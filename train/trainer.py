# train/trainer.py
import os
import sys
import math
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

sys.path.insert(0, ".")

from data.dataset_builder import build_dataloaders
from data.flank_extractor import load_reference
from tokenizer.kmer_tokenizer import KmerTokenizer, load_vocab
from model.transformer import AtlasGenSLM


# ── Paths ─────────────────────────────────────────────────────────────────────

REFERENCE_PATH = "data/reference/chr22.fa"
VCF_PATH = "data/raw/1000genomes/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
VOCAB_PATH = "tokenizer/vocab.json"
CHECKPOINT_DIR = "checkpoints"


# ── Training Config ───────────────────────────────────────────────────────────

PHYSICAL_BATCH = 2
ACCUMULATION = 32
EPOCHS = 10

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_FRACTION = 0.05

LOG_EVERY = 50

LOCAL_WINDOW = 10
MLM_PROB = 0.15
SPECIAL_IDS = set(range(11))

MAX_VARIANTS = 5000  # Smoke test active. Change to None for full run.

SEED = 1337
VALIDATION_SEED = 2026

# Set this to "checkpoints/phase1_last.pt" to resume.
RESUME_FROM = None


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── MLM Hybrid Masking ────────────────────────────────────────────────────────

def hybrid_mask(input_ids, mask_id, vocab_size, mutation_indices):
    """
    Dynamic MLM masking.

    input_ids stays on CPU here to avoid tiny Python-driven CUDA writes.
    Returns:
        masked_ids: [B, T]
        labels:     [B, T], -100 where loss is ignored
    """
    B, T = input_ids.shape

    masked = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    num_to_mask = max(1, int(MLM_PROB * T))
    num_local = num_to_mask // 2
    num_global = num_to_mask - num_local

    for i in range(B):
        ids = input_ids[i].tolist()

        maskable_set = {
            j for j in range(T)
            if ids[j] not in SPECIAL_IDS
        }

        if not maskable_set:
            continue

        mut_idx = int(mutation_indices[i])

        local_start = max(0, mut_idx - LOCAL_WINDOW)
        local_end = min(T, mut_idx + LOCAL_WINDOW + 1)

        local_positions = set(range(local_start, local_end))

        local_pool = sorted(maskable_set & local_positions)
        global_pool = sorted(maskable_set - set(local_pool))

        n_local = min(num_local, len(local_pool))
        n_global = min(num_global, len(global_pool))

        selected_local = random.sample(local_pool, n_local) if n_local > 0 else []
        selected_global = random.sample(global_pool, n_global) if n_global > 0 else []

        selected = selected_local + selected_global

        # If local/global pools were too small, fill missing masks from remaining
        # valid tokens. Usually not needed, but makes the masking count robust.
        missing = min(num_to_mask, len(maskable_set)) - len(selected)
        if missing > 0:
            remaining_pool = sorted(maskable_set - set(selected))
            selected += random.sample(
                remaining_pool,
                min(missing, len(remaining_pool))
            )

        for pos in selected:
            labels[i, pos] = ids[pos]

            r = random.random()

            # BERT-style MLM corruption:
            # 80% [MASK], 10% random token, 10% unchanged.
            if r < 0.80:
                masked[i, pos] = mask_id
            elif r < 0.90:
                masked[i, pos] = random.randint(11, vocab_size - 1)

    return masked, labels


# ── Optimizer / Scheduler ─────────────────────────────────────────────────────

def build_optimizer(model, lr, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(nd in name for nd in ["bias", "norm", "LayerNorm"]):
            no_decay.append(param)
        else:
            decay.append(param)

    return AdamW(
        [
            {
                "params": decay,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
    )


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        progress = (current_step - warmup_steps) / max(
            1,
            total_steps - warmup_steps
        )

        return max(
            0.0,
            0.5 * (1.0 + math.cos(math.pi * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def safe_torch_load(path, device):
    """
    Supports both newer and older PyTorch versions.
    Newer PyTorch may default to weights_only behavior, so we explicitly avoid it.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def clear_pad_optimizer_state(model, optimizer):
    """
    If resuming from an older checkpoint, AdamW momentum buffers may contain
    nonzero values for PAD row. Clear them so PAD stays clean.
    """
    if not hasattr(model, "embedding"):
        return

    if not hasattr(model.embedding, "token_embeddings"):
        return

    pad_param = model.embedding.token_embeddings.weight
    state = optimizer.state.get(pad_param, None)

    if not state:
        return

    for key in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
        value = state.get(key, None)

        if torch.is_tensor(value) and value.ndim >= 2 and value.size(0) > 0:
            value[0].zero_()


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    val_loss,
    best_val_loss,
    optimizer_step,
    device,
):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "optimizer_step": optimizer_step,
        "model_config": model.get_config() if hasattr(model, "get_config") else None,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": (
            torch.cuda.get_rng_state_all()
            if device.type == "cuda"
            else None
        ),
    }

    torch.save(checkpoint, path)


def load_checkpoint_if_requested(
    resume_from,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
    steps_per_epoch,
):
    if resume_from is None:
        return 0, float("inf"), 0

    if not os.path.exists(resume_from):
        raise FileNotFoundError(f"Checkpoint not found: {resume_from}")

    print(f"\nResuming from checkpoint: {resume_from}")

    checkpoint = safe_torch_load(resume_from, device)

    model.load_state_dict(checkpoint["model_state"])

    if hasattr(model, "zero_pad_embedding"):
        model.zero_pad_embedding()

    if "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        clear_pad_optimizer_state(model, optimizer)

    if "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])

    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"])

    if (
        device.type == "cuda"
        and checkpoint.get("cuda_random_state", None) is not None
    ):
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

    start_epoch = int(checkpoint.get("epoch", 0))

    best_val_loss = float(
        checkpoint.get(
            "best_val_loss",
            checkpoint.get("val_loss", float("inf"))
        )
    )

    optimizer_step = int(
        checkpoint.get(
            "optimizer_step",
            start_epoch * steps_per_epoch
        )
    )

    print(
        f"Resume state: start_epoch={start_epoch}, "
        f"optimizer_step={optimizer_step}, "
        f"best_val_loss={best_val_loss:.4f}"
    )

    return start_epoch, best_val_loss, optimizer_step


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(model, val_loader, tokenizer, device, validation_seed=None):
    """
    Validation uses dynamic masking, but with a fixed seed by default so val_loss
    is comparable across epochs.

    The Python RNG state is restored afterward so validation does not affect
    future training masks.
    """
    was_training = model.training
    model.eval()

    random_state = random.getstate()

    if validation_seed is not None:
        random.seed(validation_seed)

    total_loss = 0.0
    total_batches = 0

    amp_enabled = device.type == "cuda"

    try:
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"].to(
                    device,
                    non_blocking=True
                )

                mutation_indices = batch["mutation_index"].tolist()

                masked_ids, labels = hybrid_mask(
                    input_ids,
                    tokenizer.mask_id,
                    tokenizer.vocab_size,
                    mutation_indices,
                )

                masked_ids = masked_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast(device_type=device.type, enabled=amp_enabled):
                    logits = model(masked_ids, attention_mask)

                    loss = F.cross_entropy(
                        logits.reshape(-1, tokenizer.vocab_size),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )

                if not torch.isfinite(loss):
                    raise RuntimeError("Validation loss became NaN or Inf.")

                total_loss += loss.item()
                total_batches += 1

    finally:
        if validation_seed is not None:
            random.setstate(random_state)

        model.train(was_training)

    avg_loss = total_loss / max(1, total_batches)

    if avg_loss < 50:
        ppl = math.exp(avg_loss)
    else:
        ppl = float("inf")

    return avg_loss, ppl


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    tokenizer,
    device,
    epochs,
    lr,
    weight_decay,
    accumulation_steps,
    resume_from=None,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    optimizer = build_optimizer(model, lr, weight_decay)

    amp_enabled = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_optimizer_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(WARMUP_FRACTION * total_optimizer_steps))

    scheduler = build_scheduler(
        optimizer,
        warmup_steps,
        total_optimizer_steps,
    )

    start_epoch, best_val_loss, optimizer_step = load_checkpoint_if_requested(
        resume_from=resume_from,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        steps_per_epoch=steps_per_epoch,
    )

    running_loss = 0.0
    running_batches = 0

    print("\nStarting Phase 1 pretraining")
    print(
        f"Device: {device} | "
        f"Physical batch: {PHYSICAL_BATCH} | "
        f"Accumulation: {accumulation_steps} | "
        f"Effective batch: {PHYSICAL_BATCH * accumulation_steps} | "
        f"Total optimizer steps: {total_optimizer_steps:,}"
    )

    if start_epoch >= epochs:
        print(
            f"Checkpoint already reached epoch {start_epoch}. "
            f"Requested EPOCHS={epochs}. Nothing to train."
        )
        return

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Keep input_ids on CPU for Python-side dynamic masking.
            input_ids = batch["input_ids"]

            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True
            )

            mutation_indices = batch["mutation_index"].tolist()

            masked_ids, labels = hybrid_mask(
                input_ids,
                tokenizer.mask_id,
                tokenizer.vocab_size,
                mutation_indices,
            )

            masked_ids = masked_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            is_accumulation_step = (
                (batch_idx + 1) % accumulation_steps == 0
            )

            is_last_batch = (
                (batch_idx + 1) == len(train_loader)
            )

            group_start = (
                batch_idx // accumulation_steps
            ) * accumulation_steps

            current_accumulation = min(
                accumulation_steps,
                len(train_loader) - group_start
            )

            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(masked_ids, attention_mask)

                loss = F.cross_entropy(
                    logits.reshape(-1, tokenizer.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Training loss became NaN or Inf at "
                        f"epoch={epoch+1}, batch={batch_idx+1}."
                    )

                loss = loss / current_accumulation

            scaler.scale(loss).backward()

            raw_loss = loss.item() * current_accumulation

            running_loss += raw_loss
            epoch_loss += raw_loss

            running_batches += 1
            epoch_batches += 1

            if is_accumulation_step or is_last_batch:
                scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Gradient norm became NaN or Inf at "
                        f"epoch={epoch+1}, batch={batch_idx+1}."
                    )

                scale_before = scaler.get_scale()

                scaler.step(optimizer)
                scaler.update()

                scale_after = scaler.get_scale()

                optimizer.zero_grad(set_to_none=True)

                # If AMP skipped the optimizer step due to overflow,
                # do not advance scheduler or optimizer_step.
                step_was_not_skipped = (
                    not amp_enabled
                    or scale_after >= scale_before
                )

                if step_was_not_skipped:
                    scheduler.step()
                    optimizer_step += 1
                else:
                    print(
                        f"AMP skipped optimizer step at "
                        f"epoch={epoch+1}, batch={batch_idx+1}."
                    )

                if optimizer_step > 0 and optimizer_step % LOG_EVERY == 0:
                    avg_running = running_loss / max(1, running_batches)

                    print(
                        f"Epoch {epoch+1:02d} | "
                        f"Step {optimizer_step:05d}/{total_optimizer_steps} | "
                        f"Loss {avg_running:.4f} | "
                        f"LR {scheduler.get_last_lr()[0]:.2e}"
                    )

                    running_loss = 0.0
                    running_batches = 0

        avg_train_loss = epoch_loss / max(1, epoch_batches)

        val_loss, val_ppl = run_validation(
            model=model,
            val_loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            validation_seed=VALIDATION_SEED,
        )

        print(
            f"\nEpoch {epoch+1:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f}"
        )

        epoch_checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"phase1_epoch{epoch+1:02d}.pt"
        )

        save_checkpoint(
            path=epoch_checkpoint_path,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            optimizer_step=optimizer_step,
            device=device,
        )

        print(f"Checkpoint saved: {epoch_checkpoint_path}")

        last_path = os.path.join(CHECKPOINT_DIR, "phase1_last.pt")

        save_checkpoint(
            path=last_path,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            val_loss=val_loss,
            best_val_loss=best_val_loss,
            optimizer_step=optimizer_step,
            device=device,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            best_path = os.path.join(CHECKPOINT_DIR, "phase1_best.pt")

            save_checkpoint(
                path=best_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                val_loss=val_loss,
                best_val_loss=best_val_loss,
                optimizer_step=optimizer_step,
                device=device,
            )

            print(f"New best model saved: {best_path} | val_loss={val_loss:.4f}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ref = load_reference(REFERENCE_PATH)
    vocab = load_vocab(VOCAB_PATH)
    tokenizer = KmerTokenizer(vocab=vocab)

    print("\nBuilding dataloaders...")

    train_loader, val_loader, _ = build_dataloaders(
        vcf_paths=[VCF_PATH],
        ref=ref,
        tokenizer=tokenizer,
        batch_size=PHYSICAL_BATCH,
        max_variants=MAX_VARIANTS,
    )

    print(
        f"Train samples: {len(train_loader.dataset):,} | "
        f"Batches: {len(train_loader):,}"
    )

    print(
        f"Val samples:   {len(val_loader.dataset):,} | "
        f"Batches: {len(val_loader):,}"
    )

    model = AtlasGenSLM().to(device)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        accumulation_steps=ACCUMULATION,
        resume_from=RESUME_FROM,
    )