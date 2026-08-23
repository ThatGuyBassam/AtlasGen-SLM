# train/trainer.py
import os
import sys
import math
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast

sys.path.insert(0, ".")

from data.dataset_builder import build_dataloaders
from tokenizer.kmer_tokenizer import KmerTokenizer, load_vocab
from model.transformer import AtlasGenSLM


# ── Paths ─────────────────────────────────────────────────────────────────────

SHARD_DIR = "data/shards"
VOCAB_PATH = "tokenizer/vocab.json"
CHECKPOINT_DIR = "checkpoints"


# ── Training Config ───────────────────────────────────────────────────────────

PHYSICAL_BATCH = 32
ACCUMULATION = 2          # effective batch = 32 * 2 = 64
EPOCHS = 10

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_FRACTION = 0.15

LOG_EVERY = 50

LOCAL_WINDOW = 10
MLM_PROB = 0.15
SPECIAL_IDS = set(range(11))  # PAD, UNK, MASK, CLS, SEP, UNK_1..UNK_6

NUM_WORKERS = 2

SEED = 1337
VALIDATION_SEED = 2026

# Set this to "checkpoints/phase1_last.pt" to resume mid-phase (keeps
# optimizer/scheduler state). For starting a NEW phase, use
# WEIGHTS_ONLY_INIT_FROM instead — see below.
RESUME_FROM = None

# Set this to a weights-only checkpoint path (e.g. produced by
# save_weights_only_checkpoint at the end of Phase 1b) to initialize model
# weights for a new phase WITHOUT carrying over optimizer momentum or the
# cosine schedule. This is the phase-transition path: stale momentum from
# the previous phase's loss landscape should not contaminate the new
# phase's optimizer dynamics.
WEIGHTS_ONLY_INIT_FROM = None


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── MLM Hybrid Masking (runs inside DataLoader worker processes) ──────────────

class MLMCollator:
    """
    Turns a list of raw samples (input_ids, attention_mask, mutation_index,
    index) into a masked training batch.

    This is passed as DataLoader(collate_fn=...), which means it runs
    inside worker processes (when num_workers > 0) rather than in the main
    training loop. Two benefits:
      1. Masking overlaps with the GPU forward/backward pass of the
         previous batch instead of blocking the main process.
      2. Each sample's masking decisions are seeded deterministically from
         (seed, epoch, sample_index), not from a shared, sequentially
         mutated global random.Random. That makes masking reproducible
         given a seed + epoch, independent of worker scheduling order —
         which is NOT guaranteed if workers share one global RNG.

    Masking scheme (BERT-style, hybrid local/global):
      - MLM_PROB of maskable tokens are selected total, where the count is
        based on the REAL content length (non-special, non-PAD tokens),
        not the padded tensor length — see the note on num_to_mask below,
        this matters.
      - Half preferentially drawn from a window around the mutation
        position (LOCAL_WINDOW on each side) — this is the signal that
        matters most for the eventual pathogenicity classification task.
      - Half drawn from anywhere else in the sequence.
      - Of selected positions: 80% -> [MASK], 10% -> random token, 10% ->
        left unchanged (standard BERT corruption ratios).
    """

    def __init__(
        self,
        mask_id,
        vocab_size,
        special_ids,
        mlm_prob=MLM_PROB,
        local_window=LOCAL_WINDOW,
        seed=SEED,
        fixed_epoch=None,
    ):
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self.special_ids = special_ids
        self.mlm_prob = mlm_prob
        self.local_window = local_window
        self.seed = seed

        # If fixed_epoch is set (used for validation), masking is identical
        # every call regardless of training progress — keeps val_loss
        # comparable across epochs. Training collators leave this None and
        # rely on set_epoch() being called once per epoch instead.
        self.fixed_epoch = fixed_epoch
        self.epoch = fixed_epoch if fixed_epoch is not None else 0

    def set_epoch(self, epoch):
        if self.fixed_epoch is None:
            self.epoch = epoch

    def _mask_one(self, input_ids, mutation_index, rng):
        T = input_ids.shape[0]
        masked = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        ids = input_ids.tolist()
        maskable = [j for j in range(T) if ids[j] not in self.special_ids]

        if not maskable:
            return masked, labels

        maskable_set = set(maskable)

        # IMPORTANT: base the mask count on len(maskable_set) — the REAL
        # content length — not T (the padded tensor length, 256). Basing
        # it on T inflates the effective masking rate on real content well
        # above mlm_prob (e.g. ~22% actual vs 15% intended for a ~171-token
        # real sequence in a 256-length tensor), and as a side effect
        # saturates the local window around the mutation almost completely
        # (measured ~90% of the local window masked every sample, versus
        # ~57% with this fix — still a strong intentional bias toward the
        # mutation region, just not a near-total wipeout of local context).
        num_to_mask = max(1, int(self.mlm_prob * len(maskable_set)))
        num_local = num_to_mask // 2
        num_global = num_to_mask - num_local

        local_start = max(0, mutation_index - self.local_window)
        local_end = min(T, mutation_index + self.local_window + 1)
        local_positions = set(range(local_start, local_end))

        local_pool = sorted(maskable_set & local_positions)
        global_pool = sorted(maskable_set - set(local_pool))

        n_local = min(num_local, len(local_pool))
        n_global = min(num_global, len(global_pool))

        selected = []
        if n_local > 0:
            selected += rng.sample(local_pool, n_local)
        if n_global > 0:
            selected += rng.sample(global_pool, n_global)

        missing = min(num_to_mask, len(maskable_set)) - len(selected)
        if missing > 0:
            remaining_pool = sorted(maskable_set - set(selected))
            selected += rng.sample(remaining_pool, min(missing, len(remaining_pool)))

        for pos in selected:
            labels[pos] = ids[pos]
            r = rng.random()

            if r < 0.80:
                masked[pos] = self.mask_id
            elif r < 0.90:
                masked[pos] = rng.randint(11, self.vocab_size - 1)
            # else: leave unchanged (10% case)

        return masked, labels

    def __call__(self, batch):
        input_ids = torch.stack([s["input_ids"] for s in batch])
        attention_mask = torch.stack([s["attention_mask"] for s in batch])
        mutation_indices = [s["mutation_index"] for s in batch]
        sample_indices = [s["index"] for s in batch]

        masked_batch = torch.empty_like(input_ids)
        labels_batch = torch.full_like(input_ids, -100)

        for i in range(input_ids.shape[0]):
            # Deterministic per-sample seed: same sample + same epoch always
            # masks the same way, regardless of which worker processed it
            # or what batch it landed in.
            local_seed = (self.seed * 1_000_003 + self.epoch * 9_973 + sample_indices[i]) % (2**31)
            rng = random.Random(local_seed)

            masked, labels = self._mask_one(input_ids[i], mutation_indices[i], rng)
            masked_batch[i] = masked
            labels_batch[i] = labels

        return {
            "input_ids": masked_batch,
            "attention_mask": attention_mask,
            "labels": labels_batch,
        }


# ── Optimizer / Scheduler ─────────────────────────────────────────────────────

def build_optimizer(model, lr, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # RMSNorm modules are still named "norm"/"norm1"/"norm2"/etc, and
        # their only parameter is "weight" (no bias) — this substring check
        # still correctly routes them to no_decay.
        if any(nd in name for nd in ["bias", "norm"]):
            no_decay.append(param)
        else:
            decay.append(param)

    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def build_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Checkpointing ─────────────────────────────────────────────────────────────

def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def clear_pad_optimizer_state(model, optimizer):
    if not hasattr(model, "embedding") or not hasattr(model.embedding, "token_embeddings"):
        return

    pad_param = model.embedding.token_embeddings.weight
    state = optimizer.state.get(pad_param, None)

    if not state:
        return

    for key in ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]:
        value = state.get(key, None)
        if torch.is_tensor(value) and value.ndim >= 2 and value.size(0) > 0:
            value[0].zero_()


def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss, best_val_loss, optimizer_step, device):
    """Full training checkpoint: model + optimizer + scheduler + RNG state. Use for resuming WITHIN a phase."""
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "optimizer_step": optimizer_step,
        "model_config": model.get_config() if hasattr(model, "get_config") else None,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    torch.save(checkpoint, path)


def save_weights_only_checkpoint(path, model, val_loss):
    """
    Model-weights-only checkpoint, used specifically at phase boundaries.

    Deliberately excludes optimizer/scheduler state: starting a new phase
    (e.g. Phase 1b -> Phase 1c, or Phase 1 -> Phase 2) with fresh AdamW
    momentum and a fresh cosine schedule is intentional — stale momentum
    tuned to the previous phase's data distribution and loss landscape
    should not carry into the new phase's optimizer dynamics.
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "val_loss": val_loss,
        "model_config": model.get_config() if hasattr(model, "get_config") else None,
    }
    torch.save(checkpoint, path)
    print(f"Weights-only checkpoint saved: {path} (val_loss={val_loss:.4f})")


def load_weights_only_checkpoint(path, model, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights-only checkpoint not found: {path}")

    print(f"\nInitializing model weights from: {path}")
    checkpoint = safe_torch_load(path, device)
    model.load_state_dict(checkpoint["model_state"])

    if hasattr(model, "zero_pad_embedding"):
        model.zero_pad_embedding()

    print(f"Loaded weights (val_loss at save time: {checkpoint.get('val_loss', 'unknown')})")


def load_checkpoint_if_requested(resume_from, model, optimizer, scheduler, device, steps_per_epoch):
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

    if "python_random_state" in checkpoint:
        random.setstate(checkpoint["python_random_state"])

    if "torch_random_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_random_state"])

    if device.type == "cuda" and checkpoint.get("cuda_random_state", None) is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])

    start_epoch = int(checkpoint.get("epoch", 0))
    best_val_loss = float(checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf"))))
    optimizer_step = int(checkpoint.get("optimizer_step", start_epoch * steps_per_epoch))

    print(
        f"Resume state: start_epoch={start_epoch}, "
        f"optimizer_step={optimizer_step}, "
        f"best_val_loss={best_val_loss:.4f}"
    )

    return start_epoch, best_val_loss, optimizer_step


# ── Validation ────────────────────────────────────────────────────────────────

def run_validation(model, val_loader, vocab_size, device, amp_dtype):
    """
    val_loader's collate_fn should be an MLMCollator with fixed_epoch set,
    so masking is identical every call and val_loss is comparable epoch
    to epoch.
    """
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_batches = 0

    amp_enabled = device.type == "cuda"

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            if not torch.isfinite(loss):
                raise RuntimeError("Validation loss became NaN or Inf.")

            total_loss += loss.item()
            total_batches += 1

    model.train(was_training)

    avg_loss = total_loss / max(1, total_batches)
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")

    return avg_loss, ppl


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model,
    train_loader,
    val_loader,
    train_sampler,
    train_collator,
    vocab_size,
    device,
    epochs,
    lr,
    weight_decay,
    accumulation_steps,
    resume_from=None,
):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    optimizer = build_optimizer(model, lr, weight_decay)

    # BF16 has the dynamic range of FP32 (same exponent bits), so unlike
    # FP16 it does not need loss scaling / GradScaler — that eliminated
    # the AMP gradient-skip issue seen previously on this hardware.
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.bfloat16

    steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_optimizer_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(WARMUP_FRACTION * total_optimizer_steps))

    scheduler = build_scheduler(optimizer, warmup_steps, total_optimizer_steps)

    start_epoch, best_val_loss, optimizer_step = load_checkpoint_if_requested(
        resume_from=resume_from,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        steps_per_epoch=steps_per_epoch,
    )

    running_loss = 0.0
    running_batches = 0

    actual_batch_size = train_loader.batch_size

    print("\nStarting Phase 1 pretraining")
    print(
        f"Device: {device} | Precision: bf16 | "
        f"Physical batch: {actual_batch_size} | Accumulation: {accumulation_steps} | "
        f"Effective batch: {actual_batch_size * accumulation_steps} | "
        f"Total optimizer steps: {total_optimizer_steps:,}"
    )

    if start_epoch >= epochs:
        print(f"Checkpoint already reached epoch {start_epoch}. Requested EPOCHS={epochs}. Nothing to train.")
        return

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        train_collator.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            is_accumulation_step = (batch_idx + 1) % accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)

            group_start = (batch_idx // accumulation_steps) * accumulation_steps
            current_accumulation = min(accumulation_steps, len(train_loader) - group_start)

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Training loss became NaN or Inf at epoch={epoch+1}, batch={batch_idx+1}.")

                loss = loss / current_accumulation

            # No scaler.scale() needed under BF16 — backward directly.
            loss.backward()

            raw_loss = loss.item() * current_accumulation
            running_loss += raw_loss
            epoch_loss += raw_loss
            running_batches += 1
            epoch_batches += 1

            if is_accumulation_step or is_last_batch:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Gradient norm became NaN or Inf at epoch={epoch+1}, batch={batch_idx+1}.")

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_step += 1

                if optimizer_step > 0 and optimizer_step % LOG_EVERY == 0:
                    avg_running = running_loss / max(1, running_batches)
                    print(
                        f"Epoch {epoch+1:02d} | Step {optimizer_step:05d}/{total_optimizer_steps} | "
                        f"Loss {avg_running:.4f} | LR {scheduler.get_last_lr()[0]:.2e}"
                    )
                    running_loss = 0.0
                    running_batches = 0

        avg_train_loss = epoch_loss / max(1, epoch_batches)

        val_loss, val_ppl = run_validation(
            model=model,
            val_loader=val_loader,
            vocab_size=vocab_size,
            device=device,
            amp_dtype=amp_dtype,
        )

        print(f"\nEpoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        epoch_checkpoint_path = os.path.join(CHECKPOINT_DIR, f"phase1_epoch{epoch+1:02d}.pt")
        save_checkpoint(epoch_checkpoint_path, epoch + 1, model, optimizer, scheduler, val_loss, best_val_loss, optimizer_step, device)
        print(f"Checkpoint saved: {epoch_checkpoint_path}")

        last_path = os.path.join(CHECKPOINT_DIR, "phase1_last.pt")
        save_checkpoint(last_path, epoch + 1, model, optimizer, scheduler, val_loss, best_val_loss, optimizer_step, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "phase1_best.pt")
            save_checkpoint(best_path, epoch + 1, model, optimizer, scheduler, val_loss, best_val_loss, optimizer_step, device)
            print(f"New best model saved: {best_path} | val_loss={val_loss:.4f}\n")

    # End of phase: also write a weights-only checkpoint for the next
    # phase to initialize from (see WEIGHTS_ONLY_INIT_FROM).
    weights_only_path = os.path.join(CHECKPOINT_DIR, "phase1_final_weights_only.pt")
    save_weights_only_checkpoint(weights_only_path, model, best_val_loss)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    vocab = load_vocab(VOCAB_PATH)
    tokenizer = KmerTokenizer(vocab=vocab)

    train_collator = MLMCollator(
        mask_id=tokenizer.mask_id,
        vocab_size=tokenizer.vocab_size,
        special_ids=SPECIAL_IDS,
        seed=SEED,
        fixed_epoch=None,  # advances each epoch via set_epoch()
    )
    val_collator = MLMCollator(
        mask_id=tokenizer.mask_id,
        vocab_size=tokenizer.vocab_size,
        special_ids=SPECIAL_IDS,
        seed=VALIDATION_SEED,
        fixed_epoch=0,  # identical masking every call -> comparable val_loss across epochs
    )

    print("\nBuilding dataloaders from disk shards...")
    train_loader, val_loader, _test_loader, train_sampler = build_dataloaders(
        shard_dir=SHARD_DIR,
        batch_size=PHYSICAL_BATCH,
        num_workers=NUM_WORKERS,
        seed=SEED,
        train_collate_fn=train_collator,
        eval_collate_fn=val_collator,
    )

    print(f"Train samples: {len(train_loader.dataset):,} | Batches: {len(train_loader):,}")
    print(f"Val samples:   {len(val_loader.dataset):,} | Batches: {len(val_loader):,}")

    model = AtlasGenSLM(vocab_size=tokenizer.vocab_size).to(device)

    if WEIGHTS_ONLY_INIT_FROM is not None:
        load_weights_only_checkpoint(WEIGHTS_ONLY_INIT_FROM, model, device)

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        train_collator=train_collator,
        vocab_size=tokenizer.vocab_size,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        accumulation_steps=ACCUMULATION,
        resume_from=RESUME_FROM,
    )
