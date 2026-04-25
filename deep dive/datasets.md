# AtlasGen-SLM — Dataset Documentation

## Overview

This document tracks every dataset in the AtlasGen-SLM training corpus: what it contains, where it lives, how to access it, and why it's included. Datasets are organized by training phase.

---

## Phase 1 — Global Diversity Baseline

### 1000 Genomes Project
- **Host:** International Genome Sample Resource (IGSR)
- **URL:** https://www.internationalgenome.org
- **Access:** Open
- **Size:** 2,504 individuals, 84.4 million variants
- **Populations:** 26 populations across 5 continental groups (AFR, AMR, EAS, EUR, SAS)
- **Format:** VCF (GRCh38)
- **Why:** The foundational global diversity reference. Provides balanced continental representation for Phase 1 pretraining baseline.

---

### Simons Genome Diversity Project (SGDP)
- **Host:** Simons Foundation / European Nucleotide Archive
- **ENA Accession:** PRJEB9586
- **URL:** https://www.simonsfoundation.org/simons-genome-diversity-project/
- **Access:** Open (Fort Lauderdale principles apply)
- **Size:** 300 high-coverage WGS (avg. 43x depth), 142 populations
- **Format:** BAM / VCF
- **Why:** Superior global diversity vs. 1000 Genomes. PCR-free library prep minimizes GC bias — critical for accurate representation of GC-rich regulatory regions. Includes North African samples (Mozabite). Contains 5.8M base pairs absent from GRCh38.

---

### Human Genome Diversity Project (HGDP)
- **Host:** IGSR / Centre d'Étude du Polymorphisme Humain (CEPH)
- **URL:** https://www.internationalgenome.org/data-portal/data-collection/hgdp
- **Access:** Open via IGSR ⚠️ Use IGSR portal — not AnVIL (flagged by NHGRI 2023)
- **Size:** 929 high-coverage WGS, 54 populations
- **Format:** VCF / BAM
- **Why:** Classic population genetics reference. Includes Mozabite, Bedouin, and Palestinian samples — key for North African and Middle Eastern ancestry modeling. Deep evolutionary context including Neanderthal and Denisovan introgression signals.

---

### gnomAD
- **Host:** Broad Institute
- **URL:** https://gnomad.broadinstitute.org
- **Access:** Open (aggregated allele frequencies); controlled (individual-level)
- **Size:** ~125,000 exomes, ~15,000 genomes
- **Populations:** Multi-continental including African/African-American and Middle Eastern subsets
- **Format:** VCF
- **Why:** Best allele frequency baseline available. Critical for Phase 4 — distinguishing rare pathogenic variants from common benign ones requires knowing population-level frequencies. The African subset helps calibrate benign allele frequency thresholds for the North African classifier.

---

### TOPMed
- **Host:** NHLBI
- **Access:** Controlled (dbGaP application required)
- **URL:** https://topmed.nhlbi.nih.gov
- **Size:** 150,000+ WGS
- **Populations:** Multi-ethnic, strong African-American representation
- **Format:** VCF (GRCh38, high-depth)
- **Why:** High-depth WGS with strong rare variant coverage. The African-American cohort provides additional African ancestry signal to complement H3Africa in Phase 2.

---

## Phase 2 — African & Maghrebi Focus

### H3Africa Consortium WGS
- **Host:** European Genome-phenome Archive (EGA)
- **EGA Accession:** EGAS00001005972
- **Access:** Controlled — apply via H3Africa Data and Biospecimen Access Committee
- **URL:** https://h3africa.org
- **Size:** 426 WGS, 50 ethnolinguistic groups, 13 African countries
- **Format:** VCF / BAM
- **Why:** Largest coordinated African genomics dataset. Identified 3.4 million variants absent from all global databases. Covers previously unsampled African populations. Essential for grounding the model in the ancestral genetic pool before Maghrebi specialization.

---

### Egypt Genome Project 1K (EGP1K)
- **Host:** Egypt Center for Research and Regenerative Medicine (ECRRM)
- **Archive:** European Variation Archive (EVA)
- **Access:** Controlled — application to ECRRM Access Committee (30-day review)
- **URL:** https://www.ecrrm.ac.eg
- **Reference:** bioRxiv 2026.04.02.715521
- **Size:** 1,024 WGS, 51.3 million variants (17.1 million novel)
- **Populations:** Egyptians across 21 governorates (Upper and Lower Egypt)
- **Format:** VCF (WGS)
- **Why:** Most significant recent advance in North African genomics. Egypt sits at the genetic nexus between the Levant, Arabian Peninsula, and North Africa. The EGP1K captures critical transitional genetic architecture and documents within-country heterogeneity including runs of homozygosity in Upper Egypt — directly relevant to Maghrebi consanguinity modeling.

---

### Greater Middle East (GME) Variome
- **Host:** UC San Diego / GME Consortium
- **Access:** Controlled — primary URL (igm.ucsd.edu/gme) currently inaccessible. Contact PIs directly or access summary frequencies via Illumina Nirvana.
- **Size:** 1,111 whole exomes, ~689,299 variants
- **Populations:** Six MENA subregions including Northwest Africa (NWA) cohort — 85 samples
- **Format:** VCF (WES)
- **Why:** The NWA cohort is a direct genomic window into Berber genetic background and the mutational load associated with regional endogamy. The GME dataset documents high allele frequencies of regionally benign variants that Western databases label pathogenic — exactly the signal AtlasGen-SLM needs.

---

### Saudi Genome Program v1 (SGP)
- **Host:** Saudi Ministry of Health / Saudi Genomics Platform
- **URL:** https://saudigenomics.org
- **Access:** Controlled
- **Size:** 1,378 WGS/WES, 10 million+ variants
- **Populations:** Indigenous Arab populations, healthy individuals, pediatric genetic disease patients
- **Format:** VCF
- **Why:** Arab conquests of the 7th century left a permanent genetic signature across the Maghreb. Training on Peninsular Arab ancestral variation allows the model to contextualize the Arab admixture component of modern Moroccan genomes. Shares consanguinity-driven rare variant architecture with North Africa.

---

## Phase 3 — Moroccan-Specific

### Moroccan Genome Project (MGP) ← Anchor Dataset
- **Host:** European Genome-phenome Archive (EGA)
- **EGA Accession:** EGAD50000000750
- **URL:** https://ega-archive.org/datasets/EGAD50000000750
- **Access:** Controlled (EGA DAC)
- **Size:** 109 WGS, 30x coverage (Illumina NovaSeq 6000)
- **Populations:** Moroccan individuals
- **Format:** BAM / VCF
- **Why:** The foundational Moroccan reference genome dataset. This is the primary fine-tuning corpus for Phase 3 and the ground truth anchor for the entire project.

---

### Genome-wide Array Data — Tunisia & Morocco
- **Host:** European Genome-phenome Archive (EGA)
- **EGA Accession:** EGAD00001009071
- **DAC:** EGAC00001002770
- **Access:** Controlled
- **Size:** 45 Moroccan samples, 64 Tunisian samples
- **Populations:** Urban Moroccan and Tunisian individuals
- **Format:** Array (SNP genotypes)
- **Why:** Dense allele frequency mapping for known variants across urban North African populations. Complements the MGP WGS data with population-level frequency estimates.

---

### Amazigh & Non-Amazigh Algerian Array Data
- **Host:** European Genome-phenome Archive (EGA)
- **EGA Accession:** EGAD00001010900
- **DAC:** EGAC00001003230
- **Access:** Controlled
- **Size:** 130 Amazigh (Chaoui/Mozabite), 34 non-Amazigh Algerians
- **Populations:** Eastern and Southern Algeria (Oum El Bouaghi, Batna, Khenchela, Ghardaïa, Algiers)
- **Format:** Array (SNP genotypes)
- **Why:** Provides a cross-section of autochthonous North African background spanning the Moroccan border. The Chaoui and Mozabite subgroups capture the pure Amazigh genetic signal; the non-Amazigh samples capture the admixed urban reality.

---

### Genome Tunisia Project
- **Host:** EGA (EGAC50000000353)
- **Access:** Controlled — "hold until publication" status
- **Size:** Target 100+ WGS (Phase 1 of project)
- **Populations:** Tunisian Arab and Berber
- **Status:** ⏳ Not yet publicly accessible — monitor for release
- **Why:** Closest neighboring genomic reference to the Moroccan Genome Project. When released, Tunisian WGS will significantly strengthen Phase 3 regional embeddings.

---

## Phase 4 — Pathogenic Variant Databases

### ClinVar ← Regulatory Layer
- **Host:** NCBI
- **URL:** https://www.ncbi.nlm.nih.gov/clinvar/
- **Access:** Open
- **Size:** ~1 million variant-condition records
- **Format:** VCF / TSV
- **Role:** High-confidence labels. ClinVar's aggressive curation using ACMG/AMP guidelines and gnomAD allele frequencies makes it the gold-standard regulatory layer. In curriculum learning, ClinVar benign classifications override HGMD pathogenic calls.
- **False positive rate:** 0.22 affected per 1,000 cataloged variants (2023 audit)

---

### HGMD — Human Gene Mutation Database ← Sensitivity Layer
- **Host:** QIAGEN / Cardiff University
- **URL:** https://www.hgmd.cf.ac.uk
- **Access:** Commercial (academic license)
- **Size:** ~400,000 variants across 9,000+ genes
- **Format:** VCF / custom
- **Role:** Maximum variant breadth. HGMD covers 138% more genes and 187% more variants than ClinVar. Used in Phase 4 pre-training to expose the model to the widest possible range of mutational mechanisms.
- **Caveat:** False positive rate significantly higher than ClinVar, particularly for non-European genomes (25 affected per 1,000 cataloged variants). Never use as sole source.

---

### LOVD — Leiden Open Variation Database
- **Host:** Leiden University Medical Center
- **URL:** https://www.lovd.nl
- **Access:** Open (API available)
- **Size:** 4.6 million variants across 1.8 million individuals
- **Format:** API / downloadable per gene
- **Why:** Decentralized network of locus-specific databases. Regional clinicians submit population-specific findings here before they reach ClinVar submission threshold. Contains Arab and North African-specific entries not present in ClinVar or HGMD.

---

### Mastermind Genomic Search Engine
- **Host:** Genomenon
- **URL:** https://www.genomenon.com/mastermind
- **Access:** Free academic tier available
- **Size:** 6.1 million variants from 11 million full-text publications
- **Why:** NLP-extracted variants from the complete biomedical literature — including regional journals and supplementary tables that ClinVar manual curation misses. Sensitivity of 98.4% vs. ClinVar's 37.4% in comparative benchmarking. Particularly valuable for capturing North African case reports buried in regional publications.

---

### MGDD — Moroccan Genetic Disease Database
- **Host:** Published resource (PMC3925278)
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3925278/
- **Access:** Open
- **Format:** Literature-derived metadata
- **Why:** Moroccan-specific pathogenic variant catalogue. Note: MGDD is a relational metadata repository aggregating PubMed abstracts, not a raw sequence biobank. Use for Phase 4 fine-tuning labels, not sequence training.

---

## Access Status Tracker

| Dataset | Access Type | Status |
|---|---|---|
| 1000 Genomes | Open | ✅ Ready |
| SGDP | Open | ✅ Ready |
| HGDP | Open (IGSR) | ✅ Ready |
| gnomAD | Open | ✅ Ready |
| TOPMed | dbGaP application | ⏳ Apply |
| H3Africa | DAC application | ⏳ Apply |
| EGP1K | ECRRM application | ⏳ Apply |
| GME Variome | Contact PIs | ⏳ Pending |
| Saudi Genome | Controlled | ⏳ Apply |
| MGP (EGAD50000000750) | EGA DAC | ⏳ Apply |
| Tunisia/Morocco array | EGA DAC | ⏳ Apply |
| Algerian array | EGA DAC | ⏳ Apply |
| Genome Tunisia Project | EGA (on hold) | ⏳ Monitor |
| ClinVar | Open | ✅ Ready |
| HGMD | Commercial license | ⏳ Acquire |
| LOVD | Open | ✅ Ready |
| Mastermind | Free academic | ✅ Ready |
| MGDD | Open | ✅ Ready |
