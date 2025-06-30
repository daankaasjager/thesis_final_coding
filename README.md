# Prepend Conditioning Experiments

This README documents the setup and execution of prepend conditioning experiments for the Msc thesis of Daan Kaasjager, using various conditioning strategies for a Masked Diffusion Language Model (MDLM) to generate novel, valid, and unique molecules.

## 📚 Overview

These experiments aim to answer:

1. **RQ1.** Can an unconditioned MDLM trained on SELFIES generate valid, novel, and high-quality molecules?
2. **RQ2.** How effective is prepend conditioning at improving validity, novelty, and quality of generated molecules?

This repository focuses on **RQ2 prepend conditioning**. The core idea is to prepend scalar molecular property tokens to the input sequence as conditioning, allowing the model to learn correlations between properties and molecular structure.

---


## Training

```bash
pip install -r requirements.txt
```

### Preliminary experiments
tiny (28 M) – baseline training from scratch
```bash
scripts/run.sh model=tiny \
tokenizer.tokenizer_type=wordlevel \
experiment.name="model_size_tiny" \
checkpointing.resume_from_ckpt=False \
```

small (96 M) – baseline training from scratch
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="model_size_small" \
checkpointing.resume_from_ckpt=False
```

APE-70 (~30 % vocab increase)
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=ape \
tokenizer.max_vocab_size=70 \
checkpointing.retrain_tokenizer=True \
checkpointing.retrain_ape_vocab=True \
experiment.name="ape_70" \
checkpointing.resume_from_ckpt=False \
```


APE-80 (~50 % vocab increase)
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=ape \
tokenizer.max_vocab_size=80 \
checkpointing.retrain_tokenizer=True \
checkpointing.retrain_ape_vocab=True \
experiment.name="ape_80" \
checkpointing.resume_from_ckpt=False \
```

APE-110 (~100 % vocab increase)
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=ape \
tokenizer.max_vocab_size=110 \
checkpointing.retrain_tokenizer=True \
checkpointing.retrain_ape_vocab=True \
experiment.name="ape_110" \
checkpointing.resume_from_ckpt=False \
```

### RQ2: Prepend conditioning
1 token prepended
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="prepend_1" \
conditioning="prepend" \
conditioning.properties="['sa_score']" \
checkpointing.retrain_tokenizer=True \
checkpointing.resume_from_ckpt=False \
```

3 tokens prepended
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="prepend_3" \
conditioning="prepend" \
conditioning.properties="['sa_score','mol_wt','volume']" \
checkpointing.resume_from_ckpt=False \
```

8 tokens prepended
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="prepend_8" \
conditioning="prepend" \
conditioning.properties="['sa_score', 'mol_wt', 'volume', 'vbur_vbur', 'vmin_r', 'sterimol_L', 'sterimol_B1', 'dipolemoment']" \
checkpointing.resume_from_ckpt=False \
```

all properties prepended
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="prepend_all" \
conditioning="prepend" \
conditioning.properties="['nbo_P', 'nmr_P', 'pyr_P', 'fmo_mu', 'vmin_r', 'volume', 'fmo_eta',  'fukui_m', 'fukui_p', 'nuesp_P', 'somo_rc', 'nbo_P_rc', 'pyr_alpha', 'qpole_amp', 'vbur_vbur', 'Pint_P_min', 'sterimol_L', 'sterimol_B1', 'sterimol_B5', 'dipolemoment', 'efgtens_xx_P',  'efgtens_yy_P', 'nbo_bd_e_max', 'nbo_lp_P_occ', 'qpoletens_yy', 'E_solv_elstat', 'nbo_bds_e_avg', 'sterimol_burL', 'nbo_bd_occ_avg', 'sterimol_burB5', 'vbur_ovbur_min', 'vbur_qvbur_min', 'nbo_bds_occ_max', 'vbur_ratio_vbur_vtot', 'mol_wt', 'sa_score']" \
checkpointing.resume_from_ckpt=False \
```

### RQ2: Embedding conditioning
1 token embedding
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="embedding_1" \
conditioning="embed" \
conditioning.properties="['sa_score']" \
checkpointing.resume_from_ckpt=False \
checkpointing.retrain_tokenizer=False \
```

3 token embedding
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="embedding_3" \
conditioning="embed" \
conditioning.properties="['sa_score','mol_wt','volume']" \
checkpointing.resume_from_ckpt=False \
checkpointing.retrain_tokenizer=False \
```

8 token embedding
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="embedding_8" \
conditioning="embed" \
conditioning.properties="['sa_score', 'mol_wt', 'volume', 'vbur_vbur', 'vmin_r', 'sterimol_L', 'sterimol_B1', 'dipolemoment']" \
checkpointing.resume_from_ckpt=False \
checkpointing.retrain_tokenizer=False \
```

all token embedding
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="embedding_all" \
conditioning="embed" \
conditioning.properties="['nbo_P', 'nmr_P', 'pyr_P', 'fmo_mu', 'vmin_r', 'volume', 'fmo_eta',  'fukui_m', 'fukui_p', 'nuesp_P', 'somo_rc', 'nbo_P_rc', 'pyr_alpha', 'qpole_amp', 'vbur_vbur', 'Pint_P_min', 'sterimol_L', 'sterimol_B1', 'sterimol_B5', 'dipolemoment', 'efgtens_xx_P',  'efgtens_yy_P', 'nbo_bd_e_max', 'nbo_lp_P_occ', 'qpoletens_yy', 'E_solv_elstat', 'nbo_bds_e_avg', 'sterimol_burL', 'nbo_bd_occ_avg', 'sterimol_burB5', 'vbur_ovbur_min', 'vbur_qvbur_min', 'nbo_bds_occ_max', 'vbur_ratio_vbur_vtot', 'mol_wt', 'sa_score']" \
checkpointing.resume_from_ckpt=False \
checkpointing.retrain_tokenizer=False \
```

### RQ2: CFG conditioning
0.1 masking probability of embedding vector
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="cfg_01" \
conditioning="cfg" \
conditioning.properties="['sa_score','mol_wt','volume']" \
conditioning.cfg_prob=0.1 \
checkpointing.resume_from_ckpt=False \
checkpointing.retrain_tokenizer=False \
```

0.2 masking probability of embedding vector
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="cfg_02" \
conditioning="cfg" \
conditioning.properties="['sa_score','mol_wt','volume']" \
conditioning.cfg_prob=0.2 \
checkpointing.resume_from_ckpt=False \
checkpointing.retrain_tokenizer=False \
```

0.3 masking probability of embedding vector
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="cfg_03" \
conditioning="cfg" \
conditioning.properties="['sa_score','mol_wt','volume']" \
conditioning.cfg_prob=0.3 \
checkpointing.resume_from_ckpt=False \
checkpointing.retrain_tokenizer=False \
```


## Sampling
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
mode=evaluate \
experiment.name="experiment_name" \
conditioning="cfg" \
conditioning.properties="['list','of','target','properties']" \
conditioning.cfg_prob=0.3 \
checkpointing.resume_from_ckpt=False \
checkpointing.retrain_tokenizer=False \
```

```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
mode=generate \
experiment.name="model_size_small" \
paths.tokenizer="/scratch/s3905845/thesis_final_coding/data/kraken/tokenizers/prepend_3" #my own fuckup, delete later