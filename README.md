# Prepend Conditioning Experiments

This README documents the setup and execution of prepend conditioning experiments for the Molecular Diffusion Language Model (MDLM) project.

## 📚 Overview

These experiments aim to answer:

1. **RQ1.** Can an unconditioned MDLM trained on SELFIES generate valid, novel, and high-quality molecules?
2. **RQ2.** How effective is prepend conditioning at improving validity, novelty, and quality of generated molecules?

This repository focuses on **RQ2 prepend conditioning**. The core idea is to prepend scalar molecular property tokens to the input sequence as conditioning, allowing the model to learn correlations between properties and molecular structure.

---


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
checkpointing.resume_ckpt_path="./checkpoints/tiny-wordlevel-False.ckpt"
```

small (96 M) – baseline training from scratch
```bash
small (96 M) – baseline training from scratch
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="model_size_small" \
checkpointing.resume_from_ckpt=False \
checkpointing.resume_ckpt_path="./checkpoints/small-wordlevel-False.ckpt"
```

APE-70 (~10 % vocab increase)
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=ape \
tokenizer.max_vocab_size=70 \
checkpointing.retrain_tokenizer=True \
checkpointing.retrain_ape_vocab=True \
experiment.name="ape_70" \
checkpointing.resume_from_ckpt=False \
checkpointing.resume_ckpt_path="./checkpoints/small-ape-70-False.ckpt"
local_paths.train_data_encoding=/path/to/encoding/ape_70
```


APE-80 (~20 % vocab increase)
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=ape \
tokenizer.max_vocab_size=80 \
checkpointing.retrain_tokenizer=True \
checkpointing.retrain_ape_vocab=True \
experiment.name="ape_80" \
checkpointing.resume_from_ckpt=False \
checkpointing.resume_ckpt_path="./checkpoints/small-ape-80-False.ckpt"
local_paths.train_data_encoding=/path/to/encoding/ape_80
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
checkpointing.resume_ckpt_path="./checkpoints/small-ape-110-False.ckpt"
local_paths.train_data_encoding=/path/to/encoding/ape_110
```

### RQ2: Prepend conditioning
1 token prepended
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="prepend_1" \
conditioning="prepend" \
conditioning.properties="['sa_score']" \
checkpointing.resume_from_ckpt=False \
checkpointing.resume_ckpt_path="./checkpoints/small-wordlevel-False.ckpt" \
paths.train_data_encoding="/scratch/s3905845/thesis_final_coding/data/kraken/training_data/prepend_1"
```

3 tokens prepended
```bash
scripts/run.sh model=small \
tokenizer.tokenizer_type=wordlevel \
experiment.name="prepend_3" \
conditioning="prepend" \
conditioning.properties="['sa_score','mol_wt','volume']" \
checkpointing.resume_from_ckpt=False \
checkpointing.resume_ckpt_path="./checkpoints/small-wordlevel-False.ckpt" \
paths.train_data_encoding="/scratch/s3905845/thesis_final_coding/data/kraken/training_data/prepend_3"
```

8 tokens prepended
```bash
scripts/run.sh model=small \ 
tokenizer.tokenizer_type=wordlevel \
experiment.name="prepend_8" \
conditioning="prepend" \
conditioning.properties=['sa_score', 'mol_wt', 'volume', 'vbur_vbur', 'vmin_r', 'sterimol_L', 'sterimol_B1', 'dipolemoment'] \
checkpointing.resume_from_ckpt=False \
checkpointing.resume_ckpt_path="./checkpoints/small-wordlevel-False.ckpt" \
paths.train_data_encoding="/scratch/s3905845/thesis_final_coding/data/kraken/training_data/prepend_8"
```

all properties prepended
```bash
scripts/run.sh model=small \ 
tokenizer.tokenizer_type=wordlevel \
experiment.name="prepend_8" \
conditioning="prepend" \
conditioning.properties=['nbo_P', 'nmr_P', 'pyr_P', 'fmo_mu', 'vmin_r', 'volume', 'fmo_eta',  'fukui_m', 'fukui_p', 'nuesp_P', 'somo_rc', 'nbo_P_rc', 'pyr_alpha', 'qpole_amp', 'vbur_vbur', 'Pint_P_min', 'sterimol_L', 'sterimol_B1', 'sterimol_B5', 'dipolemoment', 'efgtens_xx_P',  'efgtens_yy_P', 'nbo_bd_e_max', 'nbo_lp_P_occ', 'qpoletens_yy', 'E_solv_elstat', 'nbo_bds_e_avg', 'sterimol_burL', 'nbo_bd_occ_avg', 'sterimol_burB5', 'vbur_ovbur_min', 'vbur_qvbur_min', 'nbo_bds_occ_max', 'vbur_ratio_vbur_vtot', 'mol_wt', 'sa_score'] \
checkpointing.resume_from_ckpt=False \
checkpointing.resume_ckpt_path="./checkpoints/small-wordlevel-False.ckpt"
paths.train_data_encoding="/scratch/s3905845/thesis_final_coding/data/kraken/training_data/prepend_all"
```
