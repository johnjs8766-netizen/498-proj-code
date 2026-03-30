# Audio Watermarking with Public Encoder and External Private Keys

This project implements a device-specific audio watermarking system with:

- a **public encoder**
- a **verifier-side decoder**
- **external per-user private keys**

The goal is to embed a device/user-specific watermark into an input audio sample and then let the verifier identify the originating device from the watermarked audio.

---

## Overview

The watermarking pipeline consists of:

- **Encoder**: used to embed a device-specific watermark into an audio sample
- **Decoder**: used by the verifier to classify the device label from the watermarked audio

A single shared encoder-decoder system is trained by the verifier.  
After training, the verifier exports:

- a **public encoder model**
- one **private secret key file** for each authorized user/device

Each authorized user receives:

- the same public encoder
- **only their own private key**

The decoder is kept by the verifier and is **not distributed to users**.

---

## Current Design

This repository uses the final deployment design:

- the **public encoder model does not store all users' secret keys**
- user-specific secret information is stored **externally** in per-user key files
- users cannot obtain other users' private key material from the public encoder alone

For each user, the private key file contains learned user-specific watermark information, including:

- a TF (time-frequency) codeword
- a waveform-branch conditioning key

During inference, watermark generation uses:

- input audio
- public encoder weights
- the user’s external secret key file

---

## Training and Deployment Flow

### Training phase

The verifier trains the watermarking model using:

```bash
python train.py
```

Training includes multiple stages for stable optimization:

- **Stage A**: initialize watermark embedding behavior
- **Stage B**: train blind device decoder
- **Stage C**: optional joint fine-tuning

All training is performed by the verifier.

### Export phase

After training, the verifier exports:

- the public encoder
- one secret key file per user

Run:

```bash
python export_public_and_keys.py
```

This produces:

- `./exported_public_model/public_generator.pth`
- `./user_keys/user_0_secret.pt`
- `./user_keys/user_1_secret.pt`
- `./user_keys/user_2_secret.pt`
- `./user_keys/user_3_secret.pt`
- `./user_keys/user_4_secret.pt`
- `./user_keys/user_5_secret.pt`

Assuming there are 6 authorized devices.

### User-side watermark generation

A user watermarks audio with:

- their input voice/audio sample
- the public encoder
- their own secret key file

Example:

```bash
python test_with_external_key.py --input_audio 3s_audio_record.wav --key_path user_keys/user_3_secret.pt --public_generator_ckpt exported_public_model/public_generator.pth --detector_ckpt checkpoints/stageC_blind_detector_epoch_8.pth --output_audio outputs/test_user3_watermarked.wav
```

This will:

- generate a watermarked audio file
- run the decoder
- print the predicted device label

### Robustness evaluation

To evaluate robustness under signal distortions and attacks, run:

```bash
python evaluate_robustness.py
```

This script evaluates performance under attacks such as:

- additive Gaussian noise
- resampling
- low-pass filtering
- amplitude scaling
- temporal cropping
- codec-like degradation

It also reports basic quality metrics such as:

- SNR
- LSD

Results are saved to:

- `./robustness_results/robustness_summary.json`

---

## Project Structure

```text
498-PROJ-CODE/
├── checkpoints/                  # trained decoder / detector checkpoints
├── dataset_libritts/             # dataset files
├── embedded_models/              # optional exported / converted models
├── exported_public_model/        # exported public encoder model
├── datasets.py                   # dataset loader
├── eval_tflite.py                # TFLite evaluation script
├── eval_tflite_requirements.txt  # requirements for TFLite evaluation
├── evaluate_robustness.py        # robustness evaluation script
├── export_public_and_keys.py     # export public encoder + per-user secret keys
├── losses.py                     # loss functions
├── models.py                     # model definitions
├── pytorch_to_tflite.py          # PyTorch -> TFLite conversion
├── pytorch_to_tflite_requirements.txt
├── readme.md
├── save_audio.py                 # auxiliary script (optional)
├── test_with_external_key.py     # test/inference using public model + key file
├── train.py                      # training script
└── user_keys/                    # exported per-user private secret key files
```

You may ignore some auxiliary scripts such as `save_audio.py` if they are not needed for your use case.

---

## Key Files

After export, the folder `./user_keys/` stores one private key file per authorized user.

Example:

```text
user_keys/
├── user_0_secret.pt
├── user_1_secret.pt
├── user_2_secret.pt
├── user_3_secret.pt
├── user_4_secret.pt
└── user_5_secret.pt
```

Each user should only receive **their own** secret key file.

---

## Checkpoints

Training outputs model checkpoints to `./checkpoints/`.

Examples:

- `stageB_blind_detector_epoch_*.pth`
- `stageC_blind_detector_epoch_*.pth`
- `stageA_generator_epoch_*.pth`
- `stageC_generator_epoch_*.pth`

In deployment, the main files of interest are usually:

- the best generator checkpoint
- the best blind detector checkpoint
- the exported public generator
- the per-user secret key files

---

## Typical Workflow

### 1. Train the model

```bash
python train.py
```

### 2. Export public model and private user keys

```bash
python export_public_and_keys.py
```

### 3. Test one user/device

```bash
python test_with_external_key.py --input_audio 3s_audio_record.wav --key_path user_keys/user_3_secret.pt --public_generator_ckpt exported_public_model/public_generator.pth --detector_ckpt checkpoints/stageC_blind_detector_epoch_8.pth --output_audio outputs/test_user3_watermarked.wav
```

### 4. Run robustness evaluation

```bash
python evaluate_robustness.py
```

---

## Notes

- The decoder is verifier-side and is not intended to be distributed to users.
- The public encoder is shared across users.
- User-specific secret information is stored externally in key files.
- This design is intended to prevent the public encoder from containing all users’ private key information.

---

## Requirements

This project is implemented in Python with PyTorch and Torchaudio.

You may also use:

- `pytorch_to_tflite.py`
- `eval_tflite.py`

if you want model conversion / deployment experiments.

If needed, install dependencies from the corresponding requirement files:

```bash
pip install -r eval_tflite_requirements.txt
pip install -r pytorch_to_tflite_requirements.txt
```

For core training/inference, make sure PyTorch and Torchaudio are installed properly.

---

## Summary

This repository implements a multi-user audio watermarking system where:

- the verifier trains the encoder/decoder
- the decoder remains private to the verifier
- users receive a public encoder and only their own private key file
- watermarked audio can be attributed back to the corresponding device/user by the verifier