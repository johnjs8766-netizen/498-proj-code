This project implements a device-specific audio watermarking system with:

- a public encoder
- a verifier-side decoder
- external per-user private keys

The goal is to embed a device-specific watermark into an input audio sample and then let the verifier identify the originating device from the watermarked audio.



## Overview

The watermarking pipeline consists of:

- **Encoder**: used to embed a device-specific watermark into an audio sample
- **Decoder**: used by the verifier to classify the device label from the watermarked audio

A single shared encoder-decoder system is trained by the verifier.  
After training, the verifier exports:

- a public encoder model
- one private secret key file for each authorized device

Each authorized user receives:

- the same public encoder
- only their own private key

The decoder is kept by the verifier and is not distributed to users.


## Design

The pipeline abides by the following design principles:

- the public encoder model does not store all users' secret keys
- user-specific secret information is stored externally in per-user key files
- users cannot obtain other users' private key material from the public encoder alone

For each user, the private key file contains learned user-specific watermark information, including:

- a TF (time-frequency) codeword
- a waveform-branch conditioning key

During inference, watermark generation uses:

- input audio
- public encoder weights
- the user’s external secret key file


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

Assuming there are 6 authorized devices. This produces:

- `./exported_public_model/public_generator.pth`
- `./user_keys/user_0_secret.pt`
- `./user_keys/user_1_secret.pt`
- `./user_keys/user_2_secret.pt`
- `./user_keys/user_3_secret.pt`
- `./user_keys/user_4_secret.pt`
- `./user_keys/user_5_secret.pt`



### User-side watermark generation

A user watermarks audio with:

- their input voice/audio sample
- the public encoder
- their own secret key file

For example, if user's ID is 3, then execute the script by:

```bash
python test_with_external_key.py --input_audio sample.wav --key_path user_keys/user_3_secret.pt --public_generator_ckpt exported_public_model/public_generator.pth --detector_ckpt checkpoints/stageC_blind_detector_epoch_8.pth --output_audio outputs/test_user3_watermarked.wav
```

### Robustness evaluation

To evaluate robustness under signal distortions and attacks, run:

```bash
python evaluate_robustness.py
```

This script will evaluate performance under common adversarial attacks toward the watermarking system such as:

- additive Gaussian noise
- resampling
- low-pass filtering
- amplitude scaling
- temporal cropping
- codec-like degradation

It also reports basic quality metrics such as:

- SNR
- LSD

Results will be shown on the terminal as well as saved json file:

- `./robustness_results/robustness_summary.json`


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


## Checkpoints

Training outputs model checkpoints to `./checkpoints/`.

Examples:

- `stageB_blind_detector_epoch_*.pth`
- `stageC_blind_detector_epoch_*.pth`
- `stageA_generator_epoch_*.pth`
- `stageC_generator_epoch_*.pth`


## Typical Workflow

### 1. Train the model

```bash
python train.py
```

### 2. Export public model and private user keys

```bash
python export_public_and_keys.py
```

### 3. Test one device

```bash
python test_with_external_key.py --input_audio 3s_audio_record.wav --key_path user_keys/user_3_secret.pt --public_generator_ckpt exported_public_model/public_generator.pth --detector_ckpt checkpoints/stageC_blind_detector_epoch_8.pth --output_audio outputs/test_user3_watermarked.wav
```

### 4. Run robustness evaluation

```bash
python evaluate_robustness.py
```

