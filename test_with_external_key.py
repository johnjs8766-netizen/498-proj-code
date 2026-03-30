import os
import argparse
import torch
import torchaudio
import torch.nn.functional as F

from models import PublicHybridWatermarker, HybridBlindDetector


def load_audio(audio_path, target_sr=16000):
    wav, sr = torchaudio.load(audio_path)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = torch.clamp(wav, -1.0, 1.0)
    return wav, target_sr


def pad_or_trim(wav, target_len):
    cur_len = wav.size(-1)
    if cur_len > target_len:
        wav = wav[:, :target_len]
    elif cur_len < target_len:
        wav = F.pad(wav, (0, target_len - cur_len))
    return wav


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audio", type=str, required=True)
    parser.add_argument("--key_path", type=str, required=True)
    parser.add_argument("--public_generator_ckpt", type=str, required=True)
    parser.add_argument("--detector_ckpt", type=str, required=True)
    parser.add_argument("--output_audio", type=str, default="outputs/watermarked.wav")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--num_devices", type=int, default=6)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # ------------------------------------
    # Load public generator + detector
    # ------------------------------------
    generator = PublicHybridWatermarker(
        n_fft=512,
        hop_length=160,
        win_length=400,
        alpha_stft=0.05,
        alpha_wave=0.02,
        code_time_steps=16,
        cond_dim=32
    ).to(device)

    detector = HybridBlindDetector(
        num_devices=args.num_devices
    ).to(device)

    generator.load_state_dict(torch.load(args.public_generator_ckpt, map_location=device))
    detector.load_state_dict(torch.load(args.detector_ckpt, map_location=device))

    generator.eval()
    detector.eval()

    # ------------------------------------
    # Load user secret
    # ------------------------------------
    secret = torch.load(args.key_path, map_location=device)
    tf_code = secret["tf_code"].to(device)       # (F,Tc)
    wave_key = secret["wave_key"].to(device)     # (cond_dim,)
    true_user_id = secret.get("user_id", None)

    # ------------------------------------
    # Load audio
    # ------------------------------------
    wav, sr = load_audio(args.input_audio, target_sr=args.sample_rate)
    target_len = int(args.sample_rate * args.duration)
    wav = pad_or_trim(wav, target_len)
    wav = wav.unsqueeze(0).to(device)   # (1,1,T)

    # ------------------------------------
    # Watermark + classify
    # ------------------------------------
    with torch.no_grad():
        watermarked_audio, residual_audio, _, _, _ = generator(wav, tf_code, wave_key)
        logits = detector(watermarked_audio)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    # ------------------------------------
    # Save outputs
    # ------------------------------------
    os.makedirs(os.path.dirname(args.output_audio) or ".", exist_ok=True)

    torchaudio.save(
        args.output_audio,
        watermarked_audio.squeeze(0).cpu(),
        args.sample_rate
    )

    residual_path = os.path.splitext(args.output_audio)[0] + "_residual.wav"
    residual_amp_path = os.path.splitext(args.output_audio)[0] + "_residual_x20.wav"

    torchaudio.save(
        residual_path,
        residual_audio.squeeze(0).cpu(),
        args.sample_rate
    )
    torchaudio.save(
        residual_amp_path,
        torch.clamp(residual_audio * 20.0, min=-1.0, max=1.0).squeeze(0).cpu(),
        args.sample_rate
    )

    # ------------------------------------
    # Print result
    # ------------------------------------
    print("\n===== Inference Result =====")
    print(f"Input audio        : {args.input_audio}")
    print(f"Key file           : {args.key_path}")
    print(f"True user id (key) : {true_user_id}")
    print(f"Predicted label    : {pred}")
    print(f"Watermarked audio  : {args.output_audio}")
    print(f"Residual audio     : {residual_path}")
    print(f"Residual x20 audio : {residual_amp_path}")

    print("\nClass probabilities:")
    for i in range(args.num_devices):
        print(f"  Device {i}: {probs[0, i].item():.6f}")


if __name__ == "__main__":
    main()
