import torch
import torch.nn as nn
import torchaudio.functional as F_audio

class AdaptiveWatermarkPipeline(nn.Module):
    def __init__(self, generators, detector, internal_sr=16000):
        super().__init__()
        self.generators = generators 
        self.detector = detector
        self.internal_sr = internal_sr

    def embed_watermark(self, original_audio, orig_sr, device_label, alpha=0.01):
        # Downsample to 16k
        audio_16k = original_audio if orig_sr == self.internal_sr else F_audio.resample(original_audio, orig_sr, self.internal_sr)
        
        # Select the specific generator for this device
        generator = self.generators[device_label]
        watermark_16k = generator(audio_16k)
        
        # Upsample watermark back to native SR
        watermark_orig_sr = watermark_16k if orig_sr == self.internal_sr else F_audio.resample(watermark_16k, self.internal_sr, orig_sr)
        
        if watermark_orig_sr.size(-1) != original_audio.size(-1):
            watermark_orig_sr = watermark_orig_sr[..., :original_audio.size(-1)]
            
        return original_audio + (alpha * watermark_orig_sr)

    def detect_watermark(self, audio, orig_sr):
        audio_16k = audio if orig_sr == self.internal_sr else F_audio.resample(audio, orig_sr, self.internal_sr)
        return self.detector(audio_16k)