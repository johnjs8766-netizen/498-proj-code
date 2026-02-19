import os
import torch
import torchaudio
import random
from torch.utils.data import Dataset

class LibriTTSDataset(Dataset):
    def __init__(self, root_dir="./data/LibriTTS", length=16000):
        """
        Args:
            root_dir: Path to the LibriTTS folder (e.g., ./data/LibriTTS)
            length: Fixed number of samples to crop/pad (default 1 sec @ 16kHz)
        """
        self.root_dir = root_dir
        self.length = length
        self.audio_files = []

        # 1. Recursive Scan
        # This works perfectly for the structure: subset -> speaker -> chapter -> wav
        print(f"Scanning {root_dir} for .wav files...")
        
        if not os.path.exists(root_dir):
            print(f"ERROR: Directory {root_dir} does not exist!")
            print("Please check if your path is correct relative to where you run the script.")
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                # Strictly filter for .wav (ignores .txt files shown in your screenshot)
                if file.endswith(".wav") and not file.startswith("._"): 
                    self.audio_files.append(os.path.join(root, file))
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No .wav files found in {root_dir}. Check your path.")
            
        print(f"Found {len(self.audio_files)} audio files.")
        print(f"Example file: {self.audio_files[0]}") # Sanity check

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        path = self.audio_files[idx]
        
        try:
            # 2. Load Audio
            # LibriTTS is usually 24kHz. torchaudio.load returns (waveform, sample_rate)
            audio, sr = torchaudio.load(path)
            
            # 3. Resample (Crucial for LibriTTS)
            # We standardize everything to 16kHz for the CELP encoder
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000).to(audio.device)
                audio = resampler(audio)

            # 4. Mix Stereo to Mono (just in case)
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # 5. Fixed Length Cropping/Padding
            # We need fixed input size for the batch training
            src_len = audio.shape[1]
            if src_len > self.length:
                # Random crop
                start = random.randint(0, src_len - self.length)
                audio = audio[:, start:start+self.length]
            elif src_len < self.length:
                # Pad with zeros
                padding = self.length - src_len
                audio = torch.nn.functional.pad(audio, (0, padding))

            # Squeeze channel dim: [1, T] -> [T]
            return audio.squeeze(0)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return silence in case of file corruption to prevent crash
            return torch.zeros(self.length)