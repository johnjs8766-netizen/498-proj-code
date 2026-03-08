import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
import os

class WatermarkDataset(Dataset):
    def __init__(self, csv_path, target_sr=16000, duration=3.0, mode='train', test_ratio=0.1):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
            
        self.df = pd.read_csv(csv_path)
        self.base_dir = os.path.dirname(csv_path)
        
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'].str.contains(mode, na=False)].reset_index(drop=True)
        else:
            # Shuffle the data predictably
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            split_index = int(len(self.df) * (1.0 - test_ratio))
            
            # Slice the dataframe 
            if mode == 'train':
                self.df = self.df.iloc[:split_index].reset_index(drop=True)
            elif mode == 'test':
                self.df = self.df.iloc[split_index:].reset_index(drop=True)

        possible_path_cols = ['filename', 'file_path', 'path', 'wav_path', 'filepath', 'audio_path']
        self.path_col = None
        
        for col in possible_path_cols:
            if col in self.df.columns:
                self.path_col = col
                break
        
        if self.path_col is None:
            raise KeyError(f"Could not find a path column. Available columns: {self.df.columns.tolist()}")
            
        self.target_sr = target_sr
        self.target_length = int(target_sr * duration)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.loc[idx, self.path_col]
        
        # Path resolution
        if "dataset_libritts" in filename or "dataset" in filename:
            audio_path = filename.lstrip('./') 
        else:
            audio_path = os.path.join(self.base_dir, filename)
        
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.target_length), 0

        # Preprocessing (Mono + Resample + Pad/Trim)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
            
        if waveform.size(1) > self.target_length:
            waveform = waveform[:, :self.target_length]
        else:
            padding = self.target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform, 0