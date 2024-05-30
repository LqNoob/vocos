from dataclasses import dataclass

import os
import glob
import math
import librosa
import random
import numpy as np
import torch
import torchaudio
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    sampling_rate: int
    n_fft: int
    win_length: int
    hop_length: int
    num_samples: int
    batch_size: int
    num_workers: int


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig, input_wavs_dir: str, input_mels_dir: str):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

        self.input_wavs_dir = input_wavs_dir
        self.input_mels_dir = input_mels_dir

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        dataset = VocosDataset(cfg, self.input_wavs_dir, self.input_mels_dir, train=train)
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train, pin_memory=True)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    #def val_dataloader(self) -> DataLoader:
    #    return self._get_dataloder(self.val_config, train=False)


class VocosDataset(Dataset):
    def __init__(self, cfg: DataConfig, input_wavs_dir, input_mels_dir, train: bool):

        with open(cfg.filelist_path, 'r', encoding='utf-8') as fi:
            wav_filelist = [os.path.join(input_wavs_dir, x.split('|')[0]) for x in fi.read().split('\n') if len(x) > 0]

        self.filelist = []
        for wav_path in wav_filelist:
            mel_path = glob.glob(os.path.join(input_mels_dir, '*', os.path.basename(wav_path).replace('.wav', '.npy')))
            if len(mel_path) == 1 and os.path.isfile(mel_path[-1]):
                assert os.path.exists(wav_path) and os.path.exists(mel_path[-1])
                self.filelist.append([wav_path, mel_path[-1]])

        self.sampling_rate = cfg.sampling_rate
        self.num_samples   = cfg.num_samples
        self.train = train

        self.n_fft      = cfg.n_fft
        self.win_length = cfg.win_length
        self.hop_length = cfg.hop_length

        self.fine_tuning = True

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> torch.Tensor:
        audio_path, mel_path = self.filelist[index]
        y, sr = librosa.load(audio_path, sr=None)

        if not self.fine_tuning:
            y = normalize(y) * 0.95

        y = torch.FloatTensor(y).unsqueeze(0)

        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)

        mel = np.load(mel_path)
        mel = torch.from_numpy(mel)
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        assert mel.ndim <= 3 and mel.shape[-2] == 80
        #if y.size(-1) < (mel.size(-1) - 1) * self.hop_length:
        #    y = torch.nn.functional.pad(y, (0, (mel.size(-1) - 1) * self.hop_length - y.size(-1)), mode="replicate")
        
        frames_per_seg = math.ceil(self.num_samples // self.hop_length)
        if y.size(-1) >= self.num_samples:
            mel_start = random.randint(0, mel.size(-1) - frames_per_seg - 1)
            mel = mel[:, :, mel_start:mel_start + frames_per_seg]
            y   = y[:, mel_start * self.hop_length:mel_start * self.hop_length + self.num_samples]
        elif self.train:
            mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(-1)), 'constant')
            y   = torch.nn.functional.pad(y, (0, self.num_samples - y.size(-1)), 'constant')
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]

        return y[0], mel.squeeze(0)
