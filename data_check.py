# -*- coding: utf-8 -*- 

import resampy
import numpy as np
import torch
import torchaudio
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(
    y,
    n_fft=1024,
    num_mels=80,
    sampling_rate=24000,
    hop_size=240,
    win_size=1024,
    fmin=0,
    fmax=8000,
    center=False,
    output_energy=False,
):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # version 1
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True,
                      return_complex=True)
    spec = torch.view_as_real(spec)

    #spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
    #                  center=center, pad_mode='reflect', normalized=False, onesided=True)

    # version 2
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_size,
        n_mels=num_mels,
        center=center,
        power=1,
        f_min=fmin, # to match matcha :X
        f_max=fmax,
        norm='slaney',
        mel_scale='slaney',
    )
    Mel_Spec = mel_spec(y)

    #spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))   # 1e-9 ??
    spec = torch.sqrt(spec.pow(2).sum(-1))
    mel_spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)

    mel_spec = spectral_normalize_torch(mel_spec)
    print(mel_spec)
    mel_spec = spectral_normalize_torch(Mel_Spec)
    print(mel_spec)

    if output_energy:
        energy = torch.norm(spec, dim=1)
        return mel_spec, energy
    else:
        return mel_spec


wav_path = 'f577cc4acd0f11ed835900163e124273+纸上的姐妹+纸上的姐妹第96集+00000120+00665350+00667850.wav'
audio, sr = load_wav(wav_path)
if sr != 24000:
    audio = resampy.resample(audio, sr, 24000)
audio = audio / MAX_WAV_VALUE
audio = normalize(audio) * 0.95
audio = torch.FloatTensor(audio).unsqueeze(0)

mel = mel_spectrogram(audio)
#print(mel.shape, mel)

