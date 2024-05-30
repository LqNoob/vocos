# -*- coding: utf-8 -*- 

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import pickle
import random
import torch
import torchaudio
import numpy as np
from vocos.pretrained import Vocos


def infer():
    audio_path = 'data/parastory_0001_speaker166_f0uvada_166lj44k_generated_e2e.wav'
    data, sample_rate = torchaudio.load(audio_path)
    
    #mel_path = 'data/parastory_0001_speaker166_f0uvada_166lj44k.npy'
    #mel_feat = torch.from_numpy(np.load(mel_path)).to(torch.float32)

    mel_path = 'data/parastory_0010_speaker166_f0uvada_166lj44k.pkl'
    with open(mel_path, 'rb') as fp:
        mel_feat = pickle.load(fp)
        mel_feat = np.concatenate(mel_feat, axis=-1)
    mel_feat = torch.from_numpy(mel_feat).to(torch.float32)
    
    #mel_path = 'data/f5699706cd0f11ed8c2500163e124273+纸上的姐妹+纸上的姐妹第9集+00000109+00715460+00723490.npy'
    #mel_feat = torch.from_numpy(np.load(mel_path)).unsqueeze(0)
    print(mel_feat.dtype, mel_feat.shape)

    # 
    config_path = 'checkpoints/cp_vocos_166_gta_discri/logs/lightning_logs/version_0/config.yaml'
    model_path = 'checkpoints/cp_vocos_166_gta_discri/vocos_checkpoint_epoch=828_step=1055000.ckpt'
    
    model = Vocos.from_pretrained(model_path, config_path)
    
    wav_data = model(data)
    torchaudio.save('test.wav', wav_data, sample_rate)
    print(wav_data)
    
    wav_data = model.decode(mel_feat)
    output_path = os.path.join('data', os.path.basename(mel_path)[:-4] + '_voco.wav')
    torchaudio.save(output_path, wav_data, sample_rate)
    print(wav_data)


def infer_copy_synthesis():

    from vocos.pretrained import Vocos
    
    config_path = 'checkpoints/cp_vocos_174_gt_22050/logs/lightning_logs/version_2/config.yaml'
    model_path  = 'checkpoints/cp_vocos_174_gt_22050/vocos_checkpoint_epoch=5714_step=1360000.ckpt'

    sample_rate = 22050
    vocos = Vocos.from_pretrained(model_path, config_path)

    audio_path = 'fadbf21ce05011ee98ca28b2bd2d5eb4+9152ff0ee0e211eeb2df00163e124273+00000001+00001870+00009590.wav'
    y, sr = torchaudio.load(audio_path)
    if y.size(0) > 1:  # mix to mono
        y = y.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=sample_rate)
    y_hat = vocos(y)

    print(os.path.basename(audio_path).replace('.wav', '_voco.wav'))
    torchaudio.save(os.path.basename(audio_path).replace('.wav', '_voco.wav'), y_hat, sample_rate)

#infer()
infer_copy_synthesis()

