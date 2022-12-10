import glob
import io
from typing import List

import librosa
import librosa.effects
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from speech_embedder_net import SpeechEmbedder


def compute_spectrogram(audio_data, sr, n_fft, hop_length, window_length, n_mels, **kwargs):    
    spec = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)
    
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_basis, mag_spec)
    
    #db mel spectrogram
    mel_db = librosa.amplitude_to_db(mel_spec).T
    
    return mel_db

def fix_length(audio, desired_samples):
    n_samples = len(audio)
    if n_samples > desired_samples:
        start = (n_samples - desired_samples) // 2
        end = (n_samples + desired_samples) // 2
        return audio[start : end]
    
    elif n_samples < desired_samples:
        pad = desired_samples - n_samples
        left_pad = pad//2 
        right_pad = pad - left_pad
        return np.pad(audio, pad_width=(left_pad, right_pad))

    return audio 

def load_audio(utterance: bytes, sr: int = None, top_db: float = None, desired_len_s: int = None):
    aud, sr= librosa.core.load(
        io.BytesIO(utterance),
        sr = sr
    )
    if top_db is not None:
        aud, _ = librosa.effects.trim(aud, top_db=10)
    
    if fix_length is not None:
        aud = fix_length(aud, desired_len_s * sr)

    return aud, sr

class SpeakerVerifier:
    def __init__(self, model_file, config):
        self.feature_ext_conf = config.get('feature_ext_conf', {})
        self.model_conf = config.get('model_conf', {})
        self.model = self.build_model(model_file)

    def build_model(self, model_file):
        embedder_net = SpeechEmbedder(self.feature_ext_conf.get('n_mels'), **self.model_conf)
        embedder_net.load_state_dict(torch.load(model_file))
        embedder_net.eval()
        return embedder_net

    def load_audio(self, utterance: bytes, sr: int = None, top_db: float = None, fixed_len: int = None):
        aud, sr= librosa.core.load(
            io.BytesIO(utterance),
            sr = self.feature_ext_conf.get('sr')
        )
        aud, _ = librosa.effects.trim(aud, top_db=10)
        return aud, sr
        
    def get_audio(self, enroll_utterances, verify_utterance):
        enroll_set = np.array([
            load_audio(
                data_bytes,
                sr = self.feature_ext_conf.get('sr'),
                top_db = self.feature_ext_conf.get('top_db'),
                desired_len_s = self.feature_ext_conf.get('desired_len_s') 
            )[0] for data_bytes in enroll_utterances
        ])

        verify_set = np.array([
            load_audio(
                verify_utterance, 
                sr = self.feature_ext_conf.get('sr'),
                top_db = self.feature_ext_conf.get('top_db'),
                desired_len_s = self.feature_ext_conf.get('desired_len_s') 
            )[0]
        ])

        return enroll_set, verify_set

    def verify(self, enroll_utterances: List[bytes], verify_utterance: bytes):
        enroll_set, verify_set = self.get_audio(enroll_utterances, verify_utterance)

        # (utt, time, mels)
        enroll_set = torch.Tensor(
            [compute_spectrogram(utt, **self.feature_ext_conf) for utt in enroll_set]
        )
        verify_set = torch.Tensor(
            [compute_spectrogram(utt, **self.feature_ext_conf) for utt in verify_set]
        )      
        # (utt, time, dim)
        enroll_emb = self.model(enroll_set)
        verify_emb = self.model(verify_set)

        # (time, dim)
        enroll_centroid = enroll_emb.mean(dim = 0)
        # print(enroll_emb)
        # print(verify_emb)

        distance = F.cosine_similarity(
            torch.flatten(enroll_centroid),
            torch.flatten(verify_emb), 
            dim=0
        )

        return distance.item()



    