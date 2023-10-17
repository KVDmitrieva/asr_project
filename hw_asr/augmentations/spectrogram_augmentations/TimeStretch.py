import torchaudio.transforms as t
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        """
        Stretch stft in time without modifying pitch for a given rate.
        :param hop_length: Length of hop between STFT windows
        :param n_freq: number of filter banks from stft
        :param fixed_rate: rate to speed up or slow down by
        """
        self._aug = t.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        return self._aug(data.squeeze(0)).unsqueeze(0)