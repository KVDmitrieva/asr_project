from torch import Tensor
from torchaudio.transforms import TimeStretch

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self, *args, **kwargs):
        """
        Stretch stft in time without modifying pitch for a given rate.
        :param hop_length: Length of hop between STFT windows
        :param n_freq: number of filter banks from stft
        :param fixed_rate: rate to speed up or slow down by
        """
        self._aug = TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)