import torchaudio

from torch import Tensor
from torchaudio.functional import add_noise
from torchaudio.utils import download_asset

from hw_asr.augmentations.base import AugmentationBase


class AddBackgroundNoise(AugmentationBase):
    def __init__(self, snr_dbs, *args, **kwargs):
        """
        Add background noise to the input audio.
        :param snr_dbs: signal-to-noise ratio
        """
        sample_noise = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
        self.noise, _ = torchaudio.load(sample_noise)
        self.snr_dbs = snr_dbs

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        x = add_noise(x, self.noise[:x.shape[1]], self.snr_dbs)
        return x.squeeze(1)
