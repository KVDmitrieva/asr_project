import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    spectrogram, text_encoded = [], []
    text, text_encoded_length = [], []

    for item in dataset_items:
        text.append(item["text"])
        spectrogram.append(item["spectrogram"].squeeze(0).T)
        text_encoded.append(item["text_encoded"].squeeze(0))
        text_encoded_length.append(item["text_encoded"].shape[1])

    return {
        "text" : text,
        "spectrogram":  torch.nn.utils.rnn.pad_sequence(spectrogram, batch_first=True).transpose(1, 2),
        "text_encoded" : torch.nn.utils.rnn.pad_sequence(text_encoded, batch_first=True),
        "text_encoded_length" : torch.tensor(text_encoded_length)
    }