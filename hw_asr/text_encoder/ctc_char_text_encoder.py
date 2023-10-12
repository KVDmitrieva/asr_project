from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
        result = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            next_char = self.ind2char[ind]
            if next_char != self.EMPTY_TOK and next_char != last_char:
                result.append(next_char)

            last_char = next_char

        return ''.join(result)


    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)
        hypos: List[Hypothesis] = [Hypothesis('', 1.0)]

        for frame in probs:
            hypos = self._extend_and_merge(frame, hypos, probs_length)
            hypos = self._truncate(hypos, beam_size)

        return sorted(hypos, key=lambda x: x.prob, reverse=True)

    def _extend_and_merge(self, frame, hypos, probs_length):
        new_hypos = defaultdict(float)
        for next_char_index, next_char_proba in enumerate(frame):
            for pref, pref_proba in hypos:
                new_pref = pref
                next_char = self.ind2char[next_char_index]
                last_char = pref[-1] if pref else self.EMPTY_TOK
                if next_char != self.EMPTY_TOK and next_char != last_char:
                    new_pref += next_char
                if len(new_pref) <= probs_length:
                    new_hypos[new_pref] += next_char_proba * pref_proba

        return [Hypothesis(text, probs) for text, probs in new_hypos.items()]

    @staticmethod
    def _truncate(hypos, beam_size):
        sorted_hypos = sorted(hypos, key=lambda x: x.prob, reverse=True)
        return sorted_hypos[:beam_size]
