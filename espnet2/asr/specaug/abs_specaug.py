from typing import Optional, Tuple

import oneflow as torch


class AbsSpecAug(torch.nn.Module):
    """Abstract class for the augmentation of spectrogram

    The process-flow:

    Frontend  -> SpecAug -> Normalization -> Encoder -> Decoder
    """

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError
