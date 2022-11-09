import copy
from typing import Optional, Tuple, Union

import logging
import humanfriendly
import numpy as np
import oneflow as torch
from oneflow_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.layers.oneflow_stft import FlowStft
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet.nets.pytorch_backend.frontends.frontend import Frontend


class DefaultFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
        use_flow_stft: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)
        self.hop_length = hop_length
        self.use_flow_stft = use_flow_stft

        if apply_stft:
            if self.use_flow_stft:
                logging.info("use oneflow STFT")
                self.stft = FlowStft(
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    center=center,
                    window=window,
                    normalized=normalized,
                    onesided=onesided,
                )
            else:
                logging.info("use PyTorch STFT")
                self.stft = Stft(
                    n_fft=n_fft,
                    win_length=win_length,
                    hop_length=hop_length,
                    center=center,
                    window=window,
                    normalized=normalized,
                    onesided=onesided,
                )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None

        self.logmel = LogMel(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.n_mels = n_mels
        self.frontend_type = "default"

    def output_size(self) -> int:
        return self.n_mels

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor,
        cpu_input: torch.Tensor, cpu_input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(input, input_lengths, cpu_input, cpu_input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real**2 + input_stft.imag**2

        # 5. Feature transform e.g. Stft -> Log-Mel-Fbank
        # input_power: (Batch, [Channel,] Length, Freq)
        #       -> input_feats: (Batch, Length, Dim)
        input_feats, _ = self.logmel(input_power, feats_lens)

        return input_feats, feats_lens

    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor,
        cpu_input: torch.Tensor, cpu_input_lengths: torch.Tensor
    ) -> torch.Tensor:
        
        if self.use_flow_stft:
            input_stft, feats_lens = self.stft(input, input_lengths)
        else:
            if input.is_cuda:
                device_str = "cuda"
            else:
                device_str = "cpu"
            torch_input = torch.utils.tensor.to_torch(cpu_input).to(device_str)
            torch_input_lengths = torch.utils.tensor.to_torch(cpu_input_lengths).to(device_str)
            torch_input_stft, torch_feats_lens = self.stft(torch_input, torch_input_lengths)
            input_stft = torch.utils.tensor.from_torch(torch_input_stft.cpu()).to(device_str)
            feats_lens = torch.utils.tensor.from_torch(torch_feats_lens.cpu()).to(device_str)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens
