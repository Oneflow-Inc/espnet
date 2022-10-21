import oneflow as torch


def pytorch_cudnn_version() -> str:
    message = (
        f"pytorch.version={torch.__version__}, "
        f"cuda.available={torch.cuda.is_available()}, "
    )
    return message
