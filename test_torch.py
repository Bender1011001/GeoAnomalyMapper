try:
    import torch
    print('Torch imported successfully:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    print('Torch dtype:', torch.get_default_dtype())
except Exception as e:
    print('Torch import failed:', str(e))
    import traceback
    traceback.print_exc()