import os
import gdown
import torch
import onnx
from pathlib import Path


def download_l2cs_pretrained_model(weights_dir: str = "pretrained_models") -> str:
    """Download L2CS pretrained model"""
    weights_folder = Path(weights_dir)
    weights_folder.mkdir(exist_ok=True)
    weights_path = weights_folder / "L2CSNet_gaze360.pkl"
    
    if weights_path.exists():
        print(f"Model already exists at {weights_path}")
        return str(weights_path)
    
    model_url = "https://drive.google.com/uc?id=18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ"
    print("Downloading L2CS pretrained model...")
    gdown.download(model_url, str(weights_path), quiet=False)
    
    return str(weights_path)


def onnx_check_and_simplify(onnx_model) -> onnx.ModelProto:
    """Check and simplify ONNX model"""
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid")
    except Exception as e:
        print(f"ONNX model validation failed: {e}")
    return onnx_model


def export_l2cs_to_onnx(model: torch.nn.Module, 
                       output_path: str,
                       input_size: tuple = (448, 448)) -> None:
    """Export L2CS model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
    
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True, do_constant_folding=True,
        input_names=['inputs0'], output_names=['outputs0', 'outputs1']
    )
    
    onnx_model = onnx.load(output_path)
    onnx_model = onnx_check_and_simplify(onnx_model)
    onnx.save(onnx_model, output_path)
    print(f"ONNX model saved to {output_path}")


def load_l2cs_model(weights_path: str, device: str = 'cpu') -> torch.nn.Module:
    """Load L2CS model from pretrained weights"""
    from l2cs.model import L2CS
    from torchvision.models.resnet import Bottleneck
    
    model = L2CS(Bottleneck, [3, 4, 6, 3], 90)
    state_dict = torch.load(weights_path, map_location=device)
    
    # Filter out fc_finetune layer if it doesn't match
    model_dict = model.state_dict()
    filtered_dict = {}
    
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                filtered_dict[k] = v
            else:
                print(f"Skipping layer {k} due to shape mismatch: {model_dict[k].shape} vs {v.shape}")
        else:
            print(f"Skipping unknown layer: {k}")
    
    model.load_state_dict(filtered_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"L2CS model loaded from {weights_path}")
    return model 