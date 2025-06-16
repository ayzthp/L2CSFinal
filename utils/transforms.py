import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Tuple, Union


class ResizePad:
    """Resize image while maintaining aspect ratio and pad to target size"""
    
    def __init__(self, target_size: Tuple[int, int], fill_color: int = 0):
        """
        Args:
            target_size: (width, height) of target size
            fill_color: Color to use for padding (0-255)
        """
        self.target_size = target_size
        self.fill_color = fill_color
    
    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Apply resize and pad transformation
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Transformed PIL Image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get original dimensions
        orig_width, orig_height = image.size
        target_width, target_height = self.target_size
        
        # Calculate scaling factor to fit image while maintaining aspect ratio
        scale = min(target_width / orig_width, target_height / orig_height)
        
        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create new image with target size and fill color
        result_image = Image.new('RGB', self.target_size, color=(self.fill_color,) * 3)
        
        # Calculate position to paste resized image (center it)
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # Paste resized image onto result
        result_image.paste(resized_image, (paste_x, paste_y))
        
        return result_image


class GazeTransforms:
    """Collection of transforms for gaze estimation preprocessing"""
    
    @staticmethod
    def get_face_preprocessing_transforms(input_size: Tuple[int, int] = (448, 448),
                                        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                                        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        Get preprocessing transforms for face images
        
        Args:
            input_size: Target input size (width, height)
            mean: Normalization mean values for RGB channels
            std: Normalization std values for RGB channels
            
        Returns:
            Composed transforms
        """
        return transforms.Compose([
            ResizePad(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    @staticmethod
    def denormalize_tensor(tensor: torch.Tensor,
                          mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                          std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> torch.Tensor:
        """
        Denormalize a normalized tensor for visualization
        
        Args:
            tensor: Normalized tensor
            mean: Mean values used for normalization
            std: Std values used for normalization
            
        Returns:
            Denormalized tensor
        """
        denorm_tensor = tensor.clone()
        for t, m, s in zip(denorm_tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(denorm_tensor, 0, 1)
    
    @staticmethod
    def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor to numpy image array
        
        Args:
            tensor: Input tensor (C, H, W)
            
        Returns:
            Numpy array image (H, W, C) in range [0, 255]
        """
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Convert to numpy and transpose to HWC format
        image = tensor.permute(1, 2, 0).cpu().numpy()
        
        # Convert to 0-255 range
        image = (image * 255).astype(np.uint8)
        
        return image


def extract_face_region(image: np.ndarray, 
                       bbox: Tuple[int, int, int, int],
                       expand_ratio: float = 0.15) -> np.ndarray:
    """
    Extract face region from image with optional expansion
    
    Args:
        image: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        expand_ratio: Ratio to expand bounding box
        
    Returns:
        Cropped face image
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Calculate expansion
    face_width = x2 - x1
    face_height = y2 - y1
    expand_w = int(face_width * expand_ratio)
    expand_h = int(face_height * expand_ratio)
    
    # Expand bounding box
    x1_exp = max(0, x1 - expand_w)
    y1_exp = max(0, y1 - expand_h)
    x2_exp = min(w, x2 + expand_w)
    y2_exp = min(h, y2 + expand_h)
    
    # Extract face region
    face_region = image[y1_exp:y2_exp, x1_exp:x2_exp]
    
    return face_region


def preprocess_face_for_gaze_estimation(face_image: Union[np.ndarray, Image.Image],
                                       target_size: Tuple[int, int] = (448, 448),
                                       mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                                       std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Preprocess face image for gaze estimation
    
    Args:
        face_image: Face image (numpy array or PIL Image)
        target_size: Target size (width, height)
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Preprocessed tensor ready for model input
    """
    transform = GazeTransforms.get_face_preprocessing_transforms(target_size, mean, std)
    
    if isinstance(face_image, np.ndarray):
        # Convert BGR to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = Image.fromarray(face_image)
    
    # Apply transforms
    tensor = transform(face_image)
    
    # Add batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def create_batch_from_faces(face_images: list, 
                          target_size: Tuple[int, int] = (448, 448),
                          mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                          std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> torch.Tensor:
    """
    Create a batch tensor from multiple face images
    
    Args:
        face_images: List of face images
        target_size: Target size for preprocessing
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Batch tensor (N, C, H, W)
    """
    batch_tensors = []
    
    for face_image in face_images:
        tensor = preprocess_face_for_gaze_estimation(face_image, target_size, mean, std)
        batch_tensors.append(tensor.squeeze(0))  # Remove batch dimension
    
    # Stack tensors to create batch
    batch_tensor = torch.stack(batch_tensors, dim=0)
    
    return batch_tensor 