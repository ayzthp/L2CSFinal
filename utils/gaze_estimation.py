import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List
import math


def softmax_temperature(tensor: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply softmax with temperature scaling"""
    return F.softmax(tensor / temperature, dim=1)


def convert_raw_gaze_predictions_to_angles(yaw_pred: np.ndarray, 
                                         pitch_pred: np.ndarray,
                                         num_bins: int = 90) -> Tuple[float, float]:
    """
    Convert raw gaze predictions to angle values
    
    Args:
        yaw_pred: Raw yaw predictions from model (90-dimensional vector)
        pitch_pred: Raw pitch predictions from model (90-dimensional vector)
        num_bins: Number of angle bins (default: 90)
        
    Returns:
        Tuple of (pitch_angle, yaw_angle) in degrees
    """
    # Convert to torch tensors if numpy arrays
    if isinstance(yaw_pred, np.ndarray):
        yaw_pred = torch.from_numpy(yaw_pred)
    if isinstance(pitch_pred, np.ndarray):
        pitch_pred = torch.from_numpy(pitch_pred)
    
    # Apply softmax to get probability distributions
    yaw_prob = F.softmax(yaw_pred, dim=-1)
    pitch_prob = F.softmax(pitch_pred, dim=-1)
    
    # Convert bin indices to angle values
    # Assuming angle range is [-90, 90] degrees divided into num_bins
    angle_bins = torch.linspace(-90, 90, num_bins)
    
    # Calculate expected angles using probability-weighted sum
    yaw_angle = torch.sum(yaw_prob * angle_bins, dim=-1)
    pitch_angle = torch.sum(pitch_prob * angle_bins, dim=-1)
    
    # Convert to scalar values
    if yaw_angle.numel() == 1:
        yaw_angle = yaw_angle.item()
    if pitch_angle.numel() == 1:
        pitch_angle = pitch_angle.item()
    
    return pitch_angle, yaw_angle


def angles_to_vector(yaw: float, pitch: float) -> Tuple[float, float, float]:
    """
    Convert yaw and pitch angles to 3D gaze vector
    
    Args:
        yaw: Yaw angle in degrees
        pitch: Pitch angle in degrees
        
    Returns:
        Tuple of (x, y, z) components of unit gaze vector
    """
    # Convert degrees to radians
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    
    # Calculate 3D vector components
    x = -math.cos(pitch_rad) * math.sin(yaw_rad)
    y = -math.sin(pitch_rad)
    z = -math.cos(pitch_rad) * math.cos(yaw_rad)
    
    return x, y, z


def vector_to_angles(x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Convert 3D gaze vector to yaw and pitch angles
    
    Args:
        x, y, z: Components of gaze vector
        
    Returns:
        Tuple of (yaw, pitch) in degrees
    """
    # Calculate pitch and yaw from vector components
    pitch = math.degrees(math.asin(-y))
    yaw = math.degrees(math.atan2(-x, -z))
    
    return yaw, pitch


class GazeEstimator:
    """Gaze estimation using L2CS-Net model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def estimate_gaze(self, face_image: torch.Tensor) -> Tuple[float, float]:
        """
        Estimate gaze angles for a single face image
        
        Args:
            face_image: Preprocessed face image tensor
            
        Returns:
            Tuple of (pitch, yaw) angles in degrees
        """
        with torch.no_grad():
            face_image = face_image.to(self.device)
            yaw_pred, pitch_pred = self.model(face_image)
            
            # Convert to numpy for processing
            yaw_pred = yaw_pred.cpu().numpy()
            pitch_pred = pitch_pred.cpu().numpy()
            
            # Convert raw predictions to angles
            pitch, yaw = convert_raw_gaze_predictions_to_angles(yaw_pred, pitch_pred)
            
        return pitch, yaw
    
    def batch_estimate_gaze(self, face_images: List[torch.Tensor]) -> List[Tuple[float, float]]:
        """
        Estimate gaze angles for multiple face images
        
        Args:
            face_images: List of preprocessed face image tensors
            
        Returns:
            List of (pitch, yaw) angle tuples in degrees
        """
        results = []
        for face_image in face_images:
            pitch, yaw = self.estimate_gaze(face_image)
            results.append((pitch, yaw))
        
        return results


def calculate_angular_error(pred_yaw: float, pred_pitch: float, 
                          gt_yaw: float, gt_pitch: float) -> float:
    """
    Calculate angular error between predicted and ground truth gaze
    
    Args:
        pred_yaw, pred_pitch: Predicted yaw and pitch angles
        gt_yaw, gt_pitch: Ground truth yaw and pitch angles
        
    Returns:
        Angular error in degrees
    """
    # Convert to 3D vectors
    pred_x, pred_y, pred_z = angles_to_vector(pred_yaw, pred_pitch)
    gt_x, gt_y, gt_z = angles_to_vector(gt_yaw, gt_pitch)
    
    # Calculate dot product
    dot_product = pred_x * gt_x + pred_y * gt_y + pred_z * gt_z
    
    # Clamp dot product to avoid numerical issues
    dot_product = max(-1.0, min(1.0, dot_product))
    
    # Calculate angular error
    angular_error = math.degrees(math.acos(abs(dot_product)))
    
    return angular_error 