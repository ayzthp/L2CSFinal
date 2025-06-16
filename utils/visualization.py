import cv2
import numpy as np
from typing import List, Tuple
import math
from PIL import Image, ImageDraw, ImageFont


def draw_bbox(image: np.ndarray, 
              detections: np.ndarray, 
              show_class: bool = True,
              show_conf: bool = True,
              color: Tuple[int, int, int] = (0, 255, 0),
              thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image as numpy array
        detections: Array of detections (x1, y1, x2, y2, confidence, ...)
        show_class: Whether to show class labels
        show_conf: Whether to show confidence scores
        color: RGB color for bounding boxes
        thickness: Line thickness for bounding boxes
        
    Returns:
        Image with bounding boxes drawn
    """
    result_image = image.copy()
    
    if len(detections) == 0:
        return result_image
    
    for detection in detections:
        if len(detection) < 4:
            continue
            
        x1, y1, x2, y2 = detection[:4].astype(int)
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness)
        
        # Add confidence score if available and requested
        if show_conf and len(detection) >= 5:
            confidence = detection[4]
            text = f"{confidence:.2f}"
            
            # Calculate text size
            font_scale = 0.6
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw text background
            cv2.rectangle(
                result_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                result_image,
                text,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
    
    return result_image


def draw_detected_gazes(image: np.ndarray,
                       boxes: np.ndarray,
                       pitches: List[float],
                       yaws: List[float],
                       arrow_length: int = 100,
                       arrow_color: Tuple[int, int, int] = (255, 0, 0),
                       arrow_thickness: int = 3) -> np.ndarray:
    """
    Draw gaze estimation arrows on detected faces
    
    Args:
        image: Input image as numpy array
        boxes: Face bounding boxes (x1, y1, x2, y2)
        pitches: List of pitch angles in degrees
        yaws: List of yaw angles in degrees
        arrow_length: Length of gaze arrows in pixels
        arrow_color: RGB color for arrows
        arrow_thickness: Thickness of arrows
        
    Returns:
        Image with gaze arrows drawn
    """
    result_image = image.copy()
    
    for i, (box, pitch, yaw) in enumerate(zip(boxes, pitches, yaws)):
        x1, y1, x2, y2 = box.astype(int)
        
        # Calculate face center
        face_center_x = (x1 + x2) // 2
        face_center_y = (y1 + y2) // 2
        
        # Convert angles to radians
        pitch_rad = math.radians(pitch)
        yaw_rad = math.radians(yaw)
        
        # Calculate arrow end point
        # Invert directions to match natural gaze interpretation
        # Positive yaw = looking right, Positive pitch = looking down
        dx = -arrow_length * math.sin(yaw_rad)  # Invert X axis for left/right
        dy = -arrow_length * math.sin(pitch_rad)  # Invert Y axis for up/down
        
        end_x = int(face_center_x + dx)
        end_y = int(face_center_y + dy)
        
        # Draw arrow
        cv2.arrowedLine(
            result_image,
            (face_center_x, face_center_y),
            (end_x, end_y),
            arrow_color,
            arrow_thickness,
            tipLength=0.3
        )
        
        # Draw face bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add gaze angle text
        text = f"Yaw:{yaw:.1f}° Pitch:{pitch:.1f}°"
        font_scale = 0.5
        font_thickness = 1
        
        # Calculate text position (above the bounding box)
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        text_x = x1
        text_y = y1 - 10
        
        # Draw text background
        cv2.rectangle(
            result_image,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(
            result_image,
            text,
            (text_x, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    return result_image


def draw_gaze_on_face_crop(face_image: np.ndarray,
                          pitch: float,
                          yaw: float,
                          arrow_length: int = 50) -> np.ndarray:
    """
    Draw gaze arrow on a cropped face image
    
    Args:
        face_image: Cropped face image
        pitch: Pitch angle in degrees
        yaw: Yaw angle in degrees
        arrow_length: Length of arrow in pixels
        
    Returns:
        Face image with gaze arrow
    """
    result_image = face_image.copy()
    h, w = face_image.shape[:2]
    
    # Face center
    center_x, center_y = w // 2, h // 2
    
    # Convert angles to radians and calculate arrow end point
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    dx = arrow_length * math.sin(yaw_rad)
    dy = arrow_length * math.sin(pitch_rad)
    
    end_x = int(center_x + dx)
    end_y = int(center_y + dy)
    
    # Draw arrow
    cv2.arrowedLine(
        result_image,
        (center_x, center_y),
        (end_x, end_y),
        (255, 0, 0),
        2,
        tipLength=0.3
    )
    
    return result_image


def create_visualization_grid(images: List[np.ndarray],
                            titles: List[str] = None,
                            grid_cols: int = 3) -> np.ndarray:
    """
    Create a grid visualization of multiple images
    
    Args:
        images: List of images to display
        titles: Optional list of titles for each image
        grid_cols: Number of columns in the grid
        
    Returns:
        Combined grid image
    """
    if not images:
        return np.array([])
    
    num_images = len(images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    # Get image dimensions (assume all images have same size)
    h, w = images[0].shape[:2]
    channels = images[0].shape[2] if len(images[0].shape) == 3 else 1
    
    # Create grid canvas
    if channels == 3:
        grid_image = np.zeros((grid_rows * h, grid_cols * w, 3), dtype=np.uint8)
    else:
        grid_image = np.zeros((grid_rows * h, grid_cols * w), dtype=np.uint8)
    
    # Place images in grid
    for i, img in enumerate(images):
        row = i // grid_cols
        col = i % grid_cols
        
        y_start = row * h
        y_end = y_start + h
        x_start = col * w
        x_end = x_start + w
        
        # Resize image if necessary
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        
        grid_image[y_start:y_end, x_start:x_end] = img
        
        # Add title if provided
        if titles and i < len(titles):
            cv2.putText(
                grid_image,
                titles[i],
                (x_start + 10, y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
    
    return grid_image 