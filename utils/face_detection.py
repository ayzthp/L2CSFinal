import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple, Optional


class FaceDetector:
    """Face detection using MediaPipe Face Detection"""
    
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in image and return bounding boxes
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of tuples (x1, y1, x2, y2, confidence) for each detected face
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                # Convert relative coordinates to absolute pixels
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                confidence = detection.score[0]
                faces.append((x1, y1, x2, y2, confidence))
        
        return faces
    
    def detect_faces_from_file(self, image_path: str) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of tuples (x1, y1, x2, y2, confidence) for each detected face
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        return self.detect_faces(image)


def face_detection_per_image(image_paths: List[str], 
                           min_detection_confidence: float = 0.5) -> dict:
    """
    Perform face detection on multiple images
    
    Args:
        image_paths: List of image file paths
        min_detection_confidence: Minimum confidence threshold for detection
        
    Returns:
        Dictionary mapping image paths to lists of detected face bounding boxes
    """
    detector = FaceDetector(min_detection_confidence)
    results = {}
    
    for image_path in image_paths:
        try:
            faces = detector.detect_faces_from_file(image_path)
            # Convert to numpy array format for compatibility
            results[image_path] = np.array(faces) if faces else np.array([])
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results[image_path] = np.array([])
    
    return results 