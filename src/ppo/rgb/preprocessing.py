import numpy as np
import cv2
from collections import deque

class ImagePreprocessor:
    """Preprocessor for RGB image observations from Flappy Bird."""
    
    def __init__(
        self, 
        method='enhanced', 
        target_size=(84, 84), 
        use_frame_stack=True, 
        frame_stack_size=4,
    ):
        """
        Initialize the preprocessor.
        
        Args:
            method (str): Preprocessing method: 'basic', 'enhanced'
            target_size (tuple): Target size for processed images (height, width)
            use_frame_stack (bool): Whether to use frame stacking
            frame_stack_size (int): Number of frames to stack
        """
        self.method = method
        self.target_size = target_size
        self.use_frame_stack = use_frame_stack
        self.frame_stack_size = frame_stack_size if use_frame_stack else 1
        
        # Initialize frame buffer for stacking
        self.frame_buffer = deque(maxlen=frame_stack_size)
        
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame.
        
        Args:
            frame (numpy.ndarray): Raw RGB frame from the environment
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        if self.method == 'basic':
            return self._basic_preprocess(frame)
        elif self.method == 'enhanced':
            return self._enhanced_preprocess(frame)
        
        else:
            # Default to enhanced
            return self._enhanced_preprocess(frame)
            
    def _basic_preprocess(self, frame):
        """Basic preprocessing: grayscale and resize"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to target size if not skipping
        if not self.skip_resize:
            resized = cv2.resize(gray, (self.target_size[1], self.target_size[0]), 
                                interpolation=cv2.INTER_AREA)
            normalized = resized / 255.0
        else:
            # Just normalize without resizing
            normalized = gray / 255.0
        
        return normalized
        
    def _enhanced_preprocess(self, frame):
        """Enhanced preprocessing with better feature preservation"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive histogram equalization to enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # Resize if not skipping
        if not self.skip_resize:
            # Resize to target size (using INTER_LINEAR for better quality)
            resized = cv2.resize(blurred, (self.target_size[1], self.target_size[0]), 
                                interpolation=cv2.INTER_LINEAR)
            normalized = resized / 255.0
        else:
            # Just normalize without resizing
            normalized = blurred / 255.0
        
        return normalized
        
        
    def process(self, frame, keep_rgb=False):
        """
        Process a frame and handle frame stacking.
        
        Args:
            frame (numpy.ndarray): Raw RGB frame from the environment
            
        Returns:
            numpy.ndarray: Processed frame or stack of frames
        """
        
        processed_frame = self.preprocess_frame(frame)
        
        if not self.use_frame_stack:
            return processed_frame[np.newaxis, :, :]
            
        # Update frame buffer
        if len(self.frame_buffer) == 0:
            for _ in range(self.frame_stack_size):
                self.frame_buffer.append(processed_frame)
        else:
            self.frame_buffer.append(processed_frame)
            
        stacked_frames = np.stack(list(self.frame_buffer), axis=0)
        
        return stacked_frames
        
    def reset(self):
        """Reset the frame buffer when environment resets"""
        self.frame_buffer.clear()