import cv2

class ImagePreprocessor:
    """Preprocessor for RGB image observations from Flappy Bird."""
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize the preprocessor.
        
        Args:
            target_size (tuple): Target size for processed images (height, width)
        """
        self.target_size = target_size
        
    def preprocess_frame(self, frame):
        """
        Preprocess a single frame.
        
        Args:
            frame (numpy.ndarray): Raw RGB frame from the environment
            
        Returns:
            numpy.ndarray: Preprocessed frame
        """
        return self._basic_preprocess(frame)
            
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
        
    def reset(self):
        return