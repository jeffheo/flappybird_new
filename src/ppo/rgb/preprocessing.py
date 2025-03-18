import cv2

class ImagePreprocessor:
    """Preprocessor for RGB image observations from Flappy Bird."""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def preprocess_frame(self, frame):
        return self._basic_preprocess(frame)
            
    def _basic_preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.target_size[1], self.target_size[0]), 
                            interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        return normalized
        
    def reset(self):
        return