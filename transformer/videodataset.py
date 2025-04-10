import cv2
import torch
from transformer.dataset import GeneralDataset
from PIL import Image  # Fix the unresolved reference issue

class VideoDataset(GeneralDataset):
    def _load_frame(self, path, frame_idx):
        """
        Load a specific frame from a video file.

        Args:
            path (str): Path to the video file.
            frame_idx (int): Index of the frame to load.

        Returns:
            torch.Tensor: Transformed frame tensor.
        """
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        frame = Image.fromarray(frame)  # Convert to PIL Image
        frame = self.resize(frame)  # Resize to 550x550
        if self.transform:
            frame = self.transform(frame)
        return frame