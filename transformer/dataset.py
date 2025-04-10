import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GeneralDataset(Dataset):
    def __init__(self, data, labels, is_video=False, sequence_length=5, transform=None):
        """
        General dataset for both image and video tasks.

        Args:
            data (list): List of file paths or preloaded data.
            labels (list): List of labels corresponding to the data.
            is_video (bool): Whether the dataset is for video tasks.
            sequence_length (int): Number of frames in a video sequence (if is_video=True).
            transform (callable): Transformations to apply to the data.
        """
        self.data = data
        self.labels = labels
        self.is_video = is_video
        self.sequence_length = sequence_length
        self.resize = transforms.Resize((550, 550))  # Resize to 550x550 pixels
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_video:
            # Load a sequence of frames for video tasks
            frames = [self._load_frame(self.data[idx], frame_idx) for frame_idx in range(self.sequence_length)]
            frames = torch.stack(frames)  # Shape: (T, C, H, W)
            label = self.labels[idx]
            return frames, label
        else:
            # Load a single image for image tasks
            image = self._load_image(self.data[idx])
            label = self.labels[idx]
            return image, label

    def _load_image(self, path):
        """
        Load an image from the given path and apply transformations.

        Args:
            path (str): Path to the image file.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        image = Image.open(path).convert("RGB")  # Open image and ensure it's RGB
        image = self.resize(image)  # Resize to 550x550
        if self.transform:
            image = self.transform(image)
        return image

    def _load_frame(self, path, frame_idx):
        """
        Load a specific frame from a video file.

        Args:
            path (str): Path to the video file.
            frame_idx (int): Index of the frame to load.

        Returns:
            torch.Tensor: Transformed frame tensor.
        """
        # Replace this with actual video frame loading logic
        raise NotImplementedError("Replace this method with actual video frame loading logic.")