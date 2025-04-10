import torch
from torch.utils.data import DataLoader
from transformer.videodataset import VideoDataset

def load_video_data(video_paths, labels, sequence_length=5, batch_size=2, shuffle=True):
    """
    Load video data using the VideoDataset class.

    Args:
        video_paths (list): List of video file paths.
        labels (list): List of labels corresponding to the videos.
        sequence_length (int): Number of frames in a video sequence.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader for the video dataset.
    """
    video_dataset = VideoDataset(data=video_paths, labels=labels, is_video=True, sequence_length=sequence_length)
    dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader