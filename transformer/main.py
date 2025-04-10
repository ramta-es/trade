import argparse
import torch
from transformer.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from transformer.video_predictor import SwinTransformerVideoPredictor
from transformer.trainer import Trainer
from transformer.use_videodataset import load_video_data  # Import the function

def main():
    parser = argparse.ArgumentParser(description="Train a Swin Transformer model for image or video tasks.")
    parser.add_argument('--model_type', type=str, choices=['image', 'video'], required=True,
                        help="Type of model to train ('image' or 'video').")
    parser.add_argument('--data', type=str, nargs='+', required=True,
                        help="List of file paths or preloaded data.")
    parser.add_argument('--labels', type=int, nargs='+', required=True,
                        help="List of labels corresponding to the data.")
    parser.add_argument('--sequence_length', type=int, default=5,
                        help="Number of frames in a video sequence (if model_type='video').")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and evaluation.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs to train.")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help="Device to use.")
    args = parser.parse_args()

    # Load data
    if args.model_type == 'video':
        dataloader = load_video_data(video_paths=args.data, labels=args.labels,
                                     sequence_length=args.sequence_length, batch_size=args.batch_size)
    else:
        from transformer.dataset import GeneralDataset  # Import GeneralDataset for image tasks
        dataset = GeneralDataset(data=args.data, labels=args.labels, is_video=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    if args.model_type == 'image':
        model = SwinTransformerSys(img_size=550, num_classes=len(set(args.labels)))
    elif args.model_type == 'video':
        model = SwinTransformerVideoPredictor(img_size=550, num_classes=len(set(args.labels)), sequence_length=args.sequence_length)

    # Initialize and run the trainer
    trainer = Trainer(model=model, dataloader=dataloader, lr=args.lr, num_epochs=args.num_epochs, device=args.device)
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()