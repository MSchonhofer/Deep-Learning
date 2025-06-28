import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json

from model import TextureAutoencoder
from data_processing import (TextureDataset, load_texture_tiles,
                             preprocess_images_from_directory,
                             load_images_from_directory)


def train_autoencoder(model, train_loader, val_loader=None, num_epochs=100,
                      learning_rate=0.001, device=None, save_dir='checkpoints'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    os.makedirs(save_dir, exist_ok=True)
    history = {'train_losses': [], 'val_losses': [] if val_loader else None, 'learning_rates': []}
    best_val_loss = float('inf')

    print(f"Starting training on {device}")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    print("-" * 50)

    early_stopper = EarlyStopping(patience=10, min_delta=0.0001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')

        avg_train_loss = train_loss / num_batches
        history['train_losses'].append(avg_train_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        val_loss = None
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
            val_loss /= len(val_loader)
            history['val_losses'].append(val_loss)
            scheduler.step(val_loss)
            early_stopper(val_loss)
            if early_stopper.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {avg_train_loss:.6f}, "
              f"{'Val Loss: {:.6f}, '.format(val_loss) if val_loss else ''}LR: {optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'history': history
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, os.path.join(save_dir, 'final_model.pth'))

    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return history


def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history['train_losses'], label='Training Loss', linewidth=2)
    if history['val_losses']:
        axes[0].plot(history['val_losses'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def load_model(model_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    model = TextureAutoencoder()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    return model, checkpoint


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--data_dir', type=str, default='texture_patches')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--num_patches', type=int, default=400)
    parser.add_argument('--enable_rotation', action='store_true', default=True)
    parser.add_argument('--force_reprocess', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.join(args.save_dir)

    if not os.path.exists(args.image_dir):
        raise ValueError(f"Image directory not found: {args.image_dir}")

    if os.path.exists(args.data_dir) and not args.force_reprocess:
        train_paths, test_paths = load_texture_tiles(args.data_dir)
    else:
        image_paths = load_images_from_directory(args.image_dir)
        if not image_paths:
            raise ValueError(f"No valid images found in {args.image_dir}")
        preprocess_images_from_directory(
            args.image_dir,
            output_dir=args.data_dir,
            num_patches=args.num_patches,
            enable_rotation=args.enable_rotation
        )
        train_paths, test_paths = load_texture_tiles(args.data_dir)

    train_dataset = TextureDataset(train_paths)
    val_dataset = TextureDataset(test_paths) if test_paths else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4) if val_dataset else None

    model = TextureAutoencoder()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history = train_autoencoder(model, train_loader, val_loader, num_epochs=args.epochs,
                                learning_rate=args.lr, device=device, save_dir=save_dir)

    plot_training_history(history, os.path.join(save_dir, 'training_history.jpg'))
    print(f"Training complete. Results saved to {save_dir}")


if __name__ == "__main__":
    main()
