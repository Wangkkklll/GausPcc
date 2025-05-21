import os
import random
import argparse
import datetime
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
from glob import glob

import torch
import torch.utils.data
from torch import nn
from torch.cuda import amp

from torchsparse.nn import functional as F
from torchsparse.utils.collate import sparse_collate_fn

from dataset import PCDataset,PCDataset_Patch
# from network_8stage import Network

seed = 11
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device='cuda:0'

# set torchsparse config
conv_config = F.conv_config.get_default_conv_config()
conv_config.kmap_mode = "hashmap"
F.conv_config.set_global_conv_config(conv_config)

parser = argparse.ArgumentParser(
    prog='train.py',
    description='Training from scratch.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument('--training_data', default='./Dataset/KITTI_detection/training/velodyne/*.ply', help='Training data (Glob pattern).')
parser.add_argument('--val_data', default='', help='Validation data folder (Glob pattern, e.g. /path/to/val/*.ply).')
parser.add_argument('--model_save_folder', default='./model/KITTIDetection', help='Directory where to save trained models.')
parser.add_argument('--log_folder', default='', help='Directory where to save log files. If empty, logs will be saved in model_save_folder.')
parser.add_argument("--is_data_pre_quantized", type=bool, default=False, help="Whether the training data is pre quantized.")
parser.add_argument("--valid_samples", type=str, default='', help="Something like train.txt/val.txt.")

parser.add_argument('--channels', type=int, help='Neural network channels.', default=32)
parser.add_argument('--kernel_size', type=int, help='Convolution kernel size.', default=3)

parser.add_argument('--batch_size', type=int, help='Batch size.', default=1)
parser.add_argument('--learning_rate', type=float, help='Learning rate.', default=0.0005)
parser.add_argument('--lr_decay', type=float, help='Decays the learning rate to x times the original.', default=0.1)
parser.add_argument('--lr_decay_steps', help='Decays the learning rate at x steps.', default=[40000, 90000])
parser.add_argument('--max_steps', type=int, help='Train up to this number of steps.', default=110000)
parser.add_argument('--val_interval', type=int, help='Validate every N steps.', default=500)
parser.add_argument('--log_interval', type=int, help='Log training info every N steps.', default=100)
parser.add_argument('--stage', type=str, help='muti stage for coding', default="2stage")
args = parser.parse_args()

# CREATE MODEL SAVE PATH
os.makedirs(args.model_save_folder, exist_ok=True)

# Set up log directory
if args.log_folder:
    log_folder = args.log_folder
    os.makedirs(log_folder, exist_ok=True)
else:
    log_folder = args.model_save_folder

# Set up log file path
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_folder, f'train_{args.stage}_{timestamp}.log')

# Configure logging
logger = logging.getLogger('training')
logger.setLevel(logging.INFO)

# Clear previous handlers (if any)
if logger.handlers:
    logger.handlers.clear()

# Create handlers
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)  # 10MB size, maximum 5 backups
console_handler = logging.StreamHandler()

# Create formatters
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Record training start information and parameters
logger.info(f"=== Training started {timestamp} ===")
logger.info(f"Parameter configuration: {vars(args)}")
logger.info(f"Log file saved at: {log_file}")

# Load training data
train_files = np.array(glob(args.training_data, recursive=True))

# Check if there are specific training samples
if args.valid_samples != '':
    valid_sample_names = np.loadtxt(args.valid_samples, dtype=str)
    valid_files = []
    for f in train_files:
        fname = f.split('/')[-1].split('.')[0]
        if fname in valid_sample_names:
            valid_files.append(f)
    train_files = valid_files

# Load validation data
val_files = []
if args.val_data:
    val_files = np.array(glob(args.val_data, recursive=True))
    
# Print dataset sizes
logger.info(f"Training set size: {len(train_files)}")
logger.info(f"Validation set size: {len(val_files)}")

# Create data loaders
train_dataflow = torch.utils.data.DataLoader(
    dataset=PCDataset_Patch(train_files, is_pre_quantized=args.is_data_pre_quantized),
    shuffle=True,
    batch_size=args.batch_size,
    collate_fn=sparse_collate_fn
)

# Only create validation dataloader if validation set is provided
if len(val_files) > 0:
    val_dataflow = torch.utils.data.DataLoader(
        dataset=PCDataset(val_files, is_pre_quantized=args.is_data_pre_quantized),
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sparse_collate_fn
    )
else:
    val_dataflow = None
    logger.warning("No validation set provided, validation will not be performed.")

if args.stage == "ue_4stage_conv":
    from network_ue_4stage_conv import Network

net = Network(channels=args.channels, kernel_size=args.kernel_size).to(device).train()
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

# Record model structure
logger.info(f"Model structure: {args.stage}")
total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
logger.info(f"Model parameters: {total_params:,}")

# Create variables to track best model
best_val_loss = float('inf')
best_model_path = os.path.join(args.model_save_folder, f'best_model_{args.stage}.pt')

# Define validation function
def validate():
    if val_dataflow is None:
        return float('inf')
        
    net.eval()
    val_losses = []
    with torch.no_grad():
        for data in val_dataflow:
            x = data['input'].to(device=device)
            loss = net(x)
            val_losses.append(loss.item())
    
    avg_val_loss = np.array(val_losses).mean()
    net.train()
    return avg_val_loss

losses = []
global_step = 0

logger.info("Starting training loop")

try:
    for epoch in range(1, 9999):
        logger.info(f"Epoch {epoch} started at {datetime.datetime.now()}")
        for data in train_dataflow:
            x = data['input'].to(device=device)
            
            optimizer.zero_grad()
            loss = net(x)

            loss.backward()
            optimizer.step()
            global_step += 1

            # Record loss
            losses.append(loss.item())

            # Periodically log training information
            if global_step % args.log_interval == 0:
                avg_loss = np.array(losses).mean()
                logger.info(f'Step: {global_step}, Training loss: {avg_loss:.5f}, Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
                
            if global_step % 500 == 0:
                logger.info(f'Epoch:{epoch} | Step:{global_step} | Train Loss:{round(np.array(losses).mean(), 5)}')
                losses = []

            # Validation evaluation - only if validation set is provided
            if val_dataflow is not None and global_step % args.val_interval == 0:
                val_loss = validate()
                logger.info(f'Validation | Epoch:{epoch} | Step:{global_step} | Val Loss:{val_loss:.5f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(net.state_dict(), best_model_path)
                    logger.info(f'New best model saved, validation loss: {best_val_loss:.5f}')

            # Learning rate decay
            if global_step in args.lr_decay_steps:
                args.learning_rate = args.learning_rate * args.lr_decay
                for g in optimizer.param_groups:
                    g['lr'] = args.learning_rate
                logger.info(f'Learning rate decay triggered, step: {global_step}, new learning rate: {args.learning_rate:.6f}')

            # Save model
            if global_step % 500 == 0:
                model_path = os.path.join(args.model_save_folder, f'ckpt_{args.stage}.pt')
                torch.save(net.state_dict(), model_path)
                logger.info(f'Model saved at: {model_path}')
            
            if global_step >= args.max_steps:
                break

        if global_step >= args.max_steps:
            logger.info(f"Maximum steps {args.max_steps} reached, training completed")
            break

except Exception as e:
    logger.exception(f"Error occurred during training: {str(e)}")
    # Save current model for possible recovery
    torch.save(net.state_dict(), os.path.join(args.model_save_folder, f'error_model_{args.stage}.pt'))

finally:
    # Evaluate and save final model after training ends
    if val_dataflow is not None:
        try:
            final_val_loss = validate()
            logger.info(f'Final validation loss: {final_val_loss:.5f}')
            logger.info(f'Best validation loss: {best_val_loss:.5f}')
        except Exception as e:
            logger.exception(f"Error during final validation: {str(e)}")

    # Save final model
    final_model_path = os.path.join(args.model_save_folder, f'final_model_{args.stage}.pt')
    torch.save(net.state_dict(), final_model_path)
    logger.info(f'Final model saved: {final_model_path}')
    logger.info("=== Training completed ===")