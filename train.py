import torch
import my_model
import followers.dataset
import argparse
from torch import nn 
from torch import optim as optim


parser = argparse.ArgumentParser(description="Train a Neural Network (NN) using transfer learning")
# 1. The directory to the image files
parser.add_argument('data_directory', default='./flowers',
                    help="The relative path to the image files to train on. It include three folders: 'train', 'test' and 'valid' for training.")
# 2. The path where shoud save the model or the checkpoint
parser.add_argument('--save_dir', default='./',
                    help="The relative path to save the neural network checkpoint")             
# 3. Choose the architecture
parser.add_argument('--arch', default="vgg16",
                    help="The model architecture supported here are  vgg16 or resnet18")
# 4. Set the hyperparameters: Learning Rate, Hidden Units, Epochs.
parser.add_argument('--lr', type=float, default="0.0005",
                    help="The learning rate for the model. Should be very small")
parser.add_argument('--num_hidden_units', type=int, default=256 ,
                    help="The number of units in the hidden layer")
parser.add_argument('--epochs', type=int, default=2,
                    help="The number of epochs you want to use")

# 5. Choose the GPU for training
parser.add_argument('--gpu', default=False, action='store_true',
                    help="If you would like to use the GPU for training. Default is False. if you want to use the gpu enter True or gpu or cuda")

args = parser.parse_args()
data_directory = args.data_directory
save_directory = args.save_dir
arch = args.arch
lr = args.lr
num_hidden_units = args.num_hidden_units
epochs = args.epochs
gpu = args.gpu


dataloaders, datatesting, datavalidation, image_datasets = followers.dataset.load_image(data_directory)

model = my_model.create_network(arch, num_hidden_units)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr )

my_model.training_val(model, criterion, optimizer, dataloaders, datavalidation, epochs, use_gpu )
my_model.training_tes(model, criterion, optimizer, dataloaders, datatesting, epochs, use_gpu)

# Save the model
my_model.save_model(model, image_datasets, epochs, optimizer, arch)
