import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

from model import StockPredictor

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StockPredictor(model_info['input_size'], model_info['hidden_size'], model_info['output_size'], model_info['num_layers'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    model.to(device).eval()

    print("Done loading model.")
    return model

def _get_train_data_loader(batch_size, output_size, training_dir):
    """Get the train data loader
       Parameters:
       ----------
       batch_size: batch size
       output_size: output size. e.g., output_size=3 indicating first 3 columns in train_data are labels
       training_dir: training dir
    """
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data.iloc[:, :output_size].values).float()
    train_X = torch.from_numpy(train_data.iloc[:, output_size:].values).unsqueeze(2).float()
    
    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)

def _get_valid_data_loader(batch_size, output_size, training_dir):
    """Get the validation data loader
       Parameters:
       ----------
       batch_size: batch size
       output_size: output size. e.g., output_size=3 indicating first 3 columns in train_data are labels
       training_dir: training dir
    """
    print("Get validate data loader.")

    valid_data = pd.read_csv(os.path.join(training_dir, "valid.csv"), header=None, names=None)

    valid_y = torch.from_numpy(valid_data.iloc[:, :output_size].values).float()
    valid_X = torch.from_numpy(valid_data.iloc[:, output_size:].values).unsqueeze(2).float()
    
    valid_ds = torch.utils.data.TensorDataset(valid_X, valid_y)

    return torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)


def train(model, train_loader, valid_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    valid_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:    

            batch_X, batch_y = batch

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            model.zero_grad()

            p = model.forward(batch_X)
            loss = loss_fn(p, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.data.item()
        print("Epoch: {}, MSELoss: {}".format(epoch, total_loss / len(train_loader)))
    
        valid(model, valid_loader, loss_fn, device)
        
def valid(model, valid_loader, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    valid_loader - The PyTorch DataLoader that should be used for validation.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    model.eval()
    valid_loss = 0
    for batch in valid_loader:
        batch_X, batch_y = batch

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        p = model.forward(batch_X)
        loss = loss_fn(p, batch_y)

        valid_loss += loss.data.item()

    print('VALIDATION_LOSS={:.4f}'.format(valid_loss / len(valid_loader)))

              
              
              
if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # Model Parameters
    parser.add_argument('--input_size', type=int, default=1, metavar='N',
                        help='size of the data point in sequence (default: 1)')
    parser.add_argument('--hidden_size', type=int, default=32, metavar='N',
                        help='size of the hidden dimension (default: 32)')
    parser.add_argument('--output_size', type=int, default=1, metavar='N',
                        help='size of the output, i.e., number of days of interests for training/prediction (default: 1)')
    parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                        help='number of stacked LSTM layers (default: 2)')
    parser.add_argument('--lstm_dropout', type=float, default=0.2, metavar='N',
                        help='drop out probability for each LSTM layer except the last layer (default: 0.2)')
    parser.add_argument('--fc_dropout', type=float, default=0, metavar='N',
                        help='drop out probability the fully connect layer for output (default: 0)')
    
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training and validation data.
    train_loader = _get_train_data_loader(args.batch_size, args.output_size, args.data_dir)
    valid_loader = _get_valid_data_loader(args.batch_size, args.output_size, args.data_dir)
                    
    # Build the model.
    model = StockPredictor(args.input_size, args.hidden_size, args.output_size, args.num_layers, args.lstm_dropout, args.fc_dropout).to(device)

    print("Model loaded with input_size {}, hidden_size {}, output_size {}, num_layers {}, lstm_dropout {} fc_dropout {} learning rate {}.".format(
        args.input_size, args.hidden_size, args.output_size, args.num_layers, args.lstm_dropout, args.fc_dropout, args.lr
    ))

    # Train the model.
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    loss_fn = torch.nn.MSELoss()

    train(model, train_loader, valid_loader, args.epochs, optimizer, loss_fn, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_size': args.input_size,
            'hidden_size': args.hidden_size,
            'output_size': args.output_size,
            'num_layers': args.num_layers,
        }
        torch.save(model_info, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
