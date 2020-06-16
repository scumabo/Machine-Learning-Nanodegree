import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
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

# Provided input data loading
def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

# Provided output data handling
def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == NP_CONTENT_TYPE:
        stream = BytesIO()
        np.save(stream, prediction_output)
        return stream.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    """ Predict stock prices
        Parameters:
        ----------
        input_data(num of samples, lookback): input data
        model: trained model
        
        Returns:
        --------
        output (num of samples, output_size): predictions
    """
    
    print('Predicting stock prices')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data = torch.from_numpy(input_data).float()
    data = data.to(device)

    # Make sure to put the model into evaluation mode
    model.eval()

    output = model.forward(data)
    
    out_np = output.cpu().detach().numpy()
    
    return out_np