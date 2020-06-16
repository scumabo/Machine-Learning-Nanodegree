import torch.nn as nn

class StockPredictor(nn.Module):
    """
    This is the simple RNN model we will be using to perform Stock Prediction.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, lstm_dropout = 0.2, fc_dropout = 0):
        """
        Initialize the model by setting up the various layers.
        
            Parameters:
            -----------
            input_size   : The number of expected features in the input.
                           The input is a sequence.
                           "input_size" specifies the dimension of each data point in the sequence.
            hidden_size  : The number of features in the hidden state.
            output_size  : The dimension of output
            num_layers   : The number of stacked LSTM layers
            lstm_dropout : Drop out for each LSTM layer except the last layer 
                           with probability equal to dropout
            fc_dropout   : Drop out probability for the final fully connected layer      
        """
        super(StockPredictor, self).__init__()
    
        # Input/output tensors shape (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_size  = input_size, 
                            hidden_size = hidden_size, 
                            num_layers  = num_layers,
                            dropout     = lstm_dropout,
                            batch_first = True)
        
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=fc_dropout)
        
    def forward(self, nn_input):
        """
        Perform a forward pass of our model on x.
        
        Parameters:
            nn_input (batch_size, seq_length, input_size): batches of input data.
            
        Returns:
            nn_output (batch_size, output_size): prediction for each batch sequence.
        """
        
        lstm_out, _ = self.lstm(nn_input)
        
        # lstm_out (batch_size, seq_length=lookback, hidden_size): 
        # contains all the hidden states of the last layer.
        # Here we just want the hidden state of the time step
        last_hidden_state = lstm_out[:, -1, :]
        
        # last_hidden_state (batch_size, hidden_size)
        nn_output = self.dropout(self.linear(last_hidden_state))
        
        return nn_output