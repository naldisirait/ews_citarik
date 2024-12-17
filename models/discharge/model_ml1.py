import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRegression(nn.Module):
    def __init__(self, input_size=72, output_size=72, hidden_size=128, num_hidden_layers=4, dropout_rate=0.2):
        """
        Initializes an advanced MLP regression model.

        Parameters:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_size (int): Number of neurons in each hidden layer.
            num_hidden_layers (int): Number of hidden layers.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(MLPRegression, self).__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        ])
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_hidden_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass of the MLP model.
        """
        # Input layer with activation
        x = F.relu(self.input_layer(x))
        
        # Hidden layers with dropout, layer normalization, and activation
        for layer, norm in zip(self.hidden_layers, self.layer_norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Output layer
        x = self.output_layer(x)
        return x

# Seq2Seq Model Components: Encoder, Decoder, Seq2Seq
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        outputs, (hidden, cell) = self.lstm(x, (h0, c0))
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers, dropout=0.3):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout=0.3):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(output_dim, hidden_dim, num_layers, dropout)

    def forward(self, src, target_len):
        batch_size = src.size(0)
        input_dim = src.size(2)
        hidden, cell = self.encoder(src)
        decoder_input = torch.zeros(batch_size, 1, input_dim).to(src.device)
        predictions = torch.zeros(batch_size, target_len, input_dim).to(src.device)
        for t in range(target_len):
            prediction, hidden, cell = self.decoder(decoder_input, hidden, cell)
            predictions[:, t:t+1, :] = prediction
            decoder_input = prediction
        return predictions

def load_model_f16(device,path_trained_f16):
    #set constants
    step = 1
    lag = 16 
    forecast_size = 8

    #defined model
    input_size = 72
    output_size = 72
    hidden_size = 128
    num_hidden_layers = 4
    dropout_rate = 0.2

    # Initialize the model
    model_f16 = MLPRegression(
        input_size=input_size, 
        output_size=output_size, 
        hidden_size=hidden_size, 
        num_hidden_layers=num_hidden_layers, 
        dropout_rate=dropout_rate)
    
    # Load the state_dict (weights) into the model
    model_f16.load_state_dict(torch.load(path_trained_f16,map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode if you are using it for inference
    model_f16.to(device)
    model_f16.eval()
    
    return model_f16

def load_model_f8(device,path_trained_f8):
    # Hyperparameters
    input_dim = 1        # Single feature for univariate time series
    output_dim = 1       # Predict one value per time step
    hidden_dim = 64      # Hidden state size in LSTM
    num_layers = 2      # Number of LSTM layers
    dropout = 0.5      # Dropout rate
    input_seq_len = 72 # Length of input sequence (lookback)
    target_seq_len = 8  # Length of target sequence (prediction steps)
    
    # Model instantiation
    model_f8 = Seq2Seq(input_dim, output_dim, hidden_dim, num_layers, dropout).to(device)
    
    #load model
    model_f8.load_state_dict(torch.load(path_trained_f8, map_location=torch.device('cpu')))
    
    # Set the model to evaluation mode if you are using it for inference
    model_f8.to(device)
    model_f8.eval()
    
    return model_f8

def inference_ml1(precip, config):
    """
    Function to predict discharge
    Args:
        precip(tensor): grided precipitation with shape (Batch=1, len_history=72)
    Returns:
        discharge(tensor): 72 hours of discharge, where 48 hours is estimated and 24 hours forcast discharge.
    """
    #set constants
    device = "cpu"
    precip = precip.float() #make sure the value type is float
    target_seq_len = 8
    length_discharge_to_extract = 72
    
    #load model
    path_trained_f16 = config['model']['path_trained_ml1_f16']
    path_trained_f8 = config['model']['path_trained_ml1_f8']
    model_f16 = load_model_f16(device,path_trained_f16=path_trained_f16)
    model_f8 = load_model_f8(device,path_trained_f8=path_trained_f8)

    #inference model with given input precipitation
    with torch.no_grad():
        output_f16 = model_f16(precip)
        B,T = output_f16.shape
        output_f16 = output_f16.view(B,T,1) # Add a new dimension, making the tensor of shape (B, T, feature)
        output_f8 = model_f8(output_f16, target_seq_len)
    discharge = torch.cat((output_f16,output_f8),axis = 1).view(-1)[-length_discharge_to_extract:]
    return discharge