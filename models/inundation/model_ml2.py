import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np

class CNNModelBN(nn.Module):
    def __init__(self, steps, features, outputs):
        super(CNNModelBN, self).__init__()
        
        # Adjust in_channels to 72 based on the provided weight shape
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.flatten = nn.Flatten()
        
        # Adjust the input dimension of fc1 to 256 based on the provided weight shape
        self.fc1 = nn.Linear(256 * steps, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, 64)
        self.bn5 = nn.BatchNorm1d(64)
        
        # Adjust fc4 to match the output dimensions of your provided weight shape
        self.fc4 = nn.Linear(64, outputs)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.fc4(x)
        return x
    
def create_model_ml2(input_size, output_size):
    steps = 1
    model = CNNModelBN(steps=steps, features=input_size, outputs=output_size)
    return model

def load_model_ml2(width, height, device):
    output_size = width * height
    features = 24
    steps = 1
    model = CNNModelBN(steps=steps, features=features, outputs=output_size)
    model.to(device)
    path_trained_model_ml2 = "./models/inundation/CNN Kabir 100 epoch 44 kasus Augmented.pth"
    model.load_state_dict(torch.load(path_trained_model_ml2, map_location=torch.device('cpu')))
    model.eval()
    print("Successfully loaded ml2")
    return model

def wse_to_depth(wse):
    """
    function to convert predicted wse into depth
    Args:
        wse(tensor): predicted wse with shape (width, height)
    Retrurns:
        depth(np.ndarray): predicted depth with shape(width, height)
    """
    # Open data dtm
    wse = wse.tolist()
    wse = np.array(wse)
    path_dtm = './data/data dtm.pkl'
    with open(path_dtm, 'rb') as file:
        loaded_data = pickle.load(file)
        arr_dtm = loaded_data['arr_dtm']

    arr_dtm = np.array(arr_dtm)
    # print(f"wse type {type(wse)}, arr_dtm type {type(arr_dtm)}")
    # print(f"wse dtype: {wse.dtype}, arr_dtm dtype: {arr_dtm.dtype}")
    # print(wse.shape, arr_dtm.shape)
    depth = wse - arr_dtm
    depth[depth<0] = 0
    depth[depth>3] = 3
    return depth

def get_non_flood_depth():
    path_non_flood = './data/depth_non_flood.pkl'
    with open(path_non_flood, 'rb') as file:
        loaded_data = pickle.load(file)
        non_flood = loaded_data['data_non_flood']
    non_flood = np.array(non_flood)
    non_flood[non_flood<0] = 0
    non_flood[non_flood>3] = 3
    return non_flood

def inference_ml2(input_debit):
    """
    Function to inference inundation 
    Args:
        input_debit(tensor): 24 hours predictiond debit from ml1, the shape (batch=1,hist=24,features=1)
    Returns:
        pred_inundation(tensor): estimated max depth given input debit
    """
    max_debit = max(input_debit.reshape(-1).tolist())
    print(f"Max debit {max_debit}")
    if max_debit <= 200:
        print("Max debit is less than 200, getting depth from non flood event")
        return get_non_flood_depth()
    device = "cpu"
    width, height = 1621, 1680
    model = load_model_ml2(width=width, height=height, device=device)
    with torch.no_grad():
        output = model(input_debit)
        output = output.view(-1)
    wse = output.view(width,height)
    pred_inundation = wse_to_depth(wse)
    print("Successfully inference ml2")
    return pred_inundation