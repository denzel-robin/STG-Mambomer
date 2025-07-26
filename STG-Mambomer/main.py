import pandas as pd
import numpy as np

from prepare import *
from train_STGmamba import *

print("\nLoading PEMS04 data...")
speed_matrix = pd.read_csv('pems04_flow.csv',sep=',')
A = np.load('pems04_adj.npy')
speed_matrix = speed_matrix.iloc[:, :-7]
A = A[:-7, :-7]

print("\nPreparing train/test data...")
train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(speed_matrix, BATCH_SIZE=48)

print("\nTraining STGmambomer model...")
STGmamba, STGmamba_loss = TrainSTG_Mamba(train_dataloader, valid_dataloader, A, K=3, num_epochs=200, mamba_features=300)
print("\nTesting STGmambomer model...")
results = TestSTG_Mamba(STGmamba, test_dataloader, max_value)
