import argparse
import os
import sys
from model import TwoLayerRNNClassifier
from preprocessing import load_data
from dataloader import get_dataloader
from train import train
import torch
# add parser arguments like input_dim, hidden_dim1, hidden_dim2, output_dim, num_epochs, batch_size, learning_rate
parser = argparse.ArgumentParser(description='Train a simple RNN model on text data')

parser.add_argument('--input_dim', type=int, default=300, help='The size of the input vectors')
parser.add_argument('--hidden_dim1', type=int, default=128, help='The size of the first hidden state')
parser.add_argument('--hidden_dim2', type=int, default=64, help='The size of the second hidden state')
parser.add_argument('--output_dim', type=int, default=5, help='The size of the output vectors')
parser.add_argument('--num_epochs', type=int, default=1000, help='The number of epochs')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0015, help='Learning rate')
parser.add_argument('--data_path', type=str, default='data.csv', help='Path to the data file')
args = parser.parse_args()

train_max_seq_length = 20
train_max_sentences = 4
valid_max_seq_length = 20
valid_max_sentences = 4

data,labels = load_data('data.csv',max_seq_length=train_max_seq_length,maxsentences=train_max_sentences,label_shifting=1)
train_loader = get_dataloader(data,labels,args.batch_size,shuffle=True)

data,labels = load_data('electronics_validation.csv',max_seq_length=valid_max_seq_length,maxsentences=valid_max_sentences,label_shifting=1)
valid_loader = get_dataloader(data,labels,args.batch_size,shuffle=False)

model = TwoLayerRNNClassifier(args.input_dim,args.hidden_dim1,args.hidden_dim2,args.output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0025)
criterion = torch.nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Training started")
train(model,train_loader,valid_loader,args.num_epochs,optimizer,criterion,device)
