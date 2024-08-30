import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TwoLayerRNNClassifier
from dataloader import get_dataloader
from preprocessing import load_data

def train(model,train_loader,valid_loader,epochs,optimizer,criterion,device):
    
    model.train()
    model.to(device)
    valid_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print('*'*30, f'Epoch {epoch+1}/{epochs}', '*'*30)
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')
        
        train_loss, train_acc = eval(model,train_loader,criterion,device)
        print(f'Training Loss: {train_loss}, Training Accuracy: {train_acc}')
        valid_loss, valid_acc = eval(model,valid_loader,criterion,device)
        print(f'Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}')
        valid_losses.append(valid_loss)
        
        # do early stopping if validation loss does not decrease for 10 epochs
        if epoch>10 and valid_loss>min(valid_losses[-10:]):
            print('Early stopping triggered')
            break
        
    print('Training completed')
    
    torch.save(model.state_dict(), 'models/model.pth')
        
        
def eval(model,dataloader,criterion,device):
        
    # model.to(device)
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            loss = criterion(output, labels)
            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    return total_loss/len(dataloader), total_correct/total