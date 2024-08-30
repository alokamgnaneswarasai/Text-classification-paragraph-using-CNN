import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import TwoLayerRNNClassifier
from dataloader import get_dataloader
from preprocessing import load_data
# use F1 score as evaluation metric
from sklearn.metrics import f1_score

def train(model,train_loader,valid_loader,epochs,optimizer,criterion,device):
    
    model.train()
    model.to(device)
    valid_losses = []
    max_F1 =0 
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
        
        train_loss, train_acc , _ = eval(model,train_loader,criterion,device)
        print(f'Training Loss: {train_loss}, Training Accuracy: {train_acc}')
        valid_loss, valid_acc ,f1 = eval(model,valid_loader,criterion,device)
        print(f'Validation Loss: {valid_loss}, Validation Accuracy: {valid_acc}')
        
        max_F1 = max(max_F1,f1)
        print(f"Max F1 score: {max_F1}")
        
        valid_losses.append(valid_loss)
        
        # do early stopping if validation loss does not decrease for 10 epochs
        if epoch>50 and valid_loss>min(valid_losses[-10:]):
            print('Early stopping triggered')
            break
        
    print('Training completed')
    
    torch.save(model.state_dict(), 'models/model.pth')
        
        
def eval(model,dataloader,criterion,device):
        
    # model.to(device)
    # model.eval()
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
            
    # calculate F1 score
    y_true = labels.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    f1 = f1_score(y_true, y_pred, average='weighted')
   
    return total_loss/len(dataloader), total_correct/total , f1
