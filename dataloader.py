import torch
from torch.utils.data import Dataset, DataLoader 

class TextDataset(Dataset):
    def __init__(self, data, labels):
       """
       Args:
            data (np.array): 4D NumPy array of shape (num_samples, max_sentences, max_seq_length, embedding_dim).
            labels (np.array): 1D array of labels.
       """
       self.data = data
       self.labels = labels
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).float(), torch.tensor(self.labels[idx]).long()
       
def get_dataloader(data, labels, batch_size=32, shuffle=True):
    """
    Args:
        data (np.array): 4D NumPy array of shape (num_samples, max_sentences, max_seq_length, embedding_dim).
        labels (np.array): 1D array of labels.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
    Returns:
        DataLoader: PyTorch DataLoader.
    """
    dataset = TextDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

