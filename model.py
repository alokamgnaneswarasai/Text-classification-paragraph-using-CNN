import torch
import torch.nn as nn

class TwoLayerRNNClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim1,hidden_dim2,output_dim):
        """
        Args:
            input_dim (int): The size of the input vectors.
            hidden_dim (int): The size of the hidden state.
            output_dim (int): The size of the output vectors.
        """
        super(TwoLayerRNNClassifier, self).__init__()
        self.hidden_dim1 = hidden_dim1  
        self.hidden_dim2 = hidden_dim2
        self.rnn1 = nn.GRU(input_dim, hidden_dim1, batch_first=True) # First RNN to process each sentence
        self.rnn2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True) # second RNN to process the seequence of sentence embeddings
        self.fc = nn.Linear(hidden_dim2, output_dim) 
        
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_sentences, max_seq_length, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        batch_size, max_sentences, max_seq_length, _ = x.size() 
        x = x.view(batch_size*max_sentences, max_seq_length, -1)  
        h1 = torch.zeros(1, batch_size*max_sentences, self.hidden_dim1).to(x.device) 
        out1, h1 = self.rnn1(x, h1)
        out1 = out1[:,-1,:] 
        out1 = out1.view(batch_size, max_sentences, -1) 
        h2 = torch.zeros(1, batch_size, self.hidden_dim2).to(x.device) 
        out2, h2 = self.rnn2(out1, h2)
        out2 = out2[:,-1,:]
        out = self.fc(out2)
        return out
        
        
        
        
    
# # just to test the model
# if __name__ == "__main__":
#     model = TwoLayerRNNClassifier(300, 128, 64, 5)
#     x = torch.rand(32, 6, 10, 300)
#     output = model(x)
#     print(output.size())