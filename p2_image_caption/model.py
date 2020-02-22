import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.init_weights()

    def init_weights(self):
        #initialize the weights
        self.embed.weight.data.uniform_(*hidden_init(self.embed))
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.embed(features))

        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
    
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = 1,
                            bias = True,
                            batch_first = True,
                            dropout = 0,
                            bidirectional=False)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embeddings.weight.data.uniform_(*hidden_init(self.embeddings))
        self.fc.weight.data.uniform_(*hidden_init(self.fc))
        self.fc.bias.data.fill_(0)
        
    def forward(self, features, captions):
        #discard the end token
        captions = captions[:,:-1]
        
        embeddings = self.embeddings(captions)
        #stack the feature and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)

        out, _ = self.lstm(embeddings)
        
        return self.fc(out)
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        
        for i in range(max_len):
            out, _ = self.lstm(inputs)
            out = self.fc(out)
            
            out = out.squeeze(1)
            wordid = out.argmax(dim=1)
            caption.add(wordid.item())
            
            inputs = self.embeddings(wordid.unsqueeze(0))
            
        return caption