#Code structure from pytorch modules and workspace.

# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import torch
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import helper
from torch import nn, optim
import torch.nn.functional as F
import argparse

#Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset ')
parser.add_argument('--epochs', type=int, help='Number of epochs')
parser.add_argument('--arch', type=str, help='Model architecture')
parser.add_argument('--learning_rate', type=float, help='Learning rate')
parser.add_argument('--hidden_units', type=int, help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')

#Train and build model.
class Classifier(nn.Module):
    def _init_(self):
        super()._init_()
        self.fc1 = nn.Linear(1054, 528)
        self.fc2 = nn.Linear(528, 264)
        self.fc3 = nn.Linear(264, 132)
        self.fc5 = nn.Linear(132, 66)
        self.fc6 = nn.Linear(66, 10)
            
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x = F.log_softmax(self.fc5(x), dim=1)
        
        return x
    
args= parser.parse_known_args()

def loadModel(model, args.epchos,lr, args.hidden_layers):
    criterion = nn.NLLLoss()
    Optimizer = optim.Adam(model.parameters(), lr = 0.003)


    for e in range(epchos):
        running_loss = 0
        for images, labels in train_loader:
            logps = model(images)
            loss = criterion(logps, labels)
        
            optimizer.zero_grad()
            loss.backward()
            optmizer.step()
        
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss}")
        
    #Save model to checkpoint.
    checkpoint = {'input_size': 1054,
                  'output_size': 10,
                  'hidden_layers': [each.out_features for each in model.hidden_layers],
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint.pth')

#Training and validation loss.
model = Classifier()
criterion = nn.NLLLoss()
Optimizer = optim.Adam(model.parameters(), lr = 0.003)

epchos = 5
step = 0

train_losses, test_losses = [], []

for e in range(epchos):
    running_loss = 0
    for images, labels in train_loader:
        logps = model(images)
        loss = criterion(logps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optmizer.step()
        
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation.
        with torch.no_grad():
            for images, labels in test_loader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/len(train_loader))
        test_losses.append(test_loss/len(test_loader))
        
        print("Epoch: {}/{}.. ".format(e+1, epochs),
             "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
             "Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
             "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))
