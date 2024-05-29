Code structure from pytorch modules, workspace and session lead.

# Imports here
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import torch
import tochvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import helper

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    gpu = False 
    
    if torch.cuda.is_available():
        gpu = True
        model = model.cuda()
    else:
        #Convert images to numpy array.
        model = model.cpu()
        image = Image.open(image_path)
        np_array = process_image(image)
        tensor = torch.from_numpy(np_array)
        
    if gpu:
        var_inputs = Variable(tensor.float().cuda(), volatile=True)
    else:       
        var_inputs = Variable(tensor, volatile=True)
        var_inputs = var_inputs.unsqueeze(0)
        output = model.forward(var_inputs)  
        ps = torch.exp(output).data.topk(topk)
        probabilities = ps[0].cpu() if gpu else ps[0]
        classes = ps[1].cpu() if gpu else ps[1]
        class_to_idx_inverted = {model.class_to_idx[k]: k for k in model.class_to_idx}
        mapped_classes = list()
        
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])
        
    return probabilities.numpy()[0], mapped_classes


probabilities = predict(image_path, model)
classes = predict(image_path, model)

print('Probabilities:'probabilities)
print('Classes: 'classes)



