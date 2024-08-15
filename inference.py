import torch.nn as nn
import torch
import os
from torchvision import transforms
import torchvision.models as models
from PIL import Image

img_path = 'tiny-imagenet-200/test/images/test_0.JPEG'

class_names = {i:x.strip('\r\n') for i,x in enumerate(open('tiny-imagenet-200/wnids.txt','r').readlines())}
class_description = {x.strip('\r\n').split('\t')[0]:x.strip('\r\n').split('\t')[1] 
                        for x in open('tiny-imagenet-200/words.txt','r').readlines()}

test_transform = transforms.Compose([
transforms.Resize(64),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 200)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)
model.load_state_dict(torch.load('models/epoch_33_valAcc_56.2.pth'))

im = Image.open(img_path).convert('RGB')
im = test_transform(im)
im = im.to(device)

model.eval()
with torch.no_grad():
    im = im.unsqueeze(0)
    outputs = model(im)
    _, preds = torch.max(outputs, 1)
    print(f'Predicted class label {preds.item()} \t class name : {class_names[preds.item()]} \t \
        class description: {class_description[class_names[preds.item()]]}')