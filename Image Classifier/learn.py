import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

train_dir = './flower_data/train'
valid_dir = './flower_data/valid'
test_dir = './flower_data/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=valid_transforms)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(OrderedDict([
    ("fc1", nn.Linear(25088, 4069)),
    ("relu1", nn.ReLU()),
    ("dropout", nn.Dropout(p=0.5)),
    ("fc2", nn.Linear(4069, 102)),
    ("output", nn.LogSoftmax(dim=1))
]))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


def train(data_loader: DataLoader):
    model.to("cpu")
    model.train()
    for index, (inputs, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        inputs, labels = inputs.to("cpu"), labels.to("cpu")
        ouputs = model.forward(inputs)
        loss = criterion(ouputs, labels)
        loss.backward()
        optimizer.step()


def valid(data_loader: DataLoader):
    model.to("cpu")
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in iter(data_loader):
            inputs = Variable(inputs.float().cuda(), volatile=True)
            labels = Variable(labels, volatile=True)
            ouputs = model.forward(inputs)
            test_loss += criterion(ouputs, labels)
            ps = torch.exp(ouputs).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()
        return test_loss / len(data_loader), accuracy / len(data_loader) * 100


epoch = 5
for batch in range(0, epoch):
    train(train_loader)


def load_checkpoint(filepath: str):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(25088, 4069)),
        ("relu1", nn.ReLU()),
        ("dropout", nn.Dropout(p=0.5)),
        ("fc2", nn.Linear(4069, 102)),
        ("output", nn.LogSoftmax(dim=1))
    ]))

    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint["class_to_idx"]


def process_image(image):
    image = Image.open(image)

    width, height = image.size
    if width > height:
        width = int(256 / height * width)
        image = image.resize((width, 256), Image.ANTIALIAS)
    else:
        height = int(256 / width * height)
        image = image.resize((256, height), Image.ANTIALIAS)

    width, height = image.size
    left = (width - 224) / 2
    right = (width + 224) / 2
    up = (height - 224) / 2
    bottom = (height + 224) / 2
    image = image.crop(box=(left, up, right, bottom))
    # 图像的颜色通道通常编码为整数 0-255，但是该模型要求值为浮点数 0-1
    data = np.array(image) / 255
    # 均值应标准化为 [0.485, 0.456, 0.406]
    mean = np.array([0.485, 0.456, 0.406])
    # 标准差应标准化为[0.229, 0.224, 0.225]
    std = np.array([0.229, 0.224, 0.225])
    # 每个颜色通道减去均值，然后除以标准差
    data = (data - mean) / std
    return np.transpose(data, (2, 0, 1))


def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    image = image.transpose((1, 2, 0))

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax


class_to_idx = {val: key for key, val in train_dataset.class_to_idx.items()}


def predict(image_path, model, topk=5):
    model.to("cuda")
    model.eval()
    img_tensor = torch.from_numpy(process_image(image_path))
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.type(torch.cuda.FloatTensor)
    output = model(Variable(img_tensor.cuda(), volatile=True))
    ps = torch.exp(output)
    probs, index = ps.topk(topk)
    probs = probs.cpu().detach().numpy().tolist()[0]
    index = index.cpu().detach().numpy().tolist()[0]

    index = [class_to_idx[i] for i in index]
    return probs, index


flower_name = cat_to_name["1"]
_, ax = plt.subplots(nrows=2, figsize=(8, 16))
img_tensor = process_image(train_dir + "/1/image_06734.jpg")
imshow(img_tensor, ax[0], flower_name)
ax[0].set_title(flower_name)
# get the prediction probs
probs, index = predict(train_dir + "/1/image_06734.jpg", model)
index = [cat_to_name[i] for i in index]

prediction = pd.Series(data=probs, index=index)
prediction.plot(kind="barh", ax=ax[1])
plt.show()
