import argparse
import json
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.autograd import Variable
from torchvision import models

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

criterion = nn.NLLLoss()

parser = argparse.ArgumentParser()

parser.add_argument(
    "-p", "--path", type=str, default="/home/workspace/aipnd-project/flowers/train/1/image_06734.jpg",
)

args = parser.parse_args()


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


if __name__ == '__main__':
    model, class_to_idx = load_checkpoint("vgg16_checkpoint.pth")
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    print(predict(args.p, model))
