import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

parser = argparse.ArgumentParser()
parser.add_argument()
parser.add_argument(
    "-d", "--dir", type=str, default="/home/workspace/aipnd-project/",
)

parser.add_argument(
    "-a", "--arch", type=str, default="vgg16"
)

parser.add_argument(
    "-s", "--save_dir", type=str, default="vgg16_checkpoint.pth"
)

parser.add_argument(
    "--gpu", type=str, default="cuda"
)

parser.add_argument(
    "--learning_rate", type=float, default=0.001
)

args = parser.parse_args()
criterion = nn.NLLLoss()


def train(gpu, model, data_loader: DataLoader, learning_rate: float):
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    if gpu == "gpu":
        model.to("cuda")
    else:
        model.to("cpu")
    model.train()
    for index, (inputs, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        if gpu == "gpu":
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
        else:
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


def save(save_file_name: str, train_dataset):
    class_to_idx = {val: key for key, val in train_dataset.class_to_idx.items()}
    checkpoint = {
        "input_size": 25088,
        "output_size": 102,
        "class_to_idx": class_to_idx,
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, save_file_name)


if __name__ == '__main__':

    train_dir = args.dir + '/flower_data/train'
    valid_dir = args.dir + '/flower_data/valid'
    test_dir = args.dir + '/flower_data/test'

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

    epoch = 10
    for batch in range(0, epoch):
        print("batch {} ---- start".format(batch))
        train(args.gpu, train_loader, args.learning_rate)
    print(valid(valid_loader))
    if args.save_dir is not None:
        save(args.save_dir, train_dataset)
        print("训练结果保存成功")
