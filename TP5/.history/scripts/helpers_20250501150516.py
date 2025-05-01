import os
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

from torch.utils.data import random_split
from scripts.custom_dataset import CustomImageDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_datasets(train = True):

    home_path = os.getcwd()
    data_path = os.path.join(home_path, "data", "skin-lesion-fake")
    label_path = os.path.join(data_path, "label")
    if train:
        train_dir_path = os.path.join(data_path, "train")
        label_train_path = os.path.join(label_path, "label_train.csv")
        dataset = CustomImageDataset(csv_path=label_train_path, data_path=train_dir_path, transform=transforms_train())
        train_size = int(0.8 * len(dataset))  # use 90% of the data for training
        val_size = len(dataset) - train_size  # use the remaining 10% for validation
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        return train_dataset, val_dataset
    else:
        test_dir_path = os.path.join(data_path, "test")
        label_test_path = os.path.join(label_path, "label_test.csv")
        dataset = CustomImageDataset(csv_path=label_test_path, data_path=test_dir_path, transform=transforms_train())
        return dataset


def get_data_loaders(train_dataset, val_dataset, batch_size = 32):

    """
    cool
    """
    # Create the PyTorch data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def save_plots(train_acc, valid_acc, train_loss, valid_loss, name):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"../models/plots/{name}_accuracy_plot.png")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"../models/plots/{name}_loss_plot.png")


# Define the transform for the training data
def transforms_train():

    trans = transforms.Compose([
    transforms.CenterCrop(400),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return trans



def transforms_eval():

    trans = transforms.Compose([
    transforms.Resize((300,300)),
    #transforms.CenterCrop(300),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return trans


def tensor_to_img(img):
    
    """
    This function take an image stored as a tensor (torch) as in CustomImageDataset
    Return the image in the good shape to be plotted (numpy)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    #std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    img = img.to(device)
    img = img.squeeze(0).permute(1, 2, 0)
    #img = img.mul(std) + mean
    img = img.cpu().numpy()
    
    return img

def to_plot(index, dataset):
    
    """
    Call to tensor_to_img on the dataset and index
    """
    
    img, _ = dataset[index]
    img = tensor_to_img(img)
    
    return img


def get_matching_indexes(dataset, model, true_label, predicted_label):
    
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    true_label_index = class_names.index(true_label)
    predicted_label_index = class_names.index(predicted_label)
    
    matching_indexes = []
    for i in range(len(dataset)):
        inputs, true_label_value = dataset[i]
        predicted_label_value = model(inputs.unsqueeze(0)).argmax(dim=1).item()
        true_label_value = true_label_value.argmax().item()

        if true_label_value == true_label_index and predicted_label_value == predicted_label_index:
            matching_indexes.append(i)
            
    return matching_indexes

def label_to_names(label):
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    class_idx = label.argmax().item()
    return class_names[class_idx]

def idx_to_predicted_label(idx, dataset, model):
    
    img, label = dataset[idx]
    model.eval()
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)    
    predicted_label = torch.zeros_like(probs[0])
    predicted_label[probs.argmax()] = 1
    
    return label_to_names(predicted_label)

def idx_to_true_label(idx, dataset):
    _, label = dataset[idx]
    return label_to_names(label)

def idx_to_proba(idx, dataset, model):
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    img, label = dataset[idx]
    
    model.eval()
    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        probs = F.softmax(logits, dim=1)
                
    # Print the outputs logits and probas
    probs, idxs = probs.sort(descending=True)
    for i in range(8):
        class_idx = idxs[0][i]
        class_prob = probs[0][i]
        class_name = class_names[class_idx]
        print(f"{i+1}. {class_name}: {class_prob:.4f} (logit: {logits[0][class_idx]:.4f})")



    return probs, logits



def inference(model, index, dataset):
    
    
    class_names = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img, label = dataset[index]
    img, label = img.to(device), label.to(device)
    
    true_label = class_names[torch.argmax(label)]
    
    model.eval()
    with torch.no_grad():
        logits = model(img.unsqueeze(0))
        probs = F.softmax(logits, dim=1)
    
    predicted_label = class_names[torch.argmax(probs)]
    
    true_prob = probs[0][torch.argmax(label)]
    predicted_prob, _ = torch.max(probs, dim=1)
    
    return true_label, predicted_label, true_prob.item(), predicted_prob.item()