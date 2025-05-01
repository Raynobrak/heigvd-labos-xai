import torch
import torchvision.transforms as transforms
import pandas as pd


from PIL import Image

# classes definiton

class CustomImageDataset(torch.utils.data.Dataset):
    """
    this class build a data based on a dir of images and csv containing the labels
    implement getitem and len to then used pytorch data loader
    """
    def __init__(self, csv_path, data_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.data_path = data_path
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):


        image_name = self.data.iloc[index, 0]
        image_path = rf"{self.data_path}/{image_name}.jpg"

        try:
            img = Image.open(image_path)

        except Exception as e:
            print(f"Error reading image at path {image_path}: {e}")
            return None

        if self.transform is not None:
            img = self.transform(img)
        else:
            transform_ = transforms.Compose([transforms.ToTensor()])
            img = transform_(img)


        # extract the one hot encoded label from the data
        label = self.data.iloc[index, 1:].values.astype(int)

        label = torch.from_numpy(label)

        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        #img = img.to(device)
        #label = label.to(device)

        return img, label
