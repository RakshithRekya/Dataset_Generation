import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_csv('image_dataset.csv')

# Split the dataset into training and testing datasets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)


class Custom_image_dataset_1(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, allowed_folders=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.allowed_folders = allowed_folders if allowed_folders else []

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        folder_name = self.dataframe.iloc[idx, 0].split('_')[0]  # Get the folder name
        if folder_name not in self.allowed_folders:
            return None  # Skip this item
        img_name = os.path.join(self.root_dir, folder_name, self.dataframe.iloc[idx, 0])
        # print(f"Attempting to open image: {img_name}")  # Debug print
        try:
            image = Image.open(img_name)
        except FileNotFoundError:
            print(f"File not found: {img_name}")  # Debug print for error
            raise
        label = self.dataframe.iloc[idx, 1]
        
        # Mapping string labels to integers
        if label == 'annmary':
            label = 0
        elif label == 'deepthi':
            label = 1
        elif label == 'jithin':
            label = 2
        elif label == 'nurettin':
            label = 3
        elif label == 'rakshith':
            label = 4
        elif label == 'yogitha':
            label = 5
        else:
            print("wrong labels used")
    
        if self.transform:
            image = self.transform(image)
    
        return image, label


class Custom_image_dataset_2(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, allowed_folders=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.allowed_folders = allowed_folders if allowed_folders else []

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        folder_name = self.dataframe.iloc[idx, 0].split('_')[0]  # Get the folder name
        if folder_name not in self.allowed_folders:
            return None  # Skip this item
        img_name = os.path.join(self.root_dir, folder_name, self.dataframe.iloc[idx, 0])
        # print(f"Attempting to open image: {img_name}")  # Debug print
        try:
            image = Image.open(img_name)
        except FileNotFoundError:
            print(f"File not found: {img_name}")  # Debug print for error
            raise

        # Extract attributes as labels
        labels = self.dataframe.iloc[idx, 2:].values.astype('float32')
        labels = torch.tensor(labels)  # Convert to tensor

        if self.transform:
            image = self.transform(image)
    
        return image, labels
    

def loaders_1():
    allowed_folders = ["annmary", "deepthi", "jithin", "nurettin", "rakshith", "yogitha"]
    
    # Define transforms for both train and test data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(95, 95)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create the dataset instances
    train_dataset = Custom_image_dataset_1(dataframe=train_data, root_dir='.', transform=train_transform, allowed_folders=allowed_folders)
    test_dataset = Custom_image_dataset_1(dataframe=test_data, root_dir='.', transform=test_transform, allowed_folders=allowed_folders)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    class_names = {0: "annmary", 1: "deepthi", 2: "jithin", 3: "nurettin", 
                  4: "rakshith", 5: "yogitha"}
    
    for loader in [train_loader, test_loader]:
        loader_name = "train Loader" if loader == train_loader else "test Loader"
        print(f"Length of {loader_name}: {len(loader)}")
        
        batch = next(iter(loader))
        inputs, labels = batch
        print(f"Shape of inputs of each batch: {inputs.shape}")
        print(f"Length of labels of each batch: {len(labels)}")
        
        min_value = inputs.min()
        max_value = inputs.max()
        print(f"Min value: {min_value.item()}")
        print(f"Max value: {max_value.item()}")
        
        # Set up the subplot dimensions
        fig, axs = plt.subplots(2, 3, figsize=(8, 6))
        axs = axs.ravel()
            
        for idx in range(6):
            image = inputs[idx]
            label = labels[idx]
                
            image_np = image.permute(1, 2, 0).numpy()
            axs[idx].imshow(image_np)
            axs[idx].axis('off')
            axs[idx].set_title(f"{class_names[label.item()]}")
        
        plt.suptitle(f"{loader_name}")
        plt.tight_layout()
        plt.show()
        
    return train_loader, test_loader

def loaders_2():
    allowed_folders = ["annmary", "deepthi", "jithin", "nurettin", "rakshith", "yogitha"]
    
    # Define transforms for both train and test data
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(95, 95)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create the dataset instances
    train_dataset = Custom_image_dataset_2(dataframe=train_data, root_dir='.', transform=train_transform, allowed_folders=allowed_folders)
    test_dataset = Custom_image_dataset_2(dataframe=test_data, root_dir='.', transform=test_transform, allowed_folders=allowed_folders)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    attribute_names = ["male", "black_hair", "mustache", "glasses", "beard"]
    
    for loader in [train_loader, test_loader]:
        loader_name = "train Loader" if loader == train_loader else "test Loader"
        print(f"Length of {loader_name}: {len(loader)}")
        
        batch = next(iter(loader))
        inputs, labels = batch
        print(f"Shape of inputs of each batch: {inputs.shape}")
        print(f"Length of labels of each batch: {len(labels)}")
        
        min_value = inputs.min()
        max_value = inputs.max()
        print(f"Min value: {min_value.item()}")
        print(f"Max value: {max_value.item()}")
        
        # Set up the subplot dimensions
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.ravel()
            
        for idx in range(6):
            image = inputs[idx]
            label = labels[idx]
                
            image_np = image.permute(1, 2, 0).numpy()
            axs[idx].imshow(image_np)
            axs[idx].axis('off')
            axs[idx].set_title(f"{label}")
        
        plt.suptitle(f"{attribute_names}")
        plt.tight_layout()
        plt.show()
        
    return train_loader, test_loader
