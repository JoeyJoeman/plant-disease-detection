from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_data = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, train_data.classes
