import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def load_cifar_10(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_size = int(0.9 * len(full_trainset))  
    val_size = len(full_trainset) - train_size 
    train_subset, val_subset = random_split(full_trainset, [train_size, val_size])

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, valloader, testloader