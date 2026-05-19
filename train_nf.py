import architectures.cnn as cnn
from architectures.utils import load_cifar_10

if __name__ == "__main__":
    trainloader, valloader, testloader = load_cifar_10()
    model = cnn.ConvNet()
    model.fit(train_loader=trainloader, val_loader=valloader, val_every=5, epochs=10, )
    