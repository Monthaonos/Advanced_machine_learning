from model_arch import CIFAR10
from data_load import load_data_CIFAR10
from train_model import model_train


def main():

    model = CIFAR10()

    train_loader, test_loader = load_data_CIFAR10(batch_size=128)

    model_train(model, 8, 0.01, train_loader, test_loader, False)

if __name__ == "__main__":
    main()