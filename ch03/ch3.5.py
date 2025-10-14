import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

"""导入全局工具依赖"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
import my_utils 



def main():
    d2l.use_svg_display()
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    print(len(mnist_train))
    print(len(mnist_test))
    print(mnist_train[0][0].shape)

    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    my_utils.show_images(X.reshape(18, 28, 28), 2, 9, titles=my_utils.get_fashion_mnist_labels(y))
    d2l.plt.show()

    """小批量"""
    batch_size = 256

    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=my_utils.get_dataloader_workers())

    timer=my_utils.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')

    train_iter, test_iter = my_utils.load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break

if __name__ == '__main__':
    main()