# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init

# utils
import math
import os
import datetime
import numpy as np
import joblib

from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    n_bands = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)



    kwargs.setdefault("patch_size", 1)
    center_pixel = True
    kwargs.setdefault("epoch", 100)
    # "The RNN was trained with the Adadelta algorithm [...] We made use of a
    # fairly  high  learning  rate  of  1.0  instead  of  the  relatively  low
    # default of  0.002 to  train the  network"
    lr = kwargs.setdefault("lr", 1.0)
    model = HSGAN_Discriminator()
    # For Adadelta, we need to load the model on GPU before creating the optimizer
    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])

    model = model.to(device)
    epoch = kwargs.setdefault("epoch", 100)
    kwargs.setdefault(
        "scheduler",
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=epoch // 4, verbose=True
        ),
    )
    # kwargs.setdefault('scheduler', None)
    kwargs.setdefault("batch_size", 100)
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs


class HSGAN_Generator(nn.Module):
    """
    Semisupervised Hyperspectral Image Classification Based on Generative Adversarial Networks
    Ying Zhan , Dan Hu, Yuntao Wang
    http://www.ieee.org/publications_standards/publications/rights/index.html
    """

    def __init__(self):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(HSGAN_Generator, self).__init__()
        self.fc1 = nn.Linear(100, 1024)

        self.fc2 = nn.Linear(1024, 6400)
        self.fc2_bn = nn.BatchNorm1d(1)

        self.up1 = nn.Upsample(size=100)
        self.up1_bn = nn.BatchNorm1d(128)

        self.conv1 = nn.Conv1d(128, 64, 1)
        self.up2 = nn.Upsample(size=200)
        self.conv2 = nn.Conv1d(64, 1, 1)

        # self.linear_blocks = nn.Sequential(
        #     nn.Linear(100, 1024),
        #     nn.Linear(1024, 6400),
        # )
        #
        # self.conv_blocks = nn.Sequential(
        #     nn.Upsample(size=100),
        #     nn.Conv1d(128, 64, 1),
        #     nn.Upsample(size=200),
        #     nn.Conv1d(64, 1, 1)
        # )
    # shape of x is 100*1*100
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2_bn(self.fc2(x)))
        x = x.view(100, 128, 50)
        x = self.up1_bn(self.up1(x))
        x = torch.tanh(self.conv1(x))
        x = self.up2(x)
        x = self.conv2(x)
        # x = self.linear_blocks(x)
        # x = x.view(100, 128, 50)
        # x = self.conv_blocks(x)
        return x

class HSGAN_Discriminator(nn.Module):
    """
    Semisupervised Hyperspectral Image Classification Based on Generative Adversarial Networks
    Ying Zhan , Dan Hu, Yuntao Wang
    http://www.ieee.org/publications_standards/publications/rights/index.html
    """

    # input_channels=200 n_classes=17
    def __init__(self):
        # The proposed network model uses a single recurrent layer that adopts our modified GRUs of size 64 with sigmoid gate activation and PRetanh activation functions for hidden representations
        super(HSGAN_Discriminator, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 1)
        self.conv2 = nn.Conv1d(32, 32, 3)



        # self.conv_blocks1 = nn.Sequential(
        #     nn.Conv1d(1, 32, 1),
        #     nn.Conv1d(32, 32, 3),
        # )
        self.max_pooling = nn.MaxPool1d(2, stride=2)
        self.conv3 = nn.Conv1d(32, 32, 3)
        self.linear_block = nn.Linear(1504, 1024)
        # real or fake

        self.real_or_fake = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        # class

        self.classify = nn.Sequential(
            nn.Linear(1024, 16),
            nn.Softmax(dim=1),
        )

    def forward(self, x):   #(100, 1, 200)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.max_pooling(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv3(x))
        x = self.max_pooling(x)
        x = x.view(100, 1, 1504)
        x = self.linear_block(x)
        # real or fake
        validity = self.real_or_fake(x)
        # classify
        label = self.classify(x)
        return validity, label

def train(
    discriminator,
    optimizer_D,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
    supervision="full",
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    # 初始化参数
    # Loss functions
    adversarial_loss = torch.nn.BCELoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()

    # Initialize generator and discriminator
    generator = HSGAN_Generator()
    # discriminator = Discriminator()  这个通过参数传过来

    # Initialize weights  暂时先不初始化
    # generator.apply(weights_init_normal)
    # discriminator.apply(weights_init_normal)
    generator.to(device)
    discriminator.to(device)

    save_epoch = epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)


    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        discriminator.train()
        avg_loss = 0.0

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):

            # Adversarial ground truths
            valid = torch.ones((100, 1), dtype=torch.float32)
            fake = torch.zeros((100, 1), dtype=torch.float32)

            # Load the data into the GPU if required
            # data, target = data.to(device), target.to(device)
            # network shape error
            real_imgs, labels = data.view(100, 1, 200).to(device), target.to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            z = torch.randn((100, 1, 100))
            gen_labels = torch.zeros((100, ), dtype=torch.int64)

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            # network shape error
            validity = validity.view(100, 1)
            pred_label = pred_label.view(100, 16)

            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            output, real_aux = discriminator(real_imgs)
            # real_pred, real_aux = discriminator(data)
            # loss = criterion(output, target)

            # network shape error
            output = output.view(100, 1)
            real_aux = real_aux.view(100, 16)


            d_real_loss = (adversarial_loss(output, valid) + auxiliary_loss(real_aux, labels)) / 2

            # Loss for fake images
            fake_pred, fake_aux = discriminator(gen_imgs.detach())

            # network shape error
            fake_pred = fake_pred.view(100, 1)
            fake_aux = fake_aux.view(100, 16)

            d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            avg_loss += d_loss.item()
            losses[iter_] = d_loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
                    win=loss_win,
                    update=update,
                    opts={
                        "title": "Training loss",
                        "xlabel": "Iterations",
                        "ylabel": "Loss",
                    },
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(
                        Y=np.array(val_accuracies),
                        X=np.arange(len(val_accuracies)),
                        win=val_win,
                        opts={
                            "title": "Validation accuracy",
                            "xlabel": "Epochs",
                            "ylabel": "Accuracy",
                        },
                    )
            iter_ += 1
            del (data, target, d_loss, output)

        # Update the scheduler
        # avg_loss /= len(data_loader)
        # if val_loader is not None:
        #     val_acc = val(discriminator, val_loader, device=device, supervision=supervision)
        #     val_accuracies.append(val_acc)
        #     metric = -val_acc
        # else:
        #     metric = avg_loss
        #
        # if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        #     scheduler.step(metric)
        # elif scheduler is not None:
        #     scheduler.step()

        # Save the weights
        # if e % save_epoch == 0:
        #     save_model(
        #         discriminator,
        #         camel_to_snake(str(discriminator.__class__.__name__)),
        #         data_loader.dataset.name,
        #         epoch=e,
        #         metric=abs(metric),
        #     )


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/"
    """
    Using strftime in case it triggers exceptions on windows 10 system
    """
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + "_epoch{epoch}_{metric:.2f}".format(
            **kwargs
        )
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        filename = time_str
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + ".pkl")


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img, **kwargs)),
        total=(iterations),
        desc="Inference on the image",
    ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                data = data.unsqueeze(1)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            real_pred, output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs


def val(net, data_loader, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == "full":
                real_pred, output = net(data)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total
