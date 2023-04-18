import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import csv

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

# Separate test dataset
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_autoencoder(encoder, decoder, save_model=False, d=10, layers=2):
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion = nn.MSELoss()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

    epochs = 20
    train_losses = []
    test_losses = []

    # Training loop
    encoder.train()
    decoder.train()
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, _ = data
            inputs = inputs.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Forward pass
            latent_vectors = encoder(inputs)
            decoded_img = decoder(latent_vectors)
            loss = criterion(decoded_img, inputs)

            # Backward pass
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

        # Print loss for this epoch
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}')
        torch.cuda.empty_cache()

        # Evaluation loop
        encoder.eval()
        decoder.eval()
        test_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(test_loader, 0):
                inputs, _ = data
                inputs = inputs.to(device)
                outputs = decoder(encoder(inputs))
                loss = criterion(outputs, inputs)
                test_loss += loss.item()

        # Print test loss
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        print(f'Test Loss: {test_loss:.4f}')

        # Compare train loss with test loss
        if epoch > 0 and test_losses[-1] > test_losses[-2]:
            print("Test loss increased. Early stopping...")
            break

    # Save trained model
    # if save_model:
    torch.save(encoder.state_dict(), f'encoder_layers{layers}_d{d}.pt')
    torch.save(decoder.state_dict(), f'decoder_layers{layers}_d{d}.pt')

    torch.cuda.empty_cache()

    return train_losses, test_losses


class Encoder_depth2(nn.Module):
    def __init__(self, d):
        super(Encoder_depth2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, d)
        self.flatten = nn.Flatten()
        self.d = d

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Decoder_depth2(nn.Module):
    def __init__(self, d):
        super(Decoder_depth2, self).__init__()
        self.fc = nn.Linear(d, 32 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 32, 7, 7)
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


def q1_1_train():
    """"q1 same architectures, different lateral space size"""
    res = [train_autoencoder(Encoder_depth2(d), Decoder_depth2(d), False, d) for d in [4, 10, 16, 64, 128, 256]]
    # res = [train_autoencoder(Encoder_depth2(d), Decoder_depth2(d), False, d) for d in [128]]
    # res = ['b','f','r']
    with open("train_results.csv", "w", newline="") as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        writer.writerow("pairs of training and test results")
        for train_test_res in res:
            writer.writerow(train_test_res[0])
            writer.writerow(train_test_res[1])


def q1_plot():
    training_loss = []
    test_loss = []
    with open('train_results.csv', "r") as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader, 0):
            losses = list(map(float, line))
            if i % 2 == 0:
                training_loss.append(losses)
            else:
                test_loss.append(losses)

    # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # Flatten axes to a 1D array
    axes = axes.flatten()
    title = ['4', '10', '16', '64', '128', '256']
    # Loop through the axes and plot the data
    for i in range(len(training_loss)):
        # Get the data for the subplot
        epochs = range(30)
        label = 'training' if i % 2 == 0 else 'test'

        # Plot the data
        axes[i].plot(epochs[1:len(training_loss[i]) + 1], training_loss[i], label=label)
        axes[i].plot(epochs[1:len(test_loss[i]) + 1], test_loss[i], label=label)
        if i == 0 or i == 3: axes[i].set_ylabel('loss')
        if i > 2: axes[i].set_xlabel('epochs')
        axes[i].set_title(label=title[i])
        axes[i].legend()

        plt.suptitle(t="loss on a 6-layers depth with latent space:")

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    # Show the plot
    plt.show()

    input('finished')


def q1_2_train():
    pass


def main():
    # q1_1_train()
    # q_1_plot()
    q1_2_train()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
