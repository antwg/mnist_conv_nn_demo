import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pylab as plt
import tkinter as tk

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageDraw, ImageOps

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(batch_size = 8192, epochs = 10):
    # Create a neural network
    model = neural_network().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters())
    loss_function = nn.CrossEntropyLoss()

    # Load the training data
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_data = DataLoader(train_dataset, batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        with tqdm(total=len(train_dataset)) as pbar:
            for imgs, labels in train_data:
                imgs, labels = imgs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(imgs)
                
                loss = loss_function(outputs, labels)
                loss.backward()
                
                optimizer.step()

                # Save for plotting
                losses.append(loss.item())
                pbar.set_postfix(loss=(sum(losses) / len(losses)))
                pbar.update(len(imgs))

    plt.plot(losses)
    plt.show()

    return model

def validate_model(model, batch_size=8192):
    # Load the validation data
    transform = transforms.Compose([transforms.ToTensor()])
    val_dataset = datasets.MNIST('.', train=False, download=True, transform=transform)
    val_data = DataLoader(val_dataset, batch_size, shuffle=True)

    model.eval()

    # Validate the model
    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0
        for imgs, labels in val_data:
            imgs, labels = imgs.to(device), labels.to(device)

            # forward
            outputs = model(imgs)

            # Get the predicted class
            _, output = torch.max(outputs, 1)

            # Count the number of correct predictions
            total_predictions += len(output)
            correct_predictions += torch.sum(output == labels)

        print(f'Validation accuracy: {correct_predictions/total_predictions*100:.2f}')


class neural_network(nn.Module):
        
    def __init__(self):
        super(neural_network, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2) # 28x28x1 tensor
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2) # 14x14x32 tensor
        
        # Pooling layer to reduce the size of the image tensor
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
       
        # Dropout layer
        self.drop = nn.Dropout(0.2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, 10)  # 10 output classes for MNIST

    def forward(self, x):
        # Convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten image input
        x = x.view(-1, 64 * 7 * 7)

        # Add dropout layer
        x = self.drop(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x
    

def draw_digit():
    # Set up the drawing canvas
    window = tk.Tk()
    canvas_width, canvas_height = 280, 280
    canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg='white')
    canvas.pack()

    image = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(image)

    def paint(event):
        x1, y1 = (event.x - 15), (event.y - 15)
        x2, y2 = (event.x + 15), (event.y + 15)
        canvas.create_oval(x1, y1, x2, y2, fill='black', width=5)
        draw.ellipse([x1, y1, x2, y2], fill='black')

    canvas.bind("<B1-Motion>", paint)

    def save():
        filename = "drawing.png"
        image.save(filename)
        window.destroy()

    button_save = tk.Button(window, text='Save', command=save)
    button_save.pack()

    window.mainloop()

def image_to_tensor(image_path):
    image = Image.open(image_path).convert('L')
    image = ImageOps.invert(image) # Invert the image to match MNIST
    
    # Define transformations: Resize to 28x28, Convert to tensor
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    # Apply transformations to the image and add a batch dimension
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

if __name__ == "__main__":
    TRAIN_MODEL = False
    if TRAIN_MODEL:
        model = train_model(epochs=10)
        validate_model(model)
        torch.save(model.state_dict(), 'mnist_cnn.pth')
    else:
        model = neural_network().to(device)
        model.load_state_dict(torch.load('mnist_cnn.pth'))
        model.eval()
        validate_model(model)
        
    # Draw a digit and classify it
    while True:
        draw_digit()
        image_tensor = image_to_tensor("drawing.png")
        with torch.no_grad():
            model.eval()
            output = model(image_tensor.to(device))
            _, predicted = torch.max(output, 1)
            print(f'Predicted: {predicted.item()}')

