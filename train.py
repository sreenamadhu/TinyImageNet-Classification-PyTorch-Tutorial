import torch.nn as nn
import torch.optim as optim
import torch
import os

def train_model(model, train_dataloader, num_epochs):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(num_epochs):
        running_loss = 0.0
        corrects = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()   
            # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-200 in this case)
            outputs = model(inputs)

            # Define our loss function, and compute the loss
            loss = criterion(outputs, labels)
            # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
            loss.backward()
            # Update the neural network weights
            optimizer.step()

            # print statistics
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            running_loss += loss.item() * inputs.size(0)
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'Epoch #{epoch+1} Iteration #{i + 1:5d} loss: {running_loss / len(train_dataloader.dataset) :.3f}')
                running_loss = 0.0
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = corrects/float(total)
        print(f'Epoch {epoch+1} :: Loss - {epoch_loss:.3f} Accuracy - {epoch_acc:.4f}')
        if not os.path.isdir('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), f'models/epoch_{epoch+1}.pth')
    return