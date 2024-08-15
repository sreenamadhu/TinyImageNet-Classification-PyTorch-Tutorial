import torch.nn as nn
import torch.optim as optim
import torch
import os

def train_model(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_acc = 0
    logs_f = open('logs.txt','w+')
    for epoch in range(num_epochs):

        #Training mode
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()   
            # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
            outputs = model(inputs)

            # Define our loss function, and compute the loss
            loss = criterion(outputs, labels)
            # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
            loss.backward()
            # Update the neural network weights
            optimizer.step()

            # Log statistics
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            running_loss += loss.item() * inputs.size(0)
            if i % 200 == 199:    # Log every 200 mini-batches
                logs_f.write(f'Epoch #{epoch+1} Iteration #{i + 1:5d} loss: {running_loss / len(train_dataloader.dataset) :.3f}\n')
                running_loss = 0.0
        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        epoch_train_acc = corrects/float(total)
        

        #Validation mode
        model.eval()
        corrects = 0
        total = 0
        for i, (inputs, labels) in enumerate(val_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        epoch_val_acc = corrects/float(total)
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            if not os.path.isdir('models'):
                os.makedirs('models')
            torch.save(model.state_dict(), f'models/epoch_{epoch+1}_valAcc_{epoch_val_acc*100:.1f}.pth')
            logs_f.write(f'Best Validation Accuracy Reported : {epoch_val_acc*100} Model Saved!!\n')
            print(f'Best Validation Accuracy Reported : {epoch_val_acc*100:.2f} Model Saved!!')

        logs_f.write(f'Epoch {epoch+1} :: Train Loss - {epoch_train_loss:.3f} Train Accuracy - {epoch_train_acc:.4f} Validation Accuracy - {epoch_val_acc:.4f}\n')
        print(f'Epoch {epoch+1} :: Train Loss - {epoch_train_loss:.3f} Train Accuracy - {epoch_train_acc:.4f} Validation Accuracy - {epoch_val_acc:.4f}')
    logs_f.close()
    return

