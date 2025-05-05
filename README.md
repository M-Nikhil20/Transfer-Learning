# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Develop a deep learning model for image classification using transfer learning. Utilize the pre-trained VGG19 model as the feature extractor, fine-tune it, and adapt it to classify images into specific categories.
</br>
</br>
</br>

## DESIGN STEPS
### **Step 1: Import Libraries and Load Dataset**
- Import the necessary libraries.
- Load the dataset.
- Split the dataset into training and testing sets.

### **Step 2: Initialize Model, Loss Function, and Optimizer**
- Define the model architecture.
- Use `CrossEntropyLoss` for multi-class classification.
- Choose the `Adam` optimizer for efficient training.

### **Step 3: Train the Model**
- Train the model using the training dataset.
- Optimize the model parameters to minimize the loss.

### **Step 4: Evaluate the Model**
- Test the model using the testing dataset.
- Measure performance using appropriate evaluation metrics.

### **Step 5: Make Predictions on New Data**
- Use the trained model to predict outcomes for new inputs.


## PROGRAM
Include your code here
```python
# Load Pretrained Model and Modify for Transfer Learning

model = models.vgg19(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_features = model.classifier[-1].in_features
num_classes = len(train_dataset.classes)

model.classifier[-1] = nn.Linear(num_features, 1)




# Modify the final fully connected layer to match the dataset classes

for param in model.features.parameters():
    param.requires_grad = False 


# Include the Loss function and optimizer

criterion = nn .BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

print(criterion,optimizer)


# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:  Sakthivel B")
    print("Register Number: 212222040141")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


```

## OUTPUT
### Training and Vision Loss
![image](https://github.com/user-attachments/assets/fdc0c379-d7da-4ca3-9a99-36aeae3f0f16)

</br>
</br>
</br>


### Confusion Matrix
![image](https://github.com/user-attachments/assets/28180b21-ea64-479a-a8ef-b7544b6d5394)

</br>
</br>
</br>

### Classification Report
![image](https://github.com/user-attachments/assets/aac2b964-e400-4ada-afd5-4fad8196eb03)

</br>
</br>
</br>

### New Sample Prediction
![image](https://github.com/user-attachments/assets/d3edbd54-09d3-4278-99ab-e927c5efddfc)


</br>
</br>
</br>

## RESULT
Thus, the transfer Learning for classification using VGG-19 architecture has succesfully implemented.
</br>
</br>
</br>
