import torch
import torch.nn as nn

from utils import Utils


training_data_path = './data/training_set/training_set'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
model, optimizer, best_loss, pth_path = Utils.get_model_optimizer_best_loss_resnet(device, learning_rate=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


train_loader, val_loader = Utils.training_data_loader(training_data_path)
print('\n✅ Data loaded!\n')


# Step 4: Training loop
epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {running_loss/len(train_loader):.8f}, Val Accuracy: {accuracy:.8f}%")

    # Save the best model based on training loss
    if epoch_loss < best_loss:
        best_loss = epoch_loss

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, pth_path)
        print("✅ Saved Best Model")

    scheduler.step()