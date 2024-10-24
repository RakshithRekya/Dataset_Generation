from dataloader import loaders_1, loaders_2
from torch.optim import Adam
from models import CNN_model_1, CNN_model_2, LNN_model_1, LNN_model_2
from training_testing import training_testing_1, training_testing_2
from plots import plots, confusion_matrix_1
import torch
import torch.nn as nn

num_epochs = 1

train_loader_1, test_loader_1 = loaders_1()
train_loader_2, test_loader_2 = loaders_2()

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(device)

# CNN
model_1 = CNN_model_1().to(device)
print(model_1)

optimizer_1 = Adam(model_1.parameters(),lr=0.001,weight_decay=0.0001)

model_2 = CNN_model_2().to(device)
print(model_2)

optimizer_2 = Adam(model_2.parameters(),lr=0.001,weight_decay=0.0001)

entropy_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()

# approach 1
print("Approach 1 with cnn begun\n")
accuracies_train_1, losses_train_1, accuracies_test_1, losses_test_1, epochs, train_labels, test_labels, train_predictions, test_predictions = training_testing_1(num_epochs, train_loader_1, test_loader_1, 
                        model_1, optimizer_1, entropy_loss, device, model_type='cnn')

plots(accuracies_train_1, losses_train_1, accuracies_test_1, losses_test_1, epochs, approach=1, model_type='cnn')
confusion_matrix_1(train_labels, train_predictions, test_labels, test_predictions, epochs, model_type='cnn')
print("Approach 1 with cnn ended\n")

# approach 2
print("Approach 2 with cnn begun\n")
accuracies_train_2, losses_train_2, accuracies_test_2, losses_test_2, epochs = training_testing_2(num_epochs, train_loader_2, test_loader_2, 
                        model_2, optimizer_2, bce_loss, device, model_type='cnn')

plots(accuracies_train_2, losses_train_2, accuracies_test_2, losses_test_2, epochs, approach=2, model_type='cnn')
print("Approach 2 with cnn ended\n")


# LNN
model_1 = LNN_model_1().to(device)
print(model_1)

optimizer_1 = Adam(model_1.parameters(),lr=0.001,weight_decay=0.0001)

model_2 = LNN_model_2().to(device)
print(model_2)

optimizer_2 = Adam(model_2.parameters(),lr=0.001,weight_decay=0.0001)

entropy_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()

# approach 1
print("Approach 1 with lnn begun\n")
accuracies_train_1, losses_train_1, accuracies_test_1, losses_test_1, epochs, train_labels, test_labels, train_predictions, test_predictions = training_testing_1(num_epochs, train_loader_1, test_loader_1, 
                        model_1, optimizer_1, entropy_loss, device, model_type='lnn')

plots(accuracies_train_1, losses_train_1, accuracies_test_1, losses_test_1, epochs, approach=1, model_type='lnn')
confusion_matrix_1(train_labels, train_predictions, test_labels, test_predictions, epochs, model_type='lnn')
print("Approach 1 with lnn ended\n")

# approach 2
print("Approach 2 with lnn begun\n")
accuracies_train_2, losses_train_2, accuracies_test_2, losses_test_2, epochs = training_testing_2(num_epochs, train_loader_2, test_loader_2, 
                        model_2, optimizer_2, bce_loss, device, model_type='lnn')

plots(accuracies_train_2, losses_train_2, accuracies_test_2, losses_test_2, epochs, approach=2, model_type='lnn')
print("Approach 2 with lnn ended\n")
