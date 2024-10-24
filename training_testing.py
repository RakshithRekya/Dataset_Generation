# Importing necessary libraries for the task
import torch
import numpy as np
from plots import confusion_matrix_2

# approach 1's training and testing method
def training_testing_1(num_epochs, train_loader, test_loader, 
                       model, optimizer, entropy_loss, device, model_type):
    # Initialising necessary lists to store data for plots
    accuracies_train = []
    losses_train = []
    accuracies_test = []
    losses_test = []
    epochs = []
    train_labels = []
    train_predictions = []
    test_labels = []
    test_predictions = []
    
    # Iterating through the number of epochs given
    for epoch in range(num_epochs):
        #Training on train dataset
        model.train()
        
        # Initializing necessary variables to calculate accuracy, loss values
        # and to store predictions and labels for confusion matrix plot
        total_train_accuracy = 0.0
        total_train_loss = 0.0
        total_train = 0
        epoch_labels_train = []
        epoch_predictions_train = []
        
        # Iterating through the train loader
        for i, data in enumerate(train_loader):
            inputs, labels = data
            
            # Setting the device for both inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            
            # Setting the optimizer's gradients to zero at the beginning of each epoch
            # so that the training of the model is not interfered.
            optimizer.zero_grad()
            
            # Getting the outputs by passing the inputs in the model
            outputs = model(inputs)
            
            # Calculating loss for train data
            loss_train = entropy_loss(outputs,labels)
            loss_train.backward()
            total_train_loss += loss_train.item()
            
            # Steeping the optimizer
            optimizer.step()
            
            # Calculating the total images in the train loader (i.e. batch sizes)
            total_train += inputs.size(0)
            
            # Calculating the prediction using torch.max from the outputs
            _, prediction = torch.max(outputs.data,1)
            
            # Storing the labels and predictions for the confusion matrix
            epoch_labels_train.extend(labels.detach().cpu().tolist())
            epoch_predictions_train.extend(prediction.detach().cpu().tolist())
            
            # Calculating  total train accuracy for the entire per epoch
            total_train_accuracy += (prediction == labels).sum().item()
        
        # Calculating the average train accuracy per epoch (not per batch)---
        train_accuracy_per_epoch = (total_train_accuracy/total_train) * 100
        
        # Similarly for loss
        train_loss_per_epoch = total_train_loss / (i + 1)
        
        # Storing for the plots
        accuracies_train.append(train_accuracy_per_epoch)
        losses_train.append(train_loss_per_epoch)
        train_labels.append(epoch_labels_train)
        train_predictions.append(epoch_predictions_train)
        
        
        # Evaluation on testing dataset
        model.eval()
        total_test_accuracy = 0
        total_test_loss = 0.0
        total_test = 0
        
        epoch_labels_test = []
        epoch_predictions_test = []
        
        for i, data in enumerate(test_loader):
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device).long()
            
            total_test += inputs.size(0)
            
            outputs = model(inputs)
            loss_test = entropy_loss(outputs,labels)
            total_test_loss += loss_test.item()
            
            _, prediction=torch.max(outputs.data,1)
            epoch_labels_test.extend(labels.detach().cpu().tolist())
            epoch_predictions_test.extend(prediction.detach().cpu().tolist())
    
            total_test_accuracy += (prediction == labels).sum().item()
    
        test_accuracy_per_epoch = (total_test_accuracy/total_test) * 100
        test_loss_per_epoch = total_test_loss / (i + 1)
        
        accuracies_test.append(test_accuracy_per_epoch)
        losses_test.append(test_loss_per_epoch)
        
        epochs.append(epoch)
        
        test_labels.append(epoch_labels_test)
        test_predictions.append(epoch_predictions_test)
        
    
        
        print(f"epoch: {epoch}/{num_epochs-1}, train_loss of this epoch: {train_loss_per_epoch:.4f}, train_accuracy of this epoch: {train_accuracy_per_epoch:.2f}, test_loss of this epoch: {test_loss_per_epoch:.4f}, accuracy_test of this epoch: {test_accuracy_per_epoch:.2f}\n")
        
    return accuracies_train, losses_train, accuracies_test, losses_test, epochs, train_labels, test_labels, train_predictions, test_predictions

# approach 2's training and testing method
def training_testing_2(num_epochs, train_loader, test_loader, 
                       model, optimizer, bce_loss, device, model_type):
    # Initialising necessary lists to store data for plots
    accuracies_train = []
    losses_train = []
    accuracies_test = []
    losses_test = []
    epochs = []
    
    # Iterating through the number of epochs given
    for epoch in range(num_epochs):
        #Training on training dataset
        model.train()
        
        # Initializing necessary variables to calculate accuracy, loss values
        # and to store predictions and labels for confusion matrix plot
        total_train_accuracy = 0
        total_train_loss = 0.0
        total_train = 0
        train_labels_epoch = []
        train_predictions_epoch = []
        
        # Iterating through the train loader
        for i, data in enumerate(train_loader):
            inputs, labels = data
            
            # Setting the device for both inputs and labels
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            # Setting the optimizer's gradients to zero at the beginning of each epoch
            # so that the training of the model is not interfered.
            optimizer.zero_grad()
            
            # Getting the outputs by passing the inputs in the model
            outputs = model(inputs)
            
            # Calculating loss for train data
            loss_train = bce_loss(outputs,labels)
            loss_train.backward()
            total_train_loss += loss_train.item()
            
            # Steeping the optimizer
            optimizer.step()
            
            # Calculating the total images in the train loader (i.e. batch sizes)
            total_train += inputs.size(0)
            
            # Calculating the prediction from the outputs
            prediction = (outputs.detach() > 0.5).float()
            train_labels_epoch.append(labels.cpu().numpy())
            train_predictions_epoch.append(prediction.cpu().numpy())
    
            accuracies_attributes = (prediction == labels).float()
            total_train_accuracy += accuracies_attributes.mean() * inputs.size(0)
            
            #revist 
        train_accuracy_per_epoch = (total_train_accuracy/total_train) * 100
        train_loss_per_epoch = total_train_loss / (i + 1)
        
        train_labels_epoch = np.vstack(train_labels_epoch)
        train_predictions_epoch = np.vstack(train_predictions_epoch)
        
        
        attribute_names = ["male", "black_hair", "mustache", "glasses", "beard"]
        for i, attribute_name in enumerate(attribute_names):
            if model_type == 'cnn':
                confusion_matrix_2(train_labels_epoch[:, i], train_predictions_epoch[:, i], attribute_name, epoch, data_type= 'train', model_type='cnn')
            else:
                confusion_matrix_2(train_labels_epoch[:, i], train_predictions_epoch[:, i], attribute_name, epoch, data_type= 'train', model_type='lnn')
        
        accuracies_train.append(train_accuracy_per_epoch.item())
        losses_train.append(train_loss_per_epoch)
        
        # Evaluation on testing dataset
        model.eval()
        total_test_accuracy = 0
        total_test_loss = 0.0
        total_test = 0
        test_labels_epoch = []
        test_predictions_epoch = []
        
        for i, data in enumerate(test_loader):
            inputs, labels = data
            
            inputs = inputs.to(device)
            labels = labels.to(device).float()
            
            total_test += inputs.size(0)
            
            outputs = model(inputs)
            loss_test = bce_loss(outputs,labels)
            total_test_loss += loss_test.item()
            
            prediction = (outputs.detach() > 0.5).float()
            test_labels_epoch.append(labels.cpu().numpy())
            test_predictions_epoch.append(prediction.cpu().numpy())
            
            accuracies_attributes = (prediction == labels).float()
            total_test_accuracy += accuracies_attributes.mean() * inputs.size(0)
    
        test_accuracy_per_epoch = (total_test_accuracy/total_test) * 100
        test_loss_per_epoch = total_test_loss / (i + 1)
        
        test_labels_epoch = np.vstack(train_labels_epoch)
        test_predictions_epoch = np.vstack(train_predictions_epoch)
        
        
        attribute_names = ["male", "black_hair", "mustache", "glasses", "beard"]
        for i, attribute_name in enumerate(attribute_names):
            if model_type == 'cnn':
                confusion_matrix_2(train_labels_epoch[:, i], train_predictions_epoch[:, i], attribute_name, epoch, data_type= 'test', model_type='cnn')
            else:
                confusion_matrix_2(train_labels_epoch[:, i], train_predictions_epoch[:, i], attribute_name, epoch, data_type= 'test', model_type='lnn')
        
        accuracies_test.append(test_accuracy_per_epoch.item())
        losses_test.append(test_loss_per_epoch)
        
        epochs.append(epoch)
        
        print(f"epoch: {epoch}/{num_epochs-1}, train_loss of this epoch: {train_loss_per_epoch:.4f}, train_accuracy of this epoch: {train_accuracy_per_epoch:.2f}, test_loss of this epoch: {test_loss_per_epoch:.4f}, accuracy_test of this epoch: {test_accuracy_per_epoch:.2f}\n")
        
    return accuracies_train, losses_train, accuracies_test, losses_test, epochs