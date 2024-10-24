import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import torch.nn as nn
import torch
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix

def plots(accuracies_train, losses_train, accuracies_test, losses_test, epochs, approach, model_type):
    if model_type == 'cnn':
        results_dir = os.path.join(os.getcwd(), 'results_cnn')
        accuracy_dir = os.path.join(results_dir, 'accuracy')
        loss_dir = os.path.join(results_dir, 'loss')
    
    else:
        results_dir = os.path.join(os.getcwd(), 'results_lnn')
        accuracy_dir = os.path.join(results_dir, 'accuracy')
        loss_dir = os.path.join(results_dir, 'loss')
    
    # Plotting the train loss values
    plt.plot(epochs, losses_train, color='red', label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(epochs)))
    plt.legend()
    plt.savefig(os.path.join(loss_dir, f'loss_plot_train_{approach}.png'))
    plt.show()
    
    # Plotting the test loss values
    plt.plot(epochs, losses_test, color='blue', label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(epochs)))
    plt.legend()
    plt.savefig(os.path.join(loss_dir, f'loss_plot_test_{approach}.png'))
    plt.show()
    
    # Plotting the training accuracy values
    plt.plot(epochs, accuracies_train, color='darkred', label='Testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy in %')
    plt.xticks(range(0, len(epochs)))
    plt.savefig(os.path.join(accuracy_dir, f'accuracy_plot_train_{approach}.png'))
    plt.show()
    
    # Plotting the testing accuracy values
    plt.plot(epochs, accuracies_test, color='darkblue', label='Testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Testing Accuracy in %')
    plt.xticks(range(0, len(epochs)))
    plt.savefig(os.path.join(accuracy_dir, f'accuracy_plot_test_{approach}.png'))
    plt.show()
    
    #Plotting testing vs training accuracy
    plt.plot(epochs, accuracies_train, color='darkred', label='Training accuracy')
    plt.plot(epochs, accuracies_test, color='darkblue', label='Testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy in %')
    plt.xticks(range(0, len(epochs)))
    plt.legend()
    plt.title('Training vs Testing accuracy of CNN model')
    plt.savefig(os.path.join(accuracy_dir, f'accuracy_plot_train_test_{approach}.png'))
    plt.show()
    
    #Plotting testing vs training loss
    plt.plot(epochs, losses_train, color='red', label='Training loss')
    plt.plot(epochs, losses_test, color='blue', label='Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(range(0, len(epochs)))
    plt.legend()
    plt.title('Training vs Testing Loss')
    plt.show()
    plt.savefig(os.path.join(loss_dir, f'loss_plot_train_test_{approach}.png'))
    plt.show()

def confusion_matrix_1(train_labels, train_predictions, test_labels, test_predictions, epochs, model_type='cnn'):
    if model_type == 'cnn':
        results_dir = os.path.join(os.getcwd(), 'results_cnn')
        cm_dir = os.path.join(results_dir, 'confusion_matrix')
        
    else:
        results_dir = os.path.join(os.getcwd(), 'results_lnn')
        cm_dir = os.path.join(results_dir, 'confusion_matrix')
        
    for epoch in epochs:
        true_labels_train = train_labels[epoch]
        predicted_labels_train = train_predictions[epoch]
        conf_matrix = confusion_matrix(true_labels_train, predicted_labels_train)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(conf_matrix, annot=True, fmt='.0%', cmap='Blues',
                    cbar_kws={'format': mtick.PercentFormatter(xmax=1.0, decimals=0)},
                    xticklabels=[i+1 for i in range(6)], 
                    yticklabels=[i+1 for i in range(6)])
        plt.title(f"Confusion matrix of train data, epoch: {epoch}")
        plt.savefig(os.path.join(cm_dir,
                f'confusion_matrix_train_approach_1_{epoch}.png'))
        plt.show()
        
        true_labels_test = test_labels[epoch]
        predicted_labels_test = test_predictions[epoch]
        conf_matrix = confusion_matrix(true_labels_test, predicted_labels_test)
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(conf_matrix, annot=True, fmt='.0%', cmap='Blues',
                    cbar_kws={'format': mtick.PercentFormatter(xmax=1.0, decimals=0)},
                    xticklabels=[i+1 for i in range(6)], 
                    yticklabels=[i+1 for i in range(6)])
        plt.title(f"Confusion matrix of test data, epoch: {epoch}")
        plt.savefig(os.path.join(cm_dir, f'confusion_matrix_test_data approach 1_{epoch}.png'))
        plt.show()
        
def confusion_matrix_2(true_labels, predictions, attribute_name, epoch, data_type='test', model_type='cnn'):
    if model_type=='cnn':
        results_dir = os.path.join(os.getcwd(), 'results_cnn')
        cm_dir = os.path.join(results_dir, 'confusion_matrix')
        
    else:
        results_dir = os.path.join(os.getcwd(), 'results_lnn')
        cm_dir = os.path.join(results_dir, 'confusion_matrix')
        
    cm = confusion_matrix(true_labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax,
                xticklabels=['No ' + attribute_name, attribute_name],
                yticklabels=['No ' + attribute_name, attribute_name])
    ax.set_title(f'Confusion Matrix for {attribute_name}, Epoch {epoch}, {data_type}') 
    plt.savefig(os.path.join(cm_dir, f'confusion_matrix_{data_type}_approach_2_{attribute_name}_epoch_{epoch}.png'))
    plt.show()

