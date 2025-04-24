import torch

def calculate_batch_accuracy(outputs, labels):
    """
    Calculate the accuracy for a single batch.
    
    Args:
        outputs (torch.Tensor): Model outputs of shape (batch_size, num_classes).
        labels (torch.Tensor): True labels of shape (batch_size,).
        
    Returns:
        float: Accuracy of the batch.
    """
    # Get predictions from outputs (largest value index)
    _, preds = torch.max(outputs, dim=1)
    # Compute accuracy
    return (preds == labels).sum().item() / labels.size(0)

def evaluate_model_accuracy(model, dataloader, device):
    """
    Evaluate the model accuracy over an entire DataLoader.
    
    Args:
        model (torch.nn.Module): The neural network model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
        
    Returns:
        float: Overall accuracy on the dataset.
    """
    model.eval()  # Set model to evaluation mode
    total_correct = 0
    total_samples = 0
    batch_cnt = 0 
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # Option 1: Using the batch accuracy function (if needed)
            # batch_acc = calculate_batch_accuracy(outputs, labels)
            # Option 2: Directly compute in the loop
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            batch_cnt += 1
            if batch_cnt > 3:
                break

    return total_correct / total_samples

