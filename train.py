import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.cuda.amp as amp
import os
from datetime import datetime
import matplotlib.pyplot as plt

from configs.default_config import config
from data.fer2013_dataset import EnhancedFER2013
from models.resemotenet import ResEmoteNetV2
from utils.metrics import calculate_metrics, save_metrics
from utils.visualization import plot_confusion_matrix, plot_learning_curves, plot_lr_schedule
from utils.helpers import set_seed, create_dirs

def main():
    # Setup directories
    create_dirs([config.CHECKPOINT_DIR, config.PLOT_DIR])
    
    # Set seed
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data loaders
    train_loader, val_loader, test_loader = create_loaders()
    
    # Model
    model = ResEmoteNetV2().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    scheduler_plateau = ReduceLROnPlateau(optimizer, mode='max', patience=config.PATIENCE//2, factor=0.5)
    scaler = amp.GradScaler()
    
    # Training variables
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    lr_history = []
    
    # Training loop
    for epoch in range(config.EPOCHS):
        # Train epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Update LR history
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch statistics
        print(f"Epoch {epoch+1}/{config.EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Update schedulers
        scheduler_cosine.step()
        scheduler_plateau.step(val_acc)
        
        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))
            print("  New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
    
    # Save and display training curves
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, 
                         os.path.join(config.PLOT_DIR, 'learning_curves.png'))
    plt.show()  # Display the plot
    
    # Save and display LR schedule
    plot_lr_schedule(lr_history, os.path.join(config.PLOT_DIR, 'lr_schedule.png'))
    plt.show()  # Display the plot
    
    # Test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')))
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device, return_predictions=True)
    
    # Calculate and save test metrics
    test_metrics = calculate_metrics(y_true, y_pred, config.CLASSES)
    save_metrics(test_metrics, os.path.join(config.PLOT_DIR, 'test'))
    
    # Print test results
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(test_metrics['classification_report']).transpose())
    
    # Save and display confusion matrix
    plot_confusion_matrix(test_metrics['confusion_matrix'], config.CLASSES, 
                          os.path.join(config.PLOT_DIR, 'confusion_matrix.png'))
    plt.show()  # Display the plot
    
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"All results and plots saved in {config.OUTPUT_DIR} directory")

def create_loaders():
    train_dataset = EnhancedFER2013(
        os.path.join(config.DATA_ROOT, 'train_labels.csv'),
        os.path.join(config.DATA_ROOT, 'train'),
        augment=True
    )
    
    val_dataset = EnhancedFER2013(
        os.path.join(config.DATA_ROOT, 'val_labels.csv'),
        os.path.join(config.DATA_ROOT, 'val')
    )
    
    test_dataset = EnhancedFER2013(
        os.path.join(config.DATA_ROOT, 'test_labels.csv'),
        os.path.join(config.DATA_ROOT, 'test')
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def train_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return train_loss / len(loader.dataset), correct / total

def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    if return_predictions:
        return loss / len(loader.dataset), correct / total, all_labels, all_preds
    return loss / len(loader.dataset), correct / total

if __name__ == '__main__':
    main()