
# Stupid first attempt at a model idk what I was thinking
# Probably not using this

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from pytorch_metric_learning import losses, miners, distances
from sklearn.manifold import TSNE

from lfw_dataloader import get_lfw_dataloaders

# Here we build a fancy Face Recognition model
class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes, embedding_size=512, pretrained=True):
        super(FaceRecognitionModel, self).__init__()
        # Use a pre-trained ResNet as the backbone because why reinvent the wheel
        backbone = models.resnet50(pretrained=pretrained)
        self.features = nn.Sequential(*list(backbone.children())[:-1]) 
        
        # Feature embedding layer to turn features into face embeddings
        self.embedding = nn.Linear(backbone.fc.in_features, embedding_size)
        
        self.classifier = nn.Linear(embedding_size, num_classes)
        
    def forward(self, x):
        x = self.features(x)  # Extract the features
        x = x.view(x.size(0), -1)  # Flatten it
        embedding = self.embedding(x)  # Make it into a low-dimensional vector

        # Normalize embeddings to make them extra sharp
        embedding_normalized = nn.functional.normalize(embedding, p=2, dim=1)
        
        logits = self.classifier(embedding)
        
        return embedding_normalized, logits

# Train time, slayyy
def train_epoch(model, dataloader, arcface_loss, classifier_loss, optimizer, device, mining_func=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels, _ in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)  # Put em on the device (Gabriel's fresh GPU)
        
        optimizer.zero_grad() 
        
        embeddings, logits = model(images)  
        
        # Calculate the loss (we're adding ArcFace and the cross-entropy loss here)
        loss = 0
        
        if mining_func:
            hard_pairs = mining_func(embeddings, labels)
            arc_loss = arcface_loss(embeddings, labels, hard_pairs)
        else:
            arc_loss = arcface_loss(embeddings, labels)
        
        # Add classifier loss for stability
        cls_loss = classifier_loss(logits, labels)
        
        loss = arc_loss + cls_loss
        
        loss.backward() 
        optimizer.step() 
        
        running_loss += loss.item()
        
        _, predicted = logits.max(1) 
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


# Evaluation
def evaluate(model, dataloader, arcface_loss, classifier_loss, device, mining_func=None):
    model.eval() 
    running_loss = 0.0
    
    all_labels = []
    all_preds = []
    all_embeddings = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(dataloader): 
            images, labels = images.to(device), labels.to(device)
            
            embeddings, logits = model(images)
            
            # Store embeddings for later analysis
            all_embeddings.append(embeddings.cpu())
            
            loss = 0
            if mining_func:
                hard_pairs = mining_func(embeddings, labels)
                arc_loss = arcface_loss(embeddings, labels, hard_pairs)
            else:
                arc_loss = arcface_loss(embeddings, labels)
            
            cls_loss = classifier_loss(logits, labels)
            loss = arc_loss + cls_loss
            
            running_loss += loss.item()
            
            _, predicted = logits.max(1) 
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    test_loss = running_loss / len(dataloader)
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Concatenate all embeddings for future t-SNE goodness
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    
    return test_loss, accuracy, precision, recall, f1, all_embeddings, all_labels


def visualize_embeddings(embeddings, labels, num_classes=10):
    unique_classes = np.unique(labels)[:num_classes] # Get a subset of classes for visualization
    mask = np.isin(labels, unique_classes)
    
    filtered_embeddings = embeddings[mask]
    filtered_labels = np.array(labels)[mask]
    
    # Apply t-SNE to project those embeddings into 2D space
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(filtered_embeddings)
    
    plt.figure(figsize=(10, 8))
    
    for i, cls in enumerate(unique_classes):
        indices = filtered_labels == cls
        plt.scatter(
            embeddings_2d[indices, 0],
            embeddings_2d[indices, 1],
            label=f"Class {cls}",
            alpha=0.7
        )
    
    plt.legend()
    plt.title("t-SNE visualization of face embeddings")
    plt.savefig("face_embeddings_tsne.png")
    plt.close()

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png") 
    plt.close()

def main():
    data_dir = 'data/lfw/'
    batch_size = 32
    img_size = 224
    num_epochs = 20
    #from the paper: reduce the learning rate at 8, 14 epochs and terminate at 18 epochs
    #but I'm going to use a simpler one for now
    lr = 1e-4
    embedding_size = 512
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or bust 💻
    print(f"Using device: {device}")
    
    # Load data
    #There are some issues with the blur sigma, so I'm commenting it out for now
    train_loader, test_loader, num_classes = get_lfw_dataloaders(
        data_dir,
        batch_size=batch_size,
        img_size=img_size,
        # blur_sigma= 5.0  # Change for burrr or non burrr
    )
    
    print(f"Dataset loaded successfully with {num_classes} unique individuals")
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    model = FaceRecognitionModel(
        num_classes=num_classes,
        embedding_size=embedding_size,
        pretrained=True
    ).to(device)
    
    # THE ARCFACE LOSS
    arcface_loss = losses.ArcFaceLoss(
        num_classes=num_classes,
        embedding_size=embedding_size,
        margin=28.6,
        scale=64
    ).to(device)
    
    classifier_loss = nn.CrossEntropyLoss()  # The old reliable
    mining_func = miners.MultiSimilarityMiner(epsilon=0.1) 
    
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': arcface_loss.parameters()}
    ], lr=lr)
    
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, 
            train_loader, 
            arcface_loss, 
            classifier_loss, 
            optimizer, 
            device,
            mining_func
        )
        
        # Evaluate
        val_loss, val_acc, val_precision, val_recall, val_f1, embeddings, labels = evaluate(
            model, 
            test_loader, 
            arcface_loss, 
            classifier_loss, 
            device,
            mining_func
        )
        
        # Update scheduler
        scheduler.step()
        
        # Store history for posterity
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    
    plot_training_history(history)
    
    visualize_embeddings(embeddings, labels)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'arcface_state_dict': arcface_loss.state_dict(),
        'num_classes': num_classes,
        'embedding_size': embedding_size
    }, "face_recognition_model.pth")
    
    print("model savedddd")


if __name__ == "__main__":
    main()
