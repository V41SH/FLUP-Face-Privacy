##########################################################################################
# AGHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
# THIS DOESN'T WORK WELL RN
##########################################################################################


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from tqdm import tqdm
import cv2
from sklearn.manifold import TSNE

# InsightFace Stuff 
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from retinaface import RetinaFace

# Only adapted it to Vaishnav's dataloader rn
from lfw_dataloader import get_lfw_dataloaders

class InsightFaceBaseline:
    """
    Baseline class for face recognition using InsightFace: RetinaFace and ArcFace
    """
    def __init__(self, det_size=(640, 640)):

        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        
        # Set detection size, higher for better accuracy but slower :')
        self.app.prepare(ctx_id=0, det_size=det_size)
        
        # mainly for storing embeddings
        self.reference_embeddings = {}
        
        print("InsightFace initialized successfully.")
    
    def get_face_embedding(self, img):
        """
        Extract face embedding from an image using InsightFace pipeline
        Returns face embedding and bounding box
        """
        # InsightFace expects BGR format like opencv

        if isinstance(img, torch.Tensor):
            # tensor to opencv format
            img = img.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            # RGB to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif isinstance(img, np.ndarray) and img.shape[2] == 3:
            if img.dtype == np.float32 and img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        
        # Detect faces :)
        faces = self.app.get(img)
        
        if len(faces) == 0:
            return None, None
        
        # Get the face with the highest detection score
        face = max(faces, key=lambda x: x.det_score)
        
        # Extract embedding(ArcFace feature)
        embedding = face.embedding
        
        # Get the bounding box
        bbox = face.bbox.astype(int)
        
        return embedding, bbox
    
    def build_reference_database(self, dataloader):
        """
        Build a reference database of face embeddings from a dataloader
        """
        self.reference_embeddings = {}
        skipped = 0
        total = 0
        
        print("Building reference database...")
        for images, labels, names in tqdm(dataloader):
            total += len(images)
            
            for i in range(len(images)):
                img = images[i]
                label = labels[i].item()
                name = names[i]
                
                # Get face embedding
                embedding, bbox = self.get_face_embedding(img)
                
                if embedding is None:
                    skipped += 1
                    continue
                
                # Store the embedding
                if label not in self.reference_embeddings:
                    self.reference_embeddings[label] = {
                        'name': name,
                        'embeddings': []
                    }
                
                self.reference_embeddings[label]['embeddings'].append(embedding)
        
        print(f"Reference database buildt. Processed {total} images, skipped {skipped}(no face detected).")
        return self.reference_embeddings
    
    def identify_face(self, img, threshold=0.1):
        """
        Identify a face by comparing with reference database embeddings
        """
        # Get the face embedding
        embedding, bbox = self.get_face_embedding(img)
        
        if embedding is None:
            return None, 0.0, None
        
        # Compare with reference database
        best_match = None
        best_similarity = -1
        
        for label, data in self.reference_embeddings.items():
            # Compare with all embeddings for this identity
            similarities = [self.cosine_similarity(embedding, ref_emb) 
                           for ref_emb in data['embeddings']]
            
            # Get max similarity
            max_similarity = max(similarities) if similarities else 0
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_match = {'label': label, 'name': data['name']}
        
        if best_similarity < threshold:
            return None, best_similarity, bbox
        
        return best_match, best_similarity, bbox
    
    def evaluate(self, dataloader, threshold=0.3):
        """
        Evaluate the recognition performance on a dataset
        """
        true_labels = []
        pred_labels = []
        confidences = []
        missed_faces = 0
        total_images = 0
        
        print("Evaluating recognition performance...")
        
        for images, labels, _ in tqdm(dataloader):
            total_images += len(images)
            
            for i in range(len(images)):
                img = images[i]
                true_label = labels[i].item()
                
                # Identify the face
                result, confidence, bbox = self.identify_face(img, threshold)
                
                if result is None:
                    missed_faces += 1
                    # Assign a special value for missed faces
                    pred_labels.append(-1)
                else:
                    pred_labels.append(result['label'])
                
                true_labels.append(true_label)
                confidences.append(confidence)
        
        # Calculate metrics(excluding the missed faces)
        valid_indices = [i for i, pred in enumerate(pred_labels) if pred != -1]
        
        # If no valid faces detected, return 0 for all metrics :(
        if not valid_indices:
            print("No valid faces detected in the evaluation set.")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'missed_ratio': 1.0
            }
        
        valid_true = [true_labels[i] for i in valid_indices]
        valid_pred = [pred_labels[i] for i in valid_indices]
        
        accuracy = accuracy_score(valid_true, valid_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            valid_true, valid_pred, average='weighted'
        )
        
        missed_ratio = missed_faces / total_images
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'missed_ratio': missed_ratio
        }
        
        print(f"Evaluation results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Missed face detection ratio: {missed_ratio:.4f}")
        
        return results
    
    @staticmethod
    def cosine_similarity(a, b):
        """
        Calculate cosine similarity between two vectors
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    #gpted it
    def visualize_results(self, dataloader, num_samples=5):
        """
        Visualize recognition results on sample images
        """
        # Get batch of images
        images, labels, names = next(iter(dataloader))
        
        # Only use a subset
        images = images[:num_samples]
        labels = labels[:num_samples]
        names = names[:num_samples]
        
        # Set up the figure
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
        
        # Unnormalize images
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        images = images * std + mean
        
        # Process each image
        for i, (img, true_label, true_name) in enumerate(zip(images, labels, names)):
            # Convert to numpy for visualization
            img_np = img.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            
            # Identify face
            result, confidence, bbox = self.identify_face(img)
            
            # Plot the image
            axes[i].imshow(img_np)
            
            # Add rectangle around face if detected
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     fill=False, edgecolor='green', linewidth=2)
                axes[i].add_patch(rect)
            
            # Add title with true and predicted name
            if result is None:
                title = f"True: {true_name}\nPred: No face"
            else:
                title = f"True: {true_name}\nPred: {result['name']}\nConf: {confidence:.2f}"
                
                # Change color based on correct/incorrect
                if result['label'] == true_label.item():
                    axes[i].set_title(title, color='green')
                else:
                    axes[i].set_title(title, color='red')
            
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("insightface_results.png")
        plt.close()
        
    def evaluate_blurring_impact(self, dataloader_original, dataloaders_blurred):
        """
        Evaluate the impact of different blur levels on recognition performance
        
        Args:
            dataloader_original: Dataloader with no burrr images
            dataloaders_blurred: Dict of dataloaders with different burrr levels
                like this: {'blur_3': dataloader_blur_3, 'blur_5': dataloader_blur_5}
        """
        # Evaluate on no burrr
        print("Evaluating on original images...")
        results_original = self.evaluate(dataloader_original)
        
        # Results for each burrr level
        all_results = {'original': results_original}
        
        # Evaluate on each burrr level
        for blur_name, dataloader in dataloaders_blurred.items():
            print(f"Evaluating on {blur_name}...")
            results = self.evaluate(dataloader)
            all_results[blur_name] = results
        
        # Visualize the results
        self.plot_blurring_results(all_results)
        
        return all_results
    
    #gpted it
    def plot_blurring_results(self, results):
        """
        Plot the impact of blurring on recognition metrics
        """
        blur_levels = list(results.keys())
        accuracy = [results[level]['accuracy'] for level in blur_levels]
        precision = [results[level]['precision'] for level in blur_levels]
        recall = [results[level]['recall'] for level in blur_levels]
        f1 = [results[level]['f1'] for level in blur_levels]
        
        plt.figure(figsize=(10, 6))
        plt.plot(blur_levels, accuracy, 'o-', label='Accuracy')
        plt.plot(blur_levels, precision, 's-', label='Precision')
        plt.plot(blur_levels, recall, '^-', label='Recall')
        plt.plot(blur_levels, f1, 'D-', label='F1 Score')
        
        plt.xlabel('Blur Level')
        plt.ylabel('Score')
        plt.title('Impact of Blurring on Face Recognition Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("blurring_impact.png")
        plt.close()


def main():
   
    data_dir = 'data/lfw/'
    batch_size = 32
    img_size = 224

    print("Creating dataloaders...")
    
    # No burrr
    train_loader_original, test_loader_original, num_classes = get_lfw_dataloaders(
        data_dir, batch_size=batch_size, img_size=img_size, blur_sigma=None
    )
    
    # Different burrrs
    blur_levels = [1, 3, 5, 7, 10]
    test_loaders_blurred = {}
    
    for blur in blur_levels:
        _, test_loader, _ = get_lfw_dataloaders(
            data_dir, batch_size=batch_size, img_size=img_size, blur_sigma=blur
        )
        test_loaders_blurred[f"blur_{blur}"] = test_loader
    
    print(f"Dataset loaded successfully with {num_classes} unique lads and lasses")
    
    # Initialize InsightFace baseline
    print("Initializing InsightFace baseline...")
    face_recognizer = InsightFaceBaseline(det_size=(640, 640))
    
    # Reference database with no burrr images
    face_recognizer.build_reference_database(train_loader_original)
    
    face_recognizer.visualize_results(test_loader_original)
    
    # Evaluate the impact of burrring
    results = face_recognizer.evaluate_blurring_impact(
        test_loader_original, test_loaders_blurred
    )
    
    print("Evaluation complete.")
    print(results)


if __name__ == "__main__":
    main()