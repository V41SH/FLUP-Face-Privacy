import argparse
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# InsightFace imports
import insightface
from insightface.app import FaceAnalysis

# Import the dataloader
from lfw_double_loader import LFWDatasetDouble
from lfw_double_loader import get_lfw_dataloaders as get_double
from lfw_triple_loaders import get_transforms
from utils.blurring_utils import blur_face


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

"""
Example Usage:
python arcface_finfin.py --root_dir data/lfw --num_pairs 100 --report_interval 25
"""

class ModelWrapper:
    def __init__(self, sharpnet_path, blurnet_path, model_name):
        self.sharpnet = torch.load(sharpnet_path, weights_only=False).to(device)
        self.blurnet = torch.load(blurnet_path, weights_only=False).to(device)
        self.model_name = model_name
        
        self.sharpnet.eval()
        self.blurnet.eval()
        
    def get_embedding(self, img, model_type='sharp'):
        """Extract face embedding from an image"""
        with torch.no_grad():
            img = self.test_transform(img).to(device).unsqueeze(0)
            
            if model_type == 'sharp':
                embedding = self.sharpnet(img)
            else:
                embedding = self.blurnet(img)
                
        embedding.requires_grad = False
        return embedding
    
    def set_transform(self, transform):
        self.test_transform = transform

class FaceVerifier:
    def __init__(self, model_configs, det_size=(640, 640)):
        self.models = []
        self.train_transform, self.test_transform = get_transforms(img_size=224)
        
        for config in model_configs:
            model = ModelWrapper(
                sharpnet_path=config['sharpnet_path'],
                blurnet_path=config['blurnet_path'],
                model_name=config['name']
            )
            model.set_transform(self.test_transform)
            self.models.append(model)
        
        print(f"Loaded {len(self.models)} models")
    
    def compare_faces(self, img1, img2, blur_sigma=None, is_same_person=True):
        """Compare two face images and return similarity scores for all models"""
        if blur_sigma is not None and blur_sigma > 0:
            img1 = blur_face(img1, blur_type='gaussian', blur_amount=blur_sigma)
        
        similarities = {}
        for model in self.models:
            # Get embeddings
            embedding1 = model.get_embedding(img1, model_type="blur")
            embedding2 = model.get_embedding(img2, model_type="sharp")
            
            if embedding1 is None or embedding2 is None:
                similarities[model.model_name] = None
                continue
            
            # Calculate similarity
            similarity = self.cosine_similarity(embedding1, embedding2)
            
            if not is_same_person:
                similarity = 1.0 - similarity
                
            similarities[model.model_name] = similarity
        
        return similarities

    def evaluate_blur_effects(self, same_person_dataset, diff_person_dataset, num_pairs=100, 
                            blur_levels=[0, 1, 3, 5, 7, 10], report_interval=10, 
                            save_path="arcface_eval_results"):
        """
        Evaluate blur effects on multiple image pairs from the dataset
        
        Args:
            num_pairs: Total number of image pairs to evaluate (50% same, 50% different)
            blur_levels: List of blur sigma values to test
            report_interval: Interval at which to save intermediate results
            save_path: Directory to save results
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize results structure
        all_results = {
            model.model_name: {
                blur: {'same': [], 'diff': [], 'all': []} 
                for blur in blur_levels
            } 
            for model in self.models
        }
        
        intermediate_results = []
        
        num_same_pairs = num_pairs // 2
        num_diff_pairs = num_pairs - num_same_pairs
        
        # Randomly select indices for evaluation
        same_indices = np.random.choice(len(same_person_dataset), num_same_pairs, replace=False)
        diff_indices = np.random.choice(len(diff_person_dataset), num_diff_pairs, replace=False)
        
        # Evaluate same-person pairs
        for i, idx in enumerate(tqdm(same_indices, desc="Evaluating same-person pairs")):
            img1, img2, name1, name2 = same_person_dataset[idx]
            
            if name1 != name2:
                continue
                
            for blur in blur_levels:
                blur_sigma = blur if blur > 0 else None
                similarities = self.compare_faces(img1, img2, blur_sigma, is_same_person=True)
                
                for model_name, similarity in similarities.items():
                    if similarity is not None:
                        all_results[model_name][blur]['same'].append(similarity)
                        all_results[model_name][blur]['all'].append(similarity)
            
            # Save intermediate results at intervals
            if (i + 1) % report_interval == 0:
                self._save_intermediate_results(all_results, blur_levels, i + 1, save_path)
                intermediate_results.append((i + 1, self._get_averages(all_results)))
        
        # Evaluate different-person pairs
        total_processed = num_same_pairs
        for i, idx in enumerate(tqdm(diff_indices, desc="Evaluating different-person pairs")):
            img1, img2, name1, name2 = diff_person_dataset[idx]
            
            if name1 == name2:
                continue
                
            for blur in blur_levels:
                blur_sigma = blur if blur > 0 else None
                similarities = self.compare_faces(img1, img2, blur_sigma, is_same_person=False)
                
                for model_name, similarity in similarities.items():
                    if similarity is not None:
                        all_results[model_name][blur]['diff'].append(similarity)
                        all_results[model_name][blur]['all'].append(similarity)
            
            # Save intermediate results at intervals
            current_count = total_processed + i + 1
            if current_count % report_interval == 0 or current_count == num_pairs:
                self._save_intermediate_results(all_results, blur_levels, current_count, save_path)
                intermediate_results.append((current_count, self._get_averages(all_results)))
        
        self._save_final_results(all_results, blur_levels, save_path)
        
        return all_results, intermediate_results
    
    def _get_averages(self, results):
        """Calculate averages for all models and categories"""
        averages = {}
        for model_name, model_results in results.items():
            averages[model_name] = {
                blur: {
                    cat: np.mean(scores) if scores else 0 
                    for cat, scores in categories.items()
                }
                for blur, categories in model_results.items()
            }
        return averages
    
    def _save_intermediate_results(self, results, blur_levels, num_processed, save_path):
        """Save intermediate results and plots"""
        averages = self._get_averages(results)
        
        # Save numerical results
        with open(os.path.join(save_path, f"intermediate_{num_processed}.txt"), "w") as f:
            f.write(f"Results after {num_processed} pairs:\n")
            for model_name, model_averages in averages.items():
                f.write(f"\nModel: {model_name}\n")
                for blur in blur_levels:
                    f.write(f"Blur {blur}:\n")
                    f.write(f"  Same Person:     Mean similarity = {model_averages[blur]['same']:.4f}\n")
                    f.write(f"  Different Person: Mean 1-similarity = {model_averages[blur]['diff']:.4f}\n")
                    f.write(f"  All Pairs:        Mean metric = {model_averages[blur]['all']:.4f}\n")
        
        # Plot current results - Same Person
        self._plot_results(averages, blur_levels, num_processed, save_path, 'same', f'Same Person Verification ({num_processed} pairs)')
        
        # Plot current results - Different Person
        self._plot_results(averages, blur_levels, num_processed, save_path, 'diff', f'Different Person Verification ({num_processed} pairs)')
    
    def _save_final_results(self, results, blur_levels, save_path):
        """Save final results and plots"""
        averages = self._get_averages(results)
        
        # Save numerical results
        with open(os.path.join(save_path, "final_results.txt"), "w") as f:
            f.write("Final Results:\n")
            for model_name, model_averages in averages.items():
                f.write(f"\nModel: {model_name}\n")
                for blur in blur_levels:
                    f.write(f"Blur {blur}:\n")
                    f.write(f"  Same Person:     Mean similarity = {model_averages[blur]['same']:.4f}\n")
                    f.write(f"  Different Person: Mean 1-similarity = {model_averages[blur]['diff']:.4f}\n")
                    f.write(f"  All Pairs:        Mean metric = {model_averages[blur]['all']:.4f}\n")
        
        # Plot final results - Same Person
        self._plot_results(averages, blur_levels, "final", save_path, 'same', 'Same Person Verification')
        
        # Plot final results - Different Person
        self._plot_results(averages, blur_levels, "final", save_path, 'diff', 'Different Person Verification')
    
    def _plot_results(self, averages, blur_levels, num_processed, save_path, category, title):
        """Helper function to plot results for a specific category"""
        plt.figure(figsize=(10, 6))
        
        for model_name, model_averages in averages.items():
            plt.plot(
                blur_levels, 
                [model_averages[blur][category] for blur in blur_levels], 
                marker='o', 
                label=model_name
            )
        
        plt.xlabel('Blur Sigma')
        plt.ylabel('Cosine Similarity' if category == 'same' else '1 - Cosine Similarity')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        
        filename = f"{category}_plot_{num_processed}.png"
        plt.savefig(os.path.join(save_path, filename))
        plt.close()
    
    @staticmethod
    def cosine_similarity(a, b):
        a = a.squeeze(0)
        b = b.squeeze(0)
        """Calculate cosine similarity between two vectors"""
        result = torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))
        return result.cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate blur effects on face verification')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--num_pairs', type=int, default=100, help='Number of image pairs to evaluate')
    parser.add_argument('--report_interval', type=int, default=10, help='Interval at which to save intermediate results')
    parser.add_argument('--blur_levels', type=int, nargs='+', default=25, help='Blur levels to evaluate')
    parser.add_argument('--det_size', type=int, default=640, help='Detection size for InsightFace')
    parser.add_argument('--save_path', type=str, default="blur_evaluation_results", help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='lfw', choices=['lfw', 'celeba'], help='Which dataset to use: lfw or celeba')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define your models here
    model_configs = [
        {
            'name': 'Experiment 4',
            'sharpnet_path': "/home/salonisaxena/work/Q3/CV/FLUP-Face-Privacy/sharpnet-1-10-49.pt",
            'blurnet_path': "/home/salonisaxena/work/Q3/CV/FLUP-Face-Privacy/blurnet-1-10-49.pt"
        },
        {
            'name': 'Experiment 2', 
            'sharpnet_path': "/home/salonisaxena/work/Q3/CV/FLUP-Face-Privacy/sharpnet-31-21-0(1).pt",
            'blurnet_path': "/home/salonisaxena/work/Q3/CV/FLUP-Face-Privacy/blurnet-31-21-0(1).pt"
        },
    ]
    
    # Initialize face verifier with multiple models
    face_verifier = FaceVerifier(model_configs, det_size=(args.det_size, args.det_size))
    
    # Create dataset with same_person=True
    if args.dataset == "lfw":
        same_person_dataset = LFWDatasetDouble(
            root_dir=args.root_dir,
            transform=None, 
            train=False,     
            same_person=True # Pairs of the same person
        )
        
        diff_person_dataset = LFWDatasetDouble(
            root_dir=args.root_dir,
            transform=None,  
            train=False,     
            same_person=False #different pairs
        )
    elif args.dataset == "celeba": 
        #create celeba dataset here
        pass
    
    args.num_pairs = min(len(same_person_dataset), len(diff_person_dataset))

    print(f"Evaluating blur effects on {args.num_pairs} image pairs...")
    
    if isinstance(args.blur_levels, int):
        args.blur_levels = list(range(0, args.blur_levels+1, 2))

    # Run evaluation
    all_results, intermediate_results = face_verifier.evaluate_blur_effects(
        same_person_dataset=same_person_dataset,
        diff_person_dataset=diff_person_dataset,
        num_pairs=args.num_pairs,
        blur_levels=args.blur_levels,
        report_interval=args.report_interval,
        save_path=args.save_path
    )
    
    print("\nFinal Results:")
    for model_name, model_results in all_results.items():
        print(f"\nModel: {model_name}")
        for blur, categories in model_results.items():
            same_mean = np.mean(categories['same']) if categories['same'] else 0
            diff_mean = np.mean(categories['diff']) if categories['diff'] else 0
            all_mean = np.mean(categories['all']) if categories['all'] else 0
            
            print(f"Blur {blur}:")
            print(f"  Same Person:      Average similarity = {same_mean:.4f}")
            print(f"  Different Person: Average 1-similarity = {diff_mean:.4f}")
            print(f"  All Pairs:        Average metric = {all_mean:.4f}")
    
    print(f"\nResults saved to {args.save_path}")

if __name__ == "__main__":
    main()