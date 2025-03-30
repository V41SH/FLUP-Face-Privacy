import argparse
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

# InsightFace imports
import insightface
from insightface.app import FaceAnalysis
from utils.blurring_utils import blur_face

# Import the dataloader
from lfw_double_loader import LFWDatasetDouble, get_lfw_dataloaders
from celebA_dataloader.dataset import *

# from utils.blurring_utils import *
"""
Example Usage:
python arcface_finfin.py --root_dir data/lfw --num_pairs 100 --report_interval 25
"""

class FaceVerifier:
    def __init__(self, det_size=(640, 640)):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=det_size)
        print("InsightFace initialized successfully.")
    
    def get_face_embedding(self, img):
        """Extract face embedding from an image"""
        # Convert to correct format(opencv style)
        if isinstance(img, Image.Image):
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Detect faces (it uses retinaface for face detection)
        faces = self.app.get(img)
        
        if len(faces) == 0:
            return None
        
        # Get the face with the highest detection score
        face = max(faces, key=lambda x: x.det_score)
        
        # Extract embedding
        embedding = face.embedding
        bbox = face.bbox.astype(int)
        
        return embedding, bbox, face.det_score
    
    def blur_face_region(self, img, bbox, sigma=3):
        """Blur a face region in an image"""
        # Make a copy to avoid modifying the original
        result = img.copy()
        
        # Extract face coordinates
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Extract the face region
        face_region = result[y1:y2, x1:x2]
        print(face_region)
        
        # Apply Gaussian blur
        blurred_face = cv2.GaussianBlur(face_region, (0, 0), sigma)
        
        # Replace the region in the original image
        result[y1:y2, x1:x2] = blurred_face
        
        return result

    def compare_faces(self, img1, img2, blur_type=None, blur_amount=None, is_same_person=True, **kwargs):
        """Compare two face images and return similarity score"""
        # Apply blur if specified
        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
            img1 = np.transpose(img1, (1, 2, 0))
            img1 = (img1 * 255).clip(0, 255).astype(np.uint8)
            img1 = Image.fromarray(np.array(img1), "RGB")
            # img1 = Image.fromarray(np.array(img1), "RGB")
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
            img2 = np.transpose(img2, (1, 2, 0))
            img2 = (img2 * 255).clip(0, 255).astype(np.uint8)
            img2 = Image.fromarray(np.array(img2), "RGB")

        if blur_type is not None and blur_amount is not None and blur_amount > 0:
            if blur_type == 'pixelation':
                kwargs['pixel_size'] = blur_amount  
            img1 = blur_face(img1, blur_type=blur_type, blur_amount=blur_amount, **kwargs)
        
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        cv2.imwrite("img1.png", img1)
        cv2.imwrite("img2.png", img2)
        # Get embeddings
        result1 = self.get_face_embedding(img1)
        result2 = self.get_face_embedding(img2)
        
        if result1 is None or result2 is None:
            return None
        
        embedding1, _, _ = result1
        embedding2, _, _ = result2
        
        # Calculate similarity
        similarity = self.cosine_similarity(embedding1, embedding2)
        
        # For different-person pairs, use 1-cosine_similarity as the metric
        if not is_same_person:
            similarity = 1.0 - similarity
        
        return similarity

    def evaluate_blur_effects(self, same_person_dataset, diff_person_dataset, num_pairs=100, 
                            # blur_levels=[0, 1, 3, 5, 7, 10], 
                            blur_levels=[0, 1, 3, 5, 7, 10],
                            report_interval=10, save_path="arcface_eval_results",
                            blur_type='gaussian', **kwargs):
        """
        Evaluate blur effects on multiple image pairs from the dataset
        
        Args:
            num_pairs: Total number of image pairs to evaluate (50% same, 50% different)
            blur_levels: List of blur amount values to test
            report_interval: Interval at which to save intermediate results
            save_path: Directory to save results
            blur_type: Type of blur to apply ('gaussian', 'black', 'pixelation')
            kwargs: Additional parameters for specific blur types
        """
        os.makedirs(save_path, exist_ok=True)
        
        all_results = {blur: {'same': [], 'diff': [], 'all': []} for blur in blur_levels}
        intermediate_results = []
        
        num_same_pairs = num_pairs // 2
        num_diff_pairs = num_pairs - num_same_pairs
        
        # Randomly select indices for evaluation
        same_indices = np.random.choice(len(same_person_dataset), num_same_pairs, replace=False)
        diff_indices = np.random.choice(len(diff_person_dataset), num_diff_pairs, replace=False)
        
        for i, idx in enumerate(tqdm(same_indices, desc="Evaluating same-person pairs")):
            img1, img2, name1, name2 = same_person_dataset[idx]
            
            if name1 != name2:
                continue
                
            for blur in blur_levels:
                blur_amount = blur if blur > 0 else None
                similarity = self.compare_faces(
                    img1, img2, 
                    blur_type=blur_type if blur_amount else None,
                    blur_amount=blur_amount,
                    is_same_person=True,
                    **kwargs
                )
                
                if similarity is not None:
                    all_results[blur]['same'].append(similarity)
                    all_results[blur]['all'].append(similarity)
            
            # Save intermediate results at intervals
            if (i + 1) % report_interval == 0:
                self._save_intermediate_results(all_results, blur_levels, i + 1, save_path, blur_type)
                intermediate_results.append((i + 1, {k: {cat: np.mean(scores) for cat, scores in v.items() if scores} 
                                                for k, v in all_results.items()}))
        
        # different-person pairs
        total_processed = num_same_pairs
        for i, idx in enumerate(tqdm(diff_indices, desc="Evaluating different-person pairs")):
            img1, img2, name1, name2 = diff_person_dataset[idx]
            
            if name1 == name2:
                continue
                
            for blur in blur_levels:
                blur_amount = blur if blur > 0 else None
                similarity = self.compare_faces(
                    img1, img2, 
                    blur_type=blur_type if blur_amount else None,
                    blur_amount=blur_amount,
                    is_same_person=False,
                    **kwargs
                )
                
                if similarity is not None:
                    all_results[blur]['diff'].append(similarity)
                    all_results[blur]['all'].append(similarity)
            
            # Save intermediate results at intervals
            current_count = total_processed + i + 1
            if current_count % report_interval == 0 or current_count == num_pairs:
                self._save_intermediate_results(all_results, blur_levels, current_count, save_path, blur_type)
                intermediate_results.append((current_count, {k: {cat: np.mean(scores) for cat, scores in v.items() if scores} 
                                                for k, v in all_results.items()}))
        
        self._save_final_results(all_results, blur_levels, save_path, blur_type)
        
        return all_results, intermediate_results
    
    def _save_intermediate_results(self, results, blur_levels, num_processed, save_path, blur_type):
        """Save intermediate results and plots"""
        # Calculate current averages
        averages = {blur: {cat: np.mean(scores) if scores else 0 
                        for cat, scores in categories.items()}
                for blur, categories in results.items()}
        
        # Save numerical results
        with open(os.path.join(save_path, f"intermediate_{blur_type}_{num_processed}.txt"), "w") as f:
            f.write(f"Results after {num_processed} pairs ({blur_type} blur):\n")
            for blur in blur_levels:
                f.write(f"Blur {blur}:\n")
                f.write(f"  Same Person:     Mean similarity = {averages[blur]['same']:.4f} (n={len(results[blur]['same'])})\n")
                f.write(f"  Different Person: Mean 1-similarity = {averages[blur]['diff']:.4f} (n={len(results[blur]['diff'])})\n")
                f.write(f"  All Pairs:        Mean metric = {averages[blur]['all']:.4f} (n={len(results[blur]['all'])})\n")
        
        # Plot current results (update plot titles to include blur type)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(blur_levels, [averages[blur]['same'] for blur in blur_levels], marker='o', label='Same Person')
        plt.plot(blur_levels, [averages[blur]['diff'] for blur in blur_levels], marker='s', label='Different Person')
        plt.plot(blur_levels, [averages[blur]['all'] for blur in blur_levels], marker='*', label='All Pairs')
        plt.xlabel('Blur Level')
        plt.ylabel('Metric Value')
        plt.title(f'{blur_type.capitalize()} Blur: Face Verification Metrics\n({num_processed} pairs processed)')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(blur_levels, [averages[blur]['all'] for blur in blur_levels], marker='o', color='green')
        plt.xlabel('Blur Level')
        plt.ylabel('Average Metric (All Pairs)')
        plt.title(f'{blur_type.capitalize()} Blur: Overall Average Metric')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'intermediate_plot_{blur_type}_{num_processed}.png'))
        plt.close()

    def _save_final_results(self, results, blur_levels, save_path, blur_type):
        """Save final results and plots for a specific blur type"""
        # Calculate final averages
        averages = {blur: {cat: np.mean(scores) if scores else 0 
                        for cat, scores in categories.items()}
                for blur, categories in results.items()}
        
        # Save numerical results
        with open(os.path.join(save_path, f"final_results_{blur_type}.txt"), "w") as f:
            f.write(f"Final Results ({blur_type} blur):\n")
            for blur in blur_levels:
                f.write(f"Blur {blur}:\n")
                f.write(f"  Same Person:      Mean similarity = {averages[blur]['same']:.4f} (n={len(results[blur]['same'])})\n")
                f.write(f"  Different Person: Mean 1-similarity = {averages[blur]['diff']:.4f} (n={len(results[blur]['diff'])})\n")
                f.write(f"  All Pairs:        Mean metric = {averages[blur]['all']:.4f} (n={len(results[blur]['all'])})\n")
        
        # Plot final results
        plt.figure(figsize=(12, 8))
        
        # Plot all categories
        plt.subplot(2, 1, 1)
        plt.plot(blur_levels, [averages[blur]['same'] for blur in blur_levels], 
                marker='o', label='Same Person')
        plt.plot(blur_levels, [averages[blur]['diff'] for blur in blur_levels], 
                marker='s', label='Different Person')
        plt.plot(blur_levels, [averages[blur]['all'] for blur in blur_levels], 
                marker='*', label='All Pairs')
        plt.xlabel('Blur Level')
        plt.ylabel('Metric Value')
        plt.title(f'{blur_type.capitalize()} Blur: Final Face Verification Metrics')
        plt.legend()
        plt.grid(True)
        
        # Plot only overall average
        plt.subplot(2, 1, 2)
        plt.plot(blur_levels, [averages[blur]['all'] for blur in blur_levels], 
                marker='o', color='green')
        plt.xlabel('Blur Level')
        plt.ylabel('Average Metric (All Pairs)')
        plt.title(f'{blur_type.capitalize()} Blur: Overall Average Metric')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'final_plot_{blur_type}.png'))
        plt.close()
        
        # Save raw data as numpy file for potential later analysis
        np.savez(
            os.path.join(save_path, f'raw_results_{blur_type}.npz'),
            results=results,
            blur_levels=blur_levels,
            blur_type=blur_type
        )
    @staticmethod
    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate blur effects on face verification')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--num_pairs', type=int, default=100, help='Number of image pairs to evaluate')
    parser.add_argument('--report_interval', type=int, default=10, help='Interval at which to save intermediate results')
    parser.add_argument('--blur_levels', type=int, nargs='+', default=[0, 1, 3, 5, 7, 10], 
                       help='Blur levels to evaluate (sigma for Gaussian, pixel size for pixelation)')
    # parser.add_argument('--blur_levels', type=int, nargs='+', default=[10, 15, 20, 25, 30, 35, 40],
    #                       help='Blur levels to evaluate (sigma for Gaussian, pixel size for pixelation)')
    parser.add_argument('--det_size', type=int, default=640, help='Detection size for InsightFace')
    parser.add_argument('--save_path', type=str, default="arcface_eval_results", help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='lfw', choices=['lfw', 'celeba'], help='Which dataset to use: lfw or celeba')
    parser.add_argument('--blur_type', type=str, default='gaussian', 
                       choices=['gaussian', 'black', 'pixelation'], 
                       help='Type of blur to apply: gaussian, black, or pixelation')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize face verifier
    face_verifier = FaceVerifier(det_size=(args.det_size, args.det_size))
    
    # Create datasets
    if args.dataset == "lfw":
        same_person_dataset = LFWDatasetDouble(
            root_dir=args.root_dir,
            transform=None, 
            train=False,     
            same_person=True
        )
        
        diff_person_dataset = LFWDatasetDouble(
            root_dir=args.root_dir,
            transform=None,  
            train=False,     
            same_person=False
        )
    elif args.dataset == "celeba": 
        tform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


        same_person_dataset = CelebADataset(
            # root_dir=args.root_dir,
            triplet=True,
            transform=tform,
            seed=42,
            blur_sigma=0,
            same_person=True,
            blur_both=None,
            anchor_blur=None
        )

        diff_person_dataset = CelebADataset(
            # root_dir=args.root_dir,
            triplet=True,
            transform=tform,
            seed=42,
            blur_sigma=0,
            same_person=False,
            blur_both=None,
            anchor_blur=None
        )
    
    print(f"Evaluating {args.blur_type} blur effects on {args.num_pairs} image pairs (50% same, 50% different)...")
    
    # For black blur, we only need to evaluate once (amount doesn't matter)
    if args.blur_type == 'black':
        blur_levels = [0, 1]  # 0=no blur, 1=black blur
    else:
        blur_levels = args.blur_levels
    
    # Additional kwargs for specific blur types
    kwargs = {}
    if args.blur_type == 'pixelation':
        kwargs['pixel_size'] = None  # Will use blur_levels directly
    
    # Run evaluation
    all_results, intermediate_results = face_verifier.evaluate_blur_effects(
        same_person_dataset=same_person_dataset,
        diff_person_dataset=diff_person_dataset,
        num_pairs=args.num_pairs,
        blur_levels=blur_levels,
        report_interval=args.report_interval,
        save_path=args.save_path,
        blur_type=args.blur_type,
        **kwargs
    )
    
    print("\nFinal Results:")
    for blur in blur_levels:
        same_mean = np.mean(all_results[blur]['same']) if all_results[blur]['same'] else 0
        diff_mean = np.mean(all_results[blur]['diff']) if all_results[blur]['diff'] else 0
        all_mean = np.mean(all_results[blur]['all']) if all_results[blur]['all'] else 0
        
        print(f"Blur {blur}:")
        print(f"  Same Person:      Average similarity = {same_mean:.4f} (n={len(all_results[blur]['same'])})")
        print(f"  Different Person: Average 1-similarity = {diff_mean:.4f} (n={len(all_results[blur]['diff'])})")
        print(f"  All Pairs:        Average metric = {all_mean:.4f} (n={len(all_results[blur]['all'])})")
    
    print(f"\nResults saved to {args.save_path}")

if __name__ == "__main__":
    main()