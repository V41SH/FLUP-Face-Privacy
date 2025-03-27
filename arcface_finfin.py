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
from lfw_double_loader import LFWDatasetDouble, get_lfw_dataloaders

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
        
        # Apply Gaussian blur
        blurred_face = cv2.GaussianBlur(face_region, (0, 0), sigma)
        
        # Replace the region in the original image
        result[y1:y2, x1:x2] = blurred_face
        
        return result

    def compare_faces(self, img1, img2, blur_sigma=None):
        """Compare two face images and return similarity score"""
        # Convert PIL images to numpy arrays if needed
        if isinstance(img1, Image.Image):
            img1 = np.array(img1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        if isinstance(img2, Image.Image):
            img2 = np.array(img2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        # Apply blur if specified
        if blur_sigma is not None and blur_sigma > 0:
            # Detect face first
            faces1 = self.app.get(img1)
            faces2 = self.app.get(img2)
            
            if len(faces1) > 0:
                face1 = max(faces1, key=lambda x: x.det_score)
                img1 = self.blur_face_region(img1, face1.bbox, blur_sigma)
            if len(faces2) > 0:
                face2 = max(faces2, key=lambda x: x.det_score)
                img2 = self.blur_face_region(img2, face2.bbox, blur_sigma)
        
        # Get embeddings
        result1 = self.get_face_embedding(img1)
        result2 = self.get_face_embedding(img2)
        
        if result1 is None or result2 is None:
            return None
        
        embedding1, _, _ = result1
        embedding2, _, _ = result2
        
        # Calculate similarity
        similarity = self.cosine_similarity(embedding1, embedding2)
        
        return similarity

    def evaluate_blur_effects(self, dataset, num_pairs=100, blur_levels=[0, 1, 3, 5, 7, 10], 
                            report_interval=10, save_path="blur_evaluation_results"):
        """
        Evaluate blur effects on multiple image pairs from the dataset
        
        Args:
            dataset: LFWDatasetDouble instance
            num_pairs: Number of image pairs to evaluate
            blur_levels: List of blur sigma values to test
            report_interval: Interval at which to save intermediate results
            save_path: Directory to save results
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Initialize results storage
        all_results = {blur: [] for blur in blur_levels}
        intermediate_results = []
        
        # Randomly select indices for evaluation
        indices = np.random.choice(len(dataset), num_pairs, replace=False)
        
        for i, idx in enumerate(tqdm(indices, desc="Evaluating pairs")):
            img1, img2, name1, name2 = dataset[idx]
            
            # Skip if the names don't match (shouldn't happen with same_person=True)
            if name1 != name2:
                continue
                
            pair_results = {}
            
            for blur in blur_levels:
                blur_sigma = blur if blur > 0 else None
                similarity = self.compare_faces(img1, img2, blur_sigma)
                
                if similarity is not None:
                    pair_results[blur] = similarity
                    all_results[blur].append(similarity)
            
            # Save intermediate results at intervals
            if (i + 1) % report_interval == 0 or (i + 1) == num_pairs:
                self._save_intermediate_results(all_results, blur_levels, i + 1, save_path)
                intermediate_results.append((i + 1, {k: np.mean(v) for k, v in all_results.items()}))
        
        # Save final results
        self._save_final_results(all_results, blur_levels, save_path)
        
        return all_results, intermediate_results
    
    def _save_intermediate_results(self, results, blur_levels, num_processed, save_path):
        """Save intermediate results and plots"""
        # Calculate current averages
        averages = {blur: np.mean(scores) if scores else 0 for blur, scores in results.items()}
        
        # Save numerical results
        with open(os.path.join(save_path, f"intermediate_{num_processed}.txt"), "w") as f:
            f.write(f"Results after {num_processed} pairs:\n")
            for blur in blur_levels:
                f.write(f"Blur {blur}: Mean similarity = {averages[blur]:.4f} (n={len(results[blur])})\n")
        
        # Plot current results
        plt.figure(figsize=(10, 5))
        plt.plot(blur_levels, [averages[blur] for blur in blur_levels], marker='o')
        plt.xlabel('Blur Sigma')
        plt.ylabel('Average Cosine Similarity')
        plt.title(f'Face Verification Similarity vs. Blur Level\n({num_processed} pairs processed)')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, f'intermediate_plot_{num_processed}.png'))
        plt.close()
    
    def _save_final_results(self, results, blur_levels, save_path):
        """Save final results and plots"""
        # Calculate final averages
        averages = {blur: np.mean(scores) if scores else 0 for blur, scores in results.items()}
        
        # Save numerical results
        with open(os.path.join(save_path, "final_results.txt"), "w") as f:
            f.write("Final Results:\n")
            for blur in blur_levels:
                f.write(f"Blur {blur}: Mean similarity = {averages[blur]:.4f} (n={len(results[blur])})\n")
        
        # Plot final results
        plt.figure(figsize=(10, 5))
        plt.plot(blur_levels, [averages[blur] for blur in blur_levels], marker='o')
        plt.xlabel('Blur Sigma')
        plt.ylabel('Average Cosine Similarity')
        plt.title('Final Face Verification Similarity vs. Blur Level')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, 'final_plot.png'))
        plt.close()
    
    @staticmethod
    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate blur effects on face verification')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the dataset')
    parser.add_argument('--num_pairs', type=int, default=100, help='Number of image pairs to evaluate')
    parser.add_argument('--report_interval', type=int, default=10, help='Interval at which to save intermediate results')
    parser.add_argument('--blur_levels', type=int, nargs='+', default=[0, 1, 3, 5, 7, 10], help='Blur levels to evaluate')
    parser.add_argument('--det_size', type=int, default=640, help='Detection size for InsightFace')
    parser.add_argument('--save_path', type=str, default="blur_evaluation_results", help='Directory to save results')
    parser.add_argument('--dataset', type=str, default='lfw', choices=['lfw', 'celeba'], help='Which dataset to use: lfw or celeba')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize face verifier
    face_verifier = FaceVerifier(det_size=(args.det_size, args.det_size))
    
    # Create dataset with same_person=True
    if args.dataset == "lfw":
        dataset = LFWDatasetDouble(
            root_dir=args.root_dir,
            transform=None,  # We'll handle transforms in the verifier
            train=True,      # Use training split
            same_person=True # Only get pairs of the same person
        )
    elif args.dataset == "celeba": 
        #create celeba dataset here
        pass
    
    print(f"Evaluating blur effects on {args.num_pairs} image pairs...")
    
    # Run evaluation
    all_results, intermediate_results = face_verifier.evaluate_blur_effects(
        dataset=dataset,
        num_pairs=args.num_pairs,
        blur_levels=args.blur_levels,
        report_interval=args.report_interval,
        save_path=args.save_path
    )
    
    print("\nFinal Results:")
    for blur in args.blur_levels:
        mean_sim = np.mean(all_results[blur]) if all_results[blur] else 0
        print(f"Blur {blur}: Average similarity = {mean_sim:.4f} (n={len(all_results[blur])})")
    
    print(f"\nResults saved to {args.save_path}")

if __name__ == "__main__":
    main()