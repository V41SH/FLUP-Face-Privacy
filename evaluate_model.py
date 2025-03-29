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

class FaceVerifier:
    def __init__(self, det_size=(640, 640)):
        # self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # self.app.prepare(ctx_id=0, det_size=det_size)
        # print("InsightFace initialized successfully.")
    
        self.sharpnet = torch.load("./sharpnet-29-9-19.pt", weights_only=False).to(device)
        self.blurnet = torch.load("./blurnet-29-9-19.pt", weights_only=False).to(device)

        self.sharpnet.eval()
        self.blurnet.eval()

        self.train_transform, self.test_transform = get_transforms(img_size=224) 

        print("Loaded custom models")
    
    def get_face_embedding(self, img, model_type = 'sharp'):
        """Extract face embedding from an image"""
        
        with torch.no_grad():
            img = self.test_transform(img).to(device).unsqueeze(0)

            if model_type=='sharp':
                embedding = self.sharpnet(img)
            else:
                embedding = self.blurnet(img)

        embedding.requires_grad = False
        return embedding
    
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

    def compare_faces(self, img1, img2, blur_sigma=None, is_same_person=True):
        """Compare two face images and return similarity score"""
        # # Convert PIL images to numpy arrays if needed
        # if isinstance(img1, Image.Image):
        #     img1 = np.array(img1)
        #     img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        # if isinstance(img2, Image.Image):
        #     img2 = np.array(img2)
        #     img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        
        # # Apply blur if specified
        # if blur_sigma is not None and blur_sigma > 0:
        #     # Detect face first
        #     faces1 = self.app.get(img1)
        #     faces2 = self.app.get(img2)
            
        #     if len(faces1) > 0:
        #         face1 = max(faces1, key=lambda x: x.det_score)
        #         img1 = self.blur_face_region(img1, face1.bbox, blur_sigma)
        #     if len(faces2) > 0:
        #         face2 = max(faces2, key=lambda x: x.det_score)
        #         img2 = self.blur_face_region(img2, face2.bbox, blur_sigma)
        
        # only img1 is blurred bro please bro
        if blur_sigma is not None and blur_sigma>0:
            img1 = blur_face(img1, blur_sigma)

        # Get embeddings
        result1 = self.get_face_embedding(img1)
        result2 = self.get_face_embedding(img2)
        
        if result1 is None or result2 is None:
            return None
        
        embedding1 = result1
        embedding2 = result2
        
        # Calculate similarity
        similarity = self.cosine_similarity(embedding1, embedding2)

        if not is_same_person:
            similarity = 1.0 - similarity
        
        return similarity

    def evaluate_blur_effects(self, same_person_dataset, diff_person_dataset, num_pairs=100, blur_levels=[0, 1, 3, 5, 7, 10], 
                            report_interval=10, save_path="arcface_eval_results"):
        """
        Evaluate blur effects on multiple image pairs from the dataset
        
        Args:
            num_pairs: Total number of image pairs to evaluate (50% same, 50% different)
            blur_levels: List of blur sigma values to test
            report_interval: Interval at which to save intermediate results
            save_path: Directory to save results
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
            
            # Skip if the names don't match (shouldn't happen with same_person=True)
            if name1 != name2:
                continue
                
            for blur in blur_levels:
                blur_sigma = blur if blur > 0 else None
                similarity = self.compare_faces(img1, img2, blur_sigma, is_same_person=True)
                
                if similarity is not None:
                    all_results[blur]['same'].append(similarity)
                    all_results[blur]['all'].append(similarity)
            
            # Save intermediate results at intervals
            if (i + 1) % report_interval == 0:
                self._save_intermediate_results(all_results, blur_levels, i + 1, save_path)
                intermediate_results.append((i + 1, {k: {cat: np.mean(scores) for cat, scores in v.items() if scores} 
                                                for k, v in all_results.items()}))
        
        # different-person pairs
        total_processed = num_same_pairs
        for i, idx in enumerate(tqdm(diff_indices, desc="Evaluating different-person pairs")):
            img1, img2, name1, name2 = diff_person_dataset[idx]
            
            if name1 == name2:
                continue
                
            for blur in blur_levels:
                blur_sigma = blur if blur > 0 else None
                similarity = self.compare_faces(img1, img2, blur_sigma, is_same_person=False)
                
                if similarity is not None:
                    all_results[blur]['diff'].append(similarity)
                    all_results[blur]['all'].append(similarity)
            
            # Save intermediate results at intervals
            current_count = total_processed + i + 1
            if current_count % report_interval == 0 or current_count == num_pairs:
                self._save_intermediate_results(all_results, blur_levels, current_count, save_path)
                intermediate_results.append((current_count, {k: {cat: np.mean(scores) for cat, scores in v.items() if scores} 
                                                for k, v in all_results.items()}))
        
        self._save_final_results(all_results, blur_levels, save_path)
        
        return all_results, intermediate_results
    
    def _save_intermediate_results(self, results, blur_levels, num_processed, save_path):
        """Save intermediate results and plots"""
        # Calculate current averages
        averages = {blur: {cat: np.mean(scores) if scores else 0 
                          for cat, scores in categories.items()}
                   for blur, categories in results.items()}
        
        # Save numerical results
        with open(os.path.join(save_path, f"intermediate_{num_processed}.txt"), "w") as f:
            f.write(f"Results after {num_processed} pairs:\n")
            for blur in blur_levels:
                f.write(f"Blur {blur}:\n")
                f.write(f"  Same Person:     Mean similarity = {averages[blur]['same']:.4f} (n={len(results[blur]['same'])})\n")
                f.write(f"  Different Person: Mean 1-similarity = {averages[blur]['diff']:.4f} (n={len(results[blur]['diff'])})\n")
                f.write(f"  All Pairs:        Mean metric = {averages[blur]['all']:.4f} (n={len(results[blur]['all'])})\n")
        
        # Plot current results
        plt.figure(figsize=(12, 8))
        
        # Plot all categories
        plt.subplot(2, 1, 1)
        plt.plot(blur_levels, [averages[blur]['same'] for blur in blur_levels], marker='o', label='Same Person')
        plt.plot(blur_levels, [averages[blur]['diff'] for blur in blur_levels], marker='s', label='Different Person')
        plt.plot(blur_levels, [averages[blur]['all'] for blur in blur_levels], marker='*', label='All Pairs')
        plt.xlabel('Blur Sigma')
        plt.ylabel('Metric Value')
        plt.title(f'Face Verification Metrics vs. Blur Level\n({num_processed} pairs processed)')
        plt.legend()
        plt.grid(True)
        
        # Plot only overall average
        plt.subplot(2, 1, 2)
        plt.plot(blur_levels, [averages[blur]['all'] for blur in blur_levels], marker='o', color='green')
        plt.xlabel('Blur Sigma')
        plt.ylabel('Average Metric (All Pairs)')
        plt.title('Overall Average Metric vs. Blur Level')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'intermediate_plot_{num_processed}.png'))
        plt.close()
    
    def _save_final_results(self, results, blur_levels, save_path):
        """Save final results and plots"""
        # Calculate final averages
        averages = {blur: {cat: np.mean(scores) if scores else 0 
                          for cat, scores in categories.items()}
                   for blur, categories in results.items()}
        
        # Save numerical results
        with open(os.path.join(save_path, "final_results.txt"), "w") as f:
            f.write("Final Results:\n")
            for blur in blur_levels:
                f.write(f"Blur {blur}:\n")
                f.write(f"  Same Person:      Mean similarity = {averages[blur]['same']:.4f} (n={len(results[blur]['same'])})\n")
                f.write(f"  Different Person: Mean 1-similarity = {averages[blur]['diff']:.4f} (n={len(results[blur]['diff'])})\n")
                f.write(f"  All Pairs:        Mean metric = {averages[blur]['all']:.4f} (n={len(results[blur]['all'])})\n")
        
        # Plot final results
        plt.figure(figsize=(12, 8))
        
        # Plot all categories
        plt.subplot(2, 1, 1)
        plt.plot(blur_levels, [averages[blur]['same'] for blur in blur_levels], marker='o', label='Same Person')
        plt.plot(blur_levels, [averages[blur]['diff'] for blur in blur_levels], marker='s', label='Different Person')
        plt.plot(blur_levels, [averages[blur]['all'] for blur in blur_levels], marker='*', label='All Pairs')
        plt.xlabel('Blur Sigma')
        plt.ylabel('Metric Value')
        plt.title('Final Face Verification Metrics vs. Blur Level')
        plt.legend()
        plt.grid(True)
        
        # Plot only overall average
        plt.subplot(2, 1, 2)
        plt.plot(blur_levels, [averages[blur]['all'] for blur in blur_levels], marker='o', color='green')
        plt.xlabel('Blur Sigma')
        plt.ylabel('Average Metric (All Pairs)')
        plt.title('Overall Average Metric vs. Blur Level')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'final_plot.png'))
        plt.close()
    
    @staticmethod
    def cosine_similarity(a, b):
        a = a.squeeze(0)
        b = b.squeeze(0)
        """Calculate cosine similarity between two vectors"""
        # return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        result = torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))
        return result.cpu().numpy()


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
    
    print(f"Evaluating blur effects on {args.num_pairs} image pairs...")
    
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
    for blur in args.blur_levels:
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