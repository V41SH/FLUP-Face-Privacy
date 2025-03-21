import argparse
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# InsightFace imports
import insightface
from insightface.app import FaceAnalysis

# only works with Vaishnav's dataloader rn
from lfw_dataloader import blur_face

"""
Face verification using InsightFace and Blur Analysis

This script performs face verification between two images using InsightFace's ArcFace implementation.
It can also do optional face blurring to analyze how it affects verification performance and whether
it actually protects your privacy.

Face verification works by:
1. Detecting faces in both images
2. Extracting deep feature embeddings using ArcFace
3. Computing similarity between the embeddings
4. Determining if they represent the same person based on similarity threshold

Example usage:
    # Basic verification of two images
    python arcface_finfin.py --inputs image1.jpg image2.jpg

    # With blur applied
    python arcface_finfin.py --inputs image1.jpg image2.jpg --blur_sigma 3

    # Compare different blur levels
    python arcface_finfin.py --inputs image1.jpg image2.jpg --compare_blur_levels

"""

def parse_args():
    parser = argparse.ArgumentParser(description='Verify faces with optional blurring')
    parser.add_argument('--inputs', nargs='+', required=True, help='Paths to input images')
    parser.add_argument('--blur_sigma', type=float, default=None, help='Blur sigma to apply')
    parser.add_argument('--compare_blur_levels', action='store_true', help='Compare different blur levels')
    parser.add_argument('--det_size', type=int, default=640, help='Detection size for InsightFace')
    return parser.parse_args()

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
        
        # Detect faces ( it uses retinaface for face detection)
        faces = self.app.get(img)
        
        if len(faces) == 0:
            return None
        
        # Get the face with the highest detection score
        face = max(faces, key=lambda x: x.det_score)
        
        # Extract embedding
        embedding = face.embedding
        bbox = face.bbox.astype(int)
        
        return embedding, bbox, face.det_score
    
    def load_and_process_image(self, img_path, blur_sigma=None):
        """Load an image, apply blur if needed, and return it"""
        # Load the image
        img = Image.open(img_path).convert('RGB')
        
        # Apply blur if specified
        if blur_sigma is not None and blur_sigma > 0:
            img = blur_face(img, blur_sigma)
        
        return img
    
    def compare_faces(self, img_path1, img_path2, blur_sigma=None):
        """Compare two face images and return similarity score"""
        # Load and process images
        img1 = self.load_and_process_image(img_path1, blur_sigma)
        img2 = self.load_and_process_image(img_path2, blur_sigma)
        
        img1_np = np.array(img1)
        img2_np = np.array(img2)
        img1_np = cv2.cvtColor(img1_np, cv2.COLOR_RGB2BGR)
        img2_np = cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)
        
        # Get embeddings
        result1 = self.get_face_embedding(img1_np)
        result2 = self.get_face_embedding(img2_np)
        
        if result1 is None or result2 is None:
            print("Face detection failed on one or both images")
            return None, None, None, None
        
        embedding1, bbox1, score1 = result1
        embedding2, bbox2, score2 = result2
        
        # Calculate similarity
        similarity = self.cosine_similarity(embedding1, embedding2)
        
        return similarity, (img1_np, bbox1), (img2_np, bbox2), (score1, score2)
    
    def compare_blur_levels(self, img_path1, img_path2, blur_levels=[0, 1, 3, 5, 7, 10]):
        """Compare face verification results at different blur levels"""
        results = []
        
        for blur in blur_levels:
            blur_sigma = blur if blur > 0 else None
            result = self.compare_faces(img_path1, img_path2, blur_sigma)
            
            if result[0] is None:
                print(f"Skipping blur level {blur} due to face detection failure")
                continue
                
            similarity = result[0]
            results.append((blur, similarity))
        
        # Plot results
        blurs, similarities = zip(*results)
        
        plt.figure(figsize=(10, 5))
        plt.plot(blurs, similarities, marker='o')
        plt.xlabel('Blur Sigma')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Face Verification Similarity vs. Blur Level')
        plt.grid(True)
        plt.savefig('blur_comparison.png')
        plt.show()
        
        return results
    
    @staticmethod
    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    #gpted it
    def visualize_comparison(self, img_path1, img_path2, blur_sigma=None):
        """Visualize the face comparison with bounding boxes and similarity"""
        result = self.compare_faces(img_path1, img_path2, blur_sigma)
        
        if result[0] is None:
            print("Face detection failed, cannot visualize")
            return
        
        similarity, (img1, bbox1), (img2, bbox2), (score1, score2) = result
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display first image
        axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        x1, y1, x2, y2 = bbox1
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             fill=False, edgecolor='green', linewidth=2)
        axes[0].add_patch(rect)
        axes[0].set_title(f"Image 1\nDetection score: {score1:.3f}")
        axes[0].axis('off')
        
        # Display second image
        axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        x1, y1, x2, y2 = bbox2
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                             fill=False, edgecolor='green', linewidth=2)
        axes[1].add_patch(rect)
        axes[1].set_title(f"Image 2\nDetection score: {score2:.3f}")
        axes[1].axis('off')
        
        # Add similarity as suptitle
        threshold = 0.5  # Adjust based on your needs
        verdict = "Same person" if similarity > threshold else "Different people"
        plt.suptitle(f"Similarity: {similarity:.4f} - {verdict}", fontsize=16)
        
        if blur_sigma:
            plt.figtext(0.5, 0.01, f"Blur sigma: {blur_sigma}", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig("face_comparison.png")
        plt.show()

def main():
    args = parse_args()
    
    # Check if we have at least two images
    if len(args.inputs) < 2:
        print("Error: Please provide at least two image paths")
        return
    
    # Initialize face verifier
    face_verifier = FaceVerifier(det_size=(args.det_size, args.det_size))
    
    if args.compare_blur_levels:
        # Compare different blur levels
        img_path1, img_path2 = args.inputs[:2]
        results = face_verifier.compare_blur_levels(img_path1, img_path2)
        print("\nBlur Level Comparison Results:")
        for blur, similarity in results:
            blur_text = f"{blur}" if blur > 0 else "No blur"
            print(f"Blur sigma {blur_text}: Similarity = {similarity:.4f}")
    else:
        # Simple verification with optional blur
        img_path1, img_path2 = args.inputs[:2]
        face_verifier.visualize_comparison(img_path1, img_path2, args.blur_sigma)
        
        # Get similarity for reporting
        result = face_verifier.compare_faces(img_path1, img_path2, args.blur_sigma)
        if result[0] is not None:
            similarity = result[0]
            # The threshold can be adjusted however we want
            # We could also tune the threshold based on confidence level
            threshold = 0.5 
            is_same_person = similarity > threshold
            
            print(f"\nFace Verification Results:")
            print(f"Image 1: {os.path.basename(img_path1)}")
            print(f"Image 2: {os.path.basename(img_path2)}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Verdict: {'Same person' if is_same_person else 'Different people'}")
            
            if args.blur_sigma is not None:
                print(f"Blur sigma applied: {args.blur_sigma}")

if __name__ == "__main__":
    main()