from model import SlayNet
import torch.nn as nn
import torch
from torch.optim.adam import Adam
import statistics
from torchvision import transforms

# from lfw_dataloader import get_lfw_dataloaders
# # Create dataloaders
# train_loader, test_loader, num_classes = get_lfw_dataloaders(
#     "../lfw", batch_size=8, blur_sigma=3
# )
# print(f"Dataset loaded successfully with {num_classes} unique individuals")
# print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# for the celebA dataset
from celebA_dataloader.dataset import CelebADual

face_transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size=15, sigma=(10, 20)),  # Apply Gaussian blur with random sigma
])


if __name__ == "__main__":

    dualdataset = CelebADual(faceTransform=face_transform, dims=128, faceFactor=0.7, basicCrop=True)
    
    print("Dataset loaded")
    
    lr = 2e-4
    epochs = 20
    epslon = 1e-6
    # loss_criterion = nn.BCELoss()
    similarity_criterion = nn.CosineEmbeddingLoss()

    blurnet = SlayNet(inputsize=128, embedding_size=512)
    sharpnet = SlayNet(inputsize=128, embedding_size=512)

    optimizer = Adam(params=list(blurnet.parameters()) + list(sharpnet.parameters()), lr=lr)


    for epoch in range(epochs):

        print(f"Epoch {epoch}")

        losses = []
        for (image_sharp, label_sharp), (image_blur, label_blur) in dualdataset:
            
            optimizer.zero_grad()
            
            embed_blur = blurnet(image_blur)
            embed_sharp = sharpnet(image_sharp)

            # loss is similar to CLIP
            # taken from https://www.reddit.com/r/MLQuestions/comments/10ohrqo/understanding_the_clip_loss_function/
            # similarity between embedding using dot product
                # normalize embeddings
            # embed_blur = embed_blur / (embed_blur.norm(dim=1) + epslon)
            # embed_sharp = embed_sharp /  (embed_sharp.norm(dim=1) + epslon)
            # print(embed_blur.shape)
            # similarity = torch.dot(embed_blur, embed_sharp)
            # print("sim shape", similarity.shape)
            
            
            # is_same_person = torch.eq(label_blur, label_sharp).float()
            is_same_person = torch.where(label_blur == label_sharp, 1, -1)

            loss = similarity_criterion(embed_blur, embed_sharp, is_same_person)

            print(loss)
            print(is_same_person)
            
            loss.backward()

            losses.append(loss.item())

            optimizer.step()

        print(f"Average loss is {statistics.mean(losses)}")

    torch.save(blurnet, "blurnet.pt")
    torch.save(sharpnet, "sharpnet.pt")

