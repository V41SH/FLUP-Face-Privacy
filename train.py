from model import SlayNet
import torch.nn as nn
import torch
from torch.optim.adam import Adam
import statistics
from torchvision import transforms

from triple_loaders import get_lfw_dataloaders
# from lfw_dataloader import get_lfw_dataloaders
# # Create dataloaders
# train_loader, test_loader, num_classes = get_lfw_dataloaders(
#     "../lfw", batch_size=8, blur_sigma=3
# )
# print(f"Dataset loaded successfully with {num_classes} unique individuals")
# print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# for the celebA dataset
# from celebA_dataloader.dataset import CelebADual

# face_transform = transforms.Compose([
#     transforms.GaussianBlur(kernel_size=15, sigma=(10, 20)),  # Apply Gaussian blur with random sigma
# ])


if __name__ == "__main__":

    # dualdataset = CelebADual(faceTransform=face_transform, dims=128, faceFactor=0.7, basicCrop=True)

    train_loader, test_loader, _ = get_lfw_dataloaders("./data/lfw", blur_sigma=3)


    print("Dataset loaded")
    
    lr = 2e-4
    epochs = 20
    epslon = 1e-6
    # loss_criterion = nn.BCELoss()
    # similarity_criterion = nn.CosineEmbeddingLoss()
    triplet_loss = nn.TripletMarginLoss()

    blurnet = SlayNet(inputsize=128, embedding_size=512)
    sharpnet = SlayNet(inputsize=128, embedding_size=512)

    optimizer = Adam(params=list(blurnet.parameters()) + list(sharpnet.parameters()), lr=lr)


    for epoch in range(epochs):

        print(f"Epoch {epoch}")

        losses = []


        # for (image_sharp, label_sharp), (image_blur, label_blur) in dualdataset:
        for idx, (anchor_sharp, positive_blur, negative_blur) in enumerate(train_loader):
            
            optimizer.zero_grad()
            
            positive_embed = blurnet(positive_blur)
            negative_embed = blurnet(negative_blur)
            anchor_embed = sharpnet(anchor_sharp)
            
            # # is_same_person = torch.eq(label_blur, label_sharp).float()
            # is_same_person = torch.where(label_blur == label_sharp, 1, -1)
            # loss = similarity_criterion(embed_blur, embed_sharp, is_same_person)
            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)

            
            loss.backward()

            losses.append(loss.item())

            optimizer.step()

        print(f"Average loss is {statistics.mean(losses)}")

    torch.save(blurnet, "blurnet.pt")
    torch.save(sharpnet, "sharpnet.pt")

