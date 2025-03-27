import torch
from slaymodel import SlayNet
import torch.nn as nn
from torch.optim.adam import Adam
import statistics
from torchvision import transforms

from lfw_triple_loaders import get_lfw_dataloaders
# from lfw_dataloader import get_lfw_dataloaders
 # # Create dataloaders
# train_loader, test_loader, num_classes = get_lfw_dataloaders(
#     "../lfw", batch_size=8, blur_sigma=3
# )
# eprint(f"Dataset loaded successfully with {num_classes} unique individuals")
# eprint(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# for the celebA dataset
# from celebA_dataloader.dataset import CelebADual

# face_transform = transforms.Compose([
#     transforms.GaussianBlur(kernel_size=15, sigma=(10, 20)),  # Apply Gaussian blur with random sigma
# ])

import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    # dualdataset = CelebADual(faceTransform=face_transform, dims=128, faceFactor=0.7, basicCrop=True)

    ba_train_loader, ba_test_loader, _ = get_lfw_dataloaders("./data/lfw", blur_sigma=3, anchor_blur=True)
    sa_train_loader, sa_test_loader, _ = get_lfw_dataloaders("./data/lfw", blur_sigma=3, anchor_blur=False)


    eprint("Datasets were loaded")
    
    lr = 2e-4
    epochs = 20
    epslon = 1e-6

    # loss_criterion = nn.BCELoss()
    # similarity_criterion = nn.CosineEmbeddingLoss()
    triplet_loss = nn.TripletMarginLoss().to(device)

    blurnet = SlayNet(inputsize=128, embedding_size=512).to(device)
    sharpnet = SlayNet(inputsize=128, embedding_size=512).to(device)

    optimizer = Adam(params=list(blurnet.parameters()) + list(sharpnet.parameters()), lr=lr)


    for epoch in range(epochs):

        eprint(f"Epoch {epoch}")

        losses = []


        # for (image_sharp, label_sharp), (image_blur, label_blur) in dualdataset:
        for idx, (
            anchor_sharp, positive_blur, negative_blur,
            anchor_blur, positive_sharp, negative_sharp
            ) in enumerate(zip(ba_train_loader, sa_train_loader)):
            
            anchor_sharp = anchor_sharp.to(device)
            positive_blur = positive_blur.to(device)
            negative_blur = negative_blur.to(device)
            anchor_blur = anchor_blur.to(device)
            positive_sharp = positive_sharp.to(device)
            negative_sharp = negative_sharp.to(device)

            optimizer.zero_grad()
            
            # train sharp network
            sharpnet.train()

            eprint("Training sharp")

            anchor_sharp_embed = sharpnet(anchor_sharp)
            eprint("Got embedding for sharp")
            with torch.no_grad():
                blurnet.eval()
                positive_blur_embed = blurnet(positive_blur)
                eprint("Got embedding for posblur")
                negative_blur_embed = blurnet(negative_blur)
                eprint("Got embedding for negblur")
            
            sharp_loss = triplet_loss(anchor_sharp_embed, positive_blur_embed, negative_blur_embed)
            eprint("Got tripletloss")


            # train blur network
            blurnet.train()


            anchor_blur_embed = blurnet(anchor_blur)
            eprint("Got embedding for blur anchor")
            with torch.no_grad():
                sharpnet.eval()
                positive_sharp_embed = sharpnet(positive_sharp)
                eprint("Got embedding for possharp")
                negative_sharp_embed = sharpnet(negative_sharp)
                eprint("Got embedding for negsharp")
            
            blur_loss = triplet_loss(anchor_blur_embed, positive_sharp_embed, negative_sharp_embed)
            eprint("Got tripletloss")

            loss = (blur_loss + sharp_loss)
            
            loss.backward()

            lossval = loss.item()
            losses.append(lossval)
            # print("Lossval is: ", lossval, end=", ")
            print("Lossval is: ", lossval)

            optimizer.step()

        eprint(f"Average loss is {statistics.mean(losses)}")

    torch.save(blurnet, "blurnet.pt")
    torch.save(sharpnet, "sharpnet.pt")

