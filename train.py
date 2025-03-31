import torch
from slaymodel import SlayNet
import torch.nn as nn

import torch.nn.functional as F

from torch.optim.adam import Adam
import statistics
from datetime import datetime
from torchvision import transforms

from lfw_triple_loaders import get_lfw_dataloaders

import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
    print(*args, **kwargs)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    # Model 2
    train_loader, test_loader, _ = get_lfw_dataloaders("./data/lfw", batch_size=10, blur_sigma=[5,20], randomize_blur=True)
    # Model 3
    # train_loader, test_loader, _ = get_lfw_dataloaders("./data/lfw", batch_size=10, blur_sigma=10, randomize_blur=False,
                                                    #    randomize_crop=True)
    # Model 4
    # train_loader, test_loader, _ = get_lfw_dataloaders("./data/lfw", batch_size=10, blur_sigma=[5,20], randomize_blur=True,
    #                                                    randomize_crop=True)

    eprint("Datasets were loaded")
    
    lr = 2e-4
    epochs = 50
    epslon = 1e-6

    triplet_loss = nn.TripletMarginLoss().to(device)

    blurnet = SlayNet(inputsize=224, embedding_size=512).to(device)
    sharpnet = SlayNet(inputsize=224, embedding_size=512).to(device)

    optimizer = Adam(params=list(blurnet.parameters()) + list(sharpnet.parameters()), lr=lr)


    for epoch in range(epochs):

        eprint(f"Epoch {epoch}")
        losses = []

        for idx, (anchor_1_sharp, anchor_2_blur, positive_1_blur, positive_2_sharp
                ), in enumerate(train_loader):
      
            # anchor 1 sharp = negative 1 sharp
            # negative 2 blur = anchor 2 blur

            anchor_1_sharp = anchor_1_sharp.to(device)
            anchor_2_blur = anchor_2_blur.to(device)
            positive_1_blur = positive_1_blur.to(device)
            positive_2_sharp = positive_2_sharp.to(device)


            optimizer.zero_grad()
            
            sharpnet.train()
            blurnet.train()


            # train sharp network
            anchor_sharp_embed = sharpnet(anchor_1_sharp)
            positive_blur_embed = blurnet(positive_1_blur)
            negative_blur_embed = blurnet(anchor_2_blur)

            # normalizing
            # anchor_sharp_embed = anchor_sharp_embed / ( anchor_sharp_embed.norm(dim=-1, keepdim=True) + epslon )
            # negative_blur_embed - negative_blur_embed / ( negative_blur_embed.norm(dim=-1, keepdim=True) + epslon )
            # positive_blur_embed = positive_blur_embed / ( positive_blur_embed.norm(dim=-1, keepdim=True) + epslon )
            anchor_sharp_embed = F.normalize(anchor_sharp_embed)
            positive_blur_embed = F.normalize(positive_blur_embed)
            negative_blur_embed = F.normalize(negative_blur_embed)

            sharp_loss = triplet_loss(anchor_sharp_embed, positive_blur_embed, negative_blur_embed)


            # train blur network
            anchor_blur_embed = negative_blur_embed
            positive_sharp_embed = sharpnet(positive_2_sharp)
            negative_sharp_embed = anchor_sharp_embed

            # normalize normalizing
            positive_sharp_embed = F.normalize(positive_sharp_embed)
            # positive_sharp_embed = positive_sharp_embed / positive_sharp_embed.norm(dim=-1, keepdim=True) + epslon

            blur_loss = triplet_loss(anchor_blur_embed, positive_sharp_embed, negative_sharp_embed)

            loss = (blur_loss + sharp_loss)
            
            loss.backward()

            lossval = loss.item()
            losses.append(lossval)
            eprint(lossval, end=", ", flush=True)

            optimizer.step()

        eprint(f"Average loss is {statistics.mean(losses)}")

        if (epoch % 10) == 0:
            torch.save(blurnet, f"blurnet-{datetime.now().day}-{datetime.now().hour}-{epoch}.pt")
            torch.save(sharpnet, f"sharpnet-{datetime.now().day}-{datetime.now().hour}-{epoch}.pt")

    # After all epochs
    torch.save(blurnet, f"blurnet-{datetime.now().day}-{datetime.now().hour}-{epoch}.pt")
    torch.save(sharpnet, f"sharpnet-{datetime.now().day}-{datetime.now().hour}-{epoch}.pt")
