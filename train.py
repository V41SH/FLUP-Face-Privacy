import torch
from slaymodel import SlayNet
import torch.nn as nn
from torch.optim.adam import Adam
import statistics
from datetime import datetime
from torchvision import transforms

from lfw_triple_loaders import get_lfw_dataloaders

import sys

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    train_loader, test_loader, _ = get_lfw_dataloaders("./data/lfw", batch_size=10, blur_sigma=3)

    eprint("Datasets were loaded")
    
    lr = 2e-4
    epochs = 20
    epslon = 1e-6

    # loss_criterion = nn.BCELoss()
    # similarity_criterion = nn.CosineEmbeddingLoss()
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
            
            sharp_loss = triplet_loss(anchor_sharp_embed, positive_blur_embed, negative_blur_embed)


            # train blur network
            anchor_blur_embed = negative_blur_embed
            positive_sharp_embed = sharpnet(positive_2_sharp)
            negative_sharp_embed = anchor_sharp_embed
            
            blur_loss = triplet_loss(anchor_blur_embed, positive_sharp_embed, negative_sharp_embed)

            loss = (blur_loss + sharp_loss)
            
            loss.backward()

            lossval = loss.item()
            losses.append(lossval)
            print(lossval, end=", ", flush=True)

            optimizer.step()

        eprint(f"Average loss is {statistics.mean(losses)}")

        torch.save(blurnet, f"blurnet-{datetime.now().day}-{datetime.now().hour}-{epoch}.pt")
        torch.save(sharpnet, f"sharpnet-{datetime.now().day}-{datetime.now().hour}-{epoch}.pt")

