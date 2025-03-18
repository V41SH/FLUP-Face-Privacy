from model import SlayNet
import torch.nn as nn
import torch
from torch.optim.adam import Adam

from lfw_dataloader import get_lfw_dataloaders

# # Create dataloaders
# train_loader, test_loader, num_classes = get_lfw_dataloaders(
#     "../", batch_size=8, blur_sigma=3
# )

# print(f"Dataset loaded successfully with {num_classes} unique individuals")
# print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

if __name__ == "__main__":

    lr = 2e-4
    epochs = 20
    epslon = 1e-6
    criterion = nn.BCELoss()

    blurnet = SlayNet(inputsize=250, embedding_size=512)
    sharpnet = SlayNet(inputsize=250, embedding_size=512)

    optimizer = Adam(params=[blurnet.parameters() + sharpnet.parameters()], lr=lr)


    for epoch in range(epochs):

        for batch_num, data in enumerate(train_loader):
            # TODO need to work on this.
            # either redesign dataloader or apply different shuffles to the same dataloader...

            image_sharp, label_sharp, image_blur, label_blur = data

            optimizer.zero_grad()
            
            embed_blur = blurnet(image_blur)
            embed_sharp = sharpnet(image_sharp)

            # loss is similar to CLIP
            # taken from https://www.reddit.com/r/MLQuestions/comments/10ohrqo/understanding_the_clip_loss_function/

            # similarity between embedding using dot product
                # normalize embeddings
            embed_blur = embed_blur / (embed_blur.norm() + epslon)
            embed_sharp = embed_sharp /  (embed_sharp.norm() + epslon)

            similarity = torch.dot(embed_blur, embed_sharp)

            is_same_person = torch.equal(label_blur, label_sharp)

            loss = criterion(similarity, is_same_person)
            loss.backward()
            optimizer.step()


    torch.save(blurnet, "blurnet.pt")
    torch.save(sharpnet, "sharpnet.pt")

