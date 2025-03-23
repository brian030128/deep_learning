import argparse
import models.unet
import oxford_pet
import torch
import models
from tqdm import tqdm
import numpy as np
import utils
import time
import evaluate as eval

import json



def train(args):
    start = int(time.time())


    metric_file = open(f"metrics/{start}_metrics.json", "w")

    dataset = oxford_pet.load_dataset(args.data_path, "train")
    val_dataset = oxford_pet.load_dataset(args.data_path, "valid")

    model = models.unet.UNet(n_channels=3, n_classes=2)
   # model.load_state_dict(torch.load("saved_models/1742732388.59257_0.02590046527431063_5_model.pth"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()


    # Train the model
    loss_history = [] # per step
    dice_history = [] # per epoch
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        model.train()
        train_loss = 0.0  # Initialize training loss for the epoch
        with tqdm(total=len(dataset), desc=f"Epoch {epoch + 1} [Training]", unit="sample") as pbar:
            for i in range(0, len(dataset), args.batch_size):
                batch = dataset[i:i+args.batch_size]

                images = torch.tensor(np.array([x['image'] for x in batch]), dtype=torch.float32).to(device)
                masks = torch.tensor(np.array([x['mask'] for x in batch]), dtype=torch.long).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                loss_history.append(loss.item())
                train_loss += loss.item()  # Accumulate training loss
                pbar.set_postfix(loss=loss.item())
                pbar.update(len(batch))

        avg_train_loss = train_loss / len(dataset)  # Calculate average training loss
        print(f"Training Loss after Epoch {epoch + 1}: {avg_train_loss:.4f}")


        # Validation phase
        eval_loss, eval_dice_score = eval.evaluate(model, val_dataset,device, args.batch_size)
        dice_history.append(eval_dice_score)

        print(f"Validation Loss after Epoch {epoch + 1}: {eval_loss:.4f}")
        print(f"Dice Score after Epoch {epoch + 1}: {eval_dice_score:.4f}")

        torch.save(model.state_dict(), f"{args.checkpoint}/{start}_{eval_loss}_{epoch}_model.pth")
    
    json.dump({"loss": loss_history, "dice": dice_history}, metric_file)



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="dataset", help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=12, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--checkpoint',type=str, default='saved_models', help='folder of checkpoints')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)

