import argparse
import models.unet
import models.resnet34_unet
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
    training_dataset = oxford_pet.load_dataset(args.data_path, "train")
    val_dataset = oxford_pet.load_dataset(args.data_path, "valid")

    #model = models.unet.UNet(n_channels=3, n_classes=2)
    model = models.resnet34_unet.Res34Unet(n_channels=3, n_classes=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    cross_entropy = torch.nn.CrossEntropyLoss()

    best_model = train_inner(model, training_dataset, val_dataset, cross_entropy, optimizer, device, epochs= args.epochs,batch_size= args.batch_size)
    #best_model = "1742793169_0.9079298377037048_6_model"
    model.load_state_dict(torch.load(f"{args.checkpoint}/{best_model}.pth"))

    dice_loss = utils.DiceLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_model = train_inner(model, training_dataset, val_dataset, dice_loss, optimizer, device, epochs=int(args.epochs/2), batch_size=args.batch_size, scheduler=scheduler)



    
    

def train_inner(model, training_dataset, validation_dataset, loss_function, optimizer, device,scheduler=None, epochs = 1, batch_size = 3) -> str:
    """
    Returns the name of the best model.
    """

    start = int(time.time())
    metric_file = open(f"metrics/{start}_metrics.jsonl", "w")

    best_model = ""
    best_dice_score = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = []
        with tqdm(total=len(training_dataset), desc=f"Epoch {epoch + 1} [Training]", unit="sample") as pbar:
            for i in range(0, len(training_dataset), batch_size):
                batch = training_dataset[i:i+batch_size]

                images = torch.tensor(np.array([x['image'] for x in batch]), dtype=torch.float32).to(device)
                masks = torch.tensor(np.array([x['mask'] for x in batch]), dtype=torch.long).to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_function(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())  # Accumulate training loss
                pbar.set_postfix(loss=loss.item())
                pbar.update(len(batch))

        avg_train_loss = sum(train_loss) / len(train_loss)  # Calculate average training loss
        print(f"Training Loss after Epoch {epoch + 1}: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        eval_loss, eval_dice_score = eval.evaluate(model, validation_dataset,device, batch_size)

        print(f"CrossEntropy Loss after Epoch {epoch + 1}: {eval_loss:.4f}")
        print(f"Dice Score after Epoch {epoch + 1}: {eval_dice_score:.4f}")

        if scheduler is not None:
            scheduler.step()


        json.dump({"loss": train_loss, "dice": eval_dice_score}, metric_file)
        metric_file.write("\n")
        metric_file.flush()

        name = f"{start}_{eval_dice_score}_{epoch}_model"
        if eval_dice_score > best_dice_score:
            best_dice_score = eval_dice_score
            best_model = name
        
        torch.save(model.state_dict(), f"{args.checkpoint}/{name}.pth")
    
    return best_model


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="dataset", help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=40, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--checkpoint',type=str, default='saved_models', help='folder of checkpoints')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)

