from typing import Tuple
import torch
import numpy as np
from tqdm import tqdm
import utils


def evaluate(model, val_dataset, device, batch_size = 1) -> Tuple[float, float]:
    """
    returns the average cross entropy loss and dice score
    """
    criterion = torch.nn.CrossEntropyLoss()
    
    model.to(device)
    model.eval()
    val_loss = []  # Initialize validation loss for the epoch
    dice_score = []  # Initialize Dice score for the epoch
    
    with torch.no_grad():
        with tqdm(total=len(val_dataset), desc=f"[Validation]", unit="sample") as pbar:
            for i in range(0, len(val_dataset), batch_size):
                batch = val_dataset[i:i+batch_size]

                images = torch.tensor(np.array([x['image'] for x in batch]), dtype=torch.float32).to(device)
                masks = torch.tensor(np.array([x['mask'] for x in batch]), dtype=torch.long).to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                output_mask = utils.output_2_mask(outputs)

                score = utils.dice_score(output_mask, masks)
                dice_score.extend(score)
                val_loss.append(loss.item()) 
                pbar.update(len(batch))

    avg_dice_score =  sum(dice_score) / len(val_dataset)  # Calculate average Dice score
    avg_val_loss = sum(val_loss) / len(val_dataset)  # Calculate average validation loss
    
    return avg_val_loss, avg_dice_score.item()


