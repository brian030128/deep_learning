# Lab 2 Report

## Implementation Detail
For both training and inference, I choose pytorch as the library because it's the deep learning library that I'm most familiar with.

### Training Flow

1. Initialization:
    The process begins by loading the training and validation datasets using the oxford_pet.load_dataset() function
A UNet model is instantiated with 3 input channels (RGB) and 2 output classes (binary segmentation)
The appropriate device (CUDA or CPU) is detected
An Adam optimizer is configured with the specified learning rate


2. Two-Phase Training Strategy:
     I observed an interesting phenomenon during training: as the loss converged, the dice score occasionally decreased even when cross-entropy loss continued to drop. Although both metrics aim to measure the difference between ground truth masks and predictions, their evaluation behaviors can diverge when model performance reaches a certain threshold. Since this task is ultimately evaluated using dice score, I implemented a two-stage training approach: initial training with cross-entropy loss followed by fine-tuning with dice score loss. This strategy yielded a slightly boost on the dice score of the test dataset on both models.
     
     Phase 1: The model is trained using Cross-Entropy loss for the full number of epochs. The best model (based on Dice score) is saved.  
     Phase 2: The best model from Phase 1 is loaded and further fine-tuned using Dice loss for 1/3 of the original epochs

    This two-phase approach helps the model first learn good representations (Cross-Entropy) and then refine the segmentation quality (Dice)




3. Core Training Loop (implemented in train_inner):

    For each epoch:

    Training stage:

    The model is set to training mode `model.train()`
Data is processed in batches of the specified size
For each batch:

    Images and masks are converted to tensors and moved to the appropriate device
Forward pass computes predictions
Loss is calculated using the current loss function
Gradients are computed via backpropagation
The optimizer updates the model parameters
Training loss is accumulated


    Average training loss for the epoch is computed and reported


    Validation stage:

    The model is set to evaluation mode `model.eval()`.
    The evaluate function computes validation loss and Dice score.
    Results are logged and reported.
    The model is saved with a filename that includes timestamp, Dice score, and epoch number. If the current model achieves a better Dice score than previous models, it's tracked as the best model






4. Model Selection:

    After all training epochs, the function returns the name of the best-performing model.
    This model is then used for the second training phase or final evaluation.

### Evaluation
The evaluation code calculates the cross entropy and dice score of the models output compared to ground truth masks.

### Inference
Loads the selected model in saved_models, and run it on the test dataset. Then it calls the evaluation code which calculates the cross entropy and dice score.

## Data Preprocessing




## Analyze Results

## Execution steps

## Discussion