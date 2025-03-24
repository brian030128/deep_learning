# Lab 2 Report

## Implementation Detail
For both training and inference, I choose pytorch as the library because it's the deep learning library that I'm most familiar with.

### Training
For the optimizer I choose Adam, as it works the best with a few tries, and for loss function, I used simply cross entropy. I noticed that dice score and cross entropy though both tries to evaluate the difference between the ground truth mask and the generated mask, in some senarios when the cross entropy loss drops , the dice score some times increases. This lead me to try combine cross entropy and dice score as training loss, but it didn't work out well, I couldn't find a meaningful and balanced method to combine both metrics, 

## Data Preprocessing

## Analyze Results

## Execution steps

## Discussion