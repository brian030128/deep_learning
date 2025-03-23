import argparse
import models.unet
import torch
import oxford_pet
import evaluate

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='saved_models/1742735028.4343746_0.02904472078966058_3_model.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, default="dataset", help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=12, help='batch size')
    
    return parser.parse_args()






if __name__ == '__main__':
    args = get_args()
    torch.set_printoptions(profile="full")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.unet.UNet(n_channels=3, n_classes=2,bilinear=True)
    model.load_state_dict(torch.load(args.model))

    dataset = oxford_pet.load_dataset(args.data_path, "test")
    print(evaluate.evaluate(model, dataset, device, args.batch_size))
