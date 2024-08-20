from dataloader.data_loader import CustomDataset
from train import Train
import torch
import argparse, json

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser('SPCNet')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='dataset path')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--end_epoch', type=int, default=990, help='end epoch')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--weight_list', type=str, default='[10, 10, 1, 10, 10, 0.1, 1]', help='weight list')
    parser.add_argument('--slice_num', type=int, default=-1, help='slice num')
    parser.add_argument('--save_path', type=str, default='test', help='save path')

    # Parse the arguments
    args = parser.parse_args()
    args.save_path = 'result/%s/' % args.save_path
    if args.batch_size < 2:
        raise ValueError("Error: batch size must be greater than 2")

    # Convert args to a dictionary
    cfg = vars(args)

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_train = CustomDataset(input_path=cfg['dataset_path'], size=cfg['size'], slice_num=cfg['slice_num'])

    print(json.dumps(cfg, indent=4, sort_keys=True))
    print('Train Data: %d' % dataset_train.total_num)

    # Start the training process
    train = Train(cfg, device, dataset_train)
    train()

