from dataloader.data_loader import CustomDataset
from test import Tester
import torch
import argparse, json


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SPCNet')
    parser.add_argument('--dataset_path', type=str, default='./dataset', help='dataset_path')
    parser.add_argument('--load_epoch', type=int, default=5, help='load epoch')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--load_path', type=str, default='test', help='load path')
    parser.add_argument('--slice_num', type=int, default=10, help='slice num')
    parser.add_argument('--save_path', type=str, default='test', help='save path')
    args = parser.parse_args()

    # Parse the arguments
    args.load_path = 'result/%s/' % args.load_path

    # Convert args to a dictionary
    cfg = vars(args)
    cfg['weight_folder'] = cfg['load_path'] + '/weight/'

    # Set the device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_test = CustomDataset(input_path=cfg['dataset_path'], size=cfg['size'], slice_num=cfg['slice_num'], shuffle=True)

    print(json.dumps(cfg, indent=4, sort_keys=True))
    print('Test Data: %d' % len(dataset_test.test_list))

    tester = Tester(cfg, device, dataset_test)

    # Save Image
    tester.create_image()

