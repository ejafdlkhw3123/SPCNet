from dataloader.data_loader import CustomDataset
from test import Tester
# from test_no_clip import Tester
from utils import load_pickle
import yaml, torch
import argparse, ast, json


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MRStyler')
    parser.add_argument('--config', type=str, default='config.yaml', help='config file name')
    parser.add_argument('--load_epoch', type=int, default=5, help='load epoch')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--dataset_name', type=str, default='sev', help='dataset name')
    parser.add_argument('--load_path', type=str, default='test', help='load path')
    parser.add_argument('--slice_num', type=int, default=10, help='dataset name')
    parser.add_argument('--save_path', type=str, default='test', help='find epoch start index')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args.load_path = 'result/%s/' % args.load_path
    cfg = config[args.dataset_name]
    cfg.update(vars(args))
    cfg['weight_folder'] = cfg['load_path'] + '/weight/'

    # Data Load
    train_list, test_list = load_pickle(folder_name=cfg['input_path'], file_name=cfg['pkl_name'])

    dataset_test = CustomDataset(input_path=cfg['input_path'], size=cfg['size'], test_list=test_list,
                                 sequence_list=cfg['sequence_list'], slice_num=cfg['slice_num'], shuffle=True)
    #
    print(json.dumps(cfg, indent=4, sort_keys=True))
    print('Test Data: %d' % len(dataset_test.test_list))

    tester = Tester(cfg, device, dataset_test)

    # # Save Image
    tester.create_image()

