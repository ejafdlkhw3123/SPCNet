from dataloader.data_loader import CustomDataset
from train import Train
from utils import load_pickle
import yaml, torch
import argparse, ast, json

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MRStyler')
    parser.add_argument('--config', type=str, default='config.yaml', help='config file name')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size must be bigger than 2')
    parser.add_argument('--end_epoch', type=int, default=990, help='end epoch')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--size', type=int, default=256, help='image size')
    parser.add_argument('--weight_list', type=str, default='[10, 10, 1, 10, 10, 0.1, 1]')
    parser.add_argument('--dataset_name', type=str, default='sev', help='dataset name')
    parser.add_argument('--slice_num', type=int, default=10, help='dataset name')
    parser.add_argument('--path', type=str, default='test', help='save path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args.path = 'result/%s/' % args.path
    weight_list = ast.literal_eval(args.weight_list)
    args.loss_weight = {'D_1': weight_list[0], 'D_2': weight_list[1], 'Recon': weight_list[2], 'Gan_1': weight_list[3],
                        'Gan_2': weight_list[4], 'Content': weight_list[5], 'Param': weight_list[6]}

    cfg = config[args.dataset_name]
    cfg.update(vars(args))

    train_list, test_list = load_pickle(folder_name=cfg['input_path'], file_name=cfg['pkl_name'])

    dataset_train = CustomDataset(input_path=cfg['input_path'], size=cfg['size'], train_list=train_list, test_list=test_list,
                                  sequence_list=cfg['sequence_list'], slice_num=cfg['slice_num'])

    print(json.dumps(cfg, indent=4, sort_keys=True))
    print('Train Data: %d' % dataset_train.total_num)

    train = Train(cfg, device, dataset_train)
    train()

