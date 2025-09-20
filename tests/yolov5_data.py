# -*- coding: utf-8 -*-

"""
@Time    : 2025/9/20 19:01
@File    : yolov5_data.py
@Author  : zj
@Description: 
"""

import os
import yaml
import numpy as np

from yolov5.data.auxiliary import check_dataset
from yolov5.data.dataloader import create_dataloader
from yolov5.utils.general import LOGGER, colorstr

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def main(opt):
    hyp, data, single_cls, imgsz, workers = opt.hyp, opt.data, opt.single_cls, opt.imgsz, opt.workers

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Config
    data_dict = check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    LOGGER.info(f"names: {names}")
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
    LOGGER.info(f"is_coco: {is_coco}")

    # Batch size
    gs = 32
    batch_size = 1

    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              prefix=colorstr('train: '),
                                              shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    LOGGER.info(f"train_loader: {train_loader}")
    LOGGER.info(f"labels: {labels}")

    val_loader = create_dataloader(val_path,
                                   imgsz,
                                   batch_size,
                                   gs,
                                   single_cls,
                                   hyp=hyp,
                                   rect=True,
                                   rank=-1,
                                   workers=workers * 2,
                                   pad=0.5,
                                   prefix=colorstr('val: '))[0]

    LOGGER.info(f"val_loader: {val_loader}")


if __name__ == '__main__':
    import argparse
    from yolov5.utils.general import print_args

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')

    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')

    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    opt = parser.parse_args()
    print_args(vars(opt))

    main(opt)
