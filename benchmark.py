#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

import random

# from repvgg_case import repvgg
# from pycinversnet_DDGL_se_norelu3 import ddgl
from DAMDNet import DAMDNet_v1
from mobilenet_v1 import mobilenet_1
# from Repvgg_ca_at_shuffle3 import repvgg
# from Repvgg_b1g4_ca_at_shuffle3 import repvgg
# from Repvgg_ca_at_shuffle3 import repvgg
# from Repvgg_ca_at_shuffle3 import repvgg
# from RepVGG import repvgg
from radanet import ddgl as repvgg
# from lxnet import repvgg
# from Repvgg_ca_noat_shuffleself3 import repvgg
import time
import numpy as np
from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw import calc_nme as calc_nme_alfw
from benchmark_aflw import ana as ana_aflw

from utils.ddfa import ToTensorGjz, NormalizeGjz, DDFATestDataset, reconstruct_vertex
import argparse


def jigsaw_generator(inputs, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 120 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = inputs.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        # print(temp.size())
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                   y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def extract_param(checkpoint_fp, root='', filelists=None, arch='', num_classes=62, device_ids=[1],
                  batch_size=128, num_workers=4):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}  # 如何重新映射存储位置
    # map_location = torch.device('cpu')
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']

    torch.cuda.set_device(device_ids[0])
    model = repvgg()
    # print(model)

    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    # model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(checkpoint)

    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True  # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, (inputs) in enumerate(data_loader):
            # inputs = inputs.cuda()
            # inputs2 = jigsaw_generator(inputs2, 4)
            output = model(inputs)
            # input1 = jigsaw_generator(inputs, 3)

            # output,_,_= model(inputs,input1)

            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()
                outputs.append(param_prediction)
        outputs = np.array(outputs, dtype=np.float32)
        # print(outputs)

    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs


def _benchmark_aflw(outputs):  # x，y
    return ana_aflw(calc_nme_alfw(outputs))


def _benchmark_aflw2000(outputs):
    return ana_alfw2000(calc_nme_alfw2000(outputs))


def benchmark_alfw_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw(outputs)


def benchmark_aflw2000_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw2000(outputs)


def benchmark_pipeline(arch, checkpoint_fp):
    device_ids = [1]
    print(device_ids)
    def aflw():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='/media/lab280/lilei/3DDFA-master/test.data/AFLW_GT_crop',
            filelists='/media/lab280/lilei/3DDFA-master/test.data/AFLW_GT_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=32)
        benchmark_alfw_params(params)

    def aflw2000():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='/media/lab280/lilei/3DDFA-master/test.data/AFLW2000-3D_crop',
            filelists='/media/lab280/lilei/3DDFA-master/test.data/AFLW2000-3D_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=32)

        benchmark_aflw2000_params(params)

    aflw2000()
    aflw()


#  E:/FaeDataSet/3DDFA/phase1_wpdc_checkpoint_epoch_46.pth.tar
#  ./models/phase1_wpdc_vdc.pth.tar
# 45
# 36 3.681 4.880
def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('--arch', default='', type=str)
    parser.add_argument('-c', '--checkpoint-fp',
                        default='/media/lab280/lilei/3DDFA-master/model/rardnet.tar',
                        # default="/media/lab280/D/lilei/3DDFA_3.52/training/snapshot_repvgg_sem01234/phase1_wpdc_repvgg_sem_checkpoint_epoch_31.pth.tar"9
                        type=str)
    # 43
#35
    args = parser.parse_args()
    #     baseline
    #     3.606  4.759
    #     45  3.603  4.703

    #     space sem
    # 37 3.573  4.750
    # 56 3.630  4.593

    #     sem
    # 45 3.614 4.577
    # 50 3.554 4.576

    #    sem01234
    # 43 3.550 4.552

    #    semspace01234
    # 48 3.521 4.844

    #     baseline_onebatch 48
    #     3.605  4.849

    #    sem01234_onebatch
    # 36 3.577 4.675

    #    semspace01234_onebatch_38
    # 36 3.517 4.641

    #    semspace01234_onebatch_gamma_beta_31
    # 36 3.528 4.641

    #    semspace01234_onebatch_fusion_40
    # 36 3.528 4.641

    #    semspace01234_onebatch_woKSIEM_47
    # 36 3.590 4.721

    #    semspace01234_onebatch_woSIRGM_36
    # 36 3.588 4.658

    #    semspace01234_onebatch_ksiem_with_topk_25
    # 36 3.514 4.613

    #    semspace01234_onebatch_ksiem_with_topk_29
    # 36 3.407 4.582

    #    semspace01234_onebatch_ksiem_with_topk_31
    # 36 3.446 4.597

    benchmark_pipeline(args.arch, args.checkpoint_fp)


if __name__ == '__main__':
    main()
