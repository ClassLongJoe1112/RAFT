import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import glob
import dataset_kittiflow
from utils import flow_viz
from utils import frame_utils

from raft import RAFT
from utils.utils import InputPadder, forward_interpolate

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

@torch.no_grad()
def inference(model, data=None, iters=24, output_path='submission'):
    """ Create submission for the Sintel leaderboard """
    
    model.eval()
    
    if not os.path.exists(output_path):
            os.makedirs(output_path)
            
    if data == 'kittiflow':
        
        test_dataset = dataset_kittiflow.KITTI(split='testing', aug_params=None)

        for test_id in range(len(test_dataset)):
            image1, image2, (frame_id, ) = test_dataset[test_id]
            padder = InputPadder(image1.shape, mode='kitti')
            image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
            output_filename = os.path.join(output_path, frame_id)
            print(output_filename)
            frame_utils.writeFlowKITTI(output_filename, flow)
    
    else:
        
        images = glob.glob(os.path.join(args.data, '*.png')) + \
                 glob.glob(os.path.join(args.data, '*.jpg'))
        
        images = sorted(images)
        
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            t1 = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            torch.cuda.synchronize()
            t2 = time.time()
            
            print("--- %s fps ---" % (1/(t2-t1)))
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            base_filename = os.path.basename(imfile1)
            output_filename = os.path.join(output_path, base_filename)
            print(output_filename)
            frame_utils.writeFlowKITTI(output_filename, flow)
            
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--data', help="path of the dataset to be inferenced, just type 'kittiflow' if you wanna inference the KittiFlow dataset")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--output_path', type=str, help='your desired output path')
    parser.add_argument('--small', action='store_true', help='use small model')
    
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()

    inference(model=model.module, data=args.data, iters=args.iters, output_path=args.output_path)