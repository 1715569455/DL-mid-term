import matplotlib.pyplot as plt
import mmcv
import cv2
from mmdet.apis import init_detector, inference_detector
import numpy as np
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize the reuslts for the first stage of Faster-RCNN')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--image-file',
        help='The image to load')
    parser.add_argument(
        '--out-dir',
        type=str,
        help='The folder to save output file')
    parser.add_argument(
        '--device',
        default='cpu',
        help='device to run the model'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    img = mmcv.imread(args.image_file)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(f'cuda:{0}')
    model = init_detector(args.config, args.checkpoint, device=args.device)

    with torch.no_grad():
        x = model.extract_feat(img)
        rpn_outs = model.rpn_head(x)

        # 解码锚框
        rpn_cls_score = rpn_outs[0]  # (list of tensors, each [N, num_anchors * 2, H, W])
        rpn_bbox_pred = rpn_outs[1]  # (list of tensors, each [N, num_anchors * 4, H, W])
        anchors = model.rpn_head.anchor_generator.grid_anchors([featmap.size()[-2:] for featmap in x], device=x[0].device)

        # 选择高得分的候选区域
        num_levels = len(rpn_cls_score)
        proposals = []
        for i in range(num_levels):
            cls_score = rpn_cls_score[i].detach().cpu().numpy().squeeze().reshape(-1)
            bbox_pred = rpn_bbox_pred[i].detach().cpu().numpy().squeeze()
            anchor = anchors[i].detach().cpu().numpy()

            anchor = anchor[cls_score > 0]
            proposals.append(anchor)

    # 合并所有特征层的候选区域
    stacked_concatenate = np.concatenate(proposals, axis=0)

    # 可视化候选区域
    plt.figure(figsize=(12, 8))
    img_np = mmcv.imread(args.image_file)
    plt.imshow(mmcv.bgr2rgb(img_np))
    for proposal in stacked_concatenate:
        x1, y1, x2, y2 = proposal
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=1.5)
        plt.gca().add_patch(rect)
    plt.axis('off')
    plt.savefig(args.out_dir, bbox_inches='tight', pad_inches=0)
