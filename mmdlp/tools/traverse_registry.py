import argparse
import os
import os.path as osp

from mmengine.registry import traverse_registry_tree, count_registered_modules
from mmpretrain.utils import register_all_modules as register_all_mmpretrain_modules
from mmseg.utils import register_all_modules as register_all_mmseg_modules
from mmdet.utils import register_all_modules as register_all_mmdet_modules
from mmyolo.utils import register_all_modules as register_all_mmyolo_modules
from mmdlp.utils import register_all_modules as register_all_mmdlp_modules


def parse_args():
    parser = argparse.ArgumentParser(description='pring registered modules')
    parser.add_argument('save_dir', help='save directory')
    return parser.parse_args()

if __name__ == "__main__":
    register_all_mmpretrain_modules()
    register_all_mmseg_modules()
    register_all_mmdet_modules()
    register_all_mmyolo_modules()
    register_all_mmdlp_modules()

    args = parse_args()
    count_registered_modules(args.save_dir)

