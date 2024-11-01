from mmengine.registry import traverse_registry_tree, count_registered_modules
from mmpretrain.utils import register_all_modules as register_all_mmpretrain_modules
# from mmseg.utils import register_all_modules as register_all_mmseg_modules
# from mmdet.utils import register_all_modules as register_all_mmdet_modules
# from mmyolo.utils import register_all_modules as register_all_mmyolo_modules
# from mmdlp.utils import register_all_modules as register_all_mmdlp_modules


if __name__ == "__main__":
    register_all_mmpretrain_modules()
    # register_all_mmseg_modules()
    # register_all_mmdet_modules()
    # register_all_mmyolo_modules()
    # register_all_mmdlp_modules()
    count_registered_modules("./")

