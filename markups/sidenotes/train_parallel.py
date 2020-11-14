
#### DATASET #####

import numpy as np
import os 
import xml.etree.ElementTree as ET
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from fvcore.common.file_io import PathManager
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
    

def load_voc_instances(dirname, split, CLASS_NAMES):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, split+".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts

def visualize_dataset(datasetname, n_samples=10):

    dataset_dicts = DatasetCatalog.get(datasetname)
    metadata = MetadataCatalog.get(datasetname)

    for d in random.sample(dataset_dicts,n_samples):
        print(d['file_name'])
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
        metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        figure(num=None, figsize=(15, 15), dpi=100, facecolor='w', edgecolor='k')
        plt.axis("off")
        plt.imshow(vis.get_image()[:, :, ::-1])
        plt.show()

        
def register_pascal_voc(name, dirname, split, CLASS_NAMES):
    if name not in DatasetCatalog.list():
        DatasetCatalog.register(name, lambda: load_voc_instances(dirname, split, CLASS_NAMES))
        
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, split=split, dirname= dirname, year=2012
    )

register_pascal_voc('signature_dataset_train', dirname='datasets/train_data', split='train', CLASS_NAMES=["signature","others"])

#### CONFIG ####

from detectron2.engine import default_argument_parser
from detectron2.engine import DefaultTrainer, launch
from detectron2.engine import default_setup
from detectron2.config import get_cfg


def setup_cfg(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


#### TRAINING ####

def main(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args("--config-file sign_config/sign_faster_rcnn_R_50_FPN_3x.yaml OUTPUT_DIR model_full_dataset ".split())

    cfg = setup_cfg(args)
    scaled_cfg = DefaultTrainer.auto_scale_workers(cfg, num_workers=4) # scale config for 4 gpus
    launch(
            main,
            num_gpus_per_machine=4,
            num_machines=1,
            machine_rank=0,
            dist_url='auto',
            args=(scaled_cfg,),
    )
