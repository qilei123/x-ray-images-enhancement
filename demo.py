from src.algorithms.runner import AlgorithmRunner, AlgorithmRunnerX
from pycocotools.coco import COCO
import os
def main():
    data_root = "/home/qilei/.TEMP/TEETH3/"
    json_ann = "train_1_3_crop.json"
    coco = COCO(os.path.join(data_root,"annotations", json_ann))
    for img in coco.imgs:
        img_dir = os.path.join(data_root,"images_crop1",img["file_name"])
        output_dir = os.path.join(data_root,"images_crop_enhance",img["file_name"])
        ar = AlgorithmRunnerX("clahe",img_dir,"",output_dir)
	    ar.run()

if __name__ == "__main__":
    main()