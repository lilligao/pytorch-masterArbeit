import sys
# setting path
sys.path.append('./')
from datasets.tless import TLESSDataset
import numpy as np
from models.mask2former import Mask2Former
import torch
import time
from itertools import groupby
import json
from torchmetrics.classification import BinaryJaccardIndex
import config
from transformers import MaskFormerImageProcessor
from lib.bop_toolkit.bop_toolkit_lib.pycoco_utils import rle_to_binary_mask



def get_bbox(binary_mask):
    binary_mask = binary_mask.numpy()

    segmentation = np.where(binary_mask == 1)

     # Bounding Box
    bbox = 0, 0, 0, 0
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        bbox = x_min, x_max, y_min, y_max
    return [x_min, y_min, x_max-x_min, y_max-y_min]

if __name__ == '__main__':
    # assert(config.LOAD_CHECKPOINTS!=None)
    # path = config.LOAD_CHECKPOINTS # path to the root dir from where you want to start searching
    # model = Mask2Former.load_from_checkpoint(path)
    # model = Mask2Former.load_from_checkpoint("./checkpoints/b5_pbrPrimesense_lr_6e-5_lr_factor_1/epoch=107-val_loss=0.14-val_iou=0.76.ckpt")
    model = Mask2Former()
    model= model.model
    if torch.cuda.is_available():
        model.cuda()
    dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="test")
    num_imgs = len(dataset)
    print("length of num imgs",num_imgs)
    
    image_processor= MaskFormerImageProcessor(
            reduce_labels=True,
            size=(512, 512),
            ignore_index=config.IGNORE_INDEX,
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )
    results = []
    
    for i in range(num_imgs):
        img, target = dataset[i]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img = img.to(device)
        img = img.unsqueeze(0)
        inputs = image_processor(img, return_tensors="pt")
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        time_pred = time.time() - start_time

        target_seg = target["label"]
        target_obj =target["labels_detection"]
        img_size = list(img.shape)[2:]
        preds = image_processor.post_process_instance_segmentation(outputs,target_sizes=img_size, return_coco_annotation=True)[0]
        # all detected objects without background
        # masks and labels of prediction
        mask_preds = preds["segmentation"]
        infos_preds = preds["segments_info"]
        # print(mask_preds)
        # print(len(infos_preds))
        for j in range(len(infos_preds)):
            segment_id =infos_preds[j]["id"]
            label_id = infos_preds[j]["label_id"]
            score_id = infos_preds[j]["score"]

            mask_visible_rle = mask_preds[j]
            mask_visible = rle_to_binary_mask(mask_visible_rle)
            bbox = get_bbox(mask_visible)
            # plt.imshow(mask_visible)
            # plt.savefig('data/tless/label_img_test_'+str(i)+'_'+str(j)+'.png')
            # plt.close()

            result_i = {}
            result_i["scene_id"] = target["scene_id"]
            result_i["image_id"] = target["image_id"]
            result_i["category_id"] = label_id
            result_i["score"] = score_id
            result_i["segmentation"] = mask_visible_rle
            result_i["time"] = time_pred

            results.append(result_i)
            
        print("scene: " + str(target["scene_id"]) + ", image: " + str(target["image_id"]) + " done, time: " + str(time_pred))
        del preds
    

    # convert into json
    # file name is mydata
    file_name = "./outputs/" + config.RUN_NAME + "_tless-test.json"
    with open(file_name, "w") as final:
        json.dump(results, final, indent=2)






