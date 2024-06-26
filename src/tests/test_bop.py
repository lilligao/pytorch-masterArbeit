import sys
# setting path
sys.path.append('./src')
from datasets.tless import TLESSDataset
import numpy as np
from models.segformer import SegFormer
import torch
import time
from itertools import groupby
import json
from torchmetrics.classification import BinaryJaccardIndex
import config
from lib.bop_toolkit.bop_toolkit_lib.pycoco_utils import binary_mask_to_rle

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
    assert(config.LOAD_CHECKPOINTS!=None)
    path = config.LOAD_CHECKPOINTS # path to the root dir from where you want to start searching
    model = SegFormer.load_from_checkpoint(path)
    #model = SegFormer.load_from_checkpoint("./checkpoints/b5_pbrPrimesense_lr_6e-5_lr_factor_1/epoch=107-val_loss=0.14-val_iou=0.76.ckpt")
    model= model.model
    if torch.cuda.is_available():
        model.cuda()
    dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="test")
    num_imgs = len(dataset)
    print("length of num imgs",num_imgs)
    
    results = []
    for i in range(num_imgs):
        img, target = dataset[i]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img = img.to(device)
        img = img.unsqueeze(0)
        #loss, logits = self.model(images, labels.squeeze(dim=1))
        # method took to segment all objects in the image
        start_time = time.time()
        with torch.no_grad():
            preds = model(img)[0]
        time_pred = time.time() - start_time

        # interpolate output of model
        preds = torch.nn.functional.interpolate(preds, size=img.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.softmax(preds, dim=1) # normalize and calculating the possibility
        scores, preds = torch.max(preds, dim=1)# delete the first dimension
        preds = preds.squeeze(0) 
        scores = scores.squeeze(0)

        # all detected objects without background
        detected_obj = torch.unique(preds).tolist()
        detected_obj.remove(0)

        target_obj = torch.unique(target["labels_detection"]).tolist()

        idx = 0

        for j in detected_obj:

            mask_visible = preds==j
            mask_visible = mask_visible.cpu()
            scores = scores.cpu()
            fortran_mask = np.asfortranarray(mask_visible)
            rle = binary_mask_to_rle(fortran_mask)

            bbox = get_bbox(mask_visible)
            # plt.imshow(mask_visible)
            # plt.savefig('data/tless/label_img_test_'+str(i)+'_'+str(j)+'.png')
            # plt.close()

            test_iou = BinaryJaccardIndex()
            if j in target_obj:
                target_mask =  target["masks_visib"][idx,:,:]==255
                idx += 1
            else: # if something detected which is not in target, create a mask with all False
                target_mask =  target["masks_visib"][1,:,:]==999
            
            iou = test_iou(mask_visible, target_mask).item()
            score = torch.mean(scores[mask_visible]).item()

            result_i = {}
            result_i["scene_id"] = target["scene_id"]
            result_i["image_id"] = target["image_id"]
            result_i["category_id"] = j
            result_i["score"] = score
            result_i["iou"] = iou
            result_i["bbox"] = bbox
            result_i["segmentation"] = rle
            result_i["time"] = time_pred

            results.append(result_i)
            
        print("scene: " + str(target["scene_id"]) + ", image: " + str(target["image_id"]) + " done, time: " + str(time_pred))
        del preds
    

    # convert into json
    # file name is mydata
    file_name = "./outputs/" + config.RUN_NAME + "_tless-test.json"
    with open(file_name, "w") as final:
        json.dump(results, final, indent=2)






