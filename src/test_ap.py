from datasets.tless import TLESSDataset
import numpy as np
from models.segformer import SegFormer
import torch
import time
from itertools import groupby
import json
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import AveragePrecision
from torchmetrics.detection import MeanAveragePrecision
import config
from torch import tensor
from pprint import pprint

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


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
    # model = SegFormer.load_from_checkpoint(path)
    model = SegFormer.load_from_checkpoint("./checkpoints/b5_pbrPrimesense_lr_6e-5_lr_factor_1/epoch=107-val_loss=0.14-val_iou=0.76.ckpt")
    model= model.model
    if torch.cuda.is_available():
        model.cuda()
    dataset = TLESSDataset(root='./data/tless', split='test_primesense',step="test")
    num_imgs = len(dataset)
    print("length of num imgs",num_imgs)
    
    results = []
     
    metric_map = MeanAveragePrecision(iou_type="segm")
    for i in range(num_imgs):
        img, target = dataset[i]
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img = img.to(device)
        img = img.unsqueeze(0)
        #loss, logits = self.model(images, labels.squeeze(dim=1))
        # method took to segment all objects in the image
        start_time = time.time()
        preds = model(img)[0]
        time_pred = time.time() - start_time

        # interpolate output of model
        preds = torch.nn.functional.interpolate(preds, size=img.shape[-2:], mode="bilinear", align_corners=False)
        preds = torch.softmax(preds, dim=1) # normalize and calculating the possibility
        preds_softmax = preds
        scores, preds = torch.max(preds, dim=1)# delete the first dimension
        preds = preds.squeeze(0) 
        scores = scores.squeeze(0)

        #print("preds.shape", preds.shape)
        #print("mask_visible.shape",mask_visible.unsqueeze(0).shape)
        #print("scores shape",scores.shape)

        # all detected objects without background
        detected_obj = torch.unique(preds).tolist()
        detected_obj.remove(0)

        target_obj = torch.unique(target["labels_detection"]).tolist()

        idx = 0

        targets_map = []
        batch_size = preds_softmax.shape[0]
        for p in range(len(target_obj)):
            for i in range(batch_size):
                mask_tgt = target["masks_visib"][p,:,:]==255
                #print("mask_tgt.shape", target["masks_visib"][p,:,:].shape)
                #print("mask_tgt.shape", mask_tgt.shape)
                mask_tgt = mask_tgt.unsqueeze(0)
                #print("mask_tgt.shape", mask_tgt.shape)
                targets_map.append(
                    dict(
                        masks=mask_tgt,
                        labels=tensor([target_obj[p]]),
                    )
                )
        #print(targets_map)
        preds_map = []
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

            metric_ap = AveragePrecision(task="multiclass", num_classes=31, average="macro")
            ap = metric_ap(preds_softmax, target["label"])
            print("preds_softmax.shape", preds_softmax.shape)
            #print("mask_visible.shape",mask_visible.unsqueeze(0).shape)
            print("target label shape",target["label"].shape)
            print(ap)

            
            for i in range(batch_size):
                # detections: detection results in a tensor with shape [max_det_per_image, 6],
                #  each row representing [x_min, y_min, x_max, y_max, score, class]
                # non_zero_indices retrieval adds extra dimension into dim=1
                #  so needs to squeeze it out
                preds_map.append(
                    dict(
                        masks=mask_visible.unsqueeze(0),
                        scores=tensor([score]),
                        labels=tensor([j]),
                    )
                )

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
            
        #print(preds_map)
        metric_map.update(preds=preds_map, target=targets_map)
        pprint(metric_map.compute())
        print("scene: " + str(target["scene_id"]) + ", image: " + str(target["image_id"]) + " done, time: " + str(time_pred))
        
        del preds,preds_softmax, preds_map, targets_map, metric_ap, ap
    






