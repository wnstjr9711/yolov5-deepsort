import numpy as np

from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from tracking import build_tracker
from yaml_parser import config_deep_sort
from tqdm import tqdm

import torch
import cv2


class Detector:
    def __init__(self, url):
        self.video = cv2.VideoCapture(url)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
        self.detector.to(self.device)
        self.detector.half()
        self.classes = self.detector.names

        self.tracker = build_tracker(config_deep_sort('configs/deep_sort.yaml'), use_cuda=True)
        self.img_size = 640
        self.acc_threshold = 0.5
        self.iou_threshold = 0.5

        self.item = dict()

    def __call__(self, ret=None, video_label=None):
        """
        run detecting with option show
        """
        ############################## gui ##############################
        from PySide2.QtGui import QImage, QPixmap
        ############################## gui ##############################
        cap = self.video
        assert cap.isOpened()
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        last_out = None
        process_bar = tqdm(range(length), bar_format='{desc}: {percentage:3.0f}')
        for current_frame_idx in process_bar:
            _, frame = cap.read()
            if current_frame_idx % 2 == 0:
                output = self.image_track(frame)
                last_out = output
            else:
                output = last_out

            ############################## gui ##############################
            new_frame = frame
            ############################## gui ##############################
            if len(output) > 0:
                bbox_xyxy = output[:, :4]
                idx = output[:, -2]
                name = output[:, -1]
                for output_idx in range(len(output)):
                    if idx[output_idx] not in self.item:
                        x1, y1, x2, y2 = bbox_xyxy[output_idx]
                        obj_frame = frame[y1:y2, x1:x2]
                        self.item[idx[output_idx]] = (name[output_idx], obj_frame)
                ############################## gui ##############################
                new_frame = self.draw_boxes(frame, bbox_xyxy, idx, name)
                ############################## gui ##############################

            # ret = (frame, process_bar, self.item)  # process ratio, detected object
            ############################## gui ##############################
            img = cv2.resize(new_frame, dsize=(480, 320), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image = QImage(img, img.shape[1], img.shape[0], img.strides[0], QImage.Format_RGB888)
            video_label.setPixmap(QPixmap.fromImage(image))
            ############################## gui ##############################

    def image_track(self, im0):
        img = letterbox(im0, new_shape=self.img_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # numpy to tensor
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device == 'cuda' else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Detection time *********************************************************
        # Inference
        with torch.no_grad():
            pred = self.detector.model(img, augment=False)[0]

        # Apply NMS and filter object other than person (cls:0)
        pred = non_max_suppression(pred, self.acc_threshold, self.iou_threshold,
                                   classes=[i for i in range(1, len(self.classes))])

        # get all obj ************************************************************
        det = pred[0]  # for video, bz is 1
        if det is not None and len(det):  # det: (#obj, 6)  x1 y1 x2 y2 conf cls

            # Rescale boxes from img_size to original im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results. statistics of number of each obj
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            bbox_xywh = xyxy2xywh(det[:, :4]).cpu()
            confs = det[:, 4:5].cpu()

            # ****************************** deepsort ****************************
            outputs = self.tracker.update(bbox_xywh, confs, im0, det[:, 5].cpu().numpy())  # x1,y1,x2,y2,track_ID
        else:
            outputs = torch.zeros((0, 5))

        return outputs

    def draw_boxes(self, img, bbox, idx, name, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            _idx = int(idx[i])
            _name = self.classes[int(name[i])]
            color = (0, 255, 0)
            label = '{}-{}'.format(_name, _idx)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img
