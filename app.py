import cv2
from tqdm import tqdm
import torch
from detecting.detector import Detector
from detecting.utils import xy2wh
from tracking import build_tracker
from yaml_parser import config_deep_sort


class V:
    def __init__(self, url):
        self.video = cv2.VideoCapture(url)
        self.detector = Detector()
        self.tracker = build_tracker(config_deep_sort('configs/deep_sort.yaml'), use_cuda=True)

    def __call__(self):
        """
        run detecting with option show
        """
        cap = self.video
        assert cap.isOpened()
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(length)):
            ret, frame = cap.read()
            frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
            objs = self.detector.detect(frame)
            x_shape, y_shape = frame.shape[1], frame.shape[0]
            for i in range(len(objs)):
                obj = objs[i]
                x1, y1, x2, y2 = int(obj[0]*x_shape), int(obj[1]*y_shape), int(obj[2]*x_shape), int(obj[3]*y_shape)
                obj_pos = xy2wh(torch.tensor([(x1, y1, x2, y2)]))
                obj_acc = obj[4]
                obj_id = int(obj[5].cpu())
                if obj_id == 21:  # bear
                    outputs = self.tracker.update(obj_pos, [obj_acc], frame)
                    bgr = (0, 255, 0)
                    for j in outputs:
                        cv2.rectangle(frame, (j[0], j[1]), (j[2], j[3]), bgr, 2)
                        cv2.putText(frame, str(j[4]), (j[0], j[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            cv2.imshow('v', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break


V('video/sample2.mp4')()

