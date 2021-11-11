from detecting.detector import Detector
from detecting.utils import xy2wh
from tracking import build_tracker
from yaml_parser import config_deep_sort
from tqdm import tqdm

import torch
import cv2


class Extractor:
    def __init__(self, url):
        self.video = cv2.VideoCapture(url)
        self.detector = Detector()
        self.tracker = build_tracker(config_deep_sort('configs/deep_sort.yaml'), use_cuda=True)
        self.items = dict()

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
                if obj_id == 0:  # bear
                    outputs = self.tracker.update(obj_pos, [obj_acc], frame)
                    for j in outputs:
                        if j[4] not in self.items:
                            self.items[j[4]] = frame[j[1]:j[3], j[0]:j[2]]
                            cv2.imshow('img', self.items[j[4]])
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

    def get_items(self):
        return self.items


