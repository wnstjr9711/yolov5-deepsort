import torch


class Detector:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect(self, frame):
        """
        :param frame:
        :return: detected objects [x1, y1, x2, y2, acc, label]
        """
        self.model.to(self.device)
        results = self.model([frame])
        return results.xyxyn[0]
