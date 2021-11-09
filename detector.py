import torch
import cv2
from tqdm import tqdm


class Detector:
    def __init__(self, url):
        self.video = cv2.VideoCapture(url)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def score_frame(self, frame):
        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.classes[int(labels[i])] + ' {:.2f}'.format(row[4].item()), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

    def __call__(self, show=False):
        """
        run detection with option show
        """
        cap = self.video
        assert cap.isOpened()
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(length)):
            ret, frame = cap.read()
            results = self.score_frame(frame)
            if show:
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                frame = self.plot_boxes(results, frame)
                cv2.imshow('video', frame)
