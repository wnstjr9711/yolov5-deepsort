import cv2
from PySide2.QtCore import QTimer, QThread
from PySide2.QtGui import QImage, QPixmap
from PySide2.QtWidgets import QMainWindow, QLabel
from .ui import Ui_MainWindow
import sys

sys.path.append("..")

from detector import Detector


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setupUi(self)
        self.setAcceptDrops(True)
        self.video.setFixedSize(480, 320)
        self.video_label = QLabel(self.video)
        self.video_label.setFixedSize(480, 320)
        self.thread_video = None

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        url = e.mimeData().urls()[0].toLocalFile()
        self.thread_video = VideoThread(url, self.video_label)
        self.thread_video.start()

    def resizeEvent(self, e):
        self.video.setGeometry((self.widget.width() - self.video.width())/2, 30, 0, 0)


class VideoThread(QThread):
    def __init__(self, video, label):
        super(VideoThread, self).__init__()
        self.video = video
        self.label = label

    def run(self):
        detector = Detector(self.video)(video_label=self.label)
