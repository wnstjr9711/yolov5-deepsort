from PySide2.QtCore import QThread
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QMainWindow, QLabel, QVBoxLayout
from .ui import Ui_MainWindow
import sys

sys.path.append("..")

from detector import Detector


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setupUi(self)
        self.setAcceptDrops(True)
        self.video.setFixedSize(480, 360)
        self.video_label = QLabel(self.video)
        self.video_label.setFixedSize(480, 360)
        self.thread_video = None

        self.label.setPixmap(QPixmap(u"gui/static/favicon.png"))

        self.loading.setText('Uploading: {:3}%'.format(0))
        self.title.setText('Video Title: -')
        self.length.setText('Video Length: 00:00')
        self.date.setText('Uploaded Date: -')
        self.labels = [[QLabel(self.scrollAreaWidgetContents), 'category'] for i in range(500)]
        for i in self.labels:
            self.verticalLayout_2.addWidget(i[0])

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        url = e.mimeData().urls()[0].toLocalFile()
        self.thread_video = VideoThread(url, self.video_label, self.loading, self.title,
                                        self.length, self.date, self.scrollArea, self.labels)
        self.thread_video.start()


class VideoThread(QThread):
    def __init__(self, video, label, status, title, length, date, objects, detected):
        super(VideoThread, self).__init__()
        self.video = video
        self.label = label
        self.status = status
        self.title = title
        self.length = length
        self.date = date
        self.objects = objects
        self.objects_key = set()
        self.detected = detected
        self.status.setText('Uploading: {:3}%'.format(0))
        self.title.setText('Video Title: -')
        self.length.setText('Video Length: 00:00')
        self.date.setText('Uploaded Date: -')

    def run(self):
        detector = Detector(self.video)(gui={'video': self.label, 'status': self.status, 'title': self.title,
                                             'length': self.length, 'date': self.date, 'objects': self.objects,
                                             'key': self.objects_key, 'detected': self.detected})
