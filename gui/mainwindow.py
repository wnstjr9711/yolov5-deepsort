from PySide2.QtCore import QThread
from PySide2.QtGui import QPixmap
from PySide2.QtWidgets import QMainWindow, QLabel, QWidget, QHBoxLayout, QCheckBox
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

        self.labels = [QWidget(self.scrollAreaWidgetContents) for i in range(100)]
        for i in self.labels:
            self.verticalLayout_2.addWidget(i)
            self.verticalLayout = QHBoxLayout(i)
            obj = QLabel(i)
            name = QLabel(i)
            self.verticalLayout.addWidget(obj)
            self.verticalLayout.addWidget(name)

        self.checkboxes = [QCheckBox(self.category) for i in range(9)]
        self.gridLayout.addWidget(self.checkboxes[0], 0, 0, 1, 1)
        self.gridLayout.addWidget(self.checkboxes[1], 0, 1, 1, 1)
        self.gridLayout.addWidget(self.checkboxes[2], 0, 2, 1, 1)
        self.gridLayout.addWidget(self.checkboxes[3], 1, 0, 1, 1)
        self.gridLayout.addWidget(self.checkboxes[4], 1, 1, 1, 1)
        self.gridLayout.addWidget(self.checkboxes[5], 1, 2, 1, 1)
        self.gridLayout.addWidget(self.checkboxes[6], 2, 0, 1, 1)
        self.gridLayout.addWidget(self.checkboxes[7], 2, 1, 1, 1)
        self.gridLayout.addWidget(self.checkboxes[8], 2, 2, 1, 1)
        for i in self.checkboxes:
            i.setVisible(False)
            i.stateChanged.connect(self.check)
        self.checkboxes[0].setText('all')
        self.checkboxes[0].setChecked(True)
        self.checkboxes[0].setVisible(True)

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        url = e.mimeData().urls()[0].toLocalFile()
        self.thread_video = VideoThread(url, self.video_label, self.loading, self.title,
                                        self.length, self.date, self.scrollArea, self.labels,
                                        self.total, self.checkboxes)
        self.thread_video.start()

    def check(self):
        if self.sender().isChecked():
            for i in self.checkboxes:
                if i.text() != self.sender().text():
                    i.setChecked(False)

            if self.sender().text() == 'all':
                for i in self.labels:
                    i.setVisible(True)
            else:
                for i in self.labels:
                    text = i.children()[2].text()
                    if text and text.split('\n\n')[1].split(': ')[-1] == self.sender().text():
                        i.setVisible(True)
                    else:
                        i.setVisible(False)






class VideoThread(QThread):
    def __init__(self, video, label, status, title, length, date, objects, detected, total, category):
        super(VideoThread, self).__init__()
        self.video = video
        self.label = label
        self.status = status
        self.title = title
        self.length = length
        self.date = date
        self.objects = objects
        self.objects_key = set()
        self.total = total
        self.detected = detected
        self.category = category
        self.status.setText('Uploading: {:3}%'.format(0))
        self.title.setText('Video Title: -')
        self.length.setText('Video Length: 00:00')
        self.date.setText('Uploaded Date: -')

    def run(self):
        detector = Detector(self.video)(gui={'video': self.label, 'status': self.status, 'title': self.title,
                                             'length': self.length, 'date': self.date, 'objects': self.objects,
                                             'key': self.objects_key, 'detected': self.detected, 'total': self.total,
                                             'category': self.category})
