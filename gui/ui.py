# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'main_categoryQjpDKN.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1000, 1000)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setStyleSheet(u"")
        self.video = QWidget(self.widget)
        self.video.setObjectName(u"video")
        self.video.setGeometry(QRect(10, 70, 480, 360))
        self.video.setStyleSheet(u"background-color: rgb(0, 0, 0);")
        self.label = QLabel(self.widget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(10, 10, 361, 41))
        self.label.setPixmap(QPixmap(u"favicon.png"))
        self.label.setScaledContents(True)
        self.loading = QLabel(self.widget)
        self.loading.setObjectName(u"loading")
        self.loading.setGeometry(QRect(10, 480, 481, 30))
        font = QFont()
        font.setPointSize(15)
        self.loading.setFont(font)
        self.title = QLabel(self.widget)
        self.title.setObjectName(u"title")
        self.title.setGeometry(QRect(10, 520, 481, 30))
        self.title.setFont(font)
        self.length = QLabel(self.widget)
        self.length.setObjectName(u"length")
        self.length.setGeometry(QRect(10, 560, 481, 30))
        self.length.setFont(font)
        self.date = QLabel(self.widget)
        self.date.setObjectName(u"date")
        self.date.setGeometry(QRect(10, 600, 481, 30))
        self.date.setFont(font)
        self.info = QLabel(self.widget)
        self.info.setObjectName(u"info")
        self.info.setGeometry(QRect(10, 440, 481, 30))
        font1 = QFont()
        font1.setPointSize(15)
        font1.setBold(True)
        font1.setWeight(75)
        self.info.setFont(font1)
        self.info.setAlignment(Qt.AlignCenter)
        self.scrollArea = QScrollArea(self.widget)
        self.scrollArea.setObjectName(u"scrollArea")
        self.scrollArea.setGeometry(QRect(500, 140, 471, 831))
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget()
        self.scrollAreaWidgetContents.setObjectName(u"scrollAreaWidgetContents")
        self.scrollAreaWidgetContents.setGeometry(QRect(0, 0, 469, 829))
        self.verticalLayout_2 = QVBoxLayout(self.scrollAreaWidgetContents)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.total = QLabel(self.widget)
        self.total.setObjectName(u"total")
        self.total.setGeometry(QRect(500, 10, 231, 16))
        self.category = QWidget(self.widget)
        self.category.setObjectName(u"category")
        self.category.setGeometry(QRect(500, 30, 471, 101))
        self.gridLayout = QGridLayout(self.category)
        self.gridLayout.setObjectName(u"gridLayout")

        self.verticalLayout.addWidget(self.widget)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label.setText("")
        self.loading.setText(QCoreApplication.translate("MainWindow", u"Uploading", None))
        self.title.setText(QCoreApplication.translate("MainWindow", u"Video Title", None))
        self.length.setText(QCoreApplication.translate("MainWindow", u"Video Length", None))
        self.date.setText(QCoreApplication.translate("MainWindow", u"Uploaded Date", None))
        self.info.setText(QCoreApplication.translate("MainWindow", u"Video Information", None))
        self.total.setText(QCoreApplication.translate("MainWindow", u"Total Count:", None))
    # retranslateUi

