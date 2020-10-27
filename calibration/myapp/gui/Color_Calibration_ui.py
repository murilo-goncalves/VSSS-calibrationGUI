# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Color_Calibration.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Color_Calibration(object):
    def setupUi(self, Color_Calibration):
        Color_Calibration.setObjectName("Color_Calibration")
        Color_Calibration.resize(1122, 565)
        self.centralwidget = QtWidgets.QWidget(Color_Calibration)
        self.centralwidget.setObjectName("centralwidget")
        self.colors = QtWidgets.QLabel(self.centralwidget)
        self.colors.setGeometry(QtCore.QRect(30, 20, 1061, 191))
        self.colors.setObjectName("colors")
        self.frame = QtWidgets.QLabel(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(200, 220, 427, 240))
        self.frame.setObjectName("frame")
        self.ok_button2 = QtWidgets.QPushButton(self.centralwidget)
        self.ok_button2.setGeometry(QtCore.QRect(960, 490, 101, 31))
        self.ok_button2.setObjectName("ok_button2")
        self.color_list = QtWidgets.QListWidget(self.centralwidget)
        self.color_list.setGeometry(QtCore.QRect(40, 240, 131, 231))
        self.color_list.setObjectName("color_list")
        self.color_insert = QtWidgets.QTextEdit(self.centralwidget)
        self.color_insert.setGeometry(QtCore.QRect(40, 480, 131, 31))
        self.color_insert.setObjectName("color_insert")
        self.frame_clusterizado = QtWidgets.QLabel(self.centralwidget)
        self.frame_clusterizado.setGeometry(QtCore.QRect(660, 220, 427, 240))
        self.frame_clusterizado.setObjectName("frame_clusterizado")
        self.clusterize_button = QtWidgets.QPushButton(self.centralwidget)
        self.clusterize_button.setGeometry(QtCore.QRect(450, 490, 241, 31))
        self.clusterize_button.setObjectName("clusterize_button")
        Color_Calibration.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Color_Calibration)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1122, 22))
        self.menubar.setObjectName("menubar")
        Color_Calibration.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Color_Calibration)
        self.statusbar.setObjectName("statusbar")
        Color_Calibration.setStatusBar(self.statusbar)

        self.retranslateUi(Color_Calibration)
        QtCore.QMetaObject.connectSlotsByName(Color_Calibration)

    def retranslateUi(self, Color_Calibration):
        _translate = QtCore.QCoreApplication.translate
        Color_Calibration.setWindowTitle(_translate("Color_Calibration", "MainWindow"))
        self.colors.setText(_translate("Color_Calibration", "TextLabel"))
        self.frame.setText(_translate("Color_Calibration", "TextLabel"))
        self.ok_button2.setText(_translate("Color_Calibration", "Ok"))
        self.frame_clusterizado.setText(_translate("Color_Calibration", "TextLabel"))
        self.clusterize_button.setText(_translate("Color_Calibration", "CLUSTERIZE"))

