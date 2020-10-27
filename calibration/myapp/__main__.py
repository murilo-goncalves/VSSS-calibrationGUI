import sys

import PyQt5
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from .gui.Border_Calibration_ui import Ui_Border_Calibration
from .gui.Color_Calibration_ui import Ui_Color_Calibration
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

import numpy as np
import cv2
import json
from subprocess import call

color_window = 0

class BorderWindow(QMainWindow, Ui_Border_Calibration):
    def __init__(self):
        super(BorderWindow, self).__init__()
        self.setupUi(self)

        self.frame_height, self.frame_width = 240, 427
        self.done = False

        logo_ger = QPixmap("/home/murilo/Documentos/PyQt5/calibration/myapp/gui/images/Logo_final.jpg")
        logo_ger = logo_ger.scaled(250, 250, PyQt5.QtCore.Qt.KeepAspectRatio)
        self.ger.setPixmap(logo_ger)
        self.ger.show()

        #       --> read JSON file "self.data" <--
        self.open_json()
        #       --> border calibration <--
        self.cap_frame()

        self.set_sliders(self.slider0x, 0, self.frame_width, 0)
        self.slider0x.valueChanged.connect(self.callback0)
        self.set_sliders(self.slider0y, 0, self.frame_height, 0)
        self.slider0y.valueChanged.connect(self.callback1)
        self.set_sliders(self.slider1x, 0, self.frame_width, self.frame_width)
        self.slider1x.valueChanged.connect(self.callback2)
        self.set_sliders(self.slider1y, 0, self.frame_height, 0)
        self.slider1y.valueChanged.connect(self.callback3)
        self.set_sliders(self.slider2x, 0, self.frame_width, self.frame_width)
        self.slider2x.valueChanged.connect(self.callback4)
        self.set_sliders(self.slider2y, 0, self.frame_height, self.frame_height)
        self.slider2y.valueChanged.connect(self.callback5)
        self.set_sliders(self.slider3x, 0, self.frame_width, 0)
        self.slider3x.valueChanged.connect(self.callback6)
        self.set_sliders(self.slider3y, 0, self.frame_height, self.frame_height)
        self.slider3y.valueChanged.connect(self.callback7)

        self.refreshTimer = QTimer()
        self.refreshTimer.timeout.connect(self.show_frame)
        self.refreshTimer.start(100)

        self.ok_button.clicked.connect(self.handle_ok)

    def open_json(self):
        try:
            with open('data.json') as f:
                self.data = json.load(f)
            self.p = self.data['points']
            for i in range(4):
                self.p[i] = (self.p[i]['x'], self.p[i]['y'])
        except:
            print("JSON file doesn't exist yet")
            self.data = {}
            self.p = []
            for i in range(4):
                ponto = ()
                self.p.append(ponto)
            # set points as corners of the cam img
            self.p[0] = (0, 0)
            self.p[1] = (self.frame_width, 0)
            self.p[2] = (self.frame_width, self.frame_height)
            self.p[3] = (0, self.frame_height)

        try:
            with open("data.json") as f:
                self.data = json.load(f)
            cam_parameters = self.data['camera_parameters']
        except:
            argv1 = sys.argv[1]
            cam_parameters = f"v4l2-ctl -d /dev/video{argv1} -c saturation=255 -c gain=255 -c \
                               exposure_auto=1 -c exposure_absolute=40 -c focus_auto=0"
        #       --> set camera parameters <--
        call(cam_parameters.split())


    def cap_frame(self):
        self.argv1 = sys.argv[1]  # get camera number as terminal argument
        cap = cv2.VideoCapture(int(self.argv1))

        if not cap.isOpened():
            print("Can't open the video cam")
            quit()

        cap.set(3, 1280)
        cap.set(4, 720)
        dWidth = int(cap.get(3))  # get the width of frames of the video
        dHeight = int(cap.get(4))  # get the height of frames of the video
        print("Frame size:", dWidth, "x", dHeight)  # print image size
        self.cap = cap


    def show_frame(self):

        if(self.done == True):
            return

        # Capture frame-by-frame
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        transformed_frame = self.transform(frame, self.p[0], self.p[1], self.p[2], self.p[3])
        transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2LAB)

        cv2.circle(frame, self.p[0], 5, (255, 0, 0), -1)
        cv2.circle(frame, self.p[1], 5, (0, 255, 0), -1)
        cv2.circle(frame, self.p[2], 5, (0, 0, 255), -1)
        cv2.circle(frame, self.p[3], 5, (255, 255, 255), -1)

        # Display the resulting frame
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        frame = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        frame = QPixmap(frame)
        self.frame_img.setPixmap(frame)
        self.frame_img.show()

        # Display the resulting trasnformed frame
        height, width, channel = transformed_frame.shape
        bytesPerLine = 3 * width
        transformed_frame = QImage(transformed_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        transformed_frame = QPixmap(transformed_frame)
        self.t_frame_img.setPixmap(transformed_frame)
        self.t_frame_img.show()


    def transform(self, img, p0, p1, p2, p3):
        # Array containing the four corners of the field
        inputQuad = np.array([p0, p1, p2, p3],  dtype="float32")
        outputQuad = np.array([(0, 0),  # array containing the four corners of the image
                               (self.frame_width-1, 0),
                               (self.frame_width-1, self.frame_height-1),
                               (0, self.frame_height-1)], dtype="float32")

        # Get the Perspective Transform Matrix i.e. lambda
        lbd = cv2.getPerspectiveTransform(inputQuad, outputQuad)

        # Apply the Perspective Transform just found to the src image
        output = cv2.warpPerspective(img, lbd, (self.frame_width, self.frame_height))

        return output


    def set_sliders(self, slider, min, max, value):
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setValue(value)

    def callback0(self):
        val = self.slider0x.value()
        self.p[0] = (val, self.p[0][1])
    def callback1(self):
        val = self.slider0y.value()
        self.p[0] = (self.p[0][0], val)
    def callback2(self):
        val = self.slider1x.value()
        self.p[1] = (val, self.p[1][1])
    def callback3(self):
        val = self.slider1y.value()
        self.p[1] = (self.p[1][0], val)
    def callback4(self):
        val = self.slider2x.value()
        self.p[2] = (val, self.p[2][1])
    def callback5(self):
        val = self.slider2y.value()
        self.p[2] = (self.p[2][0], val)
    def callback6(self):
        val = self.slider3x.value()
        self.p[3] = (val, self.p[3][1])
    def callback7(self):
        val = self.slider3y.value()
        self.p[3] = (self.p[3][0], val)

    def handle_ok(self):
        self.p2 = list(self.p)
        for i in range(4):
            self.p2[i] = {'x': self.p2[i][0], 'y': self.p2[i][1]}
        self.data['points'] = [self.p2[0], self.p2[1], self.p2[2], self.p2[3]]
        with open('data.json', 'w') as f:
            json.dump(self.data, f, indent=True, ensure_ascii=False)
        self.prox_janela()
    
    def prox_janela(self):
        global color_window
        color_window.transformed_frame = self.transformed_frame
        color_window.show()
        self.cap.release()
        self.close()
        self.done = True
    
class ColorWindow(QMainWindow, Ui_Color_Calibration):
    def __init__(self):
        super(ColorWindow, self).__init__()
        self.setupUi(self)
        self.open_json()
        self.transformed_frame = 0
        self.clusterize_button.clicked.connect(self.handle_clusterize)

    # def clusterize(self, frame):
    #     self.ret, self.label, self.center = cv2.kmeans(self.Z, self.K, None, self.criteria,
    #                                                     10, cv2.KMEANS_RANDOM_CENTERS)
    #     # Now convert back into uint8, and make original image
    #     self.center = np.uint8(self.center)
    #     self.res = center[label.flatten()]
    #     self.res2 = res.reshape((self.img.shape))

    #     # create numbered color rectangles of each cluster
    #     self.retangulos = np.zeros((200, 1061, 3), np.uint8)
    #     self.rect_size = 1061 // self.K
    #     for i in range(K):
    #         # color_rect = tuple([int(x) for x in center[i]])
    #         self.color_rect = np.uint8([[[int(x) for x in center[i]]]])
    #         self.color_rect = cv2.cvtColor(self.color_rect, cv2.COLOR_LAB2BGR)
    #         self.color_rect = tuple([int(x) for x in np.reshape(color_rect, (-1))])
    #         print(self.color_rect)

    #         cv2.rectangle(self.retangulos, (i*rect_size, 0),
    #                         ((i+1)*self.rect_size, 150), self.color_rect, thickness=-1)
    #         cv2.putText(retangulos, str(i), (i*rect_size + rect_size//2 - 15, 185),
    #                         cv2.FONT._HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=3)
        
    def handle_clusterize(self):
        print(type(self.transformed_frame))
        self.img = self.transformed_frame

        # define criteria, number of clusters(K) and apply kmeans()
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-4)
        # self.clusterize(self.img)

        #display the resulting transformed frame
        height, width = 240, 427
        bytesPerLine = 3 * width
        self.img = QImage(self.img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.img = QPixmap(self.img)
        self.frame.setPixmap(self.img)
        self.frame.show()
   
    def open_json(self):
        try:
            with open('data.json') as f:
                self.data = json.load(f)
            self.K = self.data['K']
        except:
            print("JSON file doesn't exist yet")
            self.data = {}
            self.data['colors'] = {}
            self.K = 15

def main():
    app = QApplication(sys.argv)
    border_window = BorderWindow()
    border_window.show()
    global color_window
    color_window = ColorWindow()
    
    app.exec_()

if __name__== '__main__':
    main()
