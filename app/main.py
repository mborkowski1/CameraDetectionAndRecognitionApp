import sys

import cv2
import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.uic import loadUi


class Qt5Cam(QDialog):
    def __init__(self):
        super(Qt5Cam, self).__init__()
        loadUi('gui.ui', self)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start()
        self.image = None

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.display_image(self.image, 1)

    def display_image(self, img, windows=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        out_image = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        out_image = out_image.rgbSwapped()

        if windows == 1:
            self.cameraLabel.setPixmap(QPixmap.fromImage(out_image))
            self.cameraLabel.setScaledContents(True)
        if windows == 2:
            self.cameraColorLabel.setPixmap(QPixmap.fromImage(out_image))
            self.cameraColorLabel.setScaledContents(True)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Qt5Cam()
    window.setWindowTitle('Camera Detection And Recognition App')
    window.show()
    sys.exit(app.exec_())
