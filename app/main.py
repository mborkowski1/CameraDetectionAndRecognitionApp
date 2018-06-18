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

        self.faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        self.faceDetectionButton.setCheckable(True)
        self.faceDetectionButton.toggled.connect(self.detect_webcam_face)
        self.face_Enabled = False


    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.display_image(self.image, 1)

        if self.face_Enabled:
            detected_image = self.detect_face(self.image)
            self.display_image(detected_image, 1)
        else:
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

    def detect_webcam_face(self, status):
        if status:
            self.faceDetectionButton.setText('Stop Detection')
            self.face_Enabled = True
        else:
            self.faceDetectionButton.setText('Detect Face And Eye')
            self.face_Enabled = False

    def detect_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = self.eyeCascade.detectMultiScale(roi_gray)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        return img


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Qt5Cam()
    window.setWindowTitle('Camera Detection And Recognition App')
    window.show()
    sys.exit(app.exec_())
