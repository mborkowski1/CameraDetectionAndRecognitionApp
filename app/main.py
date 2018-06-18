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

        self.objectDetectionButton.setCheckable(True)
        self.objectDetectionButton.toggled.connect(self.detect_webcam_object)
        self.object_Enabled = False

    def update_frame(self):
        ret, self.image = self.capture.read()
        self.image = cv2.flip(self.image, 1)
        self.display_image(self.image, 1)

        if self.face_Enabled:
            detected_image = self.detect_face(self.image)
            self.display_image(detected_image, 1)
        else:
            self.display_image(self.image, 1)

        if self.object_Enabled:
            detected_image2 = self.detect_object(self.image)
            self.display_image(detected_image2, 1)
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

    def detect_webcam_object(self, status):
        if status:
            self.objectDetectionButton.setText('Stop Detection')
            self.object_Enabled = True
        else:
            self.objectDetectionButton.setText('Detect Object')
            self.object_Enabled = False

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

    def detect_object(self, img):
        classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
                cv2.rectangle(img, (startX, startY), (endX, endY), colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        return img


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Qt5Cam()
    window.setWindowTitle('Camera Detection And Recognition App')
    window.show()
    sys.exit(app.exec_())
