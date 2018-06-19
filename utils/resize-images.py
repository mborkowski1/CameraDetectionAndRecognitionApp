import cv2
import os


def resize_images():
    for file_type in ['pos']:
        for img in os.listdir(file_type):
            try:
                image = cv2.imread(file_type + '/' + str(img))
                resized_image = cv2.resize(image, (50, 100))
                cv2.imwrite(file_type + '/' + str(img), resized_image)
                print(img + " resized!")

            except Exception as e:
                print(str(e))


resize_images()
