import urllib.request
import cv2
import os


def download_images():
    # http://www.image-net.org/synset?wnid=n07942152
    # http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04105893
    # http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03529175
    # http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03786313
    # http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n03077616
    # http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n04065464
    neg_images_link = "http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152"
    neg_image_urls = urllib.request.urlopen(neg_images_link).read().decode()

    if not os.path.exists('neg'):
        os.makedirs('neg')

    pic_num = 1

    for i in neg_image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, "neg/" + str(pic_num) + ".jpg")
            img = cv2.imread("neg/" + str(pic_num) + ".jpg", cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img, (640, 480))
            cv2.imwrite("neg/" + str(pic_num) + '.jpg', resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))


download_images()
