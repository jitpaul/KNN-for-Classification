import numpy as np
import cv2
import sys
import os


def segment_image(image,no_of_clusters):
    vectorized = image.reshape((-1, 3))
    z = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(z, no_of_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((image.shape))


def main(argv):
    no_of_clusters = 3  # hardcoded as 3
    img1 = cv2.imread('image1.jpg')
    img2 = cv2.imread('image2.jpg')
    img3 = cv2.imread('image3.jpg')

    # Segment the images
    result1 = segment_image(img1,no_of_clusters)
    result2 = segment_image(img2,no_of_clusters)
    result3 = segment_image(img3,no_of_clusters)

    # output the results
    d = 'clusteredImages'
    if not os.path.exists(d):
        os.mkdir(d)
    os.chdir(d)
    cv2.imwrite('image1.jpg', result1)
    cv2.imwrite('image2.jpg', result2)
    cv2.imwrite('image3.jpg', result3)

if __name__ == "__main__":
    main(sys.argv)