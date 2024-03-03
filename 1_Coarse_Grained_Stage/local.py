import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    haha = cv2.imread('haha.png')
    plt.imshow(haha)
    plt.show()
    GRID = 16
    stride =