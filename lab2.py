import cv2
import numpy as np
from PIL import Image, ImageOps
import random
import numpy

labelArray = []
label_conv = {}

def binarize(img_array, threshold=127):
    binary_mask = (img_array > threshold).astype(np.uint8)
    return binary_mask

def colorize(img):
    # Store dimensions of the image
    row, column = img.shape

    # Define colors to exclude (white and black)
    excluded_colors = {(0, 0, 0), (255, 255, 255)}

    lblColor = {0: (0, 0, 0)}
    colouredImage = np.zeros((row, column, 3), dtype=np.uint8)

    for i in range(row):
        for j in range(column):
            label = img[i, j]
            if label not in lblColor:
                while True:
                    color = (random.randint(1, 254), random.randint(1, 254), random.randint(1, 254))
                    if color not in lblColor.values() and color not in excluded_colors:
                        lblColor[label] = color
                        break

            colouredImage[i, j, :] = lblColor[label]

    return colouredImage


def DecideLabel(pixels):
    if all(x == 0 for x in pixels):
        if len(labelArray) == 0:
            labelArray.append(1)
            return max(labelArray)
        else:
            labelArray.append(max(labelArray) + 1)
            return max(labelArray)
    else:
        pixels = [x for x in pixels if x != 0]
        pixels.sort()

        minimumValue = pixels[0]
        maximumValue = pixels[len(pixels) - 1]

        if maximumValue == minimumValue:
            return minimumValue
        else:
            label_conv[maximumValue] = minimumValue
            return minimumValue


# Two pass algorithm
def TwoPassAlgorithm(img):
    # Image size
    row, column = img.shape

    # First Pass
    print("First pass started.....")

    for i in range(row):
        for j in range(column):

            # If pixels are white
            if img[i, j] == 1:

                # First element
                if i == 0 and j == 0:
                    img[i, j] = DecideLabel([])

                # First row without first element
                elif i == 0 and j > 0:
                    img[i, j] = DecideLabel([img[i, j - 1]])


                else:
                    img[i, j] = DecideLabel([img[i - 1, j], img[i, j - 1]])

    # Second Pass
    print("Second pass started.....")

    for index in range(len(label_conv)):
        for i in range(row):
            for j in range(column):
                if img[i][j] in label_conv:
                    img[i][j] = label_conv[img[i][j]]

    return img


def main():
    # Open the image. Here change the location of your image
    img = Image.open("Sample Image 1.png")
    original=img

    # Convert the image into grey luminance format
    img = img.convert('L')

    # Add borders to the image
    img = ImageOps.expand(img, border=1, fill='black')

    # Convert the image into a numpy array
    img = numpy.array(img)

    # Binarize the numpy array with 0 and 1
    img = binarize(img)

    passed_img = TwoPassAlgorithm(img)
    coloured_img=colorize(passed_img).astype(np.uint8)

    cv2.imshow("Output Image", coloured_img)

    cv2.imwrite("result.jpg", coloured_img)
    print("\n\nOutput image is in result.jpg file")
    
    cv2.waitKey()

if __name__ == "__main__": main()