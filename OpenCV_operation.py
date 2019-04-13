import cv2
import math
import numpy as np


# to get the thrashhold image--------------------------------------------------------------------------------------------
def threshold(original, HSV, grey):
    '''

    :param original: original frame image
    :param HSV: hsv image frame
    :param grey: hsv image as greyscale
    :return: 2 return data value as
            1-> threshold image and 2->eroded image
    :DOUBT: what should be we using Canny detection as edge detection or threshold image
    '''


    # retval1, paper_threshold = cv2.threshold(grey, 50, 130, cv2.THRESH_BINARY_INV)
    paper_threshold = cv2.Canny(grey, 50, 150)
    kernal = np.ones((4, 4), np.uint8)

    erosion = cv2.erode(paper_threshold, kernal, iterations=3)
    erosion = cv2.dilate(paper_threshold, kernal, iterations=2)
    return paper_threshold, erosion


# to find the conture for the image--------------------------------------------------------------------------------------
def contours(original, threshold):
    '''

    :param original: original frame
    :param threshold: threshold image from threshold def:
    :return: 3 value will be from this function

    '''
    _, cnts, hirarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 300:
            break

    # mask = np.zeros(original.shape[:2], np.uint8)
    # cv2.drawContours(mask, [cnt], -1, 255, 3)
    # dst = cv2.bitwise_and(threshold, threshold, mask=mask)
    threshold = cv2.drawContours(threshold, cnt, -1, (0, 255, 0), 3)
    return threshold, hirarchy, cnts


# to capture the image---------------------------------------------------------------------------------------------------
def capture_image(original):
    '''

    :param original: origianl frame
    :return: no runtime return but cropped image will be stored in directory
    '''
    croped_image = original[68:400, 197:444]
    cv2.imwrite("trl_IMAGE.png", croped_image)


# finding the momneyts with the help of conture--------------------------------------------------------------------------
def find_center(conture):
    '''

    :param conture: vector for total number of conture
    :return: midpoint values of conture vector

    '''
    cnt = conture[0]
    cnt2 = conture[1]

    M1 = cv2.moments(cnt, True)
    M2 = cv2.moments(cnt2, True)

    cx = int(M1['m10'] / M1['m00'])
    cy = int(M1['m01'] / M1['m00'])

    cx_2 = int(M2['m10'] / M2['m00'])
    cy_2 = int(M2['m01'] / M2['m00'])

    return cx_2, cy_2, cx, cy


# to find the distance between two point--------------------------------------------------------------------------------
def calculateDistance(x1, y1, x2, y2):
    '''

    :param x1: :param y1: :param x2: :param y2: midpoint values of conture vector
    :return: length in pixel
    '''
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist = convert_to_inches(dist)
    return dist

#this is the function that convert the pixel to inches------------------------------------------------------------------
def convert_to_inches(dist):
    dist = dist * 2.54 / 96
    dist = dist * 29.7 / 21
    myround(dist)
    dist = int(dist)
    print(dist)
    return dist

#my roundoff function--------------------------------------------------------------------------------------------------
def myround(digit):
    if (digit > 0.5):
        return digit + 1
    else:
        return digit - 0


# to find the size of the lines in image--------------------------------------------------------------------------------
def find_size():
    '''

    :return: Collabration of all the function

    '''
    captured_image = cv2.imread("trl_IMAGE.png", 1)
    captured_image_GREY = cv2.imread("trl_IMAGE.png", 0)
    captured_image_HSV = cv2.cvtColor(captured_image, cv2.COLOR_BGR2HSV)

    captured_image_threshold, captured_image_erosion = threshold(captured_image,
                                                                 captured_image_HSV,
                                                                 captured_image_GREY)
    cont, hirarchy, cnts = contours(captured_image, captured_image_erosion)

    cx_2, cy_2, cx, cy = find_center(cnts)

    length = calculateDistance(cx_2, cy_2, cx, cy)

    cv2.circle(captured_image, (102, 233), 5, (255, 0, 0), -1)
    cv2.circle(captured_image, (106, 57), 5, (255, 0, 0), -1)

    cv2.imshow('capture 1', captured_image)
    cv2.imshow('capture 2', captured_image_GREY)
    cv2.imshow('capture 3', captured_image_HSV)
    cv2.imshow('threshold', captured_image_threshold)
    cv2.imshow('erosion', captured_image_erosion)
    print(length)
