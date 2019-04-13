from OpenCV_operation import *

cv2.namedWindow("WebcamFeed")
vc = cv2.VideoCapture(0)


def run():
    if vc.isOpened():  # try to get the first frame Basically to check if the frame is available
        rval, frame = vc.read()  # this will return boolean value
        print(rval)
    else:
        rval = False

    while rval:
        rval, frame = vc.read()  # getting new frame every frame
        cv2.rectangle(frame, (197, 68), (444, 409), (0, 255, 0), 3)
        paper_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        paper_GREY = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        paper_threshold, erosion = threshold(frame, paper_HSV, paper_GREY)
        # cont=contours(frame,paper_threshold)

        cv2.imshow('frame', frame)
        cv2.imshow('paper_threshold', paper_threshold)
        cv2.imshow('erosion', erosion)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            rval = False
            break
        if key == 99:  # press c to capture image in the rectangle
            capture_image(frame)
        if key == 97:  # press a to find the size of the shoes
            find_size()

    cv2.destroyWindow("preview")
    vc.release()


if __name__ == '__main__':
    run()
