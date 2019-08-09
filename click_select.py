import cv2
import numpy as np

class Select_polygon:
    def __init__(self, img):
        self.img = img
        self.pts = []  # for storing points

    def draw_roi(self, event, x, y, flags, param):
        img2 = self.img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
            self.pts.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
            self.pts.pop()

        if event == cv2.EVENT_MBUTTONDOWN:
            mask = np.zeros(img.shape, np.uint8)
            points = np.array(self.pts, np.int32)
            points = points.reshape((-1, 1, 2))
            mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
            mask2 = cv2.fillPoly(
                mask.copy(), [points], (255, 255, 255))  # for ROI
            # for displaying images on the desktop
            mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))

            show_image = cv2.addWeighted(
                src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

            cv2.imshow("mask", mask2)
            cv2.imshow("show_img", show_image)

            ROI = cv2.bitwise_and(mask2, img)
            cv2.imshow("ROI", ROI)
            cv2.waitKey(0)

        if len(self.pts) > 0:
            # Draw the last point in pts
            cv2.circle(img2, self.pts[-1], 3, (0, 0, 255), -1)

        if len(self.pts) > 1:
            for i in range(len(self.pts) - 1):
                # x ,y is the coordinates of the mouse click place
                cv2.circle(img2, self.pts[i], 5, (0, 0, 255), -1)
                cv2.line(
                    img=img2, pt1=self.pts[i], pt2=self.pts[i + 1], color=(255, 0, 0), thickness=2)

        cv2.imshow('image', img2)

    # Create images and windows and bind windows to callback functions
    def select_polygon(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_roi)
        print("[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: determine the ROI area")
        print("[INFO] Press ‘S’ to save the selection area")
        print("[INFO] Press finish ESC to finish with previous selections")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                break
            if key == ord("s"):
                cv2.destroyAllWindows()
                return self.pts
        return None

class Select_line:
    def __init__(self, img):
        self.img = img
        self.pts = []  # for storing points
    
    def draw_point(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(self.img,(x,y),5,(0,0,255),-1)
            self.pts.append([x,y])

    def select_line(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.draw_point)
        print("[INFO] Double Click the left button: select the point")
        print("[INFO] Press finish ESC to quit")
        while True:
            cv2.imshow('image',self.img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                break
            if len(self.pts) == 2:
                cv2.destroyAllWindows()
                return self.pts
        return None

if __name__ == "__main__":
    img = cv2.imread("Test_image.jpg")
    sp = Select_polygon(img)
    result = sp.select_polygon()
    print(result)

    img = cv2.imread("Test_image.jpg")
    sl = Select_line(img)
    result = sl.select_line()
    print(result)
