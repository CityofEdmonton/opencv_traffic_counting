from pipeline import (
    PipelineRunner,
    ContourDetection,
    Visualizer,
    CsvWriter,
    VehicleCounter)
from click_select import Select_polygon, Select_line
import os
import logging
import logging.handlers
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import utils
random.seed(123)


# ============================================================================
IMAGE_DIR = "./out"
VIDEO_SOURCE = "waterdale_long.mp4"
VIDEO_OUT_DEST = "output_waterdale_long.mp4"
# EXIT_PTS = np.array([
#     [[0, 240], [320, 240], [320, 180], [0, 180]]
# ]) # 320*240
MIN_CONTOUR_RATIO = 35./720
AVG_SPEED_INTERVAL = 2  # interval in seconds
USE_PHYSICAL_SPEED = True
# ============================================================================


def select_exit_zones(video_sourse):
    print("Select polygons as the exit zones!")
    cap = cv2.VideoCapture(video_sourse)
    print("Press Enter to go to the next frame, input 'y' to pick this frame")
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            plt.imshow(img)
            plt.show(block=False)
            res = input()
            if res == "y":
                plt.close()
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

    exit_pts = []
    while True:
        sp = Select_polygon(img)
        result = sp.select_polygon()
        if result is None:
            break
        else:
            exit_pts += [result]
    return exit_pts


def select_pixel_distance(video_sourse):
    print("Select two points to calculate Pixel Distance between them!")
    cap = cv2.VideoCapture(video_sourse)
    print("Press Enter to go to the next frame, input 'y' to pick this frame")
    while(cap.isOpened()):
        ret, img = cap.read()
        if ret == True:
            plt.imshow(img)
            plt.show(block=False)
            res = input()
            if res == "y":
                plt.close()
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()

    sl = Select_line(img)
    line_pts = sl.select_line()
    pixel_distance = utils.distance(line_pts[0], line_pts[1])
    return pixel_distance


def train_bg_subtractor(inst, cap, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print('Training BG Subtractor...')
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            inst.apply(frame, None, 0.001)
            i += 1
            if i >= num:
                break
        else:
            break


def main():
    log = logging.getLogger("main")

    # pick pixel distance start&end by double clicking
    PIXEL_DISTANCE = select_pixel_distance(VIDEO_SOURCE)
    if not PIXEL_DISTANCE:
        print("No selection of PIXEL_DISTANCE!")
        return
    else:
        print('PIXEL_DISTANCE: ')
        print(PIXEL_DISTANCE)

    PYHSICAL_DISTANCE = float(input(
        "Please enter the physical distance in meters of the pixel distance selected:"))

    METER_PER_PIXEL = PYHSICAL_DISTANCE/PIXEL_DISTANCE

    # draw polygons using mouse to pick exit points
    EXIT_PTS = select_exit_zones(VIDEO_SOURCE)
    if not EXIT_PTS:
        print("No selection of exit zone!")
        return
    else:
        EXIT_PTS = np.array(EXIT_PTS)
        print('EXIT_PTS: ')
        print(EXIT_PTS)

    # Set up image source
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(width, height)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUT_DEST, fourcc, fps, (width, height))

    # creating exit mask from points, where we will be counting our vehicles
    base = np.zeros((height, width) + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]

    # there is also bgslibrary, that seems to give better BG substruction, but
    # not tested it yet
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, detectShadows=True)

    # processing pipline for programming conviniance
    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor,
                         min_contour_width=int(MIN_CONTOUR_RATIO*height),
                         min_contour_height=int(MIN_CONTOUR_RATIO*height),
                         save_image=False, image_dir=IMAGE_DIR),
        # we use y_weight == 2.0 because traffic are moving vertically on video
        # use x_weight == 2.0 for horizontal.
        VehicleCounter(use_physical_speed=USE_PHYSICAL_SPEED, meter_per_pixel=METER_PER_PIXEL, fps=fps, avg_speed_interval=AVG_SPEED_INTERVAL,
                       exit_masks=[exit_mask], y_weight=2.0),
        Visualizer(use_physical_speed=USE_PHYSICAL_SPEED,
                   video_out=out, image_dir=IMAGE_DIR, save_image=False),
        CsvWriter(path='./', name='report.csv')
    ], log_level=logging.DEBUG)

    # skipping 500 frames to train bg subtractor, close video and reopen
    train_bg_subtractor(bg_subtractor, cap, num=500)
    cap.release()

    frame_number = -1
    frame_time_sec = -1.0/fps
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # calculate the frame time in second
            frame_time_sec += 1.0/fps

            # frame number that will be passed to pipline
            # this needed to make video from cutted frames
            frame_number += 1

            # plt.imshow(frame)
            # plt.show()
            # return

            pipeline.set_context({
                'frame': frame,
                'frame_number': frame_number,
                'frame_time_sec': frame_time_sec
            })
            pipeline.run()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ============================================================================


if __name__ == "__main__":
    log = utils.init_logging()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
