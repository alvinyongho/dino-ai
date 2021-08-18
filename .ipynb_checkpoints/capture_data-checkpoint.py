#conda activate dino-ai

import mss
import time
import cv2
import numpy
import keyboard
from PIL import Image
import uuid

num_screenshots = 0
image_height = 150
image_width = 320
monitor = {"top": 190, "left": 950, "width": image_width, "height": image_height}


def saveSnapshot(session_id, category, img_arr):
    global num_screenshots
    num_screenshots += 1
    output = f"images/{category}/{session_id}_{num_screenshots}.png"
    im = Image.fromarray(img_arr)
    im.save(output)


with mss.mss() as sct:
    # Grab the data
    sct_img = sct.grab(monitor)
    session_id = uuid.uuid4()

    while "Screen capturing":
        last_time = time.time()
        raw_img = sct.grab(monitor)
        img_arr = numpy.array(raw_img)
        cv2.imshow("OpenCV/Numpy normal", img_arr)

        # We can do image processing here if we need to.
        print("fps: {}".format(1/(time.time()-last_time)))

        if keyboard.is_pressed(keyboard.KEY_UP) or keyboard.is_pressed("space"):
            print("Pressed up")
            saveSnapshot(session_id, "jump", img_arr)
            time.sleep(0.1)

        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            print("Pressed down")
            saveSnapshot(session_id, "duck", img_arr)
            time.sleep(0.1)

        elif keyboard.is_pressed("right"):
            print("Do nothing")
            saveSnapshot(session_id, "nothing", img_arr)
            time.sleep(0.1)

        # Press "q" to quit
        elif cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
