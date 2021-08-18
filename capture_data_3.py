#conda activate dino-ai

import mss
import time
import cv2
import numpy
import keyboard
from PIL import Image
import uuid
from collections import deque

# Last number of dinos that we want to capture within the last second.
FPS = 14
last_screenshots = deque(maxlen=FPS)

num_screenshots = 0
image_height = 150
image_width = 600
monitor = {"top": 190, "left": 950, "width": image_width, "height": image_height}

def saveSnapshot(session_id, category, last_screenshots):
    global num_screenshots
    num_screenshots += 1

    output = f"fps_images/{category}/{session_id}_{num_screenshots}.png"
    
    if len(last_screenshots) < FPS:
        stacked_snapshots = numpy.concatenate(last_screenshots[0]*FPS)
        im = Image.fromarray(stacked_snapshots)

        # Compress via scale
        im = im.resize(size=(image_height, image_width), resample=Image.NEAREST)

        im.save(output)
    else:
        stacked_snapshots = numpy.concatenate(last_screenshots)
        im = Image.fromarray(stacked_snapshots)

        # Compress via scale
        im = im.resize(size=(image_height, image_width))

        im.save(output)



with mss.mss() as sct:
    # Grab the data
    sct_img = sct.grab(monitor)
    session_id = uuid.uuid4()

    while "Screen capturing":
        last_time = time.time()
        raw_img = sct.grab(monitor)
        img_arr = numpy.array(raw_img)

        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        last_screenshots.append(gray)
        cv2.imshow("OpenCV/Numpy normal", gray)


        # We can do image processing here if we need to.
        print("fps: {}".format(1/(time.time()-last_time)))

        if keyboard.is_pressed(keyboard.KEY_UP) or keyboard.is_pressed("space"):
            print("Pressed up")
            saveSnapshot(session_id, "jump", last_screenshots)
            time.sleep(0.1)

        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            print("Pressed down")
            saveSnapshot(session_id, "duck", last_screenshots)
            time.sleep(0.1)

        elif keyboard.is_pressed("right"):
            print("Do nothing")
            saveSnapshot(session_id, "nothing", last_screenshots)
            time.sleep(0.1)

        # Press "q" to quit
        elif cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        time.sleep(1/FPS)
