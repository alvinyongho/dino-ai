import mss
import time
import cv2
import numpy
import keyboard
from PIL import Image
import uuid
from tensorflow import keras
import tensorflow as tf
from collections import deque

is_exit = False

def duck():
    keyboard.release("right")
    keyboard.release(keyboard.KEY_UP)
    keyboard.press(keyboard.KEY_DOWN)

# A function for go up in the game
def jump():
    keyboard.release("right")
    keyboard.release(keyboard.KEY_DOWN)
    keyboard.press(keyboard.KEY_UP)


# A function for go right in the game
def nothing():
    keyboard.release(keyboard.KEY_UP)
    keyboard.release(keyboard.KEY_DOWN)
    keyboard.press("right")

def exit():
    global is_exit
    is_exit = True

keyboard.add_hotkey("esc", exit)    # If user clik the 'esc', the program will stop


image_height = 150
image_width = 480
num_screenshots = 0
monitor = {"top": 190, "left": 950, "width": image_width, "height": image_height}

# Load the model
# Why do we load them separately?
model = keras.models.model_from_json(open("model.json", "r").read())
model.load_weights("weights.h5")

last_screenshots = deque(maxlen=4)
predictable_image = None

with mss.mss() as sct:
    while "Screen capturing":
        if is_exit == True:
            keyboard.release("right")
            keyboard.release(keyboard.KEY_DOWN)
            keyboard.release(keyboard.KEY_UP)
            cv2.destroyAllWindows()
            break

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

        # Todo: Grab monitor asynchronously then we can control
        # fps capture rate.

        # Grab is so that we can take a screenshot of just part of the screen.
        raw_img = sct.grab(monitor)

        # Get raw pixels from the screen and turn it into a numpy array
        img_array = numpy.array(raw_img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        last_screenshots.append(gray)

        if len(last_screenshots) < 4:
            continue
        else:
            predictable_image = numpy.concatenate(last_screenshots)
            im = Image.fromarray(predictable_image)
            gray = im.convert("L")
            img_array = tf.expand_dims(predictable_image, 0) # Create a batch
            prediction = model.predict(img_array)
            result = numpy.argmax(prediction)

            print(prediction)

            if result == 0:
                duck()
                print("duck")
            elif result == 1:
                jump()
                print("jump")
            elif result == 2:
                nothing()
                print("nothing")


        # This causes performance issues.
        time.sleep(0.15)
