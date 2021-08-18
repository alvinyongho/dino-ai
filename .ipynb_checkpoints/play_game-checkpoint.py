import mss
import time
import cv2
import numpy
import keyboard
from PIL import Image
import uuid
from tensorflow import keras
import tensorflow as tf

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
image_width = 320
num_screenshots = 0
monitor = {"top": 190, "left": 950, "width": image_width, "height": image_height}



# Load the model
# Why do we load them separately?
model = keras.models.model_from_json(open("model.json", "r").read())
model.load_weights("weights.h5")


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

        last_time = time.time()
        raw_img = sct.grab(monitor)

        # Get raw pixels from the screen and turn it into a numpy array
        img_array = numpy.array(raw_img)
        
        # Converts to grayscale
        grayscale = cv2.cvtColor(img_array, cv2.COLOR_BGRA2GRAY)

        image = Image.frombytes("RGB", raw_img.size, raw_img.rgb)
        gray = image.convert("L")
        img_array = keras.preprocessing.image.img_to_array(gray)

        # Inserts a new axis that will appear at the axis position in the expanded array shape
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        prediction = model.predict(img_array)
        
        result = numpy.argmax(prediction)
        print(prediction)
        # print(result)

        if result == 0:
            duck()
            print("duck")
        elif result == 1:
            jump()
            print("jump")
        elif result == 2:
            nothing()
            print("nothing")


        # cv2.imshow("OpenCV/Numpy normal", gray)

        time.sleep(.00001)
