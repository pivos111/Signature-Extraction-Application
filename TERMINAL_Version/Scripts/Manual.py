import os
import sys
import cv2
import numpy as np
import Inserter
from PIL import Image
from matplotlib import pyplot as plt
from pdf2image import convert_from_bytes

canrun = False

# load input
path = sys.argv[1]
name, extension= os.path.splitext(path)
head, tail = os.path.split(name)

# convert to png
if extension == ".pdf":
    images = convert_from_bytes(open(path,'rb').read())
    for i, image in enumerate(images):
        image.save("temp/"+tail+".png", "PNG")
    filename = "temp/"+tail+".png"
    canrun = True
elif extension == ".jpg" or extension == ".png":
    image = Image.open(path)
    image.save("temp/"+tail+".png", "PNG")
    filename = "temp/"+tail+".png"
    canrun = True
else:
    print("file isn't .pdf or .jpg or .png")
print("Press C to Exit Image selection")


# The coordinates defining the square selected will be kept in this list.
select_coords = []
# While we are in the process of selecting a region, this flag is True.
selecting = False

def get_square_coords(x, y, cx, cy):
    """
    Get the diagonally-opposite coordinates of the square.
    (cx, cy) are the coordinates of the square centre.
    (x, y) is a selected point to which the largest square is to be matched.

    """

    # Selected square edge half-length; don't stray outside the image boundary.
    a = max(abs(cx-x), abs(cy-y))
    a = min(a, w-cx, cx, h-cy, cy)
    return cx-a, cy-a, cx+a, cy+a


def region_selection(event, x, y, flags, param): 
    """Callback function to handle mouse events related to region selection."""
    global select_coords, selecting, image

    if event == cv2.EVENT_LBUTTONDOWN: 
        # Left mouse button down: begin the selection.
        # The first coordinate pair is the centre of the square.
        select_coords = [(x, y)]
        selecting = True

    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        # If we're dragging the selection square, update it.
        image = clone.copy()
        x0, y0, x1, y1 = get_square_coords(x, y, *select_coords[0])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        # Left mouse button up: the selection has been made.
        select_coords.append((x, y))
        selecting = False


# Load the image and get its filename without path and dimensions.
basename = os.path.basename(filename)
image = cv2.imread(filename)
h, w = image.shape[:2]
# The cropped image will be saved with this filename.
cropped_filename = filename
cropped_basename = os.path.basename(cropped_filename)
# Store a clone of the original image (without selected region annotation).
clone = image.copy() 
# Name the main image window after the image filename.
cv2.namedWindow(basename) 
cv2.setMouseCallback(basename, region_selection)

# Keep looping and listening for user input until 'c' is pressed.
while True: 
    # Display the image and wait for a keypress 
    cv2.imshow(basename, image) 
    key = cv2.waitKey(1) & 0xFF
    # If 'c' is pressed, break from the loop and handle any region selection.
    if key == ord("c"): 
        break

# Did we make a selection?
if len(select_coords) == 2: 
    cx, cy = select_coords[0]
    x, y = select_coords[1]
    x0, y0, x1, y1 = get_square_coords(x, y, cx, cy)
    # Crop the image to the selected region and display in a new window.
    cropped_image = clone[y0:y1, x0:x1]
    cv2.imshow(cropped_basename, cropped_image) 
    cv2.imwrite(cropped_filename, cropped_image)
    # Wait until any key press.
    cv2.waitKey(0)

# We're done: close all open windows before exiting.
cv2.destroyAllWindows()

# check extension
if canrun !=True:
    print("file isn't .png .jpg or .pdf")
else:
    # formatting parameters
    img = Image.open(filename)
    width, height = img.size
    basewidth = 399
    baseheight = 399

    # check the size of picture to normalize the biggest side
    if width > height:
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    else:
        hpercent = (baseheight / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, baseheight), Image.ANTIALIAS)

    # save resized image
    img.save('temp/resized.png')

    # load resized image as grayscale
    gray = cv2.imread('temp/resized.png', cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    #print(h,w)

   # load background image as grayscale
    img = np.zeros([400,400,3],dtype=np.uint8)
    img.fill(255) # or img[:] = 255
    plt.imsave('temp/background1.png', img)
    back = cv2.imread('temp/background1.png', cv2.IMREAD_GRAYSCALE)
    hh, ww = back.shape
    #print(hh,ww)

    # compute xoff and yoff for placement of upper left corner of resized image   
    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)
    #print(yoff,xoff)

    # use numpy indexing to place the resized image in the center of background image
    result = back.copy()
    result[yoff:yoff+h, xoff:xoff+w] = gray

    # save resulting centered image in DB
    cv2.imwrite('temp/formatgrey.png', result)
    #read image
    img_grey = cv2.imread('temp/formatgrey.png', cv2.IMREAD_GRAYSCALE)

    # define a threshold, 128 is the middle of black and white in grey scale
    thresh = 200

    # threshold the image
    img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]

    #save image
    cv2.imwrite(filename,img_binary)

    try:
        os.remove("temp/background1.png")
        os.remove("temp/resized.png")
        os.remove("temp/formatgrey.png")
    except Exception as e:
        print(e)
    
    print("1. Yes")
    print("2. No")
    choice = input("is "+filename+" ok?\n")

    if choice == "1":
        # Check number of AFIMI
        while(True):
            afimi = input("Insert your afimi (9 digits)\n")
            if(len(afimi)==9):
                break
            else:
                print("wrong number of digits")

        # Move image to SQL db folder
        head, tail = os.path.split(filename)
        print(tail)
        image = Image.open(filename)
        image.save("SQL_DB/"+tail, "PNG")
        try:
            os.remove(filename)
        except Exception as e:
            print(e)
        filename = "SQL_DB/"+tail
        Inserter.insert(filename, afimi)

    elif choice == "2":
        print("Use Manual.py")
        try:
            os.remove(filename)
        except Exception as e:
            print(e)
    else:
        print("Wrong input")
        try:
            os.remove(filename)
        except Exception as e:
            print(e)


