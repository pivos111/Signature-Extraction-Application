import os
import sys
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# load input
path = "signature.png"
name, extension= os.path.splitext(path)

# check extension
if extension != ".png":
    print("file isn't .png")
else:
    # formatting parameters
    img = Image.open(path)
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
    img.save('resized.png')

    # load resized image as grayscale
    gray = cv2.imread('resized.png', cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    #print(h,w)

   # load background image as grayscale
    img = np.zeros([400,400,3],dtype=np.uint8)
    img.fill(255) # or img[:] = 255
    plt.imsave('background1.png', img)
    back = cv2.imread('background1.png', cv2.IMREAD_GRAYSCALE)
    hh, ww = back.shape
    #print(hh,ww)

    # compute xoff and yoff for placement of upper left corner of resized image   
    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)
    #print(yoff,xoff)

    # use numpy indexing to place the resized image in the center of background image
    result = back.copy()
    result[yoff:yoff+h, xoff:xoff+w] = gray

    # view result
    # cv2.imshow('CENTERED', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save resulting centered image in DB
    cv2.imwrite('formatgrey.png', result)
    #read image
    img_grey = cv2.imread('formatgrey.png', cv2.IMREAD_GRAYSCALE)

    # define a threshold, 128 is the middle of black and white in grey scale
    thresh = 200

    # threshold the image
    img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]

    #save image
    cv2.imwrite('formatblack-and-white.png',img_binary)

    try:
        os.remove("background1.png")
        os.remove("resized.png")
        os.remove("formatgrey.png")
        os.remove("signature.png")
        os.rename('formatblack-and-white.png','signature.png')
    except Exception as e:
        print(e)
