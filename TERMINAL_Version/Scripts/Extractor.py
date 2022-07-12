import os
import sys
import cv2
import numpy as np
from signature_detect.loader import Loader
from signature_detect.extractor import Extractor
from signature_detect.cropper import Cropper
from signature_detect.judger import Judger
from matplotlib import pyplot as plt
from pdf2image import convert_from_bytes
from PIL import Image
from torch import true_divide
import Inserter

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

if canrun:
    # Load image and HSV color threshold to cut non handwritten text
    image = cv2.imread(filename)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([90, 38, 0])
    upper = np.array([145, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask==0] = (255, 255, 255)
    plt.imsave('temp/masked.png', result)

    # Extraction parameters
    loader = Loader(
        low_threshold=(0, 0, 250), 
        high_threshold=(255, 255, 255))
    extractor = Extractor(
        outlier_weight=3, 
        outlier_bias=100, 
        amplfier=8, 
        min_area_size=10)
    cropper = Cropper(
        min_region_size=10000, 
        border_ratio=0.0)
    judger = Judger(
        size_ratio=[1, 2], 
        pixel_ratio=[0.01, 1])

    # Start Extraction process
    try:
        masks = loader.get_masks("temp/masked.png")
        is_signed = False
        for mask in masks:
            labeled_mask = extractor.extract(mask)
            # plt.imsave('mask.png', mask)
            results = cropper.run(labeled_mask)
            for result in results.values():
                is_signed = judger.judge(result["cropped_mask"])
                if is_signed:
                    # plt.imshow(result["cropped_mask"], interpolation='nearest')
                    plt.gray()
                    plt.imsave('temp/signature.png', result["cropped_mask"])
                    break
            if is_signed:
                print("File is signed")
                break
        if is_signed == False:
            # gives blank if there is no signs
            img = np.zeros([400,400,3],dtype=np.uint8)
            img.fill(255) # or img[:] = 255
            plt.imsave('temp/signature.png', img)  
        # plt.show()
    except Exception as e:
        print(e)

    # Cropping parameters
    img = Image.open('temp/signature.png')
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
        os.remove("temp/masked.png")
        os.remove("temp/resized.png")
        os.remove("temp/signature.png")
        os.remove("temp/formatgrey.png")
        #os.remove(filename)
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
