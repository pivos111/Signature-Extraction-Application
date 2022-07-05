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


# load input
path = sys.argv[2]
name, extension= os.path.splitext(path)

# convert to png
filename = "fullpdf.png"
if extension == ".pdf":
    images = convert_from_bytes(open(path,'rb').read(),poppler_path="bin")
    for i, image in enumerate(images):
        image.save(filename, "PNG")
        image.save('fullPdfResize.png',"PNG")
        # print("it was pdf")
elif extension == ".jpg" or extension == ".png":
    image = Image.open(path)
    image.save(filename, "PNG")
    image.save('fullPdfResize.png',"PNG")
    # print("it was image")
else:
    print("file isn't .pdf .jpg or .png")

# Load image and HSV color threshold to cut non handwritten text
image = cv2.imread(filename)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([90, 38, 0])
upper = np.array([145, 255, 255])
mask = cv2.inRange(hsv, lower, upper)
result = cv2.bitwise_and(image, image, mask=mask)
result[mask==0] = (255, 255, 255)
plt.imsave('masked.png', result)

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
    masks = loader.get_masks("masked.png")
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
                plt.imsave('signature.png', result["cropped_mask"])
                break
        if is_signed:
            break
    if is_signed == False:
        # gives blank if there is no signs
        img = np.zeros([400,400,3],dtype=np.uint8)
        img.fill(255) # or img[:] = 255
        plt.imsave('signature.png', img)
    # print("File is", is_signed)
    # plt.show()
except Exception as e:
    print(e)

# Cropping parameters
img = Image.open('signature.png')
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
    os.remove("masked.png")
    os.remove("resized.png")
    os.remove("formatgrey.png")
    os.remove("signature.png")
    os.rename('formatblack-and-white.png','signature.png')
except Exception as e:
    print(e)
    
img = Image.open('fullPdfResize.png')
baseheight=1000
hpercent = (baseheight / float(img.size[1]))
wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((wsize, baseheight), Image.ANTIALIAS)

img.save('fullPdfResize.png')