import os
import sys
import cv2
import imagehash
import numpy as np
import mysql.connector
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity

# load input
path = "SQL_DB/test2.png"
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

    # view result
    # cv2.imshow('CENTERED', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save resulting centered image in DB
    cv2.imwrite('temp/formatgrey.png', result)
    #read image
    img_grey = cv2.imread('temp/formatgrey.png', cv2.IMREAD_GRAYSCALE)

    # define a threshold, 128 is the middle of black and white in grey scale
    thresh = 200

    # threshold the image
    img_binary = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)[1]

    #save image
    cv2.imwrite('temp/Comparison.png',img_binary)

    try:
        os.remove("temp/background1.png")
        os.remove("temp/resized.png")
		os.remove("temp/formatgrey.png")
    except Exception as e:
        print(e)

# Take full DB

grammes = 0 # rows of DB

try:
  connection = mysql.connector.connect(host="localhost",
                                      database='signatures',
                                      user="root",
                                      password="")

  sql_select_query = """select * from signatures"""

  cursor = connection.cursor(buffered=True)
  cursor.execute(sql_select_query)
  connection.commit()
  # print(cursor.fetchone()[1])
  records = cursor.fetchall()
  print(cursor.rowcount, "Record selected successfully into signatures table")
  grammes = cursor.rowcount
  cursor.close()

except mysql.connector.Error as error:
  print("Failed to select record into signatures table {}".format(error))

finally:
  if connection.is_connected():
    connection.close()
    print("MySQL connection is closed")

# Check all images in db

topresults = 5 # top x to keep
image_from_User = "temp/Comparison.png"
# maxscores = [0] *topresults
# maxafimi = ["" for x in range(topresults)]
# maxpaths = ["" for x in range(topresults)]
listamax = [[0 for i in range(3)] for j in range(grammes)] # for top x

counte = 0
for row in records:
    
  image_from_DB = row[1] #Each image in db

  # --- Method 1: imagehash ---

  hash0 = imagehash.average_hash(Image.open(image_from_User)) 
  hash1 = imagehash.average_hash(Image.open(image_from_DB)) 
  cutoff = 15 # maximum bits that could be different between the hashes. 

  if hash0 - hash1 < cutoff:
    print('images are similar')
  else:
    print('images are not similar')

  # --- Method 2: SSIM ---

  before = cv2.imread(image_from_User)
  after = cv2.imread(image_from_DB)

  # Convert images to grayscale
  before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
  after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

  # Compute SSIM between two images
  (score, diff) = structural_similarity(before_gray, after_gray, full=True)
  print("Image similarity", score)

  # The diff image contains the actual image differences between the two images
  # and is represented as a floating point data type in the range [0,1] 
  # so we must convert the array to 8-bit unsigned integers in the range
  # [0,255] before we can use it with OpenCV
  diff = (diff * 255).astype("uint8")

  # Threshold the difference image, followed by finding contours to
  # obtain the regions of the two input images that differ
  thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
  contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]

  mask = np.zeros(before.shape, dtype='uint8')
  filled_after = after.copy()

  for c in contours:
      area = cv2.contourArea(c)
      if area > 40:
          x,y,w,h = cv2.boundingRect(c)
          cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
          cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
          cv2.drawContours(mask, [c], 0, (0,255,0), -1)
          cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

  # cv2.imshow('before', before)
  # cv2.imshow('after', after)
  # cv2.imshow('diff',diff)
  # cv2.imshow('mask',mask)
  # cv2.imshow('filled after',filled_after)
  # cv2.waitKey(0)
  

  scoretxt = int(score * 100)
  print("score is :", scoretxt)
  listamax[counte][0] = scoretxt #score
  listamax[counte][1] = row[1] #path
  listamax[counte][2] = row[0] #afimi
 
  counte += 1
  #check for better score
  # for i in range(topresults):
  #   if(scoretxt > maxscores[topresults-1-i]):
      
  #     # shift all cells left
  #     upperlimit = topresults-1-i
  #     for j in range(upperlimit):
  #       if j > 0:
  #         maxscores[j-1] =  maxscores[j]
  #         maxafimi[j-1] = maxafimi[j]
  #         maxpaths[j-1] = maxpaths[j]

  #     # add new bigger value
  #     maxscores[topresults-1-i] = scoretxt
  #     maxafimi[topresults-1-i] = row[0]
  #     maxpaths[topresults-1-i] = row[1]
    
maxx = sorted(listamax, key=lambda x : x[0])

# with open("results.txt", "w") as file:
#     file.write(scoretxt +"|"+ image_from_DB +"|"+ afimi)

f = open("results.txt", "w")   # 'r' for reading and 'w' for writing
for i in range(1,6):
  if i < 5:
    f.write(str(maxx[grammes-i][0]) +"% | Path: "+ maxx[grammes-i][1] +"| AFIMI: "+ str(maxx[grammes-i][2])+"\n")    # creating results
  else:
    f.write(str(maxx[grammes-i][0]) +"% | Path: "+ maxx[grammes-i][1] +"| AFIMI: "+ str(maxx[grammes-i][2]))
f.close()  


filename = "temp/results.txt"

with open(filename) as f_input:
    data = f_input.read().rstrip('\n')

with open(filename, 'w') as f_output:    
    f_output.write(data)

try:
  os.remove("temp/Comparison.png")
except Exception as e:
  print(e)