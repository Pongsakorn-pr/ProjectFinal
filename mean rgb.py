import cv2
from pathlib import Path
import numpy as np
from rembg import remove
import PIL
from tabulate import tabulate
import os

def get_rgb(path):
    # Read image
   img = cv2.imread(path)
   hh, ww = img.shape[:2]

   # threshold on white
   # Define lower and uppper limits
   lower = np.array([95,0,0])
   upper = np.array([255, 90,90])
   # Create mask to only select black
   thresh = cv2.inRange(img, lower, upper)

   # apply morphology
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
   morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

   # get contours
   contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   contours = contours[0] if len(contours) == 2 else contours[1]

   # draw white contours on black background as mask
   mask = np.zeros((hh,ww), dtype=np.uint8)
   for cntr in contours:
      cv2.drawContours(mask, [cntr], 0, (255,255,255), -1)

   # get convex hull
   points = np.column_stack(np.where(thresh.transpose() > 0))
   hullpts = cv2.convexHull(points)
   ((centx,centy), (width,height), angle) = cv2.fitEllipse(hullpts)

   # draw convex hull on image
   hull = img.copy()
   cv2.polylines(hull, [hullpts], True, (0,0,255), 1)

   # create new circle mask from ellipse 
   circle = np.zeros((hh,ww), dtype=np.uint8)
   cx = int(centx)
   cy = int(centy)
   radius = ((width-300)+(height-400))/4
   cv2.circle(circle, (cx,cy), int(radius), 255, -1)

   # erode circle a bit to avoid a white ring
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
   circle = cv2.morphologyEx(circle, cv2.MORPH_ERODE, kernel)

   # combine inverted morph and circle
   mask2 = cv2.bitwise_and(255-morph, 255-morph, mask=circle)

   # apply mask to image
   result = cv2.bitwise_and(img, img, mask=mask2)
   result = cv2.mean(img,mask=mask2)
   avgR = result[2]
   avgG = result[1]
   avgB = result[0]
   print(avgR,avgG,avgB)
   return avgR,avgG,avgB

images = Path("./imagess").glob("*.jpg")
image_strings = [str(p) for p in images]
data = []
r = ''
g = ''
b = ''
for i in range(len(image_strings)):
    img = cv2.imread(image_strings[i])
    img = remove(img)
    cv2.imwrite('image.png',img)
    avgR, avgG, avgB = get_rgb('image.png')
    data.append([os.path.basename(image_strings[i]), avgR, avgG, avgB])
    result = tabulate(data, headers=['img_name','R', 'G', 'B'], tablefmt='fancy_grid', showindex="always")
    r += str(avgR) + '\n'
    g += str(avgG) + '\n'
    b += str(avgB) + '\n'
    print(str(i)+'/'+ str(len(image_strings)))
    print(image_strings[i])
    print(data)
with open('rgb.txt', 'w', encoding="utf-8") as f:
        f.write(result)
os.startfile("rgb.txt", "print") 
with open('r.txt', 'w', encoding="utf-8") as a:
        a.write(r)
os.startfile("r.txt", "print")   
with open('g.txt', 'w', encoding="utf-8") as c:
        c.write(g)
os.startfile("g.txt", "print") 
with open('b.txt', 'w', encoding="utf-8") as d:
        d.write(b)
os.startfile("b.txt", "print")         
print(result)