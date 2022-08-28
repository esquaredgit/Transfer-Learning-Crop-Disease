import sys
import cv2

def smallestSquare(image, point1, point2):
    points = [point1, point2]
    points.sort(key=lambda p: p[0])
    L, R = points
    points.sort(key=lambda p: p[1])
    T, B = points
    xlen = R[0]-L[0]
    ylen = B[1]-T[1]
    new1 = None
    new2 = None
    # If the x side is longer than the y side
    if xlen > ylen:
        # get amount of pixels to be added to each side
        total = xlen-ylen
        half = int(total/2)
        # If we're going to go outside the top boundary of the photo by extending the crop
        if T[1]-half < 0:
            top = T[1]
            bottom = total-top
            new1 = [T[0], T[1]-top]
            new2 = [B[0], B[1]+bottom]
            return new1, new2
        # If we're going to go outside the bottom boundary
        elif B[1]+half > len(image):
            bottom = len(image)-1-B[1]
            top = total-bottom
            new1 = [T[0], T[1]-top]
            new2 = [B[0], B[1]+bottom]
            return new1, new2
        # If we're in the clear
        else: 
            new1 = [T[0], T[1]-half]
            new2 = [B[0], B[1]+half]
            return new1, new2
    # If the y side is longer than the x side
    elif xlen < ylen:
        total = ylen-xlen
        half = int(total/2)
        # If we're going to go outside the left boundary
        if L[0]-half < 0:
            left = L[0]
            right = total-left
            new1 = [L[0]-left, L[1]]
            new2 = [R[0]+right, R[1]]
            return new1, new2
        # If we're going to go outside the right boundary
        elif R[0]+half > len(image[0]):
            right = len(image[0])-R[0]
            left = total-right
            new1 = [L[0]-left, L[1]]
            new2 = [R[0]+right, R[1]]
            return new1, new2
        # If we're in the clear
        else: 
            new1 = [L[0]-half, L[1]]
            new2 = [R[0]+half, R[1]]
            return new1, new2
    # if it's already a square
    else: 
        return point1, point2



# Transforming OSF pics according to annotations
# Importing annotations
with open("annotations_handheld.csv") as file:
    annotations = file.readlines()
    del annotations[0]
    for i in range(0, len(annotations)):
        temp = annotations[i].replace("\n","").split(",")
        points = [temp[j] for j in range(0,5)]
        for j in range(1, len(points)):
            points[j] = int(points[j])
        annotations[i] = points

# Transforming images
def get_points(filename):
    global annotations
    all = []
    indexes = []
    for i in range(len(annotations)):
        if annotations[i][0] == filename:
            points = annotations[i][1:]
            all.append(points)
            indexes.append(i)
            continue
        else:
            continue
    annotations = [annotations[i] for i in range(len(annotations)) if not(i in indexes)]
    return all

# DEBUGGING CODE:
img = cv2.imread("/Users/ethaneldridge/Documents/CSCI-P556/final_project/training_images/osf/DSC00202.JPG")
print(len(img), len(img[0]))
crops = get_points("DSC00202.JPG")
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
count = 0
print(crops)
for c in crops:
    count+=1
    x1, y1, x2, y2 = c
    square = smallestSquare(img, (x1,y1), (x2,y2))
    xs = [square[0][0], square[1][0]]
    xs.sort()
    ys = [square[0][1], square[1][1]]
    ys.sort()
    print(xs)
    print(ys)
    img = cv2.circle(img, [x1, y1], 50, (0,0,255), thickness=-1)
    img = cv2.circle(img, [x2, y2], 50, (0,0,255), thickness=-1)

    cropped = img[ys[0]:ys[1], xs[0]:xs[1]]
    cv2.imshow("crop", img)
    cv2.waitKey(0)

cv2.imshow("img", img)
cv2.waitKey(0)
print(smallestSquare(img, p1, p2))

# IMAGE PROCESSING
# Cropping images
import os
dir = '/Users/ethaneldridge/Documents/CSCI-P556/final_project/training_images/osf'
path = "/Users/ethaneldridge/Documents/CSCI-P556/final_project/training_images/osfSquare"

healthyCount = 0
progCount = 0
total = len(os.listdir(dir))
bads = []
for filename in os.listdir(dir):
    progCount+=1
    print(round((progCount/total)*100,2))
    # print(filename)
    crops = get_points(filename)
    count = 0
    for c in crops:
        count+=1
        x1, y1, x2, y2 = c
        if x1==0 and y1==0: 
            healthyCount += 1
            # print(x1, y1, x2, y2, end=" (PASSED)\n")
            continue
        # print(x1, y1, x2, y2)
        img = cv2.imread(os.path.join(dir, filename))
        if (x1 > len(img[0])) or (x2 > len(img[0])) or (y1 > len(img)) or (y2 > len(img)):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            bads.append(filename)
        square = smallestSquare(img, (x1,y1), (x2,y2))
        xs = [square[0][0], square[1][0]]
        xs.sort()
        ys = [square[0][1], square[1][1]]
        ys.sort()
        # print(xs)
        # print(ys)
        cropped = img[ys[0]:ys[1], xs[0]:xs[1]]
        prefix, suffix = filename.split(".")
        cv2.imwrite(os.path.join(path, prefix+"_"+str(count)+"."+suffix), cropped)
    # print()
print(healthyCount)
print(bads)
sys.exit()
