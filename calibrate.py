import argparse
import glob
import numpy as np
import cv2

"""
    This code is based on the below tutorial
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
"""

parser = argparse.ArgumentParser(description='Simple canera calibration script.')
parser.add_argument('--images', type=str, help='Directory path of images', required = True)
parser.add_argument('--width', type=int, help='# of shells of a checker board', required = True)
parser.add_argument('--height', type=int, help='# of shells of a checker board', required = True)

args = parser.parse_args()

images = glob.glob("{}/*.png".format(args.images))

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
num_width = args.width
num_height = args.height

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((num_width*num_height,3), np.float32)
objp[:,:2] = np.mgrid[0:num_width,0:num_height].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for fn in images:
    img = cv2.imread(fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (num_width,num_height), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (num_width,num_height), corners2, ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)
print("Camera Matrix")
print(mtx)
print("Dist Coeffs")
print(dist)
# undistorted examples
for fn in images:
    img = cv2.imread(fn)
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('undistorted',dst)
    cv2.waitKey(500)
