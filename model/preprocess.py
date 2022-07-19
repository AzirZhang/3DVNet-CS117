import pickle
import numpy as np
import cv2
import glob
import io

import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

import whatimage
import pyheif
from skimage.exposure import rescale_intensity
from PIL import Image
import matplotlib.pyplot as plt


def calibrate(calibimgfiles):
    # checkerboard coordinates in 3D
    objp = np.zeros((7*7, 3), np.float32)
    objp[:, :2] = 2.8*np.mgrid[0:7, 0:7].T.reshape(-1,2)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(calibimgfiles)
    # masks = glob.glob(maskfiles)
    # print(images)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        # print(fname)
        # cv2.imwrite(f'./data/test/img_{idx}.png', img)
        # mask = cv2.imread(masks[idx], cv2.IMREAD_UNCHANGED)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # img[mask == 0] = 0
        # cv2.imwrite(f'./data/test/mask_{idx}.png', mask)



        ####################################
        # following codes are from
        # https://stackoverflow.com/questions/66225558/cv2-findchessboardcorners-fails-to-find-corners

        lwr = np.array([0, 0, 143])
        upr = np.array([179, 61, 252])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        msk = cv2.inRange(hsv, lwr, upr)
#
        krn = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
        dlt = cv2.dilate(msk, krn, iterations=5)
        res = 255 - cv2.bitwise_and(dlt, msk)
#
        img = np.uint8(res)


        #########

        cv2.imwrite(f'./data/test/img_{idx}.png', img)
        img_size = (img.shape[1], img.shape[0])
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # cv2.imwrite(f'./data/test/{idx}.png', img)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(img, (7,7), None)  # flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                                    # cv2.CALIB_CB_FAST_CHECK +
                                                                    # cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points
        # print(ret)
        # print(corners.shape)
        if ret == True:
            print(idx)
            objpoints.append(objp)
            imgpoints.append(corners)

            # Display image with the corners overlayed
            cv2.drawChessboardCorners(img, (7,7), corners, ret)
            cv2.waitKey(500)

    # cv2.destroyAllWindows()
    # a = input()
    # now perform the calibration
    print("Start Calibration")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    print("Estimated camera intrinsic parameter matrix K")
    print(K)
    print("Estimated radial distortion coefficients")
    print(dist)

    print("Individual intrinsic parameters")
    print("fx = ", K[0][0])
    print("fy = ", K[1][1])
    print("cx = ", K[0][2])
    print("cy = ", K[1][2])


    # save the results out to a file for later use
    calib = {}
    calib["f"] = (K[0][0] + K[1][1])/2
    calib["c"] = torch.tensor([[K[0][2]], [K[1][2]]])
    calib["dist"] = dist
    calib["K"] = K

    return calib


def remove_rd(in_dir, out_dir, calib):
    images = glob.glob(in_dir)
    out_images = glob.glob(out_dir)
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        # img_size = (img.shape[1], img.shape[0])

        dst = cv2.undistort(img, calib['K'], calib['dist'], None, calib['K'])
        cv2.imwrite(out_images[idx], dst)


def heic_to_png(in_dir, out_dir):
    img = pyheif.read(in_dir)
    pi = Image.frombytes(mode=img.mode, size=img.size, data=img.data)
    pi.save(out_dir, format="png")


def remove_green_screen(in_dir, mask_dir, resize=True):
    '''
    Edited from the codes: https://stackoverflow.com/a/72280828
    '''
    img = cv2.imread(in_dir)
    if resize:
        img = crop_and_resize(img)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = lab[:, :, 1]
    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    blur = cv2.GaussianBlur(thresh, (0, 0), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)
    mask = rescale_intensity(blur, in_range=(127.5, 255), out_range=(0, 255)).astype(np.uint8)
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    # save output
    cv2.imwrite(mask_dir, mask)
    cv2.imwrite(in_dir, result)
    cv2.waitKey(0)
    # image = Image.open(in_dir)
    # img_tensor = transforms.ToTensor()(image)
    # save_image(img_tensor, in_dir)

    # image = Image.open(mask_dir)
    # img_tensor = transforms.ToTensor()(image)
    # save_image(img_tensor, mask_dir)


def crop_and_resize(img):
    img = img[126:3654, :]
    img = cv2.resize(img, (640, 512), interpolation=cv2.INTER_NEAREST)

    return img


if __name__ == '__main__':
    # for i in range(5):
    #     heic_to_png(f'./raw_data/object/{i}.HEIC', f'./data/object/{i}.png')
    #     remove_green_screen(f'./data/object/{i}.png', f'./data/mask/{i}.png')
    calib = calibrate('./raw_data/chessboard/*.png')
    # remove_rd('./raw_data/chessboard/*.png', './data/chessboard/*.png', calib)
    # remove_rd('./data/object/*.png', './data/object/*.png', calib)
    # remove_rd('./data/mask/*.png', './data/mask/*.png', calib)
    print(calib)
    # param = np.load(".\\calibration.pickle", allow_pickle=True)