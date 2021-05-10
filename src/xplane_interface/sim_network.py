from xpc3 import *
from xpc3_helper import *
from xpc3_helper_SK import *
from nnet import *
from PIL import Image

import numpy as np
import h5py
import time
import mss
import cv2
import os

filename = "../../models/KJ_TaxiNet.nnet"
network = NNet(filename)

### IMPORTANT PARAMETERS FOR IMAGE PROCESSING ###
stride = 16             # Size of square of pixels downsampled to one grayscale value
numPix = 16             # During downsampling, average the numPix brightest pixels in each square
width  = 256//stride    # Width of downsampled grayscale image
height = 128//stride    # Height of downsampled grayscale image

screenShot = mss.mss()
monitor = {'top': 100, 'left': 100, 'width': 1720, 'height': 960}
screen_width = 360  # For cropping
screen_height = 200  # For cropping

OUTFILE = "/scratch/smkatz/NASA_ULI/controller_detailed_run_v2.h5"

def getCurrentImage():
    time.sleep(1)
    # Get current screenshot
    img = cv2.cvtColor(np.array(screenShot.grab(monitor)), cv2.COLOR_BGRA2BGR)[230:,:,:]
    img = cv2.resize(img,(screen_width,screen_height))
    img = img[:,:,::-1]
    img = np.array(img)
    orig = np.copy(img)

    # Convert to grayscale, crop out nose, sky, bottom of image, resize to 256x128, scale so 
    # values range between 0 and 1
    img = np.array(Image.fromarray(img).convert('L').crop((55, 5, 360, 135)).resize((256, 128)))/255.0

    # Downsample image
    # Split image into stride x stride boxes, average numPix brightest pixels in that box
    # As a result, img2 has one value for every box
    img2 = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            img2[i,j] = np.mean(np.sort(img[stride*i:stride*(i+1),stride*j:stride*(j+1)].reshape(-1))[-numPix:])

    # Ensure that the mean of the image is 0.5 and that values range between 0 and 1
    # The training data only contains images from sunny, 9am conditions.
    # Biasing the image helps the network generalize to different lighting conditions (cloudy, noon, etc)
    img2 -= img2.mean()
    img2 += 0.5
    img2[img2>1] = 1
    img2[img2<0] = 0
    return orig, img2.flatten()

def dynamics(x, y, theta, phi_deg, dt = 0.05, v = 5, L = 5):
    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi_deg)

    x_dot = v * np.sin(theta_rad)
    y_dot = v * np.cos(theta_rad)
    theta_dot = (v / L) * np.tan(phi_rad)

    x_prime = x + x_dot * dt
    y_prime = y + y_dot * dt
    theta_prime = theta + np.rad2deg(theta_dot) * dt

    return x_prime, theta_prime, y_prime


def rollout(client, network, num_steps = 500, dt = 0.05, ctrl_every = 20):
    cte, he = getErrors(client)
    _, dtp = getHomeXY(client)

    print("Performing Rollout Simulation...\n")

    phi_deg = 0.0

    num_control = int(num_steps / ctrl_every - 1)

    xs = np.zeros(num_steps + 1)
    pred_xs = np.zeros(num_control + 1)
    ys = np.zeros(num_steps + 1)
    thetas = np.zeros(num_steps + 1)
    pred_thetas = np.zeros(num_control + 1)

    orig_images = np.zeros((num_steps + 1, 200, 360, 3))
    downsampled_images = np.zeros((num_steps + 1, 128))

    xs[0] = cte
    ys[0] = dtp
    thetas[0] = he

    for i in range(num_steps):
        orig, image = getCurrentImage()
        if i % ctrl_every == 0:
            # Get network prediction and control
            #image = getCurrentImage()
            pred = network.evaluate_network(image)
            cte_pred = pred[0]
            he_pred = pred[1]

            pred_xs[int(i / ctrl_every)] = cte_pred
            pred_thetas[int(i / ctrl_every)] = he_pred

            phi_deg = -0.74 * cte_pred - 0.44 * he_pred # Steering angle

        # Get next states and go to them
        cte, he, dtp = dynamics(cte, dtp, he, phi_deg, dt)
        setHomeXYhe(client, cte, dtp, he)
        # time.sleep(0.05)
        # print("cte: ", cte, "\n")
        # print("dtp: ", dtp, "\n")
        # print("he: ", he, "\n")

        xs[i + 1] = cte
        ys[i + 1] = dtp
        thetas[i + 1] = he
        orig_images[i, :, :, :] = orig
        downsampled_images[i, :] = image

        time.sleep(0.05)
        # if i % ctrl_every == 0:
        #     setHomeXYhe(client, cte, dtp, he)
        #     time.sleep(0.5)

    return xs, ys, thetas, pred_xs, pred_thetas, orig_images, downsampled_images

client = xpc3.XPlaneConnect()
xpc3_helper.reset(client)
time.sleep(1)

TIME_OF_DAY = 9.0 # am
client.sendDREF("sim/time/zulu_time_sec", TIME_OF_DAY*3600+8*3600)

start_cte = 8.0
start_he = 0.0

setHomeXYhe(client, start_cte, 322.0, start_he)
time.sleep(3)

xs, ys, thetas, pred_xs, pred_thetas, orig_images, downsampled_images = rollout(client, network, 
                                                        num_steps = 500, dt = 0.05, ctrl_every = 20)

# xmat = np.zeros((9, 501))
# ymat = np.zeros((9, 501))
# thetamat = np.zeros((9, 501))

# #start_ctes = [8.0, 6.0, 4.0, 2.0, 0.0, -2.0, -4.0, -6.0, -8.0]
# start_ctes = [6.0]

# ind = 0
# for start_cte in start_ctes:
#     setHomeXYhe(client, start_cte, 322.0, start_he)
#     time.sleep(3)
#     xs, ys, thetas = rollout(client, network)
#     xmat[ind,:] = xs
#     ymat[ind,:] = ys
#     thetamat[ind,:] = thetas
#     ind += 1

# with h5py.File(OUTFILE, 'w') as f:
#     f.create_dataset('xs', data = xmat)
#     f.create_dataset('ys', data = ymat)
#     f.create_dataset('thetas', data = thetamat)

with h5py.File(OUTFILE, 'w') as f:
    f.create_dataset('xs', data = xs)
    f.create_dataset('ys', data = ys)
    f.create_dataset('thetas', data = thetas)
    f.create_dataset('pred_xs', data = pred_xs)
    f.create_dataset('pred_thetas', data = pred_thetas)
    f.create_dataset('orig_images', data = orig_images)
    f.create_dataset('downsampled_images', data = downsampled_images)