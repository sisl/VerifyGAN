import numpy as np
import random
import time
import mss
import cv2
import os

import xpc3
import xpc3_helper
import xpc3_helper_SK

OUT_DIR = "/scratch/smkatz/NASA_ULI/data_GAN_focus_area/" # Make sure this already exists

# 0=Clear, 1=Cirrus, 2=Scattered, 3=Broken, 4=Overcast (higher numbers are cloudier/darker)
CLOUD_TYPE = 2

TIME_OF_DAY = 9.0 # 9am

# Constant cross track error
CTE = 4

# Number of images to grab
NUM_POINTS_CTE = 50
NUM_POINTS_DTP = 200

def main():
    with xpc3.XPlaneConnect() as client:
    	genDataFocusArea(client, OUT_DIR, NUM_POINTS_CTE, NUM_POINTS_DTP)


def genDataFocusArea(client, outDir, num_points_cte, num_points_dtp):
    # Reset simulator
    xpc3_helper.reset(client)

    client.sendDREF("sim/time/zulu_time_sec", TIME_OF_DAY*3600+8*3600)
    client.sendDREF("sim/weather/cloud_type[0]", CLOUD_TYPE)

    # Give a few seconds to get terminal out of the way
    time.sleep(5)

    xpc3_helper.reset(client)

    # Screenshot parameters
    screenShot = mss.mss()
    monitor = {'top': 100, 'left': 100, 'width': 1720, 'height': 960}
    width = 360  # For cropping
    height = 200  # For cropping

    states = []
    csv_file = outDir + 'errors.csv'
    with open(csv_file, 'w') as fd:
        fd.write("filename,CTE,HE,DTP\n")

    # Generate random downtrack positions for data
    dtps = [random.uniform(322, 522) for _ in range(num_points_dtp)]
    dtps.sort()

    idx = 0
    # Go to those positions and take screenshots
    for dtp in dtps:
        # Sample ctes
        ctes = [random.uniform(-11, 11) for _ in range(num_points_cte)]
        ctes.sort()
        for cte in ctes:
            # Sample he
            he = random.uniform(-30, 30)

            # Get to the new position and heading
            xpc3_helper_SK.setHomeXYhe(client, cte, dtp, he)

            s = np.array([cte, he, dtp])
            states.append(s)

            # Pause to allow time for it to get there before taking screenshot
            time.sleep(0.5)

            # Take and save screenshot
            img = cv2.cvtColor(np.array(screenShot.grab(monitor)),
                            cv2.COLOR_BGRA2BGR)[230:, :, :]
            img = cv2.resize(img, (width, height))
            cv2.imwrite('%s%d.png' % (outDir, idx), img)
            idx += 1

            # Save state to the CSV file
            with open(csv_file, 'a') as fd:
                fd.write("%d,%f,%f,%f\n" %
                        (idx, states[-1][0], states[-1][1], states[-1][2]))

def genDataCTE(client, outDir, cte, num_points):
    # Reset simulator
    xpc3_helper.reset(client)
    
    client.sendDREF("sim/time/zulu_time_sec",TIME_OF_DAY*3600+8*3600) 
    client.sendDREF("sim/weather/cloud_type[0]",CLOUD_TYPE)
    
    # Give a few seconds to get terminal out of the way
    time.sleep(5)
    
    xpc3_helper.reset(client)

    num_points_1 = int(num_points / 2)
    num_points_2 = num_points - num_points_1

    # Screenshot parameters
    screenShot = mss.mss()
    monitor = {'top':100, 'left': 100, 'width':1720, 'height':960}
    width = 360 # For cropping
    height = 200 # For cropping

    # Generate random downtrack positions for data
    dtps_1 = [random.uniform(0, 1340) for _ in range(num_points_1)]
    dtps_2 = [random.uniform(1540, 2900) for _ in range(num_points_2)]
    dtps = dtps_1 + dtps_2
    dtps.sort()

    idx = 0
    # Go to those positions and take screenshots
    for dtp in dtps:
        # Get to the new position
        xpc3_helper_SK.setHomeXY(client, cte, dtp)
        # Pause to allow time for it to get there before taking screenshot
        time.sleep(0.5)

        # Take and save screenshot
        img = cv2.cvtColor(np.array(screenShot.grab(monitor)),
			cv2.COLOR_BGRA2BGR)[230:, :, :]
        img = cv2.resize(img,(width,height))
        cv2.imwrite('%s%d.png'%(outDir,idx),img)
        idx += 1

if __name__ == "__main__":
	main()
