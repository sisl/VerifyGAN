import xpc3
import xpc3_helper

import time

def rotateToHome(x, y):
    rotx = 0.583055934597441 * x + 0.8124320138514389 * y
    roty = -0.8124320138514389 * x + 0.583055934597441 * y
    return rotx, roty

def rotateToLocal(x, y):
    rotx = 0.583055934597441 * x + -0.8124320138514389 * y
    roty = 0.8124320138514389 * x + 0.583055934597441 * y
    return rotx, roty

def getHomeXY(client):
    x = client.getDREF("sim/flightmodel/position/local_x")[0]
    y = client.getDREF("sim/flightmodel/position/local_z")[0]

    # Translate to make start x and y the origin
    startX, startY = xpc3_helper.getStartXY()
    transx = startX - x
    transy = startY - y

    # Rotate to align runway with y axis
    rotx, roty = rotateToHome(transx, transy)
    return rotx, roty

def homeToLocal(x, y):
    # Rotate back
    rotx, roty = rotateToLocal(x, y)

    # Translate back
    startX, startY = xpc3_helper.getStartXY()
    transx = startX - rotx
    transy = startY - roty

    return transx, transy 

def setHomeXY(client, x, y):
    localx, localz = homeToLocal(x, y)

    client.sendDREF("sim/flightmodel/position/local_x",localx)
    client.sendDREF("sim/flightmodel/position/local_z",localz)

    # Place perfectly on the ground
    curr_agly = client.getDREF("sim/flightmodel/position/y_agl")[0]
    curr_localy = client.getDREF("sim/flightmodel/position/local_y")[0]
    client.sendDREF("sim/flightmodel/position/local_y", curr_localy - curr_agly)

def setHomeXYhe(client, x, y, he):
    localx, localz = homeToLocal(x, y)

    client.sendDREF("sim/flightmodel/position/local_x", localx)
    client.sendDREF("sim/flightmodel/position/local_z", localz)
    client.sendDREF("sim/flightmodel/position/psi", 53.7 - he)

    # Place perfectly on the ground
    curr_agly = client.getDREF("sim/flightmodel/position/y_agl")[0]
    curr_localy = client.getDREF("sim/flightmodel/position/local_y")[0]
    client.sendDREF("sim/flightmodel/position/local_y",
                    curr_localy - curr_agly)

def driveDownCenter(client):
    setHomeXY(client, 0.0, 0.0)
    for y in range(0, 500):
        setHomeXY(client, 0.0, y)
        time.sleep(0.1)
        if y % 5 == 0:
            print(y)


def driveDownFocusArea(client):
    setHomeXY(client, 0.0, 0.0)
    for y in range(322, 522):
        setHomeXY(client, 0.0, y)
        time.sleep(0.1)
        if y % 5 == 0:
            print(y)
