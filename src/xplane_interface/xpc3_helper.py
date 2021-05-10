import numpy as np
import xpc3
import time
import pandas as pd

def sendCTRL(client, elev, aileron, rudder, throttle):
    client.sendCTRL([elev,aileron,rudder,throttle])

def sendBrake(client, brake):
    client.sendDREF("sim/flightmodel/controls/parkbrake",brake)

def getStartPosition(cte=0.0, he=0.0, dr=0.0):
    lat, lon = getStartCoord()
    alt=362.14444; heading=53.7;pitch=0.31789625; roll=0.10021035;
    dlat = -0.026/3620; dlon = 0.0285/3620
    dlatDR = -0.027/3620/1.4151; dlonDR = -0.055/3620/1.4151
    lat -= cte*dlat; lon -= cte*dlon
    lat -= dr*dlatDR; lon -= dr*dlonDR
    heading -= he
    alt -= 0.001*dr
    return lat, lon, alt, pitch, roll, heading

def getStartCoord():
    return 47.196890, -119.33260

def getStartXY():
    return -25159.26953125, 33689.8125
    #return -24968.23828125, 33553.0234375

def getEndXY():
    return -22736.160390625, 31950.826171875

def getEndCoord():
    return 47.2126, -119.30066

def coordToXY(lat,lon):
    startLat,startLon = getStartCoord()
    y = 111111*(lat-startLat)
    x = 111111*np.cos(startLat/180.0*np.pi)*(lon-startLon)
    return x,y

def getPercDownRunway(lat,lon):
    x,y = coordToXY(lat,lon)
    endLat, endLon = getEndCoord()
    endX, endY = coordToXY(endLat,endLon)
    return np.sqrt(x*x+y*y)/np.sqrt(endX*endX+endY*endY)*100

def getMetersDownRunway(lat,lon):
    return (getPercDownRunway(lat,lon)-7.87)*2960/100

def getErrors(client):
    lat,lon,alt,pitch,roll,yaw,brk = client.getPOSI()
    cte = getCTE(client)
    he = getHE(yaw)
    return np.array([cte,he])

def setCTE(client, cte):

    endX, endY = getEndXY()
    startX, startY = getStartXY()

    x = client.getDREF("sim/flightmodel/position/local_x")[0]
    y = client.getDREF("sim/flightmodel/position/local_z")[0]

    endX -= startX; endY -= startY; x -= startX; y -= startY
    dist = np.abs(endY*x-endX*y)/np.sqrt(endX*endX+endY*endY)
    sgn = 1.0-2*(np.arctan2(y,x)<np.arctan2(endY,endX))
    cteOld = -dist*sgn
    alpha = np.arctan2(endY,endX)

    # Remove old CTE attempt
    xCenter = x + startX - np.sin(alpha)*cteOld
    yCenter = y + startY + np.cos(alpha)*cteOld

    # Add exact CTE with local_x/local_z
    xCenter += np.sin(alpha)*cte
    yCenter -= np.cos(alpha)*cte

    client.sendDREF("sim/flightmodel/position/local_x",xCenter)
    client.sendDREF("sim/flightmodel/position/local_z",yCenter)

def getCTE(client):
    endX, endY = getEndXY()
    startX, startY = getStartXY()
    x = client.getDREF("sim/flightmodel/position/local_x")[0]
    y = client.getDREF("sim/flightmodel/position/local_z")[0]
    endX -= startX; endY -= startY; x -= startX; y -= startY
    dist = np.abs(endY*x-endX*y)/np.sqrt(endX*endX+endY*endY)
    sgn = 1.0-2*(np.arctan2(y,x)<np.arctan2(endY,endX))
    cteNew = dist*sgn
    return -cteNew #cteNom

def getHE(heading):
    endX, endY = getEndXY()
    startX, startY = getStartXY()
    endX -= startX; endY -= startY;
    he = heading+np.arctan2(-endX,-endY)*180.0/np.pi
    if he < -180.0:
        he+= 360.0
    elif he > 180.0:
        he -= 360.0
    return -he

def getSpeed(client):
    return client.getDREF("sim/flightmodel/position/groundspeed")[0]

def getBrake(client):
    return client.getDREF("sim/flightmodel/controls/parkbrake")[0]

def getControl(client,angLimit,turn,centerCTE):
    lat,lon,alt,pitch,roll,yaw,brk = client.getPOSI()

    speed = getSpeed(client)
    cte = getCTE(client)
    he = getHE(yaw)

    throttle=0.1
    if speed>5:
        throttle=0.0
    elif speed<3:
        throttle=0.2

    # Amount of rudder needed to go straight, roughly
    rudder  = 0.008 
    if he<angLimit and cte<centerCTE:
        rudder -= turn*1
    elif he>-angLimit and cte>centerCTE:
        rudder += turn*1
        
    sendCTRL(client,0,rudder,rudder,throttle)

def reset(client, atEnd=False, cteInit=0, heInit=0, drInit=0, noBrake=True):
    client.pauseSim(True)
    lat,lon, alt, pitch, roll, heading = getStartPosition(cteInit, heInit, drInit); 
    if atEnd:
        lat,lon = getEndCoord(); 
        alt=350.7371; heading=234.1; pitch=0.009598253294825554; roll=-0.1025998443365097;

    # Turn off joystick "+" mark from screen
    client.sendDREF("sim/operation/override/override_joystick", 1)

    # Zero out control inputs
    sendCTRL(client,0,0,0,0)

    # Set parking brake
    if noBrake:
        sendBrake(client,0)
    else:
        sendBrake(client,1)

    # Zero out moments and forces
    initRef = "sim/flightmodel/position/"
    drefs = []
    refs = ['theta','phi','psi','local_vx','local_vy','local_vz','local_ax','local_ay','local_az',
    'Prad','Qrad','Rrad','q','groundspeed',
    'indicated_airspeed','indicated_airspeed2','true_airspeed','M','N','L','P','Q','R','P_dot',
    'Q_dot','R_dot','Prad','Qrad','Rrad']
    for ref in refs:
        drefs += [initRef+ref]
    values = [0]*len(refs)
    client.sendDREFs(drefs,values)

    # Set position and orientation
    client.sendPOSI([lat,lon,alt,pitch,roll,heading,1],0)

    # Fine-tune position
    # Setting position with lat/lon gets you within 0.3m. Setting local_x, local_z is more accurate)
    setCTE(client, cteInit)

    # Fix the plane if you "crashed" or broke something
    client.sendDREFs(["sim/operation/fix_all_systems"],[1])

    # Set fuel mixture for engine
    client.sendDREF("sim/flightmodel/engine/ENGN_mixt",0.61)

    # Set speed of aircraft to be 5 m/s in current heading direction
    client.sendDREF("sim/flightmodel/position/local_vx",5.0*np.sin(heading*np.pi/180.0))
    client.sendDREF("sim/flightmodel/position/local_vz",-5.0*np.cos(heading*np.pi/180.0))

    # Reset fuel levels
    client.sendDREFs(["sim/flightmodel/weight/m_fuel1","sim/flightmodel/weight/m_fuel2"],[232,232])

def saveState(client, folder, filename='test.csv'):
    """
    Save the current state of the simulator to a CSV file.
    Pulls all relevant DREFs and stores them in a CSV.
    Can use loadState to set simulator back to a saved state.
    """

    # Position/orientation/speeds/turn rates/etc datarefs
    drefs = []
    initRef = "sim/flightmodel/position/"
    refs = ['theta','phi','psi','local_x','local_y','local_z','local_vx','local_vy','local_vz','local_ax','local_ay',
    'local_az','Prad','Qrad','Rrad','q','groundspeed'
    'indicated_airspeed','indicated_airspeed2','M','N','L','P','Q','R','P_dot',
    'Q_dot','R_dot','Prad','Qrad','Rrad']
    for ref in refs:
        drefs += [initRef+ref]

    # Engine related datarefs
    initRef = "sim/flightmodel/engine/"
    refs = ['ENGN_N2_','ENGN_N1_','ENGN_EGT','ENGN_ITT','ENGN_CHT','ENGN_EGT_c','ENGN_ITT_c',
            'ENGN_CHT_C','ENGN_FF_','ENGN_EPR','ENGN_MPR','ENGN_oil_press_psi',
            'ENGN_oil_press','ENGN_oil_press','ENGN_power','ENGN_prop','ENGN_TRQ',
            'ENGN_thro','ENGN_thro_use',
            'POINT_thrust','POINT_tacrad','ENGN_mixt','ENGN_prop','ENGN_propmode']
    for ref in refs:
        drefs += [initRef+ref]

    # Force related dataref
    initRef = "sim/flightmodel/forces/"
    refs = ['fside_prop','fnrml_prop','faxil_prop','L_prop','M_prop','N_prop',
            'L_total','M_total','N_total']
    for ref in refs:
        drefs += [initRef+ref]

    # parking brake, time of day, and fuel level data ref
    drefs += ["sim/flightmodel/controls/parkbrake"]
    drefs += ["sim/time/zulu_time_sec"]
    drefs += ["sim/flightmodel/weight/m_fuel1","sim/flightmodel/weight/m_fuel2"]

    # Get the datarefs
    values = client.getDREFs(drefs)
    valuesFilt = []
    drefsFilt = []
    for i,val in enumerate(values):
        if len(val)>0:
            valuesFilt += [val[0]]
            drefsFilt += [drefs[i]]

    # Get position and controller settings
    valuesFilt += client.getPOSI()
    valuesFilt += client.getCTRL()
    drefsFilt += ["lat","lon","alt","pitch","roll","heading","gear"]
    drefsFilt += ["elev","aileron","rudder","throttle","gear","flaps","speedbrakes"]
    values = np.array(valuesFilt).reshape((1,len(valuesFilt)))

    # Save to CSV
    outData = pd.DataFrame(values,index=[0],columns=drefsFilt)
    csv_file = folder + "/"+filename
    outData.to_csv(csv_file,index=False,index_label=False)


def loadState(client, folder, filename='test.csv'):
    """
    Read csv file of saved simulator state, load the simulator with saved values
    """

    # Read CSV file
    tab = pd.read_csv(folder + "/" +filename)
    drefs = list(tab.columns)
    values = list(tab.values[0])

    # Separate out values
    pos = values[-14:-7]
    ctrl = values[-7:]
    values = values[:-14]
    drefs = drefs[:-14]

    # Send values to simulator
    client.sendPOSI(pos)
    client.sendDREFs(drefs,values)
    ctrl[4] = int(ctrl[4]); ctrl[5] = int(ctrl[5]); ctrl[6] = int(ctrl[6]);
    ctrl[3]*=3
    client.sendCTRL(ctrl)
    time.sleep(0.05)

if __name__ == "__main__":
    main()
