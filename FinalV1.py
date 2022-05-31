# Make sure to have the add-on "ZMQ remote API" running in
# CoppeliaSim and have following scene loaded:
#
# scenes/messaging/synchronousImageTransmissionViaRemoteApi.ttt
#
# Do not launch simulation, but run this script
#
# All CoppeliaSim commands will run in blocking mode (block
# until a reply from CoppeliaSim is received). For a non-
# blocking example, see simpleTest-nonBlocking.py

import time
import math
import numpy as np
import cv2 as cv

from pupil_apriltags import Detector
from PIL import Image, ImageMorph
from PIL.ImageMorph import LutBuilder, MorphOp
import numpy as np
import os
import argparse

from zmqRemoteApi import RemoteAPIClient



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--families", type=str, default='tag36h11')
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--quad_decimate", type=float, default=2.0)
    parser.add_argument("--quad_sigma", type=float, default=0.0)
    parser.add_argument("--refine_edges", type=int, default=1)
    parser.add_argument("--decode_sharpening", type=float, default=0.25)
    parser.add_argument("--debug", type=int, default=0)

    args = parser.parse_args()

    return args


print('Program started')

TargetPos = [0.513, -0.521, 0.753]
CamPos = 0

L1 = 0.3
L2 = 0.38
L3 = 0.1

clawHandle = 44

Joint16Angle = 0
client = RemoteAPIClient()
sim = client.getObject('sim')

# When simulation is not running, ZMQ message handling could be a bit
# slow, since the idle loop runs at 8 Hz by default. So let's make
# sure that the idle loop runs at full speed for this program:
defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
#sim.setInt32Param(sim.intparam_idle_fps, 0)

# Run a simulation in stepping mode:
client.setStepping(True)
sim.startSimulation()

sim.setJointTargetPosition(16, 0.8)
sim.setJointTargetPosition(19, 0)
sim.setJointTargetPosition(24, 0)
sim.setJointTargetPosition(28, 0.5)

def draw_tags(
    image,
    tags,
    elapsed_time,
):
    for tag in tags:
        tag_id = tag.tag_id
        center = tag.center
        corners = tag.corners

        center = (int(center[0]), int(center[1]))
        corner_01 = (int(corners[0][0]), int(corners[0][1]))
        corner_02 = (int(corners[1][0]), int(corners[1][1]))
        corner_03 = (int(corners[2][0]), int(corners[2][1]))
        corner_04 = (int(corners[3][0]), int(corners[3][1]))

        # Center
        cv.circle(image, (center[0], center[1]), 5, (150, 150, 150), 2)

        # 
        cv.line(image, (corner_01[0], corner_01[1]),
                (corner_02[0], corner_02[1]), (255, 0, 0), 2)
        cv.line(image, (corner_02[0], corner_02[1]),
                (corner_03[0], corner_03[1]), (255, 0, 0), 2)
        cv.line(image, (corner_03[0], corner_03[1]),
                (corner_04[0], corner_04[1]), (0, 255, 0), 2)
        cv.line(image, (corner_04[0], corner_04[1]),
                (corner_01[0], corner_01[1]), (0, 255, 0), 2)

        cv.putText(image, str(tag_id), (center[0] - 10, center[1] - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv.LINE_AA)
    return image

args = get_args()
    
families = args.families
nthreads = args.nthreads
quad_decimate = args.quad_decimate
quad_sigma = args.quad_sigma
refine_edges = args.refine_edges
decode_sharpening = args.decode_sharpening
debug = args.debug

def detec():
    at_detector = Detector(
        families=families,
        nthreads=nthreads,
        quad_decimate=quad_decimate,
        quad_sigma=quad_sigma,
        refine_edges=refine_edges,
        decode_sharpening=decode_sharpening,
        debug=debug,
    )
    return at_detector


def getCam():
    
    img, resX, resY = sim.getVisionSensorCharImage(51)
    img = np.frombuffer(img, dtype=np.uint8).reshape(resY, resX, 3)
    img = cv.flip(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 0)
    image = np.uint8(img)

    tags = detec().detect(
                image,
                estimate_tag_pose=False,
                camera_params=None,
                tag_size=None,
        )
    
    image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    debug_image = draw_tags(image, tags, 0)
    cv.imshow('AprilTag Detect Demo', debug_image)
        
    cv.waitKey(5)

def step1(time1):
    rotate = -5
    while (sim.getSimulationTime() - time1 <= 8):
        getCam()
        
        sim.setJointTargetPosition(19, 0)
        sim.setJointTargetPosition(24, 0)
        sim.setJointTargetPosition(28, 0.5)
        sim.setJointTargetPosition(16, rotate)
        rotate = rotate + 0.05
        client.step() 

def step2(time1):
    while (sim.getSimulationTime() - time1 < 0.5):
        getCam()
        
        TargetPos = sim.getObjectPosition(88, 15) 
        
        Joint16Angle = math.atan((TargetPos[1]) / (TargetPos[0]))
        Joint16swag = ((0.9*((Joint16Angle)/0.785))*-1)+0.8
        
        distance = math.sqrt(TargetPos[0]**2 + TargetPos[1]**2)
        Angle19 = math.asin(((distance/1.8))/(2*L1))
        Angle24 = -3.2 + (2.2 * Angle19)
        Joint19angle = Angle19 - 0.5
        Joint24angle = Angle24 - 0.5
        Joint28angle = 3.1 - Angle19 
        
        sim.setJointTargetPosition(16, Joint16swag)
        sim.setJointTargetPosition(19, Joint19angle)
        sim.setJointTargetPosition(24, Joint24angle)
        sim.setJointTargetPosition(28, Joint28angle)
        
        client.step()  # triggers next simulation step

def step3(time1):
    while (sim.getSimulationTime() - time1 <= 0.8):
        getCam()
        TargetPos = sim.getObjectPosition(88, 15)
        displacement = sim.getObjectPosition(38, 88)
        distance = math.sqrt(TargetPos[0]**2 + TargetPos[1]**2)
        forward = math.sqrt(displacement[0]**2 + displacement[1]**2)
        total = distance + forward + 0.1
        
        Angle19 = math.asin(((total/1.8))/(2*L1))
        Angle24 = -3.2 + (2.2 * Angle19)
        Joint19angle = Angle19 - 0.5
        Joint24angle = Angle24 - 0.5
        Joint28angle = 3.1 - Angle19 
        
        
        sim.setJointTargetPosition(19, Joint19angle)
        sim.setJointTargetPosition(24, Joint24angle)
        sim.setJointTargetPosition(28, Joint28angle)
        client.step()
    
def step4(time1):
    while (sim.getSimulationTime() - time1 <= 0.8):
        getCam()
        TargetPos = sim.getObjectPosition(88, 15)
        displacement = sim.getObjectPosition(38, 88)
        distance = math.sqrt(TargetPos[0]**2 + TargetPos[1]**2)
        forward = math.sqrt(displacement[0]**2 + displacement[1]**2)
        total = distance + forward + 0.1
        
        Angle19 = math.asin(((total/1.7))/(2*L1))
        Angle24 = -3.3 + (2.2 * Angle19)
        Joint19angle = Angle19 - 0.25
        Joint24angle = Angle24 - 0.5
        Joint28angle = 3.45 - Angle19 
        
        
        
        sim.setJointTargetPosition(19, Joint19angle)
        sim.setJointTargetPosition(24, Joint24angle)
        sim.setJointTargetPosition(28, Joint28angle)
        client.step()
        
def step5(time1):
    while (sim.getSimulationTime() - time1 <= 0.2):
        getCam()
        sim.setJointTargetVelocity(clawHandle,-0.2)
        client.step()

def step6(time1, position):
    while (sim.getSimulationTime() - time1 <= 1):
        getCam()
        distance = math.sqrt(position[0]**2 + position[1]**2)
        Angle19 = math.asin(((distance - L3))/(2*L1))
        Angle24 = -3.2 + (2 * Angle19)
        Joint19angle = Angle19 - 0.6
        Joint24angle = Angle24 - 0.4
        Joint28angle = 3.1 - Angle19 
        
        sim.setJointTargetPosition(19, Joint19angle)
        sim.setJointTargetPosition(24, Joint24angle)
        sim.setJointTargetPosition(28, Joint28angle)
        client.step()

def step7(time1):
    while (sim.getSimulationTime() - time1 <= 0.5):
        getCam()
        position = sim.getObjectPosition(91, 15)
        
        Joint16Angle = math.atan((position[1]) / (position[0]))
        Joint16swag = ((1*((Joint16Angle)/0.785))*-1)+0.9
        
        
        distance = math.sqrt(position[0]**2 + position[1]**2)
        Angle19 = math.asin(((distance - L3))/(2*L1))
        Angle24 = -3.2 + (2 * Angle19)
        Joint19angle = Angle19 - 0.6
        Joint24angle = Angle24 - 0.6
        Joint28angle = 3.2 - Angle19 
        
        
        sim.setJointTargetPosition(16, Joint16swag)
        sim.setJointTargetPosition(19, Joint19angle)
        sim.setJointTargetPosition(24, Joint24angle)
        sim.setJointTargetPosition(28, Joint28angle)
        client.step()

def step8(time1):
    while (sim.getSimulationTime() - time1 <= 0.2):
        getCam()
        # position = sim.getObjectPosition(91, 15)
                
        # Angle24 = -3.2 
        # Joint19angle = 0
        # Joint24angle = Angle24 - 0.6
        # Joint28angle = 3.2 - 0
        
        # sim.setJointTargetPosition(19, Joint19angle)
        # sim.setJointTargetPosition(24, Joint24angle)
        # sim.setJointTargetPosition(28, Joint28angle)
        
        sim.setJointTargetVelocity(clawHandle,0.2)
        
        client.step()
    
def start():
#    args = get_args()
#    
#    families = args.families
#    nthreads = args.nthreads
#    quad_decimate = args.quad_decimate
#    quad_sigma = args.quad_sigma
#    refine_edges = args.refine_edges
#    decode_sharpening = args.decode_sharpening
#    debug = args.debug
##    
#    at_detector = Detector(
#        families=families,
#        nthreads=nthreads,
#        quad_decimate=quad_decimate,
#        quad_sigma=quad_sigma,
#        refine_edges=refine_edges,
#        decode_sharpening=decode_sharpening,
#        debug=debug,
#    )
#    
#    step1(sim.getSimulationTime())  #Scan
    step2(sim.getSimulationTime())  #visual servoing
    step3(sim.getSimulationTime())  #approach
    step4(sim.getSimulationTime())  #lower onto object
    step5(sim.getSimulationTime())  #grasp
    step6(sim.getSimulationTime(), sim.getObjectPosition(88, 15))   #stand
    step7(sim.getSimulationTime())  #Relocate to drop zone
    step8(sim.getSimulationTime())  #Lower
    step1(sim.getSimulationTime())


start()
sim.stopSimulation(sim.getSimulationTime())

# Restore the original idle loop frequency:
sim.setInt32Param(sim.intparam_idle_fps, defaultIdleFps)

cv.destroyAllWindows()

print('Program ended')
