from os.path import join, dirname, abspath
import sys
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
import pybullet as p
import numpy as np

def approaching_sensing(sensor_transform_matrix):
    sensorPosition = []
    rayEndPosition = []
    rayRange = 0.05
    armRadius = 0.042
    sideOffset = 0.05
    rayMissColor = [1, 0, 0]

    for key in sensor_transform_matrix:
        if "m9" in key or "link11sensor_num6" in key:
            sensorPosition.append(sensor_transform_matrix[key].dot([0, 0, sideOffset, 1])[:3])
            rayEndPosition.append(sensor_transform_matrix[key].dot([0, 0, sideOffset + rayRange, 1])[:3])
        else:
            sensorPosition.append(sensor_transform_matrix[key].dot([armRadius, 0, 0, 1])[:3])
            rayEndPosition.append(sensor_transform_matrix[key].dot([armRadius + rayRange, 0, 0, 1])[:3])

    approaching_datas = p.rayTestBatch(sensorPosition, rayEndPosition)
    # print("approachingDataset:{}".format(approachingDataset))
    # touch = []
    # for i in range(len(sensorPosition)):
    #     touch.append(p.addUserDebugLine(sensorPosition[i], rayEndPosition[i], rayMissColor, bodyId))
    return approaching_datas

def contact_sensing(robots, objectsId):
    contact_datas = []
    contact_data = []
    for i in range(len(objectsId)):
        points = p.getContactPoints(bodyA=robots,
                                   bodyB=objectsId[i], linkIndexA=[11])
        body_states = p.getLinkState(objectsId[i], 0)
        if len(points) == 1:
            if len(points[0]) == 14:
                contact_data.append([list(points[0][5]),
                                     [list(body_states[0][:2]),
                                     list(body_states[4][:2]),
                                     [0, list(points[0][0][9])]]])

            elif len(points[0]) >= 1 and len(points[0]) <= 5:
                contact_data.append([list(points[0][0][5]),
                                     [list(body_states[0][:2]),
                                     list(body_states[4][:2]),
                                     [0, list(points[0][0][9])]]])
        elif len(points) == 14:
            contact_data.append([list(points[5]),
                                     [list(body_states[0][:2]),
                                     list(body_states[4][:2]),
                                     [0, list(points[9])]]])
        contact_datas.append(contact_data)
    # print("contactData:{}".format(contactDataset))
    return contact_datas


