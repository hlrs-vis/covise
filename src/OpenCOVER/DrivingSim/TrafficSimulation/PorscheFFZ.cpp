/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "../../plugins/drivingsim/SteeringWheel/UDPComm.h"
#include "PorscheFFZ.h"

#include "VehicleManager.h"
#include "HumanVehicle.h"
#include "AgentVehicle.h"
#include "VehicleUtils.h"

#include "projects/radarcones.hpp"

#include "UDPBroadcast.h"
// some alternative classes (haven't got broadcast functionality?):
// #include <util/UDP_Sender.h>
// #include "UDPComm.h"

// RoadSystem //
//
#include "RoadSystem/Road.h"
#include "RoadSystem/LaneSection.h"
#include "RoadSystem/Lane.h"

// For Receive Thread //
//
#include <OpenThreads/Barrier>
#include <OpenThreads/Mutex>

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>

#include <math.h>

using namespace covise;
using namespace opencover;

/** UDP BROADCASTER. Sends data via UDP.
 * This Broadcast was initially designed for sending data
 * via UDP to the Porsche dSPACE and KMS. A list of vehicles
 * will be created sendFrequency times per second. The data
 * of the 32 vehicles that are closest to the human driver
 * will be transmitted.
 * Data:
 * Number of obstacles (so far only vehicles);
 * Weather type and visibility range;
 * 32 Blocks of vehicle data:
 * Position, orientation and velocity relative to human driver
 * Size, unique ID and type (car, van, pedestrian,...)
*/
PorscheFFZ::PorscheFFZ(double sendFrequency)
    : sendFrequency_(sendFrequency)
    , dSpaceComm_(NULL)
    , kmsComm_(NULL)
    , doRun_(false)
    , cones_(NULL)
{
    startTime_ = (float)cover->frameTime();

    radarConesData_ = (RadarConesData *)malloc(sizeof(RadarConesData));
    memset(radarConesData_, 0, sizeof(RadarConesData));

    if (coVRMSController::instance()->isMaster())
    {
        incomingRadarConesData_ = (RadarConesData *)malloc(sizeof(RadarConesData));
        memset(incomingRadarConesData_, 0, sizeof(RadarConesData));
    }
    else
    {
        incomingRadarConesData_ = NULL;
    }
};

/** Destructor.
*/
PorscheFFZ::~PorscheFFZ()
{
    if (doRun_)
    {
        doRun_ = false;
        endBarrier.block(2); // wait until communication thread finishes (???)
        delete dSpaceComm_;
        delete kmsComm_;
    }

    free(radarConesData_);

    if (coVRMSController::instance()->isMaster())
    {
        free(incomingRadarConesData_);
    }
}

void
PorscheFFZ::setupDSPACE(std::string destinationIP, int port, int localPort)
{
    dSpaceIp_ = destinationIP;
    dSpacePort_ = port;
    dSpaceLocalPort_ = localPort;

    // Start thread and open connection //
    //
    if (coVRMSController::instance()->isMaster() && !destinationIP.empty())
    {
        std::cout << "\nPorscheFFZ: Creating UPD Connection with dSPACE." << std::endl;
        dSpaceComm_ = new UDPBroadcast(destinationIP.c_str(), port, localPort, NULL, -1, false);
        std::cout << "		return Message: " << dSpaceComm_->errorMessage() << std::endl;

        // For reading //
        //
        doRun_ = true;
        startThread();
    }
}

void
PorscheFFZ::setupKMS(std::string destinationIP, int port, int localPort)
{
    kmsIp_ = destinationIP;
    kmsPort_ = port;
    kmsLocalPort_ = localPort;

    // Start thread and open connection //
    //
    if (coVRMSController::instance()->isMaster() && !destinationIP.empty())
    {
        std::cout << "\nPorscheFFZ: Creating UPD Connection with KMS." << std::endl;
        std::cout << "  " << kmsIp_ << ":" << kmsPort_ << ":" << kmsLocalPort_ << std::endl;
        kmsComm_ = new UDPBroadcast(destinationIP.c_str(), port, localPort, NULL, -1, false);
        std::cout << "  return Message: " << kmsComm_->errorMessage() << "\n" << std::endl;
    }
}

/** Master only. Thread sleeps until it receives data from UDP.
* Triggers READING (not sending). LIFO.
*/
void
PorscheFFZ::run()
{
    memset(incomingRadarConesData_, 0, sizeof(RadarConesData));

    while (doRun_)
    {
        //sendData();
        while (doRun_ && !readData())
        {
            microSleep(10);
        }
    }
    endBarrier.block(2); // wait until communication thread finishes (???)
}

/** Master only.
*/
bool
PorscheFFZ::readData()
{
#if 0

	int ret = dSpaceComm_->readMessage();
	if(ret < 4)
	{
		return false;
	}


//	int *msgType = (int *)comm_->rawBuffer();
//	switch (*msgType)
//	{
//		case DSpaceToVis:
			if(dSpaceComm_->messageSize() == sizeof(RadarConesData))
			{
				memcpy((void *)incomingRadarConesData_, dSpaceComm_->rawBuffer(), sizeof(RadarConesData));
			}
			else
			{
				std::cout << "received wrong UDP Message, expected size " << sizeof(RadarConesData) << "but received size " << dSpaceComm_->messageSize() << std::endl;
			}
//			break;
//		default:
//			cerr << "received unknown message type " << *msgType << " size " << comm_->messageSize() << endl;
//			break;
//	}
	return true;
#else
    return false;
#endif
}

/** Send data via UDP.
 * Send data to destinationIP_ with port_.
*/
void
PorscheFFZ::sendData(const VehicleList &vehicleList)
{
    if (coVRMSController::instance()->isMaster())
    {
        //Test ob hoehe aus Steeringwheel gelesen werden kann 14.04.2011

        //cout << "TrafficSim - hoehe[0]: "<< rayHeight << endl;

        // HumanVehicle //
        //
        HumanVehicle *human = VehicleManager::Instance()->getHumanVehicle();
        if (!human)
            return; // no human driver found

        Vector3D l_human = human->getVehicleTransform().v();
        Quaternion q_human = human->getVehicleTransform().q();

        // List of AgentVehicles //
        //
        // Go through official list of vehicles and create a new one, sorted by distance
        std::map<double, AgentVehicle *> vehicleMapDistanceSorted;
        VehicleList::const_iterator vehIt = vehicleList.begin();
        while (vehIt != vehicleList.end())
        {
            AgentVehicle *veh = dynamic_cast<AgentVehicle *>(*vehIt);
            if (veh)
            {
                vehicleMapDistanceSorted.insert(std::pair<double, AgentVehicle *>(veh->getSquaredDistanceTo(l_human), veh));
            }
            ++vehIt; // go to next one in list
        }

        // Prepare dataSet //
        //
        VehicleBroadcastOutData dataSet;

        dataSet.nObstacles = (unsigned short)vehicleMapDistanceSorted.size();
        dataSet.weatherType = VehicleBroadcastOutData::WEATHER_SUNNY;
        dataSet.visibilityRange = 10000.0f;
        dataSet.time = int((cover->frameTime() - startTime_) * 1000.0); // [ms]

        // Human Vehicle //
        //
        double w = q_human.w();
        double x = q_human.x();
        double y = q_human.y();
        double z = q_human.z();
        double pitch = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
        double roll = -asin(2 * (w * y - x * z));
        double yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)) - M_PI / 2.0;
        //	double yaw = atan2(2*(w*z + x*y), 1-2*(y*y + z*z));
        double yawHuman = yaw;
        ObstacleData ob(65535, ObstacleData::OBSTACLE_HUMAN,
                        (float)l_human[0], (float)l_human[1], (float)l_human[2],
                        //			(float)sizeX, (float)sizeY, (float)sizeZ,
                        // TODO: calculate!
                        (float)4.970, (float)1.931, (float)1.418, // panamera
                        (float)roll, (float)pitch, (float)yaw,
                        (float)0.0 // speed
                        );
        dataSet.obstacle[0] = ob;

        // Lane //
        //
        Road *road = human->getRoad();
        dataSet.weatherType = 0;
        dataSet.visibilityRange = 10.0;
        if (road)
        {
            double s = human->getU();
            double t = human->getV();
            LaneSection *laneSection = road->getLaneSection(s);
            int l = human->getLane();
            if (l != Lane::NOLANE)
            {
                //geändert 07.08.12
                /*
//				double width = laneSection->getLaneWidth(s, l);
//				Vector2D center = laneSection->getLaneCenter(laneId, human->getU());
				double center = laneSection->getDistanceToLane(s, l) + 0.5*laneSection->getLaneWidth(s, l);
//				std::cout << "width: " << width << ", center: " << center << ", t: " << t << std::endl;
//				std::cout << "distance: " << center-t << std::endl;
				dataSet.weatherType = l + 1000; // unsigned... 1000 = 0
				dataSet.visibilityRange = center-t;
*/
                //double width = laneSection->getLaneWidth(s, l);
                double center = laneSection->getDistanceToLane(s, l) + 0.5 * laneSection->getLaneWidth(s, l);

                dataSet.visibilityRange = center - t;
                dataSet.weatherType = l + 1000; // unsigned... 1000 = 0
            }
        }

        // Agent Vehicles //
        //
        if (vehicleMapDistanceSorted.size() > 0)
        {
            int i = 1; // maximum of 31 cars, the first is the human driver
            std::map<double, AgentVehicle *>::const_iterator distIt = vehicleMapDistanceSorted.begin();
            //while(distIt != vehicleMapDistanceSorted.end() && i < 32) {
            while (distIt != vehicleMapDistanceSorted.end() && i < 10)
            {
                AgentVehicle *veh = (*distIt).second;
                //std::cout << "ID: " << (*distIt).first << ", " << veh->getVehicleID() << std::endl;

                // size (length, width, height) //
                //
                osg::BoundingBox bbox = veh->getCarGeometry()->getBoundingBox();
                double sizeX = bbox.xMax() - bbox.xMin();
                double sizeY = bbox.yMax() - bbox.yMin();
                double sizeZ = bbox.zMax() - bbox.zMin();
                //NEU 03-02-11
                //if (isinf(sizeX)) std::cout << "********** FALSCHES FAHRZEUG!! ************" << std::endl;
                //if (!(isinf(sizeX))&&!(isinf(sizeY))&&!(isinf(sizeZ))) << AUSKOMMENTIERT: Mit Falschmeldungen
                //{
                //std::cout << " sizeX: " << sizeX << " sizeY: " << sizeY << " sizeZ: " << sizeZ << std::endl;
                //

                // translation //
                //
                // translation and rotation of the car's boundingbox center relative to
                // the center of gravity of the human driver.
                Quaternion q_car = veh->getVehicleTransform().q();
                Vector3D l_car = veh->getVehicleTransform().v();

                Vector3D l_car_center = (q_car * Vector3D(bbox.xMin() + sizeX / 2.0, bbox.yMin() + sizeY / 2.0, bbox.zMin() + sizeZ / 2.0) * q_car.T()).getVector(); // offset between car origin (ground) and car boundingbox center, local to global

                Vector3D dl_global = l_car + l_car_center - l_human; // global distance
                Vector3D dl_local = (q_human.T() * dl_global * q_human).getVector(); // transform to local
                //std::cout << "dl_local: " << dl_local[0] << ", " << dl_local[1] << ", " << dl_local[2] << std::endl;

                Vector3D l_center_local = (q_car.T() * Vector3D(bbox.xMin() + sizeX / 2.0, bbox.yMin() + sizeY / 2.0, bbox.zMin() + sizeZ / 2.0) * q_car).getVector(); // offset between car origin (ground) and car boundingbox center
                //std::cout << "l_cog_local: " << l_cog_local[0] << ", " << l_cog_local[1] << ", " << l_cog_local[2] << std::endl;

                //Vector3D loc = dl_local + l_center_local; edit 10-09-10
                Vector3D loc = (q_human.T() * dl_global * q_human).getVector(); // transform to local
                //std::cout << "loc: " << loc[0] << ", " << loc[1] << ", " << loc[2] << std::endl;

                // rotation //
                //
                Quaternion dq = q_human.T() * q_car;
                //Quaternion dq = veh->getVehicleTransform().q();
                double w = dq.w();
                double x = dq.x();
                double y = dq.y();
                double z = dq.z();
                double pitch = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
                double roll = -asin(2 * (w * y - x * z));

                // geändert Stefan: 24.07.2012
                //double yaw = atan2(2*(w*z + x*y), 1-2*(y*y + z*z)) - M_PI/2.0;
                //std::cout << "rot: " << 360.0*pitch/6.28 << ", " << 360.0*roll/6.28 << ", " << 360.0*yaw/6.28 << std::endl;
                double yaw = veh->getYaw() - yawHuman;

                // speed (absolute) //
                //
                // note that the car is always oriented in the direction of the velocity
                VehicleState vp = veh->getVehicleState();
                double speed = sqrt(vp.du * vp.du + vp.dv * vp.dv);

                // Save //
                //
                /*		ObstacleData ob(veh->getVehicleID(), ObstacleData::Obstacle_TRUCK
					(float)loc[0], (float)loc[1], (float)loc[2],
					(float)sizeX, (float)sizeY, (float)sizeZ,
					(float)roll, (float)pitch, (float)yaw,
					(float)speed
				);*/ //NEU 02-02-2011

                ObstacleData ob(veh->getVehicleID(), veh->getVehType(),
                                (float)loc[0], (float)loc[1], (float)loc[2],
                                (float)sizeX, (float)sizeY, (float)sizeZ,
                                (float)roll, (float)pitch, (float)yaw,
                                (float)speed);
                dataSet.obstacle[i] = ob;

                // Iteration //
                //
                ++i;
                //	} // NEU 03-02-11 Fahrzeug wird nur hinzugefuegt,wenn Wert korrekt << AUSKOMMENTIERT: Mit Falschmedlungen
                ++distIt; // go to next one in list
            }

            // SEND //
            //
            //std::cout << "SENDING: " << comm_->send(&dataSet, sizeof(dataSet)) << std::endl;
            if (dSpaceComm_)
            {
                dSpaceComm_->send(&dataSet, sizeof(dataSet));
            }

            if (kmsComm_)
            {
                kmsComm_->send(&dataSet, sizeof(dataSet));
            }
        }
    }
}

/** Receive UDP data.
 *
*/
void
PorscheFFZ::receiveData()
{
    // Syncronizes Master and Slaves //
    //
    if (coVRMSController::instance()->isMaster())
    {
        memcpy((void *)radarConesData_, (void *)incomingRadarConesData_, sizeof(RadarConesData));
        coVRMSController::instance()->sendSlaves((char *)radarConesData_, sizeof(RadarConesData));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)radarConesData_, sizeof(RadarConesData));
    }

    // Example //
    //
    //radarConesData_->nSensors = 1;
    //radarConesData_->cones[0].installPosX = 0.5;
    //radarConesData_->cones[0].installPosY = -0.5;
    //radarConesData_->cones[0].installPosZ = 0.5;
    //radarConesData_->cones[0].installRoll = 0.0;
    //radarConesData_->cones[0].installPitch = 0.0;
    //radarConesData_->cones[0].installYaw = 0.0;
    //radarConesData_->cones[0].minRange = 0.5;
    //radarConesData_->cones[0].maxRange = 40.0;
    //radarConesData_->cones[0].azimuth = 5.0;
    //radarConesData_->cones[0].elevation = 2;
    //radarConesData_->cones[0].color = 0x01ff0e0f;
    //radarConesData_->cones[0].color = 0xff0000e0;

    //	std::cout << "nSens: " << radarConesData_->nSensors << std::endl;
    //	std::cout << "  max: " <<  radarConesData_->cones[0].maxRange << std::endl;

    HumanVehicle *human = VehicleManager::Instance()->getHumanVehicle();
    if (!human)
    {
        return; // no human driver found
    }

    // Porsche Radar Cones //
    //
    if (!cones_)
    {
        cones_ = new RadarCones(human);
    }
    cones_->update(radarConesData_);

    return;
}

//
