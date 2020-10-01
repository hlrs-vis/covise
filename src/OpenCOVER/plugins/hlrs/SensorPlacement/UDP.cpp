#include "UDP.h"
#include "Helper.h"
#include <string>
#include <config/CoviseConfig.h>
#include <cover/coVRMSController.h>
#include "DataManager.h"
#include "Factory.h"
#include "UI.h"


#include <osgDB/ReadFile>
#include <iostream>
#ifndef WIN32
#include <experimental/filesystem>
#else
#include <filesystem>
#endif
using namespace opencover;

UDP::UDP()
{
    initUDP();
}

UDP::~UDP()
{
    _doRun = false;
    if (coVRMSController::instance()->isMaster())
    {
        fprintf(stderr, "waiting1\n");
        _endBarrier.block(2); // wait until communication thread finishes
        fprintf(stderr, "done1\n");
    }
}

void UDP::initUDP()
{
    _udp.reset(nullptr);
    const std::string host = covise::coCoviseConfig::getEntry("value", "SensorPlacement.serverHost", "10.42.0.2");
    _serverPort = covise::coCoviseConfig::getInt("SensorPlacement.serverPort", 5555);
    _localPort = covise::coCoviseConfig::getInt("SensorPlacement.serverHost", 5555);
    std::cerr << "SensorPlacement: UDP: serverHost: " << host << ", localPort: " << _localPort << ", serverPort: " << _serverPort << std::endl;
    _doRun = false;
    if(coVRMSController::instance()->isMaster())
    {
        _doRun = true;
        _udp = myHelpers::make_unique<UDPComm>(host.c_str(), _serverPort, _localPort);
        startThread();
        std::cout <<" start thread" <<std::endl;
    }
}

void UDP::processIncomingMessage(const Message& message)
{
    double timestamp = cover->frameTime();

    if(message._type == MessageType::Camera)
    {  
        int updatePos = replaceMessage(_udpCameras, message, timestamp);
        if(updatePos == -1) //if the id is not in the vector than add it
        {
            DataManager::AddUDPSensor(createSensor(SensorType::Camera, message._matrix,true,osg::Vec4(0.5,0.5,1,1)));
            _udpCameras.push_back(DetectedCameraOrObject(message, timestamp));
        }
        else    //if the id was found then update the position
        {
            osg::Matrix matrix;
            if(UI::m_showShortestUDPPositions)
                matrix =  _udpCameras.at(updatePos).getMatrixFromClosestCamera();
            else if(UI::m_showAverageUDPPositions)
                matrix = _udpCameras.at(updatePos).getAverageMatrix();

            DataManager::UpdateUDPSensorPosition(updatePos,matrix );  
        }
    }
    else if(message._type == MessageType::ROI)
    {
        osg::Matrix translate = osg::Matrix::translate(osg::Vec3(0,0,0.02)); // translate ROI in z direction, that it is flat on table
        int updatePos = replaceMessage(_udpROI, message, timestamp);

        if(updatePos == -1) //if the id is not in the vector than add it
        {
            DataManager::AddUDPZone(createSafetyZone(SafetyZone::Priority::PRIO1, translate * message._matrix, 0.297, 0.210, 0.02)); 
            _udpROI.push_back(DetectedCameraOrObject(message, timestamp));

        }
        else    //if the id was found then update the position
        {
            osg::Matrix matrix;
            if(UI::m_showShortestUDPPositions)
                matrix =  translate * _udpROI.at(updatePos).getMatrixFromClosestCamera();
            else if(UI::m_showAverageUDPPositions)
                matrix = translate *_udpROI.at(updatePos).getAverageMatrix();

            DataManager::UpdateUDPZone(updatePos, matrix, _udpROI.at(updatePos).getNbrOfMarkers() );  
        }
    }
    else if(message._type == MessageType::Obstacle)
    {
        int updatePos = replaceMessage(_udpObstacle, message, timestamp);
        if(updatePos == -1) //if the id is not in the vector than add it
        {
            _udpObstacle.push_back(DetectedCameraOrObject(message, timestamp));
            const char *covisedir = getenv("COVISEDIR");
            osg::ref_ptr<osg::Node> node = osgDB::readNodeFile( std::string(covisedir)+ "/obstacle.3ds" );
            if (!node.valid())
            {
                osg::notify( osg::FATAL ) << "Unable to load node data file. Exiting." << std::endl;
            }
            DataManager::AddUDPObstacle(std::move(node), message._matrix);
        }
         else    //if the id was found then update the position
         {
            osg::Matrix matrix;
            if(UI::m_showShortestUDPPositions)
                matrix =  _udpObstacle.at(updatePos).getMatrixFromClosestCamera();
            else if(UI::m_showAverageUDPPositions)
                matrix = _udpObstacle.at(updatePos).getAverageMatrix();

            DataManager::UpdateUDPObstacle(updatePos, matrix ); 
         }
    }
}

// return -1 if object id was not found and can't update message
int UDP::replaceMessage(std::vector<DetectedCameraOrObject>& vec, const Message& message, double timestamp )
{
    if(vec.empty())
        return -1;

    bool replaced{false}; //found message with can be replaced with newer one  ?
    int updatePos{0};    
    size_t iterator{0}; 

    MessageType msgType = vec.front()._type;
    for(auto& obj : vec) // go over all objects
    {
        if((msgType == MessageType::Camera && obj._id == message._cameraID) || (msgType != MessageType::Camera && obj._id == message._id )) // check if object already exists
        {
            replaced = true;
            updatePos = iterator;

            bool cameraAlreadyAvailable{false};
            for(auto& marker : obj._markers) // go over all cameras which can see this object
            {
                if( (msgType == MessageType::Camera && marker._markerID == message._id) || (msgType != MessageType::Camera && marker._markerID == message._cameraID) )
                {
                    marker._distance = message._distanceCamera;
                    marker._Matrix = message._matrix;
                    marker._timestamp = timestamp;

                    cameraAlreadyAvailable = true;
                    break; // camera was found jump out of loop
                }
            }
            if(!cameraAlreadyAvailable)
                obj.addMarker(message, timestamp);
            
            break; // object was found break out! 
        }
        if(replaced == true)
        {
            return updatePos;
        }
        iterator ++;
    }
    
    if(!replaced) 
        return -1;
    else
        return updatePos;
}

void UDP::deleteOutOfDateMessages()
{
    const float maxValue{3.0}; // after "maxValue" seconds of no new data delete objects
    double timestamp = cover->frameTime();

    if(!_udpCameras.empty())
    {
        int count{0};
        for(auto& camera : _udpCameras)
        {
            if(camera.isVisible(maxValue, timestamp))
            {
                _udpCameras.erase(_udpCameras.begin()+count);
                DataManager::RemoveUDPSensor(count); 
            }
            count++;
        }
    }

    if(!_udpObstacle.empty())
    {
        int count{0};
        for(auto& obstacle : _udpObstacle)
        {
            if(obstacle.isVisible(maxValue, timestamp))
            {
                _udpObstacle.erase(_udpObstacle.begin()+count);
                DataManager::RemoveUDPObstacle(count); 
            }
            count++;
        }
    }

    if(!_udpROI.empty())
    {
        int count{0};
        for(auto& object : _udpROI)
        {
            if(object.isVisible(maxValue, timestamp))
            {
                _udpROI.erase(_udpROI.begin()+count);
                DataManager::RemoveUDPZone(count); 
            }
            count++;    
        }
    }
 
    //this didn't work !!! --> try with algorithm !!
    // _udpCameras.erase( std::remove_if(_udpCameras.begin(), _udpCameras.end(), [&timestamp](const MessageWithTimestamp& obst)
            // {   
                // std::cout <<"diff" <<timestamp - obst._timestamp <<std::endl;
                // std::cout <<"bool"<<(timestamp - obst._timestamp >= 5.0 ? true : false) <<std::endl;
                // bool value;
                // if(timestamp - obst._timestamp >= 5.0) 
                    // value = true;
                // else 
                    // value = false;
                // return (timestamp - obst._timestamp >= 5.0 ? true : false);
                //  std::cout <<"value"<<value<<std::endl;;
                //  return true;
            // }
            // ));
}

bool UDP::update()
{
    //OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);   ->>> threads !!!!
    bool returnValue{true};
    if(_udp != nullptr)
    {
        Message temp; 
        int bytes = _udp->receive(&temp, sizeof(Message));

        if(bytes == sizeof(Message))
        {
            UDPMatrix2CoviseMatrix(temp);
            processIncomingMessage(temp);
            std::cout<< "SensorPlacement::update: successfully received message" <<std::endl;
            //temp.printMessage();
        }
        else if(bytes == -1)
        {
            std::cerr << "SensorPlacement::update: no incoming data" << std::endl;   
            returnValue = false;
        }
        else
        {
            std::cerr << "SensorPlacement::update: received invalid no. of bytes: recv=" << bytes << ", expected=" << Message::getSize() << std::endl;
            returnValue = false;
        }

        deleteOutOfDateMessages(); // --> maybe don't do this every Frame ! 

        for(const auto& x : _udpROI)
        {
            std::cout << "Nbr Cameras for ROI id " << x._id << " :" << x.getNbrOfMarkers() <<std::endl;
        }

        for(const auto& x : _udpObstacle)
        {
            std::cout << "Nbr Cameras for Obstacle id " << x._id << " :" << x.getNbrOfMarkers() <<std::endl;
        }

        for(const auto& x : _udpCameras)
        {
            std::cout << "Nbr calibMarkers for camera id " << x._id << " :" << x.getNbrOfMarkers() <<std::endl;
        }
    }

    //Message::printSize();

    return returnValue;
}

void UDP::run()
{   
    while(_doRun)
    {
        update();
        int worked = microSleep(10);
        //fprintf(stderr, "running\n");
    }

    fprintf(stderr, "waiting2\n");
    _endBarrier.block(2);
    fprintf(stderr, "done2\n");
}

void UDP::UDPMatrix2CoviseMatrix(Message& input) const 
{
    if(input._type == MessageType::Camera)
    {
        osg::Matrix udpCam2CoviseCam = osg::Matrix::rotate(osg::DegreesToRadians(-90.0), osg::X_AXIS);
        input._matrix = udpCam2CoviseCam * input._matrix ;
    } 
    
}