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
/*#ifndef WIN32
#include <experimental/filesystem>
#else
#include <filesystem>
#endif
*/using namespace opencover;

// void calcAverageMatrix(osg::Matrixf& average, const osg::Matrixf& input)
// {
    // average(0,0) = (average(0,0) + input(0,0) ) / 2;
    // average(0,1) = (average(0,1) + input(0,1) ) / 2;
    // average(0,2) = (average(0,2) + input(0,2) ) / 2;
    // average(0,3) = (average(0,3) + input(0,3) ) / 2;
    // average(1,0) = (average(1,0) + input(1,0) ) / 2;
    // average(1,1) = (average(1,1) + input(1,1) ) / 2;
    // average(1,2) = (average(1,2) + input(1,2) ) / 2;
    // average(1,3) = (average(1,3) + input(1,3) ) / 2;
    // average(2,0) = (average(2,0) + input(2,0) ) / 2;
    // average(2,1) = (average(2,1) + input(2,1) ) / 2;
    // average(2,2) = (average(2,2) + input(2,2) ) / 2;
    // average(2,3) = (average(2,3) + input(2,3) ) / 2;
    // average(3,0) = (average(3,0) + input(3,0) ) / 2;
    // average(3,1) = (average(3,1) + input(3,1) ) / 2;
    // average(3,2) = (average(3,2) + input(3,2) ) / 2;
    // average(3,3) = (average(3,3) + input(3,3) ) / 2;
// }

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
        auto updatePos1 = std::find_if(_udpCameras.begin(), _udpCameras.end(),[&message ,&timestamp](DetectedCameraOrObject& object){return object.update(message, timestamp);});
        if(updatePos1 == _udpCameras.end())
        {
            _udpCameras.push_back(DetectedCameraOrObject(message, timestamp));
            DataManager::AddUDPSensor(Factory::createSensor(SensorType::Camera, message._matrix,true,osg::Vec4(0.5,0.5,1,1)));
            DataManager::GetUDPSensors().back().get()->showInteractor(false);
        }
        else
        {
            osg::Matrix updateMatrix;
            bool send = _udpCameras.at(std::distance(_udpCameras.begin(), updatePos1)).getMatrixFromClosestCamera(updateMatrix);
            if(send)
                DataManager::UpdateUDPSensorPosition(std::distance(_udpCameras.begin(), updatePos1), updateMatrix ); 
        }

        // int updatePos = replaceMessage(_udpCameras, message, timestamp);
        // if(updatePos == -1) //if the id is not in the vector than add it
        // {
        //     DataManager::AddUDPSensor(Factory::createSensor(SensorType::Camera, message._matrix,true,osg::Vec4(0.5,0.5,1,1)));
        //     DataManager::GetUDPSensors().back().get()->showInteractor(false);
        //     _udpCameras.push_back(DetectedCameraOrObject(message, timestamp));
        // }
        // else    //if the id was found then update the position
        // {
        //     osg::Matrix matrix;
        //     if(UI::m_showShortestUDPPositions)
        //         _udpCameras.at(updatePos).getMatrixFromClosestCamera(matrix);
        //     else if(UI::m_showAverageUDPPositions)
        //         matrix = _udpCameras.at(updatePos).getAverageMatrix();

        //      if(_udpCameras.at(updatePos)._frameCounter == 10)
        //      {    
        //         DataManager::UpdateUDPSensorPosition(updatePos,matrix ); 
        //         _udpCameras.at(updatePos)._frameCounter = 0;
        //         _udpCameras.at(updatePos)._markers.front()._Matrix = matrix;
        //      }
        //       std::cout <<"counter:" << _udpCameras.at(updatePos)._frameCounter <<std::endl;

        // }
    }
    else if(message._type == MessageType::ROI)
    {
        osg::Matrix translate = osg::Matrix::translate(osg::Vec3(0,0,0.02)); // translate ROI in z direction, that it is flat on table

        auto updatePos1 = std::find_if(_udpROI.begin(), _udpROI.end(),[&message ,&timestamp](DetectedCameraOrObject& object){return object.update(message, timestamp);});
        if(updatePos1 == _udpROI.end())
        {
            _udpROI.push_back(DetectedCameraOrObject(message, timestamp));
            DataManager::AddUDPZone(Factory::createSafetyZone(SafetyZone::Priority::PRIO1, translate * message._matrix, 0.297, 0.210, 0.02)); 
            DataManager::GetUDPSafetyZones().back().get()->hide();
        }
        else
        {
            osg::Matrix updateMatrix;
            bool send = _udpROI.at(std::distance(_udpROI.begin(), updatePos1)).getMatrixFromClosestCamera(updateMatrix);
            if(send)
                DataManager::UpdateUDPZone(std::distance(_udpROI.begin(), updatePos1), updateMatrix, _udpROI.at(std::distance(_udpROI.begin(), updatePos1)).getNbrOfMarkers() );
        }

        // osg::Matrix translate = osg::Matrix::translate(osg::Vec3(0,0,0.02)); // translate ROI in z direction, that it is flat on table
        // int updatePos = replaceMessage(_udpROI, message, timestamp);

        // if(updatePos == -1) //if the id is not in the vector than add it
        // {
        //     DataManager::AddUDPZone(Factory::createSafetyZone(SafetyZone::Priority::PRIO1, translate * message._matrix, 0.297, 0.210, 0.02)); 
        //     DataManager::GetUDPSafetyZones().back().get()->hide();

        //     _udpROI.push_back(DetectedCameraOrObject(message, timestamp));

        // }
        // else    //if the id was found then update the position
        // {
        //     osg::Matrix matrix;
        //     if(UI::m_showShortestUDPPositions)
        //     {
        //         _udpROI.at(updatePos).getMatrixFromClosestCamera(matrix);
        //         matrix = translate * matrix;
        //     }
        //     else if(UI::m_showAverageUDPPositions)
        //         matrix = translate *_udpROI.at(updatePos).getAverageMatrix();

        //     DataManager::UpdateUDPZone(updatePos, matrix, _udpROI.at(updatePos).getNbrOfMarkers() );  
        // }
    }
    else if(message._type == MessageType::Obstacle)
    {

        auto updatePos1 = std::find_if(_udpObstacle.begin(), _udpObstacle.end(),[&message ,&timestamp](DetectedCameraOrObject& object){return object.update(message, timestamp);});
        if(updatePos1 == _udpObstacle.end())
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
        else
        {
            osg::Matrix updateMatrix;
            bool send = _udpObstacle.at(std::distance(_udpObstacle.begin(), updatePos1)).getMatrixFromClosestCamera(updateMatrix);
            if(send)
                DataManager::UpdateUDPObstacle(std::distance(_udpObstacle.begin(), updatePos1), updateMatrix );
        }   

        // int updatePos = replaceMessage(_udpObstacle, message, timestamp);
        // if(updatePos == -1) //if the id is not in the vector than add it
        // {
        //     _udpObstacle.push_back(DetectedCameraOrObject(message, timestamp));
        //     const char *covisedir = getenv("COVISEDIR");
        //     osg::ref_ptr<osg::Node> node = osgDB::readNodeFile( std::string(covisedir)+ "/obstacle.3ds" );
        //     if (!node.valid())
        //     {
        //         osg::notify( osg::FATAL ) << "Unable to load node data file. Exiting." << std::endl;
        //     }
        //     DataManager::AddUDPObstacle(std::move(node), message._matrix);
        // }
        //  else    //if the id was found then update the position
        //  {
        //     osg::Matrix matrix;
        //     if(UI::m_showShortestUDPPositions)
        //         _udpObstacle.at(updatePos).getMatrixFromClosestCamera(matrix);
        //     else if(UI::m_showAverageUDPPositions)
        //         matrix = _udpObstacle.at(updatePos).getAverageMatrix();

        //     DataManager::UpdateUDPObstacle(updatePos, matrix ); 
        //  }
    }
}
bool DetectedCameraOrObject:: s_frameAverage{true};
bool DetectedCameraOrObject::update(const Message& newMessage, const double& timestamp)
{
    bool updated{false};

    if(newMessage._type == this->_type)
    {
        if((this->_type == MessageType::Camera && this->_id  == newMessage._cameraID) || (this->_type != MessageType::Camera && this->_id == newMessage._id )) // check if object already exists
        {    //does object already exist ? 

            auto found = std::find_if(_markers.begin(), _markers.end(),[&newMessage, &timestamp, this](Marker& marker)
            {   
                if( (this->_type == MessageType::Camera && marker._markerID == newMessage._id) || (this->_type != MessageType::Camera && marker._markerID == newMessage._cameraID) )
                {
                    std::cout <<"Old pos: " << marker._Matrix.getTrans().x() <<", "<<marker._Matrix.getTrans().y() <<", "<<marker._Matrix.getTrans().z() <<std::endl;
                    std::cout <<"New pos: " <<newMessage._matrix.getTrans().x() <<", "<<newMessage._matrix.getTrans().y() <<", "<<newMessage._matrix.getTrans().z() <<std::endl;
                    
                    marker._distance = newMessage._distanceCamera;
                    marker._timestamp = timestamp;
                    if(s_frameAverage)
                       marker.checkDifferenceOfMatrixes(newMessage);// marker.calcAverageMatrix(newMessage);
                    else
                        marker._Matrix = newMessage._matrix;

                    //cameraAlreadyAvailable = true;
                    //break; // camera was found jump out of loop
                    

                    return true;
                }
                else
                    return false;
            

            });

            if(found == std::end(_markers))
                addNewMarker(newMessage, timestamp);
            
            updated = true;
        }
    }
    return updated;    
}

void DetectedCameraOrObject::Marker::calcAverageMatrix(const Message& input)
{
    if(_frameCounter > 10)
    {
        _Matrix = input._matrix;
        _frameCounter = 0;
        _send = true;
    }
    else
    {
        _Matrix(0,0) = (_Matrix(0,0) + input._matrix(0,0) ) / 2;
        _Matrix(0,1) = (_Matrix(0,1) + input._matrix(0,1) ) / 2;
        _Matrix(0,2) = (_Matrix(0,2) + input._matrix(0,2) ) / 2;
        _Matrix(0,3) = (_Matrix(0,3) + input._matrix(0,3) ) / 2;
        _Matrix(1,0) = (_Matrix(1,0) + input._matrix(1,0) ) / 2;
        _Matrix(1,1) = (_Matrix(1,1) + input._matrix(1,1) ) / 2;
        _Matrix(1,2) = (_Matrix(1,2) + input._matrix(1,2) ) / 2;
        _Matrix(1,3) = (_Matrix(1,3) + input._matrix(1,3) ) / 2;
        _Matrix(2,0) = (_Matrix(2,0) + input._matrix(2,0) ) / 2;
        _Matrix(2,1) = (_Matrix(2,1) + input._matrix(2,1) ) / 2;
        _Matrix(2,2) = (_Matrix(2,2) + input._matrix(2,2) ) / 2;
        _Matrix(2,3) = (_Matrix(2,3) + input._matrix(2,3) ) / 2;
        _Matrix(3,0) = (_Matrix(3,0) + input._matrix(3,0) ) / 2;
        _Matrix(3,1) = (_Matrix(3,1) + input._matrix(3,1) ) / 2;
        _Matrix(3,2) = (_Matrix(3,2) + input._matrix(3,2) ) / 2;
        _Matrix(3,3) = (_Matrix(3,3) + input._matrix(3,3) ) / 2;

        _frameCounter++;
        _send = false;
    }

    std::cout <<"send: " <<_send <<std::endl;
    
}

void DetectedCameraOrObject::Marker::checkDifferenceOfMatrixes(const Message& input)
{
    //coCoord eulerIn = input._matrix;
    //coCoord eulerOld = _Matrix;

    osg::Quat inQuat = input._matrix.getRotate();
    osg::Quat oldQuat = _Matrix.getRotate();
    double const eps = 1e-12; // some error threshold
    std::cout <<"Old pos: " <<_Matrix.getTrans().x() <<", "<<_Matrix.getTrans().y() <<", "<<_Matrix.getTrans().z() <<std::endl;
    std::cout <<"New pos: " <<input._matrix.getTrans().x() <<", "<<input._matrix.getTrans().y() <<", "<<input._matrix.getTrans().z() <<std::endl;
    
    double angle;
    osg::Vec3 axis;
    inQuat.getRotate(angle, axis);
    double angleRad = osg::RadiansToDegrees(angle);
    std::cout <<"New quat to angle Angle(rad): " <<angle <<" Angle(deg): "<<angleRad << " x:"<<axis.x() <<" y:"<<axis.y() <<" z:"<<axis.z() <<std::endl;
    _send = true;
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
                    //calcAverageMatrix(marker._Matrix, message._matrix);
                    marker._Matrix = message._matrix;
                    marker._timestamp = timestamp;
                    obj._frameCounter++;
                    cameraAlreadyAvailable = true;
                    break; // camera was found jump out of loop
                }
            }
            if(!cameraAlreadyAvailable)
                obj.addNewMarker(message, timestamp);
            
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


