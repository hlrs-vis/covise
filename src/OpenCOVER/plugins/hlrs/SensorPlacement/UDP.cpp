#include "UDP.h"
#include "Helper.h"
#include <string>
#include <config/CoviseConfig.h>
#include <cover/coVRMSController.h>
#include "DataManager.h"
#include "Factory.h"


#include <osgDB/ReadFile>
#include <iostream>
#include <experimental/filesystem>
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
    const std::string host = covise::coCoviseConfig::getEntry("value", "SensorPlacement.serverHost", "10.65.5.241");
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
    MessageWithTimestamp msgWithTimestamp(message, timestamp);   

    if(message.type == MessageType::Camera)
    {  
        int updatePos = replaceMessage(_udpCameras, msgWithTimestamp);
        if(updatePos == -1) //if the id is not in the vector than add it
        {
            _udpCameras.push_back(msgWithTimestamp);
            DataManager::AddUDPSensor(createSensor(SensorType::Camera, MessageToMatrix(msgWithTimestamp._message),true,osg::Vec4(0.5,0.5,1,1)));  
        }
        else    //if the id was found then update the position
            DataManager::UpdateUDPSensorPosition(updatePos, MessageToMatrix(msgWithTimestamp._message) );  
    }
    else if(message.type == MessageType::ROI)
    {
        int updatePos = replaceMessage(_udpROI, msgWithTimestamp);
        if(updatePos == -1) //if the id is not in the vector than add it
        {
            _udpROI.push_back(msgWithTimestamp);
            DataManager::AddUDPZone(createZone(ZoneType::ROIzone,MessageToMatrix(msgWithTimestamp._message), 0.324, 0.163, 0.02));
        }
        else    //if the id was found then update the position
            DataManager::UpdateUDPZone(updatePos, MessageToMatrix(msgWithTimestamp._message) );  
    }
    else if(message.type == MessageType::Obstacle)
    {
        int updatePos = replaceMessage(_udpObstacle, msgWithTimestamp);
        if(updatePos == -1) //if the id is not in the vector than add it
        {
            _udpObstacle.push_back(msgWithTimestamp);
            const char *covisedir = getenv("COVISEDIR");
            osg::ref_ptr<osg::Node> node = osgDB::readNodeFile( std::string(covisedir)+ "/obstacle.3ds" );
            if (!node.valid())
            {
                osg::notify( osg::FATAL ) << "Unable to load node data file. Exiting." << std::endl;
            }
            DataManager::AddUDPObstacle(std::move(node), MessageToMatrix(msgWithTimestamp._message));
        }
         else    //if the id was found then update the position
             DataManager::UpdateUDPObstacle(updatePos, MessageToMatrix(msgWithTimestamp._message)); 
    }
}

int UDP::replaceMessage(std::vector<MessageWithTimestamp>& vec, const MessageWithTimestamp& message   ) 
{
    if(vec.empty())
        return -1;

    bool replaced{false}; //found message with can be replaced with newer one  ?
    int iterator{0};
    int updatePos{0};     
    // replace old message with new message if id is the same! 
    std::replace_if(vec.begin(), vec.end(),[&message,&replaced,&iterator, &updatePos](const MessageWithTimestamp& msgIt)
                    {   
                        bool sameID = message._message.id == msgIt._message.id;
                        if(sameID)
                        {
                            replaced = true;
                            updatePos = iterator;
                        }  
                        iterator++;
                        return sameID;
                    }
    , message);                    // hier könnte man nach dem ersten Fund aufhören ! Algorithmus geht aber ganzen Vec durch!

    if(!replaced)
        return -1;
    
    else return updatePos;
}

void UDP::deleteOutOfDateMessages()
{
    const float maxValue{3.0}; // after "maxValue" seconds of no new data delete objects
    double timestamp = cover->frameTime();

    if(!_udpCameras.empty())
    {
        int count{0};
        for(const auto& camera : _udpCameras)
        {
            if( timestamp - camera._timestamp > maxValue)
            {
                _udpCameras.erase(_udpCameras.begin()+count);
                DataManager::RemoveUDPSensor(count);
            }
            count++;
        }
    }

    if(!_udpROI.empty())
    {
        int count{0};
        for(const auto& roi : _udpROI)
        {
            if( timestamp - roi._timestamp > maxValue)
            {
                _udpROI.erase(_udpROI.begin()+count);
                DataManager::RemoveUDPZone(count);
            }
            count++;
        }
    }

      if(!_udpObstacle.empty())
    {
        int count{0};
        for(const auto& obstacle : _udpObstacle)
        {
            if( timestamp - obstacle._timestamp > maxValue)
            {
                _udpObstacle.erase(_udpObstacle.begin()+count);
                DataManager::RemoveUDPObstacle(count);
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
            processIncomingMessage(temp);
            std::cout<< "SensorPlacement::update: successfully received message" <<std::endl;
            temp.printMessage();
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
    }

    Message::printSize();

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

void UDP::syncMasterSlave()
{
    //coVRMSController::instance()->syncData(&_message,sizeof(Message));
}


osg::Matrix UDP::MessageToMatrix( const Message& input)
{
    coCoord euler;
    euler.xyz[0] = input.translation[0];
    euler.xyz[1] = input.translation[1];
    euler.xyz[2] = input.translation[2];

    euler.hpr[0] = 180 + input.rotation[2];
    euler.hpr[1] = -90 + input.rotation[0];
    euler.hpr[2] = input.rotation[1]; 


    osg::Matrix matrix;
    euler.makeMat(matrix);

    return matrix;
}