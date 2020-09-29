#pragma once

#include "UDPComm.h"
#include <cover/coVRPluginSupport.h>


enum class MessageType
{
    Camera = 0,         // detected a calibration marker
    Obstacle = 1,       // detected an obstacle
    ROI = 2             // detected a ROI
};


// this is the data structure of the incoming UDP message
struct Message
{   
    MessageType _type;
    int _cameraID;              // id of camera, which can see the object
    int _id;                    // each Objects has it's own id, in case of a detected calibration marker that's the marker id
    osg::Matrixf _matrix;       // matrix of the detected object. In case of a detected calibration marker that's the camera matrix
    float _distanceCamera;      // distance from camera to center of the object or the calibration marker

    void printMessage()
    {
        std::cout << "type:"<<(int)this->_type <<" id:"<<this->_id<<
        " x:"<<this->_matrix.getTrans().x() <<" y:" << this->_matrix.getTrans().y()<<" z:" <<this->_matrix.getTrans().z() << std::endl;
    }
    static void printSize()
    {
        size_t sum = sizeof(_type) + sizeof(_cameraID) + sizeof(_id) + sizeof(_matrix) + sizeof(_distanceCamera);
        size_t sumMatrix = sizeof(sumMatrix);

        std::cout<<"Size of datatype 'Message': "<< sizeof(Message) <<" bytes" << ", sum of the individual datatypes: " << sum <<"bytes" <<std::endl;
        //std::cout<<"Size of osg::Matrixf: "<< sizeof(matrix) <<" bytes" << std::endl;
    }
    static size_t getSize()
    {
        return sizeof(Message);
    }
};


/* 
    This is either a detected camera, ROI or obstacle
    if (Type == Camera): _id == cameraId, _markers == all calibrations markers which this camera can see
    else : _id == obstacle or ROI id, _markers == all cameras which can see this object
*/
struct DetectedCameraOrObject
{
    struct Marker
    {
        int _markerID;
        double _timestamp;
        osg::Matrixf _Matrix;        
        float _distance;             // distance marker - camera
        
        Marker(const Message& message, const double& timestamp)
        :_timestamp(timestamp), _Matrix(message._matrix), _distance(message._distanceCamera)
        {
            if(message._type == MessageType::Camera)
                _markerID = message._id;
            else
                _markerID = message._cameraID;
        };
    };

    DetectedCameraOrObject(const Message& message, const double& timestamp)
    :_type(message._type)
    {
        if(message._type == MessageType::Camera)
            _id = message._cameraID;
        else
            _id = message._id;

        _markers.push_back(Marker(message, timestamp));
    };

    MessageType _type;
    int _id;                        
    std::vector<Marker> _markers;

    void addMarker(const Message& message, const double& timestamp)
    {
        _markers.push_back(Marker(message, timestamp));
    };
    
    int getNbrOfMarkers()const {return _markers.size();}

    osg::Matrix getAverageMatrix()
    {
        std::cout << "Function not implementetd yet" << std::endl;
        return osg::Matrix();
    };

    osg::Matrix getMatrixFromClosestCamera()
    {
        std::vector<Marker>::iterator result = std::min_element(_markers.begin(), _markers.end(),[](const Marker& marker1, const Marker& marker2)
        {
            return marker1._distance < marker2._distance;
        }); 

        int pos = std::distance(_markers.begin(), result);
        if(_type == MessageType::Camera)
            std::cout<<"Camera Id: " << _id << " closest Marker: "<< _markers.at(pos)._distance << "meters"<<" from id: "<<_markers.at(pos)._markerID<< std::endl;
        else
            std::cout<<"Marker Id: " << _id << " closest Camera: "<< _markers.at(pos)._distance << "meters"<<" from id: "<<_markers.at(pos)._markerID<< std::endl;

        return _markers.at(pos)._Matrix;
    };

    bool isVisible(const float& maxValue, const double& timestamp) // check if this camera / marker is still alive
    {
        if(!_markers.empty())
        {
            size_t count{0};
            for(const auto& marker : _markers)
            {
                if(timestamp - marker._timestamp > maxValue)
                {
                    _markers.erase(_markers.begin() + count);
                    if(_markers.empty())
                    {
                        return true;
                    }
                }
                count++;
            }
        }
        return false;
    }
};

class UDP : public OpenThreads::Thread
{
private:
    std::unique_ptr<UDPComm> _udp;
    //OpenThreads::Mutex _mutex; need this ? 

    int _serverPort;
    int _localPort;
    bool _doRun; //if true thread is running, should only be true on master


    std::vector<DetectedCameraOrObject> _udpCameras;
    std::vector<DetectedCameraOrObject> _udpObstacle;
    std::vector<DetectedCameraOrObject> _udpROI;

    OpenThreads::Barrier _endBarrier; // braucht man das ??? 
    void processIncomingMessage(const Message& message);

    // returns positions in vec which was replaced, if no message was replaced adds new message to vec and returns -1
    int replaceMessage(std::vector<DetectedCameraOrObject>& vec, const Message& message, double timestamp); 
    
    // delete all messages and the corresponding Objects where no new data is received
    void deleteOutOfDateMessages();

    // make cooSystem of the demonstrator fit to Covise cooSystem
    void UDPMatrix2CoviseMatrix(Message& input) const ;       
public:
    UDP();
    ~UDP();

    void initUDP();
    void run() override;          //Master only sending and receiving Thread! 

    bool update();

};


