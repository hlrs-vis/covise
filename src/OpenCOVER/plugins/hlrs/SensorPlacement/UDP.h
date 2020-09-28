#pragma once

#include "UDPComm.h"
#include <cover/coVRPluginSupport.h>


enum class MessageType
{
    Camera = 0,
    Obstacle = 1,
    ROI = 2
};


// this is the data structure of the incoming UDP message
struct Message
{   
    MessageType type;
    int cameraID;              // id of camera, which can see the object
    int id;                    // each Objects has it's own id
    osg::Matrixf matrix;
    float distanceCamera;      // distance from camera to center of object 

    void printMessage()
    {
        std::cout << "type:"<<(int)this->type <<" id:"<<this->id<<
        " x:"<<this->matrix.getTrans().x() <<" y:" << this->matrix.getTrans().y()<<" z:" <<this->matrix.getTrans().z() << std::endl;
    }
    static void printSize()
    {
        size_t sum = sizeof(type) + sizeof(cameraID) + sizeof(id) + sizeof(matrix) + sizeof(distanceCamera);
        size_t sumMatrix = sizeof(sumMatrix);

        std::cout<<"Size of datatype 'Message': "<< sizeof(Message) <<" bytes" << ", sum of the individual datatypes: " << sum <<"bytes" <<std::endl;
        //std::cout<<"Size of osg::Matrixf: "<< sizeof(matrix) <<" bytes" << std::endl;
    }
    static size_t getSize()
    {
        return sizeof(Message);
    }
};

// camera id with corresponding timestamp of last incoming message
struct CameraIDwithTimestamp
{
    int _cameraID;
    double _timestamp;
    osg::Matrixf _matrix;
};

struct MessageWithTimestamp
{
    double _timestamp;
    Message _message;

   MessageWithTimestamp(Message message, double timestamp)
    :_message(message), _timestamp(timestamp){};
};

struct DetectedObject
{
    struct Camera //Camera which can see the detected Object
    {
        int _cameraID;
        double _timestamp;
        osg::Matrixf _objectMatrix; // Matrix of the detected Object calculated from this camera
        float _distance; // Distance from this camera to the object

        Camera(Message message, double timestamp)
        :_cameraID(message.cameraID),_timestamp(timestamp), _objectMatrix(message.matrix), _distance(message.distanceCamera)
        {};
    };

    MessageType _type;
    int _id;                      // id of this object
    std::vector<Camera> _cameras; // all cameras, which can see this object

    DetectedObject(Message message, double timestamp)
    :_type(message.type), _id(message.id)
    {
        _cameras.push_back(Camera(message, timestamp));
    };

    void addCamera(Message message, double timestamp)
    {
        _cameras.push_back(Camera(message, timestamp));
    };

    bool deleteObject(const float& maxValue, const double& timestamp) // delete this object if no camera can see it
    {
        if(!_cameras.empty())
        {
            size_t count{0};
            for(const auto& cam : _cameras)
            {
                if(timestamp - cam._timestamp > maxValue)
                {
                    _cameras.erase(_cameras.begin() + count);
                    if(_cameras.empty())
                    {
                        return true;
                    }
                }
                count++;
            }
        }

        return false;
    }
    

    int getNbrOfCameras()const {return _cameras.size();}

    osg::Matrix getAverageMatrix()
    {
        std::cout << "Function not implementetd yet" << std::endl;
        return osg::Matrix();
    };

    osg::Matrix getMatrixFromClosestCamera()
    {
        std::vector<Camera>::iterator result = std::min_element(_cameras.begin(), _cameras.end(),[](const Camera& cam1, const Camera& cam2)
        {
            return cam1._distance < cam2._distance;
        }); 

        std::cout<<"smallest distance" << _cameras.at(std::distance(_cameras.begin(), result))._distance << std::endl;
        return _cameras.at(std::distance(_cameras.begin(), result))._objectMatrix;
    };
};

class UDP : public OpenThreads::Thread
{
private:
    std::unique_ptr<UDPComm> _udp;
    //OpenThreads::Mutex _mutex; need this ? 

    int _serverPort;
    int _localPort;
    bool _doRun; //if true thread is running, should only be true on master


    std::vector<MessageWithTimestamp> _udpCameras;
    std::vector<DetectedObject> _udpObstacle;
    std::vector<DetectedObject> _udpROI;

    OpenThreads::Barrier _endBarrier; // braucht man das ??? 
    void processIncomingMessage(const Message& message);

    // returns positions in vec which was replaced, if no message was replaced adds new message to vec and returns -1
    int replaceMessage(std::vector<MessageWithTimestamp>& vec, const MessageWithTimestamp& message);    // don't need this anymore!
    int replaceMessage(std::vector<DetectedObject>& vec, const Message& message, double timestamp); 


    void deleteOutOfDateMessages(); // delete all messages and the corresponding Objects where no new data is received


    //osg::Matrix MessageToMatrix( const Message& input);
    void UDPMatrix2CoviseMatrix(Message& input) const ;        // make cooSytstem of the demonstrator fit to Covise cooSystem
    bool isSameId(const Message &ob1, const Message& ob2) const;
public:
    UDP();
    ~UDP();

    void initUDP();
    void run() override;          //Master only sending and receiving Thread! 

    void syncMasterSlave();       // sync master and slaves!
    bool update();

};


