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
    int id;                    // each Objects has it's own id
    float translation[3];      // translation x,y,z
    float rotation[3];         // rotation as Euler rot um x, y, z

    void printMessage()
    {
        std::cout << "type:"<<(int)this->type <<" id:"<<this->id<<
        " x:"<<this->translation[0] <<" y:" << this->translation[1]<<" z:" <<this->translation[2] <<
        " euler x:"<<this->rotation[0] <<" euler y:" << this->rotation[1]<<" euler z:" <<this->rotation[2] <<
        std::endl;
    }
    static void printSize()
    {
        size_t sum = sizeof(type) + sizeof(id) + sizeof(translation) + sizeof(rotation);
        std::cout<<"Size of datatype 'Message': "<< sizeof(Message) <<" bytes" << ", sum of the individual datatypes: " << sum <<"bytes" <<std::endl;
    }
    static size_t getSize()
    {
        return sizeof(Message);
    }
};

struct MessageWithTimestamp
{
    double _timestamp;
    Message _message;

   MessageWithTimestamp(Message message, double timestamp)
    :_message(message), _timestamp(timestamp){};
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
    std::vector<MessageWithTimestamp> _udpROI;

    OpenThreads::Barrier _endBarrier; // braucht man das ??? 
    void processIncomingMessage(const Message& message);

    // returns positions in vec which was replaced, if no message was replaced adds new message to vec and returns -1
    int replaceMessage(std::vector<MessageWithTimestamp>& vec, const MessageWithTimestamp& message); 

    void deleteOutOfDateMessages(); // delete all messages and the corresponding Objects where no new data is received

    osg::Matrix MessageToMatrix( const Message& input);
    bool isSameId(const Message &ob1, const Message& ob2) const;
public:
    UDP();
    ~UDP();

    void initUDP();
    void run() override;          //Master only sending and receiving Thread! 

    void syncMasterSlave();       // sync master and slaves!
    bool update();

};


