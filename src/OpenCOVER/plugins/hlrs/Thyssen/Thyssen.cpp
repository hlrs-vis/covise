/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "Thyssen.h"
#include "Elevator.h"
#include "Exchanger.h"
#include "Landing.h"
#include "Car.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>

using namespace covise;

ThyssenPlugin *ThyssenPlugin::plugin = NULL;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeThyssen(scene);
}

// Define the built in VrmlNodeType:: "Thyssen" fields

VrmlNodeType *VrmlNodeThyssen::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Thyssen", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    char nameBuf[200];
    for(int i=0;i<4;i++)
    {
        sprintf(nameBuf,"carPos%d",i);
        t->addEventOut(nameBuf, VrmlField::SFVEC3F);
        sprintf(nameBuf,"carDoorClose%d",i);
        t->addEventOut(nameBuf, VrmlField::SFTIME);
        sprintf(nameBuf,"carDoorOpen%d",i);
        t->addEventOut(nameBuf, VrmlField::SFTIME);
        sprintf(nameBuf,"landingDoorClose%d",i);
        t->addEventOut(nameBuf, VrmlField::SFTIME);
        sprintf(nameBuf,"landingDoorOpen%d",i);
        t->addEventOut(nameBuf, VrmlField::SFTIME);
        sprintf(nameBuf,"carAngle%d",i);
        t->addEventOut(nameBuf, VrmlField::SFFLOAT);
        sprintf(nameBuf,"exchangerAngle%d",i);
        t->addEventOut(nameBuf, VrmlField::SFFLOAT);
    }
    

    return t;
}

VrmlNodeType *VrmlNodeThyssen::nodeType() const
{
    return defineType(0);
}

VrmlNodeThyssen::VrmlNodeThyssen(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_enabled(true)
{
    setModified();
}

VrmlNodeThyssen::VrmlNodeThyssen(const VrmlNodeThyssen &n)
    : VrmlNodeChild(n.d_scene)
    , d_enabled(n.d_enabled)
{
    setModified();
}

VrmlNodeThyssen::~VrmlNodeThyssen()
{
}

VrmlNode *VrmlNodeThyssen::cloneMe() const
{
    return new VrmlNodeThyssen(*this);
}

VrmlNodeThyssen *VrmlNodeThyssen::toThyssen() const
{
    return (VrmlNodeThyssen *)this;
}

ostream &VrmlNodeThyssen::printFields(ostream &os, int indent)
{
    if (!d_enabled.get())
        PRINT_FIELD(enabled);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeThyssen::setField(const char *fieldName,
                               const VrmlField &fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeThyssen::getField(const char *fieldName)
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "floats_changed") == 0)
        return &d_floats;
    else if (strcmp(fieldName, "ints_changed") == 0)
        return &d_ints;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeThyssen::eventIn(double timeStamp,
                              const char *eventName,
                              const VrmlField *fieldValue)
{

    // Check exposedFields
    //else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeThyssen::render(Viewer *)
{
    if (!d_enabled.get())
        return;
    
    char pname[100];
    double timeStamp = System::the->time();
    for(int i=0;i<ThyssenPlugin::plugin->cars.size();i++)
    {
        carData &cd = ThyssenPlugin::plugin->cars[i];
        d_carPos[i].set( 0, cd.posZ / 1000.0, -cd.posY / 1000.0);
        sprintf(pname,"carPos%d",i);
        eventOut(timeStamp, pname, d_carPos[i]);
        if(cd.doorState != cd.oldDoorState)
        {
           /* if(i==0)
            {
                fprintf(stderr,"doorState: %d , oldState = %d\n",cd.doorState,cd.oldDoorState);
            }*/
            if(cd.doorState == ThyssenPlugin::opening)
            {
                d_carDoorOpen[i].set(timeStamp);
                sprintf(pname,"carDoorOpen%d",i);
                eventOut(timeStamp, pname, d_carDoorOpen[i]);
            }
            if(cd.doorState == ThyssenPlugin::closing)
            {
                d_carDoorClose[i].set(timeStamp);
                sprintf(pname,"carDoorClose%d",i);
                eventOut(timeStamp, pname, d_carDoorClose[i]);
            }
            cd.oldDoorState = cd.doorState;
        }
    }
    for(int i=0;i<ThyssenPlugin::plugin->exchangers.size();i++)
    {
        exchangerData &ed = ThyssenPlugin::plugin->exchangers[i];
        if(ed.oldAngle != ed.swvlRotaryMotor)
        {
            ed.oldAngle = ed.swvlRotaryMotor;
            d_exchangerAngle[i].set(ed.swvlRotaryMotor/180.0*M_PI);
            sprintf(pname,"exchangerAngle%d",i);
            eventOut(timeStamp, pname, d_exchangerAngle[i]);
            if(ed.linkedCar >=0 && ed.linkedCar < ThyssenPlugin::plugin->cars.size())
            {
                carData &cd = ThyssenPlugin::plugin->cars[ed.linkedCar];
                d_carAngle[i].set(ed.swvlRotaryMotor/180.0*M_PI);
                sprintf(pname,"carAngle%d",i);
                eventOut(timeStamp, pname, d_carAngle[i]);
            }
            else
            {
                fprintf(stderr,"no car in exchanger %d\n",ed.exID);
            }
        }
    }
}

ThyssenPlugin::ThyssenPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "ThyssenPlugin::ThyssenPlugin\n");

    plugin = this;
    conn = NULL;
    sConn=NULL;
    if(coVRMSController::instance()->isMaster())
    {

    port = coCoviseConfig::getInt("COVER.Plugin.ThyssenPlugin.TCPPort", 52051);

    sConn = new ServerConnection(port, 1234, 0);

    if (!sConn->getSocket())
    {
        cout << "tried to open server Port " << port << endl;
        cout << "Creation of server failed!" << endl;
        cout << "Port-Binding failed! Port already bound?" << endl;
        delete sConn;
        sConn = NULL;
    }
    }
    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    cout << "Set socket options..." << endl;
    if (sConn)
    {
        setsockopt(sConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

        cout << "Set server to listen mode..." << endl;
        sConn->listen();
        if (!sConn->is_connected()) // could not open server port
        {
            fprintf(stderr, "Could not open server port %d\n", port);
            delete sConn;
            sConn = NULL;
        }
    }

    numFloats = 62;
    numInts = 61;

    floatValues = new float[numFloats];
    intValues = new int[numInts];
}

// this is called if the plugin is removed at runtime
ThyssenPlugin::~ThyssenPlugin()
{
    fprintf(stderr, "ThyssenPlugin::~ThyssenPlugin\n");

    delete[] floatValues;
    delete[] intValues;
}

bool ThyssenPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeThyssen::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeElevator::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeCar::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeExchanger::defineType());
    VrmlNamespace::addBuiltIn(VrmlNodeLanding::defineType());

    return true;
}

bool ThyssenPlugin::readVal(void *buf, unsigned int numBytes)
{
    unsigned int toRead = numBytes;
    unsigned int numRead = 0;
    int readBytes = 0;
    while (numRead < numBytes)
    {
        readBytes = conn->getSocket()->Read(((unsigned char *)buf) + readBytes, toRead);
        if (readBytes < 0)
        {
            cerr << "error reading data from socket" << endl;
            return false;
        }
        numRead += readBytes;
        toRead = numBytes - numRead;
    }
    return true;
}

carData::carData(int id)
{
    carID = id;
}
carData::~carData()
{
}
void carData::setData(TokenBuffer &tb)
{
    tb >> posY;
    tb >> posZ;
    tb >> speed;
    tb >> accel;
    tb >> doorState;
    tb >> direction;
    tb >> floor;
    tb >> hzllockState;
    tb >> vtllockState;
    tb >> hzlproxState;
    tb >> vtlproxState;
}

exchangerData::exchangerData(int id)
{
    exID = id;
}
exchangerData::~exchangerData()
{
}
void exchangerData::setData(TokenBuffer &tb)
{
    tb >> posY;
    tb >> posZ;
    tb >> swvlhzllckStatus;
    tb >> swvlvtllckStatus;
    tb >> swvlproxStatus;
    tb >> cbnlckStatus;
    tb >> cbnlckproxStatus;
    tb >> swvlRotaryMotor;
    tb >> linkedCar;
    tb >> destnPos;
}

brakeData::brakeData(int id)
{
    brakeID = id;
}
brakeData::~brakeData()
{
}
void brakeData::setData(TokenBuffer &tb)
{
    tb >> type;
    tb >> status;
}

int ThyssenPlugin::readData(char *buf,unsigned int size)
{
    int numBytes;
    unsigned int numReceived=0;
    if (coVRMSController::instance()->isMaster())
    {
        do {
            numBytes = conn->getSocket()->Read(buf,size);
            if(numBytes >0)
            {
                numReceived += numBytes;
            }
            else 
            {
                conn = NULL;
                buf[0]=-128;
                buf[1]=-128;
                coVRMSController::instance()->sendSlaves(buf, size);
                return -1;
            }
        } while(conn !=NULL && numReceived < size);
        coVRMSController::instance()->sendSlaves(buf, size);
    }
    else
    {
        coVRMSController::instance()->readMaster(buf, size);
        if((unsigned char)buf[0]==0xff && (unsigned char)buf[1]==0xff)
        {
            return -1;
        }
    }
    return(size);
}

void
ThyssenPlugin::preFrame()
{

    if (sConn && sConn->is_connected() && sConn->check_for_input()) // we have a server and received a connect
    {
        //   std::cout << "Trying serverConn..." << std::endl;
        conn = sConn->spawnSimpleConnection();
        if (conn && conn->is_connected())
        {
            std::cerr << "Client connected" << endl;
        }
    }

    if (conn)
    {
        while (conn->check_for_input())
        {
            char buf[200 * sizeof(int)];

            int numBytes;
            if((numBytes = readData(buf,2 * sizeof(int)))<0)
                break;
            if (buf[0] == 't' && buf[1] == 'h' && buf[2] == 'i')
            {
                conn->getSocket()->Read(buf, 6);
                break;
            }
            TokenBuffer tb(buf, numBytes, true);
            int msSize;
            int msType;
            int totalSize=numBytes;
            tb >> msSize;
            tb >> msType;
            if (msType == CAR_DATA)
            {
                if((numBytes = readData(buf,4 * sizeof(int)))<0)
                    break;
                totalSize+=numBytes;
                TokenBuffer tb(buf, numBytes, true);
                int numberOfCars;
                int numberOfExchangers;
                int numberOfcarBrakes;
                int numberOfexchBrakes;
                tb >> numberOfCars;
                tb >> numberOfExchangers;
                tb >> numberOfcarBrakes;
                tb >> numberOfexchBrakes;
                numberOfcarBrakes /= numberOfCars;
                numberOfexchBrakes /= numberOfExchangers;
                for (int i = 0; i < numberOfCars; i++)
                {
                    if((numBytes = readData(buf,8 * sizeof(int) + 4 * sizeof(float)))<0)
                        break;
                    
                    totalSize+=numBytes;
                    TokenBuffer tb(buf, numBytes, true);
                    int carID;
                    tb >> carID;
                    if (i >= cars.size())
                    {
                        carData d(carID);
                        cars.push_back(d);
                    }
                    cars.at(i).setData(tb);
                }
                if(conn==NULL)
                    break;
                for (int i = 0; i < numberOfExchangers; i++)
                {
                    if((numBytes = readData(buf,7 * sizeof(int) + 4 * sizeof(float)))<0)
                        break;
                    totalSize+=numBytes;
                    TokenBuffer tb(buf, numBytes, true);
                    int exID;
                    tb >> exID;
                    if (i >= exchangers.size())
                    {
                        exchangerData d(exID);
                        exchangers.push_back(d);
                    }
                    exchangers.at(i).setData(tb);
                }
                if(conn==NULL)
                    break;
                for (int i = 0; i < numberOfCars; i++)
                {
                    if((numBytes = readData(buf,sizeof(int) * 3 * numberOfcarBrakes))<0)
                        break;
                    totalSize+=numBytes;
                    TokenBuffer tb(buf, numBytes, true);
                    carData &car = cars.at(i);
                    for (int n = 0; n < numberOfcarBrakes; n++)
                    {
                        int brakeID;
                        tb >> brakeID;
                        if (car.brakes.size() <= n)
                        {
                            brakeData d(brakeID);
                            car.brakes.push_back(d);
                        }
                        car.brakes.at(n).setData(tb);
                    }
                }
                if(conn==NULL)
                    break;
                for (int i = 0; i < numberOfExchangers; i++)
                {
                    if((numBytes = readData(buf,sizeof(int) * 3 * numberOfexchBrakes))<0)
                        break;
                    totalSize+=numBytes;
                    TokenBuffer tb(buf, numBytes, true);
                    exchangerData &exchanger = exchangers.at(i);
                    for (int n = 0; n < numberOfexchBrakes; n++)
                    {
                        int brakeID;
                        tb >> brakeID;
                        if (exchanger.brakes.size() <= n)
                        {
                            brakeData d(brakeID);
                            exchanger.brakes.push_back(d);
                        }
                        exchanger.brakes.at(n).setData(tb);
                    }
                }
                if(conn==NULL)
                    break;
                if(totalSize!=msSize)
                {
                    fprintf(stderr,"Wrong messageSize %d in Message Type %d, expected %d\n",totalSize,msType,msSize);
                }
            }

        }
    }

    
}

COVERPLUGIN(ThyssenPlugin)
