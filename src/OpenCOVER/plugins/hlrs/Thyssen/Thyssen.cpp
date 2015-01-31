/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "Thyssen.h"

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
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addEventOut("car0Pos", VrmlField::SFVEC3F);
    t->addEventOut("car1Pos", VrmlField::SFVEC3F);
    t->addEventOut("car2Pos", VrmlField::SFVEC3F);
    t->addEventOut("ints_changed", VrmlField::MFINT32);
    t->addEventOut("floats_changed", VrmlField::MFFLOAT);

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

    double timeStamp = System::the->time();
    if (ThyssenPlugin::plugin->numFloats)
    {
        d_floats.set(ThyssenPlugin::plugin->numFloats, ThyssenPlugin::plugin->floatValues);
        eventOut(timeStamp, "floats_changed", d_floats);
    }
    if (ThyssenPlugin::plugin->numInts)
    {
        d_ints.set(ThyssenPlugin::plugin->numInts, ThyssenPlugin::plugin->intValues);
		eventOut(timeStamp, "ints_changed", d_ints);
	}
	if(ThyssenPlugin::plugin->cars.size()>0)
	{
		d_car0Pos.set(ThyssenPlugin::plugin->cars[0].posY / 1000.0, ThyssenPlugin::plugin->cars[0].posZ / 1000.0, 0);
		eventOut(timeStamp, "car0Pos", d_car0Pos);
	}
	if(ThyssenPlugin::plugin->cars.size()>1)
	{
		d_car0Pos.set(ThyssenPlugin::plugin->cars[1].posY / 1000.0, ThyssenPlugin::plugin->cars[1].posZ / 1000.0, 0);
		eventOut(timeStamp, "car1Pos", d_car1Pos);
	}
	if(ThyssenPlugin::plugin->cars.size()>2)
	{
		d_car0Pos.set(ThyssenPlugin::plugin->cars[2].posY / 1000.0, ThyssenPlugin::plugin->cars[2].posZ / 1000.0, 0);
		eventOut(timeStamp, "car2Pos", d_car2Pos);
	}
}

ThyssenPlugin::ThyssenPlugin()
{
    fprintf(stderr, "ThyssenPlugin::ThyssenPlugin\n");

    plugin = this;
    conn = NULL;

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
    delete conn;
}

bool ThyssenPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeThyssen::defineType());

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
			char buf[200*sizeof(int)];
			
			int numBytes;
			if (coVRMSController::instance()->isMaster())
			{
				numBytes = conn->getSocket()->Read(buf,2*sizeof(int));
				coVRMSController::instance()->sendSlaves(buf,2*sizeof(int));
			}
			else
			{
				coVRMSController::instance()->readMaster(buf,2*sizeof(int));
				numBytes = 2*sizeof(int);
			}
			if(buf[0]=='t' && buf[1]=='h' && buf[2]=='i')
			{
			    conn->getSocket()->Read(buf,6);
				break;
			}
			TokenBuffer tb(buf,numBytes,true);
			int msSize;
			int msType;
			tb >> msSize;
			tb >> msType;
			if(msType == CAR_DATA)
			{
				if (coVRMSController::instance()->isMaster())
				{
					numBytes = conn->getSocket()->Read(buf,4*sizeof(int));
					coVRMSController::instance()->sendSlaves(buf,4*sizeof(int));
				}
				else
				{
					coVRMSController::instance()->readMaster(buf,4*sizeof(int));
				    numBytes = 4*sizeof(int);
				}
				TokenBuffer tb(buf,numBytes,true);
				int numberOfCars;
				int numberOfExchangers;
				int numberOfcarBrakes;
				int numberOfexchBrakes;
				tb >> numberOfCars;
				tb >> numberOfExchangers;
				tb >> numberOfcarBrakes;
				tb >> numberOfexchBrakes;
				numberOfcarBrakes/=numberOfCars;
				numberOfexchBrakes/=numberOfExchangers;
			    for (int i=0; i<numberOfCars; i++)
				{
					if (coVRMSController::instance()->isMaster())
					{
						numBytes = conn->getSocket()->Read(buf,8*sizeof(int)+8*sizeof(float));
						coVRMSController::instance()->sendSlaves(buf,8*sizeof(int)+4*sizeof(float));
					}
					else
					{
						coVRMSController::instance()->readMaster(buf,8*sizeof(int)+4*sizeof(float));
						numBytes = 8*sizeof(int)+4*sizeof(float);
					}
					TokenBuffer tb(buf,numBytes,true);
					int carID;
					tb >> carID;
					if(i >= cars.size())
					{
						carData d(carID);
						cars.push_back(d);
					}
					cars.at(i).setData(tb);
	
				}
				for (int i=0; i<numberOfExchangers; i++)
				{
					if (coVRMSController::instance()->isMaster())
					{
						numBytes = conn->getSocket()->Read(buf,7*sizeof(int)+8*sizeof(float));
						coVRMSController::instance()->sendSlaves(buf,7*sizeof(int)+4*sizeof(float));
					}
					else
					{
						coVRMSController::instance()->readMaster(buf,7*sizeof(int)+4*sizeof(float));
						numBytes = 7*sizeof(int)+4*sizeof(float);
					}
					TokenBuffer tb(buf,numBytes,true);
					int exID;
					tb >> exID;
					if(i >= exchangers.size())
					{
						exchangerData d(exID);
						exchangers.push_back(d);
					}
					exchangers.at(i).setData(tb);
				}
				for (int i=0; i<numberOfCars; i++)
				{
					if (coVRMSController::instance()->isMaster())
					{
						numBytes = conn->getSocket()->Read(buf,sizeof(int)*3*numberOfcarBrakes);
						coVRMSController::instance()->sendSlaves(buf,sizeof(int)*3*numberOfcarBrakes);
					}
					else
					{
						coVRMSController::instance()->readMaster(buf,sizeof(int)*3*numberOfcarBrakes);
						numBytes = sizeof(int)*3*numberOfcarBrakes;
					}
					TokenBuffer tb(buf,numBytes,true);
					carData &car = cars.at(i);
					for(int n=0;n<numberOfcarBrakes;n++)
					{
						int brakeID;
						tb >> brakeID;
						if(car.brakes.size()<=n)
						{
						    brakeData d(brakeID);
						    car.brakes.push_back(d);
						}
						car.brakes.at(n).setData(tb);
					}
	
				}
				for (int i=0; i<numberOfExchangers; i++)
				{
					if (coVRMSController::instance()->isMaster())
					{
						numBytes = conn->getSocket()->Read(buf,sizeof(int)*3*numberOfcarBrakes);
						coVRMSController::instance()->sendSlaves(buf,sizeof(int)*3*numberOfcarBrakes);
					}
					else
					{
						coVRMSController::instance()->readMaster(buf,sizeof(int)*3*numberOfcarBrakes);
						numBytes = sizeof(int)*3*numberOfcarBrakes;
					}
					TokenBuffer tb(buf,numBytes,true);
					exchangerData &exchanger = exchangers.at(i);
					for(int n=0;n<numberOfexchBrakes;n++)
					{
						int brakeID;
						tb >> brakeID;
						if(exchanger.brakes.size()<=n)
						{
						    brakeData d(brakeID);
						    exchanger.brakes.push_back(d);
						}
						exchanger.brakes.at(n).setData(tb);
					}
	
				}
			}

        /* old ASCII Protocoll    const char *line = conn->readLine();
            if (line != NULL)
            {
                if (strncmp(line, "CARID0", 6) == 0)
                {
                    sscanf(line + 12, "%f, PosY %f", &zpos, &ypos);

                    //cerr << line <<  "z" << zpos << endl;
                }

				ExCHID4 PosZ 153850.000000 ,  PosY 3000.000000
CARID0 PosZ 113020.000000 ,  PosY 3000.000000
CARID1 PosZ 153850.000000 ,  PosY 3000.000000
CARID2 PosZ 5000.000000 ,  PosY 3000.000000
CARID3 PosZ 34420.000000 ,  PosY 3000.000000
ExCHID0 PosZ 5000.000000 ,  PosY 3000.000000
ExCHID1 PosZ 34420.000000 ,  PosY 3000.000000
ExCHID2 PosZ 44200.000000 ,  PosY 5700.000000
ExCHID3 PosZ 94200.000000 ,  PosY 3000.000000
ExCHID4 PosZ 153850.000000 ,  PosY 3000.000000
CARID0 PosZ 113050.000000 ,  PosY 3000.000000
CARID1 PosZ 153850.000000 ,  PosY 3000.000000
CARID2 PosZ 5000.000000 ,  PosY 3000.000000
CARID3 PosZ 34420.000000 ,  PosY 3000.000000
ExCHID0 PosZ 5000.000000 ,  PosY 3000.000000
ExCHID1 PosZ 34420.000000 ,  PosY 3000.000000
ExCHID2 PosZ 44200.000000 ,  PosY 5700.000000
ExCHID3 PosZ 94200.000000 ,  PosY 3000.000000
ExCHID4 PosZ 153850.000000 ,  PosY 3000.000000
CARID0 PosZ 113230.000000 ,  PosY 3000.000000*/
            
        }
    }

    if (coVRMSController::instance()->isMaster())
    {
        //cerr << numFloats << endl;
        //cerr << numInts << endl;
        /*   coVRMSController::instance()->sendSlaves((char *)&numFloats,sizeof(int));
      coVRMSController::instance()->sendSlaves((char *)&numInts,sizeof(int));
      if(numFloats)
         coVRMSController::instance()->sendSlaves((char *)floatValues,numFloats*sizeof(float));
      if(numInts)
         coVRMSController::instance()->sendSlaves((char *)intValues,numInts*sizeof(int));*/
    }
    else
    {
        /*  int newNumFloats=0;
      int newNumInts=0;
      coVRMSController::instance()->readMaster((char *)&newNumFloats,sizeof(int));
      coVRMSController::instance()->readMaster((char *)&newNumInts,sizeof(int));
      //cerr << newNumFloats << endl;
      //cerr << newNumInts << endl;
      if(newNumFloats>0 && newNumFloats != numFloats)
      {
      cerr << "resize" << endl;
         numFloats=newNumFloats;
         delete[] floatValues;
         floatValues = new float[numFloats];
      }
      if(newNumInts > 0 && newNumInts != numInts)
      {
      cerr << "resize" << endl;
         numInts=newNumInts;
         delete[] intValues;
         intValues = new int[numInts];
      }
      if(newNumFloats>0 && numFloats)
      {
      //cerr << "rf" << endl;
         coVRMSController::instance()->readMaster((char *)floatValues,numFloats*sizeof(float));
      }
      if(newNumFloats>0 && numInts)
      {
      //cerr << "ri" << endl;
         coVRMSController::instance()->readMaster((char *)intValues,numInts*sizeof(int));
      }*/
    }
}

COVERPLUGIN(ThyssenPlugin)
