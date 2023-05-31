/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "Simulink.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <util/unixcompat.h>

SimulinkPlugin *SimulinkPlugin::plugin = NULL;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeSimulink(scene);
}

// Define the built in VrmlNodeType:: "Simulink" fields

VrmlNodeType *VrmlNodeSimulink::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Simulink", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addEventOut("ints_changed", VrmlField::MFINT32);
    t->addEventOut("floats_changed", VrmlField::MFFLOAT);
    t->addEventIn("intsIn", VrmlField::MFINT32);
    t->addEventIn("floatsIn", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeSimulink::nodeType() const
{
    return defineType(0);
}

VrmlNodeSimulink::VrmlNodeSimulink(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_enabled(true)
{
    setModified();
}

VrmlNodeSimulink::VrmlNodeSimulink(const VrmlNodeSimulink &n)
    : VrmlNodeChild(n.d_scene)
    , d_enabled(n.d_enabled)
{
    setModified();
}

VrmlNodeSimulink::~VrmlNodeSimulink()
{
}

VrmlNode *VrmlNodeSimulink::cloneMe() const
{
    return new VrmlNodeSimulink(*this);
}

VrmlNodeSimulink *VrmlNodeSimulink::toSimulink() const
{
    return (VrmlNodeSimulink *)this;
}

ostream &VrmlNodeSimulink::printFields(ostream &os, int indent)
{
    if (!d_enabled.get())
        PRINT_FIELD(enabled);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeSimulink::setField(const char *fieldName,
                               const VrmlField &fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeSimulink::getField(const char *fieldName)
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "floats_changed") == 0)
        return &d_floats;
    else if (strcmp(fieldName, "intsIn") == 0)
        return &d_intsIn;
    else if (strcmp(fieldName, "floatsIn") == 0)
        return &d_floatsIn;
    else if (strcmp(fieldName, "ints_changed") == 0)
        return &d_ints;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeSimulink::eventIn(double timeStamp,
                              const char *eventName,
                              const VrmlField *fieldValue)
{

    // Check exposedFields
    //else
    if (strcmp(eventName, "floatsIn") == 0 || strcmp(eventName, "intsIn") == 0)
    {
        SimulinkPlugin::plugin->sendData(d_floatsIn.size(), d_floatsIn.get(),d_intsIn.size(),d_intsIn.get());
    }

    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void VrmlNodeSimulink::render(Viewer *)
{
    if (!d_enabled.get())
        return;

    double timeStamp = System::the->time();
    if (SimulinkPlugin::plugin->numFloats)
    {
        d_floats.set(SimulinkPlugin::plugin->numFloats, SimulinkPlugin::plugin->floatValues);
        eventOut(timeStamp, "floats_changed", d_floats);
    }
    if (SimulinkPlugin::plugin->numInts)
    {
        d_ints.set(SimulinkPlugin::plugin->numInts, SimulinkPlugin::plugin->intValues);
        eventOut(timeStamp, "ints_changed", d_ints);
    }

	/*fprintf(stderr, "%d", SimulinkPlugin::plugin->numFloats);
	for (int i = 0; i < SimulinkPlugin::plugin->numFloats; i++)
		fprintf(stderr, "%f;", SimulinkPlugin::plugin->floatValues[i]);
	fprintf(stderr, "\n");
	fprintf(stderr, "%d", SimulinkPlugin::plugin->numInts);
	for (int i = 0; i < SimulinkPlugin::plugin->numInts; i++)
		fprintf(stderr, "%d;", SimulinkPlugin::plugin->intValues[i]);
	fprintf(stderr, "\n");*/
}

void SimulinkPlugin::sendData(int numFloats, float* floats, int numInts, int* ints)
{
    int bufSize= numFloats * sizeof(float) + numInts * sizeof(int) + 2 * sizeof(int);
    char* buf = new char[bufSize];
    *((int*)buf) = numFloats;
    *(((int*)buf)+1) = numInts;
    memcpy(buf + 2 * sizeof(int), floats, numFloats * sizeof(float));
    memcpy(buf + 2 * sizeof(int)+ numFloats * sizeof(float), ints, numInts * sizeof(int));
    udp->send(buf, bufSize);
}

SimulinkPlugin::SimulinkPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "SimulinkPlugin::SimulinkPlugin\n");

    plugin = this;
    conn = NULL;

    serverHost = NULL;
    localHost = new Host("localhost");

    serverPort = covise::coCoviseConfig::getInt("COVER.Plugin.Simulink.serverPort", 31319);
    localPort = covise::coCoviseConfig::getInt("COVER.Plugin.Simulink.localPort", 12345);
    port = coCoviseConfig::getInt("COVER.Plugin.Simulink.TCPPort", 12345);
    std::string line = coCoviseConfig::getEntry("COVER.Plugin.Simulink.Server");
    if (!line.empty())
    {
        if (strcasecmp(line.c_str(), "NONE") == 0)
            serverHost = NULL;
        else
            serverHost = new Host(line.c_str());
        cerr << serverHost->getName() << endl;
    }
    else
    {
        serverHost = new Host("localhost");
    }

    numFloats = 0;
    numInts = 0;

    floatValues = new float[numFloats];
    intValues = new int[numInts];
}

// this is called if the plugin is removed at runtime
SimulinkPlugin::~SimulinkPlugin()
{
    fprintf(stderr, "SimulinkPlugin::~SimulinkPlugin\n");

    delete[] floatValues;
    delete[] intValues;
    delete conn;
    delete serverHost;
}

bool SimulinkPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeSimulink::defineType());
    bool ret = false;
    if (coVRMSController::instance()->isMaster())
    {
        udp = new UDPComm(serverHost->getName(), serverPort, localPort);
        if (!udp->isBad())
        {
            ret = true;
        }
        else
        {
            std::cerr << "Simulink Plugin: falided to open local UDP port" << localPort << std::endl;
            ret = false;
        }
        coVRMSController::instance()->sendSlaves(&ret, sizeof(ret));
    }
    else
    {
        coVRMSController::instance()->readMaster(&ret, sizeof(ret));
    }
    return ret;
}

bool SimulinkPlugin::readVal(void *buf, unsigned int numBytes)
{
    unsigned int toRead = numBytes;
    unsigned int numRead = 0;
    int readBytes = 0;
    while (numRead < numBytes)
    {
		if (conn == NULL)
			return false;
		if (conn->is_connected() == false)
			return false;
        readBytes = conn->getSocket()->Read(((unsigned char *)buf) + numRead, toRead);
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

bool
SimulinkPlugin::update()
{
    if (udp!=nullptr)
    {
        
        char tmpbuf[1000];
        int ret = 0;
        bool receivedData = false;
            do {
                ret = udp->receive(tmpbuf, 1000,0); if (ret > -1) {
                    if (ret != *((int*)tmpbuf) * sizeof(float) + *(((int*)tmpbuf) + 1) * sizeof(int) + 2 * sizeof(int)) {
                        std::cerr << "wrong number of bytes";
                    }
                    else {
                        receivedData = true;
                    }
                }
            } while (ret >= 2 * 4);
        if(receivedData)
        {
            int newNumFloats = 0; // should read these numbers from the server!!
            int newNumInts = 0; // should read these numbers from the server!!
            newNumFloats = *((int*)tmpbuf);
            newNumInts = *(((int*)tmpbuf)+1);
            if (newNumFloats > 0 && newNumFloats != numFloats)
            {
                numFloats = (int)newNumFloats;
                delete[] floatValues;
                floatValues = new float[numFloats];
            }
            if (newNumInts > 0 && newNumInts != numInts)
            {
                numInts = (int)newNumInts;
                delete[] intValues;
                intValues = new int[numInts];
            }
            memcpy(floatValues, tmpbuf + 2 * sizeof(int), numFloats * sizeof(float));
            memcpy(intValues, tmpbuf + 2 * sizeof(int)+ numFloats * sizeof(float), numInts * sizeof(int));
        }

    }
    /*if (conn)
    {
        while (conn->check_for_input())
        {
            int newNumFloats = 6; // should read these numbers from the server!!
            int newNumInts = 1; // should read these numbers from the server!!
            if (!readVal(&newNumFloats, sizeof(int)))
            {
                delete conn;
                conn = NULL;
                newNumFloats = 0;
                newNumInts = 0;
                numFloats = 0;
                numInts = 0;
                cerr << "reset " << newNumInts << endl;
            }
            if (!readVal(&newNumInts, sizeof(int)))
            {
                delete conn;
                conn = NULL;
                newNumFloats = 0;
                newNumInts = 0;
                numFloats = 0;
                numInts = 0;
                cerr << "reseti " << newNumInts << endl;
            }
            if (newNumFloats > 0 && newNumFloats != numFloats)
            {
                numFloats = (int)newNumFloats;
                delete[] floatValues;
                floatValues = new float[numFloats];
            }
            if (newNumInts > 0 && newNumInts != numInts)
            {
                numInts = (int)newNumInts;
                delete[] intValues;
                intValues = new int[numInts];
            }
            if (!readVal(floatValues, numFloats * sizeof(float)))
            {
                delete conn;
                conn = NULL;
                newNumFloats = 0;
                newNumInts = 0;
                numFloats = 0;
                numInts = 0;
                cerr << "reseti2 " << newNumInts << endl;
            }
            if (!readVal(intValues, numInts * sizeof(int)))
            {
                delete conn;
                conn = NULL;
                newNumFloats = 0;
                newNumInts = 0;
                numFloats = 0;
                numInts = 0;
                cerr << "reseti2 " << newNumInts << endl;
            }
        }
    }*/
    /*else if ((coVRMSController::instance()->isMaster()) && (serverHost != NULL))
    {
        // try to connect to server every 2 secnods
        if ((cover->frameTime() - oldTime) > 2)
        {
            conn = new SimpleClientConnection(serverHost, port, 0);

            if (!conn->is_connected()) // could not open server port
            {
#ifndef _WIN32
                if (errno != ECONNREFUSED)
                {
                    fprintf(stderr, "Could not connect to Simulink on %s; port %d\n", serverHost->getName(), port);
                    delete serverHost;
                    serverHost = NULL;
                }
#endif
                delete conn;
                conn = NULL;
                conn = new SimpleClientConnection(localHost, port, 0);
                if (!conn->is_connected()) // could not open server port
                {
#ifndef _WIN32
                    if (errno != ECONNREFUSED)
                    {
                        fprintf(stderr, "Could not connect to Simulink on %s; port %d\n", localHost->getName(), port);
                    }
#endif
                    delete conn;
                    conn = NULL;
                }
                else
                {
                    fprintf(stderr, "Connected to Simulink on %s; port %d\n", localHost->getName(), port);
                }
            }
            else
            {
                fprintf(stderr, "Connected to Simulink on %s; port %d\n", serverHost->getName(), port);
            }
            if (conn && conn->is_connected())
            {
                int id = 2;
                conn->getSocket()->write(&id, sizeof(id));
            }
            oldTime = cover->frameTime();
        }
    }*/
    if (coVRMSController::instance()->isMaster())
    {
        //cerr << numFloats << endl;
        //cerr << numInts << endl;
        coVRMSController::instance()->sendSlaves((char *)&numFloats, sizeof(int));
        coVRMSController::instance()->sendSlaves((char *)&numInts, sizeof(int));
        if (numFloats)
            coVRMSController::instance()->sendSlaves((char *)floatValues, numFloats * sizeof(float));
        if (numInts)
            coVRMSController::instance()->sendSlaves((char *)intValues, numInts * sizeof(int));
    }
    else
    {
        int newNumFloats = 0;
        int newNumInts = 0;
        coVRMSController::instance()->readMaster((char *)&newNumFloats, sizeof(int));
        coVRMSController::instance()->readMaster((char *)&newNumInts, sizeof(int));
        //cerr << newNumFloats << endl;
        //cerr << newNumInts << endl;
        if (newNumFloats > 0 && newNumFloats != numFloats)
        {
            cerr << "resize" << endl;
            numFloats = newNumFloats;
            delete[] floatValues;
            floatValues = new float[numFloats];
        }
        if (newNumInts > 0 && newNumInts != numInts)
        {
            cerr << "resize" << endl;
            numInts = newNumInts;
            delete[] intValues;
            intValues = new int[numInts];
        }
        if (newNumFloats > 0 && numFloats)
        {
            //cerr << "rf" << endl;
            coVRMSController::instance()->readMaster((char *)floatValues, numFloats * sizeof(float));
        }
        if (newNumFloats > 0 && numInts)
        {
            //cerr << "ri" << endl;
            coVRMSController::instance()->readMaster((char *)intValues, numInts * sizeof(int));
        }
    }
	return true;
}

COVERPLUGIN(SimulinkPlugin)
