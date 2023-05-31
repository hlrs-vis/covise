/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "LabView.h"

#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <util/unixcompat.h>

LabViewPlugin *LabViewPlugin::plugin = NULL;

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeLabView(scene);
}

// Define the built in VrmlNodeType:: "LabView" fields

VrmlNodeType *VrmlNodeLabView::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("LabView", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addEventOut("ints_changed", VrmlField::MFINT32);
    t->addEventOut("floats_changed", VrmlField::MFFLOAT);

    return t;
}

VrmlNodeType *VrmlNodeLabView::nodeType() const
{
    return defineType(0);
}

VrmlNodeLabView::VrmlNodeLabView(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_enabled(true)
{
    setModified();
}

VrmlNodeLabView::VrmlNodeLabView(const VrmlNodeLabView &n)
    : VrmlNodeChild(n.d_scene)
    , d_enabled(n.d_enabled)
{
    setModified();
}

VrmlNodeLabView::~VrmlNodeLabView()
{
}

VrmlNode *VrmlNodeLabView::cloneMe() const
{
    return new VrmlNodeLabView(*this);
}

VrmlNodeLabView *VrmlNodeLabView::toLabView() const
{
    return (VrmlNodeLabView *)this;
}

ostream &VrmlNodeLabView::printFields(ostream &os, int indent)
{
    if (!d_enabled.get())
        PRINT_FIELD(enabled);

    return os;
}

// Set the value of one of the node fields.

void VrmlNodeLabView::setField(const char *fieldName,
                               const VrmlField &fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField *VrmlNodeLabView::getField(const char *fieldName)
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

void VrmlNodeLabView::eventIn(double timeStamp,
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

void VrmlNodeLabView::render(Viewer *)
{
    if (!d_enabled.get())
        return;

    double timeStamp = System::the->time();
    if (LabViewPlugin::plugin->numFloats)
    {
        d_floats.set(LabViewPlugin::plugin->numFloats, LabViewPlugin::plugin->floatValues);
        eventOut(timeStamp, "floats_changed", d_floats);
    }
    if (LabViewPlugin::plugin->numInts)
    {
        d_ints.set(LabViewPlugin::plugin->numInts, LabViewPlugin::plugin->intValues);
        eventOut(timeStamp, "ints_changed", d_ints);
    }

	fprintf(stderr, "%d", LabViewPlugin::plugin->numFloats);
	for (int i = 0; i < LabViewPlugin::plugin->numFloats; i++)
		fprintf(stderr, "%f;", LabViewPlugin::plugin->floatValues[i]);
	fprintf(stderr, "\n");
	fprintf(stderr, "%d", LabViewPlugin::plugin->numInts);
	for (int i = 0; i < LabViewPlugin::plugin->numInts; i++)
		fprintf(stderr, "%d;", LabViewPlugin::plugin->intValues[i]);
	fprintf(stderr, "\n");
}

LabViewPlugin::LabViewPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "LabViewPlugin::LabViewPlugin\n");

    plugin = this;
    conn = NULL;

    serverHost = NULL;
    localHost = new Host("localhost");

    port = coCoviseConfig::getInt("COVER.Plugin.LabViewPlugin.TCPPort", 12345);
    std::string line = coCoviseConfig::getEntry("COVER.Plugin.LabViewPlugin.Server");
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
        serverHost = new Host("visper.hlrs.de");
    }

    numFloats = 62;
    numInts = 61;

    floatValues = new float[numFloats];
    intValues = new int[numInts];
}

// this is called if the plugin is removed at runtime
LabViewPlugin::~LabViewPlugin()
{
    fprintf(stderr, "LabViewPlugin::~LabViewPlugin\n");

    delete[] floatValues;
    delete[] intValues;
    delete conn;
    delete serverHost;
}

bool LabViewPlugin::init()
{
    VrmlNamespace::addBuiltIn(VrmlNodeLabView::defineType());

    return true;
}

bool LabViewPlugin::readVal(void *buf, unsigned int numBytes)
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
LabViewPlugin::update()
{
    if (conn)
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
    }
    else if ((coVRMSController::instance()->isMaster()) && (serverHost != NULL))
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
                    fprintf(stderr, "Could not connect to LabView on %s; port %d\n", serverHost->getName(), port);
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
                        fprintf(stderr, "Could not connect to LabView on %s; port %d\n", localHost->getName(), port);
                    }
#endif
                    delete conn;
                    conn = NULL;
                }
                else
                {
                    fprintf(stderr, "Connected to LabView on %s; port %d\n", localHost->getName(), port);
                }
            }
            else
            {
                fprintf(stderr, "Connected to LabView on %s; port %d\n", serverHost->getName(), port);
            }
            if (conn && conn->is_connected())
            {
                int id = 2;
                conn->getSocket()->write(&id, sizeof(id));
            }
            oldTime = cover->frameTime();
        }
    }
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

COVERPLUGIN(LabViewPlugin)
