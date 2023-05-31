/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//
#ifdef WIN32
#include <WinSock2.h>
#endif
#include "Sensor.h"
#include <cover/coVRTui.h>
#include <cover/coVRCommunication.h>
#include <osg/LineSegment>
#include <osg/MatrixTransform>

#if !defined(_WIN32) && !defined(__APPLE__)
//#define USE_X11
#define USE_LINUX
#endif

// standard linux sockets
#include <sys/types.h>
#ifndef WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif
#include <pthread.h>
#include <stdio.h>
//#include <unistd. h.>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>

#ifdef USE_X11
#include <X11/Xlib.h>
#include <X11/extensions/XInput.h>
#include <X11/cursorfont.h>
#endif

#include <PluginUtil/PluginMessageTypes.h>

// socket (listen)
#define LOCAL_SERVER_PORT 9930

#define REMOTE_IP "141.58.8.26"
#define REMOTE_PORT 9931

#define BUF 512

int s, rc, n, len;
struct sockaddr_in cliAddr, servAddr;
char puffer[BUF];
time_t time1;
char loctime[BUF];
char *ptr;
const int y = 1;
pthread_t thdGetSensorSpeedIns;
static volatile int rpiGSR;
static volatile int rpiBPM;
static volatile int rpiSPO2;

void *thdGetSensorSpeed(void *ptr);

// socket (send)
#define BUFLEN 512

struct sockaddr_in si_other;
int ssend, i, slen = sizeof(si_other);
char buf[BUFLEN];

void setSensorBreak(int breakvalue);

SensorPlugin *SensorPlugin::plugin = NULL;

void *thdGetSensorSpeed(void *ptr)
{
    while (1)
    {
        memset(puffer, 0, BUF);
        len = sizeof(cliAddr);

        n = recvfrom(s, puffer, BUF, 0, (struct sockaddr *)&cliAddr, (socklen_t *)&len);

        if (n >= 0)
        {
            sscanf(puffer, "%d %d %d", &rpiBPM, &rpiSPO2, &rpiGSR);
            fprintf(stderr, "message received: %d bpm  %d   %d mV\n", rpiBPM, rpiSPO2, rpiGSR);
        }
    }

    // char *message;
    // message = (char *) ptr;
    // printf("%s \n", message);
    return NULL;
}

// --------------------------------------------------------------------

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeSensor(scene);
}

// --------------------------------------------------------------------
// Define the built in VrmlNodeType:: "Sensor" fields
// --------------------------------------------------------------------

VrmlNodeType *VrmlNodeSensor::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Sensor", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addEventIn("sensor1", VrmlField::SFTIME);
    t->addEventIn("sensor2", VrmlField::SFTIME);
    t->addEventIn("sensor3", VrmlField::SFTIME);
    t->addEventIn("sensor4", VrmlField::SFTIME);
    t->addEventIn("sensor5", VrmlField::SFTIME);
    t->addEventIn("sensor6", VrmlField::SFTIME);
    t->addEventIn("sensor7", VrmlField::SFTIME);
    t->addEventIn("sensor8", VrmlField::SFTIME);
    t->addEventIn("sensor9", VrmlField::SFTIME);
    t->addEventIn("sensor10", VrmlField::SFTIME);
    t->addEventIn("sensor11", VrmlField::SFTIME);
    t->addEventIn("sensor12", VrmlField::SFTIME);
    t->addEventIn("sensor13", VrmlField::SFTIME);
    t->addEventIn("sensor14", VrmlField::SFTIME);
    t->addEventIn("sensor15", VrmlField::SFTIME);
    t->addEventIn("sensor16", VrmlField::SFTIME);

    return t;
}

// --------------------------------------------------------------------

VrmlNodeType *VrmlNodeSensor::nodeType() const
{
    return defineType(0);
}

// --------------------------------------------------------------------

VrmlNodeSensor::VrmlNodeSensor(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    fp = NULL;
    if (coVRMSController::instance()->isMaster())
    {
        fp = fopen("logfile.txt", "a");
    }
    oldTime = 0;
    setModified();
}

// --------------------------------------------------------------------

VrmlNodeSensor::VrmlNodeSensor(const VrmlNodeSensor &n)
    : VrmlNodeChild(n.d_scene)
{
    //fp =fopen("logfile.txt","a");
    setModified();
}

// --------------------------------------------------------------------

VrmlNodeSensor::~VrmlNodeSensor()
{
    if (fp)
    {
        fprintf(fp, "The End!\n");
        fclose(fp);
    }
}

// --------------------------------------------------------------------

VrmlNode *VrmlNodeSensor::cloneMe() const
{
    return new VrmlNodeSensor(*this);
}

// --------------------------------------------------------------------

VrmlNodeSensor *VrmlNodeSensor::toSensor() const
{
    return (VrmlNodeSensor *)this;
}

// --------------------------------------------------------------------

ostream &VrmlNodeSensor::printFields(ostream &os, int indent)
{
    return os;
}

// --------------------------------------------------------------------
// Set the value of one of the node fields.
// --------------------------------------------------------------------

void VrmlNodeSensor::setField(const char *fieldName, const VrmlField &fieldValue)
{
    if
        TRY_FIELD(sensor1, SFTime)
    else if
        TRY_FIELD(sensor2, SFTime)
}

//

const VrmlField *VrmlNodeSensor::getField(const char *fieldName)
{
    /* if (strcmp(fieldName,"enabled")==0)
      return &d_enabled;
   else if (strcmp(fieldName,"joystickNumber")==0)
      return &d_joystickNumber;
   else if (strcmp(fieldName,"axes_changed")==0)
      return &d_axes;
   else if (strcmp(fieldName,"buttons_changed")==0)
      return &d_buttons;
   else
      cout << "Node does not have this eventOut or exposed field " << nodeType()->getName()<< "::" << name() << "." << fieldName << endl;*/
    return 0;
}

void VrmlNodeSensor::eventIn(double timeStamp,
                             const char *eventName,
                             const VrmlField *fieldValue)
{
    fprintf(stderr, "event %s\n", eventName);
    if (fp)
    {
        fprintf(fp, "event %s\n", eventName);
        fflush(fp);
    }
    if (strcmp(eventName, "sensor1") == 0)
    {
    }
    else if (strcmp(eventName, "sensor2") == 0)
    {
    }
    // Check exposedFields
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

// --------------------------------------------------------------------

void VrmlNodeSensor::render(Viewer *)
{

    time_t currentTime = time(NULL);
    if (currentTime > oldTime && fp)
    {
        osg::Vec3 pos;
        osg::Matrix m = cover->getViewerMat();
        m = m * cover->getInvBaseMat();
        pos = m.getTrans();
        fprintf(fp, "pos %f %f %f time %ld values %d %d %d\n", pos[0], pos[1], pos[2], long(currentTime), rpiBPM, rpiSPO2, rpiGSR);
        fflush(fp);
        oldTime = currentTime;
    }
    setModified();
}

//-----------------------------------------------------------------------------
// Name: UpdateInputState()
// Desc: Get the input device's state and display it.
//-----------------------------------------------------------------------------
void SensorPlugin::UpdateInputState()
{

    return;
}

void SensorPlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

void SensorPlugin::tabletEvent(coTUIElement * /*tUIItem*/)
{
    {
    }
}

SensorPlugin::SensorPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    startThread();
    running = true;
}

bool SensorPlugin::init()
{
    fprintf(stderr, "SensorPlugin::SensorPlugin\n");
    if (plugin)
        return false;

    plugin = this;
    SensorTab = new coTUITab("Sensor", coVRTui::instance()->mainFolder->getID());
    SensorTab->setPos(0, 0);
    if (coVRMSController::instance()->isMaster())
    {

        // socket init (tx)

        if ((ssend = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
        {
            printf("bicyce: socket error\n");
        }

        memset((char *)&si_other, 0, sizeof(si_other));
        si_other.sin_family = AF_INET;
        si_other.sin_port = htons(REMOTE_PORT);
#ifdef WIN32
        if (InetPton(AF_INET,REMOTE_IP, &si_other.sin_addr)==0) 
#else
        if (inet_aton(REMOTE_IP, &si_other.sin_addr) == 0)
#endif
        {
            printf("bicyce: inet_aton failed\n");
        }

        // socket init (rx)

        s = socket(AF_INET, SOCK_DGRAM, 0);
        if (s < 0)
        {
            printf("Sensor: Kann Socket nicht öffnen ...(%s)\n",
                   strerror(errno));
            exit(EXIT_FAILURE);
        }

        servAddr.sin_family = AF_INET;
        servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
        servAddr.sin_port = htons(LOCAL_SERVER_PORT);
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *)&y, sizeof(int));
        rc = bind(s, (struct sockaddr *)&servAddr,
                  sizeof(servAddr));
        if (rc < 0)
        {
            printf("Sensor: Kann Portnummern %d nicht binden (%s)\n",
                   LOCAL_SERVER_PORT, strerror(errno));
            exit(EXIT_FAILURE);
        }

        const char *message = "getSpeed";

        int iret1;
        iret1 = pthread_create(&thdGetSensorSpeedIns, NULL, thdGetSensorSpeed, (void *)message);
        if (iret1)
        {
            fprintf(stderr, "Sensor: pthread_create() return code: %d\n", iret1);
            exit(EXIT_FAILURE);
        }
    }

    initUI();

    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
SensorPlugin::~SensorPlugin()
{
    fprintf(stderr, "SensorPlugin::~SensorPlugin\n");
    running = false;
}

int SensorPlugin::initUI()
{
    VrmlNamespace::addBuiltIn(VrmlNodeSensor::defineType());
    return 1;
}

void
SensorPlugin::stop()
{
    doStop = true;
}
void
SensorPlugin::run()
{
    running = true;
    doStop = false;
    while (running)
    {
        usleep(20000);
    }
}
void
SensorPlugin::preFrame()
{
    if (coVRMSController::instance()->isMaster())
    {
    }
    UpdateInputState();
}

void SensorPlugin::key(int /*type*/, int /*keySym*/, int /*mod*/)
{
}

COVERPLUGIN(SensorPlugin)
