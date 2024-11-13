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

void VrmlNodeSensor::initFields(VrmlNodeSensor *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    for (size_t i = 0; i < numSensors; i++)
    {
        initFieldsHelper(node, t,
                        eventInCallBack("sensor" + std::to_string(i), node->d_sensor[i]));
    }
}

const char *VrmlNodeSensor::name()
{
    return "Sensor";
}

// --------------------------------------------------------------------

VrmlNodeSensor::VrmlNodeSensor(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
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
    : VrmlNodeChild(n)
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

VrmlNodeSensor *VrmlNodeSensor::toSensor() const
{
    return (VrmlNodeSensor *)this;
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
    VrmlNamespace::addBuiltIn(VrmlNode::defineType<VrmlNodeSensor>());
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
