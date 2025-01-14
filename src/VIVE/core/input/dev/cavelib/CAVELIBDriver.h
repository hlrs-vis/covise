/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * CAVELIBDriver.h
 *
 *  Created on: Feb 5, 2014
 *      Author: hpcwoess
 */

#ifndef CAVELIB_DRIVER_H
#define CAVELIB_DRIVER_H

#include <OpenThreads/Thread>
#include <osg/Matrix>
#include <string>

#include <cover/input/inputdevice.h>
#include <util/coExport.h>


/* Miscellaneous constants for utility functions & CAVEConfig entries */
/* New names should be added to the end of the list, to maintain
   backward-compatibility of values */

#define TRACKD_MAX_SENSORS 8
#define CAVE_MAX_BUTTONS 32
#define CAVE_MAX_VALUATORS 32

typedef enum
{
    CAVE_NULL = 0,
    CAVE_HEAD,
    CAVE_WAND,
    CAVE_LEFT_EYE,
    CAVE_RIGHT_EYE,
    CAVE_FRONT,
    CAVE_BACK,
    CAVE_LEFT,
    CAVE_RIGHT,
    CAVE_UP,
    CAVE_DOWN,
    CAVE_FRONT_NAV,
    CAVE_BACK_NAV,
    CAVE_LEFT_NAV,
    CAVE_RIGHT_NAV,
    CAVE_UP_NAV,
    CAVE_DOWN_NAV,
    CAVE_HEAD_FRONT,
    CAVE_HEAD_BACK,
    CAVE_HEAD_LEFT,
    CAVE_HEAD_RIGHT,
    CAVE_HEAD_UP,
    CAVE_HEAD_DOWN,
    CAVE_WAND_FRONT,
    CAVE_WAND_BACK,
    CAVE_WAND_LEFT,
    CAVE_WAND_RIGHT,
    CAVE_WAND_UP,
    CAVE_WAND_DOWN,
    CAVE_HEAD_NAV,
    CAVE_WAND_NAV,
    CAVE_LEFT_EYE_NAV,
    CAVE_RIGHT_EYE_NAV,
    CAVE_HEAD_FRONT_NAV,
    CAVE_HEAD_BACK_NAV,
    CAVE_HEAD_LEFT_NAV,
    CAVE_HEAD_RIGHT_NAV,
    CAVE_HEAD_UP_NAV,
    CAVE_HEAD_DOWN_NAV,
    CAVE_WAND_FRONT_NAV,
    CAVE_WAND_BACK_NAV,
    CAVE_WAND_LEFT_NAV,
    CAVE_WAND_RIGHT_NAV,
    CAVE_WAND_UP_NAV,
    CAVE_WAND_DOWN_NAV,
    /* CAVESetOption() arguments */
    CAVE_PROJ_USEWINDOW,
    CAVE_TRACKER_SIGNALRESET,
    CAVE_DIST_NETWORKSLAVE,
    CAVE_GL_SAMPLES,
    CAVE_GL_STENCILSIZE,
    CAVE_GL_ACCUMSIZE,
    CAVE_SCRAMNET_ARENASIZE,
    CAVE_SHMEM_SIZE,
    CAVE_SHMEM_ADDRESS,
    CAVE_NET_BUFFERSIZE,
    CAVE_NET_NUMBUFFERS,
    CAVE_NET_UPDATELOCALDATA,
    /* Values for CAVEConfig->DisplayMode  */
    CAVE_MONO,
    CAVE_STEREO,
    /* Values for CAVEConfig->ControllerType */
    CAVE_MOUSE_CONTROLLER,
    CAVE_PC_CONTROLLER,
    CAVE_SIMULATOR_CONTROLLER,
    CAVE_LOGITECH_CONTROLLER,
    CAVE_CUSTOM_CONTROLLER,
    CAVE_DAEMON_CONTROLLER,
    CAVE_SCRAMNET_CONTROLLER,
    /* Values for CAVEConfig->TrackerType */
    CAVE_POLHEMUS_TRACKER_OBSOLETE,
    CAVE_BIRDS_TRACKER,
    CAVE_LOGITECH_TRACKER,
    CAVE_MOUSE_TRACKER,
    CAVE_SIMULATOR_TRACKER,
    CAVE_SPACEBALL_TRACKER,
    CAVE_BOOM_TRACKER,
    CAVE_DAEMON_TRACKER,
    CAVE_SCRAMNET_TRACKER,
    CAVE_SPACEPAD_TRACKER,
    /* Values for CAVEConfig->BirdsHemisphere */
    CAVE_FRONT_HEMI,
    CAVE_LEFT_HEMI,
    CAVE_RIGHT_HEMI,
    CAVE_AFT_HEMI,
    CAVE_UPPER_HEMI,
    CAVE_LOWER_HEMI,
    /* Flock-of-birds sync type, for CAVEConfig->SyncBirds */
    CAVE_BIRDS_SYNC1,
    CAVE_BIRDS_SYNC2,
    /* Frames of reference, for CAVE_SENSOR_ST.frame */
    CAVE_TRACKER_FRAME,
    CAVE_NAV_FRAME,
    /* Values for CAVEConfig->Units */
    CAVE_FEET,
    CAVE_METERS,
    CAVE_INCHES,
    CAVE_CENTIMETERS,
    /* Additional eye value for CAVEConfig->WallEyes */
    CAVE_BOTH_EYES,
    /* Callback types, for CAVEAddCallback */
    CAVE_DISPLAY_CALLBACK,
    CAVE_INITGRAPHICS_CALLBACK,
    CAVE_PERFRAME_CALLBACK,
    CAVE_NETADDUSER_CALLBACK,
    CAVE_NETDELETEUSER_CALLBACK,
    CAVE_NETAPPDATA_CALLBACK,
    /* Distributed CAVE method, for CAVEConfig->Distribution */
    CAVE_DIST_SCRAMNET,
    CAVE_DIST_TCP,
    /* Networking types, for CAVEConfig->Network */
    CAVE_NET_MCAST,
    CAVE_NET_TCP,
    CAVE_NET_UDP,
    /* Process types; returned by CAVEProcessType() */
    CAVE_APP_PROCESS,
    CAVE_DISPLAY_PROCESS,
    CAVE_TRACKER_PROCESS,
    CAVE_NETWORK_PROCESS,
    CAVE_DISTRIB_PROCESS,
    /* Projection types (for screen# walls) */
    CAVE_WALL_PROJECTION,
    CAVE_HMD_PROJECTION,
    /* Further 2.6beta additions */
    CAVE_PROJ_USEMODELVIEW,
    CAVE_SIM_DRAWWAND,
    CAVE_SIM_DRAWOUTLINE,
    CAVE_SIM_DRAWUSER,
    CAVE_SIM_DRAWTIMING,
    CAVE_SIM_VIEWMODE,
    CAVE_PROJ_INCLUDENAVIGATION
} CAVEID;

typedef struct
{
    uint32_t version; /* CAVElib version (see constants below) */
    uint32_t numSensors; /* Total number of sensors */
    uint32_t sensorOffset; /* Byte offset from of header to start of sensor array */
    uint32_t sensorSize; /* sizeof() of a sensor struct */
    uint32_t timestamp[2]; /* NB: *Not* a struct timeval - that changes */
    /*      size between 32 & 64 bit mode */
    uint32_t command; /* For sending commands (such as 'reset') to daemon */
} CAVE_TRACKDTRACKER_HEADER;

/* Sensor timestamp (can't use struct timeval because its size is different in 64-bit mode!) */
typedef struct
{
    uint32_t sec;
    uint32_t usec;
} CAVE_TIMESTAMP_ST;

/* Position & orientation data for trackers */
typedef struct
{
    float x, y, z;
    float azim, elev, roll;
    CAVE_TIMESTAMP_ST timestamp;
    int calibrated;
    CAVEID frame; /* CAVE_TRACKER_FRAME or CAVE_NAV_FRAME */
} CAVE_SENSOR_ST;

struct TRACKD_TRACKING
{
    CAVE_TRACKDTRACKER_HEADER header;
    CAVE_SENSOR_ST sensor[TRACKD_MAX_SENSORS];
};

/* Collection of buttons & valuators associated with wand */
typedef struct
{
    uint32_t num_buttons, num_valuators;
    int32_t button[CAVE_MAX_BUTTONS];
    float valuator[CAVE_MAX_VALUATORS];
} CAVE_CONTROLLER_ST;

struct TRACKD_WAND
{
    CAVE_TRACKDTRACKER_HEADER header;
    int num;
    CAVE_CONTROLLER_ST controller[1];
};

/**
 * @brief The CAVELIBDriver class interacts with input hardware
 *
 * This class interacts with input hardware and stores the data
 * about all configured input hardware e.g. tracking systems,
 * button devices etc.
 *
 * Main interaction loop runs in its own thread
 */
class CAVELIBDriver : public opencover::InputDevice
{
   

    bool init();
    virtual bool needsThread() const;
    
    int key;
#ifndef WIN32
    int tracker_shmid;
#endif
    double scaleFactor;
    bool Yup;
    
    TRACKD_TRACKING *CaveLibTracker;
    TRACKD_WAND *CaveLibWand;
    int CaveLibWandController;

public:
    CAVELIBDriver(const std::string &name);
    virtual ~CAVELIBDriver();
    
    virtual void update();
};
#endif
