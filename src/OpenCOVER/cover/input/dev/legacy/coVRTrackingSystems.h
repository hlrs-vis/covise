/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			coVRTrackingSystems.h (Performer 2.0)	*
 *									*
 *	Description		polhemus tracker class			*
 *				supports stylus and sensor		*
 *									*
 *	Author			D. Rainer				*
 *									*
 *	Date			20.08.97				*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#ifndef __VR_POLHEMUS_TRACKER_H
#define __VR_POLHEMUS_TRACKER_H

#include <util/common.h>

extern "C" {
#include "flock.h"
}
#ifdef WIN32
#include "coRawMouse.h"
#endif

class CGVTrack;
class Tarsus;
class PvrSSD;
class VRPN;
class DynaSight;

#include <cover/input/VRKeys.h>
#include "fobalt.h"
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include "polhemusdrvr.h"
#include "birdTracker.h"
#include "DTrack.h"
#include "coVRTrackingUtil.h"
class VRCTracker;

#include "bgLib.h"
namespace opencover
{
class ButtonDevice;
class coVRPlugin;
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

typedef struct
{
    float origin[3];
    float alpha;
    float beta0, beta1;
    float gamma;
    float delta;
    float up_scale;
    int useFlag;
    int swapFlag;

    int filterType;

} emFilterStruct;

class INPUT_LEGACY_EXPORT coVRTrackingSystems
{
private:
    void *motion;
    fob *fo;
    DTrack *dtrack;
    CGVTrack *cgvtrack;
    Tarsus *tarsus;
    PvrSSD *ssd;
    DynaSight *dynasight;
    VRPN *vrpn;
    ButtonDevice *mousebuttons;
#ifdef WIN32
    coRawMouse *rawTarsusMouse;
#endif
    struct
    {
        float position[3],
            angles[3],
            matrix[4][4];
        int button;
    } mouse;
    int buttonMask;
    birdTracker *tracker;
    class fastrak *fs;
    int sensorStation, stylusStation, worldXFormStation;
    int trackingSystem;
    int buttonSystem;
    coVRPlugin *buttonSystemPlugin;
    char portname[20]; // serial port for trackers
    int numStations;
    int baudrate;
    char devicename[50];
    TRACKD_TRACKING *CaveLibTracker;
    TRACKD_WAND *CaveLibWand;
    int CaveLibWandController;
    int handDevice, headDevice;
    float scaleFactor;
    bool Yup;
    VRCTracker *vrcTracker;

    // hemisphere direction
    float hx, hy, hz;

    // correction factors for the magnetic field
    float correct_x, correct_y, correct_z;

    // interpolation
    char interp_message[500];
    const char *interpolationFile;
    //int write_counter;		//for the out put of the trackr position (mat)
    bool interpolationFlag, orientInterpolationFlag;
    bool orien_interp_files_flag;
    const char *ori_file_name_x, *ori_file_name_y, *ori_file_name_z;
    float *x_coord, *y_coord, *z_coord; // deformierte Koordinaten
    float *n1, *n2, *n3; // exakte Koordinaten
    int nx, ny, nz, ne; // Anzahl Punkte in x, y, z, Numerierung
    float ***trans_basis; // for the orientation
    float val8[2][2][2]; // 8 values for the 8 corners of a cell
    float val4[2][2]; // 4 interpolated values,1 for each pair of corners
    float val2[2]; // 2 interpolated values,1 for each pair of the last 4 values
    float val1; // 1 finterolated
    //write calibration file
    bool write_calibration_flag;
    int w_ni, w_nj, w_nk;
    float calib_pos_i[100];
    float calib_pos_j[100];
    float calib_pos_k[100];

    char calib_name_x[200], calib_name_y[200], calib_name_z[200], calib_name_p[200];
    char end_file_name[8];
    fstream calib_file_x, calib_file_y, calib_file_z, calib_file_p;

    bool readInterpolationFile(const char *);
    float dis_pn_pt(const float[3], const float[3], const float[3], const float[3]);
    //void interpolate (const float [3] , float[3]);
    void interpolate(const float[4][4], float[4][4]);
    void find_closest_g_point(const float[3], int[3]);
    int find_xyz_velocity(void);
    void reorganize_data(int);
    void create_trans_basis(void);
    void read_trans_basis(void);
    void transform_orientation(const float[3], const float[3], const float[3],
                               const float[3], float[3]);
    void linear_equations_sys(const float[3], const float[3], const float[3],
                              const float[3], float[3]);
    float determinante(const float c0[3], const float c1[3], const float c2[3]);

    //
    int readConfigFile();

    // stuff for approximating/improving the magnetic field
    void filterEMField(osg::Matrix &mat);
    void filterEMPoint(float &x, float &y, float &z, float bx, float by, float bz);
    void filterEMPoint(float &x, float &y, float &z, int filterType);
    emFilterStruct emFilterInfo;

    int XValuator, YValuator;
    int rawButton_micha;

    int dtrackWheel;

public:
    coVRTrackingSystems(int numStations, int stylusStation, int sensorStation, int worldXFormStation = -1);

    ~coVRTrackingSystems();

#ifdef WIN32
    HANDLE pinchfd;

    HWND AWindow;
#else
    int pinchfd;
#endif
    char *buttonDevice; // serial port for button devices
    bglv bgdata;
    void config();
    void config_save();

    void reset();
    void getRotationMatrix(osg::Matrix &rotMat);

    void getTranslationMatrix(osg::Matrix &transMat);

    void getMatrix(int station, osg::Matrix &mat);
    void getMatrix_save(int station, osg::Matrix &mat);

    void getButton(int station, unsigned int *buttons);
    void getWheel(int station, int *wheel);
    void getAnalog(int station, float &x, float &y);
    void getCyberWheel(int station, int &count);
    void getCerealAnalog(int station, float **value);
    void getRawButtonMicha(int station, int *rawButtonMicha);

    /** 
       * Get DTrack finger data, return 0 if no data exists
       * @param hand use handDev for first hand, hand2Dev for second hand
       */

    const DTrack::FingerData *getFingerData(coVRTrackingUtil::IDOfDevice hand);

    // marker data for optical trackers (currently VICON only)
    int getNumMarkers();
    bool getMarker(int index, float *pos); // true if marker is visible
};
}
#endif
