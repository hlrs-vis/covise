/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// class VRTracker
// intitialize and read input devices
// authors: F. Foehl, D. Rainer, A. werner, U. Woessner
// (C) 1996-2003 University of Stuttgart
// (C) 1997-2003 Vircinity GmbH

#ifndef VR_TRACKER_H
#define VR_TRACKER_H

#ifdef WIN32
#include "coRawMouse.h"
#endif
#include <util/common.h>
#include "coVRTrackingUtil.h"

#include <osg/Matrix>
#include <osg/Vec3>

#include "cover/input/VRKeys.h"

// Button Definitions
#include <OpenVRUI/sginterface/vruiButtons.h>

#define MAX_BUTTONS 32

// Button systems
#define B_NONE 0
#define B_MIKE 1
#define B_DIVISION 2
#define B_CAVELIB 3
#define B_CEREAL 4
#define B_PINCH 5
#define B_VP 6
#define B_CYBER 7
#define B_HORNET 8
#define B_MOUSE 9
#define B_WIIMOTE 10
#define B_PLUGIN 11
#define B_VRC 12
#define B_PRESENTER 13

class VRSpacePointer;
class VRCTracker;

namespace opencover
{
class coVRTrackingSystems;
class coVRTrackingUtil;

class angleStruct;

class VRVruiRenderInterface;

class INPUT_LEGACY_EXPORT VRTracker
{
private:
// flags

// tracker stuff
#ifdef WIN32
    coRawMouseManager *rawMouseManager;
#endif
    class coVRTrackingSystems *trackingSystems;
    VRCTracker *vrcTracker;
    int baud;
    char serialPortname[500];
    const char *interpolationFile;
    int buttonStation;
    int analogStation;
    bool enableTracking_;
    //Array for the station matrices
    osg::Matrix **trackerStationMat;
    osg::Matrix mouseMat;
    // this Matrix stores the Identity. for error return in getStationMat()
    osg::Matrix myIdentity;
    //int buttonMap[MAX_BUTTONS];
    std::map<int, int> buttonMap;
    typedef std::pair<int, int> Data_Pair;

    osg::Vec3 screenToTransmitterOffset;

    VRSpacePointer *spacePointer;
    bool doJoystick;
    bool doJoystickVisenso;
    bool visensoJoystickAnalog;
    float visensoJoystickSpeed;
    bool doCalibrate;
    bool doCalibrateOrientation;

    float *x_coord, *y_coord, *z_coord; // deformierte Koordinaten
    float *n1, *n2, *n3; // exakte Koordinaten
    int nx, ny, nz, ne; // Anzahl Punkte in x, y, z, Numerierung
    float ***trans_basis; // for the orientation
    float val8[2][2][2]; // 8 values for the 8 corners of a cell
    float val4[2][2]; // 4 interpolated values,1 for each pair of corners
    float val2[2]; // 2 interpolated values,1 for each pair of the last 4 values
    float val1; // 1 finterolated

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

    void calibrate(osg::Matrix &mat);
    int readConfigFile();

    void updateHead();

    void updateHand();

    void updateWheel();

    void updateDevice(int device);

    bool d_debugTracking, d_debugButtons; //enables debug outputs for APP tracking and mapped debug buttons
    int d_debugStation;

    //returns the maximum of a and b
    int maxInt(int a, int b)
    {
        return (((a) > (b)) ? (a) : (b));
    };
    //tracking information now can be recorded or replayed
    //File for saving Trackinginfo
    FILE *d_saveTracking;
    //File to replay Trackinginfo from
    FILE *d_loadTracking;

    unsigned int m_buttonState;
    void setButtonState(unsigned int s);

public:
    static VRTracker *instance();

    VRTracker();

    ~VRTracker();

    void update();

    void reset();

    void enableTracking(bool on);
    bool isTrackingOn()
    {
        return enableTracking_;
    }

    void initCereal(angleStruct *screen_angle); // CEREAL under Polhemus
    osg::Matrix &getHandMat();
    osg::Matrix &getViewerMat();
    osg::Matrix &getCameraMat();
    osg::Matrix &getWorldMat();
    osg::Matrix &getStationMat(int station);
    void setHandMat(const osg::Matrix &);
    void setViewerMat(const osg::Matrix &);
    void setStationMat(const osg::Matrix &, int station);
    void setWorldMat(const osg::Matrix &);
    void setMouseMat(const osg::Matrix &);
    osg::Matrix &getMouseMat();
    osg::Matrix getDeviceMat(coVRTrackingUtil::IDOfDevice device);

    int getNumStation();

    void updateHandPublic();
    int getHandSensorStation();
    int getHeadSensorStation();
    int getCameraSensorStation();
    int getWorldSensorStation();
    coVRTrackingSystems *getTrackingSystemsImpl();
    coVRTrackingUtil *trackingUtil;

    bool hasHand() const;
    bool hasHead() const;

    unsigned int getButtonState() const;
};
}
#endif
