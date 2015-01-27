/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ****************************************** /
// class coVRTrackingUtil
// intitialize and read input device
// Date: 2007-10-16
// ****************************************** /

#ifndef CO_VR_TRACKINGUTIL_H
#define CO_VR_TRACKINGUTIL_H

#include <util/common.h>

#include <OpenVRUI/osg/mathUtils.h>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Vec4>

#include <cover/coTabletUI.h>

namespace opencover
{
class coVRPlugin;

struct offsets
{
    float trans[3];
    float rot[3];
};

class INPUT_LEGACY_EXPORT coVRTrackingUtil : public coTUIListener
{
public:
    enum IDOfDevice
    {
        trackingSys = 0,
        headDev,
        handDev,
        secondHandDev,
        worldDev,
        hmdDev,
        cameraDev,
        objectDev,
        numDevices
    };
    enum TrackingSystemType
    {
        T_NONE = 0,
        T_POLHEMUS,
        T_MOTIONSTAR,
        T_FOB,
        T_CAVELIB,
        T_SPACEPOINTER,
        T_COVER_BEEBOX,
        T_COVER_FLYBOX,
        T_PHANTOM,
        T_SPACEBALL,
        T_DTRACK,
        T_MOUSE,
        T_VRC_DAEMON,
        T_TARSUS,
        T_SEEREAL,
        T_SSD,
        T_VRPN,
        T_DYNASIGHT,
        T_CGVTRACK,
        T_PLUGIN,
        numTrackingSystems
    };
    coVRTrackingUtil();
    ~coVRTrackingUtil();

    int readOffset();
    osg::Matrix computeDeviceOffsetMat(IDOfDevice device_ID);
    virtual void tabletEvent(coTUIElement *tUIItem);
    void createTUI();
    int getDeviceAddress(IDOfDevice device_ID) const
    {
        return deviceAddress[device_ID];
    }
    bool haveDevice(IDOfDevice device_ID) const
    {
        return deviceAddress[device_ID] != -1;
    }
    int getNumStations()
    {
        return numStations;
    };
    int getTrackingSystem()
    {
        return trackingSystem;
    };
    coVRPlugin *getTrackingSystemPlugin() const
    {
        return trackingSystemPlugin;
    }
    bool hasHead() const;
    bool hasHand() const;
    bool hasSecondHand() const;

    static coVRTrackingUtil *instance()
    {
        if (myInstance == NULL)
            myInstance = new coVRTrackingUtil();
        return myInstance;
    };
    float *getDeviceOffsetTrans(IDOfDevice device_ID);
    float *getDeviceOffsetRot(IDOfDevice device_ID);
    void setDeviceOffset(IDOfDevice device_ID, osg::Vec3 &trans, osg::Vec3 &rot);

private:
    static const char *deviceNames[numDevices];
    static const char *cmDeviceNames[numDevices];
    static int deviceAddress[numDevices];
    int currentDevice;
    int numStations;
    void updateTUI();
    struct offsets deviceOffsets[numDevices];
    coTUITab *trackingTab;
    coTUIComboBox *deviceChoice;
    coTUILabel *deviceLabel;
    coTUIEditFloatField *trans[3];
    coTUILabel *transLabel[3];
    coTUIEditFloatField *rot[3];
    coTUILabel *rotLabel[3];
    TrackingSystemType trackingSystem;
    coVRPlugin *trackingSystemPlugin;
    static coVRTrackingUtil *myInstance;
};
}
#endif
