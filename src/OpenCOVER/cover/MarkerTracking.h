/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
\brief MarkerTracking optical tracking system interface classes

\author Uwe Woessner <woessner@hlrs.de>
\author (C) 2002
Computer Centre University of Stuttgart,
Allmandring 30,
D-70550 Stuttgart,
Germany

\date    July 2002
*/

#ifndef MarkerTracking_H
#define MarkerTracking_H

#include <list>
#include <string>
#include <array>

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Drawable>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Vec3>

#include <OpenConfig/array.h>
#include <OpenConfig/value.h>
#include <OpenConfig/file.h>

#include <cover/coTabletUI.h>
namespace opencover
{
class MarkerTrackingMarker;
class coTUILabel;
class coTUIToggleButton;
class coTUIEditFloatField;
class coTUIEditFloatField;
class coTUITab;
class coTUIElement;
class coTUIGroupBox;
}

#include <net/covise_connect.h>
#include <net/message.h>

#include "coTUIListener.h"
#include "coTabletCovConfig.h"

namespace opencover
{
class COVEREXPORT MarkerTrackingNode : public osg::Drawable
{
private:
    bool displayVideo; // true if CoviseConfig.displayVideo is set
    bool renderTextures;
    std::string m_MarkerTrackingVariant;

public:
    MarkerTrackingNode(std::string MarkerTrackingVariant);
    virtual ~MarkerTrackingNode();
    static MarkerTrackingNode *theNode;
    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
    /** Clone the type of an object, with Object* return type.
		Must be defined by derived classes.*/
    virtual osg::Object *cloneType() const;

    /** Clone the an object, with Object* return type.
		Must be defined by derived classes.*/
    virtual osg::Object *clone(const osg::CopyOp &) const;
};

class COVEREXPORT MarkerTrackingInterface
{
public:
    MarkerTrackingInterface(){};
    virtual ~MarkerTrackingInterface(){};
    virtual std::string loadPattern(const std::string& /*patternId*/)
    {
        return "";
    };
    virtual bool isVisible(const MarkerTrackingMarker *marker) = 0;

    virtual osg::Matrix getMat(const MarkerTrackingMarker *marker) = 0;

    virtual void updateViewerPos(const osg::Vec3f &vp)
    {
        (void)vp;
    }
    virtual void updateMarkerParams(){};
    virtual bool isMarkerTracking()
    {
        return false;
    };
};

class COVEREXPORT RemoteARInterface
{
public:
    RemoteARInterface()
    {
    }
    virtual ~RemoteARInterface()
    {
    }
    virtual void update() = 0;
    virtual void receiveImage(const char *data) = 0;
    virtual void updateBitrate(const int bitrate) = 0;
    virtual bool usesIRMOS() const
    {
        return false;
    }
    virtual bool isReceiver() const
    {
        return false;
    }
    virtual covise::ClientConnection *getIRMOSClient() const
    {
        return NULL;
    }
};

class COVEREXPORT MarkerTracking
{
public:
    bool running = false;
    ~MarkerTracking();
    static MarkerTracking *instance();
    MarkerTrackingMarker *getMarker(const std::string &name);
    MarkerTrackingMarker *getOrCreateMarker(const std::string &name, const std::string &pattern, double size, const osg::Matrix &offset, bool vrml, bool isObjectMarker = false);
    
    coTUITab *artTab;
    bool flipH;
    MarkerTrackingInterface *arInterface = nullptr;
    RemoteARInterface *remoteAR = nullptr;
    void config();
    void update();
    bool isRunning();
    opencover::config::File &markerDatabase();
    unsigned char *videoData = nullptr;
    unsigned char *videoDataRight = nullptr;
    int videoWidth = 0;
    int videoHeight = 0;
    int videoMode = GL_RGB;
    int videoDepth;
    bool stereoVideo = false;
    bool videoMirrorLeft = false;
    bool videoMirrorRight = false;
    std::string m_MarkerTrackingVariant = "MarkerTracking";
    std::map<std::string, std::unique_ptr<MarkerTrackingMarker>> markers;
    bool doMerge;
    bool testImage = false;
private:
    static MarkerTracking *art;
    MarkerTracking();
    void changeObjectMarker(MarkerTrackingMarker *m, bool state);
    std::string m_configPath;
    covise::Message msg;

    bool objTracking = false;
    int numObjectMarkers;
    std::list<MarkerTrackingMarker *> objectMarkers;
    std::unique_ptr<opencover::config::File> m_markerDatabase;
    std::unique_ptr<opencover::config::File> m_trackingConfig;
};

class COVEREXPORT MarkerTrackingMarker : public coTUIListener
{
friend class MarkerTracking;

private:

    int pattGroup = -1;
    float oldpattGroup = -1;
    double pattCenter[2] = {0.0, 0.0};
    double pattTrans[3][4];
    bool objectMarker = false;
    osg::Matrix offset;
    osg::Matrix cameraTransform;
    std::unique_ptr<covTUIToggleButton> vrmlToPf;
    std::unique_ptr<covTUIEditField> pattID;
    std::unique_ptr<covTUIEditFloatField> size;
    // std::array<std::shared_ptr<Coord>, 6> m_transform;
    std::unique_ptr<covTUIEditFloatFieldVec3> m_xyz; 
    std::unique_ptr<covTUIEditFloatFieldVec3> m_hpr; 
    void createUiandConfigValues(const std::string &configName);
    void matToEuler(const osg::Matrix &mat);
    osg::Matrix eulerToMat() const;
    void updateMatrices();
    void init();
	void updateData(double markerSize, const osg::Matrix& mat, bool vrmlToOsg);


	MarkerTrackingMarker(const std::string &configName, const std::string &pattern, double size, const osg::Matrix &mat, bool vrml);
	MarkerTrackingMarker(const std::string &Name);
public:
    virtual ~MarkerTrackingMarker();
    MarkerTrackingMarker(const MarkerTrackingMarker&) = delete;
    MarkerTrackingMarker& operator=(const MarkerTrackingMarker&) = delete;
    MarkerTrackingMarker(MarkerTrackingMarker&&) = default;
    MarkerTrackingMarker& operator=(MarkerTrackingMarker&&) = default;

    const osg::Matrix &getCameraTrans();
    osg::Matrix getMarkerTrans();
    const osg::Matrix &getOffset() const
    {
        return offset;
    };
    int getMarkerGroup() const;
    virtual void tabletEvent(coTUIElement *tUIItem);
    double getSize() const;
    std::string getPattern() const;
    bool isVisible() const;
    bool isObjectMarker() const
    {
        return objectMarker;
    };
    void setObjectMarker(bool o)
    {
        objectMarker = o;
    };
    void setColor(float r, float g, float b);
    osg::Geode *quadGeode = nullptr;
    osg::ref_ptr<osg::MatrixTransform> markerQuad;
    osg::MatrixTransform *posSize=nullptr;
    osg::Vec4Array *colors = nullptr;
    osg::Geometry *geom = nullptr;
    coTUIToggleButton *displayQuad = nullptr;
    coTUIToggleButton *calibrate = nullptr;
    int numCalibSamples = 0;
    osg::Matrix matrixSumm;
    bool lastVisible = false;
    void setOffset(const osg::Matrix &mat);
    void stopCalibration();
};
}
#endif
