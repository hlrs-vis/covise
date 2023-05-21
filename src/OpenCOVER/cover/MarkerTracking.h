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
    virtual void createUnconfiguredTrackedMarkers() = 0;
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

class COVEREXPORT MarkerTracking : public coTUIListener
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
    void changeObjectMarker(MarkerTrackingMarker *m);
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
    bool testImage = false;
    coTUIFrame* m_trackingFrame = nullptr;
private:
    static MarkerTracking *art;
    MarkerTracking();
    void tabletPressEvent(coTUIElement* tUIItem) override;

    std::string m_configPath;
    covise::Message msg;

    bool objTracking = false;
    int numObjectMarkers;
    std::vector<MarkerTrackingMarker *> objectMarkers;
    std::unique_ptr<opencover::config::File> m_markerDatabase;
    coTUIButton* m_configureMarkerBtn = nullptr;
    coTUIButton* m_saveBtn = nullptr;
    coTUIFrame* m_buttonsFrame = nullptr;
};

constexpr int noMarkerGroup = -1;

class COVEREXPORT MarkerTrackingMarker : public coTUIListener
{
friend class MarkerTracking;

private:

    float m_oldpattGroup = -1;
    double m_pattCenter[2] = {0.0, 0.0};
    double m_pattTrans[3][4];
    osg::Matrix m_offset;
    osg::Matrix m_cameraTransform;

    coTUIButton *m_toggleConfigOff = nullptr;
    coTUIButton *m_toggleConfigOn = nullptr;
    coTUIFrame *m_layoutGroup = nullptr;
    std::unique_ptr<coTUILabel> m_configLabel;
    std::unique_ptr<covTUIToggleButton> m_vrmlToPf;
    std::unique_ptr<covTUIToggleButton> m_objectMarker;
    std::unique_ptr<covTUIEditField> m_pattID;
    std::unique_ptr<covTUIEditFloatField> m_size;
    std::unique_ptr<covTUIEditIntField> m_pattGroup;
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
        return m_offset;
    };
    int getMarkerGroup() const;
    void tabletEvent(coTUIElement *tUIItem) override;
    double getSize() const;
    std::string getPattern() const;
    bool isVisible() const;
    bool isObjectMarker() const;
    void setObjectMarker(bool o);
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
