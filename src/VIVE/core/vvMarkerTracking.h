/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include <list>
#include <string>
#include <array>

#include <vsg/maths/mat4.h>
#include <vsg/nodes/MatrixTransform.h>
#include <vsg/maths/vec3.h>

#include <OpenConfig/array.h>
#include <OpenConfig/value.h>
#include <OpenConfig/file.h>

#include "vvTabletUI.h"
namespace vive
{
class MarkerTrackingMarker;
class vvTUILabel;
class vvTUIToggleButton;
class vvTUIEditFloatField;
class vvTUIEditFloatField;
class vvTUITab;
class vvTUIElement;
class vvTUIGroupBox;
}

#include <net/covise_connect.h>
#include <net/message.h>

#include "vvTUIListener.h"
#include "vvTabletCovConfig.h"

namespace vive
{
class VVCORE_EXPORT MarkerTrackingNode : public vsg::Inherit<vsg::Node, MarkerTrackingNode>
{
private:
    bool displayVideo; // true if CoviseConfig.displayVideo is set
    bool renderTextures;
    std::string m_MarkerTrackingVariant;

public:
    MarkerTrackingNode(std::string MarkerTrackingVariant);
    virtual ~MarkerTrackingNode();
    static MarkerTrackingNode *theNode;
    //virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
};

class VVCORE_EXPORT MarkerTrackingInterface
{
public:
    MarkerTrackingInterface(){};
    virtual ~MarkerTrackingInterface(){};
    virtual std::string loadPattern(const std::string& /*patternId*/)
    {
        return "";
    };
    virtual bool isVisible(const MarkerTrackingMarker *marker) = 0;

    virtual vsg::dmat4 getMat(const MarkerTrackingMarker *marker) = 0;

    virtual void updateViewerPos(const vsg::vec3 &vp)
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

class VVCORE_EXPORT RemoteARInterface
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

class VVCORE_EXPORT MarkerTracking : public vvTUIListener
{
public:

    enum ColorMode
    {
        RGB,
        RGBA,
        BGRA
    };
    bool running = false;
    ~MarkerTracking();
    static MarkerTracking *instance();
    MarkerTrackingMarker *getMarker(const std::string &name);
    MarkerTrackingMarker *getOrCreateMarker(const std::string &name, const std::string &pattern, double size, const vsg::dmat4 &offset, bool vrml, bool isObjectMarker = false);
    
    vvTUITab *artTab;
    bool flipH;
    MarkerTrackingInterface *arInterface = nullptr;
    RemoteARInterface *remoteAR = nullptr;
    void config();
    void update();
    bool isRunning();
    void changeObjectMarker(MarkerTrackingMarker *m);
    vive::config::File &markerDatabase();
    unsigned char *videoData = nullptr;
    unsigned char *videoDataRight = nullptr;
    int videoWidth = 0;
    int videoHeight = 0;
    ColorMode videoMode = RGB;
    int videoDepth;
    bool stereoVideo = false;
    bool videoMirrorLeft = false;
    bool videoMirrorRight = false;
    std::string m_MarkerTrackingVariant = "MarkerTracking";
    std::map<std::string, std::unique_ptr<MarkerTrackingMarker>> markers;
    bool testImage = false;
    vvTUIFrame* m_trackingFrame = nullptr;
private:
    static MarkerTracking *art;
    MarkerTracking();
    void tabletPressEvent(vvTUIElement* tUIItem) override;

    std::string m_configPath;
    covise::Message msg;

    bool objTracking = false;
    int numObjectMarkers;
    std::vector<MarkerTrackingMarker *> objectMarkers;
    std::unique_ptr<vive::config::File> m_markerDatabase;
    vvTUIButton* m_configureMarkerBtn = nullptr;
    vvTUIButton* m_saveBtn = nullptr;
    vvTUIFrame* m_buttonsFrame = nullptr;
};

constexpr int noMarkerGroup = -1;

class VVCORE_EXPORT MarkerTrackingMarker : public vvTUIListener
{
friend class MarkerTracking;

private:

    float m_oldpattGroup = -1;
    double m_pattCenter[2] = {0.0, 0.0};
    double m_pattTrans[3][4];
    vsg::dmat4 m_offset;
    vsg::dmat4 m_cameraTransform;

    vvTUIButton *m_toggleConfigOff = nullptr;
    vvTUIButton *m_toggleConfigOn = nullptr;
    vvTUIFrame *m_layoutGroup = nullptr;
    std::unique_ptr<vvTUILabel> m_configLabel;
    std::unique_ptr<covTUIToggleButton> m_vrmlToPf;
    std::unique_ptr<covTUIToggleButton> m_objectMarker;
    std::unique_ptr<covTUIEditField> m_pattID;
    std::unique_ptr<covTUIEditFloatField> m_size;
    std::unique_ptr<covTUIEditIntField> m_pattGroup;
    // std::array<std::shared_ptr<Coord>, 6> m_transform;
    std::unique_ptr<covTUIEditFloatFieldVec3> m_xyz; 
    std::unique_ptr<covTUIEditFloatFieldVec3> m_hpr; 
    void createUiandConfigValues(const std::string &configName);
    void matToEuler(const vsg::dmat4 &mat);
    vsg::dmat4 eulerToMat() const;
    void updateMatrices();
    void init();
	void updateData(double markerSize, const vsg::dmat4& mat, bool vrmlToOsg);

	MarkerTrackingMarker(const std::string &configName, const std::string &pattern, double size, const vsg::dmat4 &mat, bool vrml);
	MarkerTrackingMarker(const std::string &Name);
public:
    virtual ~MarkerTrackingMarker();
    MarkerTrackingMarker(const MarkerTrackingMarker&) = delete;
    MarkerTrackingMarker& operator=(const MarkerTrackingMarker&) = delete;
    MarkerTrackingMarker(MarkerTrackingMarker&&) = default;
    MarkerTrackingMarker& operator=(MarkerTrackingMarker&&) = default;

    const vsg::dmat4 &getCameraTrans();
    vsg::dmat4 getMarkerTrans();
    const vsg::dmat4 &getOffset() const
    {
        return m_offset;
    };
    int getMarkerGroup() const;
    void tabletEvent(vvTUIElement *tUIItem) override;
    double getSize() const;
    std::string getPattern() const;
    bool isVisible() const;
    bool isObjectMarker() const;
    void setObjectMarker(bool o);
    void setColor(float r, float g, float b);
    osg::Node *quadGeode = nullptr;
    vsg::ref_ptr<vsg::MatrixTransform> markerQuad;
    vsg::MatrixTransform *posSize=nullptr;
    vsg::vec4Array *colors = nullptr;
    vsg::Node *geom = nullptr;
    vvTUIToggleButton *displayQuad = nullptr;
    vvTUIToggleButton *calibrate = nullptr;
    int numCalibSamples = 0;
    vsg::dmat4 matrixSumm;
    bool lastVisible = false;
    void setOffset(const vsg::dmat4 &mat);
    void stopCalibration();
};
}
EVSG_type_name(vive::MarkerTrackingNode);
