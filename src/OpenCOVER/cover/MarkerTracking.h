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
    virtual int loadPattern(const char *)
    {
        return -1;
    };
    virtual bool isVisible(int)
    {
        return false;
    };
    virtual osg::Matrix getMat(int /* pattID */, double /*pattCenter*/[2], double /*pattSize*/, double /* pattTrans */[3][4])
    {
        osg::Matrix m;
        return m;
    };
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
private:
    static MarkerTracking *art;
    MarkerTracking();
    std::string m_configPath;
    covise::Message msg;

    bool objTracking = false;
    int numObjectMarkers;
    std::list<MarkerTrackingMarker *> objectMarkers;

public:
    bool running = false;
    virtual ~MarkerTracking();
    static MarkerTracking *instance();
    coTUITab *artTab;
    bool flipH;
    MarkerTrackingInterface *arInterface = nullptr;
    RemoteARInterface *remoteAR = nullptr;

    void update();
    void config();
    bool isRunning();

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
    std::list<MarkerTrackingMarker *> markers;
    void addMarker(MarkerTrackingMarker *);
    bool doMerge;
    bool testImage = false;
};

class COVEREXPORT MarkerTrackingMarker : public coTUIListener
{
private:
struct Coord{
    Coord() = default;
    Coord(const std::string &name, coTUIGroupBox* group);
    Coord &operator=(const Coord&);
    std::string name;
    float value() const;
    coTUIEditFloatField * edit = nullptr;
};
    int pattGroup = -1;
    float oldpattGroup = -1;
    double pattCenter[2] = {0.0, 0.0};
    double pattTrans[3][4];
    bool objectMarker = false;
    osg::Matrix offset;
    osg::Matrix Ctrans;
    osg::Matrix Mtrans;
    coTUILabel *markerLabel = nullptr;
    coTUIToggleButton *vrmlToPf = nullptr;
    coTUIEditFloatField * pattID = nullptr;
    coTUIEditFloatField *size;
    std::array<Coord, 6> m_transform; 
    void createUi(const std::string &configName);
    void matToEuler(const osg::Matrix &mat);
    osg::Matrix eulerToMat() const;


public:
	MarkerTrackingMarker(const std::string &configName,int MarkerID,double size, const osg::Matrix &mat,bool VrmlToOSG);
	MarkerTrackingMarker(const std::string &Name);
	void updateData(double markerSize, osg::Matrix& mat, osg::Matrix& hostMat, bool vrmlToOsg);
    virtual ~MarkerTrackingMarker();
    osg::Matrix &getCameraTrans();
    osg::Matrix &getMarkerTrans();
    const osg::Matrix &getOffset() const
    {
        return offset;
    };
    int getMarkerGroup() const;
    virtual void tabletEvent(coTUIElement *tUIItem);
    double getSize() const;
    int getPattern() const;
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
    void setOffset(osg::Matrix &mat);
    void stopCalibration();

	osg::Matrix OpenGLToOSGMatrix;
	osg::Matrix PfToOpenGLMatrix;
};
}
#endif
