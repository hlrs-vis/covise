/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
\brief ARToolKit optical tracking system interface classes

\author Uwe Woessner <woessner@hlrs.de>
\author (C) 2002
Computer Centre University of Stuttgart,
Allmandring 30,
D-70550 Stuttgart,
Germany

\date    July 2002
*/

#ifndef ARTOOLKIT_H
#define ARTOOLKIT_H

#include <list>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Drawable>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Vec3>

namespace opencover
{
class ARToolKitMarker;
class coTUILabel;
class coTUIToggleButton;
class coTUIEditFloatField;
class coTUIEditFloatField;
class coTUITab;
class coTUIElement;
}

#include <net/covise_connect.h>
#include <net/message.h>

#include "coTabletUI.h"

namespace opencover
{
class COVEREXPORT ARToolKitNode : public osg::Drawable
{
private:
    bool displayVideo; // true if CoviseConfig.displayVideo is set
    bool renderTextures;
    std::string m_artoolkitVariant;

public:
    ARToolKitNode(std::string artoolkitVariant);
    virtual ~ARToolKitNode();
    static ARToolKitNode *theNode;
    virtual void drawImplementation(osg::RenderInfo &renderInfo) const;
    /** Clone the type of an object, with Object* return type.
		Must be defined by derived classes.*/
    virtual osg::Object *cloneType() const;

    /** Clone the an object, with Object* return type.
		Must be defined by derived classes.*/
    virtual osg::Object *clone(const osg::CopyOp &) const;
};

class COVEREXPORT ARToolKitInterface
{
public:
    ARToolKitInterface(){};
    virtual ~ARToolKitInterface(){};
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
    virtual bool isARToolKit()
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

class COVEREXPORT ARToolKit
{
private:
    static ARToolKit *art;
    std::string m_configPath;
    covise::Message msg;

    bool objTracking;
    int numObjectMarkers;
    std::list<ARToolKitMarker *> objectMarkers;

public:
    bool running;
    ARToolKit();
    virtual ~ARToolKit();
    static ARToolKit *instance();
    coTUITab *artTab;
    bool flipH;
    ARToolKitInterface *arInterface;
    RemoteARInterface *remoteAR;

    void update();
    void config();
    bool isRunning();

    unsigned char *videoData;
    unsigned char *videoDataRight;
    int videoWidth;
    int videoHeight;
    int videoMode;
    int videoDepth;
    bool stereoVideo;
    bool videoMirrorLeft;
    bool videoMirrorRight;
    std::string m_artoolkitVariant;
    std::list<ARToolKitMarker *> markers;
    void addMarker(ARToolKitMarker *);
    bool doMerge;
    bool testImage;
};

class COVEREXPORT ARToolKitMarker : public coTUIListener
{
private:
    int pattID;
    double pattSize;
    double pattCenter[2];
    double pattTrans[3][4];
    bool VrmlToPf;
    bool objectMarker;
    float x, y, z, h, p, r;
    bool visible;
    osg::Matrix offset;
    osg::Matrix Ctrans;
    osg::Matrix Mtrans;
    osg::Matrix OpenGLToOSGMatrix;
    osg::Matrix PfToOpenGLMatrix;
    coTUILabel *markerLabel;
    coTUIToggleButton *vrmlToPfFlag;
    coTUIEditFloatField *size;
    coTUIEditFloatField *posX;
    coTUIEditFloatField *posY;
    coTUIEditFloatField *posZ;
    coTUIEditFloatField *rotH;
    coTUIEditFloatField *rotP;
    coTUIEditFloatField *rotR;

public:
    ARToolKitMarker(const char *Name);
    virtual ~ARToolKitMarker();
    osg::Matrix &getCameraTrans();
    osg::Matrix &getMarkerTrans();
    osg::Matrix &getOffset()
    {
        return offset;
    };
    virtual void tabletEvent(coTUIElement *tUIItem);
    double getSize();
    int getPattern();
    bool isVisible();
    bool isObjectMarker()
    {
        return objectMarker;
    };
    void setObjectMarker(bool o)
    {
        objectMarker = o;
    };
    void setColor(float r, float g, float b);
    osg::Geode *quadGeode;
    osg::ref_ptr<osg::MatrixTransform> markerQuad;
    osg::MatrixTransform *posSize;
    osg::Vec4Array *colors;
    osg::Geometry *geom;
    coTUIToggleButton *displayQuad;
    coTUIToggleButton *calibrate;
    int numCalibSamples;
    osg::Matrix matrixSumm;
    bool lastVisible;
    void setOffset(osg::Matrix &mat);
};
}
#endif
