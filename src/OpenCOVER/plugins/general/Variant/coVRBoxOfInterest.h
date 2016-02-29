/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* 
 * File:   coVRBoxOfInterest.h
 * Author: hpcagott
 *
 * Created on 21. September 2009, 14:45
 */
#ifndef _COVRBOXOFInterest_H
#define _COVRBOXOFInterest_H

#include <iostream>
#include <osg/Group>
#include <osg/ClipPlane>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Array>
#include <PluginUtil/coSphere.h>
#include <PluginUtil/coSensor.h>
#include <cover/coVRPluginSupport.h>
#include <osg/GL>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osgDB/ReadFile>
#include <osg/Program>
#include <osg/Shader>
#include <osg/Point>
#include <osg/ShadeModel>

#include <OpenVRUI/coTrackerButtonInteraction.h>

using namespace std;
using namespace vrui;
using namespace opencover;

enum TRANS
{
    XTRANS,
    YTRANS,
    ZTRANS
};


class VariantPlugin;
class interactorSpheres;
class mySensor;

class coVRBoxOfInterest
{
    friend class mySensor;

public:
    coVRBoxOfInterest(VariantPlugin *plug, coTrackerButtonInteraction *_interactionA);
    ~coVRBoxOfInterest();
    osg::ClipNode *createClipNode(std::string cnName);
    void showHide(bool state);
    void setMatrix(osg::Matrix mat);
    //TODO std::string getActiveSensorName();
    bool isSensorActiv(std::string sensorName);
    osg::Matrix getMat();
    osg::Matrix getinvMat();
    //     void setCoord(osg::Vec3 center,osg::Vec3 length);
    void setScale(osg::Matrix startMat, osg::Vec3 scale, TRANS direction);
    static void loadTransparentGeostate(osg::StateSet *stateSet);
    void setStartMatrix();
    osg::MatrixTransform *getBoxCenterMt();
    osg::MatrixTransform *getBoxGeoMt();
    osg::MatrixTransform *getBoxaaMt();
    osg::MatrixTransform *getBoxbbMt();
    osg::MatrixTransform *getBoxccMt();
    osg::MatrixTransform *getBoxddMt();
    osg::MatrixTransform *getBoxeeMt();
    osg::MatrixTransform *getBoxffMt();
    osg::Vec3 getLength();
    void setLentgh(osg::Vec3 scale);
    void attachClippingPlanes(osg::Node *varNode, osg::ClipNode *mycn);
    void releaseClippingPlanes(osg::Node *varNode, osg::ClipNode *mycn);
    void updateClippingPlanes();
    osg::Matrix getGlobalMat(osg::Node *node);

    coTrackerButtonInteraction *_interactionA; ///< interaction for first button

private:
    void createLines(osg::Geode *node, osg::Vec3Array *vertices);
    void createQuads(osg::Geode *node, osg::Vec3Array *vertices);

    VariantPlugin *plugin;
    coSensorList sensorList;
    osg::ClipNode *parent;
    osg::MatrixTransform *boiNode;
    osg::MatrixTransform *bMt;
    // osg::ClipNode *mycn;
    osg::Box *b;
    interactorSpheres *bSphere;
    osg::Vec3 length;
    osg::ref_ptr<osg::ClipPlane> cp[6]; //the clipplanes
};

class interactorSpheres
{
private:
    osg::ref_ptr<osg::TessellationHints> hint;
    void updateSpheres(osg::Vec3 center, osg::Vec3 length);

    osg::MatrixTransform *centerMt;
    osg::Matrix startCenterMat;
    osg::Sphere *centerSphere;
    mySensor *centerSensor;

    osg::MatrixTransform *aaMt;
    osg::Matrix startaaMat;
    osg::Sphere *aaSphere;
    mySensor *aaSensor;

    osg::MatrixTransform *bbMt;
    osg::Matrix startbbMat;
    osg::Sphere *bbSphere;
    mySensor *bbSensor;

    osg::MatrixTransform *ccMt;
    osg::Matrix startccMat;
    osg::Sphere *ccSphere;
    mySensor *ccSensor;

    osg::MatrixTransform *ddMt;
    osg::Matrix startddMat;
    osg::Sphere *ddSphere;
    mySensor *ddSensor;

    osg::MatrixTransform *eeMt;
    osg::Matrix starteeMat;
    osg::Sphere *eeSphere;
    mySensor *eeSensor;

    osg::MatrixTransform *ffMt;
    osg::Matrix startffMat;
    osg::Sphere *ffSphere;
    mySensor *ffSensor;

    osg::MatrixTransform *mt;

public:
    interactorSpheres(osg::Node *node, osg::Vec3 center, osg::Vec3 length, coTrackerButtonInteraction *_interactionA);
    ~interactorSpheres();
    bool isSensorActiv(std::string sensorName);
    void setStateSet(osg::StateSet *stateSet);
    void updateSpherePos(osg::Vec3 scaleVec, osg::Vec3 size, TRANS direction);
    void printMatrix(osg::Matrix ma);
    void setStartMatrix();
    osg::MatrixTransform *getCenterMt();
    osg::MatrixTransform *getaaMt();
    osg::MatrixTransform *getbbMt();
    osg::MatrixTransform *getccMt();
    osg::MatrixTransform *getddMt();
    osg::MatrixTransform *geteeMt();
    osg::MatrixTransform *getffMt();
};
class mySensor : public coPickSensor
{
public:
    mySensor(osg::Node *node, std::string name, coTrackerButtonInteraction *_interactionA, osg::ShapeDrawable *cSphDr);
    ~mySensor();

    void activate();
    void disactivate();
    std::string getSensorName();
    bool isSensorActive();

private:
    std::string sensorName;
    bool isActive;
    coTrackerButtonInteraction *_interA;
    osg::ShapeDrawable *shapDr;
};
#endif /* _COVRBOXOFInterest_H */

//                                  7    ........................ 6
//                                     . .                    . .
//                                 .     .     f           .    .
//                              .        .   f         .        .
//                          4  ........................  5      .
//                             .         .         cc .         .
//                             .    d    .            .    b    .
//                             .   d     .      cen   .   b     .
//                             .     3   .      ter   .         .
//                             .         ..aa.................... 2
//                             .       .              .        .
//                             .    .            e    .      .
//                             .  .            e      .  .
//                           0 ........................ 1
