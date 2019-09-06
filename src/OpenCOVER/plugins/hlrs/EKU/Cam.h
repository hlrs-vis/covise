/* This file is part of COVISE.
 *

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <unistd.h>

#include<cover/coVRPluginSupport.h>
#include<osg/ShapeDrawable>
#include<osg/Vec4>
#include<osg/NodeCallback>
#include<osg/PositionAttitudeTransform>
#include<osg/Material>
#include<osg/MatrixTransform>
#include<osg/Quat>
#include<osg/BlendFunc>
#include<osgText/Font>
#include<osgText/Text>

#include<Sensor.h>

class mySensor;
class Cam
{   
public:
    static double imgWidth;
    static double imgHeight;
    static double imgWidthPixel;
    static double imgHeightPixel;
    static double fov;
    static double depthView;
    static double focalLengthPixel;

    Cam(const osg::Vec3 pos, const osg::Vec2 rot, const osg::Vec3Array &observationPoints,const std::string name);
    Cam(const osg::Vec3 pos, const osg::Vec2 rot,const std::string name);
    ~Cam();


    const osg::Vec2 rot; // [0]=alpha =zRot, [1]=beta =yRot
    const osg::Vec3 pos;

    //const osg::Vec3Array* obsPoints =nullptr; // NOTE: remove later

    //check if points are visible for this camera
    void calcVisMat(const osg::Vec3Array &observationPoints);
    std::vector<int> visMat;
    std::string getName()const{return name;}
protected:
    const std::string name;
private:

    // Calculates if Obstacles are in line of sigth betwenn camera and observation Point
    bool calcIntersection(const osg::Vec3d& end);


};

class CamDrawable
{
private:
    osg::Vec3Array* verts;
    osg::Vec4Array* colors;
    osg::ref_ptr<osg::Group> group;
    osg::ref_ptr<osg::Geode> camGeode;
    osg::ref_ptr<osg::MatrixTransform> transMat;
    osg::ref_ptr<osg::MatrixTransform> rotMat;
    osg::ref_ptr<osgText::Text> text;

public:
    static size_t count;
    Cam* cam=nullptr;
    osg::Geode* plotCam();
    void updateFOV(float value);
    void updateVisibility(float value);
    void updateColor();
    void resetColor();
    //CamDrawable(const osg::Vec3 pos, const osg::Vec2 rot,const std::string name);
    CamDrawable(Cam* cam);
    //CamDrawable(Cam cam);
    ~CamDrawable();

    osg::ref_ptr<osg::Group> getCamDrawable()const{return group;}
    osg::ref_ptr<osg::Geode> getCamGeode()const{return camGeode;}
};


