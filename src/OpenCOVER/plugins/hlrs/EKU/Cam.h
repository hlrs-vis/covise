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

    void calcVisMat(const osg::Vec3Array &observationPoints);
    std::vector<int> visMat;

protected:
    const std::string name;
private:

    // Calculates if Obstacles are in line of sigth betwenn camera and observation Point
    bool calcIntersection(const osg::Vec3d& end);


};

class CamDrawable: public Cam
{
private:
    osg::Vec3Array* verts;
    osg::ref_ptr<osg::Group> group;
    osg::ref_ptr<osg::Geode> camGeode;
    osg::ref_ptr<osg::MatrixTransform> transMat;
    osg::ref_ptr<osg::MatrixTransform> rotMat;
    osg::ref_ptr<osgText::Text> text;
    //osg::PositionAttitudeTransform* revolution;


public:
    static size_t count;

    osg::Geode* plotCam();
    void updateFOV(float value);
    void updateVisibility(float value);

    CamDrawable(const osg::Vec3 pos, const osg::Vec2 rot,const std::string name);
    //CamDrawable(Cam cam);
    ~CamDrawable();

    osg::ref_ptr<osg::Group> getCamDrawable()const{return group;}
};

/*class RotationCallback : public osg::NodeCallback
{
public:
   // Default constructor.
   RotationCallback()
      : osg::NodeCallback()
      , angle(0.0f)
   {}
   // This updater function rotates the geometry by incrementing the current
   // angle by one degree each rendering update.
   virtual void operator()(osg::Node* node, osg::NodeVisitor* nv)
   {
#ifndef _WIN32
      // On Linux, only update at 36 frames per second.
      usleep((unsigned long)((1.0 / 36.0) * 1000.0));
#endif
      osg::PositionAttitudeTransform* pat =
         dynamic_cast<osg::PositionAttitudeTransform*>(node);
      if ( pat ) {
         pat->setAttitude( osg::Quat( osg::DegreesToRadians(angle), osg::Vec3(0.0f, 1.0f, 0.0f) ) );
         angle += 1.0f;
      }
      // Always call base class traverse to send the visitor on its way.
      traverse(node, nv);
   }
private:
   float angle;
};
*/
