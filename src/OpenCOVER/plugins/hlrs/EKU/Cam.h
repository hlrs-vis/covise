/* This file is part of COVISE.
 *

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <unistd.h>

#include <cover/coVRPluginSupport.h>
#include <osg/ShapeDrawable>
#include <osg/Vec4>
#include <osg/NodeCallback>
#include <osg/PositionAttitudeTransform>
#include <osg/Material>



class Cam
{
private:

    osg::Vec3Array* verts;
    osg::Vec2i rot;
    osg::Vec3i pos;
    osg::ref_ptr<osg::Geode> camGeode;
    osg::PositionAttitudeTransform* revolution;
public:

    static int imgWidthPixel;
    static int imgHeigthPixel;
    static int fov;
    static int depthView;
    static int focalLengthPixel;
    //osg::Matrix Rz = osg::Matrix::rotate(rot(0),0,0,1);// Z rotation
    //osg::Matrix Ry = osg::Matrix::rotate(rot(1),0,1,0);// Y rotation
    //osg::Matrix T =  osg::Matrix::translate(pos); //Translation



    Cam();
    ~Cam();

    osg::Geode* plotCam();
    void setFOV(float radius);
    void setVisibility(float vis);
    void updateFOV(float value);
    void updateVisibility(float value);




};

class RotationCallback : public osg::NodeCallback
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
