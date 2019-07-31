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
public:

    Cam();
    ~Cam();

    osg::Geode* createPyramid();
    void setFOV(float radius);
    void setVisibility(float vis);
    void updateFOV(float value);
    void updateVisibility(float value);

private:

    osg::Vec3Array* verts;
    osg::ref_ptr<osg::Geode> camGeode;

    osg::PositionAttitudeTransform* revolution;


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
