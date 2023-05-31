/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once
/****************************************************************************\ 
 **                                                            (C)2008 HLRS  **
 **                                                                          **
 ** Description: Gizmo OpenCOVER Plugin (draws a gizmo)                      **
 **                                                                          **
 **                                                                          **
 ** Author: Matthias Epple		                                             **
 **                                                                          **
 ** History:  								                                 **
 ** July 2020  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <PluginUtil/coVR3DTransGizmo.h>
#include <PluginUtil/coVR3DRotGizmo.h>
#include <PluginUtil/coVR3DScaleGizmo.h>
#include <PluginUtil/coVR3DGizmo.h>



class GizmoTest : public opencover::coVRPlugin 
{
public:
  GizmoTest();
  void preFrame() override;

private:
  osg::ref_ptr<osg::MatrixTransform> _scene;
  osg::ref_ptr<osg::MatrixTransform> _t1;
  osg::ref_ptr<osg::MatrixTransform> _t2;
  osg::ref_ptr<osg::MatrixTransform> _t21;
  osg::ref_ptr<osg::MatrixTransform> _t22;

  osg::ref_ptr<osg::Geode> _gizmoGeode;
  osg::ref_ptr<osg::MatrixTransform> _root;
  osg::ref_ptr<osg::Node> _node;

  osg::ref_ptr<osg::MatrixTransform> _scale;

  osg::ref_ptr<osg::Geode> _cube1;
  osg::ref_ptr<osg::Geode> _cube2;  
  osg::ref_ptr<osg::Geode> _cube21;
  osg::ref_ptr<osg::Geode> _cube22;


  opencover::coVR3DTransGizmo* _transgizmo;
  opencover::coVR3DRotGizmo* _rotgizmo;
  opencover::coVR3DScaleGizmo* _scalegizmo;
  opencover::coVR3DTransRotInteractor* _transRotInteractor;
  opencover::coVR3DGizmo* _gizmo;





  osg::ref_ptr<osg::Geode> _circle;

  osg::Vec3Array* circleVerts(int plane, int approx);
  osg::Geode* circles( int plane, int approx );

};
