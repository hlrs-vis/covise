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
#include "coVRGizmo.h"
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <PluginUtil/coVR3DTransGizmo.h>

class GizmoTest :public opencover::coVRPlugin 
{
public:
  GizmoTest();
  void preFrame() override;

private:
  osg::ref_ptr<osg::MatrixTransform> _scene;
  osg::ref_ptr<osg::Geode> _gizmoGeode;
  osg::ref_ptr<osg::MatrixTransform> _root;
  osg::ref_ptr<GizmoDrawable> _gizmo;
  osg::ref_ptr<osg::Node> _node;

  osg::ref_ptr<osg::Geode> _cube1;
  osg::ref_ptr<osg::Geode> _cube2;

  opencover::coVR3DTransGizmo* _transgizmo;
  opencover::coVR3DTransRotInteractor* _transRotInteractor;


};
