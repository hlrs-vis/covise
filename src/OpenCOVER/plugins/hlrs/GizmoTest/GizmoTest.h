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

class GizmoTest :public opencover::coVRPlugin 
{
public:
  GizmoTest();

private:
  osg::ref_ptr<osg::Node> _node;
  osg::ref_ptr<osg::Geode> _geode;
  osg::ref_ptr<osg::MatrixTransform> _transform;
  osg::ref_ptr<GizmoDrawable> _gizmo;

};