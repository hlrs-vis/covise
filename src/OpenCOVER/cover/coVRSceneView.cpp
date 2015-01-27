/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2003 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
 */

#include "coVRPluginSupport.h"
#include "coVRSceneView.h"
#include <osgUtil/SceneView>
#include <osgUtil/UpdateVisitor>
#include <cover/coVRConfig.h>
#include <cover/coVRTui.h>

#include <osg/Timer>
#include <osg/Notify>
#include <osg/Texture>
#include <osg/VertexProgram>
#include <osg/FragmentProgram>
#include <osg/AlphaFunc>
#include <osg/TexEnv>
#include <osg/ColorMatrix>
#include <osg/LightModel>
#include <osg/CollectOccludersVisitor>

#include <osg/GLU>

using namespace osg;
using namespace osgUtil;
using namespace opencover;
using namespace covise;

coVRSceneView::coVRSceneView(DisplaySettings *ds, int c)
    : osgUtil::SceneView(ds)
{
    setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
    channel = c;
}

coVRSceneView::~coVRSceneView()
{
}

bool coVRSceneView::cullStage(const osg::Matrixd &projection, const osg::Matrixd &modelview, osgUtil::CullVisitor *cullVisitor,
                              osgUtil::StateGraph *rendergraph, osgUtil::RenderStage *renderStage, osg::Viewport *viewport)
{

    if (!_camera || !viewport)
        return false;

    osg::ref_ptr<RefMatrix> proj = new osg::RefMatrix(projection);
    osg::ref_ptr<RefMatrix> mv = new osg::RefMatrix(modelview);

    osg::Matrix rotonly = *(mv.get());
    rotonly(3, 0) = 0;
    rotonly(3, 1) = 0;
    rotonly(3, 2) = 0;
    rotonly(3, 3) = 1;
    osg::Matrix invRot;

    osg::Matrix nmv;
    osg::Matrix npm;
    invRot.invert(rotonly);
    nmv = (*(mv.get()) * invRot) * cover->invEnvCorrectMat;
    npm = cover->envCorrectMat * rotonly * *(proj.get());
    if (coVRConfig::instance()->screens[channel].render == false)
        return false;
    bool retval = SceneView::cullStage(npm, nmv, cullVisitor, rendergraph, renderStage, viewport);

    if (coVRTui::instance()->binList->size() > 0)
    {
        coVRTui::instance()->binList->updateBins();
    }
    return retval;
}
