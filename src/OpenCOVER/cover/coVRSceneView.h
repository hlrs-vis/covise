/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef coVRSceneView_H
#define coVRSceneView_H
#include <osg/Version>
/*! \file
 \brief  modified osgUtil::SceneView for dealing with modified view and projection matrices

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2004
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

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

#include <util/coExport.h>
#include <osgUtil/SceneView>

namespace opencover
{

/**
 * coVRSceneView is literally a view of a scene, encapsulating the
 * camera (modelview+projection matrices), global state, lights and the scene itself.  Provides
 * methods for setting up the view and rendering it.
 */
class COVEREXPORT coVRSceneView : public osgUtil::SceneView
{
public:
    /** Construct a default scene view.*/
    coVRSceneView(osg::DisplaySettings *ds = NULL, int channel = -1);
    void createUniforms(osg::StateSet *stateset);
    static void destroyUniforms();
    static osg::Uniform *coEnvCorrectMatrixUniform;
    static osg::Uniform *coInvEnvCorrectMatrixUniform;

protected:
    virtual ~coVRSceneView();

    
    /** Do cull traversal of attached scene graph using Cull NodeVisitor.*/
    virtual bool cullStage(const osg::Matrixd &projection, const osg::Matrixd &modelview, osgUtil::CullVisitor *cullVisitor,
                           osgUtil::StateGraph *stategraph, osgUtil::RenderStage *renderStage,
                           osg::Viewport *viewport);
    int screen;
};
}
#endif
