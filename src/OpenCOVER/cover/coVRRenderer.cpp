/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 1996					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			VRViewer.C (Performer 2.0)		*
 *									*
 *	Description		stereo viewer class			*
 *									*
 *	Author			D. Rainer				*
 *									*
 *	Date			20.08.97				*
 *				09.01.97 general viewing frustum	*
 *									*
 *	Status			in dev					*
 *									*
 ************************************************************************/

#include <util/common.h>
#include <osgViewer/View>
#include "coVRRenderer.h"
#include "coVRSceneView.h"
#include "coCullVisitor.h"
#include "coVRConfig.h"

using namespace opencover;

// OpenCOVER
/* Callback for overidding the default method for compute the offset projection and view matrices.*/
struct MyComputeStereoMatricesCallback : public osgUtil::SceneView::ComputeStereoMatricesCallback
{
    int channelNum;
    virtual osg::Matrixd computeLeftEyeProjection(const osg::Matrixd &projection) const
    {
        (void)projection;
        return coVRConfig::instance()->channels[channelNum].leftProj;
    }

    virtual osg::Matrixd computeLeftEyeView(const osg::Matrixd &view) const
    {
        (void)view;
        return coVRConfig::instance()->channels[channelNum].leftView;
    }

    virtual osg::Matrixd computeRightEyeProjection(const osg::Matrixd &projection) const
    {
        (void)projection;
        return coVRConfig::instance()->channels[channelNum].rightProj;
    }

    virtual osg::Matrixd computeRightEyeView(const osg::Matrixd &view) const
    {
        (void)view;
        return coVRConfig::instance()->channels[channelNum].rightView;
    }
};

coVRRenderer::coVRRenderer(osg::Camera *camera, int channel)
    : osgViewer::Renderer(camera)
{
    // alles gleich wie in Renderer, nur wird hier eine coVRSceneView gemacht statt der osgUtil::SceneView

    // lock the mutex for the current cull SceneView to
    // prevent the draw traversal from reading from it before the cull traversal has been completed.
    _availableQueue._queue.clear();

    coVRSceneView *sv1 =  new coVRSceneView(NULL, channel);
    coVRSceneView *sv2 = new coVRSceneView(NULL, channel);
    _sceneView[0] = sv1;
    _sceneView[1] = sv2;
    unsigned int sceneViewOptions = 0; // no HeadLight

    osg::Camera *masterCamera = _camera->getView() ? _camera->getView()->getCamera() : camera;
    osg::StateSet *stateset = masterCamera->getOrCreateStateSet();
    osgViewer::View *view = dynamic_cast<osgViewer::View *>(_camera->getView());
    
    sv1->createUniforms(stateset);
    sv2->coEnvCorrectMatrixUniform = sv1->coEnvCorrectMatrixUniform;
    sv2->coInvEnvCorrectMatrixUniform = sv1->coInvEnvCorrectMatrixUniform;

    osg::DisplaySettings *ds = _camera->getDisplaySettings() ? _camera->getDisplaySettings() : ((view && view->getDisplaySettings()) ? view->getDisplaySettings() : osg::DisplaySettings::instance().get());

    for (int i = 0; i < 2; i++)
    {
        _sceneView[i]->setGlobalStateSet(stateset);
        _sceneView[i]->setDefaults(sceneViewOptions);
        _sceneView[i]->setInheritanceMask(_sceneView[i]->getInheritanceMask() & ~osg::CullSettings::LIGHTING_MODE);
        _sceneView[i]->setInheritanceMask(_sceneView[i]->getInheritanceMask() & ~osg::CullSettings::LIGHT);
        _sceneView[i]->setCamera(_camera.get(), false);
        int channel = -1;
        for (int n = 0; n < coVRConfig::instance()->numChannels(); n++)
        {
            if (coVRConfig::instance()->channels[n].camera.get() == camera)
            {
                channel = n;
                break;
            }
        }
        if (channel >= 0)
        {
            MyComputeStereoMatricesCallback *sCallback = new MyComputeStereoMatricesCallback;
            sCallback->channelNum = channel;
            _sceneView[i]->setComputeStereoMatricesCallback(sCallback);
        }
        else
        {
            MyComputeStereoMatricesCallback *sCallback = new MyComputeStereoMatricesCallback;
            sCallback->channelNum = 0;
            _sceneView[i]->setComputeStereoMatricesCallback(sCallback);
        }
        _sceneView[i]->setDisplaySettings(ds);
        //sceneView->setSceneData(_scene.get()->getSceneData());
        _sceneView[i]->setCullVisitor(new coCullVisitor());
        _sceneView[i]->setCullVisitorLeft(new coCullVisitor());
        _sceneView[i]->setCullVisitorRight(new coCullVisitor());
    }

    // lock the mutex for the current cull SceneView to
    // prevent the draw traversal from reading from it before the cull traversal has been completed.
    _availableQueue.add(_sceneView[0].get());
    _availableQueue.add(_sceneView[1].get());
}

coVRRenderer::~coVRRenderer()
{
}
