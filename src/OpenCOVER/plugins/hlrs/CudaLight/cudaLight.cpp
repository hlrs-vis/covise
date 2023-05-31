/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//#include "cudaEngine.h"
#include "cudaLight.h"

#include <OpenVRUI/coMenuItem.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>

#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Material>
#include <osg/TexEnv>
#include <osg/Texture1D>

#include <sysdep/opengl.h>

#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coPotiToolboxItem.h>

#include <cover/coVRMSController.h>
#include <cover/coVRShader.h>

cudaLight *cudaLight::plugin = NULL;


cudaLight::cudaLight()
: coVRPlugin(COVER_PLUGIN_NAME)
, initDone(false)
, menu(NULL)
{
    plugin = this;
    fprintf(stderr, "cudaLight::cudaLight()\n");
    tuiTab = new coTUITab("cudaLight", coVRTui::instance()->mainFolder->getID());
    tuiTab->setPos(0, 0);
    tuiSumm = new coTUIToggleButton("summ", tuiTab->getID());
    tuiSumm->setEventListener(this);
    tuiSumm->setPos(0, 0);
    tuiSumm->setState(true);

}

cudaLight::~cudaLight()
{
}

void cudaLight::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiSumm)
    {
        
    fprintf(stderr, "cudaLight::simulate\n");
    }
}

bool cudaLight::init()
{
    fprintf(stderr, "cudaLight::init\n");
    dD = new dustDrawable();
    return true;
}

void cudaLight::preDraw(osg::RenderInfo &)
{

    
}

void cudaLight::preFrame()
{
    if (!initDone)
        return;

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        dustDrawable *drawable = dynamic_cast<dustDrawable *>(i->second->getDrawable(0));
        if (drawable)
        {
            drawable->preFrame();
        }
    }

    
    if (tuiSumm->getState())
    {

    
    }
}

void cudaLight::postFrame()
{
    if (!initDone)
        return;

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        dustDrawable *drawable = dynamic_cast<dustDrawable *>(i->second->getDrawable(0));
        if (drawable)
            drawable->postFrame();
    }
    if (tuiSumm->getState())
    {
    }
}

dustDrawable::dustDrawable()
    : osg::Drawable()
    , coMenuListener()
    , coTUIListener()
    , state(NULL)
    , animate(false)
    , anim(0)
    , threshold(0)
{

    //box = osg::BoundingBox(b[0], b[1], b[2], b[3], b[4], b[5]);
    renderer = NULL;

    setDataVariance(Object::DYNAMIC);
    threshold = FLT_MAX;
    changed = true;
    doReset = false;
}

dustDrawable::~dustDrawable()
{
    delete renderer;
}

void dustDrawable::menuReleaseEvent(coMenuItem * /*item*/)
{
}

void dustDrawable::menuEvent(coMenuItem * /*item*/)
{
}

void dustDrawable::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiSlider)
    {

        //float thresh = tuiSlider->getValue();
        //slider->setValue(thresh);
    }
}

void dustDrawable::preFrame()
{
    //float thresh = slider->getValue();
    /*
	if (animate && anim < 5)
	anim ++;
	anim %= 5;
	*/
}

void dustDrawable::reset()
{

    doReset = true;
}

void dustDrawable::preDraw()
{
    numParticles = particles.size();

    if (doReset)
    {
        doReset = false;
    }
    if (renderer == NULL && numParticles > 0)
    {

        renderer = new ParticleRenderer();

        //renderer->setPositions(particleCoords, particles.size());
        //renderer->setPBO(cudaParticles->getPosVBO(), particles.size(), CUDA_USE_DOUBLE);
        /*float color[4];
        color[0] = 1;
        color[1] = 1;
        color[2] = 1;
        color[3] = 1;
        renderer->setBaseColor(color);*/
        for (int i = 0; i < numParticles; i++)
        {
            box.expandBy(osg::Vec3(particles[i].xc * 2, particles[i].yc * 2, particles[i].zc * 2));
            box.expandBy(osg::Vec3(-particles[i].xc * 2, -particles[i].yc * 2, -particles[i].zc * 2));
        }
        dirtyBound();
    }
    if (cudaLight::instance()->tuiSumm->getState())
    {
        int deltaT = 900;
        int numIterationsPerFrame = 960;
        time_t sTime = cudaLight::instance()->SimulationTime;
        
    }
}

void dustDrawable::postFrame()
{
}

dustDrawable::dustDrawable(const dustDrawable &draw, const osg::CopyOp &op)
    : osg::Drawable(draw, op)
    , coMenuListener()
    , coTUIListener()
{
}

void dustDrawable::drawImplementation(osg::RenderInfo &renderInfo) const
{
    if (renderer != NULL)
    {

        osg::ref_ptr<osg::StateSet> currentState = new osg::StateSet;
        renderInfo.getState()->captureCurrentState(*currentState);
        renderInfo.getState()->pushStateSet(currentState.get());

        // Save the current state
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        /* enum DisplayMode
		{
		PARTICLE_POINTS,
		PARTICLE_SPRITES,
		PARTICLE_SPRITES_COLOR,
		PARTICLE_NUM_MODES
		
        if (cudaLight::instance()->tuiRender->getState())
        {
            renderer->setSpriteSize(cudaLight::instance()->particleSize);
            renderer->display(ParticleRenderer::PARTICLE_SPRITES, cudaParticles->getNumActiveParticles());
            //renderer->display(ParticleRenderer::PARTICLE_POINTS);
        }*/

        // Restore the current state
        glPopAttrib();
        glPopClientAttrib();
        renderInfo.getState()->popStateSet();
    }
}

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
osg::BoundingBox dustDrawable::computeBoundingBox() const
#else
osg::BoundingBox dustDrawable::computeBound() const
#endif
{
    return box;
}

osg::Object *dustDrawable::cloneType() const
{
    return new dustDrawable();
}

osg::Object *dustDrawable::clone(const osg::CopyOp &op) const
{
    return new dustDrawable(*this, op);
}

COVERPLUGIN(cudaLight)
