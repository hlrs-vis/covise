/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//#include "cudaEngine.h"

#include "SPH.h"
#include "osgSimBuffer.h"

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

SPH *SPH::plugin = NULL;
SPH::SPH()
: coVRPlugin(COVER_PLUGIN_NAME)
: initDone(false)
, menu(NULL)
{
    plugin = this;
    fprintf(stderr, "SPH::SPH()\n");
    tuiTab = new coTUITab("SPH", coVRTui::instance()->mainFolder->getID());
    tuiTab->setPos(0, 0);
    tuiSimulate = new coTUIToggleButton("simulate", tuiTab->getID());
    tuiSimulate->setEventListener(this);
    tuiSimulate->setPos(0, 0);
    tuiSimulate->setState(true);

    tuiReset = new coTUIButton("reset", tuiTab->getID());
    tuiReset->setEventListener(this);
    tuiReset->setPos(1, 0);

    tuiRender = new coTUIToggleButton("render", tuiTab->getID());
    tuiRender->setEventListener(this);
    tuiRender->setPos(0, 1);
    tuiRender->setState(true);

    tuiSimTime = new coTUILabel("Year:", tuiTab->getID());
    tuiSimTime->setPos(2, 0);

    tuiNumParticlesLabel = new coTUILabel("numParticles:", tuiTab->getID());
    tuiNumParticlesLabel->setPos(3, 0);

    tuiNumParticles = new coTUIEditFloatField("numParticles:", tuiTab->getID());
    tuiNumParticles->setPos(4, 0);
    tuiNumParticles->setEventListener(this);
    tuiNumParticles->setValue(128 * 1024);

    deltaT = 900;
    tuiTestSceneLabel = new coTUILabel("TestScene:", tuiTab->getID());
    tuiTestSceneLabel->setPos(3, 1);

    tuiTestScene = new coTUIEditFloatField("TestScene:", tuiTab->getID());
    tuiTestScene->setPos(4, 1);
    tuiTestScene->setEventListener(this);
    tuiTestScene->setValue(1);

    numIterationsPerFrame = 960;
    tuiIterationsPerFrameLabel = new coTUILabel("IterationsPerFrame:", tuiTab->getID());
    tuiIterationsPerFrameLabel->setPos(3, 2);

    tuiIterationsPerFrame = new coTUIEditFloatField("IterationsPerFrame:", tuiTab->getID());
    tuiIterationsPerFrame->setPos(4, 2);
    tuiIterationsPerFrame->setEventListener(this);
    tuiIterationsPerFrame->setValue(numIterationsPerFrame);

    tuiSimulateToYearLabel = new coTUILabel("SimulateToYear:", tuiTab->getID());
    tuiSimulateToYearLabel->setPos(3, 3);

    tuiSimulateToYear = new coTUIEditFloatField("SimulateToYear:", tuiTab->getID());
    tuiSimulateToYear->setPos(4, 3);
    tuiSimulateToYear->setEventListener(this);
    tuiSimulateToYear->setValue(4000);
    simulateTo = (4000 - 2000) * 365.0 * 24.0 * 60.0 * 60.0;

    tuiActivePlanetsLabel = new coTUILabel("numActivePlanets:", tuiTab->getID());
    tuiActivePlanetsLabel->setPos(3, 4);
    numActivePlanets = 5;
    tuiActivePlanets = new coTUIEditIntField("numActivePlanets:", tuiTab->getID());
    tuiActivePlanets->setPos(4, 4);
    tuiActivePlanets->setEventListener(this);
    tuiActivePlanets->setValue(numActivePlanets);

    particleSize = 10000.0;
    tuiParticleSizeLabel = new coTUILabel("ParticleSize:", tuiTab->getID());
    tuiParticleSizeLabel->setPos(3, 5);

    tuiParticleSize = new coTUIEditFloatField("ParticleSize:", tuiTab->getID());
    tuiParticleSize->setPos(4, 5);
    tuiParticleSize->setEventListener(this);
    tuiParticleSize->setValue(particleSize);

    simCudaHelper = new SimLib::SimCudaHelper();
    simCudaHelper->Initialize(0);

    system = NULL;

    fD = new fluidDrawable();

    //totalTimer->stop();

    doReset = false;
}

SPH::~SPH()
{
}

void SPH::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiReset)
    {

        doReset = true;
    }
    else if (tUIItem == tuiNumParticles)
    {
        planetScale = tuiNumParticles->getValue();
    }
    else if (tUIItem == tuiIterationsPerFrame)
    {
        numIterationsPerFrame = tuiIterationsPerFrame->getValue();
    }
    else if (tUIItem == tuiTestScene)
    {

        deltaT = tuiTestScene->getValue();
    }
    else if (tUIItem == tuiSimulateToYear)
    {
        simulateTo = (tuiSimulateToYear->getValue() - 2000) * 365 * 24 * 60 * 60;
    }
    else if (tUIItem == tuiActivePlanets)
    {
        numActivePlanets = tuiActivePlanets->getValue();
    }
    else if (tUIItem == tuiParticleSize)
    {
        particleSize = tuiParticleSize->getValue();
    }
}

bool SPH::init()
{
    fprintf(stderr, "SPH::init\n");

    osg::Geode *g = new osg::Geode;
    g->setName("SPHGeode");
    geode["mySPH"] = g;
    fD->setName("SPHDrawable");
    if (fD.get() != NULL)
    {
        g->addDrawable(fD.get());
    }
    fD->setUseDisplayList(false);
    /*if(parent)
	parent->addChild(g);
	else*/
    cover->getObjectsRoot()->addChild(g);

    return true;
}

void SPH::preDraw(osg::RenderInfo &)
{

    if (doReset)
    {
        system->SetScene(tuiTestScene->getValue());

        system->GetSettings()->SetValue("Particles Number", tuiNumParticles->getValue());
        doReset = false;
    }
    if (system)
    {
        system->Simulate(true, true);
    }

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        fluidDrawable *drawable = dynamic_cast<fluidDrawable *>(i->second->getDrawable(0));

        if (drawable)
        {
            drawable->preDraw();
        }
    }
}

void SPH::preFrame()
{
    if (!initDone)
        return;

    if (system != NULL)
    {
        std::map<std::string, osg::Geode *>::iterator i;
        for (i = geode.begin(); i != geode.end(); i++)
        {
            fluidDrawable *drawable = dynamic_cast<fluidDrawable *>(i->second->getDrawable(0));
            if (drawable)
            {
                drawable->preFrame();
            }
        }
    }
}

void SPH::postFrame()
{
    if (!initDone)
        return;

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        fluidDrawable *drawable = dynamic_cast<fluidDrawable *>(i->second->getDrawable(0));
        if (drawable)
            drawable->postFrame();
    }
    if (tuiSimulate->getState())
    {
        SimulationTime += deltaT * numIterationsPerFrame; // +=10 Tage (default)

        if (SimulationTime > simulateTo)
            tuiSimulate->setState(false);
    }
}

fluidDrawable::fluidDrawable()
    : osg::Drawable()
    , coMenuListener()
    , coTUIListener()
    , state(NULL)
    , animate(false)
    , anim(0)
    , threshold(0)
{

    box = osg::BoundingBox(0, 0, 0, 1024, 1024, 1024);
    renderer = NULL;

    setDataVariance(Object::DYNAMIC);
    threshold = FLT_MAX;
    changed = true;
    doReset = false;
    //	cudaParticles = NULL;
}

fluidDrawable::~fluidDrawable()
{
    //delete renderer;
}

void fluidDrawable::menuReleaseEvent(coMenuItem * /*item*/)
{
}

void fluidDrawable::menuEvent(coMenuItem * /*item*/)
{
}

void fluidDrawable::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiSlider)
    {

        //float thresh = tuiSlider->getValue();
        //slider->setValue(thresh);
    }
}

void fluidDrawable::preFrame()
{
    //float thresh = slider->getValue();
    /*
	if (animate && anim < 5)
	anim ++;
	anim %= 5;
	*/
}

void fluidDrawable::reset()
{

    doReset = true;
    //	cudaParticles->setNumActiveParticles(0);
}

void fluidDrawable::preDraw()
{
    //numParticles =  particles.size();

    if (doReset)
    {
        doReset = false;
        //	cudaParticles->copyInitialData(particleCoords,particleVelos);
    }
    if (renderer == NULL)
    {

        renderer = new ParticleRenderer(this);
        SimLib::SimulationSystem *system;
        SPH::instance()->system = system = new SimLib::SimulationSystem(true, SPH::instance()->simCudaHelper, false);
        system->SetFluidPosition(make_float3(0, 0, 0));
        osgSimBuffer *sb = SPH::instance()->fD->getRenderer()->GetCudaBufferPosition();
        //sb->Alloc(32*1024*sizeof(float));
        system->SetExternalBuffer(SimLib::Sim::BufferPosition, sb);
        sb = SPH::instance()->fD->getRenderer()->GetCudaBufferColors();
        //sb->Alloc(32*1024*sizeof(int));
        system->SetExternalBuffer(SimLib::Sim::BufferColor, sb);

        system->Init();
        system->GetSettings()->SetValue("Timestep", 0.002);
        system->GetSettings()->SetValue("Particles Number", 128 * 1024);
        system->GetSettings()->SetValue("Grid World Size", 1024);
        system->GetSettings()->SetValue("Simulation Scale", 0.002);
        system->GetSettings()->SetValue("Rest Density", 1000);
        system->GetSettings()->SetValue("Rest Pressure", 0);
        system->GetSettings()->SetValue("Ideal Gas Constant", 1.5);
        system->GetSettings()->SetValue("Viscosity", 1);
        system->GetSettings()->SetValue("Boundary Stiffness", 20000);
        system->GetSettings()->SetValue("Boundary Dampening", 256);

        system->SetFluidPosition(make_float3(0.5, 0.5, 0.5));
        system->SetNumParticles(128 * 1024);

        //system->Clear();
        //system->GetSettings()->Print();
        cout << "\n";
        //system->SetPrintTiming(true);
        system->SetScene(1);

        //ocu::GPUTimer *totalTimer = new ocu::GPUTimer();
        //totalTimer->start();

        system->PrintMemoryUse();

        cudaThreadSynchronize();

        /*for(int i=0;i<numParticles;i++)
		{
		box.expandBy(osg::Vec3(particles[i].xc*2,particles[i].yc*2,particles[i].zc*2));
		box.expandBy(osg::Vec3(-particles[i].xc*2,-particles[i].yc*2,-particles[i].zc*2));
		}*/
        dirtyBound();
        SPH::instance()->initDone = true;
        sb = SPH::instance()->fD->getRenderer()->GetCudaBufferPosition();
        hostPtr = new char[sb->GetSize()];
    }
    else
    {
        static bool firsttime = true;
        if (firsttime)
        {
            SimLib::SimBuffer *sb = SPH::instance()->fD->getRenderer()->GetCudaBufferPosition();
            void *devPtr = sb->GetPtr();
            cudaMemcpy(hostPtr, devPtr, sb->GetSize(), cudaMemcpyDeviceToHost);
            float *pos = (float *)hostPtr;
            SPH::instance()->fD->getRenderer()->setPos(pos);
            box.init();
            int numV = (sb->GetSize() / sizeof(float)) / 4;
            for (int i = 0; i < numV; i++)
            {
                pos[i * 4 + 3] = 1.0;
                box.expandBy(pos[i * 4] * 5, pos[i * 4 + 1] * 5, pos[i * 4 + 2] * 5);
            }

            cudaMemcpy(devPtr, hostPtr, sb->GetSize(), cudaMemcpyHostToDevice);
            firsttime = false;
        }
    }
    /*if(SPH::instance()->tuiSimulate->getState())
	{
	int deltaT = 900;
	int numIterationsPerFrame = 960;
	time_t sTime = SPH::instance()->SimulationTime;
	if(cudaParticles)
	{
	if(cudaParticles->getNumActiveParticles() < numParticles)
	{
	int i;
	for(i=cudaParticles->getNumActiveParticles();i<numParticles;i++)
	{
	if(particles[i].s > sTime)
	break;
	}
	cudaParticles->setNumActiveParticles(i);
	static int oldna = 0;
	if(oldna !=cudaParticles->getNumActiveParticles())
	{
	fprintf(stderr,"numActiveParticles = %d\n",cudaParticles->getNumActiveParticles());
	oldna = cudaParticles->getNumActiveParticles();
	}
	}
	cudaParticles->integrate(SPH::instance()->deltaT,SPH::instance()->numIterationsPerFrame,SPH::instance()->numActivePlanets); //1/4 Stunde
	}
	}*/
}

void fluidDrawable::postFrame()
{
}

fluidDrawable::fluidDrawable(const fluidDrawable &draw, const osg::CopyOp &op)
    : osg::Drawable(draw, op)
    , coMenuListener()
    , coTUIListener()
{
    renderer = NULL;
}

void fluidDrawable::drawImplementation(osg::RenderInfo &renderInfo) const
{
    if (renderer != NULL)
    {

        osg::ref_ptr<osg::StateSet> currentState = new osg::StateSet;
        renderInfo.getState()->captureCurrentState(*currentState);
        renderInfo.getState()->pushStateSet(currentState.get());

        // Save the current state
        glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        if (SPH::instance()->tuiRender->getState())
        {
            //renderer->display(ParticleRenderer::PARTICLE_POINTS);

            renderer->setSpriteSize(4);
            renderer->display(ParticleRenderer::PARTICLE_SPRITES); // ,cudaParticles->getNumActiveParticles());
        }

        // Restore the current state
        glPopAttrib();
        glPopClientAttrib();
        renderInfo.getState()->popStateSet();
    }
}

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
osg::BoundingBox fluidDrawable::computeBoundingBox() const
#else
osg::BoundingBox fluidDrawable::computeBound() const
#endif
{
    return box;
}

osg::Object *fluidDrawable::cloneType() const
{
    return new fluidDrawable();
}

osg::Object *fluidDrawable::clone(const osg::CopyOp &op) const
{
    return new fluidDrawable(*this, op);
}

COVERPLUGIN(SPH)
