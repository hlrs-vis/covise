/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: PSO Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                OSGPSOParticleOnResponseSurface OSGPSOParticleOnResponseSurface                                                         **
\****************************************************************************/

#include "PSOPlugin.h"
#include "OSGPSOParticleOnResponseSurface.h"
#include "OSGResponseSurface.h"
#include "PSOPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>

double PSOPlugin::quadratic(double *x)
{
    return (x[0] * x[0] + x[1] * x[1]);
}

double PSOPlugin::noisy(double *x)
{
    return (x[0] * x[0]) - 100.0 * cos(x[0]) * cos(x[0]) - 100.0 * cos(x[0] * x[0] / 30.0) + x[1] * x[1] - 100.0 * cos(x[1]) * cos(x[1]) - 100.0 * cos(x[1] * x[1] / 30.0) + 1400.0;
}

PSOPlugin::PSOPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "PSOPlugin::PSOPlugin\n");
    oldTime = 0.0;

    isSetup = false;
    isRunning = false;
}

bool PSOPlugin::init()
{
    std::cerr << "Particle Swarm Optimization Plugin" << std::endl;
    PSOTab = new coTUITab("PSO", coVRTui::instance()->mainFolder->getID());
    PSOTab->setPos(0, 0);

    startButton = new coTUIButton("Start Optimization", PSOTab->getID());
    startButton->setEventListener(this);
    startButton->setPos(0, 0);
    stopButton = new coTUIButton("Stop Optimization", PSOTab->getID());
    stopButton->setEventListener(this);
    stopButton->setPos(0, 1);
    resetButton = new coTUIButton("Reset Optimization", PSOTab->getID());
    resetButton->setEventListener(this);
    resetButton->setPos(0, 2);

    functionComboBox = new coTUIComboBox("Response Function", PSOTab->getID());
    functionComboBox->setEventListener(this);
    functionComboBox->setPos(1, 0);
    functionComboBox->addEntry("2D Input Space Quadratic function");
    functionComboBox->addEntry("2D Input Space Griewank function");
    functionComboBox->addEntry("3D Input Space Rosenbrock function");
    functionComboBox->setSelectedEntry(0);

    inertiaFunctionToggleButton = new coTUIToggleButton("Inertia update", PSOTab->getID());
    inertiaFunctionToggleButton->setEventListener(this);
    inertiaFunctionToggleButton->setPos(2, 0);
    crazyFunctionToggleButton = new coTUIToggleButton("Craziness operator", PSOTab->getID());
    crazyFunctionToggleButton->setEventListener(this);
    crazyFunctionToggleButton->setPos(2, 1);
    odsFunctionToggleButton = new coTUIToggleButton("One dimensional search", PSOTab->getID());
    odsFunctionToggleButton->setEventListener(this);
    odsFunctionToggleButton->setPos(2, 2);

    initPSO();

    return true;
}

void PSOPlugin::initPSO()
{
    if (!isSetup)
    {
        //int nvar = pso::rosenbrock3_nvar;
        //int niter = 120;
        applyInertiaUpdate = inertiaFunctionToggleButton->getState();
        applyCrazinessOperator = crazyFunctionToggleButton->getState();
        applyODS = odsFunctionToggleButton->getState();

        //	surfaceGeode = new OSGResponseSurface(noisy, lbound, ubound, integer, 0.1);
        if (functionComboBox->getSelectedEntry() == 0)
        {
            npar = 5;
            nvar = 2;
            double lbound[2] = { -1, -1 };
            double ubound[2] = { 1, 1 };
            bool integer[2] = { false, false };

            OSGPSOParticleOnResponseSurface::init(quadratic, lbound, ubound, integer, (long int)(1e6 * cover->frameTime()));
            surfaceGeode = new OSGResponseSurface(quadratic, lbound, ubound, integer, 0.01);
			surfaceGeode->setName("ResponseSurface");
        }
        else if (functionComboBox->getSelectedEntry() == 1)
        {
            npar = 20;
            nvar = 2;
            double lbound[2] = { -100, -100 };
            double ubound[2] = { 100, 100 };
            bool integer[2] = { false, false };

            surfaceGeode = new OSGResponseSurface(pso::griewank2, lbound, ubound, integer, 0.5);
			surfaceGeode->setName("ResponseSurface");
            OSGPSOParticleOnResponseSurface::init(pso::griewank2, lbound, ubound, integer, (long int)(1e6 * cover->frameTime()));
        }
        else if (functionComboBox->getSelectedEntry() == 2)
        {
            npar = 20;
            nvar = 3;
            double lbound[3] = { -100, -100, -100 };
            double ubound[3] = { 100, 100, 100 };
            bool integer[3] = { false, false, false };

            OSGPSOParticle::init(pso::rosenbrock3, lbound, ubound, integer, (long int)(1e6 * cover->frameTime()));
        }

        else
        {
            npar = 20;
            nvar = 2;
            double lbound[2] = { -1, -1 };
            double ubound[2] = { 1, 1 };
            bool integer[2] = { false, false };

            OSGPSOParticleOnResponseSurface::init(quadratic, lbound, ubound, integer, (long int)(1e6 * cover->frameTime()));
            surfaceGeode = new OSGResponseSurface(quadratic, lbound, ubound, integer, 0.01);
			surfaceGeode->setName("ResponseSurface");
        }

        if (nvar == 2)
        {
            cover->getObjectsRoot()->addChild(surfaceGeode);

            par2D = new OSGPSOParticleOnResponseSurface *[npar];
            for (int i = 0; i < npar; ++i)
            {
                par2D[i] = new OSGPSOParticleOnResponseSurface;
				char name[100];
				sprintf(name, "p%d", i);
				par2D[i]->setName(name);
                cover->getObjectsRoot()->addChild(par2D[i]);
            }

            OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::computePath);
            OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::move);
            for (int i = 0; i < npar; ++i)
            {
                osg::Vec3 pos = (cover->getInvBaseMat()) * (par2D[i]->getPosition());
                double scale = cover->getInteractorScale(pos);
                par2D[i]->setScale(osg::Vec3(scale, scale, scale));
            }
        }
        else if (nvar == 3)
        {
            par3D = new OSGPSOParticle *[npar];
            for (int i = 0; i < npar; ++i)
            {
                par3D[i] = new OSGPSOParticle;
				char name[100];
				sprintf(name, "p%d", i);
				par3D[i]->setName(name);
                cover->getObjectsRoot()->addChild(par3D[i]);
            }

            OSGPSOParticle::all(&OSGPSOParticle::computePath);
            OSGPSOParticle::all(&OSGPSOParticle::move);
            for (int i = 0; i < npar; ++i)
            {
                osg::Vec3 pos = (cover->getInvBaseMat()) * (par3D[i]->getPosition());
                double scale = cover->getInteractorScale(pos);
                par3D[i]->setScale(osg::Vec3(scale, scale, scale));
            }
        }

        isSetup = true;
    }
}
void PSOPlugin::destroyPSO()
{
    if (isSetup)
    {
        isRunning = false;

        if (nvar == 2)
        {
            cover->getObjectsRoot()->removeChild(surfaceGeode);
            for (int i = 0; i < npar; ++i)
                cover->getObjectsRoot()->removeChild(par2D[i]);
            delete[] par2D;

            OSGPSOParticleOnResponseSurface::destroy();
        }
        else if (nvar == 3)
        {
            for (int i = 0; i < npar; ++i)
                cover->getObjectsRoot()->removeChild(par3D[i]);
            delete[] par3D;

            OSGPSOParticle::destroy();
        }

        isSetup = false;
    }
}

// this is called if the plugin is removed at runtime
PSOPlugin::~PSOPlugin()
{
    fprintf(stderr, "PSOPlugin::~PSOPlugin\n");
    destroyPSO();
}

void
PSOPlugin::preFrame()
{

	applyInertiaUpdate = inertiaFunctionToggleButton->getState();
	applyCrazinessOperator = crazyFunctionToggleButton->getState();
	applyODS = odsFunctionToggleButton->getState();
    if (isSetup && isRunning)
    {
        if ((cover->frameTime() > oldTime + 1.0))
        {
            if (nvar == 2)
            {
                PSOCycle2D();
                OSGPSOParticleOnResponseSurface::t = 0;
            }
            else if (nvar == 3)
            {
                PSOCycle3D();
                OSGPSOParticle::t = 0;
            }

            oldTime = cover->frameTime();
        }

        if (nvar == 2)
        {
            OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::move);
            OSGPSOParticleOnResponseSurface::t += cover->frameDuration();
        }
        else if (nvar == 3)
        {
            OSGPSOParticle::all(&OSGPSOParticle::move);
            OSGPSOParticle::t += cover->frameDuration();
        }
    }
    for (int i = 0; i < npar; ++i)
    {
        if (nvar == 2)
        {
            osg::Vec3 pos = (cover->getInvBaseMat()) * (par2D[i]->getPosition());
            double scale = cover->getInteractorScale(pos);
            par2D[i]->setScale(osg::Vec3(scale, scale, scale));
        }
        else if (nvar == 3)
        {
            osg::Vec3 pos = (cover->getInvBaseMat()) * (par3D[i]->getPosition());
            double scale = cover->getInteractorScale(pos);
            par3D[i]->setScale(osg::Vec3(scale, scale, scale));
        }
    }
}

void PSOPlugin::PSOCycle2D()
{
    // On every particle: add evaluation job at particle's present position
    OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::updateVal);

    // On every particle: Update particle's own best value as well as the global best value
    OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::updateBestVal);

    // On every particle: Update the particle velocity
    OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::updateVelocity);

    if (applyODS)
    {
        // On every particle: determine two additional to points to evaluate
        OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::updateBeta);

        // On every particle: add evaluation job at the two additional points of particle
        OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::updateValBeta);

        // On every particle: Approximate response curve along velocoity vector, determine minimum of it,
        //						apply determined minimum to velocity vector (scale it)
        OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::approximateV);
    }

    // On every particle: Update the particle position
    OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::updatePosition);

    // Status output
    //std::cerr << "Iterasie " << j << ":" << "\t";
    //std::cerr << "Beste waarde: " << OSGPSOParticleOnResponseSurface::getGBestVal() << "\t";
    //std::cerr << "Beste posisie: ";
    //for(int i=0; i<nvar; ++i)
    //	std::cerr << bestx[i] << "\t";
    //std::cerr << std::endl;

    OSGPSOParticleOnResponseSurface::all(&OSGPSOParticleOnResponseSurface::computePath);

    if (applyInertiaUpdate)
    {
        // Update inertia of particles
        OSGPSOParticleOnResponseSurface::updateInertia();
    }

    if (applyCrazinessOperator)
    {
        // Apply craziness operator
        OSGPSOParticleOnResponseSurface::goCrazy();
    }
}

void PSOPlugin::PSOCycle3D()
{
    // On every particle: add evaluation job at particle's present position
    OSGPSOParticle::all(&OSGPSOParticle::updateVal);

    // On every particle: Update particle's own best value as well as the global best value
    OSGPSOParticle::all(&OSGPSOParticle::updateBestVal);

    // On every particle: Update the particle velocity
    OSGPSOParticle::all(&OSGPSOParticle::updateVelocity);

    if (applyODS)
    {
        // On every particle: determine two additional to points to evaluate
        OSGPSOParticle::all(&OSGPSOParticle::updateBeta);

        // On every particle: add evaluation job at the two additional points of particle
        OSGPSOParticle::all(&OSGPSOParticle::updateValBeta);

        // On every particle: Approximate response curve along velocoity vector, determine minimum of it,
        //						apply determined minimum to velocity vector (scale it)
        OSGPSOParticle::all(&OSGPSOParticle::approximateV);
    }

    // On every particle: Update the particle position
    OSGPSOParticle::all(&OSGPSOParticle::updatePosition);

    // Status output
    //std::cerr << "Iterasie " << j << ":" << "\t";
    //std::cerr << "Beste waarde: " << OSGPSOParticle::getGBestVal() << "\t";
    //std::cerr << "Beste posisie: ";
    //for(int i=0; i<nvar; ++i)
    //	std::cerr << bestx[i] << "\t";
    //std::cerr << std::endl;

    OSGPSOParticle::all(&OSGPSOParticle::computePath);

    if (applyInertiaUpdate)
    {
        // Update inertia of particles
        OSGPSOParticle::updateInertia();
    }

    if (applyCrazinessOperator)
    {
        // Apply craziness operator
        OSGPSOParticle::goCrazy();
    }
}

void PSOPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == startButton)
    {
        std::cerr << "Starting PSO optimization..." << std::endl;
        isRunning = true;
    }

    if (tUIItem == stopButton)
    {
        std::cerr << "Stopping PSO optimization..." << std::endl;
        isRunning = false;
    }

    if (tUIItem == resetButton)
    {
        std::cerr << "Reseting PSO optimization..." << std::endl;
        destroyPSO();
        initPSO();
    }
}

COVERPLUGIN(PSOPlugin)
