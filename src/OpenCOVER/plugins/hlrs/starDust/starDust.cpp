/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//#include "cudaEngine.h"
#include "starDust.h"

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
#include "solarSystemNode.h"

starDust *starDust::plugin = NULL;

FileHandler fileHandler[] = {
    { NULL,
      starDust::loadFile,
      starDust::unloadFile,
      "starDust" }
};

int starDust::loadFile(const char *fn, osg::Group *parent, const char *)
{
    if (plugin)
        return plugin->loadData(fn, parent);

    return -1;
}

int starDust::unloadFile(const char *fn, const char *)
{
    if (plugin)
    {
        plugin->unloadData(fn);
        return 0;
    }

    return -1;
}

int starDust::loadData(std::string particlepath, osg::Group *parent)
{
    unloadData(particlepath);
    FILE *fp;
    fp = fopen(particlepath.c_str(), "r");
    if (fp)
    {
#define LINE_LEN 1000
        char buf[1001];
        buf[1000] = '\0';
        dD = new dustDrawable();
        while (!feof(fp))
        {
            fgets(buf, LINE_LEN, fp);
            if ((buf[0] != '%') && (buf[0] != '#'))
            {
                double xc, yc, zc, vx, vy, vz, mass, f, startTime;
                sscanf(buf, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", &xc, &yc, &zc, &vx, &vy, &vz, &mass, &f, &startTime);
                dD->particles.push_back(particleData(xc, yc, zc, vx, vy, vz, mass, f, startTime));
            }
        }
        dD->particleCoords = new CUDA_DATATYPE[dD->particles.size() * 4];
        dD->particleVelos = new CUDA_DATATYPE[dD->particles.size() * 3];
        dD->planetCoords = new CUDA_DATATYPE[starDust::instance()->planets.size() * 4];
        dD->planetVelos = new CUDA_DATATYPE[starDust::instance()->planets.size() * 3];
        int n = dD->particles.size();
        for (int i = 0; i < n; i++)
        {
            dD->particleCoords[i * 4 + 0] = dD->particles[i].xc;
            dD->particleCoords[i * 4 + 1] = dD->particles[i].yc;
            dD->particleCoords[i * 4 + 2] = dD->particles[i].zc;
            dD->particleCoords[i * 4 + 3] = dD->particles[i].m;
            dD->particleVelos[i * 3] = dD->particles[i].vx;
            dD->particleVelos[i * 3 + 1] = dD->particles[i].vy;
            dD->particleVelos[i * 3 + 2] = dD->particles[i].vz;
        }
        osg::Geode *g = new osg::Geode;
        g->setName("startDustGeo");
        geode[particlepath] = g;

        g->addDrawable(dD.get());
        dD->setUseDisplayList(false);
        if (parent)
            parent->addChild(g);
        else
            cover->getObjectsRoot()->addChild(g);

        geode[std::string(particlepath)] = g;
        initDone = true;
    }
    else
    {
        fprintf(stderr, "Could not open file %s\n", particlepath.c_str());
    }

    return 0;
}

void starDust::unloadData(std::string particlepath)
{
}

starDust::starDust()
: coVRPlugin(COVER_PLUGIN_NAME)
, initDone(false)
, menu(NULL)
{
    plugin = this;
    fprintf(stderr, "starDust::starDust()\n");
    tuiTab = new coTUITab("starDust", coVRTui::instance()->mainFolder->getID());
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

    tuiPlanetScaleLabel = new coTUILabel("PlanetScale:", tuiTab->getID());
    tuiPlanetScaleLabel->setPos(3, 0);

    tuiPlanetScale = new coTUIEditFloatField("PlanetScale:", tuiTab->getID());
    tuiPlanetScale->setPos(4, 0);
    tuiPlanetScale->setEventListener(this);
    tuiPlanetScale->setValue(1000000);
    planetScale = 1000000;

    deltaT = 900;
    tuiIntegrationTimestepLabel = new coTUILabel("IterationTimestep:", tuiTab->getID());
    tuiIntegrationTimestepLabel->setPos(3, 1);

    tuiIntegrationTimestep = new coTUIEditFloatField("IterationTimestep:", tuiTab->getID());
    tuiIntegrationTimestep->setPos(4, 1);
    tuiIntegrationTimestep->setEventListener(this);
    tuiIntegrationTimestep->setValue(deltaT);

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

    FILE *fp;
    fp = fopen("/data/irs/planets.txt", "r");
    if (fp)
    {
#define LINE_LEN 1000
        char buf[1001];
        buf[1000] = '\0';
        while (!feof(fp))
        {
            fgets(buf, LINE_LEN, fp);
            if ((buf[0] != '%') && (buf[0] != '#'))
            {
                double xc, yc, zc, vx, vy, vz, mass;
                char name[LINE_LEN];
                sscanf(buf, "%s %lf %lf %lf %lf %lf %lf %lf %lf", name, &xc, &yc, &zc, &vx, &vy, &vz, &mass, &startTime);
                planetData pd = planetData(xc, yc, zc, vx, vy, vz, mass, startTime);
                planets.push_back(pd);
                initialPlanets.push_back(pd);
                fprintf(stderr, "startTime %lf\n", startTime);
            }
        }
        SimulationTime = startTime;
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "Could not open file %s\n", "/data/irs/planets.txt");
    }

    fp = fopen("/data/irs/objects.txt", "r");
    if (fp)
    {
        char buf[1001];
        buf[1000] = '\0';
        while (!feof(fp))
        {
            fgets(buf, LINE_LEN, fp);
            if ((buf[0] != '%') && (buf[0] != '#'))
            {
                double xc, yc, zc, vx, vy, vz, mass, startT;
                char name[LINE_LEN];
                sscanf(buf, "%s %lf %lf %lf %lf %lf %lf %lf %lf", name, &xc, &yc, &zc, &vx, &vy, &vz, &mass, &startT);
                objectData od = objectData(xc, yc, zc, vx, vy, vz, mass, startT);
                objects.push_back(od);
                initialObjects.push_back(od);
            }
        }
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "Could not open file %s\n", "/data/irs/objects.txt");
    }

    VrmlNamespace::addBuiltIn(VrmlNodeSolarSystem::defineType());
}

starDust::~starDust()
{
}

void starDust::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiReset)
    {

        for (int i = 0; i < initialPlanets.size(); i++)
        {
            planets[i] = initialPlanets[i];
        }
        for (int i = 0; i < initialObjects.size(); i++)
        {
            objects[i] = initialObjects[i];
        }
        std::map<std::string, osg::Geode *>::iterator i;
        for (i = geode.begin(); i != geode.end(); i++)
        {
            dustDrawable *drawable = dynamic_cast<dustDrawable *>(i->second->getDrawable(0));
            if (drawable)
                drawable->reset();
        }

        SimulationTime = startTime;
    }
    else if (tUIItem == tuiPlanetScale)
    {
        planetScale = tuiPlanetScale->getValue();
    }
    else if (tUIItem == tuiIterationsPerFrame)
    {
        numIterationsPerFrame = tuiIterationsPerFrame->getValue();
    }
    else if (tUIItem == tuiIntegrationTimestep)
    {
        deltaT = tuiIntegrationTimestep->getValue();
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

bool starDust::init()
{
    fprintf(stderr, "starDust::init\n");
    coVRFileManager::instance()->registerFileHandler(&fileHandler[0]);
    return true;
}

void starDust::preDraw(osg::RenderInfo &)
{

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        dustDrawable *drawable = dynamic_cast<dustDrawable *>(i->second->getDrawable(0));
        if (drawable && drawable->cudaParticles != NULL)
        {
            for (int n = 0; n < planets.size(); n++)
            {
                drawable->planetCoords[n * 4] = planets[n].xc;
                drawable->planetCoords[n * 4 + 1] = planets[n].yc;
                drawable->planetCoords[n * 4 + 2] = planets[n].zc;
                drawable->planetCoords[n * 4 + 3] = planets[n].m;
                drawable->planetVelos[n * 3] = planets[n].vx;
                drawable->planetVelos[n * 3 + 1] = planets[n].vx;
                drawable->planetVelos[n * 3 + 2] = planets[n].vz;
            }
            drawable->cudaParticles->copyPlanetData(drawable->planetCoords, drawable->planetVelos);
        }
        if (drawable)
        {
            drawable->preDraw();
        }
    }
}

void starDust::preFrame()
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

    int years = SimulationTime / (60 * 60 * 24 * 365);
    time_t simTime = SimulationTime - years * 60 * 60 * 24 * 365;
    years = 2000 + years;
    int days = simTime / (60 * 60 * 24);
    simTime = simTime - days * (60 * 60 * 24);
    int hours = simTime / (60 * 60);
    simTime = simTime - hours * (60 * 60);
    int minutes = simTime / 60;
    static int oldYears = 0;
    if (oldYears != years)
    {
        oldYears = years;
        fprintf(stderr, "SimTime: %d\n", years);
        char label[1000];
        sprintf(label, "Year: %d", years);
        starDust::instance()->tuiSimTime->setLabel(label);
    }

    if (tuiSimulate->getState())
    {

        double ms = 1.98892E30;
        double G = 6.67384E-11;
        double G_km = G * 0.000000001;

        for (int i = 0; i < numIterationsPerFrame; i++)
        {
            for (int p = 0; p < planets.size(); p++)
            {
                for (int pt = 0; pt < planets.size(); pt++) // to other planets
                {
                    if (pt != p)
                    {
                        double dx = planets[pt].xc - planets[p].xc;
                        double dy = planets[pt].yc - planets[p].yc;
                        double dz = planets[pt].zc - planets[p].zc;

                        double distSqr = dx * dx + dy * dy + dz * dz;
                        double len = sqrt(distSqr);
                        double factor = G_km * (planets[pt].m) / len / distSqr;

                        double ax = dx * factor;
                        double ay = dy * factor;
                        double az = dz * factor;
                        planets[p].vx += ax * deltaT;
                        planets[p].vy += ay * deltaT;
                        planets[p].vz += az * deltaT;
                    }
                }
                { // to sun
                    double dx = planets[p].xc;
                    double dy = planets[p].yc;
                    double dz = planets[p].zc;

                    double distSqr = dx * dx + dy * dy + dz * dz;
                    double len = sqrt(distSqr);
                    double factor = -G_km * (ms) / len / distSqr;

                    double ax = dx * factor;
                    double ay = dy * factor;
                    double az = dz * factor;
                    planets[p].vx += ax * deltaT;
                    planets[p].vy += ay * deltaT;
                    planets[p].vz += az * deltaT;
                }
            }
            for (int p = 0; p < planets.size(); p++)
            {
                planets[p].xc += planets[p].vx * deltaT;
                planets[p].yc += planets[p].vy * deltaT;
                planets[p].zc += planets[p].vz * deltaT;
            }

            for (int p = 0; p < objects.size(); p++)
            {
                if (SimulationTime > objects[p].s)
                {
                    for (int pt = 0; pt < planets.size(); pt++) // to other planets
                    {
                        double dx = planets[pt].xc - objects[p].xc;
                        double dy = planets[pt].yc - objects[p].yc;
                        double dz = planets[pt].zc - objects[p].zc;

                        double distSqr = dx * dx + dy * dy + dz * dz;
                        double len = sqrt(distSqr);
                        double factor = G_km * (planets[pt].m) / len / distSqr;

                        double ax = dx * factor;
                        double ay = dy * factor;
                        double az = dz * factor;
                        objects[p].vx += ax * deltaT;
                        objects[p].vy += ay * deltaT;
                        objects[p].vz += az * deltaT;
                    }
                    { // to sun
                        double dx = objects[p].xc;
                        double dy = objects[p].yc;
                        double dz = objects[p].zc;

                        double distSqr = dx * dx + dy * dy + dz * dz;
                        double len = sqrt(distSqr);
                        double factor = -G_km * (ms) / len / distSqr;

                        double ax = dx * factor;
                        double ay = dy * factor;
                        double az = dz * factor;
                        objects[p].vx += ax * deltaT;
                        objects[p].vy += ay * deltaT;
                        objects[p].vz += az * deltaT;
                    }
                }
            }
            for (int p = 0; p < objects.size(); p++)
            {
                if (SimulationTime > objects[p].s)
                {
                    objects[p].xc += objects[p].vx * deltaT;
                    objects[p].yc += objects[p].vy * deltaT;
                    objects[p].zc += objects[p].vz * deltaT;
                }
            }
        }
        std::map<std::string, osg::Geode *>::iterator di;
        for (di = geode.begin(); di != geode.end(); di++)
        {
            dustDrawable *drawable = dynamic_cast<dustDrawable *>(di->second->getDrawable(0));
            if (drawable)
            {
                drawable->preFrame();
            }
        }

        VrmlNodeSolarSystem *vn = VrmlNodeSolarSystem::instance();
        if (vn)
        {
            vn->setJupiterPosition(planets[0].xc, planets[0].yc, planets[0].zc);
            vn->setSaturnPosition(planets[1].xc, planets[1].yc, planets[1].zc);
            vn->setEarthPosition(planets[2].xc, planets[2].yc, planets[2].zc);
            vn->setMarsPosition(planets[3].xc, planets[3].yc, planets[3].zc);
            vn->setVenusPosition(planets[4].xc, planets[4].yc, planets[4].zc);
            vn->setComet_CG_Position(objects[0].xc, objects[0].yc, objects[0].zc);
            vn->setRosettaPosition(objects[1].xc, objects[1].yc, objects[1].zc);
            vn->setPlanetScale(planetScale);
        }
    }
}

void starDust::postFrame()
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
    if (tuiSimulate->getState())
    {
        SimulationTime += deltaT * numIterationsPerFrame; // +=10 Tage (default)

        if (SimulationTime > simulateTo)
            tuiSimulate->setState(false);
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
    cudaParticles = NULL;
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
    cudaParticles->setNumActiveParticles(0);
}

void dustDrawable::preDraw()
{
    numParticles = particles.size();

    if (doReset)
    {
        doReset = false;
        cudaParticles->copyInitialData(particleCoords, particleVelos);
    }
    if (renderer == NULL && numParticles > 0)
    {

        renderer = new ParticleRenderer();
        cudaParticles = new CudaParticles<CUDA_DATATYPE>(numParticles);
        cudaParticles->setInitialData(particleCoords, particleVelos);

        for (int n = 0; n < starDust::instance()->planets.size(); n++)
        {
            planetCoords[n * 4] = starDust::instance()->planets[n].xc;
            planetCoords[n * 4 + 1] = starDust::instance()->planets[n].yc;
            planetCoords[n * 4 + 2] = starDust::instance()->planets[n].zc;
            planetCoords[n * 4 + 3] = starDust::instance()->planets[n].m;
            planetVelos[n * 3] = starDust::instance()->planets[n].vx;
            planetVelos[n * 3 + 1] = starDust::instance()->planets[n].vx;
            planetVelos[n * 3 + 2] = starDust::instance()->planets[n].vz;
        }
        cudaParticles->setInitialPlanetData(planetCoords, planetVelos);

        //renderer->setPositions(particleCoords, particles.size());
        renderer->setPBO(cudaParticles->getPosVBO(), particles.size(), CUDA_USE_DOUBLE);
        float color[4];
        color[0] = 1;
        color[1] = 1;
        color[2] = 1;
        color[3] = 1;
        renderer->setBaseColor(color);
        for (int i = 0; i < numParticles; i++)
        {
            box.expandBy(osg::Vec3(particles[i].xc * 2, particles[i].yc * 2, particles[i].zc * 2));
            box.expandBy(osg::Vec3(-particles[i].xc * 2, -particles[i].yc * 2, -particles[i].zc * 2));
        }
        dirtyBound();
    }
    if (starDust::instance()->tuiSimulate->getState())
    {
        int deltaT = 900;
        int numIterationsPerFrame = 960;
        time_t sTime = starDust::instance()->SimulationTime;
        if (cudaParticles)
        {
            if (cudaParticles->getNumActiveParticles() < numParticles)
            {
                int i;
                for (i = cudaParticles->getNumActiveParticles(); i < numParticles; i++)
                {
                    if (particles[i].s > sTime)
                        break;
                }
                cudaParticles->setNumActiveParticles(i);
                static int oldna = 0;
                if (oldna != cudaParticles->getNumActiveParticles())
                {
                    fprintf(stderr, "numActiveParticles = %d\n", cudaParticles->getNumActiveParticles());
                    oldna = cudaParticles->getNumActiveParticles();
                }
            }
            cudaParticles->integrate(starDust::instance()->deltaT, starDust::instance()->numIterationsPerFrame, starDust::instance()->numActivePlanets); //1/4 Stunde
        }
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
		*/
        if (starDust::instance()->tuiRender->getState())
        {
            renderer->setSpriteSize(starDust::instance()->particleSize);
            renderer->display(ParticleRenderer::PARTICLE_SPRITES, cudaParticles->getNumActiveParticles());
            //renderer->display(ParticleRenderer::PARTICLE_POINTS);
        }

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

COVERPLUGIN(starDust)
