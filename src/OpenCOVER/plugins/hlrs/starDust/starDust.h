/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef starDust_H
#define starDust_H

#include <map>

#include <osg/Drawable>
#include <osg/Version>

#include <util/coTypes.h>
#include <cover/coVRPlugin.h>

#include <OpenVRUI/coMenu.h>

#include <cover/coVRTui.h>
#include <sysdep/opengl.h>
#include "renderParticles.h"
#include "bodysystem.h"

#define CUDA_DATATYPE double
#define CUDA_USE_DOUBLE true

struct State;
class dustDrawable;

namespace opencover
{
class coVRPlugin;
}

namespace vrui
{
class coMenuItem;
class coDoUnstructuredGrid;
class coRowMenu;
class coSliderMenuItem;
class coPotiToolboxItem;
class coMenuItem;
}

using namespace vrui;
using namespace opencover;

struct minmax
{
    float min, max;
};

struct particleData
{
    double xc, yc, zc, vx, vy, vz, m, f, s;
    particleData(double xc_, double yc_, double zc_, double vx_, double vy_, double vz_, double m_, double f_, double s_)
    {
        xc = xc_;
        yc = yc_;
        zc = zc_;
        vx = vx_;
        vy = vy_;
        vz = vz_;
        m = m_;
        f = f_;
        s = s_;
    };
};

struct planetData
{
    double xc, yc, zc, vx, vy, vz, m, s;
    planetData(double xc_, double yc_, double zc_, double vx_, double vy_, double vz_, double m_, double s_)
    {
        xc = xc_;
        yc = yc_;
        zc = zc_;
        vx = vx_;
        vy = vy_;
        vz = vz_;
        m = m_;
        s = s_;
    };
};

struct objectData
{
    double xc, yc, zc, vx, vy, vz, m, s;
    objectData(double xc_, double yc_, double zc_, double vx_, double vy_, double vz_, double m_, double s_)
    {
        xc = xc_;
        yc = yc_;
        zc = zc_;
        vx = vx_;
        vy = vy_;
        vz = vz_;
        m = m_;
        s = s_;
    };
};

class starDust : public coVRPlugin, public coTUIListener
{
public:
    starDust();
    ~starDust();

    virtual bool init();
    virtual void preDraw(osg::RenderInfo &);
    virtual void preFrame();
    virtual void postFrame();

    int loadData(std::string particlepath, osg::Group *parent);
    void unloadData(std::string particlepath);
    static int loadFile(const char *name, osg::Group *parent, const char *covise_key);
    static int unloadFile(const char *name, const char *covise_key);
    static starDust *instance()
    {
        return plugin;
    };
    coTUIToggleButton *tuiSimulate;
    coTUIToggleButton *tuiRender;
    coTUIButton *tuiReset;
    coTUILabel *tuiSimTime;
    coTUILabel *tuiPlanetScaleLabel;
    coTUIEditFloatField *tuiPlanetScale;
    coTUILabel *tuiIntegrationTimestepLabel;
    coTUIEditFloatField *tuiIntegrationTimestep;
    coTUILabel *tuiIterationsPerFrameLabel;
    coTUIEditFloatField *tuiIterationsPerFrame;
    coTUILabel *tuiSimulateToYearLabel;
    coTUIEditFloatField *tuiSimulateToYear;
    coTUILabel *tuiActivePlanetsLabel;
    coTUIEditIntField *tuiActivePlanets;
    coTUILabel *tuiParticleSizeLabel;
    coTUIEditFloatField *tuiParticleSize;
    virtual void tabletEvent(coTUIElement *tUIItem);
    double startTime;
    float particleSize;
    time_t SimulationTime;

    int deltaT;
    int numIterationsPerFrame;
    int numActivePlanets;
    std::vector<planetData> planets;
    std::vector<objectData> objects;

private:
    time_t simulateTo;
    bool initDone;
    coRowMenu *menu;
    coTUITab *tuiTab;
    std::map<std::string, osg::Geode *> geode;
    std::map<std::string, coPotiToolboxItem *> sliders;
    std::map<std::string, coTUIFloatSlider *> tuiSliders;
    std::map<std::string, coTUIToggleButton *> tuiButtons;
    std::map<std::string, osg::Group *> groups;
    osg::ref_ptr<dustDrawable> dD;

    std::map<std::string, struct minmax> minMax;
    std::vector<planetData> initialPlanets;
    std::vector<objectData> initialObjects;
    float planetScale;

    static starDust *plugin;
};

class dustDrawable : public osg::Drawable, public coMenuListener, public coTUIListener
{
public:
    //   dustDrawable(coSliderMenuItem *slider, coTUIFloatSlider *tui,
    dustDrawable();
    dustDrawable(const dustDrawable &draw, const osg::CopyOp &op = osg::CopyOp::SHALLOW_COPY);
    virtual ~dustDrawable();
    std::vector<particleData> particles;
    CUDA_DATATYPE *particleCoords;
    CUDA_DATATYPE *particleVelos;
    CUDA_DATATYPE *planetCoords;
    CUDA_DATATYPE *planetVelos;

    virtual void drawImplementation(osg::RenderInfo &info) const;
    virtual osg::Object *cloneType() const;
    virtual osg::Object *clone(const osg::CopyOp &op) const;
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

    void preFrame();
    void preDraw();
    void postFrame();
    void reset();

    virtual void menuEvent(coMenuItem *item);
    virtual void menuReleaseEvent(coMenuItem *item);
    virtual void tabletEvent(coTUIElement *tUIItem);
    CudaParticles<CUDA_DATATYPE> *cudaParticles;

private:
    State *state;
    bool doReset;
    RenderObject *geom;
    ParticleRenderer *renderer;

    bool changed;
    bool animate;
    int anim;
    float threshold;
    float min, max;
    int numParticles;

    //   coSliderMenuItem *slider;
    coPotiToolboxItem *slider;
    coTUIFloatSlider *tuiSlider;

    osg::BoundingBox box;
};

#endif
