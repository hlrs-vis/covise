/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CUTRACER_H
#define CUTRACER_H

#include <map>

#include <GL/gl.h>
#include <osg/Drawable>
#include <osg/Geometry>

#include <util/coTypes.h>
#include <cover/coVRPlugin.h>

#include <cover/coVR3DTransInteractor.h>
#include <OpenVRUI/coMenu.h>

class TracerDrawable;

namespace opencover
{
class coVRPlugin;
class coColorBar;
}

namespace covise
{
class coMenuItem;
class coDoUnstructuredGrid;
class coRowMenu;
class coSubMenu;
class coMenuItem;
class coCheckboxMenuItem;
}

using namespace covise;
using namespace opencover;

struct minmax
{
    float min, max;
};

/*
 * Author: Florian Niebling
 */
class cuTracer : public coVRPlugin
{
public:
    cuTracer();
    virtual ~cuTracer();
    coCheckboxMenuItem *getMenu(RenderObject *container, RenderObject *data,
                                RenderObject *tex);
    void removeObject(const char *name, bool replace);
    void addObject(RenderObject *container,
                   RenderObject *geometry, RenderObject * /*normObj*/,
                   RenderObject *colorObj, RenderObject * /*texObj*/,
                   osg::Group * /*setName*/, int /*numCol*/,
                   int, int /*colorPacking*/, float *, float *, float *, int *,
                   int, int, float *, float *, float *, float);

    virtual bool init();
    virtual void preFrame();
    virtual void postFrame();

private:
    coRowMenu *menu;
    std::map<std::string, osg::Geode *> geode;
    std::map<std::string, coVR3DTransInteractor *> interactors;
    std::map<std::string, osg::Group *> groups;
    std::map<std::string, coRowMenu *> menus;

    std::map<std::string, struct minmax> minMax;
};

class TracerDrawable : public osg::Geometry
{
public:
    TracerDrawable(RenderObject *geo, RenderObject *vel, coVR3DTransInteractor *inter);
    TracerDrawable(const TracerDrawable &draw, const osg::CopyOp &op = osg::CopyOp::SHALLOW_COPY);
    ~TracerDrawable();

    virtual void drawImplementation(osg::RenderInfo &info) const;
    virtual osg::Object *cloneType() const;
    virtual osg::Object *clone(const osg::CopyOp &op) const;
    virtual osg::BoundingBox computeBound();

    void preFrame();
    void postFrame();

private:
    RenderObject *geom;

    bool interactorChanged;
    float distance;

    coCheckboxMenuItem *menu;
    coVR3DTransInteractor *interactor;

    osg::BoundingBox box;

    unsigned char *texData;
    mutable GLuint texture;
    mutable bool tex;

    std::string name;

    struct usg cuda_usg;
    int numParticles, numSteps;

    mutable int step, stepsComputed;
    mutable bool initialized;

    float3 *startPos;
};

#endif
