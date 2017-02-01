/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CUCUTTINGSURFACE_H
#define CUCUTTINGSURFACE_H

#include <map>
#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#endif
#include <sysdep/opengl.h>
#include <osg/Drawable>
#include <osg/Geometry>
#include <osg/Version>
#include <cudpp.h>

#include <util/coTypes.h>
#include <cover/coVRPlugin.h>

#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <OpenVRUI/coMenu.h>

struct State;
class CuttingDrawable;

namespace opencover
{
class coVRPlugin;
class coColorBar;
}

namespace vrui
{
class coMenuItem;
class coRowMenu;
class coSubMenu;
class coMenuItem;
class coCheckboxMenuItem;
}

using namespace vrui;
using namespace opencover;

struct minmax
{
    float min, max;
};

/*
 * Author: Florian Niebling
 */
class cuCuttingSurface : public coVRPlugin
{
public:
    cuCuttingSurface();
    virtual ~cuCuttingSurface();
    coCheckboxMenuItem *getMenu(const RenderObject *container, const RenderObject *data,
                                const RenderObject *tex);
    void removeObject(const char *name, bool replace);
    void addObject(const RenderObject *container, osg::Group * /*setName*/, const RenderObject *, const RenderObject *, const RenderObject *, const RenderObject *);

    virtual bool init();
    virtual void preFrame();
    virtual void preDraw(osg::RenderInfo &);
    virtual void postFrame();
    virtual void message(int type, int len, const void *buf); // receive messages for Cuttinsurface updates from remote or the script plugin
    /*
   static void getMinMax(const float *data, int numElem, float &min,
                         float &max, float minV = -FLT_MAX, float maxV = FLT_MAX);
   static void removeSpikesAdaptive(const float *data, int numElem,
                                    float &min, float &max);
   static void countBins(const float *data, int numElem, float min, 
                         float max, int numBins, int *bins);
   */
private:
    bool initDone;
    coRowMenu *menu;
    std::map<std::string, osg::Geode *> geode;
    std::map<std::string, coVR3DTransRotInteractor *> interactors;
    std::map<std::string, osg::Group *> groups;
    std::map<std::string, coRowMenu *> menus;

    std::map<std::string, struct minmax> minMax;
};

class CuttingDrawable : public osg::Geometry, public coMenuListener
{
public:
    CuttingDrawable(coCheckboxMenuItem *menu,
                    coVR3DTransRotInteractor *interactor,
                    const RenderObject *geo, const RenderObject *map, const RenderObject *data,
                    float *bbox, float min, float max);

    CuttingDrawable(const CuttingDrawable &draw,
                    const osg::CopyOp &op = osg::CopyOp::SHALLOW_COPY);
    ~CuttingDrawable();

    virtual void drawImplementation(osg::RenderInfo &info) const;
    virtual osg::Object *cloneType() const;
    virtual osg::Object *clone(const osg::CopyOp &op) const;
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
    virtual osg::BoundingBox computeBoundingBox() const;
#else
    virtual osg::BoundingBox computeBound() const;
#endif

    void preDraw();
    void preFrame();
    void postFrame();

    virtual void menuEvent(coMenuItem *item);
    virtual void menuReleaseEvent(coMenuItem *item);
    void setMatrix(osg::Matrix &mat)
    {
        remoteMatrixChanged = true;
        remoteMatrix = mat;
    };

private:
    State *state;

    const RenderObject *geom;

    bool remoteMatrixChanged;
    osg::Matrix remoteMatrix;

    bool interactorChanged;
    float distance;

    coCheckboxMenuItem *menu;
    coVR3DTransRotInteractor *planeInteractor;

    osg::BoundingBox box;
    std::string name;
};

#endif
