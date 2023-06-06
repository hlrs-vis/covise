/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "cudaEngine.h"
#include "cuIsoSurface.h"

#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

#include <PluginUtil/ColorBar.h>

#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>

#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Material>
#include <osg/TexEnv>
#include <osg/Texture1D>

#include <sysdep/opengl.h>

#include <cover/coVRMSController.h>
#include <cover/coVRShader.h>

using namespace covise;
using vrui::coInteraction;

CUDAEngine isoEngine;

void RenderCUDAState(State *);

void getMinMax(const float *data, int numElem, float *min,
               float *max, float minV = -FLT_MAX, float maxV = FLT_MAX);

void removeSpikesAdaptive(const float *data, int numElem,
                          float *min, float *max);

cuIsoSurface::cuIsoSurface()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("cuIsoSurface", cover->ui)
, initDone(false)
, menu(NULL)
{
    tuiTab = new coTUITab("cuIsoSurface", coVRTui::instance()->mainFolder->getID());
    tuiTab->setPos(0, 0);
}

cuIsoSurface::~cuIsoSurface()
{
}

bool cuIsoSurface::init()
{
    return true;
}

void cuIsoSurface::removeObject(const char *objName, bool /*replace*/)
{
    minMax.erase(objName);

    std::map<std::string, osg::Group *>::iterator i = groups.find(std::string(objName));
    if (i != groups.end())
    {
        for (unsigned int index = 0; index < i->second->getNumChildren(); index++)
        {
            std::string name = i->second->getChild(index)->getName();
            geode.erase(name);
        }
        // TODO: delete group contents + drawable
        cover->getObjectsRoot()->removeChild(i->second);
        groups.erase(i);
    }

    std::map<std::string, ui::Slider *>::iterator si = sliders.find(objName);

    if (si != sliders.end())
        delete si->second;
}

void cuIsoSurface::addObject(const RenderObject *container, osg::Group * /*setName*/, const RenderObject *geometry, const RenderObject *normals, const RenderObject *colorObj, const RenderObject *texObj)
{
    osg::Group *group = NULL;

    std::map<std::string, osg::Group *>::iterator gi = groups.find(container->getName());
    if (gi == groups.end())
    {
        group = new osg::Group();
        group->setName(container->getName());
        cover->getObjectsRoot()->addChild(group);
        groups[container->getName()] = group;
    }
    else
        group = gi->second;

    if (!menu)
    {
        menu = new ui::Menu("cuIsoSurfaceUSG", this);
        cover->visMenu->add(menu);
    }

    if (container)
    {
        // get min and max from COLORMAP parameter of the container object
        RenderObject *tex = container->getTexture();
        if (tex && tex->isSet())
        {
            const char *attr = tex->getAttribute("COLORMAP");
            if (attr)
            {
                std::string species;
                std::vector<float> r, g, b, a;
                int numColors;
                float min, max;
                ColorBar::parseAttrib(attr, species, min, max,
                                      numColors, r, g, b, a);
                struct minmax m = { min, max };
                minMax[container->getName()] = m;
            }
        }
    }

    if (geometry && geometry->isUnstructuredGrid())
    {

        if (colorObj)
        {
            const float *red = colorObj->getFloat(Field::Red);
            const float *green = colorObj->getFloat(Field::Green);
            const float *blue = colorObj->getFloat(Field::Blue);
            const int *pc = colorObj->getInt(Field::RGBA);

            if (red && (!pc && !green && !blue))
            {
                float box[6];
                geometry->getMinMax(box[0], box[1], box[2], box[3], box[4], box[5]);

                float min = 0.0, max = 1.0;
                float dataMin = 0.0, dataMax = 1.0;
                getMinMax(red, colorObj->getNumElements(), &dataMin, &dataMax);
                removeSpikesAdaptive(red, colorObj->getNumElements(), &dataMin, &dataMax);

                std::map<std::string, struct minmax>::iterator mi = minMax.find(container->getName());
                if (mi != minMax.end())
                {
                    min = mi->second.min;
                    max = mi->second.max;
                }
                else
                {
                    const char *attrMin = container->getAttribute("MIN");
                    const char *attrMax = container->getAttribute("MAX");
                    if (attrMin && attrMax)
                    {
                        min = atof(attrMin);
                        max = atof(attrMax);
                    }
                }
                //fprintf("stderr, Iso minmax: %f %f\n", min, max);

                osg::ref_ptr<osg::StateSet> state = new osg::StateSet();
                state->setGlobalDefaults();

                if (texObj)
                {

                    osg::Texture1D *colorTex = new osg::Texture1D();
                    colorTex->setDataVariance(osg::Object::DYNAMIC);

                    int wx, wy, wz;
                    texObj->getSize(wx, wy, wz);

                    osg::Image *texImage = new osg::Image();
                    texImage->allocateImage(wx, wy, 1, GL_RGB, GL_UNSIGNED_BYTE);
                    // copy texture
                    unsigned char *it = new unsigned char[wx * wy * wz];
                    memcpy(it, texObj->getByte(Field::Texture), wx * wy * wz);

                    texImage->setImage(wx, wy, 1, 4, GL_RGBA, GL_UNSIGNED_BYTE, it, osg::Image::USE_NEW_DELETE);

                    colorTex->setImage(texImage);
                    colorTex->setFilter(osg::Texture1D::MIN_FILTER,
                                        osg::Texture1D::LINEAR);
                    colorTex->setFilter(osg::Texture1D::MAG_FILTER,
                                        osg::Texture1D::LINEAR);

                    state->setTextureAttributeAndModes(0, colorTex,
                                                       osg::StateAttribute::ON);

                    state->setTextureMode(0, GL_TEXTURE_1D,
                                          osg::StateAttribute::ON);
                }
                osg::ref_ptr<osg::Geode> g = new osg::Geode();

                state->setTextureMode(0, GL_TEXTURE_1D, osg::StateAttribute::ON);
                state->setMode(GL_BLEND, osg::StateAttribute::OFF);
                state->setMode(GL_LIGHTING, osg::StateAttribute::ON);
                state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
                state->setRenderBinDetails(10, "RenderBin");
                state->setNestRenderBins(false);
                /*
            osg::Material *mtl = new osg::Material;
            mtl->setColorMode(osg::Material::OFF);
            mtl->setAmbient(osg::Material::FRONT_AND_BACK,
                            osg::Vec4(1.0, 0.0, 0.0, 0.5));
            mtl->setDiffuse(osg::Material::FRONT_AND_BACK,
                            osg::Vec4(1.0, 0.0, 0.0, 0.5));
            mtl->setSpecular(osg::Material::FRONT_AND_BACK,
                             osg::Vec4(0.8, 0.8, 0.8, 0.5));
            mtl->setEmission(osg::Material::FRONT_AND_BACK,
                             osg::Vec4(0.0, 0.0, 0.0, 0.5));
            mtl->setAlpha(osg::Material::FRONT_AND_BACK, 1.0);
            state->setAttributeAndModes(mtl, osg::StateAttribute::ON);
            osg::TexEnv *env = new osg::TexEnv();
            env->setMode(osg::TexEnv::REPLACE);
            state->setTextureAttributeAndModes(0, env, osg::StateAttribute::ON);
            */
                g->setStateSet(state.get());

                ui::Slider *slider = NULL;
                coTUIFloatSlider *tuiSlider = NULL;
                coTUIToggleButton *tuiButton = NULL;

                std::map<std::string, ui::Slider *>::iterator i = sliders.find(container->getName());
                std::map<std::string, coTUIFloatSlider *>::iterator ti = tuiSliders.find(container->getName());
                std::map<std::string, coTUIToggleButton *>::iterator tbi = tuiButtons.find(container->getName());

                if (i == sliders.end())
                {
                    const char *name = colorObj->getAttribute("LABEL");
                    if (!name)
                        name = container->getName();

                    slider = new ui::Slider(menu, name);
                    slider->setBounds(dataMin, dataMax);
                    slider->setValue(dataMin);
                    sliders[container->getName()] = slider;
                }
                else
                    slider = i->second;

                if (ti == tuiSliders.end())
                {
                    const char *name = colorObj->getAttribute("LABEL");
                    if (!name)
                        name = container->getName();

                    tuiSlider = new coTUIFloatSlider(name, tuiTab->getID());
                    tuiSlider->setMin(min);
                    tuiSlider->setMax(max);
                    tuiSlider->setValue(min);
                    tuiSlider->setPos(0, tuiSliders.size());
                    tuiSliders[container->getName()] = tuiSlider;
                }
                else
                    tuiSlider = ti->second;

                if (tbi == tuiButtons.end())
                {
                    const char *name = colorObj->getAttribute("LABEL");
                    if (!name)
                        name = container->getName();

                    tuiButton = new coTUIToggleButton("animate", tuiTab->getID());
                    tuiButton->setState(false);
                    tuiButton->setPos(1, tuiButtons.size());
                    tuiButtons[container->getName()] = tuiButton;
                }
                else
                    tuiSlider = ti->second;

                osg::ref_ptr<IsoDrawable> draw = new IsoDrawable(slider, tuiSlider, tuiButton,
                                                                 geometry, normals, colorObj, box, min, max);

                opencover::coVRShader *shader = coVRShaderList::instance()->get("texture1d");
                if (shader)
                    shader->apply(g);

                g->addDrawable(draw.get());
                draw->setUseDisplayList(false);
                group->addChild(g.get());
                char name[256];
                snprintf(name, 256, "%s_%s", container->getName(),
                         geometry->getName());
                geode[std::string(name)] = g.get();
                g->setName(strdup(name));
                //printf("added geode [%s]\n", name);
            }
            else
                cerr << "no/wrong data received" << endl;
        }
    }
}

void cuIsoSurface::preDraw(osg::RenderInfo &)
{
    if (!initDone)
    {
        isoEngine.Init(0);
        initDone = true;
    }

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        IsoDrawable *drawable = dynamic_cast<IsoDrawable *>(i->second->getDrawable(0));
        if (drawable)
            drawable->preDraw();
    }
}

void cuIsoSurface::preFrame()
{
    if (!initDone)
        return;

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        IsoDrawable *drawable = dynamic_cast<IsoDrawable *>(i->second->getDrawable(0));
        if (drawable)
            drawable->preFrame();
    }
}

void cuIsoSurface::postFrame()
{
    if (!initDone)
        return;

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        IsoDrawable *drawable = dynamic_cast<IsoDrawable *>(i->second->getDrawable(0));
        if (drawable)
            drawable->postFrame();
    }
}

IsoDrawable::IsoDrawable(ui::Slider *s, coTUIFloatSlider *tui,
                         coTUIToggleButton *button,
                         const RenderObject *g, const RenderObject *map,
                         const RenderObject *data,
                         float *b, float mi = 0.0, float ma = 0.0)
    : osg::Drawable()
    , coTUIListener()
    , state(NULL)
    , geom(g)
    , changed(false)
    , animate(false)
    , anim(0)
    , threshold(0)
    , min(mi)
    , max(ma)
    , slider(s)
    , tuiSlider(tui)
    , tuiButton(button)
{
    if (geom && geom->isUnstructuredGrid())
    {

        const float *red = data->getFloat(Field::Red);
        const float *green = data->getFloat(Field::Green);
        const float *blue = data->getFloat(Field::Blue);

        int numElem, numConn, numCoord;
        geom->getSize(numElem, numConn, numCoord);
        const int *connList = geom->getInt(Field::Connections);
        const int *elemList = geom->getInt(Field::Elements);
        const int *typeList = geom->getInt(Field::Types);
        const float *x = geom->getFloat(Field::X);
        const float *y = geom->getFloat(Field::Y);
        const float *z = geom->getFloat(Field::Z);

        const char *mapName = NULL;
        const float *xm = NULL, *ym = NULL, *zm = NULL;
        if (map)
        {
            xm = map->getFloat(Field::X);
            ym = map->getFloat(Field::Y);
            zm = map->getFloat(Field::Z);

            mapName = map->getName();
        }

        const char *dataName = NULL;
        if (data)
            dataName = data->getName();

        state = isoEngine.InitState(geom->getName(), dataName, mapName,
                                    typeList, elemList, connList, x, y, z,
                                    numElem, numConn, numCoord, red,
                                    xm, ym, zm, min, max);

        box = osg::BoundingBox(b[0], b[1], b[2], b[3], b[4], b[5]);

        setDataVariance(Object::DYNAMIC);
        threshold = FLT_MAX;
        changed = true;
        if (tuiSlider)
            tuiSlider->setEventListener(this);
        if (tuiButton)
            tuiButton->setEventListener(this);
    }
}

IsoDrawable::~IsoDrawable()
{
    CleanupState(state);
}

void IsoDrawable::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == tuiSlider)
    {

        //float thresh = tuiSlider->getValue();
        //slider->setValue(thresh);
    }

    else if (tUIItem == tuiButton)
    {

        animate = tuiButton->getState();
    }
}

void IsoDrawable::preFrame()
{
    float thresh = slider->value();
    /*
   if (animate && anim < 5)
      anim ++;
   anim %= 5;
   */
    if (animate && !anim)
    {

        changed = true;
        threshold += (max - min) / 500;
        if (threshold > max)
            threshold = min;
        //slider->setValue(threshold);
        tuiSlider->setValue(threshold);
    }
    else if (thresh != threshold)
    {
        changed = true;
        threshold = thresh;

        if (tuiSlider->getValue() != thresh)
            tuiSlider->setValue(thresh);
    }
}

void IsoDrawable::preDraw()
{
    if (changed)
    {

        int numVertices;
        isoEngine.computeIsoMesh(state, threshold,
                                 &numVertices, 0, 0);
        printf("num: %d\n", numVertices);
    }
}

void IsoDrawable::postFrame()
{
    changed = false;
}

IsoDrawable::IsoDrawable(const IsoDrawable &draw, const osg::CopyOp &op)
    : osg::Drawable(draw, op)
    , coTUIListener()
{
}

void IsoDrawable::drawImplementation(osg::RenderInfo & /*info*/) const
{
    RenderCUDAState(state);
}

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
osg::BoundingBox IsoDrawable::computeBoundingBox() const
#else
osg::BoundingBox IsoDrawable::computeBound() const
#endif
{
    return box;
}

osg::Object *IsoDrawable::cloneType() const
{
    return new IsoDrawable(NULL, NULL, NULL, NULL, NULL, NULL, NULL);
}

osg::Object *IsoDrawable::clone(const osg::CopyOp &op) const
{
    return new IsoDrawable(*this, op);
}

COVERPLUGIN(cuIsoSurface)
