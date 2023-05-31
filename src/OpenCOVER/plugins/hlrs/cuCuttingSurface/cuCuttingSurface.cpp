/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "cudaEngine.h"
#include "cuCuttingSurface.h"

#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

#include <PluginUtil/ColorBar.h>

#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Manager.h>

#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Texture1D>
#include <osg/Texture2D>
#include <osg/Material>
#include <osg/TexEnv>

#include <osgDB/FileUtils>
#include <osgDB/ReadFile>

#include <sysdep/opengl.h>

#include <PluginUtil/PluginMessageTypes.h>

#include <cover/coVRMSController.h>
#include <cover/coVRShader.h>
#include <net/tokenbuffer.h>

using namespace covise;

using vrui::coInteraction;

CUDAEngine cuttingEngine;

void RenderCUDAState(State *);

void getMinMax(const float *data, int numElem, float *min,
               float *max, float minV = -FLT_MAX, float maxV = FLT_MAX);

void removeSpikesAdaptive(const float *data, int numElem,
                          float *min, float *max);

cuCuttingSurface::cuCuttingSurface()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("cuCuttingSurface", cover->ui)
, initDone(false)
, menu(NULL)
{
}

cuCuttingSurface::~cuCuttingSurface()
{
    std::map<std::string, osg::Geode *>::iterator i;

    for (i = geode.begin(); i != geode.end(); i++)
    {
        cover->getObjectsRoot()->removeChild((*i).second);
    }
}

bool cuCuttingSurface::init()
{
    return true;
}

void cuCuttingSurface::preDraw(osg::RenderInfo &)
{
    if (!initDone)
    {
        cuttingEngine.Init(0);
        initDone = true;
    }

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        osg::Geode *geode = i->second;
        if (geode->getNumDrawables() > 0)
        {
            CuttingDrawable *drawable = dynamic_cast<CuttingDrawable *>(geode->getDrawable(0));
            if (drawable)
                drawable->preDraw();
        }
    }
}

ui::Button *cuCuttingSurface::getMenu(const RenderObject *container,
                                              const RenderObject * /*data*/,
                                              const RenderObject *tex)
{
    ui::Button *check = NULL;

    if (!tex)
        tex = container->getTexture();

    if (!tex)
        return NULL;

    std::map<std::string, ui::Menu *>::iterator i = menus.find(container->getName());

    if (i == menus.end())
    {
        const char *name = NULL;
        if (tex)
        {
            if (tex->isSet())
            {
                // TODO
                //int numElements;
                //RenderObject **e = tex->getAllElements(&numElements);
                //if (numElements > 0)
                //   name = e[0]->getAttribute("LABEL");
            }
            else
                name = tex->getAttribute("LABEL");
        }
        if (!name)
            name = container->getName();

        ui::Menu *subMenu = new ui::Menu(menu, name); 
        check = new ui::Button(subMenu, "enable");
        check->setState(true);
        menus[container->getName()] = subMenu;

        const char *attr = tex->getAttribute("COLORMAP");
        if (attr)
        {
            std::string species;
            std::vector<float> r,g,b,a;
            int numColors;
            float min, max;
            ColorBar::parseAttrib(attr, species, min, max,
                                  numColors, r, g, b, a);
            struct minmax m = { min, max };
            minMax[container->getName()] = m;

            //ColorBar *bar = new ColorBar(name, species, min, max, numColors, r, g, b, a);
        }
    }
    else
    {
        auto sm = i->second;
        auto path = sm->path()+"."+"enable";
        check = dynamic_cast<ui::Button *>(cover->ui->getByPath(path));
    }

    return check;
}

void cuCuttingSurface::removeObject(const char *objName, bool /*replace*/)
{
    minMax.erase(objName);

    std::map<std::string, osg::Group *>::iterator i = groups.find(std::string(objName));
    if (i != groups.end())
    {
        for (unsigned int index = 0; index < i->second->getNumChildren(); index++)
        {
            osg::Node *node = i->second->getChild(index);

            std::string name = node->getName();
            geode.erase(name);
            //delete(dynamic_cast<osg::Geode *>(node));
        }
        // TODO: delete group contents + drawable
        //printf("objectsroot removechild [%s]\n", i->second->getName().c_str());
        cover->getObjectsRoot()->removeChild(i->second);
        groups.erase(i);
    }

    // erase interactor
    std::map<std::string, coVR3DTransRotInteractor *>::iterator pi = interactors.find(objName);
    if (pi != interactors.end())
    {
        pi->second->hide();
        interactors.erase(pi);
    }

    std::map<std::string, ui::Menu *>::iterator mi = menus.find(objName);
    if (mi != menus.end())
    {
        delete mi->second;
        menus.erase(mi);
    }
}

void cuCuttingSurface::addObject(const RenderObject *container, osg::Group *, const RenderObject *geometry, const RenderObject *normals, const RenderObject *colorObj, const RenderObject *texObj)
{
    /*
   const char * variant = container->getAttribute("VARIANT");
   if (!variant || strncmp(variant, "GPGPU", 5))
      return;
*/
    osg::Group *group = NULL;

    std::map<std::string, osg::Group *>::iterator gi = groups.find(container->getName());
    if (gi == groups.end())
    {
        group = new osg::Group();
        group->setName(container->getName());
        //printf("objectsroot addchild [%s]\n", container->getName());
        cover->getObjectsRoot()->addChild(group);
        groups[container->getName()] = group;
    }
    else
        group = gi->second;

    if (!menu)
    {
        menu = new ui::Menu("cuCuttingSurfaceUSG", this);
        if (cover->visMenu)
            cover->visMenu->add(menu);
    }

    if (container)
    {
        // get min and max from COLORMAP parameter of the container object
        getMenu(container, colorObj, texObj);
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

                float min = 0.0, max = 0.0;
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
                state->setMode(GL_BLEND, osg::StateAttribute::ON);
                state->setMode(GL_LIGHTING, osg::StateAttribute::ON);
                state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
                state->setRenderBinDetails(10, "RenderBin");
                state->setNestRenderBins(false);

                osg::TexEnv *tEnv = new osg::TexEnv;
                tEnv->setMode(osg::TexEnv::REPLACE);
                state->setTextureAttributeAndModes(0, tEnv, osg::StateAttribute::ON);

#ifdef LIC
                osg::Texture1D *colorTex = new osg::Texture1D();
                colorTex->setDataVariance(osg::Object::DYNAMIC);

                osg::Image *colorImage = new osg::Image();
                if (texObj)
                {
                    int x, y, z;
                    texObj->getSize(x, y, z);

                    colorImage->allocateImage(x, y, z, GL_RGBA, GL_UNSIGNED_BYTE);
                    colorImage->setImage(x, y, z, 4, GL_RGBA, GL_UNSIGNED_BYTE, tex, osg::Image::NO_DELETE);
                }
                else
                    colorImage = osgDB::readImageFile(coVRFileManager::instance()->getName("share/covise/materials/map.png"));

                colorTex->setImage(colorImage);
                colorTex->setFilter(osg::Texture1D::MIN_FILTER, osg::Texture1D::LINEAR);
                colorTex->setFilter(osg::Texture1D::MAG_FILTER, osg::Texture1D::LINEAR);

                osg::Texture2D *noiseTex = new osg::Texture2D;
                const char *texFile = coVRFileManager::instance()->getName("share/covise/materials/noise.png");
                if (texFile)
                {
                    osg::Image *noiseImage = osgDB::readImageFile(texFile);
                    if (noiseImage)
                        noiseTex->setImage(noiseImage);
                }
                noiseTex->setFilter(osg::Texture1D::MIN_FILTER,
                                    osg::Texture1D::LINEAR);
                noiseTex->setFilter(osg::Texture1D::MAG_FILTER,
                                    osg::Texture1D::LINEAR);

                state->setTextureAttributeAndModes(0, colorTex, osg::StateAttribute::ON);
                state->setTextureAttributeAndModes(1, noiseTex, osg::StateAttribute::ON);
                state->setTextureMode(0, GL_TEXTURE_1D, osg::StateAttribute::ON);
                state->setTextureMode(1, GL_TEXTURE_2D, osg::StateAttribute::ON);
#endif

                g->setStateSet(state.get());

                ui::Button *check = getMenu(container, colorObj, texObj);
                coVR3DTransRotInteractor *interactor = NULL;

                std::map<std::string, coVR3DTransRotInteractor *>::iterator pi = interactors.find(container->getName());
                if (pi == interactors.end())
                {
                    osg::Matrix m;
                    m.makeTranslate(osg::Vec3f(-1.0, 2.0, 0.5));
                    //interactor = new coVR3DTransRotInteractor(m, cover->getSceneSize() / 80.0, coVR3DTransRotInteractor::TwoD, coInteraction::ButtonA, "hand", "Plane_S0", coInteraction::High);
                    interactor = new coVR3DTransRotInteractor(m, cover->getSceneSize() / 50.0, coInteraction::ButtonA, "hand", "Plane_S0", coInteraction::High);

                    interactor->show();
                    interactor->enableIntersection();
                    interactors[container->getName()] = interactor;
                }
                else
                    interactor = pi->second;

                osg::ref_ptr<CuttingDrawable> draw = new CuttingDrawable(check, interactor, geometry, normals,
                                                                         colorObj, box, min, max);

                char name[256];
                snprintf(name, 256, "%s_%s", container->getName(),
                         geometry->getName());
                geode[std::string(name)] = g.get();
                g->setName(strdup(name));

                g->addDrawable(draw.get());
                draw->setUseDisplayList(false);
                group->addChild(g.get());
#ifdef LIC
                opencover::coVRShader *shader = coVRShaderList::instance()->get("lic");
                if (shader)
                    shader->apply(g);
#else
                opencover::coVRShader *shader = coVRShaderList::instance()->get("tex1dreplace");
                shader->apply(g);
#endif
                //printf(" group [%s] addchild [%p]\n", g->getName().c_str(), name);
            }
            else
                cerr << "no/wrong data received" << endl;
        }
    }
}

void cuCuttingSurface::preFrame()
{
    if (!initDone)
        return;

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        osg::Geode *geode = i->second;
        if (geode->getNumDrawables() > 0)
        {
            CuttingDrawable *drawable = dynamic_cast<CuttingDrawable *>(geode->getDrawable(0));
            if (drawable)
                drawable->preFrame();
        }
    }

    std::map<std::string, coVR3DTransRotInteractor *>::iterator pi;
    for (pi = interactors.begin(); pi != interactors.end(); pi++)
        pi->second->preFrame();
}

void cuCuttingSurface::postFrame()
{
    if (!initDone)
        return;

    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        osg::Geode *geode = i->second;
        if (geode->getNumDrawables() > 0)
        {
            CuttingDrawable *drawable = dynamic_cast<CuttingDrawable *>(geode->getDrawable(0));
            if (drawable)
                drawable->postFrame();
        }
    }
}

// receive messages for Cuttinsurface updates from remote or the script plugin
void cuCuttingSurface::message(int toWhom, int type, int len, const void *buf)
{
    (void)toWhom;

    if (type != PluginMessageTypes::HLRS_cuCuttingSurface)
        return;
    int cuttingPlaneNumber;
    TokenBuffer tb((const char *)buf, len);
    tb >> cuttingPlaneNumber;
    double dmat[16];
    for (int i = 0; i < 16; i++)
        tb >> dmat[i];
    osg::Matrix mat(dmat);
    int num = 0;
    std::map<std::string, osg::Geode *>::iterator i;
    for (i = geode.begin(); i != geode.end(); i++)
    {
        if (num == cuttingPlaneNumber)
        {
            osg::Geode *geode = i->second;
            if (geode->getNumDrawables() > 0)
            {
                CuttingDrawable *drawable = dynamic_cast<CuttingDrawable *>(geode->getDrawable(0));
                if (drawable)
                    drawable->setMatrix(mat);
            }
        }
        num++;
    }
}

CuttingDrawable::CuttingDrawable(ui::Button *m,
                                 coVR3DTransRotInteractor *i, const RenderObject *g,
                                 const RenderObject *map, const RenderObject *data,
                                 float *b, float min = 0.0, float max = 0.0)
    : osg::Geometry()
    , state(NULL)
    , geom(g)
    , interactorChanged(false)
    , distance(0)
    , menu(m)
    , planeInteractor(i)
{
    if (geom && geom->isUnstructuredGrid())
    {

        const float *red = data->getFloat(Field::Red);
        const float *green = data->getFloat(Field::Green);
        const float *blue = data->getFloat(Field::Blue);

        int numElem, numConn, numCoord;
        geom->getSize(numElem, numConn, numCoord);

        const float *x = geom->getFloat(Field::X);
        const float *y = geom->getFloat(Field::Y);
        const float *z = geom->getFloat(Field::Z);
        const int *connList = geom->getInt(Field::Connections);
        const int *elemList = geom->getInt(Field::Elements);
        const int *typeList = geom->getInt(Field::Types);

        const float *xm = NULL, *ym = NULL, *zm = NULL;
        const char *mapName = NULL;
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

        state = cuttingEngine.InitState(geom->getName(), dataName, mapName,
                                        typeList, elemList, connList, x, y, z,
                                        numElem, numConn, numCoord, red,
                                        xm, ym, zm, min, max);

        box = osg::BoundingBox(b[0], b[1], b[2], b[3], b[4], b[5]);

        distance = FLT_MAX;

        name = geom->getName();
    }
	if (menu)
	{
		menu->setCallback([this](bool state) {
			if (!state)
				interactorChanged = true;
			});
	}
}

CuttingDrawable::~CuttingDrawable()
{
    CleanupState(state);
}

void CuttingDrawable::preFrame()
{
    if (menu && !menu->state())
        return;

    interactorChanged = false;
    if (planeInteractor->isRunning())
        interactorChanged = true;
}

void CuttingDrawable::preDraw()
{
    if (menu && !menu->state())
        return;

    if (interactorChanged || remoteMatrixChanged)
    {

        osg::Matrix m;
        if (remoteMatrixChanged)
        {
            m = remoteMatrix;
            planeInteractor->updateTransform(m);
            remoteMatrixChanged = false;
        }
        else
            m = planeInteractor->getMatrix();
        osg::Vec3 point = m.getTrans();
        osg::Vec4 axis(0, 0, 1, 0);
        osg::Vec4 normal = axis * m;
        normal.normalize();

        //const osg::Quat::value_type *r = m.getRotate().inverse()._v;
        const osg::Quat invRot = m.getRotate().inverse();
        const double *r = invRot._v;
        //printf("......... %f (%f %f %f)\n", r[0], r[1], r[2], r[3]);
        float rot[4];
        for (int index = 0; index < 4; index++)
            rot[index] = r[index];

        int numVertices;
        distance = (point * normal);
        cuttingEngine.computeCuttingMesh(state, rot,
                                         normal.x(), normal.y(), normal.z(),
                                         distance, &numVertices, 0, 0);
        //printf("numVertices: %d\n", numVertices);

        interactorChanged = false;
    }
}

void CuttingDrawable::postFrame()
{
}

CuttingDrawable::CuttingDrawable(const CuttingDrawable &draw,
                                 const osg::CopyOp &op)
    : osg::Geometry(draw, op)
{
}

void CuttingDrawable::drawImplementation(osg::RenderInfo & /*info*/) const
{
    if (menu && !menu->state())
        return;

    RenderCUDAState(state);
}

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
osg::BoundingBox CuttingDrawable::computeBoundingBox() const
#else
osg::BoundingBox CuttingDrawable::computeBound() const
#endif
{
    return box;
}

osg::Object *CuttingDrawable::cloneType() const
{
    return new CuttingDrawable(NULL, NULL, NULL, NULL, NULL, NULL);
}

osg::Object *CuttingDrawable::clone(const osg::CopyOp &op) const
{
    return new CuttingDrawable(*this, op);
}

COVERPLUGIN(cuCuttingSurface)
