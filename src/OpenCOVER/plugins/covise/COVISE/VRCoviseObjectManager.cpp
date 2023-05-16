/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                                                                      *
 *                                                                      *
 *                            (C) 1996                                  *
 *              Computer Centre University of Stuttgart                 *
 *                         Allmandring 30                               *
 *                       D-70550 Stuttgart                              *
 *                            Germany                                   *
 *                                                                      *
 *                                                                      *
 * File   VRCoviseConnection.C                                     *
 *                                                                      *
 * Description  covise interface class                                *
 *                                                                      *
 * Author   D. Rantzau                                            *
 *          D. Rainer                                             *
 *          F. Foehl                                              *
 *                                                      *
 * Date           20.08.97                                        *
 *                                                      *
 * Status   in dev                                      *
 *                                                                      *
 ************************************************************************/

#include <cover/input/VRKeys.h>
#include "VRCoviseObjectManager.h"
#include <CovisePluginUtil/VRCoviseGeometryManager.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRShader.h>
#include "CoviseSG.h"
#include "VRRotator.h"
#include "VRSlider.h"
#include "coVRTUIParam.h"
#include "VRVectorInteractor.h"
#include <cover/coVRMSController.h>
#include "coVRMenuList.h"
#include <cover/coVRPluginList.h>
#include <CoviseRenderObject.h>
#include <PluginUtil/coSphere.h>
#include <cover/coVRPluginSupport.h>
#include "coCoviseInteractor.h"
#include <cover/coVRAnimationManager.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <cover/VRRegisterSceneGraph.h>
#include <PluginUtil/coLOD.h>

#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <config/CoviseConfig.h>
#include <util/coLog.h>
#include <do/coDistributedObject.h>
#include <appl/RenderInterface.h>

#include <osg/Sequence>
#include <osg/MatrixTransform>
#include <osg/PolygonOffset>
#include <osg/Texture1D>
#include <osg/Texture2D>
#include <osg/LOD>

#include <stdio.h>
#include <cstring>
//#include <cover/coVRDePee.h>

using namespace std;
using namespace opencover;
using namespace covise;
using namespace vrui;

ColorMap::ColorMap()
: min(0.)
, max(1.)
, vertexMapShader(NULL)
, textureMapShader(NULL)
{
    tex = new osg::Texture1D;
    img = new osg::Image;
    tex->setImage(img);
    tex->setInternalFormat(GL_RGBA);
    img->allocateImage(2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE);
    unsigned char *rgba = img->data();
    rgba[0] = 16;
    rgba[1] = 16;
    rgba[2] = 16;
    rgba[3] = 255;
    rgba[4] = 200;
    rgba[5] = 200;
    rgba[6] = 200;
    rgba[7] = 255;

    tex->setBorderWidth( 0 );
    tex->setFilter( osg::Texture::MIN_FILTER, osg::Texture::LINEAR );
    tex->setFilter( osg::Texture::MAG_FILTER, osg::Texture::LINEAR );
    tex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);

    textureMapShader = coVRShaderList::instance()->getUnique("MapColorsTexture");
    setMinMax(min, max);
}

void ColorMap::setMinMax(float min, float max)
{
    this->min = min;
    this->max = max;

    if (vertexMapShader)
    {
        vertexMapShader->setFloatUniform("rangeMin", min);
        vertexMapShader->setFloatUniform("rangeMax", max);
    }
    if (textureMapShader)
    {
        textureMapShader->setFloatUniform("rangeMin", min);
        textureMapShader->setFloatUniform("rangeMax", max);
    }
}


static ObjectManager *singleton = NULL;

//================================================================
// ObjectManager methods
//================================================================
ObjectManager *ObjectManager::instance()
{
    assert(singleton);
    return singleton;
}

ObjectManager::ObjectManager(coVRPlugin *plugin): materialList(NULL), coviseSG(new CoviseSG(plugin))
{
    assert(!singleton);
    singleton = this;
    if (cover->debugLevel(2))
        fprintf(stderr, "new ObjectManager\n");

    added_a_rotate_flag = 0;

    c_feedback = false;
    i_feedback = false;
    t_feedback = false;
    noFrameBuffer = new osg::ColorMask(false, false, false, false);
    interactionA = new coTrackerButtonInteraction(coInteraction::ButtonA, "CoviseInteractions");

    anzset = 0;
    depthPeeling = coCoviseConfig::isOn("COVER.DepthPeeling", false);
}

ObjectManager::~ObjectManager()
{
    delete coviseSG;
    if (cover->debugLevel(2))
        fprintf(stderr, "delete ObjectManager\n");
    delete interactionA;
    delete materialList;
    delete GeometryManager::instance();
    singleton = nullptr;
}

/*______________________________________________________________________*/
void
ObjectManager::update(void)
{

    // newCode
    osg::Matrix invBase = cover->getInvBaseMat();
    osg::Matrix pointer = cover->getPointerMat();
    pointer.postMult(invBase);
    // pointer matrix now in object koordinates

    if (coVRNavigationManager::instance()->isSnapping() && !coVRNavigationManager::instance()->isDegreeSnapping())
    {
        snapTo45Degrees(&pointer);
    }
    else if (coVRNavigationManager::instance()->isSnapping() && coVRNavigationManager::instance()->isDegreeSnapping())
    {
        snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &pointer);
    }
    // done snaping

    osg::Vec3 position(pointer(3, 0),
                       pointer(3, 1),
                       pointer(3, 2));

    osg::Vec3 normal(pointer(1, 0), pointer(1, 1), pointer(1, 2));

    osg::Vec3 normal2(pointer(0, 0), pointer(0, 1), pointer(0, 2));

    float c;
    c = position * (normal);
    c /= normal.length();
    normal.normalize();

    char buf[1000];

    if ((c_feedback | t_feedback | i_feedback))
    {
        CoviseRender::set_feedback_info(this->currentFeedbackInfo);

        char ch;
        switch (ch = CoviseRender::get_feedback_type())
        {
        case 'C':
            /* button just pressed */
            if (interactionA->wasStarted())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "vertex\nFloatVector\n%f %f %f", normal[0], normal[1], normal[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "scalar\nFloatScalar\n%f", c);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
            }
            break;
        case 'G': // CutGeometry with new parameter names
            /* button just pressed */
            if (interactionA->wasStarted())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "normal\nFloatVector\n%f %f %f", normal[0], normal[1], normal[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "distance\nFloatScalar\n%f", c);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
            }
            break;
        case 'Z':
            /* button just pressed */
            if (interactionA->wasStarted())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "vertex\nFloatVector\n%f %f %f", position[0], position[1], position[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
            }
            break;

        case 'A':
            /* button just pressed */
            if (interactionA->wasStarted())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "position\nFloatVector\n%f %f %f", position[0], position[1], position[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "normal\nFloatVector\n%f %f %f", normal[0], normal[1], normal[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "normal2\nFloatVector\n%f %f %f", normal2[0], normal2[1], normal2[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
            }
            break;

        case 'S': // S for 'steady cam'
            /* button just pressed */
            if (interactionA->wasStarted())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "position\nFloatVector\n%f %f %f", position[0], position[1], position[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    sprintf(buf, "direction\nFloatVector\n%f %f %f", normal[0], normal[1], normal[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);

                    int frame_no = 0;
                    // somehow find out current frame-id

                    // anderes Problem: wie dem Renderer sagen, wo er
                    // die Kameraposition setzen soll ?

                    sprintf(buf, "timestep\nIntScalar\n%d", frame_no);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
            }
            break;

        case 'T':
            /* button just pressed */
            if (interactionA->wasStarted())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "startpoint1\nFloatVector\n%f %f %f", position[0], position[1], position[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                }
            }
            /* button just released */
            if (interactionA->wasStopped())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "startpoint2\nFloatVector\n%f %f %f", position[0], position[1], position[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
            }
            break;
        case 'P':
            /* button just pressed */
            if (interactionA->wasStarted())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "startpoint1\nFloatVector\n%f %f %f", position[0], position[1], position[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                }
            }
            /* button just released */
            if (interactionA->wasStopped())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "startpoint2\nFloatVector\n%f %f %f", position[0], position[1], position[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    // TracerUsg has no normal parameter (nor Tracer)...
                    /* && strcmp(currentFeedbackInfo+1,"Tracer")!= 0 */
                    if (strncmp(currentFeedbackInfo + 1, "Tracer", strlen("Tracer")) != 0)
                    {
                        sprintf(buf, "normal\nFloatVector\n%f %f %f", normal[0], normal[1], normal[2]);
                        CoviseRender::send_feedback_message("PARAM", buf);
                    }
                    // ... but has direction
                    sprintf(buf, "direction\nFloatVector\n%f %f %f", normal2[0], normal2[1], normal2[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
            }
            break;

        case 'I':
            /* button just pressed */
            if (interactionA->wasStarted())
            {
                if (coVRMSController::instance()->isMaster())
                {
                    fprintf(stdout, "\a");
                    fflush(stdout);
                    sprintf(buf, "isopoint\nFloatVector\n%f %f %f", position[0], position[1], position[2]);
                    CoviseRender::send_feedback_message("PARAM", buf);
                    buf[0] = '\0';
                    CoviseRender::send_feedback_message("EXEC", buf);
                }
            }
            break;

        default:
            printf("unknown feedback type %c\n", ch);
        }
    }
    else
    {
    }
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void ObjectManager::addObject(const char *object, const coDistributedObject *data_obj)
{
#ifdef DBGPRINT
    printf("...... ObjectManager::addObject %s\n" object);
#endif

    if (cover->debugLevel(4))
        fprintf(stderr, "--ObjectManager (%s)::addObject %s\n", getenv("HOST") ? getenv("HOST") : "unknown", object);
    if ((!data_obj) && (coVRMSController::instance()->isMaster()))
    {
        data_obj = coDistributedObject::createFromShm(object);
    }
    CoviseRenderObject *ro = new CoviseRenderObject(data_obj);

    if (ro != NULL)
    {
        m_roMap[object] = ro;
        //fprintf(stderr, "++++++ObjectManager(%s)::addObject %s  data_obj=%s renderObj=%s\n", getenv("HOST"), object,data_obj->getName(), ro->getName() );

        std::string gtype = ro->getType();
        if (gtype == "DOTEXT")
        {
        }
        else if (gtype == "COLMAP")
        {
            addColorMap(object, ro);
        }
        else
        {
            // also send container object 'ro' for plugin usage
            if (osg::Node *n = addGeometry(object, NULL, ro, NULL, NULL, NULL, NULL, ro, NULL))
            {
                coviseSG->addNode(n, (osg::Group *)NULL, ro);
            }
        }
    }
}

coInteractor *ObjectManager::handleInteractors(CoviseRenderObject *container, CoviseRenderObject *geomObj, CoviseRenderObject *normObj, CoviseRenderObject *colorObj, CoviseRenderObject *texObj) const
{

    CoviseRenderObject *ro[4] = {
        geomObj,
        normObj,
        colorObj,
        texObj
    };

    coInteractor *ret = nullptr;
    if (geomObj)
    {
        // a new object arrived, look for interactors

        for (int k = 0; k < 4; ++k)
        {
            if (!ro[k])
                continue;

            char **name, **value;
            int n = ro[k]->getAllAttributes(name, value);
            for (int i = 0; i < n; i++)
            {
                if (strcmp(name[i], "MODULE") == 0 || strcmp(name[i], "PLUGIN") == 0)
                {
                    cover->addPlugin(value[i]);
                }

                if (strcmp(name[i], "INTERACTOR") == 0)
                {
                    coInteractor *it = new coCoviseInteractor(ro[k]->getName(), ro[k], value[i]);
                    if (it->getPluginName())
                        coVRPluginList::instance()->addPlugin(it->getPluginName());

                    it->incRefCount();
                    coVRPluginList::instance()->newInteractor(container, it);
                    if (it->refCount() > 1)
                    {
                        if (strcmp(it->getModuleName(), "Colors") != 0)
                            ret = it;
                    }
                    it->decRefCount();
                }
            }
        }
    }
    return ret;
}

const ColorMap &ObjectManager::getColorMap(const std::string &species)
{
   return colormaps[species];
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void ObjectManager::coviseError(const char *error)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "ObjectManager::coviseError\n");

    coVRPluginList::instance()->coviseError(error);
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void ObjectManager::deleteObject(const char *name, bool groupobject)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "--ObjectManager::deleteObject: %s\n", name);

    //printf("ObjectManager::deleteObject\n");
    //printf("\t object = %s\n", name);

    int i, n;
    for (i = 0; i < anzset; i++)
    {
        if (strcmp(setnames[i], name) == 0)
        {
            for (n = 0; n < elemanz[i]; n++)
            {
                deleteObject(elemnames[i][n]);
                delete[] elemnames[i][n];
            }
            delete[] elemnames[i];
            n = i;
            anzset--;
            while (n < (anzset))
            {
                elemanz[n] = elemanz[n + 1];
                elemnames[n] = elemnames[n + 1];
                setnames[n] = setnames[n + 1];
                n++;
            }
        }
    }
    removeGeometry(name, groupobject);
#ifdef PHANTOM_TRACKER
    if (feedbackList)
        feedbackList->removeData(name);
#endif
    RenderObjectMap::iterator it = m_roMap.find(name);
    if (it != m_roMap.end())
    {
        RenderObject *ro = it->second;
        m_roMap.erase(it);
        delete ro;
    }
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void ObjectManager::removeGeometry(const char *name, bool groupobject)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "ObjectManager::removeGeometry=%s\n", name);

    // remove all Menus attached to this geometry
    coVRMenuList::instance()->removeAll(name);

    coVRPluginList::instance()->removeObject(name, CoviseRender::isReplace());
    coviseSG->deleteNode(name, groupobject);
}

//======================================================================
// create a color data object for a named color
//======================================================================
FILE *fp = NULL;
int isopen = 0;
static int create_named_color(const char *cname)
{
    int r = 255, g = 255, b = 255;
    unsigned int rgba;
    char line[80];
    char *tp, *token[15], *chptr;
    int count;
    const int tmax = 15;
    // first check if we get the name of the color or the RGB values
    const char *first_blank = strchr(cname, ' ');
    const char *last_blank = strrchr(cname, ' ');

    if (first_blank && last_blank && (first_blank - cname <= 3)
        && (last_blank - first_blank > 0) && (last_blank - first_blank <= 4))
    {
        if (sscanf(cname, "%d %d %d", &r, &g, &b) != 3)
        {
            if (cover->debugLevel(2))
                cerr << "create_named_color: sscanf failed" << endl;
        }
    }

    else if (strcasecmp(cname, "white") == 0)
    {
        r = g = b = 150;
    }
    else
    {
        while (fgets(line, sizeof(line), fp) != NULL)
        {
            count = 0;
            tp = strtok(line, " \t");
            for (count = 0; count < tmax && tp != NULL;)
            {
                token[count] = tp;
                tp = strtok(NULL, " \t");
                count++;
            }
            token[count] = NULL;
            std::string token3;
            if (count > 3)
            {
                token3 = token[3];
                if (count == 5)
                {
                    token3 += " ";
                    token3 += token[4];
                }
            }
            if (strstr(token3.c_str(), cname) != NULL)
            {
                r = atoi(token[0]);
                g = atoi(token[1]);
                b = atoi(token[2]);
                fseek(fp, 0L, SEEK_SET);
                break;
            }
        }
        fseek(fp, 0L, SEEK_SET);
    }
#ifdef BYTESWAP
    chptr = (char *)&rgba;
    *chptr = (char)(unsigned char)(255);
    chptr++;
    *(chptr) = (unsigned char)(b);
    chptr++;
    *(chptr) = (unsigned char)(g);
    chptr++;
    *(chptr) = (unsigned char)(r);
#else
    chptr = (char *)&rgba;
    *chptr = (unsigned char)(r);
    chptr++;
    *(chptr) = (unsigned char)(g);
    chptr++;
    *(chptr) = (unsigned char)(b);
    chptr++;
    *(chptr) = (unsigned char)(255);
#endif
    return rgba;
}

//----------------------------------------------------------------
//
//----------------------------------------------------------------
void ObjectManager::addColorMap(const char *object, CoviseRenderObject *cmap)
{
    std::string species;
    if (const char *sp = cmap->getAttribute("SPECIES"))
    {
        species = sp;
    }

    handleInteractors(cmap, cmap, NULL, NULL, NULL);
    coVRPluginList::instance()->addObject(cmap, NULL, NULL, NULL, NULL, NULL);

    ColorMap &cm = colormaps[species];
    cm.setMinMax(cmap->getMin(0), cmap->getMax(0));
    const float *cols = cmap->getFloat(Field::ColorMap);
    cm.lut.resize(cmap->getNumColors());
    for (int i=0; i<cm.lut.size(); ++i)
    {
        cm.lut[i].r = cols[i*5+0]*255.99;
        cm.lut[i].g = cols[i*5+1]*255.99;
        cm.lut[i].b = cols[i*5+2]*255.99;
        cm.lut[i].a = cols[i*5+3]*255.99;
        //std::cerr << "  " << i << ": " << (int)cm.lut[i].r << " " << (int)cm.lut[i].g << " " << (int)cm.lut[i].b << " " << (int)cm.lut[i].a << std::endl;
    }
    cm.img->allocateImage(cm.lut.size(), 1, 1, GL_RGBA, GL_UNSIGNED_BYTE);
    memcpy(cm.img->data(), &cm.lut[0], cm.lut.size()*4);
    cm.tex->setInternalFormat(GL_RGBA);
    cm.img->dirty();
    std::cerr << "colormap for species " << species << ": range " << cm.min << " - " << cm.max << ", #steps: " << cmap->getNumColors() << std::endl;
}

osg::Node *ObjectManager::addGeometry(const char *object, osg::Group *root, CoviseRenderObject *geometry,
                                      CoviseRenderObject *normals, CoviseRenderObject *colors, CoviseRenderObject *texture, CoviseRenderObject *vertexAttribute, CoviseRenderObject *container, const char *lod)
{
    CoviseRenderObject *const *dobjsg = NULL; // Geometry Set elements
    CoviseRenderObject *const *dobjsc = NULL; // Color Set elements
    CoviseRenderObject *const *dobjsn = NULL; // Normal Set elements
    CoviseRenderObject *const *dobjst = NULL; // Normal Set elements
    CoviseRenderObject *const *dobjsva = NULL; // VertexAttribute Set elements
    // number of elements per geometry,color and normal set
    bool is_animation = false;
    int no_elems = 0, no_c = 0, no_n = 0, no_t = 0, no_va = 0;
    int normalbinding = Bind::None, colorbinding = Bind::None;
    int colorpacking = Pack::None;
    int vertexOrder = 0;
    int no_poly = 0;
    int no_strip = 0;
    int no_vert = 0;
    int no_points = 0;
    int no_lines = 0;
    int no_faces = 0; // for deciding if per-vertex or per-face binding is to be used
    int curset;
    int iRenderMethod = coSphere::RENDER_METHOD_CPU_BILLBOARDS;
    int *type_c = NULL;
    float pointsize = 2.f;
    float linewidth = 2.f;
    // sphere specific data properties
    float *radii_c = NULL;
    int *v_l = NULL, *l_l = NULL, *el, *vl;
    int xsize = 0, ysize = 0, zsize = 0 /*,i_dim,j_dim,k_dim*/;
    float xmax = 0.f, xmin = 0.f, ymax = 0.f, ymin = 0.f, zmax = 0.f, zmin = 0.f;
    float *rc = NULL, *gc = NULL, *bc = NULL, *xn = NULL, *yn = NULL, *zn = NULL;
    float *xva = NULL, *yva = NULL, *zva = NULL;
    int *pc = NULL;
    float *x_c = NULL, *y_c = NULL, *z_c = NULL;
    float *t_cp[2] = { NULL, NULL };
    float **t_c = t_cp;
    float transparency = 0.0;
    const char *gtype, *ntype, *ctype, *ttype, *vatype;
    const char *vertexOrderStr, *transparencyStr;
    const char *bindingType, *objName;
    char buf[300];
    const char *tstep_attrib = NULL;
    const char *feedback_info = NULL;
    unsigned int rgba;
    curset = anzset;
    osg::Texture::WrapMode wrapMode = osg::Texture::CLAMP_TO_EDGE;
    osg::Texture::FilterMode minfm = osg::Texture::NEAREST;
    osg::Texture::FilterMode magfm = osg::Texture::NEAREST;

    Rotator *cur_rotator;
    coMaterial *material = NULL;

    int texW = 0, texH = 0; // texture width and height
    int pixS = 0; // size of pixels in texture map (= number of bytes per pixel)
    unsigned char *texImage = NULL; // texture map

//fprintf(stderr, "++++++ObjectManager::addGeometry1  container=%s geometry=%s object =%s\n", container->getName(), geometry->getName(), object );
#ifdef DBGPRINT
    printf("... ObjectManager::addGeometry=s\n", object);
#endif
    gtype = geometry->getType();

    //fprintf ( stderr,"ObjectManager::addGeometry=%s type=%s ......... \n", geometry->getName(), gtype);
    // check for plugin to load
	if (const char* pluginName = geometry->getAttribute("PLUGIN"))
	{
		cover->addPlugin(pluginName);
	}
    if (const char *pluginName = geometry->getAttribute("MODULE"))
    {
        cover->addPlugin(pluginName);
    }

    if (const char *currentLod = geometry->getAttribute("LOD"))
    {
        lod = currentLod;
    }

#if 0
    // check for FRAME_ANGLE attribute
    if (geometry->getAttribute("FRAME_ANGLE") != NULL && geometry->isAssignedToMe())
    {
        const char *bfr = geometry->getAttribute("ROTATION_POINT");
        if (bfr)
        {
            if (sscanf(bfr, "%f %f %f", &(VRSceneGraph::instance()->rotationPoint[0]),
                       &(VRSceneGraph::instance()->rotationPoint[1]),
                       &(VRSceneGraph::instance()->rotationPoint[2])) != 3)
            {
                if (cover->debugLevel(2))
                    cerr << "ObjectManager::addGeometry: sscanf1 failed" << endl;
            }
        }

        bfr = geometry->getAttribute("ROTATION_AXIS");
        if (bfr)
        {
            if (sscanf(bfr, "%f %f %f", &(VRSceneGraph::instance()->rotationAxis[0]),
                       &(VRSceneGraph::instance()->rotationAxis[1]),
                       &(VRSceneGraph::instance()->rotationAxis[2])) != 3)
            {
                if (cover->debugLevel(2))
                    cerr << "ObjectManager::addGeometry: sscanf2 failed" << endl;
            }
        }

        bfr = geometry->getAttribute("FRAME_ANGLE");
        if (bfr)
        {
            if (sscanf(bfr, "%f", &(VRSceneGraph::instance()->frameAngle)) != 1)
            {
                if (cover->debugLevel(2))
                    cerr << "ObjectManager::addGeometry: sscanf3 failed" << endl;
            }
        }
    }
#endif

    if (texture && geometry->isAssignedToMe())
    {
        const char *wm = texture->getAttribute("WRAP_MODE");
        wrapMode = osg::Texture::CLAMP_TO_EDGE;
        if (wm && strncasecmp(wm, "repeat", 6) == 0)
        {
            wrapMode = osg::Texture::REPEAT;
        }
        const char *fm = texture->getAttribute("MIN_FILTER");
        minfm = osg::Texture::NEAREST;
        if (fm && strncasecmp(fm, "nearest", 7) == 0)
        {
            minfm = osg::Texture::NEAREST;
        }
        else if (fm && strncasecmp(fm, "linear", 6) == 0)
        {
            minfm = osg::Texture::LINEAR;
        }
        else if (fm && strncasecmp(fm, "linear_mipmap_linear", 20) == 0)
        {
            minfm = osg::Texture::LINEAR_MIPMAP_LINEAR;
        }
        else if (fm && strncasecmp(fm, "linear_mipmap_nearest", 21) == 0)
        {
            minfm = osg::Texture::LINEAR_MIPMAP_NEAREST;
        }

        fm = texture->getAttribute("MAG_FILTER");
        magfm = osg::Texture::NEAREST;
        if (fm && strncasecmp(fm, "nearest", 7) == 0)
        {
            magfm = osg::Texture::NEAREST;
        }
        else if (fm && strncasecmp(fm, "linear", 6) == 0)
        {
            magfm = osg::Texture::LINEAR;
        }
        else if (fm && strncasecmp(fm, "linear_mipmap_linear", 20) == 0)
        {
            magfm = osg::Texture::LINEAR_MIPMAP_LINEAR;
        }
        else if (fm && strncasecmp(fm, "linear_mipmap_nearest", 21) == 0)
        {
            magfm = osg::Texture::LINEAR_MIPMAP_NEAREST;
        }
    }

    // check for rotate attributes
    if (geometry->getAttribute("ROTATE_POINT") != NULL && geometry->isAssignedToMe())
    {

        // create rotator and pass it to the add... Functions
        // in the GeometryManager in order to keep the
        // rotatorlist up to date

        cur_rotator = new Rotator;
        cur_rotator->feedback_information = NULL;

        if (geometry->getAttribute("FEEDBACK"))
        {
            char *bfr = new char[strlen(geometry->getAttribute("FEEDBACK")) + 1];
            strcpy(bfr, geometry->getAttribute("FEEDBACK"));
            cur_rotator->feedback_information = bfr;
        }

        const char *bfr = geometry->getAttribute("ROTATE_POINT");
        if (sscanf(bfr, "%f %f %f", &(cur_rotator->point[0]),
                   &(cur_rotator->point[1]),
                   &(cur_rotator->point[2])) != 3)
        {
            if (cover->debugLevel(2))
                cerr << "ObjectManager::addGeometry: sscanf4 failed" << endl;
        }

        bfr = geometry->getAttribute("ROTATE_VECTOR");
        if (bfr)
        {
            if (sscanf(bfr, "%f %f %f", &(cur_rotator->vector[0]),
                       &(cur_rotator->vector[1]),
                       &(cur_rotator->vector[2])) != 3)
            {
                if (cover->debugLevel(2))
                    cerr << "ObjectManager::addGeometry: sscanf5 failed" << endl;
            }
        }

        bfr = geometry->getAttribute("ROTATE_ANGLE");
        if (bfr)
        {
            if (sscanf(bfr, "%f", &(cur_rotator->angle)) != 1)
            {
                if (cover->debugLevel(2))
                    cerr << "ObjectManager::addGeometry: sscanf6 failed" << endl;
            }
        }
        else
        {
            cur_rotator->angle = 0.0;
        }
        bfr = geometry->getAttribute("ROTATE_SPEED");
        if (bfr)
        {
            if (sscanf(bfr, "%f", &(cur_rotator->speed)) != 1)
            {
                if (cover->debugLevel(2))
                    cerr << "ObjectManager::addGeometry: sscanf7 failed" << endl;
            }
        }
        else
        {
            cur_rotator->speed = 0.0;
        }

        if ((!added_a_rotate_flag) && (cur_rotator->speed == 0.0))
        {
            added_a_rotate_flag = 1;

            //VRSceneGraph::instance()->add_rotate_controls();
        }
    }
    else
        cur_rotator = NULL;

    bool cullBackfaces = false;
    if (geometry->isAssignedToMe())
    {
        //fprintf(stderr,"ObjectManager::addGeometryif NOT SETELE\n");
        const char *attr = geometry->getAttribute("POINTSIZE");
        if (attr)
        {
            //fprintf ( stderr, "pointsize: %s\n", attr );
            pointsize = atof(attr);
        }
        attr = geometry->getAttribute("CULL_BACKFACES");
        if (attr)
        {
            cullBackfaces = true;
        }
        attr = geometry->getAttribute("LINEWIDTH");
        if (attr)
        {
            //fprintf ( stderr, "linewidth: %s\n", attr );
            linewidth = atof(attr);
        }
        // default (see VRSceneGraph) is view all (-1.0)
        if ((attr = geometry->getAttribute("SCALE")) != NULL)
        {
            if (!strcasecmp(attr, "viewAll"))
                VRSceneGraph::instance()->setScaleMode(-1.0);
            else if (!strcasecmp(attr, "keep"))
                VRSceneGraph::instance()->setScaleMode(0.0);
            else
            {
                float scaleMode = 0.f;
                if (sscanf(attr, "%f", &scaleMode) == 1)
                {
                    VRSceneGraph::instance()->setScaleMode(scaleMode);
                }
                else
                {
                    if (cover->debugLevel(2))
                        cerr << "ObjectManager::addGeometry: sscanf8 failed" << endl;
                }
            }
        }

        feedback_info = geometry->getAttribute("FEEDBACK");
        if (feedback_info && !cur_rotator)
        {
            const char *name = geometry->getAttribute("OBJECTNAME");
            if (container->getAttribute("OBJECTNAME"))
                name = container->getAttribute("OBJECTNAME");
        }
        // check for VertexOrderStr
        vertexOrderStr = geometry->getAttribute("vertexOrder");
        if (vertexOrderStr == NULL)
            vertexOrder = 0;
        else
            vertexOrder = vertexOrderStr[0] - '0';

        // check for Transparency
        transparencyStr = geometry->getAttribute("TRANSPARENCY");
        transparency = 0.0;
        if (transparencyStr != NULL)
        {
            if ((transparency = atof(transparencyStr)) < 0.0)
                transparency = 0.0;
            if (transparency > 1.0)
                transparency = 1.0;
        }
        material = NULL;

        // check for Material
        const char *materialStr = geometry->getAttribute("MATERIAL");
        if (materialStr != NULL)
        {
            if (strncmp(materialStr, "MAT:", 4) == 0)
            {
                char dummy[32];
                char material_name[256];
                float ambientColor[3];
                float diffuseColor[3];
                float specularColor[3];
                float emissiveColor[3];
                float shininess;
                float transparency;

                int ret = sscanf(materialStr, "%s%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f",
                                 dummy, material_name,
                                 &ambientColor[0], &ambientColor[1], &ambientColor[2],
                                 &diffuseColor[0], &diffuseColor[1], &diffuseColor[2],
                                 &specularColor[0], &specularColor[1], &specularColor[2],
                                 &emissiveColor[0], &emissiveColor[1], &emissiveColor[2],
                                 &shininess, &transparency);
                if (ret != 16)
                {
                    if (cover->debugLevel(2))
                        cerr << "ObjectManager::addGeometry: sscanf9 failed" << endl;
                }
                const char *mat_colorStr = geometry->getAttribute("MAT_COLOR");
                if (mat_colorStr != NULL)
                {
                    // change base color of material from white to the given color
                    float r, g, b;
                    int ri, gi, bi;
                    if (sscanf(mat_colorStr, "%d %d %d", &ri, &gi, &bi) != 3)
                    {
                        if (cover->debugLevel(2))
                            cerr << "ObjectManager::addGeometry: sscanf10 failed" << endl;
                    }
                    r = (float)ri / 255.;
                    g = (float)gi / 255.;
                    b = (float)bi / 255.;

                    diffuseColor[0] = r * diffuseColor[0];
                    diffuseColor[1] = g * diffuseColor[1];
                    diffuseColor[2] = b * diffuseColor[2];
                    specularColor[0] = r * specularColor[0];
                    specularColor[1] = g * specularColor[1];
                    specularColor[2] = b * specularColor[2];
                    ambientColor[0] = r * ambientColor[0];
                    ambientColor[1] = g * ambientColor[1];
                    ambientColor[2] = b * ambientColor[2];
                }

                material = new coMaterial(material_name, ambientColor, diffuseColor, specularColor,
                                          emissiveColor, shininess, transparency);
            }
            else
            {
                if (!materialList)
                    materialList = new coMaterialList("metal");

                material = materialList->get(materialStr);
                if (!material)
                {
                    char category[500];
                    if (sscanf(materialStr, "%s", category) != 1)
                    {
                        if (cover->debugLevel(2))
                            cerr << "ObjectManager::addGeometry: sscanf11 failed" << endl;
                    }
                    materialList->add(category);
                    material = materialList->get(materialStr);
                    if (!material)
                    {
                        if (cover->debugLevel(2))
                            fprintf(stderr, "Material %s not found!\n", materialStr);
                    }
                }
            }
        }
    }

    if (strcmp(gtype, "GEOMET") == 0)
    {
        //fprintf(stderr,"ObjectManager::addGeometry if GEOMET\n");
        CoviseRenderObject *dobjg = geometry->getGeometry();
        CoviseRenderObject *dobjn = geometry->getNormals();
        CoviseRenderObject *dobjc = geometry->getColors();
        CoviseRenderObject *dobjt = geometry->getTexture();
        CoviseRenderObject *dobjv = geometry->getVertexAttribute();
        gtype = dobjg->getType();
        // use correct name for container object (necessary for COVER-GUI comunication)
        return addGeometry(object, root, dobjg, dobjn, dobjc, dobjt, dobjv, container, lod);
    }
    else if (strcmp(gtype, "SETELE") == 0)
    {
        std::vector<std::vector<int> > assignments;

        //fprintf(stderr,"ObjectManager::addGeometry if SETELE\n");

        // TODO change all Plugins to user RenderObjects
        auto inter = handleInteractors(container, geometry, normals, colors, texture);
        //fprintf(stderr, "++++++ObjectManager::addGeometry3  container=%s geometry=%s\n", container->getName(), geometry->getName() );
        coVRPluginList::instance()->addObject(container, root, geometry, normals, colors, texture);
        // retrieve the whole set
        dobjsg = (CoviseRenderObject **)geometry->getAllElements(no_elems, assignments);

        // look if it is a timestep series
        tstep_attrib = geometry->getAttribute("TIMESTEP");
        if (tstep_attrib != NULL)
        {
            //fprintf(stderr,"Found TIMESTEP Attrib\n");
            is_animation = true;

            if (const char *tsunit = geometry->getAttribute("TIMESTEPUNIT"))
                coVRAnimationManager::instance()->setTimestepUnit(tsunit);
            if (const char *tsbase = geometry->getAttribute("TIMESTEPBASE"))
                coVRAnimationManager::instance()->setTimestepBase(atof(tsbase));
            if (const char *tsscale = geometry->getAttribute("TIMESTEPSCALE"))
                coVRAnimationManager::instance()->setTimestepScale(atof(tsscale));
        }

        // Uwe Woessner
        feedback_info = geometry->getAttribute("FEEDBACK");
        if (feedback_info && geometry->isAssignedToMe())
        {
            const char *name = geometry->getAttribute("OBJECTNAME");
            if (container->getAttribute("OBJECTNAME"))
                name = container->getAttribute("OBJECTNAME");
        }
        if (normals != NULL)
        {
            ntype = normals->getType();
            if (strcmp(ntype, "SETELE") != 0)
            {
                print_comment(__LINE__, __FILE__, "ERROR: ...did not get a normal set");
            }
            else
            {
                // Get Set
                dobjsn = (CoviseRenderObject **)normals->getAllElements(no_n, assignments);
                if (no_n != no_elems)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: number of normal elements does not match geometry set");
                    no_n = 0;
                }
            }
        }

        if (colors != NULL)
        {
            ctype = colors->getType();

            if (strcmp(ctype, "SETELE") != 0)
            {
                print_comment(__LINE__, __FILE__, "ERROR: ...did not get a color set");
            }
            else
            {

                // Get Set
                dobjsc = (CoviseRenderObject **)colors->getAllElements(no_c, assignments);
                if (no_c != no_elems)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: number of colorelements does not match geometry set");
                    std::cerr << "ERROR: number of colorelements does not match geometry set" << std::endl;

                    no_c = 0;
                }
            }
        }

        if (texture != NULL)
        {
            ttype = texture->getType();
            if (strcmp(ttype, "SETELE") != 0)
            {
                print_comment(__LINE__, __FILE__, "ERROR: ...did not get a texture set");
            }
            else
            {
                // Get Set
                dobjst = (CoviseRenderObject **)texture->getAllElements(no_t, assignments);
                if (no_t != no_elems)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: number of texture-elements does not match geometry set");
                    no_t = 0;
                }
            }
        }

        if (vertexAttribute != NULL)
        {
            vatype = vertexAttribute->getType();
            if (strcmp(vatype, "SETELE") != 0)
            {
                print_comment(__LINE__, __FILE__, "ERROR: ...did not get a vertexAttribute set");
            }
            else
            {
                // Get Set
                dobjsva = (CoviseRenderObject **)vertexAttribute->getAllElements(no_va, assignments);
                if (no_va != no_elems)
                {
                    print_comment(__LINE__, __FILE__, "ERROR: number of normal elements does not match geometry set");
                    no_va = 0;
                }
            }
        }
        if (coVRMSController::instance()->isMaster())
        {
            if (tstep_attrib != NULL)
            {
                CoviseBase::sendInfo("Adding a sequence: %d timesteps", no_elems);
            }
            else
            {
                CoviseBase::sendInfo("Adding a group: %d elements", no_elems);
            }
        }
        setnames[curset] = new char[strlen(object) + 1];
        strcpy(setnames[curset], object);
        elemanz[curset] = no_elems;
        elemnames[curset] = new char *[no_elems];
        if (no_elems <= 1)
        {
            is_animation = false;
            CoviseBase::sendInfo("timesteps <=1 --> static");
        }
        osg::Group* groupNode = nullptr;
        if (container->isAssignedToMe())
        {
            groupNode = GeometryManager::instance()->addGroup(object, is_animation);
            if (cur_rotator)
            {
                cur_rotator->node = groupNode;
                RotatorList::instance()->append(cur_rotator);
            }
        }
        anzset++;
        for (int i = 0; i < no_elems; i++)
        {
            strcpy(buf, dobjsg[i]->getName());
            objName = buf;
            elemnames[curset][i] = new char[strlen(objName) + 1];
            strcpy(elemnames[curset][i], objName);

            //std::cerr << "ObjectManager::addGeometry info: calling addGeometry for " << objName << " (" << i << ")" << std::endl;
            osg::Node *node = addGeometry(objName, groupNode, dobjsg[i],
                                          no_n > 0 ? dobjsn[i] : NULL,
                                          no_c > 0 ? dobjsc[i] : NULL,
                                          no_t > 0 ? dobjst[i] : NULL,
                                          no_va > 0 ? dobjsva[i] : NULL,
                                          container, lod);
            if (groupNode && node)
            {
                groupNode->addChild(node);
            }
            else
            {
                std::cerr << "ignoring Set element " << objName << ": no " << (node ? "" : "group ") << "node" << std::endl;
            }

            if (dobjsg)
                delete dobjsg[i];
            if (dobjsn && i < no_n)
                delete dobjsn[i];
            if (dobjsc && i < no_c)
                delete dobjsc[i];
            if (dobjst && i < no_t)
                delete dobjst[i];
            if (dobjsva && i < no_va)
                delete dobjsva[i];
        }

        const char *polyOffset = geometry->getAttribute("POLYGON_OFFSET");
        if (polyOffset && geometry->isAssignedToMe())
        {
            osg::StateSet *stateset;
            stateset = groupNode->getOrCreateStateSet();
            float factor = atof(polyOffset);
            osg::PolygonOffset *po = new osg::PolygonOffset();
            po->setFactor(factor);
            stateset->setAttributeAndModes(po, osg::StateAttribute::ON);
        }

#if 0
      if (container->isAssignedToMe())
         coviseSG->addNode ( groupNode,root,geometry );
#endif

        // might have to add AnimationSpeed and SteadyCam Feedback to
        // Pinboard
        const char *attr = geometry->getAttribute("MULTIROT");
        if (attr && geometry->isAssignedToMe())
        {
            float degrees;
            int numInstances;
            sscanf(attr, "%d %f", &numInstances, &degrees);
            int i, n;
            for (n = 0; n < numInstances; n++)
            {
                for (i = 0; i < no_elems; i++)
                {
                    osg::MatrixTransform *mt = new osg::MatrixTransform();
                    osg::Matrix m;
                    m.makeRotate((((n + 1) * (45)) / 180.0 / M_PI /*degrees*/), osg::Vec3(0, 0, 1));
                    mt->setMatrix(m);
                    mt->addChild(groupNode->getChild(i));
                    groupNode->addChild(mt);
                }
            }
        }

        if (inter && groupNode)
        {
            //std::cerr << "setting interactor user data on Group " << groupNode->getName() << std::endl;
            groupNode->setUserData(new InteractorReference(inter));
        }
        if (groupNode)
        {
            if (osg::Sequence * pSequence = dynamic_cast<osg::Sequence*>(groupNode)) // timesteps
            {
                coVRAnimationManager::instance()->addSequence(pSequence, coVRAnimationManager::Cycle);
            }
        }


        if (groupNode)
        {
            if (osg::Sequence * pSequence = dynamic_cast<osg::Sequence*>(groupNode)) // timesteps
            {
                coVRAnimationManager::instance()->addSequence(pSequence, coVRAnimationManager::Cycle);
            }
        }

        return groupNode;
    }

    else if (geometry->isAssignedToMe()) // not a set
    {
        if (texture != NULL)
        {
            colorbinding = Bind::None;
            colorpacking = Pack::Texture;
        }
        if (strcmp(gtype, "POLYGN") == 0)
        {
            no_poly = geometry->getNumPolygons();
            no_faces = no_poly;
            no_vert = geometry->getNumVertices();
            no_points = geometry->getNumPoints();
            geometry->getAddresses(x_c, y_c, z_c, v_l, l_l);
        }
        else if (strcmp(gtype, "TRIANG") == 0)
        {
            no_strip = geometry->getNumStrips();
            no_faces = no_strip;
            no_vert = geometry->getNumVertices();
            no_points = geometry->getNumPoints();
            geometry->getAddresses(x_c, y_c, z_c, v_l, l_l);
        }
        else if (strcmp(gtype, "TRITRI") == 0)
        {
            no_vert = geometry->getNumVertices();
            no_faces = no_vert / 3;
            no_points = geometry->getNumPoints();
            geometry->getAddresses(x_c, y_c, z_c, v_l);
        }
        else if (strcmp(gtype, "QUADS") == 0)
        {
            no_vert = geometry->getNumVertices();
            no_faces = no_vert / 4;
            no_points = geometry->getNumPoints();
            geometry->getAddresses(x_c, y_c, z_c, v_l);
        }
        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            cover->addPlugin("Volume");
            geometry->getSize(xsize, ysize, zsize);
            geometry->getMinMax(xmin, xmax, ymin, ymax, zmin, zmax);

#ifdef PHANTOM_TRACKER
/*
            if(ugrid->getAttribute("DataObject"))
            {
            if(feedbackList==NULL)
            {
            feedbackList= new ForceFeedbackList();
            feedbackList->init();
            }
            feedbackList->addData(new ForceData(object,ugrid));
            }
            */
#endif
        }
        else if (strcmp(gtype, "UNSGRD") == 0)
        {
            geometry->getSize(no_points, no_points, no_points);
            geometry->getAddresses(x_c, y_c, z_c, vl, el);
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            geometry->getSize(xsize, ysize, zsize);
            geometry->getAddresses(x_c, y_c, z_c, vl, el);
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            geometry->getSize(xsize, ysize, zsize);
            geometry->getAddresses(x_c, y_c, z_c, vl, el);
        }
        else if (strcmp(gtype, "POINTS") == 0)
        {
            no_points = geometry->getNumPoints();
            geometry->getAddresses(x_c, y_c, z_c, vl, el);
        }
        else if (strcmp(gtype, "SPHERE") == 0)
        {
            geometry->getSize(no_points);
            iRenderMethod = geometry->getRenderMethod();

            geometry->getAddresses(x_c, y_c, z_c, radii_c, type_c);
        }

        else if (strcmp(gtype, "LINES") == 0)
        {
            no_lines = geometry->getNumLines();
            no_faces = no_lines;
            no_vert = geometry->getNumVertices();
            no_points = geometry->getNumPoints();
            geometry->getAddresses(x_c, y_c, z_c, v_l, l_l);
        }
        else if (strcmp(gtype, "USTSDT") == 0)
        {
            if (cover->debugLevel(2))
                fprintf(stderr, "not handled by COVER but maybe plugin\n");
        }
        else
        {
            if (cover->debugLevel(2))
                fprintf(stderr, "++++GTYPE:[%s]\n", gtype);
            print_comment(__LINE__, __FILE__, "ERROR: ...got unknown geometry");
            return NULL;
        }

        if (normals)
        {
            normals->getSize(no_n);
            normals->getAddresses(xn, yn, zn);

            /// now get this attribute junk done
            if (no_faces > 0 && no_faces == no_n)
                normalbinding = Bind::PerFace;
            else if (no_n >= no_points)
                normalbinding = Bind::PerVertex;
            else if (no_n > 1 && no_n >= no_faces)
                normalbinding = Bind::PerFace;
            else if (no_n == 1)
                normalbinding = Bind::OverAll;
            else
                normalbinding = Bind::None;
        }

        if (vertexAttribute)
        {
            vertexAttribute->getSize(no_va);
            vertexAttribute->getAddresses(xva, yva, zva);
        }
        if (texture) // colors by texture map
        {
            if (vertexAttribute == NULL) // if we have vertex attributes, allow color and texture to be mixed, otherwise change to
            {
                colorpacking = Pack::None;
                colorbinding = Bind::None;
            }

            texImage = texture->texture;

            no_t = texture->numTC;

            // a Texture with 0 coordinates is as good as no texture object
            // can occurr e.g. with dummy objects in pipeline
            if (no_t > 0)
            {
                texture->getSize(texW, texH, pixS);
                t_c = texture->textureCoords;
                if (vertexAttribute == NULL) // if we have vertex attributes, allow color and texture to be mixed, otherwise change to
                {
                    colorbinding = Bind::PerVertex;
                    colorpacking = Pack::Texture;
                }
            }
        }
        if (colors && (!texture || vertexAttribute != NULL))
        {

            colors->getSize(no_c);
            ctype = colors->getType();
            if (strcmp(ctype, "STRVDT") == 0)
            {
                colors->getSize(no_c);
                colors->getAddresses(rc, gc, bc);
                colorpacking = Pack::None;
            }
            else if (strcmp(ctype, "USTSTD") == 0)
            {
                colors->getSize(no_c);
                colors->getAddresses(rc, gc, bc);
                bc = NULL;
                colorpacking = Pack::None;
            }
            else if (strcmp(ctype, "USTVDT") == 0)
            {
                colors->getSize(no_c);
                colors->getAddresses(rc, gc, bc);
                colorpacking = Pack::None;
            }
            else if (strcmp(ctype, "RGBADT") == 0)
            {
                colors->getSize(no_c);
                pc = colors->pc;
                colorpacking = Pack::RGBA;
            }
            else if (strcmp(ctype, "STRSDT") == 0)
            {
                colors->getSize(no_c);
                colors->getAddresses(rc, gc, bc);
                gc = NULL;
                bc = NULL;
                colorpacking = Pack::Float;
            }
            else if (strcmp(ctype, "USTSDT") == 0)
            {
                colors->getSize(no_c);
                colors->getAddresses(rc, gc, bc);
                gc = NULL;
                bc = NULL;
                colorpacking = Pack::Float;
            }
            else
            {
                colorbinding = Bind::None;
                colorpacking = Pack::None;
                no_c = 0;
                print_comment(__LINE__, __FILE__, "ERROR: DataTypes other than structured and unstructured are not yet implemented");
                //   sendError("ERROR: DataTypes other than structured and unstructured are not jet implemented");
            }

            /// now get this attribute junk done
            if (no_c == 0)
            {
                // sendWarning("WARNING: Data object 'Color' is empty");
                colorbinding = Bind::None;
                colorpacking = Pack::None;
            }
            else if (no_faces > 0 && no_faces == no_c)
                colorbinding = Bind::PerFace;
            else if (no_c >= no_points)
                colorbinding = Bind::PerVertex;
            else if (no_c > 1 && no_c >= no_faces)
                colorbinding = Bind::PerFace;
            else if (no_c == 1)
                colorbinding = Bind::OverAll;
            else
                colorbinding = Bind::None;
        }
        else //if(container==NULL) // we got an object without colors
        {
            bindingType = geometry->getAttribute("COLOR");
            if (bindingType != NULL)
            {
                colorbinding = Bind::OverAll;
                colorpacking = Pack::RGBA;
                no_c = 1;
                // open ascii file for color names
                if (!isopen)
                {
                    const char *fileName = coVRFileManager::instance()->getName("share/covise/rgb.txt");
                    if (fileName)
                    {
                        fp = fopen(fileName, "r");
                    }
                    if (fp != NULL)
                        isopen = 1;
                }
                if (isopen)
                {
                    rgba = create_named_color(bindingType);
                    pc = (int *)&rgba;
                }
            }
        }

        //
        // add object to VRSceneGraph::instance() depending on type
        //
        // TODO Change all Plugins
        coInteractor *inter = handleInteractors(container, geometry, normals, colors, texture);
        coVRPluginList::instance()->addObject(container, root, geometry, normals, colors, texture);

        osg::Node *newNode = NULL;

        bool skipGeometryCreation = false;

        if (!skipGeometryCreation)
        {
            if (strcmp(gtype, "UNIGRD") == 0)
            {
                newNode = GeometryManager::instance()->addUGrid(object, xsize, ysize, zsize, xmin, xmax, ymin, ymax, zmin, zmax,
                                                                no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                no_n, normalbinding, xn, yn, zn, transparency);
                if (!newNode)
                    cover->addPlugin("Volume");
            }
            else if (strcmp(gtype, "RCTGRD") == 0)
                newNode = GeometryManager::instance()->addRGrid(object, xsize, ysize, zsize, x_c, y_c, z_c,
                                                                no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                no_n, normalbinding, xn, yn, zn, transparency);
            else if (strcmp(gtype, "STRGRD") == 0)
                newNode = GeometryManager::instance()->addSGrid(object, xsize, ysize, zsize, x_c, y_c, z_c,
                                                                no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                no_n, normalbinding, xn, yn, zn, transparency);
            else if (strcmp(gtype, "POLYGN") == 0)
                newNode = GeometryManager::instance()->addPolygon(object, no_poly, no_vert,
                                                                  no_points, x_c, y_c, z_c,
                                                                  v_l, l_l,
                                                                  no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                  no_n, normalbinding, xn, yn, zn, transparency,
                                                                  vertexOrder, material,
                                                                  texW, texH, pixS, texImage,
                                                                  no_t, t_c[0], t_c[1], wrapMode, minfm, magfm, no_va, xva, yva, zva, cullBackfaces);
            else if (strcmp(gtype, "TRIANG") == 0)
                newNode = GeometryManager::instance()->addTriangleStrip(object, no_strip, no_vert,
                                                                        no_points, x_c, y_c, z_c,
                                                                        v_l, l_l,
                                                                        no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                        no_n, normalbinding, xn, yn, zn, transparency,
                                                                        vertexOrder, material,
                                                                        texW, texH, pixS, texImage,
                                                                        no_t, t_c[0], t_c[1], wrapMode, minfm, magfm, no_va, xva, yva, zva, cullBackfaces);
            else if (strcmp(gtype, "TRITRI") == 0)
                newNode = GeometryManager::instance()->addTriangles(object, no_vert,
                                                                    no_points, x_c, y_c, z_c,
                                                                    v_l,
                                                                    no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                    no_n, normalbinding, xn, yn, zn, transparency,
                                                                    vertexOrder, material,
                                                                    texW, texH, pixS, texImage,
                                                                    no_t, t_c[0], t_c[1], wrapMode, minfm, magfm, no_va, xva, yva, zva, cullBackfaces);
            else if (strcmp(gtype, "QUADS") == 0)
                newNode = GeometryManager::instance()->addTriangles(object, no_vert,
                                                                    no_points, x_c, y_c, z_c,
                                                                    v_l,
                                                                    no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                    no_n, normalbinding, xn, yn, zn, transparency,
                                                                    vertexOrder, material,
                                                                    texW, texH, pixS, texImage,
                                                                    no_t, t_c[0], t_c[1], wrapMode, minfm, magfm, no_va, xva, yva, zva, cullBackfaces);
            else if (strcmp(gtype, "LINES") == 0)
            {
                // feedback_info + line  -> trace
                int isTrace = false;
                if (feedback_info)
                    isTrace = true;

                newNode = GeometryManager::instance()->addLine(object, no_lines, no_vert, no_points,
                                                               x_c, y_c, z_c, v_l, l_l, no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                               no_n, normalbinding, xn, yn, zn, isTrace, material,
                                                               texW, texH, pixS, texImage, no_t, t_c[0], t_c[1], wrapMode, minfm, magfm,
                                                               linewidth);
            }
            //else if ( ( strcmp ( gtype,"POINTS" ) == 0 ) || ( strcmp ( gtype,"UNSGRD" ) == 0 ) )
            else if ((strcmp(gtype, "POINTS") == 0))
            {
                const char *var = container ? container->getAttribute("VARIANT") : 0;
                if (!var || strncmp(var, "GPGPU", 5))
                    newNode = GeometryManager::instance()->addPoint(object, no_points,
                                                                    x_c, y_c, z_c, no_c, colorbinding, colorpacking, rc, gc, bc, pc,
                                                                    material,
                                                                    texW, texH, pixS, texImage, no_t, t_c[0], t_c[1], wrapMode, minfm, magfm,
                                                                    pointsize);
            }
            else if (strcmp(gtype, "SPHERE") == 0)
            {
                newNode = GeometryManager::instance()->addSphere(object, no_points,
                                                                 x_c, y_c, z_c, iRenderMethod, radii_c, colorbinding, rc, gc, bc, pc, no_n, xn, yn, zn, no_va, xva, yva, zva, material);
            }
        }

        if (newNode)
        {
            if(depthPeeling)
            {
               /* osg::Group *g = new osg::Group;
                coVRDePee *dp = new coVRDePee(g,newNode);
                newNode = g;*/
            }
            if (cur_rotator)
            {
                cur_rotator->node = newNode;
                RotatorList::instance()->append(cur_rotator);
            }
            //fprintf(stderr,"newNode\n");
            const char *attr = geometry->getAttribute("BOUNDING_BOX");
            if (!attr)
                attr = container->getAttribute("BOUNDING_BOX");
            if (attr)
            {
                osg::BoundingSphere bs;
                bool ignore = true;
                if (strcmp(attr, "ignore"))
                {
                    float xMin = 0.f, yMin = 0.f, zMin = 0.f, xMax = 0.f, yMax = 0.f, zMax = 0.f;
                    if (sscanf(attr, "%f %f %f %f %f %f",
                               &xMin, &yMin, &zMin,
                               &xMax, &yMax, &zMax) != 6)
                    {
                        CoviseBase::sendInfo("failed to read bounding box");
                    }
                    else
                    {
                        bs.expandBy(osg::Vec3(xMin, yMin, zMin));
                        bs.expandBy(osg::Vec3(xMax, yMax, zMax));
                        ignore = false;
                    }
                }

                if (!ignore)
                {
                    VRSceneGraph::instance()->setNodeBounds(newNode, &bs);
                }
            }

            attr = geometry->getAttribute("DEPTH_ONLY");
            if (!attr)
                attr = container->getAttribute("DEPTH_ONLY");
            if (attr)
            {
                osg::StateSet *stateset;
                stateset = newNode->getOrCreateStateSet();
                stateset->setRenderBinDetails(-1, "RenderBin");
                stateset->setNestRenderBins(false);
                stateset->setAttributeAndModes(noFrameBuffer, osg::StateAttribute::ON);
            }
            if (const char *polyOffset = geometry->getAttribute("POLYGON_OFFSET"))
            {
                osg::StateSet *stateset;
                stateset = newNode->getOrCreateStateSet();
                float factor = atof(polyOffset);
                osg::PolygonOffset *po = new osg::PolygonOffset();
                po->setFactor(factor);
                stateset->setAttributeAndModes(po, osg::StateAttribute::ON);
            }
            const char *shaderName = geometry->getAttribute("SHADER");
            if (texture != NULL && !shaderName)
                shaderName = texture->getAttribute("SHADER");
            if (!shaderName)
                shaderName = container->getAttribute("SHADER");
            if (shaderName || colorpacking==Pack::Float)
            {
                coVRShader *shader = NULL;
                if (shaderName)
                {
                    shader = coVRShaderList::instance()->get(shaderName);
                }
                else if (colorpacking == Pack::Float)
                {
                    const char *sp= colors->getAttribute("SPECIES");
                    std::string species;
                    if (sp)
                        species = sp;
                    const ColorMap &cm = getColorMap(species);
                    shader = cm.textureMapShader;
                    osg::StateSet *stateset = newNode->getOrCreateStateSet();
                    stateset->setTextureAttributeAndModes(1, cm.tex, osg::StateAttribute::ON);
                }
                if (shader)
                {

                    const char *uniformValues = geometry->getAttribute("UNIFORMS");
                    if (texture && uniformValues == NULL)
                    {
                        uniformValues = texture->getAttribute("UNIFORMS");
                    }
                    if (uniformValues)
                    {
                        shader->setUniformesFromAttribute(uniformValues);
                    }
                    shader->apply(newNode);

                    // add attributes
                }
                else
                {
                    cerr << "ERROR: no shader found with name:" << shaderName << endl;
                }
            }
            const char *modelName = geometry->getAttribute("MODEL_FILE");
            //fprintf(stderr, "modelName=%s\n", modelName?modelName:"(null)");
            // check for additional Model to load
            if (modelName)
            {

                // SceneGraphItems startID
                const char *startIndex = geometry->getAttribute("SCENEGRAPHITEMS_STARTINDEX");
                if (startIndex)
                {
                    VRRegisterSceneGraph::instance()->setRegisterStartIndex(atoi(startIndex));
                }
                osg::Node *modelNode = NULL;

                const char *modelPath = geometry->getAttribute("MODEL_PATH");
                if (modelPath)
                {
                    std::string tmpName = std::string(modelPath) + "/" + modelName;
                    if(!coVRFileManager::instance()->findFile(tmpName).empty())
                    {
                        modelNode = coVRFileManager::instance()->loadFile(tmpName.c_str(), NULL, NULL, geometry->getName());
                        coviseSG->attachNode(object, modelNode, tmpName.c_str());
                    }
                }
                else if(!coVRFileManager::instance()->findFile(modelName).empty())
                {
                    modelNode = coVRFileManager::instance()->loadFile(modelName, NULL, NULL, geometry->getName());
                    coviseSG->attachNode(object, modelNode, modelName);
                }
            }
            /*.----------------------------------------------------------------------------------------------------------------------------
           Attributes for read and modifying CAD-Datafiles, e.g. JT-Data-Files
           ------------------------------------------------------------------------*/
            const char *CAD_FILE = geometry->getAttribute("CAD_FILE");
            //fprintf(stderr, "modelName=%s\n", modelName?modelName:"(null)");
            // check for additional Model to load
            if (CAD_FILE)
            {
                cout << "LoadCADData: Reading CAD-File: " << CAD_FILE << endl;

                osg::MatrixTransform *mt = new osg::MatrixTransform; //New MatrixTransform Object
                osg::Matrix matrix;

                /*Building the modifying Matrix: First translation, then rotation and last resizing */

                if (const char *resize = geometry->getAttribute("RESIZE_OBJECT"))
                {
                    float sc_x, sc_y, sc_z;
                    sscanf(resize, "%f %f %f", &sc_x, &sc_y, &sc_z);
                    cout << endl << "CAD File is scaled by: " << sc_x << "-" << sc_y << "-" << sc_z << endl << endl;
                    matrix = osg::Matrix::scale(sc_x, sc_y, sc_z);
                    mt->setMatrix(matrix);
                }
                if (const char *rotation = geometry->getAttribute("ROTATE_OBJECT"))
                {
                    const char *rotangle = geometry->getAttribute("ROTANGLE_OBJECT");
                    float alpha, u, v, w;
                    sscanf(rotation, "%f %f %f", &u, &v, &w);
                    sscanf(rotangle, "%f", &alpha);
                    cout << endl << "CAD File is rotated by: " << u << "-" << v << "-" << w << endl << endl;
                    matrix = matrix * osg::Matrix::rotate(alpha, u, v, w);
                    mt->setMatrix(matrix);
                }
                if (const char *translation = geometry->getAttribute("TRANSLATE_OBJECT"))
                {
                    float x, y, z;
                    sscanf(translation, "%f %f %f", &x, &y, &z);
                    cout << endl << "CAD File is translated by: " << x << "-" << y << "-" << z << endl << endl;
                    matrix = matrix * osg::Matrix::translate(x, y, z);
                    mt->setMatrix(matrix);
                }

                /*setting the name of the modification matrix <mt>*/
                string CAD_F(CAD_FILE);
                int numchild = cover->getObjectsRoot()->getNumChildren();
                char buf[1024];
                sprintf(buf, "_%i", numchild);
                string objName(CAD_F.substr(CAD_F.rfind("/") + 1, CAD_F.size() - CAD_F.rfind("/")));
                //if the Name is already used append numbering
                int a = 0;
                for (int index = 0; index < numchild; index++)
                {
                    string newobjName(cover->getObjectsRoot()->getChild(index)->getName());
                    newobjName = newobjName.substr(0, newobjName.rfind("."));
                    string oldobjName(objName.substr(0, objName.rfind(".")));
                    if (oldobjName == newobjName)
                    {
                        a++;
                    }
                }
                if (a == 0)
                {
                    mt->setName(objName);
                }
                else
                {
                    sprintf(buf, "%i", a);
                    mt->setName(objName += buf);
                }
                // new Node <mt> under ObjectRoot
                cover->getObjectsRoot()->addChild(mt);
                //reading CAD-File and adding the new CAD-File structure under the <mt> node
                coVRFileManager::instance()->loadFile(CAD_FILE, NULL, mt, geometry->getName());
                //attaching <mt> with <object> to get the correct name of the node in the SceneGraphBrowser
                coviseSG->attachNode(object, mt, CAD_FILE);
            }
            //..----------------------------------------------------------------------------------------------------------------------------------
            //------------------------------------------------------------------------------------------------------------------------------------
            if (const char *label = geometry->getAttribute("LABEL"))
            {
                coviseSG->attachLabel(object, label);
            }

            // ---------------------------------------------------------------------------------
            /*  if(char *translation=geometry->getAttribute("TRANSLATE_OBJECT"))
             {

             float x,y,z;
             sscanf(translation,"%f %f %f",&x,&y,&z);
             cout<<endl<<"Gottlieb-----------------------Translationsattribute gefunden!:" << x << "-" << y << "-"  << z <<endl<<endl;
             osg::MatrixTransform *mt = new osg::MatrixTransform;
             mt->setMatrix(osg::Matrix::translate(x,y,z));
             mt->addChild(newNode);

         // newNode->getParent(0)->addChild(mt);

         newNode = mt;


         }*/
            //------------------------------------------------------------------------------------
            SliderList::instance()->add(geometry, newNode);
            vectorList.add(geometry, newNode);
            tuiParamList.add(geometry, newNode);

            if (lod)
            {
                float min, max;
                if (sscanf(lod, "%f %f", &min, &max) == 2)
                {
                    coLOD *lodNode = new coLOD();
                    lodNode->addChild(newNode, min, max);
                    lodNode->setName(newNode->getName());
                    newNode->setName(newNode->getName() + "_LOD");
                    newNode = lodNode;
                }
            }

            bool addNode = coVRMenuList::instance()->add(geometry, newNode);
            if (addNode)
            {
                if (inter && newNode)
                {
                    //std::cerr << "setting interactor user data on Node " << newNode->getName() << std::endl;
                    newNode->setUserData(new InteractorReference(inter));
                }
                return newNode;
            }
        }

        return NULL;
        //else
        //   fprintf(stderr,"!newNode\n");
    }
    //fprintf(stderr,"........ObjectManager::addGeometry=%s type=%s done\n", object, gtype );

    return NULL;
}
