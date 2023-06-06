/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#include <winsock2.h>
#include <windows.h>
#include <direct.h>
#endif
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRAnimationManager.h>
#include <PluginUtil/coSphere.h>
#include <cover/coVRTui.h>
#include <config/CoviseConfig.h>
#include "vranimPlugin.h"

#include <osg/GL>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osgDB/ReadFile>
#include <osg/Program>
#include <osg/Shader>
#include <osg/Point>
#include <osg/ShadeModel>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>

#include <osgGA/GUIEventAdapter>

#ifndef WIN32
// for chdir
#include <unistd.h>
#endif

#ifdef _WINDOWS
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

MultiBodyPlugin *MultiBodyPlugin::plugin_ = NULL;
int MultiBodyPlugin::debugLevel_ = 0;
double MultiBodyPlugin::updatetime_ = -999.0;
extern "C" {
}
/* ------------------------------------------------------------------ */
/* Definition of global variables                                     */
/* ------------------------------------------------------------------ */
/* geometry data */
struct geometry geo = { GL_TRUE, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                        NULL, NULL, 0, NULL, NULL, NULL, NULL, NULL };

/* inventor files for VR */
struct ivfiles iv = { 0, NULL };

/* animation data */
struct animation str = { GL_TRUE, 0, 1, 0.0, 0.0, 0, GL_FALSE, NULL, NULL, NULL, NULL };

/* sensor data */
struct sensors sensor = { 0, NULL, NULL, NULL, NULL };

/* plotter's appearence */
struct plotter plo = { 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, { 0, 0, 0, 0, 0, 0 }, { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } }, { 0, 0, 0, 0, 0, 0 } };

/* plotter's information */
struct plotdata dat = { GL_TRUE, 0, 1, 0.0, 0, 0, NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL };

/* elastic geometry data */
struct elgeometry elgeo = { GL_TRUE, 0, 0, NULL, NULL, NULL, NULL, NULL,
                            NULL, NULL, NULL, NULL };

/*line element data*/
struct lineelem lin = { 0, 0, NULL, NULL, NULL, NULL, NULL, NULL,
                        NULL, NULL, NULL, NULL, NULL };

/* colors */
float colorindex[MAXCOLORS + MAXNODYNCOLORS + 1][4];

/* dynamic color information */
struct dyncolor dyn = { NULL, NULL };

struct menuentries menus = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             NULL, NULL, NULL };

struct flags flag = {
    GL_FALSE, /* oitl_animation */
    GL_FALSE, /* simula */
    0, /* video */
    GL_FALSE, /* leftmbut */
    GL_FALSE /* middlembut */
};

struct modes mode = {
    ANIM_AUTO, /* anim */
    ZOOM, /* mmb */
    SHADE_OFF, /* shade */
    SHADE_WIRE, /* shade_el */
    GL_TRUE, /* toggle: rigid */
    1, /* selected plotter */
    MVGEO, /* move geometry */
    GL_TRUE, /* toggle: display timestep */
    GL_FALSE, /* toggle: show coordinate systems */
    0.0 /* scaling coordinate system */
};

FILE *outfile = stderr;

/* global viewing transformation (incl. scaling) */
float globalmat[17] = { 999, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
/* plotter viewing transformation */
float plotmat[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

extern int *lenList_plotterline;

/* ------------------------------------------------------------------ */
/* Prototypes                                                         */
/* ------------------------------------------------------------------ */
/* functions defined in read.c */
int anim_read_geo_file(char *);
int anim_read_str_file(char *);
int anim_read_trmat_file(char *);
int anim_read_sns_file(char *);
int anim_read_cmp_file(char *);
int anim_read_lig_file(char *);
int anim_read_set_file(char *);
int anim_read_data_file(char *);
int anim_read_dyncolor_file(char *);
int anim_read_elgeo_file(char *);
int anim_read_lin_file(char *);

/* functions defined in plot.c */
void anim_ini_plotter(void);
void anim_ini_viewmat(float *);
void anim_draw_plotter(void);

/* functions defined in auxil.c */
int index_to_color_frame(int);
int index_to_color_polygon(int);
double gettime(void);
void minvert(float *, float *);
void transback(float *, float *);
void matmult(anim_vector, float *, anim_vector);
void mult(float *, float *, float *);
void vcopy(float *, float *);
void save_transmat(char *);
void save_frame(int, int);
void upk_transformation(void);
void writeileaffile(void);
void output_defines(FILE *file);

static FileHandler handlers[] = {
    { NULL,
      MultiBodyPlugin::loadDyn,
      MultiBodyPlugin::unloadDyn,
      "dyn" },
    { NULL,
      MultiBodyPlugin::loadDyn,
      MultiBodyPlugin::unloadDyn,
      "geoall" },
    { NULL,
      MultiBodyPlugin::loadDyn,
      MultiBodyPlugin::unloadDyn,
      "str" },
    { NULL,
      MultiBodyPlugin::loadDyn,
      MultiBodyPlugin::unloadDyn,
      "sensor" }
};

int MultiBodyPlugin::loadDyn(const char *filename, osg::Group *parent, const char *)
{

    cerr << "Read the file " << filename << endl;
    char *dir = new char[strlen(filename) + 1];
    strcpy(dir, filename);
    char *basename = new char[strlen(filename) + 1];
    char *c = dir + strlen(dir);
    while (c >= dir)
    {
        if (*c == '/' || *c == '\\')
        {
            *c = '\0';
            strcpy(basename, c + 1);
            break;
        }
        c--;
    }
    if (c <= dir)
    {
        dir[0] = '\0';
        strcpy(basename, filename);
    }
    c = basename + strlen(basename);
    while (c >= basename)
    {
        if (*c == '.')
        {
            *c = '\0';
            break;
        }
        c--;
    }
    return plugin_->loadFile(dir, basename, parent);
}

int MultiBodyPlugin::unloadDyn(const char *filename, const char *)
{

    cerr << "unload the file " << filename << endl;
    return plugin_->unloadFile();
}

int MultiBodyPlugin::loadFile(const char *dir, const char *basename, osg::Group *parent)
{
    (void)parent;

    char currentPath[1024];
    GetCurrentDir(currentPath, sizeof(currentPath));

    cerr << "Set of files is " << basename << endl;
    cerr << "Working directory is " << dir << endl;

    // read current working directory
    const char *cwdName;
    cwdName = (const char *)dir;
    if (cwdName != NULL)
    {
        char *cwd_filename = new char[strlen(cwdName) + 1];
        strcpy(cwd_filename, cwdName);
//fprintf(stderr, "WorkingDirectory read as >>%s<<\n",
//						cwd_filename);
#ifndef WIN32
        chdir(cwd_filename);
#else
        _chdir(cwd_filename);
#endif
    }
    else
    {
        char *cwd_filename = NULL;
        size_t size = 0;
#ifdef WIN32
        cwd_filename = _getcwd(NULL, size); // interne Allokierung
#else
        cwd_filename = getcwd(NULL, size); // interne Allokierung
#endif
        fprintf(stderr, "Could not read WorkingDirectory\n");

        fprintf(stderr,
                "  -> program is run in the directory >>%s<< on both computers\n",
                cwd_filename); // geht so noch nicht!
        // chdir("~");
        free(cwd_filename);
    }

    // read set of files
    const char *setofFilesName;
    setofFilesName = (const char *)basename;
    if (setofFilesName != NULL)
    {
        char *sof_filename = new char[strlen(setofFilesName) + 1];
        strcpy(sof_filename, setofFilesName);
        //fprintf(stderr,"SetofFiles read as >>%s<<\n",
        //sof_filename);
        anim_read_set_file(sof_filename);
        vranimroot_->setName(setofFilesName);
    }
    else
    {
        fprintf(stderr, "Could not read SetofFiles\n");
    }

    // set colors
    int numColors = MAXNODYNCOLORS + MAXCOLORS;
    for (int i = 0; i < numColors; i++)
    {
        colArr->push_back(osg::Vec4(colorindex[i][0], colorindex[i][1], colorindex[i][2], colorindex[i][3]));
    }

    // set global trafo if available (read from viewing.mat in read.cpp)
    if (globalmat[0] != 999)
    {
        osg::Matrix m_mat;
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m_mat(i, j) = globalmat[4 * i + j];
            }
        }
        cover->getObjectsXform()->setMatrix(m_mat);
        //cover->setXformMat(m_mat);
        cover->setScale(globalmat[16]);
    }

    // covise.config does not exist for opencover,
    // so we cannot set additional geofile names,
    // color maps, animation files, stride and dt=timeint

    str.act_step = 0;

    // Update animation frame:
    coVRAnimationManager::instance()->setNumTimesteps(str.timesteps, this);

    scenegr_create();
    menus_create();
    vranimtab_create();

#ifndef WIN32
    chdir(currentPath);
#else
    _chdir(currentPath);
#endif

    return 1;
}

int MultiBodyPlugin::unloadFile()
{

    return 0;
}

//------------------------------------------------------------------------------
void MultiBodyPlugin::key(int type, int keySym, int mod)
{
    if (type == osgGA::GUIEventAdapter::KEYDOWN)
    {
        //fprintf(stdout,"--- coVRKey called (KeyPress, keySym=%d, mod=%d)\n",
        //	keySym,mod);
        return;
        //}else{
        //fprintf(stdout,"--- coVRKey called (KeyRelease, keySym=%d)\n",keySym);
    }

    switch (keySym)
    {

    case ('r'): /* r: reset animation */
        mode.anim = ANIM_OFF;
        reset_timestep();
        break;

    case ('s'): // s
        mode.anim = ANIM_OFF;
        break;

    case ('o'): // o
        for (int i = 0; i < geo.nfiles; i++)
        {
            // show bodies
            vranimbodywireSwitch_[i]->setAllChildrenOff();
            vranimbodygeoSwitch_[i]->setAllChildrenOn();
        }
        break;

    case ('w'): // w
        // show wire frames
        for (int i = 0; i < geo.nfiles; i++)
        {
            vranimbodywireSwitch_[i]->setAllChildrenOn();
            vranimbodygeoSwitch_[i]->setAllChildrenOff();
        }
        break;

    case ('g'): // g
        mode.anim = ANIM_AUTO;
        break;

    case ('i'): //i
        str.stride *= -1;
        break;

    case ('d'): // d
        mode.anim = ANIM_STEP;
        break;

    case (' '): /* space key */
        if (mode.anim == ANIM_STEP)
        {
            str.act_step = str.act_step + str.stride;
            if ((str.act_step >= str.timesteps) || (str.act_step < 0))
            {
                reset_timestep();
            }
        }
        break;
    }
}

//-----------------------------------------------------------------------------
MultiBodyPlugin::MultiBodyPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    plugin_ = this;
    vranimTab = NULL;
}

bool MultiBodyPlugin::init()
{
    //fprintf(stdout,"--- MultiBodyPlugin::MultiBodyPlugin called\n");
    //#ifdef VERBOSE
    fprintf(stderr, "\n\n\n********************************************\n");
    fprintf(stderr, "***                                      ***\n");
    fprintf(stderr, "***                VRANIM                ***\n");
    fprintf(stderr, "***                ------                ***\n");
    fprintf(stderr, "***                                      ***\n");
    fprintf(stderr, "***  based on ANIM Version : " VERSION "         ***\n");
    fprintf(stderr, "***                                      ***\n");
    fprintf(stderr, "***  by P.Eberhard                        ***\n");
    fprintf(stderr, "***     ...                              ***\n");
    fprintf(stderr, "***     F.Fleissner (V 3.6 -      )     ***\n");
    fprintf(stderr, "***     Uwe Woessner(OpenSceneGraph Port)***\n");
    fprintf(stderr, "***                                      ***\n");
    fprintf(stderr, "********************************************\n\n\n");
    //#endif
    plotItem = NULL;
    multiBodyMenuButton_ = NULL;
    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
    coVRFileManager::instance()->registerFileHandler(&handlers[1]);
    coVRFileManager::instance()->registerFileHandler(&handlers[2]);
    coVRFileManager::instance()->registerFileHandler(&handlers[3]);
    //----------------------------------------------------------------------------

    hint = new osg::TessellationHints();
    float ratio = coCoviseConfig::getFloat("COVER.Plugin.VRAnim.DetailRatio", 0.1);
    hint->setDetailRatio(ratio);

    /* 64 colors (0 is background color default black) */
    int counter = 0;
    for (int i = 0; i < 4; i++)
    {
        for (int ii = 0; ii < 4; ii++)
        {
            for (int iii = 0; iii < 4; iii++)
            {
                colorindex[counter][0] = i * 0.3333;
                colorindex[counter][1] = ii * 0.3333;
                colorindex[counter][2] = iii * 0.3333;
                colorindex[counter][3] = 1;
                counter++;
            }
        }
    }

    /* dynamic colors (default red) */
    for (int i = 0; i < MAXNODYNCOLORS; i++)
    {
        colorindex[MAXCOLORS + i][0] = 1;
        colorindex[MAXCOLORS + i][1] = 0;
        colorindex[MAXCOLORS + i][2] = 0;
        colorindex[MAXCOLORS + i][3] = 1;
    }

    colArr = new osg::Vec4Array;

    globalmtl = new osg::Material;
    globalmtl->ref();
    globalmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    globalmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
    globalmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    globalmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    globalmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
    globalmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    unlightedStateSet = new osg::StateSet();
    unlightedStateSet->ref();
    unlightedStateSet->setAttributeAndModes(globalmtl.get(), osg::StateAttribute::ON);
    unlightedStateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    unlightedStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    unlightedStateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    unlightedStateSet->setNestRenderBins(false);
    osg::ShadeModel *shadeModel = new osg::ShadeModel;
    shadeModel->setMode(osg::ShadeModel::FLAT);
    unlightedStateSet->setAttributeAndModes(shadeModel, osg::StateAttribute::ON);

    flatStateSet = new osg::StateSet();
    flatStateSet->ref();
    flatStateSet->setAttributeAndModes(globalmtl.get(), osg::StateAttribute::ON);
    flatStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    flatStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    flatStateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    flatStateSet->setNestRenderBins(false);
    shadeModel = new osg::ShadeModel;
    shadeModel->setMode(osg::ShadeModel::FLAT);
    flatStateSet->setAttributeAndModes(shadeModel, osg::StateAttribute::ON);

    shadedStateSet = new osg::StateSet();
    shadedStateSet->ref();
    shadedStateSet->setAttributeAndModes(globalmtl.get(), osg::StateAttribute::ON);
    shadedStateSet->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    shadedStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    //for transparency, we need a transparent bin
    shadedStateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    shadedStateSet->setNestRenderBins(false);
    shadeModel = new osg::ShadeModel;
    shadeModel->setMode(osg::ShadeModel::SMOOTH);
    shadedStateSet->setAttributeAndModes(shadeModel, osg::StateAttribute::ON);

    if (cover->debugLevel(3))
    {
        fprintf(stderr, "MultiBodyPlugin::MultiBodyPlugin\n");
    }

    if (coVRMSController::instance()->isMaster())
    {
        MultiBodyPlugin::debugLevel_ = coCoviseConfig::getInt("COVER.Plugin.VRAnim.MultiBody.DebugLevel", 0);
    }

    vranimroot_ = new osg::Group;
    cover->getObjectsRoot()->addChild(vranimroot_.get());

    return true;
}

//------------------------------------------------------------------------------
MultiBodyPlugin::~MultiBodyPlugin()
{
}

bool MultiBodyPlugin::destroy()
{

    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[1]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[2]);
    coVRFileManager::instance()->unregisterFileHandler(&handlers[3]);

    cover->getObjectsRoot()->removeChild(vranimroot_.get());

    scenegr_delete();
    menus_delete();
    vranimtab_delete();

    return true;
}

//------------------------------------------------------------------------------
void MultiBodyPlugin::preFrame()
{
    static int updatedstep = -999;

    // update only when necessary
    if (updatedstep != str.act_step)
    {
        update();
        updatedstep = str.act_step;
    }

    // pe: check intersection
    //
    //   cover->intersectObjects(true);
    //   pfNode *tmp=NULL;
    //   tmp=cover->getIntersectedNode();
    //   if(tmp !=NULL){
    //     for (int i=0; i<geo.nfiles; i++){
    //       if(tmp == vranimbodyGeometry_[i]){
    // 	printf("cover->getIntersectedNode() hits body %d\n",i);
    // 	vranimbodygeoSwitch_[i]->setAllChildrenOff(); // hide body
    //       }
    //     }
    //   }
}

//------------------------------------------------------------------------------
void MultiBodyPlugin::postFrame()
{
    // we do not need to care about animation (auto or step) here,
    // because it's in the main program
}

//----------------------------------------------------------------------------
osg::Geode *MultiBodyPlugin::createBodyGeometry(int bodyId, int what)
{
    //fprintf(outfile,
    // "--- MultiBodyPlugin::createBodyGeometry (%d %s %d) called\n",
    //	  bodyId, geo.name[bodyId], what);

    osg::Geode *geode;
    int numPolygons, numCoords, numIndices = 0;

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
    osg::DrawArrayLengths *primitives;
    if (what == CREATE_GEO)
    {
        primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::POLYGON);
    }
    if (what == CREATE_WIRE)
    {
        primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::LINE_STRIP);
    }
    numPolygons = geo.nf[bodyId];
    numCoords = geo.nvertices[bodyId];

    osg::Vec3Array *vert = new osg::Vec3Array;
    int jj = 0, index;
    for (int j = 0; j < numPolygons; j++)
    {
        primitives->push_back(geo.npoints[bodyId][j]);
        for (int k = 0; k < geo.npoints[bodyId][j]; k++)
        {
            index = geo.face[bodyId][j][k] - 1; // array indices start with 0
            vert->push_back(osg::Vec3(geo.vertex[bodyId][index][0], geo.vertex[bodyId][index][1], geo.vertex[bodyId][iindex][2]));
            jj++;
        }
    }
    geom->setVertexArray(vert);

    double tmp = mode.coord_scaling / 0.2; // size coord.system 20% of largest value
    for (int j = 0; j < numCoords; j++)
    {
        for (int jj = 0; jj < 3; jj++)
        {
            if (tmp < geo.vertex[bodyId][j][jj])
            {
                tmp = geo.vertex[bodyId][j][jj];
            }
        }
    }
    mode.coord_scaling = tmp * 0.2;

    if (what == CREATE_GEO)
    {
        osg::Vec3Array *normalArray = new osg::Vec3Array();
        int jj = 0, index;
        for (int j = 0; j < numPolygons; j++)
        {
            for (int k = 0; k < geo.npoints[bodyId][j]; k++)
            {
                index = geo.face[bodyId][j][k] - 1; // array indices start with 0
                normalArray->push_back(osg::Vec3(geo.norm[bodyId][index][0], geo.norm[bodyId][index][1], geo.norm[bodyId][iindex][2]));
                jj++;
            }
        }
        geom->setNormalArray(normalArray);
        geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    geom->addPrimitiveSet(primitives);

    geom->setColorArray(colArr);
    osg::UShortArray *cIndices = new osg::UShortArray(numPolygons);
    for (int j = 0; j < numPolygons; j++)
    {
        if (what == CREATE_GEO)
        {
            (*cIndices)[j] = geo.fcolor[bodyId][j];
        }
        if (what == CREATE_WIRE)
        {
            (*cIndices)[j] = geo.ecolor[bodyId][j];
        }
    }
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->setColorIndices(cIndices);

    geode = new osg::Geode();
    geode->addDrawable(geom.get());
    geode->setName(geo.name[bodyId]);
    if (what == CREATE_WIRE)
        geode->setStateSet(unlightedStateSet.get());
    else
        geode->setStateSet(shadedStateSet.get());
    //geode->setStateSet(flatStateSet.get());
    return (geode);
}

//----------------------------------------------------------------------------
osg::Geode *MultiBodyPlugin::createelBodyGeometry(int bodyId, int what)
{
    //fprintf(outfile,
    // "--- MultiBodyPlugin::createelBodyGeometry (%d %s %d) called\n",
    //	  bodyId, geo.name[bodyId], what);

    osg::Geode *geode;
    int numPolygons, numCoords, numIndices = 0;

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
    osg::DrawArrayLengths *primitives;
    if (what == CREATE_GEO)
    {
        primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::POLYGON);
    }
    if (what == CREATE_WIRE)
    {
        primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::LINE_STRIP);
    }
    numPolygons = elgeo.nf[bodyId];
    numCoords = elgeo.nvertices[bodyId];

    if (global_vert[bodyId] == NULL)
    {
        global_vert[bodyId] = new osg::Vec3Array;
        for (int i = 0; i < numCoords; i++)
        {
            global_vert[bodyId]->push_back(osg::Vec3(elgeo.vertex[bodyId][str.act_step][i][0], elgeo.vertex[bodyId][str.act_step][i][1], elgeo.vertex[bodyId][str.act_step][i][2]));
        }
    }
    geom->setVertexArray(global_vert[bodyId]);

    double tmp = mode.coord_scaling / 0.2; // size coord.system 20% of largest value
    for (int j = 0; j < numCoords; j++)
    {
        for (int jj = 0; jj < 3; jj++)
        {
            if (tmp < elgeo.vertex[bodyId][str.act_step][j][jj])
            {
                tmp = elgeo.vertex[bodyId][str.act_step][j][jj];
            }
        }
    }
    mode.coord_scaling = tmp * 0.2;

    if (what == CREATE_GEO)
    {
        osg::Vec3Array *normalArray = new osg::Vec3Array();

        for (int i = 0; i < numCoords; i++)
        {
            normalArray->push_back(osg::Vec3(elgeo.norm[bodyId][str.act_step][i][0], elgeo.norm[bodyId][str.act_step][i][1], elgeo.norm[bodyId][str.act_step][i][2]));
        }
        geom->setNormalArray(normalArray);
        geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }
    // coordinate index list - attention: unsigned short-> list size <65535
    for (int j = 0; j < numPolygons; j++)
    {
        numIndices += elgeo.npoints[bodyId][j];
    }
    if (numCoords < USHRT_MAX)
    {
        osg::UShortArray *indices = new osg::UShortArray(numIndices);
        int jj = 0;

        for (int j = 0; j < numPolygons; j++)
        {
            primitives->push_back(elgeo.npoints[bodyId][j]);
            for (int k = 0; k < elgeo.npoints[bodyId][j]; k++)
            {
                (*indices)[jj] = elgeo.face[bodyId][j][k] - 1; // array indices start with 0
                jj++;
            }
        }
        geom->setVertexIndices(indices);
        if (what == CREATE_GEO)
        {
            geom->setNormalIndices(indices);
        }
    }
    else
    {
        osg::UIntArray *indices = new osg::UIntArray(numIndices);
        int jj = 0;

        for (int j = 0; j < numPolygons; j++)
        {
            primitives->push_back(elgeo.npoints[bodyId][j]);
            for (int k = 0; k < elgeo.npoints[bodyId][j]; k++)
            {
                (*indices)[jj] = elgeo.face[bodyId][j][k] - 1; // array indices start with 0
                jj++;
            }
        }
        geom->setVertexIndices(indices);

        if (what == CREATE_GEO)
        {
            geom->setNormalIndices(indices);
        }
    }
    geom->addPrimitiveSet(primitives);

    geom->setColorArray(colArr);
    osg::UShortArray *cIndices = new osg::UShortArray(numPolygons);
    for (int j = 0; j < numPolygons; j++)
    {
        if (what == CREATE_GEO)
        {
            (*cIndices)[j] = elgeo.fcolor[bodyId][j];
        }
        if (what == CREATE_WIRE)
        {
            (*cIndices)[j] = elgeo.ecolor[bodyId][j];
        }
    }
    geom->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);
    geom->setColorIndices(cIndices);

    geode = new osg::Geode();
    geode->addDrawable(geom.get());
    geode->setName(elgeo.name[bodyId]);
    if (what == CREATE_WIRE)
        geode->setStateSet(unlightedStateSet.get());
    else
        geode->setStateSet(shadedStateSet.get());
    return (geode);
}

//----------------------------------------------------------------------------
osg::Geode *MultiBodyPlugin::createBallGeometry(int ballId)
{

    //------------------------
    // ! function not used
    //------------------------
    //fprintf(outfile,
    // "--- MultiBodyPlugin::createBallGeometry (%d) called\n",ballId);

    osg::Geode *geode;
    float color3[3];
    int col;

    osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), (float)geo.ballsradius[ballId]);
    osg::ShapeDrawable *mySphereDrawable = new osg::ShapeDrawable(mySphere, hint.get());
    col = geo.ballscolor[ballId];
    if (col < 0)
    {
        ballcolor(color3, str.balldyncolor[str.act_step][ballId]);
        mySphereDrawable->setColor(osg::Vec4(color3[0], color3[1], color3[2], 1.0f));
    }
    else
    {
        mySphereDrawable->setColor(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    }
    geode = new osg::Geode();
    geode->addDrawable(mySphereDrawable);
    geode->setStateSet(shadedStateSet.get());

    return (geode);
}
//----------------------------------------------------------------------------
void MultiBodyPlugin::update(void)
{
    updateRigidTransform();
    updateBallTransform();
    updateIvTransform();
    updateElast();
    updateSensor();
    updateLineElements();
    updatePlotter();
    updatefixed();
    updateDynColors();
}

//----------------------------------------------------------------------------
void MultiBodyPlugin::updatefixed(void)
{
    osg::Matrix m;
    int k;
    float a_inv[16]; // hold inverted matrix

    // weshalb nie auf einheitsmatrix zuruecksetzen???

    // fix object by inverting its movements
    for (int i = 0; i < geo.nfiles; i++)
    {
        if (geo.fixmotion[i] != 0)
        {
            minvert(str.a[str.act_step][i], a_inv);
            k = 0;
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    m(r, c) = a_inv[k];
                    k++;
                }
            }
            vranimfixedTransform_.get()->setMatrix(m);
            break;
        }
    }

    // fix object by inverting its translation
    for (int i = 0; i < geo.nfiles; i++)
    {
        if (geo.fixtranslation[i] != 0)
        {
            transback(str.a[str.act_step][i], a_inv);
            k = 0;
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    m(r, c) = a_inv[k];
                    k++;
                }
            }
            vranimfixedTransform_.get()->setMatrix(m);
        }
    }
}

//----------------------------------------------------------------------------
void MultiBodyPlugin::updateRigidTransform(void)
{
    osg::Matrix m;
    int k;

    for (int bodyId = 0; bodyId < geo.nfiles; bodyId++)
    {
        k = 0;
        if (str.a)
        {
            for (int r = 0; r < 4; r++)
            {
                for (int c = 0; c < 4; c++)
                {
                    m(r, c) = str.a[str.act_step][bodyId][k];
                    k++;
                }
            }
        }
        vranimbodyTransform_[bodyId].get()->setMatrix(m);
    }
}

//----------------------------------------------------------------------------
void MultiBodyPlugin::updateIvTransform(void)
{
    osg::Matrix m;
    int k;

    for (int i = 0; i < iv.nfiles; i++)
    {
        k = 0;
        for (int r = 0; r < 4; r++)
        {
            for (int c = 0; c < 4; c++)
            {
                m(r, c) = str.a /*richtig spaeter: _iv*/[str.act_step][i][k];
                k++;
            }
        }
        vranimivTransform_[i]->setMatrix(m);
    }
}
//----------------------------------------------------------------------------
void MultiBodyPlugin::updateBallTransform(void)
{
    osg::Matrix m;
    //int      k;

    if (geo.nballs > 0)
    {
        if (str.a_ball)
        {
            sphere->updateCoordsFromMatrices(str.a_ball[str.act_step]);
            /*
         for (int ballId=0; ballId< geo.nballs; ballId++){
            k=0;
            for (int r=0; r<4; r++){
               for (int c=0; c<4; c++){  
                  m(r,c)=str.a_ball[str.act_step][ballId][k];
               }
               k++;
            }
            vranimballTransform_[ballId].get()->setMatrix(m);    
         }*/
        }
    }
}

//----------------------------------------------------------------------------
void MultiBodyPlugin::updateElast(void)
{
    if (elgeo.nfiles > 0)
    {
        for (int i = 0; i < elgeo.nfiles; i++)
        {
            for (int j = 0; j < elgeo.nvertices[i]; j++)
            {
                float cx = elgeo.vertex[i][str.act_step][j][0];
                float cy = elgeo.vertex[i][str.act_step][j][1];
                float cz = elgeo.vertex[i][str.act_step][j][2];
                if (global_vert[i])
                {
                    (*global_vert[i])[j].set(cx, cy, cz);
                }
                vranimelbodywireGeometry_[i]->getDrawable(0)->dirtyDisplayList();
                vranimelbodywireGeometry_[i]->dirtyBound();
                vranimelbodyGeometry_[i]->getDrawable(0)->dirtyDisplayList();
                vranimelbodyGeometry_[i]->dirtyBound();
            }
        }
    }
}

//----------------------------------------------------------------------------
void MultiBodyPlugin::updateSensor(void)
{
    osg::Drawable *drawable;

    if (sensor.nr > 0)
    {
        for (int i = 0; i < sensor.nr; i++)
        {
            drawable = vranimsensorGeometry_[i].get()->getDrawable(0);
            osg::Geometry *geom = dynamic_cast<osg::Geometry *>(drawable);
            if (geom)
            {
                osg::PrimitiveSet *prim = geom->getPrimitiveSet(0);
                if (prim)
                {
                    osg::DrawArrayLengths *da = dynamic_cast<osg::DrawArrayLengths *>(prim);
                    if (da)
                    {
                        (*da)[0] = str.act_step;
                        //prim->setModifiedCount(str.act_step);
                    }
                }
                geom->dirtyDisplayList();
            }
        }
    }
}

//----------------------------------------------------------------------------
void MultiBodyPlugin::updateLineElements(void)
{
    osg::Matrix m, m1, m2, m3;

    if (lin.nr > 0)
    {
        for (int i = 0; i < lin.nr; i++)
        {
            //      pfVec3 dir_0,dir_act;
            osg::Vec3f dir_0, dir_act;
            //      dir_0.set(1.0f,0.0f,0.0f);
            dir_0.set(1.0f, 0.0f, 0.0f);
            float rx = lin.dir[i][str.act_step][0] / lin.length[i][str.act_step];
            float ry = lin.dir[i][str.act_step][1] / lin.length[i][str.act_step];
            float rz = lin.dir[i][str.act_step][2] / lin.length[i][str.act_step];
            //      dir_act.set(rx,ry,rz);
            dir_act.set(rx, ry, rz);
            float tx = lin.pkt1[i][str.act_step][0];
            float ty = lin.pkt1[i][str.act_step][1];
            float tz = lin.pkt1[i][str.act_step][2];
            //      m1.makeVecRotVec(dir_0,dir_act);
            m1.makeRotate(dir_0, dir_act);
            //      m2.makeScale(lin.length[i][str.act_step],1.0,1.0);
            m2.makeScale(lin.length[i][str.act_step], 1.0, 1.0);
            //      m3.makeTrans(tx,ty,tz);
            m3.makeTranslate(tx, ty, tz);
            m.mult(m2, m1);
            m.mult(m, m3);
            //      vranimlinElTransform_[i]->setMat(m);
            vranimlinElTransform_[i]->setMatrix(m);
        }
    }
}

//----------------------------------------------------------------------
void MultiBodyPlugin::updatePlotter(void)
{
    if (plotItem)
    {
        for (int i = 0; i < dat.ndata; i++)
        {
            plotItem[i]->lineDrawArray->setCount(str.act_step);
        }
    }
    /*if(lenList_plotterline != NULL){
    lenList_plotterline[0] = str.act_step; // todo wie oben dirtyDisplayList
  }*/
}

//----------------------------------------------------------------------
void MultiBodyPlugin::updateDynColors(void)
{
    //fprintf(outfile, "updateDynColors\n");
    if (dyn.isset != NULL)
    {
        for (int j = MAXCOLORS; j < MAXCOLORS + MAXNODYNCOLORS; j++)
        {
            if (dyn.isset[j - MAXCOLORS] == GL_TRUE)
            {
                for (int i = 0; i < 3; i++)
                {
                    colorindex[j][i] = dyn.rgb[j - MAXCOLORS][str.act_step][i];
                }
                float r = colorindex[j][0];
                float g = colorindex[j][1];
                float b = colorindex[j][2];
                float a = colorindex[j][3];
                (*colArr)[j].set(r, g, b, a);
            }
        }
    }
}
//------------------------------------------------------------------------------
void MultiBodyPlugin::menuEvent(coMenuItem *item)
{
    //fprintf(stdout,"--- MultiBodyPlugin::menuEvent called\n");

    if (item == anim_shadewire_Checkbox_)
    {
        if (anim_shadewire_Checkbox_->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodywireSwitch_[i]->setAllChildrenOn();
            }
        }
        else
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodywireSwitch_[i]->setAllChildrenOff();
            }
        }
    }

    if (item == anim_shadeoff_Checkbox_)
    {
        if (anim_shadeoff_Checkbox_->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodygeoSwitch_[i]->setAllChildrenOff();
            }
        }
        else
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
            }
        }
    }

    if (item == anim_shadeunlighted_Checkbox_)
    {
        if (anim_shadeunlighted_Checkbox_->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                vranimbodyGeometry_[i]->setStateSet(unlightedStateSet.get());
            }
        }
    }

    if (item == anim_shadeflat_Checkbox_)
    {
        if (anim_shadeflat_Checkbox_->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                vranimbodyGeometry_[i]->setStateSet(flatStateSet.get());
            }
        }
    }

    if (item == anim_shadegouraud_Checkbox_)
    {
        if (anim_shadegouraud_Checkbox_->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                vranimbodyGeometry_[i]->setStateSet(shadedStateSet.get());
            }
        }
    }

    if (item == anim_shadeflexwire_Checkbox_)
    {
        if (anim_shadeflexwire_Checkbox_->getState() == true)
        {
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodywireSwitch_[i]->setAllChildrenOn();
            }
        }
        else
        {
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodywireSwitch_[i]->setAllChildrenOff();
            }
        }
    }

    if (item == anim_shadeflexoff_Checkbox_)
    {
        if (anim_shadeflexoff_Checkbox_->getState() == true)
        {
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodygeoSwitch_[i]->setAllChildrenOff();
            }
        }
        else
        {
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodygeoSwitch_[i]->setAllChildrenOn();
            }
        }
    }

    if (item == anim_shadeflexunlighted_Checkbox_)
    {
        if (anim_shadeflexunlighted_Checkbox_->getState() == true)
        {
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodygeoSwitch_[i]->setAllChildrenOn();
                vranimelbodyGeometry_[i]->setStateSet(unlightedStateSet.get());
            }
        }
    }

    if (item == anim_shadeflexflat_Checkbox_)
    {
        if (anim_shadeflexflat_Checkbox_->getState() == true)
        {
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodygeoSwitch_[i]->setAllChildrenOn();
                vranimelbodyGeometry_[i]->setStateSet(flatStateSet.get());
            }
        }
    }

    if (item == anim_shadeflexgouraud_Checkbox_)
    {
        if (anim_shadeflexgouraud_Checkbox_->getState() == true)
        {
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodygeoSwitch_[i]->setAllChildrenOn();
                vranimelbodyGeometry_[i]->setStateSet(shadedStateSet.get());
            }
        }
    }

    if (item == anim_interval_Button_)
    {
        fprintf(stderr, "anim_interval_Button_ was pressed\n");
    }

    if (item == anim_calcstride_Button_)
    {
        fprintf(stderr, "anim_calcstride_Button_ was pressed\n");
    }

    if (item == anim_savetrafo_Button_)
    {
        saveTrafo();
    }

    if (sensor.nr > 0)
    {
        if (item == anim_showsensors_Checkbox_)
        {
            if (anim_showsensors_Checkbox_->getState() == true)
            {
                vranimsensorSwitch_->setAllChildrenOn();
            }
            else
            {
                vranimsensorSwitch_->setAllChildrenOff();
            }
        }
    }

    if (dat.ndata > 0)
    {
        if (item == anim_showplotters_Checkbox_)
        {
            if (anim_showplotters_Checkbox_->getState() == true)
            {
                for (int i = 0; i < dat.ndata; i++)
                {
                    plotHandle[i]->setVisible(true);
                }
            }
            else
            {
                for (int i = 0; i < dat.ndata; i++)
                {
                    plotHandle[i]->setVisible(false);
                }
            }
        }
    }

    if (item == anim_showcoordsystem_Checkbox_)
    {
        if (anim_showcoordsystem_Checkbox_->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodycsSwitch_[i]->setAllChildrenOn();
            }
        }
        else
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodycsSwitch_[i]->setAllChildrenOff();
            }
        }
    }

    // --hide--------------------------
    if (item == anim_nohide_Checkbox_)
    {
        if (anim_nohide_Checkbox_->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                geo.hide[i] = false;
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                if (anim_shadewire_Checkbox_->getState() == true) //if wire is selected
                    vranimbodywireSwitch_[i]->setAllChildrenOn();
            }
            for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
            {
                anim_hide_Checkbox_[i]->setState(false);
                // The result ist not always correct because of limited bodylist in the menu:
                // if you unselect in the hide-menu all bodies, the 'do not hide'-checkbox
                // is automatically selected.
                // But there can still be bodies hidden, selected in the tabletUI
            }
        }
        else
        {
            anim_nohide_Checkbox_->setState(true); // cannot be set to false, if there is no body hidden
        }
    }
    for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
    {
        if (item == anim_hide_Checkbox_[i])
        {
            if (anim_hide_Checkbox_[i]->getState() == true)
            {
                geo.hide[i] = true;
                vranimbodygeoSwitch_[i]->setAllChildrenOff();
                vranimbodywireSwitch_[i]->setAllChildrenOff();
                anim_nohide_Checkbox_->setState(false);
            }
            else
            {
                geo.hide[i] = false;
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                if (anim_shadewire_Checkbox_->getState() == true) //if wire is selected
                    vranimbodywireSwitch_[i]->setAllChildrenOn();

                //loop over if there are any more bodies hidden
                bool nobodyHidden = true;
                for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
                    if (anim_hide_Checkbox_[i]->getState() == true)
                        nobodyHidden = false;
                if (nobodyHidden)
                    anim_nohide_Checkbox_->setState(true); // is set to true, if there is no more body hidden
            }
        }
    }

    // --fixmotion--------------------------
    if (item == anim_nofixmotion_Checkbox_)
    {
        if (anim_nofixmotion_Checkbox_->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                geo.fixmotion[i] = false;
                geo.fixtranslation[i] = false;
            }
            for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
            {
                anim_fixtranslation_Checkbox_[i]->setState(false);
            }
            anim_nofixtranslation_Checkbox_->setState(true);
        }
    }
    for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
    {
        if (item == anim_fixmotion_Checkbox_[i])
        {
            if (anim_fixmotion_Checkbox_[i]->getState() == true)
            {
                for (int j = 0; j < geo.nfiles; j++)
                {
                    geo.fixmotion[j] = false;
                    geo.fixtranslation[j] = false;
                }
                for (int j = 0; j < geo.nfiles && j < VRANIM_MAXNOMENUITEMS; j++)
                {
                    anim_fixtranslation_Checkbox_[j]->setState(false);
                }
                geo.fixmotion[i] = true;
                anim_nofixtranslation_Checkbox_->setState(true);
            }
        }
    }

    // --fixtranslation--------------------------
    if (item == anim_nofixtranslation_Checkbox_)
    {
        if (anim_nofixtranslation_Checkbox_->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                geo.fixmotion[i] = false;
                geo.fixtranslation[i] = false;
            }
            for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
            {
                anim_fixmotion_Checkbox_[i]->setState(false);
            }
            anim_nofixmotion_Checkbox_->setState(true);
        }
    }
    for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
    {
        if (item == anim_fixtranslation_Checkbox_[i])
        {
            if (anim_fixtranslation_Checkbox_[i]->getState() == true)
            {
                for (int j = 0; j < geo.nfiles; j++)
                {
                    geo.fixmotion[j] = false;
                    geo.fixtranslation[j] = false;
                }
                for (int j = 0; j < geo.nfiles && j < VRANIM_MAXNOMENUITEMS; j++)
                {
                    anim_fixmotion_Checkbox_[j]->setState(false);
                }
                geo.fixtranslation[i] = true;
                anim_nofixmotion_Checkbox_->setState(true);
            }
        }
    }
}

//--------------------------------------------------------------------
void MultiBodyPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == rigidBodiesAsWireButton)
    {
        if (rigidBodiesAsWireButton->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodywireSwitch_[i]->setAllChildrenOn();
            }
            //menu
            anim_shadewire_Checkbox_->setState(true);
        }
        else
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodywireSwitch_[i]->setAllChildrenOff();
            }
            //menu
            anim_shadewire_Checkbox_->setState(false);
        }
    }

    if (tUIItem == rigidComboBox)
    {
        //menu
        anim_shadeoff_Checkbox_->setState(false);
        anim_shadeunlighted_Checkbox_->setState(false);
        anim_shadeflat_Checkbox_->setState(false);
        anim_shadegouraud_Checkbox_->setState(false);

        switch (rigidComboBox->getSelectedEntry())
        {

        case 0:
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodygeoSwitch_[i]->setAllChildrenOff();
            }
            anim_shadeoff_Checkbox_->setState(true);
            break;

        case 1:
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                vranimbodyGeometry_[i]->setStateSet(unlightedStateSet.get());
            }
            anim_shadeunlighted_Checkbox_->setState(true);
            break;

        case 2:
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                vranimbodyGeometry_[i]->setStateSet(flatStateSet.get());
            }
            anim_shadeflat_Checkbox_->setState(true);
            break;

        case 3:
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                vranimbodyGeometry_[i]->setStateSet(shadedStateSet.get());
            }
            anim_shadegouraud_Checkbox_->setState(true);
            break;
        }
    }

    if (tUIItem == flexBodiesAsWireButton)
    {
        if (flexBodiesAsWireButton->getState() == true)
        {
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodywireSwitch_[i]->setAllChildrenOn();
            }
            //menu
            anim_shadeflexwire_Checkbox_->setState(true);
        }
        else
        {
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodywireSwitch_[i]->setAllChildrenOff();
            }
            //menu
            anim_shadeflexwire_Checkbox_->setState(false);
        }
    }

    if (tUIItem == flexComboBox)
    {
        //menu
        anim_shadeflexoff_Checkbox_->setState(false);
        anim_shadeflexunlighted_Checkbox_->setState(false);
        anim_shadeflexflat_Checkbox_->setState(false);
        anim_shadeflexgouraud_Checkbox_->setState(false);

        switch (flexComboBox->getSelectedEntry())
        {

        case 0:
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodygeoSwitch_[i]->setAllChildrenOff();
            }
            anim_shadeflexoff_Checkbox_->setState(true);
            break;

        case 1:
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodygeoSwitch_[i]->setAllChildrenOn();
                vranimelbodyGeometry_[i]->setStateSet(unlightedStateSet.get());
            }
            anim_shadeflexunlighted_Checkbox_->setState(true);
            break;

        case 2:
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodygeoSwitch_[i]->setAllChildrenOn();
                vranimelbodyGeometry_[i]->setStateSet(flatStateSet.get());
            }
            anim_shadeflexflat_Checkbox_->setState(true);
            break;

        case 3:
            for (int i = 0; i < elgeo.nfiles; i++)
            {
                vranimelbodygeoSwitch_[i]->setAllChildrenOn();
                vranimelbodyGeometry_[i]->setStateSet(shadedStateSet.get());
            }
            anim_shadeflexgouraud_Checkbox_->setState(true);
            break;
        }
    }

    if (tUIItem == saveTrafoButton)
    {
        saveTrafo();
    }

    if (sensor.nr > 0)
    {
        if (tUIItem == showSensorsButton)
        {
            if (showSensorsButton->getState() == true)
            {
                vranimsensorSwitch_->setAllChildrenOn();
                //menu
                anim_showsensors_Checkbox_->setState(true);
            }
            else
            {
                vranimsensorSwitch_->setAllChildrenOff();
                //menu
                anim_showsensors_Checkbox_->setState(false);
            }
        }
    }

    if (dat.ndata > 0)
    {
        if (tUIItem == showPlottersButton)
        {
            if (showPlottersButton->getState() == true)
            {
                for (int i = 0; i < dat.ndata; i++)
                {
                    plotHandle[i]->setVisible(true);
                }
                //menu
                anim_showplotters_Checkbox_->setState(true);
            }
            else
            {
                for (int i = 0; i < dat.ndata; i++)
                {
                    plotHandle[i]->setVisible(false);
                }
                //menu
                anim_showplotters_Checkbox_->setState(false);
            }
        }
    }

    if (tUIItem == showCoordSystemsButton)
    {
        if (showCoordSystemsButton->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodycsSwitch_[i]->setAllChildrenOn();
            }
            //menu
            anim_showcoordsystem_Checkbox_->setState(true);
        }
        else
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                vranimbodycsSwitch_[i]->setAllChildrenOff();
            }
            //menu
            anim_showcoordsystem_Checkbox_->setState(false);
        }
    }

    // --hide--------------------------
    if (tUIItem == doNotHideButton)
    {
        if (doNotHideButton->getState() == true)
        {
            for (int i = 0; i < geo.nfiles; i++)
            {
                geo.hide[i] = false;
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                if (anim_shadewire_Checkbox_->getState() == true) //if wire is selected
                    vranimbodywireSwitch_[i]->setAllChildrenOn();
                //menu
                if (i < VRANIM_MAXNOMENUITEMS)
                    anim_hide_Checkbox_[i]->setState(false);
                hideBodyButton[i]->setState(false);
            }
            doNotHideButton->setState(true);
            //menu
            anim_nohide_Checkbox_->setState(true);
        }
        else
        {
            bool nobodyHidden = true;
            for (int i = 0; i < geo.nfiles; i++)
                if (hideBodyButton[i]->getState() == true)
                    nobodyHidden = false;
            if (nobodyHidden)
                doNotHideButton->setState(true); // cannot be set to false, if there is no body hidden
        }
    }

    for (int i = 0; i < geo.nfiles; i++)
    {
        if (tUIItem == hideBodyButton[i])
        {
            if (hideBodyButton[i]->getState() == true)
            {
                geo.hide[i] = true;
                vranimbodygeoSwitch_[i]->setAllChildrenOff();
                vranimbodywireSwitch_[i]->setAllChildrenOff();
                doNotHideButton->setState(false);
                //menu
                if (i < VRANIM_MAXNOMENUITEMS)
                    anim_hide_Checkbox_[i]->setState(true);
                anim_nohide_Checkbox_->setState(false);
            }
            else
            {
                geo.hide[i] = false;
                vranimbodygeoSwitch_[i]->setAllChildrenOn();
                if (anim_shadewire_Checkbox_->getState() == true) //if wire is selected
                    vranimbodywireSwitch_[i]->setAllChildrenOn();
                if (i < VRANIM_MAXNOMENUITEMS)
                    anim_hide_Checkbox_[i]->setState(false); //menu

                //loop over if there are any more bodies hidden
                bool nobodyHidden = true;
                for (int i = 0; i < geo.nfiles; i++)
                    if (hideBodyButton[i]->getState() == true)
                        nobodyHidden = false;
                if (nobodyHidden)
                {
                    doNotHideButton->setState(true); // is set to true, if there is no more body hidden
                    anim_nohide_Checkbox_->setState(true); // and also in the menu
                }
            }
        }
    }

    // --do not fix--------------------------
    if (tUIItem == doNotFixRadioButton)
    {
        doNotFix();
    }

    // --fixmotion--------------------------
    for (int i = 0; i < geo.nfiles; i++)
    {
        if (tUIItem == fixMotionButton[i])
        {
            for (int j = 0; j < geo.nfiles; j++)
            {
                geo.fixmotion[j] = false;
                geo.fixtranslation[j] = false;
            }
            for (int j = 0; j < geo.nfiles; j++)
            {
                if (i != j)
                {
                    fixMotionButton[j]->setState(false);
                    //menu, nur vom tabletui aus
                    if (j < VRANIM_MAXNOMENUITEMS)
                        anim_fixmotion_Checkbox_[j]->setState(false);
                }
                else
                {
                    //menu, nur vom tabletui aus
                    if (j < VRANIM_MAXNOMENUITEMS)
                        anim_fixmotion_Checkbox_[j]->setState(true);
                }
                //tabletUI
                fixTranslationButton[j]->setState(false);
                //menu
                if (j < VRANIM_MAXNOMENUITEMS)
                    anim_fixtranslation_Checkbox_[j]->setState(false);
            }
            geo.fixmotion[i] = true;
            doNotFixRadioButton->setState(false);
            //menu, nur vom tabletui aus
            anim_nofixmotion_Checkbox_->setState(false);
            anim_nofixtranslation_Checkbox_->setState(true); //<-menu
            if (i < VRANIM_MAXNOMENUITEMS)
                anim_fixmotion_Checkbox_[i]->setState(true);
        }
    }

    // --fixtranslation--------------------------
    for (int i = 0; i < geo.nfiles; i++)
    {
        if (tUIItem == fixTranslationButton[i])
        {
            for (int j = 0; j < geo.nfiles; j++)
            {
                geo.fixtranslation[j] = false;
                geo.fixmotion[j] = false;
            }
            for (int j = 0; j < geo.nfiles; j++)
            {
                if (i != j)
                {
                    fixTranslationButton[j]->setState(false);
                    //menu, nur vom tabletui aus
                    if (j < VRANIM_MAXNOMENUITEMS)
                        anim_fixtranslation_Checkbox_[j]->setState(false);
                }
                else
                {
                    //menu, nur vom tabletui aus
                    if (j < VRANIM_MAXNOMENUITEMS)
                        anim_fixtranslation_Checkbox_[j]->setState(true);
                }
                //tabletUI
                fixMotionButton[j]->setState(false);
                //menu
                if (j < VRANIM_MAXNOMENUITEMS)
                    anim_fixmotion_Checkbox_[j]->setState(false);
            }
            geo.fixtranslation[i] = true;
            doNotFixRadioButton->setState(false);
            //menu, nur vom tabletui aus
            anim_nofixtranslation_Checkbox_->setState(false);
            anim_nofixmotion_Checkbox_->setState(true); //<-menu
            if (i < VRANIM_MAXNOMENUITEMS)
                anim_fixtranslation_Checkbox_[i]->setState(true);
        }
    }
}

//--------------------------------------------------------------------
void MultiBodyPlugin::saveTrafo(void)
{
    fprintf(stderr, "anim_savetrafo_Button_ was pressed\n");
    FILE *fp_mat;
    if ((fp_mat = fopen("viewing.mat", "w")) == NULL)
    {
        fprintf(outfile, "error opening file viewing.mat,\n");
        fprintf(outfile, "  Transformation matrix not saved\n");
    }

    osg::Matrix m_mat;
    m_mat = cover->getObjectsXform()->getMatrix();
    //     fprintf(outfile,"write matrix and scaling factor:\n ");
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            fprintf(fp_mat, "%f ", m_mat(i, j));
            // 	fprintf(outfile,"%f ",m_mat[i][j]);
        }
        //       fprintf(outfile,"\n");
    }
    //./covise/src/renderer/COVER/cover/coVRPluginSupport.h
    float scalingfactor;
    scalingfactor = cover->getScale();
    fprintf(fp_mat, " \n%f\n", scalingfactor);
    //     fprintf(outfile,"scene scaling factor: %f\n",scalingfactor);

    //     fprintf(outfile,"done\n ");

    (void)fclose(fp_mat);
    fprintf(outfile, "  global transformation matrix saved in file viewing.mat\n");
}
//--------------------------------------------------------------------
void MultiBodyPlugin::doNotFix(void)
{
    doNotFixRadioButton->setState(true);
    for (int i = 0; i < geo.nfiles; i++)
    {
        if (fixMotionButton[i]->getState() == true)
        {
            geo.fixmotion[i] = false;
            //tabletUI
            fixMotionButton[i]->setState(false);
            //menu, nur vom tabletui aus
            if (i < VRANIM_MAXNOMENUITEMS)
                anim_fixmotion_Checkbox_[i]->setState(false);
        }
        geo.fixtranslation[i] = false;
        fixTranslationButton[i]->setState(false);
        if (i < VRANIM_MAXNOMENUITEMS)
            anim_fixtranslation_Checkbox_[i]->setState(false);
    }
    anim_nofixtranslation_Checkbox_->setState(true);
    //menu, nur vom tabletui aus
    anim_nofixmotion_Checkbox_->setState(true);
}

//--------------------------------------------------------------------
void MultiBodyPlugin::reset_timestep(void)
{
    if (str.stride < 0)
    {
        str.act_step = str.timesteps - 1;
    }
    else
    {
        str.act_step = 0;
    }
}

//--------------------------------------------------------------------
void MultiBodyPlugin::setTimestep(int t)
{
    if (t < str.timesteps)
        str.act_step = t;
}

//--------------------------------------------------------------------
void MultiBodyPlugin::ballcolor(float *color, float fcolor)
{
    color[0] = fcolor;
    color[1] = 0;
    color[2] = 1 - fcolor;
}

//--------------------------------------------------------------------
void MultiBodyPlugin::menus_create()
{
    // add button to cover main menu
    multiBodyMenuButton_ = new coSubMenuItem("vranim ...");
    cover->getMenu()->add(multiBodyMenuButton_);

    // submenu
    multiBodyMenu_ = new coRowMenu("vranim", NULL);
    multiBodyMenuButton_->setMenu(multiBodyMenu_);

    mode.anim = ANIM_AUTO;

    // ----------------------------
    // shadecontrol submenu
    anim_shaderadio_Button_ = new coSubMenuItem("shading ...");
    multiBodyMenu_->add(anim_shaderadio_Button_);

    // shade
    anim_shaderadio_Menu_ = new coRowMenu("shading");
    anim_shaderadio_Button_->setMenu(anim_shaderadio_Menu_);

    anim_shadewire_Checkbox_ = new coCheckboxMenuItem("rigid bodies as wire", false);
    anim_shadewire_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadewire_Checkbox_);

    anim_shadewire_Checkbox_->setState(false);

    // shade checkbox
    anim_shade_Radio_Group_ = new coCheckboxGroup();

    anim_shadeoff_Checkbox_ = new coCheckboxMenuItem("rigid off", false, anim_shade_Radio_Group_);
    anim_shadeoff_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadeoff_Checkbox_);

    anim_shadeunlighted_Checkbox_ = new coCheckboxMenuItem("rigid unlighted", false, anim_shade_Radio_Group_);
    anim_shadeunlighted_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadeunlighted_Checkbox_);

    anim_shadeflat_Checkbox_ = new coCheckboxMenuItem("rigid flat", false, anim_shade_Radio_Group_);
    anim_shadeflat_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadeflat_Checkbox_);

    anim_shadegouraud_Checkbox_ = new coCheckboxMenuItem("rigid gouraud", false, anim_shade_Radio_Group_);
    anim_shadegouraud_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadegouraud_Checkbox_);

    anim_shadeoff_Checkbox_->setState(false);
    anim_shadeunlighted_Checkbox_->setState(false);
    anim_shadeflat_Checkbox_->setState(false);
    anim_shadegouraud_Checkbox_->setState(true);

    // flex
    anim_shadeflexwire_Checkbox_ = new coCheckboxMenuItem("flex bodies as wire", true);
    anim_shadeflexwire_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadeflexwire_Checkbox_);

    anim_shadeflexwire_Checkbox_->setState(true);

    // shadeflex checkbox
    anim_shadeflex_Radio_Group_ = new coCheckboxGroup();

    anim_shadeflexoff_Checkbox_ = new coCheckboxMenuItem("flex off", false, anim_shadeflex_Radio_Group_);
    anim_shadeflexoff_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadeflexoff_Checkbox_);

    anim_shadeflexunlighted_Checkbox_ = new coCheckboxMenuItem("flex unlighted", false, anim_shadeflex_Radio_Group_);
    anim_shadeflexunlighted_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadeflexunlighted_Checkbox_);

    anim_shadeflexflat_Checkbox_ = new coCheckboxMenuItem("flex flat", false, anim_shadeflex_Radio_Group_);
    anim_shadeflexflat_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadeflexflat_Checkbox_);

    anim_shadeflexgouraud_Checkbox_ = new coCheckboxMenuItem("flex gouraud", false, anim_shadeflex_Radio_Group_);
    anim_shadeflexgouraud_Checkbox_->setMenuListener(this);
    anim_shaderadio_Menu_->add(anim_shadeflexgouraud_Checkbox_);

    anim_shadeflexoff_Checkbox_->setState(false);
    anim_shadeflexflat_Checkbox_->setState(true);
    anim_shadeflexunlighted_Checkbox_->setState(false);
    anim_shadeflexgouraud_Checkbox_->setState(false);

    // ----------------------------
    anim_interval_Button_ = new coButtonMenuItem("interval");
    anim_interval_Button_->setMenuListener(this);
    // multiBodyMenu_->add(anim_interval_Button_);

    anim_calcstride_Button_ = new coButtonMenuItem("calc stride");
    anim_calcstride_Button_->setMenuListener(this);
    // multiBodyMenu_->add(anim_calcstride_Button_);

    // ----------------------------
    // show sensors
    if (sensor.nr > 0)
    {
        anim_showsensors_Checkbox_ = new coCheckboxMenuItem("show sensors", false);
        anim_showsensors_Checkbox_->setMenuListener(this);
        multiBodyMenu_->add(anim_showsensors_Checkbox_);
    }

    // ----------------------------
    // show plotters
    if (dat.ndata > 0)
    {
        anim_showplotters_Checkbox_ = new coCheckboxMenuItem("show plotters", false);
        anim_showplotters_Checkbox_->setMenuListener(this);
        multiBodyMenu_->add(anim_showplotters_Checkbox_);
    }

    // ----------------------------
    // show coord. systems
    anim_showcoordsystem_Checkbox_ = new coCheckboxMenuItem("show coord.systems", false);
    anim_showcoordsystem_Checkbox_->setMenuListener(this);
    multiBodyMenu_->add(anim_showcoordsystem_Checkbox_);

    // ----------------------------
    // save trafo
    anim_savetrafo_Button_ = new coButtonMenuItem("save trafo");
    anim_savetrafo_Button_->setMenuListener(this);
    multiBodyMenu_->add(anim_savetrafo_Button_);

    // ----------------------------
    // hidecontrol submenu
    anim_hideradio_Button_ = new coSubMenuItem("hide bodies ...");
    multiBodyMenu_->add(anim_hideradio_Button_);

    // hide
    anim_hideradio_Menu_ = new coRowMenu("hide bodies");
    anim_hideradio_Button_->setMenu(anim_hideradio_Menu_);

    anim_nohide_Checkbox_ = new coCheckboxMenuItem("do not hide any bodies", false);
    anim_nohide_Checkbox_->setMenuListener(this);
    anim_hideradio_Menu_->add(anim_nohide_Checkbox_);

    int numberofitems;
    if (geo.nfiles > VRANIM_MAXNOMENUITEMS)
    {
        numberofitems = VRANIM_MAXNOMENUITEMS;
    }
    else
    {
        numberofitems = geo.nfiles;
    }
    anim_hide_Checkbox_ = new coCheckboxMenuItem *[numberofitems];
    for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
    {
        anim_hide_Checkbox_[i] = new coCheckboxMenuItem(geo.name[i], false);
        anim_hide_Checkbox_[i]->setMenuListener(this);
        anim_hideradio_Menu_->add(anim_hide_Checkbox_[i]);
    }

    anim_nohide_Checkbox_->setState(true);

    // ----------------------------
    // fixmotion control submenu
    anim_fixmotionradio_Button_ = new coSubMenuItem("fix motion ...");
    multiBodyMenu_->add(anim_fixmotionradio_Button_);

    // fixmotion
    anim_fixmotionradio_Menu_ = new coRowMenu("fix motion");
    anim_fixmotionradio_Button_->setMenu(anim_fixmotionradio_Menu_);

    // fixmotion checkbox group
    anim_fixmotion_Radio_Group_ = new coCheckboxGroup();

    anim_nofixmotion_Checkbox_ = new coCheckboxMenuItem("do not fix any motion",
                                                        true, anim_fixmotion_Radio_Group_);
    anim_nofixmotion_Checkbox_->setMenuListener(this);
    anim_fixmotionradio_Menu_->add(anim_nofixmotion_Checkbox_);

    anim_fixmotion_Checkbox_ = new coCheckboxMenuItem *[numberofitems];
    for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
    {
        anim_fixmotion_Checkbox_[i] = new coCheckboxMenuItem(geo.name[i],
                                                             false, anim_fixmotion_Radio_Group_);
        anim_fixmotion_Checkbox_[i]->setMenuListener(this);
        anim_fixmotionradio_Menu_->add(anim_fixmotion_Checkbox_[i]);
    }
    anim_nofixmotion_Checkbox_->setState(true);
    for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
    {
        anim_fixmotion_Checkbox_[i]->setState(false);
    }

    // ----------------------------
    // fixtranslation control submenu
    anim_fixtranslationradio_Button_ = new coSubMenuItem("fix translation ...");
    multiBodyMenu_->add(anim_fixtranslationradio_Button_);

    // fixtranslation
    anim_fixtranslationradio_Menu_ = new coRowMenu("fix translation");
    anim_fixtranslationradio_Button_->setMenu(anim_fixtranslationradio_Menu_);

    // fixtranslation checkbox group
    anim_fixtranslation_Radio_Group_ = new coCheckboxGroup();

    anim_nofixtranslation_Checkbox_ = new coCheckboxMenuItem("do not fix any translation",
                                                             true, anim_fixtranslation_Radio_Group_);
    anim_nofixtranslation_Checkbox_->setMenuListener(this);
    anim_fixtranslationradio_Menu_->add(anim_nofixtranslation_Checkbox_);
    anim_fixtranslation_Checkbox_ = new coCheckboxMenuItem *[numberofitems];
    for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
    {
        anim_fixtranslation_Checkbox_[i] = new coCheckboxMenuItem(geo.name[i],
                                                                  false, anim_fixtranslation_Radio_Group_);
        anim_fixtranslation_Checkbox_[i]->setMenuListener(this);
        anim_fixtranslationradio_Menu_->add(anim_fixtranslation_Checkbox_[i]);
    }
    anim_nofixtranslation_Checkbox_->setState(true);
    for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
    {
        anim_fixtranslation_Checkbox_[i]->setState(false);
    }

    if (dat.ndata > 0)
    {
        plotHandle = new coPopupHandle *[dat.ndata];
        plotItem = new coPlotItem *[dat.ndata];
        for (int i = 0; i < dat.ndata; i++)
        {
            char *name = new char[MAXLENGTH];
            sprintf(name, "%d: %s", i, dat.name[i]);
            plotHandle[i] = new coPopupHandle(name);
            plotHandle[i]->setPos(0, i * 2.0, i * 100.0);
            //coFrame *panelFrame = new coFrame("UI/Frame");
            plotItem[i] = new coPlotItem(i);

            //panelFrame->addElement(plotItem[i]);
            //pe org plotHandle[i]->addElement(panelFrame);
            plotHandle[i]->addElement(plotItem[i]);
            plotHandle[i]->setVisible(false);
        }
    }
}

//--------------------------------------------------------------------
void MultiBodyPlugin::menus_delete()
{
    if (multiBodyMenuButton_)
    {
        for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
            delete anim_hide_Checkbox_[i];
        for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
            delete anim_fixmotion_Checkbox_[i];
        for (int i = 0; i < geo.nfiles && i < VRANIM_MAXNOMENUITEMS; i++)
            delete anim_fixtranslation_Checkbox_[i];

        if (dat.ndata > 0)
        {
            for (int i = 0; i < dat.ndata; i++)
            {
                delete plotHandle[i];
                delete plotItem[i];
            }
            delete plotHandle;
            delete plotItem;
        }

        delete multiBodyMenuButton_;
        delete multiBodyMenu_;

        delete anim_shaderadio_Button_;
        delete anim_shaderadio_Menu_;
        delete anim_shadewire_Checkbox_;
        delete anim_shadeflexwire_Checkbox_;

        delete anim_shadeoff_Checkbox_;
        delete anim_shadeunlighted_Checkbox_;
        delete anim_shadeflat_Checkbox_;
        delete anim_shadegouraud_Checkbox_;
        delete anim_shade_Radio_Group_;

        delete anim_shadeflexoff_Checkbox_;
        delete anim_shadeflexunlighted_Checkbox_;
        delete anim_shadeflexflat_Checkbox_;
        delete anim_shadeflexgouraud_Checkbox_;
        delete anim_shadeflex_Radio_Group_;

        delete anim_interval_Button_;
        delete anim_calcstride_Button_;

        delete anim_savetrafo_Button_;
        delete anim_showcoordsystem_Checkbox_;
        if (sensor.nr > 0)
            delete anim_showsensors_Checkbox_;
        if (dat.ndata > 0)
            delete anim_showplotters_Checkbox_;

        delete anim_hideradio_Button_;
        delete anim_nohide_Checkbox_;
        delete anim_hide_Checkbox_;
        delete anim_hideradio_Menu_;
        delete anim_fixmotionradio_Button_;

        delete anim_nofixmotion_Checkbox_;
        delete anim_fixmotion_Radio_Group_;
        delete anim_fixmotionradio_Menu_;
        delete anim_fixmotion_Checkbox_;

        delete anim_fixtranslationradio_Button_;

        delete anim_nofixtranslation_Checkbox_;
        delete anim_fixtranslation_Radio_Group_;
        delete anim_fixtranslation_Checkbox_;
        delete anim_fixtranslationradio_Menu_;
    }
}

//--------------------------------------------------------------------
void MultiBodyPlugin::scenegr_create(void)
{
    vranimglobalTransform_ = new osg::MatrixTransform;
    osg::Matrix m;
    //m.makeRotate(90.0* M_PI / 180.0, 1, 0, 0);
    //vranimglobalTransform_->setMatrix(m);
    vranimroot_->addChild(vranimglobalTransform_.get());

    vranimfixedTransform_ = new osg::MatrixTransform;
    vranimglobalTransform_->addChild(vranimfixedTransform_.get());

    vranimallbodiesGroup_ = new osg::Group;
    vranimfixedTransform_->addChild(vranimallbodiesGroup_.get());

    vranimbodyGroup_ = new osg::Group;
    vranimallbodiesGroup_->addChild(vranimbodyGroup_.get());

    vranimelbodyGroup_ = new osg::Group;
    vranimallbodiesGroup_->addChild(vranimelbodyGroup_.get());

    vranimballGroup_ = new osg::Group;
    vranimallbodiesGroup_->addChild(vranimballGroup_.get());
    //vranimroot_->addChild(vranimballGroup_.get());

    vranimivGroup_ = new osg::Group;
    vranimallbodiesGroup_->addChild(vranimivGroup_.get());

    vranimlinElGroup_ = new osg::Group;
    vranimallbodiesGroup_->addChild(vranimlinElGroup_.get());

    vranimsensorSwitch_ = new osg::Switch;
    vranimsensorSwitch_->setAllChildrenOff();
    vranimallbodiesGroup_->addChild(vranimsensorSwitch_.get());

    vranimsensorGroup_ = new osg::Group;
    vranimsensorSwitch_->addChild(vranimsensorGroup_.get());

    vranimbodyTransform_ = new osg::ref_ptr<osg::MatrixTransform>[geo.nfiles];

    if (geo.nfiles > 0)
    {

        vranimbodygeoSwitch_ = new osg::ref_ptr<osg::Switch>[geo.nfiles];
        vranimbodyGeometry_ = new osg::ref_ptr<osg::Geode>[geo.nfiles];

        vranimbodycsSwitch_ = new osg::ref_ptr<osg::Switch>[geo.nfiles];

        vranimbodywireSwitch_ = new osg::ref_ptr<osg::Switch>[geo.nfiles];
        vranimbodywireGeometry_ = new osg::ref_ptr<osg::Geode>[geo.nfiles];

        for (int i = 0; i < geo.nfiles; i++)
        {
            vranimbodyTransform_[i] = new osg::MatrixTransform;
            vranimbodyGroup_->addChild(vranimbodyTransform_[i].get());

            vranimbodygeoSwitch_[i] = new osg::Switch;
            vranimbodygeoSwitch_[i]->setAllChildrenOn();
            vranimbodyTransform_[i]->addChild(vranimbodygeoSwitch_[i].get());

            vranimbodywireSwitch_[i] = new osg::Switch;
            vranimbodywireSwitch_[i]->setAllChildrenOff();
            vranimbodyTransform_[i]->addChild(vranimbodywireSwitch_[i].get());

            vranimbodyGeometry_[i] = createBodyGeometry(i, CREATE_GEO);
            vranimbodygeoSwitch_[i]->addChild(vranimbodyGeometry_[i].get());

            vranimbodywireGeometry_[i] = createBodyGeometry(i, CREATE_WIRE);
            vranimbodywireSwitch_[i]->addChild(vranimbodywireGeometry_[i].get());
        }

        vranimbodycsGeometry_ = new osg::MatrixTransform();
        vranimbodycsGeometry_->addChild(coVRFileManager::instance()->loadIcon("Axis"));
        osg::Matrix scaleMat;
        scaleMat.makeScale(osg::Vec3(0.001, 0.001, 0.001));
        vranimbodycsGeometry_->setMatrix(scaleMat);
        for (int i = 0; i < geo.nfiles; i++)
        {
            vranimbodycsSwitch_[i] = new osg::Switch;
            vranimbodycsSwitch_[i]->setAllChildrenOff();
            vranimbodyTransform_[i]->addChild(vranimbodycsSwitch_[i].get());
            vranimbodycsSwitch_[i]->addChild(vranimbodycsGeometry_.get());
        }
    }

    if (elgeo.nfiles > 0)
    {

        vranimelbodygeoSwitch_ = new osg::ref_ptr<osg::Switch>[elgeo.nfiles];
        vranimelbodywireSwitch_ = new osg::ref_ptr<osg::Switch>[elgeo.nfiles];
        vranimelbodyGeometry_ = new osg::ref_ptr<osg::Geode>[elgeo.nfiles];
        vranimelbodywireGeometry_ = new osg::ref_ptr<osg::Geode>[elgeo.nfiles];

        global_vert = new osg::Vec3Array *[elgeo.nfiles];

        for (int i = 0; i < elgeo.nfiles; i++)
            global_vert[i] = NULL;

        for (int i = 0; i < elgeo.nfiles; i++)
        {
            vranimelbodygeoSwitch_[i] = new osg::Switch;
            vranimelbodygeoSwitch_[i]->setAllChildrenOn();
            vranimelbodyGroup_->addChild(vranimelbodygeoSwitch_[i].get());

            vranimelbodywireSwitch_[i] = new osg::Switch;
            vranimelbodywireSwitch_[i]->setAllChildrenOn();
            vranimelbodyGroup_->addChild(vranimelbodywireSwitch_[i].get());

            vranimelbodyGeometry_[i] = createelBodyGeometry(i, CREATE_GEO);
            vranimelbodygeoSwitch_[i]->addChild(vranimelbodyGeometry_[i].get());

            vranimelbodywireGeometry_[i] = createelBodyGeometry(i, CREATE_WIRE);
            vranimelbodywireSwitch_[i]->addChild(vranimelbodywireGeometry_[i].get());
        }
    }

    if (geo.nballs > 0)
    {

        //	float      color3[3];
        int col;
        osg::Geode *geode = new osg::Geode();
        geode->setName("balls");

        osg::StateSet *geoState = geode->getOrCreateStateSet();

        setDefaultMaterial(geoState, true);

        osg::BlendFunc *blendFunc = new osg::BlendFunc();
        blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
        geoState->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
        osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
        alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);
        //blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
        geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);
        geode->setStateSet(geoState);

        sphere = new coSphere();
        sphere->setNumberOfSpheres(geo.nballs);
        sphere->updateRadii(geo.ballsradius);

        float xmin = FLT_MAX, ymin = FLT_MAX, zmin = FLT_MAX, xmax = -FLT_MAX, ymax = -FLT_MAX, zmax = -FLT_MAX;
        // TODO filter out some spheres
        for (int i = 0; i < geo.nballs; i++)
        {
            for (int t = 0; t < str.timesteps; t++)
            {
                float xp, yp, zp;
                xp = str.a_ball[t][i][12];
                yp = str.a_ball[t][i][13];
                zp = str.a_ball[t][i][14];
                if (xp + geo.ballsradius[i] > xmax)
                    xmax = xp + geo.ballsradius[i];
                if (yp + geo.ballsradius[i] > ymax)
                    ymax = yp + geo.ballsradius[i];
                if (zp + geo.ballsradius[i] > zmax)
                    zmax = zp + geo.ballsradius[i];
                if (xp - geo.ballsradius[i] < xmin)
                    xmin = xp - geo.ballsradius[i];
                if (yp - geo.ballsradius[i] < ymin)
                    ymin = yp - geo.ballsradius[i];
                if (zp - geo.ballsradius[i] < zmin)
                    zmin = zp - geo.ballsradius[i];
            }
        }
        sphere->overrideBoundingBox(osg::BoundingBox(osg::Vec3(xmin, ymin, zmin), osg::Vec3(xmax, ymax, zmax)));

        for (int i = 0; i < geo.nballs; i++)
        {
            col = geo.ballscolor[i];
            if (col >= 0 && col < MAXCOLORS)
            {
                //ballcolor(color3,str.balldyncolor[str.act_step][i]);
                //sphere->setColor(i,color3[0], color3[1], color3[2], 1.0f);
                //sphere->setColor(i,(*colArr)[col][0],(*colArr)[col][1],(*colArr)[col][2],1.0f);
                sphere->setColor(i, colorindex[col][0], colorindex[col][1], colorindex[col][2], 1.0f);
            }
            else
            {
                sphere->setColor(i, 1.0f, 1.0f, 1.0f, 1.0f);
            }
        }

        geode->addDrawable(sphere);
        vranimballGroup_->addChild(geode);

        /*
		vranimballTransform_ = new osg::ref_ptr<osg::MatrixTransform>[geo.nballs];
		vranimballGeometry_= new osg::ref_ptr<osg::Geode>[geo.nballs];
		for (int i=0; i< geo.nballs; i++){
			vranimballTransform_[i] = new osg::MatrixTransform;
			vranimballGroup_->addChild(vranimballTransform_[i].get());
			vranimballGeometry_[i] = createBallGeometry(i);
			vranimballTransform_[i]->addChild(vranimballGeometry_[i].get());
		}*/
    }

    if (iv.nfiles > 0)
    {
        vranimivTransform_ = new osg::ref_ptr<osg::MatrixTransform>[iv.nfiles];
        vranimivGeometry_ = new osg::ref_ptr<osg::Node>[iv.nfiles];
        for (int i = 0; i < iv.nfiles; i++)
        {
            vranimivTransform_[i] = new osg::MatrixTransform;
            vranimivGroup_->addChild(vranimivTransform_[i].get());
            vranimivGeometry_[i] = osgDB::readNodeFile(iv.name[i]);
            vranimivTransform_[i]->addChild(vranimivGeometry_[i].get());
        }
    }

    if (sensor.nr > 0)
    {
        vranimsensorGeometry_ = new osg::ref_ptr<osg::Geode>[sensor.nr];
        for (int i = 0; i < sensor.nr; i++)
        {
            vranimsensorGeometry_[i] = createSensorGeometry(i);
            vranimsensorGroup_->addChild(vranimsensorGeometry_[i].get());
        }
    }

    if (lin.nr > 0)
    {
        vranimlinElTransform_ = new osg::ref_ptr<osg::MatrixTransform>[lin.nr];
        if (lin.n_iv > 0)
        {
            vranimlinivGeometry_ = new osg::ref_ptr<osg::Node>[lin.n_iv];
        }
        if ((lin.nr - lin.n_iv) > 0)
        {
            vranimlinstdGeometry_ = new osg::ref_ptr<osg::Geode>[(lin.nr - lin.n_iv)];
        }
        int count_iv = 0; //count iv-Nodes added
        for (int i = 0; i < lin.nr; i++)
        {
            vranimlinElTransform_[i] = new osg::MatrixTransform;
            vranimlinElGroup_->addChild(vranimlinElTransform_[i].get());
            if (lin.type[i] == 3)
            {
                vranimlinivGeometry_[count_iv] = osgDB::readNodeFile(lin.name[i]);
                fprintf(outfile, "%s\n", lin.name[i]);
                vranimlinElTransform_[i]->addChild(vranimlinivGeometry_[count_iv].get());
                count_iv++;
            }
            else
            {
                vranimlinstdGeometry_[i - count_iv] = createlinstdGeometry(i);
                vranimlinElTransform_[i]->addChild(vranimlinstdGeometry_[i - count_iv].get());
            }
        }
    }

    update();
}

void MultiBodyPlugin::setDefaultMaterial(osg::StateSet *geoState, bool transparent)
{
    if (globalmtl.get() == NULL)
    {
        globalmtl = new osg::Material;
        globalmtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
        globalmtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
        globalmtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0));
        globalmtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.4f, 0.4f, 0.4f, 1.0));
        globalmtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
        globalmtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    }

    {
        geoState->setAttributeAndModes(globalmtl.get(), osg::StateAttribute::ON);
        if (transparent)
        {
            geoState->setMode(GL_BLEND, osg::StateAttribute::ON);
            geoState->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
            geoState->setNestRenderBins(false);
        }
        else
        {
            geoState->setMode(GL_BLEND, osg::StateAttribute::OFF);
            geoState->setRenderingHint(osg::StateSet::OPAQUE_BIN);
            geoState->setNestRenderBins(false);
        }
    }
    geoState->setMode(GL_LIGHTING, osg::StateAttribute::ON);
}

//--------------------------------------------------------------------
void MultiBodyPlugin::scenegr_delete(void)
{
}

//--------------------------------------------------------------------
void MultiBodyPlugin::vranimtab_create(void)
{
    vranimTab = new coTUITab("VRAnim", coVRTui::instance()->mainFolder->getID());
    vranimTab->setPos(0, 0);

    infoLabel = new coTUILabel("VRAnim Version " VERSION, vranimTab->getID());
    infoLabel->setPos(0, 0);

    leftFrame = new coTUIFrame("leftFrame", vranimTab->getID());
    leftFrame->setPos(0, 1);

    bodiesFrame = new coTUIFrame("bodiesFrame", leftFrame->getID());
    bodiesFrame->setPos(0, 0);
    bodiesFrame->setShape(6);
    bodiesFrame->setStyle(0x20);

    hideLabel = new coTUILabel("Hide", bodiesFrame->getID());
    hideLabel->setPos(1, 0);
    fixMotionLabel = new coTUILabel("Fix Motion", bodiesFrame->getID());
    fixMotionLabel->setPos(2, 0);
    fixTranslationLabel = new coTUILabel("Fix Translation", bodiesFrame->getID());
    fixTranslationLabel->setPos(3, 0);
    bodyLabel = new coTUILabel *[geo.nfiles];
    hideBodyButton = new coTUIToggleButton *[geo.nfiles];
    fixMotionButton = new coTUIToggleButton *[geo.nfiles];
    fixTranslationButton = new coTUIToggleButton *[geo.nfiles];

    noBodyLabel = new coTUILabel("NOT ANY BODIES", bodiesFrame->getID());
    noBodyLabel->setPos(0, 1);

    doNotHideButton = new coTUIToggleButton("CheckBox", bodiesFrame->getID());
    doNotHideButton->setEventListener(this);
    doNotHideButton->setState(true);
    doNotHideButton->setPos(1, 1);

    doNotFixRadioButton = new coTUIToggleButton("RadioButton", bodiesFrame->getID());
    doNotFixRadioButton->setEventListener(this);
    doNotFixRadioButton->setSize(2, 1);
    doNotFixRadioButton->setState(true);
    doNotFixRadioButton->setPos(2, 1);

    for (int i = 0; i < geo.nfiles; i++)
    {
        bodyLabel[i] = new coTUILabel(geo.name[i], bodiesFrame->getID());
        bodyLabel[i]->setPos(0, i + 2);
        hideBodyButton[i] = new coTUIToggleButton("CheckBox", bodiesFrame->getID());
        hideBodyButton[i]->setEventListener(this);
        hideBodyButton[i]->setState(false);
        hideBodyButton[i]->setPos(1, i + 2);
        fixMotionButton[i] = new coTUIToggleButton("RadioButton", bodiesFrame->getID());
        fixMotionButton[i]->setEventListener(this);
        fixMotionButton[i]->setState(false);
        fixMotionButton[i]->setPos(2, i + 2);
        fixTranslationButton[i] = new coTUIToggleButton("RadioButton", bodiesFrame->getID());
        fixTranslationButton[i]->setEventListener(this);
        fixTranslationButton[i]->setState(false);
        fixTranslationButton[i]->setPos(3, i + 2);
    }

    rightFrame = new coTUIFrame("rightFrame", vranimTab->getID());
    rightFrame->setPos(1, 1);

    // show
    showFrame = new coTUIFrame("showFrame", rightFrame->getID());
    showFrame->setPos(0, 0);
    showFrame->setShape(6);
    showFrame->setStyle(0x20);

    showLabel = new coTUILabel("Show", showFrame->getID());
    showLabel->setPos(0, 0);

    if (sensor.nr > 0)
    {
        showSensorsButton = new coTUIToggleButton("sensors", showFrame->getID());
        showSensorsButton->setEventListener(this);
        showSensorsButton->setState(false);
        showSensorsButton->setPos(0, 1);
    }

    if (dat.ndata > 0)
    {
        showPlottersButton = new coTUIToggleButton("plotters", showFrame->getID());
        showPlottersButton->setEventListener(this);
        showPlottersButton->setState(false);
        showPlottersButton->setPos(0, 2);
    }

    showCoordSystemsButton = new coTUIToggleButton("coord. systems", showFrame->getID());
    showCoordSystemsButton->setEventListener(this);
    showCoordSystemsButton->setState(false);
    showCoordSystemsButton->setPos(0, 3);

    //shading
    shadingFrame = new coTUIFrame("shadingFrame", rightFrame->getID());
    shadingFrame->setPos(0, 1);
    shadingFrame->setShape(6);
    shadingFrame->setStyle(0x20);

    shadingLabel = new coTUILabel("Shading", shadingFrame->getID());
    shadingLabel->setPos(0, 0);

    rigidComboBox = new coTUIComboBox("rigid", shadingFrame->getID());
    rigidComboBox->setEventListener(this);
    rigidComboBox->addEntry("rigid off");
    rigidComboBox->addEntry("rigid unlighted");
    rigidComboBox->addEntry("rigid flat");
    rigidComboBox->addEntry("rigid gouraud");
    rigidComboBox->setSelectedEntry(3);
    rigidComboBox->setPos(0, 1);

    rigidBodiesAsWireButton = new coTUIToggleButton("rigid bodies as wire", shadingFrame->getID());
    rigidBodiesAsWireButton->setEventListener(this);
    rigidBodiesAsWireButton->setPos(0, 2);

    flexComboBox = new coTUIComboBox("flex", shadingFrame->getID());
    flexComboBox->setEventListener(this);
    flexComboBox->addEntry("flex off");
    flexComboBox->addEntry("flex unlighted");
    flexComboBox->addEntry("flex flat");
    flexComboBox->addEntry("flex gouraud");
    flexComboBox->setSelectedEntry(2);
    flexComboBox->setPos(0, 3);

    flexBodiesAsWireButton = new coTUIToggleButton("flex bodies as wire", shadingFrame->getID());
    flexBodiesAsWireButton->setEventListener(this);
    flexBodiesAsWireButton->setPos(0, 4);

    //trafo
    trafoFrame = new coTUIFrame("trafoFrame", rightFrame->getID());
    trafoFrame->setPos(0, 2);
    trafoFrame->setShape(6);
    trafoFrame->setStyle(0x20);

    saveTrafoButton = new coTUIButton("save trafo", trafoFrame->getID());
    saveTrafoButton->setEventListener(this);
    saveTrafoButton->setPos(0, 0);
}

//--------------------------------------------------------------------
void MultiBodyPlugin::vranimtab_delete(void)
{
    if (vranimTab)
    {
        delete infoLabel;

        delete hideLabel;
        delete fixMotionLabel;
        delete fixTranslationLabel;
        delete noBodyLabel;
        delete doNotHideButton;
        delete doNotFixRadioButton;
        for (int i = 0; i < geo.nfiles; i++)
        {
            delete bodyLabel[i];
            delete hideBodyButton[i];
            delete fixMotionButton[i];
            delete fixTranslationButton[i];
        }
        delete bodyLabel;
        delete hideBodyButton;
        delete fixMotionButton;
        delete fixTranslationButton;
        delete bodiesFrame;

        delete showLabel;
        if (sensor.nr > 0)
            delete showSensorsButton;
        if (dat.ndata > 0)
            delete showPlottersButton;
        delete showCoordSystemsButton;
        delete showFrame;

        delete shadingLabel;
        delete rigidComboBox;
        delete rigidBodiesAsWireButton;
        delete flexComboBox;
        delete flexBodiesAsWireButton;
        delete shadingFrame;

        delete saveTrafoButton;
        delete trafoFrame;

        delete leftFrame;
        delete rightFrame;
        delete vranimTab;
    }
}

//--------------------------------------------------------------------
osg::Geode *MultiBodyPlugin::createSensorGeometry(int sensorId)
{
    //fprintf(outfile,
    // "--- MultiBodyPlugin::createSensorGeometry (%d %s) called\n",
    //	  bodyId, geo.name[sensorId]);

    osg::Geode *geode;

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setUseDisplayList(coVRConfig::instance()->useDisplayLists());
    geom->setUseVertexBufferObjects(coVRConfig::instance()->useVBOs());
    osg::DrawArrayLengths *primitives;
    primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::LINE_STRIP);

    osg::Vec3Array *vert = new osg::Vec3Array;
    for (int j = 0; j < str.timesteps; j++)
    {
        vert->push_back(osg::Vec3(sensor.pkt[sensorId][j][0],
                                  sensor.pkt[sensorId][j][1],
                                  sensor.pkt[sensorId][j][2]));
    }
    geom->setVertexArray(vert);
    primitives->push_back(str.timesteps);
    geom->addPrimitiveSet(primitives);

    osg::Vec4Array *localColArr = new osg::Vec4Array();
    localColArr->push_back(osg::Vec4(colorindex[sensor.col[sensorId]][0], colorindex[sensor.col[sensorId]][1], colorindex[sensor.col[sensorId]][2], colorindex[sensor.col[sensorId]][3]));
    geom->setColorArray(localColArr);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    geode = new osg::Geode();
    geode->addDrawable(geom.get());
    geode->setName("sensor");
    geode->setStateSet(unlightedStateSet.get());
    return (geode);
}

//--------------------------------------------------------------------
osg::Geode *MultiBodyPlugin::createlinstdGeometry(int linElId)
{
    //fprintf(outfile,
    // "--- MultiBodyPlugin::createlinElGeometry (%d) called\n",
    //	  linElId);

    osg::Geode *geode;
    osg::Vec3Array *coordList, *normalArray;
    osg::Vec3f c1, c2;
    osg::UShortArray *coordIndexList, *colorIndexList;
    osg::Matrix scaleMat, rotMat, transMat;

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setUseDisplayList(false);
    geom->setUseVertexBufferObjects(false);

    osg::DrawArrayLengths *primitives; // = lenList
    primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::POLYGON);

    /*create elastic band*/
    if (lin.type[linElId] == 1 || lin.type[linElId] == 2)
    {

        // 8 vertices
        coordList = new osg::Vec3Array();
        coordList->push_back(osg::Vec3(0, -0.1, -0.1));
        coordList->push_back(osg::Vec3(0, 0.1, -0.1));
        coordList->push_back(osg::Vec3(0, 0.1, 0.1));
        coordList->push_back(osg::Vec3(0, -0.1, 0.1));
        coordList->push_back(osg::Vec3(1, -0.1, -0.1));
        coordList->push_back(osg::Vec3(1, 0.1, -0.1));
        coordList->push_back(osg::Vec3(1, 0.1, 0.1));
        coordList->push_back(osg::Vec3(1, -0.1, 0.1));
        geom->setVertexArray(coordList);

        // 6 polygons
        for (int i = 0; i < 6; i++)
        {
            primitives->push_back(4);
        }

        // 6*4 polygon-vertices
        coordIndexList = new osg::UShortArray();
        coordIndexList->push_back(0);
        coordIndexList->push_back(3);
        coordIndexList->push_back(2);
        coordIndexList->push_back(1);
        coordIndexList->push_back(2);
        coordIndexList->push_back(3);
        coordIndexList->push_back(7);
        coordIndexList->push_back(6);
        coordIndexList->push_back(4);
        coordIndexList->push_back(5);
        coordIndexList->push_back(6);
        coordIndexList->push_back(7);
        coordIndexList->push_back(4);
        coordIndexList->push_back(0);
        coordIndexList->push_back(1);
        coordIndexList->push_back(5);
        coordIndexList->push_back(2);
        coordIndexList->push_back(6);
        coordIndexList->push_back(5);
        coordIndexList->push_back(1);
        coordIndexList->push_back(7);
        coordIndexList->push_back(3);
        coordIndexList->push_back(0);
        coordIndexList->push_back(4);
        geom->setVertexIndices(coordIndexList);

        // 6 normals
        normalArray = new osg::Vec3Array();
        normalArray->push_back(osg::Vec3(-1, 0, 0));
        normalArray->push_back(osg::Vec3(0, 0, 1));
        normalArray->push_back(osg::Vec3(1, 0, 0));
        normalArray->push_back(osg::Vec3(0, 0, -1));
        normalArray->push_back(osg::Vec3(0, -1, 0));
        normalArray->push_back(osg::Vec3(0, 1, 0));
        geom->setNormalArray(normalArray);
        geom->setNormalBinding(osg::Geometry::BIND_PER_PRIMITIVE);

        // 6 colors
        colorIndexList = new osg::UShortArray();
        for (int i = 0; i < 6; i++)
        {
            colorIndexList->push_back(lin.color[linElId]);
        }
        geom->setColorArray(colArr);
        geom->setColorBinding(osg::Geometry::BIND_PER_PRIMITIVE);
        geom->setColorIndices(colorIndexList);

        // scale, rotate and translate
        float cx = lin.dir[linElId][0][0] / lin.length[linElId][0];
        float cy = lin.dir[linElId][0][1] / lin.length[linElId][0];
        float cz = lin.dir[linElId][0][2] / lin.length[linElId][0];
        c2.set(cx, cy, cz);
        c1.set(1.0, 0.0, 0.0);
        scaleMat.makeScale(lin.length[linElId][0], 1.0, 1.0);
        rotMat.makeRotate(c1, c2);
        transMat.makeTranslate(lin.pkt1[linElId][0][0], lin.pkt1[linElId][0][1], lin.pkt1[linElId][0][2]);

        //draw polygons
        geom->addPrimitiveSet(primitives);
    }

    /*Create spring*/
    /*(Geometry of spring was defined by using mkobject)*/
    //  if(lin.type[linElId]==2) ..........

    geode = new osg::Geode();
    geode->addDrawable(geom.get());
    geode->setName("lineElement");
    geode->setStateSet(shadedStateSet.get());
    return (geode);
}

//--------------------------------------------------------------------

COVERPLUGIN(MultiBodyPlugin)
