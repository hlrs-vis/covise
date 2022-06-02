/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\
**                                                            (C)2005 HLRS  **
**                                                                          **
** Description: RecordPath Plugin (records viewpoints and viewing directions and targets)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** April-05  v1	    				       		                         **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "TrajectoriesPlugin.h"
#define USE_MATH_DEFINES
#include <math.h>
#include <config/coConfig.h>
#include <config/CoviseConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRFileManager.h>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/LineWidth>

#include <osg/LineSegment>
#include <osg/Matrix>
#include <osg/Vec3>
#include <cover/coVRAnimationManager.h>
#define MAXSAMPLES 1200
using namespace osg;
using namespace osgUtil;
#include <stdio.h> /* gets, etc. */
#include <stdlib.h> /* exit       */
#include <string.h> /* strcpy     */

TrajectoriesPlugin *TrajectoriesPlugin::thePlugin = NULL;

TrajectoriesPlugin *TrajectoriesPlugin::instance()
{
    if (!thePlugin)
        thePlugin = new TrajectoriesPlugin();
    return thePlugin;
}

TrajectoriesPlugin::TrajectoriesPlugin()
{
    thePlugin = this;
}

static FileHandler handlers[] = {
    { NULL,
      TrajectoriesPlugin::sloadTrajectories,
      TrajectoriesPlugin::unloadTrajectories,
      "gcode" }
};

int TrajectoriesPlugin::sloadGCode(const char *filename, osg::Group *loadParent, const char *)
{

    instance()->loadGCode(filename, loadParent);
    return 0;
}

int TrajectoriesPlugin::loadGCode(const char *filename, osg::Group *loadParent)
{

    frameNumber = 0;
    //delete[] positions;
    //positions = new float [3*MAXSAMPLES+3];

    geode = new Geode();
    geom = new Geometry();
    geode->setStateSet(geoState.get());

    geom->setColorBinding(Geometry::BIND_OFF);

    geode->addDrawable(geom.get());
    geode->setName("Viewer Positions");

    // set up geometry
    vert = new osg::Vec3Array;
    color = new osg::Vec4Array;

    int status;
    int do_next; /* 0=continue, 1=mdi, 2=stop */
    int block_delete;
    char buffer[80];
    int tool_flag;
    int gees[RS274NGC_ACTIVE_G_CODES];
    int ems[RS274NGC_ACTIVE_M_CODES];
    double sets[RS274NGC_ACTIVE_SETTINGS];
    char default_name[] SET_TO "rs274ngc.var";
    int print_stack;

    do_next SET_TO 2; /* 2=stop */
    block_delete SET_TO RS_OFF;
    print_stack SET_TO RS_OFF;
    tool_flag SET_TO 0;
    
    const char *varFileName = opencover::coVRFileManager::instance()->getName("share/covise/rs274ngc.var");
    strcpy(_parameter_file_name, varFileName);
    _outfile SET_TO stdout; /* may be reset below */

    fprintf(stderr, "executing\n");
    if (tool_flag IS 0)
    {
        const char *toolFileName = opencover::coVRFileManager::instance()->getName("share/covise/rs274ngc.tool_default");

        if (read_tool_file(toolFileName) ISNT 0)
            exit(1);
    }

    if ((status SET_TO rs274ngc_init())ISNT RS274NGC_OK)
    {
        report_error(status, print_stack);
        exit(1);
    }

    status SET_TO rs274ngc_open(filename);
    if (status ISNT RS274NGC_OK) /* do not need to close since not open */
    {
        report_error(status, print_stack);
        exit(1);
    }
    status SET_TO interpret_from_file(do_next, block_delete, print_stack);
    rs274ngc_file_name(buffer, 5); /* called to exercise the function */
    rs274ngc_file_name(buffer, 79); /* called to exercise the function */
    rs274ngc_close();
    rs274ngc_line_length(); /* called to exercise the function */
    rs274ngc_sequence_number(); /* called to exercise the function */
    rs274ngc_active_g_codes(gees); /* called to exercise the function */
    rs274ngc_active_m_codes(ems); /* called to exercise the function */
    rs274ngc_active_settings(sets); /* called to exercise the function */
    rs274ngc_exit(); /* saves parameters */
    primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
    primitives->push_back(vert->size());

    // Update animation frame:
    coVRAnimationManager::instance()->setNumTimesteps(vert->size(), this);

    geom->setVertexArray(vert);
    geom->setColorArray(color);
    geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    geom->addPrimitiveSet(primitives);
    geom->dirtyDisplayList();
    geom->setUseDisplayList(false);
    parentNode = loadParent;
    if (parentNode == NULL)
        parentNode = cover->getObjectsRoot();
    parentNode->addChild(geode.get());
    ;
    return 0;
}

//--------------------------------------------------------------------
void TrajectoriesPlugin::setTimestep(int t)
{
    if (primitives)
        primitives->at(0) = t;
}

int TrajectoriesPlugin::unloadTrajectories(const char *filename, const char *)
{
    (void)filename;

    return 0;
}

void TrajectoriesPlugin::deleteColorMap(const std::string &name)
{
    auto it = mapValues.find(name);
    if (it != mapValues.end())
        delete[] it->second;
    mapSize.erase(name);
    mapValues.erase(name);
}

bool TrajectoriesPlugin::init()
{
    fprintf(stderr, "TrajectoriesPlugin::TrajectoriesPlugin\n");

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);

    recordRate = 1;
    primitives = NULL;

    coConfig *config = coConfig::getInstance();

    // read the name of all colormaps in file
    covise::coCoviseConfig::ScopeEntries keys = coCoviseConfig::getScopeEntries("Colormaps");
#ifdef NO_COLORMAP_PARAM
    mapNames.push_back("COVISE");
#else
    mapNames.push_back("Editable");
#endif
    for (const auto &key: keys)
        mapNames.push_back(key.first);

    // read the values for each colormap
    for (int k = 1; k < mapNames.size(); k++)
    {
        // get all definition points for the colormap
        string name = "Colormaps." + mapNames[k];
        coCoviseConfig::ScopeEntries keys = coCoviseConfig::getScopeEntries(name);

        int no = keys.size();
        mapSize.emplace(mapNames[k], no);
        float *cval = new float[no * 5];
        mapValues.emplace(mapNames[k], cval);

        // read all sampling points
        float diff = 1.0f / (no - 1);
        float pos = 0.0f;
        float *cur = cval;
        for (int j = 0; j < no; j++)
        {
            ostringstream out;
            out << j;
            string tmp = name + ".Point:" + out.str();

            bool rgb = false;
            string rgba = coCoviseConfig::getEntry("rgba", tmp);
            if (rgba.empty())
            {
                rgb = true;
                rgba = coCoviseConfig::getEntry("rgb", tmp);
            }
            if (!rgba.empty())
            {
                float a = 1.;
                uint32_t c = strtol(rgba.c_str(), NULL, 16);
                if (!rgb)
                {
                    a = (c & 0xff) / 255.0f;
                    c >>= 8;
                }
                float b = (c & 0xff) / 255.0f;
                c >>= 8;
                float g = (c & 0xff) / 255.0f;
                c >>= 8;
                float r = (c & 0xff) / 255.0f;
                float x = coCoviseConfig::getFloat("x", tmp, pos);
                *cur++ = r;
                *cur++ = g;
                *cur++ = b;
                *cur++ = a;
                *cur++ = x;
            }
            else
            {
                *cur++ = 1.;
                *cur++ = 1.;
                *cur++ = 1.;
                *cur++ = 1.;
                *cur++ = pos;
            }
            pos = pos + diff;
        }
    }
    TrajectoriesTab = new ui::Menu("Midi", this);

    play = new ui::Button(TrajectoriesTab, "Play");
    play->setText("Play");
    play->setCallback([this](bool) {
        
        });
    frameNumber = 0;
    play->setState(false);
    playing = false;

    geoState = new osg::StateSet();
    linemtl = new osg::Material;
    lineWidth = new osg::LineWidth(2.0);
    linemtl.get()->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    linemtl.get()->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
    linemtl.get()->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0));
    linemtl.get()->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.0));
    linemtl.get()->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
    linemtl.get()->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    geoState->setAttributeAndModes(linemtl.get(), StateAttribute::ON);

    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    geoState->setAttributeAndModes(lineWidth.get(), StateAttribute::ON);

    return true;
}

// this is called if the plugin is removed at runtime
TrajectoriesPlugin::~TrajectoriesPlugin()
{
    fprintf(stderr, "TrajectoriesPlugin::~TrajectoriesPlugin\n");

    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);

    delete play;
    if (geode && geode->getNumParents() > 0)
    {
        TrajectoriesRoot = geode->getParent(0);
        if (TrajectoriesRoot)
            TrajectoriesRoot->removeChild(geode.get());
    }
}

bool
TrajectoriesPlugin::update()
{
    if (play->getState())
    {
    }
}

osg::Vec4 TrajectoriesPlugin::getColor(float pos)
{

    osg::Vec4 actCol;
    int idx = 0;
    //cerr << "name: " << (const char *)mapNames[currentMap].toAscii() << endl;
    float *map = mapValues[mapNames[currentMap]];
    int mapS = mapSize[mapNames[currentMap]];
    if (map == NULL)
    {
        return actCol;
    }
    while (map[(idx + 1) * 5] <= pos)
    {
        idx++;
        if (idx > mapS - 2)
        {
            idx = mapS - 2;
            break;
        }
    }
    double d = (pos - map[idx * 5]) / (map[(idx + 1) * 5] - map[idx * 5]);
    actCol[0] = (float)((1 - d) * map[idx * 5 + 1] + d * map[(idx + 1) * 5 + 1]);
    actCol[1] = (float)((1 - d) * map[idx * 5 + 2] + d * map[(idx + 1) * 5 + 2]);
    actCol[2] = (float)((1 - d) * map[idx * 5 + 3] + d * map[(idx + 1) * 5 + 3]);
    actCol[3] = (float)((1 - d) * map[idx * 5 + 4] + d * map[(idx + 1) * 5 + 4]);

    return actCol;
}

COVERPLUGIN(TrajectoriesPlugin)
