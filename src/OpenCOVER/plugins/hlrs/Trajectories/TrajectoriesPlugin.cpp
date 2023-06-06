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
#include <OpenVRUI/osg/mathUtils.h>
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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

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

TrajectoriesPlugin* TrajectoriesPlugin::thePlugin = NULL;

TrajectoriesPlugin* TrajectoriesPlugin::instance()
{
    if (!thePlugin)
        thePlugin = new TrajectoriesPlugin();
    return thePlugin;
}

TrajectoriesPlugin::TrajectoriesPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("TrajectoriesPlugin", cover->ui)
{
    thePlugin = this;
}
Trajectory::Trajectory()
{
}

int Trajectory::readData(int fd, int& first_start_time)
{

    if (read(fd, &h1, sizeof(h1)) < sizeof(h1))
        return -1;
    if (read(fd, &h2, sizeof(h2)) < sizeof(h2))
        return -1;
    if (strcmp(h1.type, "car") == 0)
    {
        type = T_CAR;
    }
    else if (strcmp(h1.type, "person") == 0)
    {
        type = T_PERSON;
    }
    else if (strcmp(h1.type, "bicycle") == 0)
    {
        type = T_BICYCLE;
    }
    int numBytes = h2.numTimesteps * sizeof(timestep);
    if (h2.numTimesteps < 30) // too short
    {
        lseek(fd, numBytes, SEEK_CUR);
        return -2;
    }
    timesteps = new timestep[h2.numTimesteps];
    int nr = 0;
    while (nr < numBytes)
    {
        int r = read(fd, ((char*)timesteps) + nr, numBytes - nr);
        if (r < 0)
            return -1;
        nr += r;
    }
    if (nr < numBytes)
    {
        return -1;
    }
    //update first start time for the first trajectory
    if (first_start_time == -1)
    {
        first_start_time = h2.firstTimestep;
    }

    // set up geometry
    vert = new osg::Vec3Array();
    vert->resize(h2.numTimesteps);
    for (int i = 0; i < h2.numTimesteps; i++)
    {
        (*vert)[i].set(timesteps[i].x, timesteps[i].y, 0.0);
    }

    geode = new Geode();
    geom = new Geometry();
    geode->setStateSet(TrajectoriesPlugin::instance()->geoState.get());

    geom->setColorBinding(Geometry::BIND_OFF);

    geode->addDrawable(geom.get());
    geode->setName("trajectories");



    primitives = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
    primitives->push_back(vert->size());

    // Update animation frame:
    coVRAnimationManager::instance()->setNumTimesteps((h2.lastTimestep - first_start_time) / TrajectoriesPlugin::timeStepLength, this);

    geom->setVertexArray(vert);
    //geom->setColorArray(color);
    //geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    color = new osg::Vec4Array(1);
    if (type == T_CAR)
    {
        (*color)[0].set(1, 0, 0, 1);
    }
    if (type == T_PERSON)
    {
        (*color)[0].set(0, 1, 0, 1);
    }
    if (type == T_BICYCLE)
    {
        (*color)[0].set(0, 0, 1, 1);
    }
    geom->setColorArray(color);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);
    geom->addPrimitiveSet(primitives);
    geom->dirtyDisplayList();
    geom->setUseDisplayList(false);
    return h2.numTimesteps;
}
Trajectory::~Trajectory()
{
    if (geode && geode->getNumParents() > 0)
    {
        osg::Group* parent = geode->getParent(0);
        if (parent)
            parent->removeChild(geode.get());
    }
    delete[] timesteps;
}

static FileHandler handlers[] = {
    { NULL,
      TrajectoriesPlugin::sloadTrajectories,
      TrajectoriesPlugin::unloadTrajectories,
      "bcrtf" }
};

int TrajectoriesPlugin::sloadTrajectories(const char* filename, osg::Group* loadParent, const char*)
{

    instance()->loadTrajectories(filename, loadParent);
    return 0;
}
int TrajectoriesPlugin::loadTrajectories(const char* filename, osg::Group* loadParent)
{
    int numTrajectories = 0;
    TrajectoriesRoot = new osg::Group();
    if (loadParent != NULL)
    {
        loadParent->addChild(TrajectoriesRoot);
    }
    else {
        TrajectoriesRoot = cover->getObjectsRoot();
    }

    
#ifdef _WIN32
    int fd = open(filename, O_RDONLY | O_BINARY);
#else
    int fd = open(filename, O_RDONLY);
#endif
    if (fd >= 0)
    {
        while (1)//for (int i = 0; i < numTrajectories; i++)
        {
            Trajectory* tr = new Trajectory();
            int numTimesteps = tr->readData(fd, first_start_time);
            if (numTimesteps < 0)
            {
                delete tr;
                if (numTimesteps != -2)
                    break;
            }
            else
            {
                //TrajectoriesRoot->addChild(tr->getGeometry());
                trajectories.push_back(tr);
            }
        }
        fprintf(stderr, "%s num Trajectories: %lu\n", filename, (unsigned long)trajectories.size());
    }
    else
    {
        return -1;
    }
    return -1;
}

//--------------------------------------------------------------------
void TrajectoriesPlugin::setTimestep(int t)
{

    //TrajectoriesRoot = cover->getObjectsRoot();
    double timeStepTime = first_start_time + TrajectoriesPlugin::timeStepLength * t;
    for (const auto& tr : trajectories)
    {

        // tr is in between
        if (tr->h2.firstTimestep < timeStepTime + threshold && timeStepTime - threshold < tr->h2.lastTimestep)
        {
            //display it
            if (tr->h3.visible == 0) {
                TrajectoriesRoot->addChild(tr->getGeometry());
                tr->h3.visible = 1;
            }
        }

        else
        {
            if (tr->h3.visible != 0)
            {
                TrajectoriesRoot->removeChild(tr->getGeometry());
                tr->h3.visible = 0;
            }

        }
    }
}



int TrajectoriesPlugin::unloadTrajectories(const char* filename, const char*)
{
    (void)filename;

    return 0;
}

void TrajectoriesPlugin::deleteColorMap(const std::string& name)
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

    coConfig* config = coConfig::getInstance();

    // read the name of all colormaps in file
    covise::coCoviseConfig::ScopeEntries keys = coCoviseConfig::getScopeEntries("Colormaps");
#ifdef NO_COLORMAP_PARAM
    mapNames.push_back("COVISE");
#else
    mapNames.push_back("Editable");
#endif
    for (const auto& key : keys)
        mapNames.push_back(key.first);

    // read the values for each colormap
    for (int k = 1; k < mapNames.size(); k++)
    {
        // get all definition points for the colormap
        string name = "Colormaps." + mapNames[k];
        coCoviseConfig::ScopeEntries keys = coCoviseConfig::getScopeEntries(name);

        int no = keys.size();
        mapSize.emplace(mapNames[k], no);
        float* cval = new float[no * 5];
        mapValues.emplace(mapNames[k], cval);

        // read all sampling points
        float diff = 1.0f / (no - 1);
        float pos = 0.0f;
        float* cur = cval;
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
    for (const auto& t : trajectories)
    {
        delete t;
    }
}

bool
TrajectoriesPlugin::update()
{
    if (play->state())
    {
        return true;
    }
    return false;
}

osg::Vec4 TrajectoriesPlugin::getColor(float pos)
{

    osg::Vec4 actCol;
    int idx = 0;
    //cerr << "name: " << (const char *)mapNames[currentMap].toAscii() << endl;
    float* map = mapValues[mapNames[currentMap]];
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
