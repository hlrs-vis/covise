/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RecordPath_PLUGIN_H
#define _RecordPath_PLUGIN_H
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
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>

#include <osg/Geode>
#include <osg/ref_ptr>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/LineWidth>
#include <PluginUtil/coSphere.h>
#include <map>
#include <string>
#include <vector>
#include <stdio.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Button.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Label.h>
#include <cover/ui/Slider.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Owner.h>

using namespace covise;
using namespace opencover;
class Trajectory
{
    enum trajectoryType {
        T_CAR,
        T_PERSON,
        T_BICYCLE
    };
#pragma pack(push, 1)
    struct header1 {
        int32_t globalID;
        int32_t perClassID;
        char type[21];
    };
    struct header2 {
        int32_t numTimesteps;
        double firstTimestep;
        double lastTimestep;
    };
    struct header3 {
        bool visible = 0;
    };
#pragma pack(pop)
    struct timestep {
        float x;
        float y;
        double timestep;
    };

    timestep* timesteps = nullptr;
    osg::Vec3Array* vert;
    osg::Vec4Array* color;
    osg::DrawArrayLengths* primitives;
    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;
    double startTime;
    double endTime;
    int numPoints;
    int objectType;
    osg::Vec3Array* coordinates;
    double* timestamps;
    trajectoryType type;
public:
    header1 h1;
    header2 h2;
    header3 h3;
    Trajectory();
    ~Trajectory();
    int readData(int fd, int& first_start_time);
    osg::Geode* getGeometry() { return geode; };

};

class TrajectoriesPlugin : public coVRPlugin, public ui::Owner
{
public:
    TrajectoriesPlugin();
    virtual ~TrajectoriesPlugin();
    static TrajectoriesPlugin* instance();
    bool init();

    int loadTrajectories(const char* filename, osg::Group* loadParent);
    static int sloadTrajectories(const char* filename, osg::Group* loadParent, const char* covise_key);
    static int unloadTrajectories(const char* filename, const char* covise_key);


    bool update();

    ui::Menu* TrajectoriesTab = nullptr;
    ui::Button* play = nullptr;
    ui::Label* numTrajectories = nullptr;




    std::vector<std::string> mapNames;
    std::map<std::string, int> mapSize;
    std::map<std::string, float*> mapValues;

    osg::ref_ptr<osg::StateSet> geoState;
    osg::ref_ptr<osg::Material> linemtl;
    osg::ref_ptr<osg::LineWidth> lineWidth;
    void setTimestep(int t);

    osg::Vec4 getColor(float pos);
    void deleteColorMap(const std::string& name);
    static const int timeStepLength = 60;
    static const int threshold = 10;
    int first_start_time = -1;
private:
    bool playing;
    double recordRate;
    int currentMap;
    std::list< Trajectory*> trajectories;
    osg::Group* TrajectoriesRoot;

    static TrajectoriesPlugin* thePlugin;


};
#endif
