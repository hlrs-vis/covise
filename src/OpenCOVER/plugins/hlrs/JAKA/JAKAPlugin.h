/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#ifndef _JAKAPlugin_H
#define _JAKAPlugin_H
/****************************************************************************\
**                                                            (C)2009 HLRS  **
**                                                                          **
** Description: Varant plugin                                               **
**                                                                          **
this plugin uses the "JAKAPlugin" attribute, setted from the VarianMarker module
to show/hide several JAKAPlugins in the cover menu (JAKAPlugins item)
**                                                                          **
** Author: A.Gottlieb                                                       **
**                                                                          **
** History:                                                                 **
** Jul-09  v1                                                               **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include <cover/coVRPlugin.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <cover/coVRSelectionManager.h>
#include "cover/coVRPluginSupport.h"
#include <util/coExport.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <cover/coVRLabel.h>
#include <util/UDPComm.h>
#include "JAKAZuRobot.h"
#include <OpenThreads/Thread>
#include <OpenThreads/Mutex>

#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Manager.h>


using namespace covise;
using namespace opencover;
using namespace vrui;
using namespace std;
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

#pragma pack(push, 1)
struct linearMessage
{
    int size;
    int type;
    float x;
    float y;
    float z;
    float h;
    float p;
    float r;
};
#pragma pack(pop)
class JAKAPlugin;
class Pose : public CartesianPose
{
public:
    Pose(){ tran.x = 0; tran.y = 0; tran.z = 0; rpy.rx = 0; rpy.ry = 0; rpy.rz = 0; };
    Pose(double x, double y, double z, double rx, double ry, double rz ) { tran.x = x; tran.y = y; tran.z = z; rpy.rx = rx; rpy.ry = ry; rpy.rz = rz; };
    Pose(const Pose& p) { tran.x = p.tran.x; tran.y = p.tran.y; tran.z = p.tran.z; rpy.rx = p.rpy.rx; rpy.ry = p.rpy.ry; rpy.rz = p.rpy.rz; };
    ~Pose() {};
    Pose& operator=(const Pose& p)
    {
        tran.x = p.tran.x;
        tran.y = p.tran.y;
        tran.z = p.tran.z;
        rpy.rx = p.rpy.rx;
        rpy.ry = p.rpy.ry;
        rpy.rz = p.rpy.rz;
        return *this;
    }
    Pose operator+(const Pose& p) const
    {
        Pose result;
        result.tran.x = tran.x + p.tran.x;
        result.tran.y = tran.y + p.tran.y;
        result.tran.z = tran.z + p.tran.z;
        result.rpy.rx = rpy.rx + p.rpy.rx;
        result.rpy.ry = rpy.ry + p.rpy.ry;
        result.rpy.rz = rpy.rz + p.rpy.rz;
        return result;
    }
    Pose operator-(const Pose& p) const
    {
        Pose result;
        result.tran.x = tran.x - p.tran.x;
        result.tran.y = tran.y - p.tran.y;
        result.tran.z = tran.z - p.tran.z;
        result.rpy.rx = rpy.rx - p.rpy.rx;
        result.rpy.ry = rpy.ry - p.rpy.ry;
        result.rpy.rz = rpy.rz - p.rpy.rz;
        return result;
    }
    bool operator==(const Pose& p)
    {
        if (tran.x == p.tran.x && tran.y == p.tran.y && tran.z == p.tran.z && rpy.rx == p.rpy.rx && rpy.ry == p.rpy.ry && rpy.rz == p.rpy.rz)
            return true;
        else
            return false;
    }
    std::string to_string() const {
        char buf[200];
        sprintf(buf, "%f %f %f %f %f %f", tran.x, tran.y, tran.z, rpy.rx / M_PI * 180.0, rpy.ry / M_PI * 180.0, rpy.rz / M_PI * 180.0);
        return std::string(buf);

    }
    std::string to_Taiwanstring() const {
        char buf[200];
        Pose offset(0.15, 0.66, 0.0, 2.191, 2.253, 0.0);
        Pose temp = *this + offset;
        if (temp.rpy.rz > M_PI)
            temp.rpy.rz =(- 2 * M_PI) + temp.rpy.rz;
        if (temp.rpy.rz < -M_PI)
            temp.rpy.rz = (2 * M_PI) + temp.rpy.rz;
        if (temp.rpy.ry > M_PI)
            temp.rpy.ry = (-2 * M_PI) + temp.rpy.ry;
        if (temp.rpy.ry < -M_PI)
            temp.rpy.ry = (2 * M_PI) + temp.rpy.ry;
        if (temp.rpy.rx > M_PI)
            temp.rpy.rx = (-2 * M_PI) + temp.rpy.rx;
        if (temp.rpy.rx < -M_PI)
            temp.rpy.rx = (2 * M_PI) + temp.rpy.rx;
        sprintf(buf, "%f,%f,%f,%f,%f,%f", temp.tran.x/1000.0, temp.tran.y/1000.0 , temp.tran.z/1000.0, temp.rpy.rx, temp.rpy.ry, temp.rpy.rz);
        return std::string(buf);

    }
};

class Trajectory;


class JAKAPlugin :public coVRPlugin, public coMenuListener, public coTUIListener, public OpenThreads::Thread, public opencover::ui::Owner
{
    friend class mySensor;
public:
    static JAKAPlugin *plugin;

    JAKAPlugin();
    ~JAKAPlugin();

    // this will be called in PreFrame
    bool update();
    // this will be called if a COVISE object arrives
    bool init();
    virtual void run(); // robot thread loop
    
    unsigned int numActiveActor;

    linearMessage lm;
    Pose origin;
    void startTrajectory(std::string name);
    void loadTrajectories();
    ui::Menu* JAKATab;
    Trajectory* currentTrajectory = nullptr;
    Trajectory* lastTrajectory = nullptr;
    int currentPosition = 0;
private:
    ui::Button* testButton;
    ui::Button* singleStep;
    ui::Action* reload;
    ui::Action* forward;
    ui::Action* backward;
    bool testMode = false;
    bool stepMode = false;

    JAKAZuRobot* robot=nullptr;
    std::map<std::string, coTUIToggleButton *> tui_header_trans;

    UDPComm* udpclient;
    int UDPPort;

    bool doRun = true;
    
    bool loadWRL(const char *path);
    std::vector<Trajectory*> trajectories;
    void loadTrajectory(std::string filename);

    void sendPose(const Pose& pose);
    Pose testPose;
};

class TrajectoryEntry
{
public:
    TrajectoryEntry(Pose& p, bool w, int s) { pose = p; wait = w; speed = s; };
    Pose pose;
    bool wait;
    int speed;

};
class Trajectory
{
public:
    std::string name;
    std::vector<TrajectoryEntry> entries;
    JAKAZuRobot* robot;
    int currentPosition;
    JAKAPlugin* plugin;
    ui::Action* startButton;
    Trajectory(char* line, JAKAZuRobot* r, JAKAPlugin* p);
    virtual ~Trajectory();
    void addPose(float x, float y, float z, float h, float p, float r, bool wait, int speed)
    {
        Pose pose(x, y, z, h, p, r);
        entries.push_back(TrajectoryEntry(pose, wait, speed));
    }
};
#endif

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
