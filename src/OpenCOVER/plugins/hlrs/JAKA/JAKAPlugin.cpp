/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

/*********************************************************************************\
**                                                            2009 HLRS         **
**                                                                              **
** Description:  Show/Hide of JAKAPlugins, defined in Collect Module               **
**                                                                              **
**                                                                              **
** Author: A.Gottlieb                                                           **
**                                                                              **
** Jul-09  v1                                                                   **
**                                                                              **
**                                                                              **
\*********************************************************************************/

#include "JAKAPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <osg/CullStack>
#include <iostream>
#include <cover/coTabletUI.h>
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <net/tokenbuffer.h>
#include <osg/Node>
#include <algorithm>
#include <map>
#include <vector>
#include <iterator>
#include <numeric>
#include "VrmlNodeJAKA.h"
#include <vrml97/vrml/VrmlNamespace.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
//#include <OpenVRUI/coMenuChangeListener.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coToolboxMenuItem.h>
#include <util/threadname.h>
#include <filesystem>

#define Remote_HOST "140.110.136.126"
//#define Remote_HOST "172.31.86.227"
//#define Local_robot "192.168.178.127"
#define Local_robot "10.203.24.17"
//#define Local_robot "141.58.8.231"
using namespace covise;
using namespace opencover;
using namespace vrui;

JAKAPlugin *JAKAPlugin::plugin = NULL;

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

Trajectory::~Trajectory()
{
    delete startButton;
    entries.clear();
}

Trajectory::Trajectory(char* line, JAKAZuRobot* r, JAKAPlugin* p)
{
    name = line;
    if (name[name.length() - 1] == '\n')
        name.pop_back();
    robot = r;
    plugin = p;

    startButton = new ui::Action(plugin->JAKATab, name);
    startButton->setText(name);
    startButton->setCallback([this]() {
        plugin->currentPosition = 0;
        plugin->startTrajectory(name);
        });
}

// Overload the << operator for std::ostream
std::ostream& operator<<(std::ostream& os, const Pose& p) {
    os << p.to_string();
    return os;
}

JAKAPlugin::JAKAPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("JAKAPlugin", cover->ui)
{
    plugin = this;

    VrmlNamespace::addBuiltIn(VrmlNodeJAKA::defineType<VrmlNodeJAKA>());



}
//------------------------------------------------------------------------------------------------------------------------------

bool JAKAPlugin::init()
{
    JAKATab = new ui::Menu("JAKA", this);
    reload = new ui::Action(JAKATab, "Reload");
    reload->setText("Reload");
    reload->setCallback([this]() {
        loadTrajectories();
        });

    singleStep = new ui::Button(JAKATab, "SingleStep");
    singleStep->setText("SingleStep");
    singleStep->setCallback([this](bool state) {
        stepMode = state;
        });
    forward = new ui::Action(JAKATab, "forward");
    forward->setText("forward");
    forward->setCallback([this]() {
        if (lastTrajectory != nullptr)
        {
            if (currentPosition < lastTrajectory->entries.size() - 1)
                currentPosition++;
            currentTrajectory = lastTrajectory;
            cerr << "currentPosition" << currentPosition << endl;
        }
        });

    backward = new ui::Action(JAKATab, "backward");
    backward->setText("backward");
    backward->setCallback([this]() {
        if (lastTrajectory != nullptr)
        {
            if (currentPosition > 0)
                currentPosition--;
            currentTrajectory = lastTrajectory;
            cerr << "currentPosition" << currentPosition << endl;
        }
        });

    testButton = new ui::Button(JAKATab, "Test");
    testButton->setText("Test");
    testButton->setCallback([this](bool state) {
        testMode = state;
        });

    UDPPort = 30001;
    udpclient = new UDPComm(Remote_HOST,UDPPort,UDPPort);
    robot = new JAKAZuRobot();
    errno_t ret = robot->login_in(Local_robot);
    if (ret != ERR_SUCC)
    {
        std::cout << "login failed !.\n";
    }
    RobotState status;
    ret = robot->get_robot_state(&status);
    if (ret != ERR_SUCC)
    {
        std::cout << "get_state failed !.\n";
    }
    if (!status.poweredOn)
    {
        ret = robot->power_on();
        if (ret != ERR_SUCC)
        {
            std::cout << "power on failed !.\n";
        }
        while (!status.poweredOn)
        {
            robot->get_robot_state(&status);
            Sleep(1);
        }
    }
    if (!status.servoEnabled)
    {
        ret = robot->enable_robot();
        if (ret != ERR_SUCC)
        {
            std::cout << "enable failed !.\n";
        }
        while (!status.poweredOn)
        {
            robot->get_robot_state(&status);
            Sleep(1);
        }
    }

    origin.tran.x = -232.0;
    origin.tran.y = -277.0;
    origin.tran.z = 0;
    origin.rpy.rx = M_PI;
    origin.rpy.ry = 0;
    origin.rpy.rz = 0;

    lm.type = 1;
    lm.size = sizeof(lm);
    lm.x = 1.0;
    lm.y = 0.0;
    lm.z = 0.0;
    lm.h = 0.0;
    lm.p = 0.0;
    lm.r = 0.0;
    testPose.tran.x = 1;
    loadTrajectories();
    start();
    return true;
}
inline bool ends_with(std::string const& value, std::string const& ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

void JAKAPlugin::loadTrajectories()
{
    for (auto& t : trajectories)
    {
        delete t;
    }   
    trajectories.clear();
    std::string path = "/data/NCHC";
    for (const auto& entry : filesystem::directory_iterator(path))
    {
        if (ends_with(entry.path().string(), ".txt"))
        {
            if (entry.path().string().rfind("trajectory", 0) != 0)
            {
                loadTrajectory(entry.path().string());

                std::cout << entry.path() << std::endl;
            }
        }
    }

}

//------------------------------------------------------------------------------------------------------------------------------
// this is called if the plugin is removed at runtime

JAKAPlugin::~JAKAPlugin()
{
    doRun = false;
    //fprintf ( stderr,"JAKAPlugin::~JAKAPlugin\n" );

}
//------------------------------------------------------------------------------------------------------------------------------
void JAKAPlugin::sendPose(const Pose& pose)
{

    char buf[200];
    sprintf(buf, "%d,%s\n", 1, pose.to_Taiwanstring().c_str());
    //udpclient->send(&lm, sizeof(lm));
    udpclient->send(&buf, strlen(buf));
    cout << buf << std::endl;
}

void JAKAPlugin::run()
{
    setThreadName("JAKA Robot Thread");

    while (doRun)
    {
        static double lastTime = 0.0;
        if (currentTrajectory)
        {
            if (!stepMode)
                currentPosition = 0;
            for (; currentPosition < currentTrajectory->entries.size(); )
            {
                Pose pose = currentTrajectory->entries[currentPosition].pose;

                Pose newPose = pose + origin;
                sendPose(pose);
                errno_t ret = robot->linear_move(&newPose, MoveMode::ABS, currentTrajectory->entries[currentPosition].wait, currentTrajectory->entries[currentPosition].speed);
                if (ret != ERR_SUCC)
                {
                    std::cout << "linear move failed !.\n";
                }
                if (stepMode)
                {
                    break;
                }
                currentPosition++;
            }
            currentTrajectory = nullptr;
        }
        else
        {
            if (testMode)
            {
                if (cover->frameTime() > (lastTime + 1.0))
                {
                    lastTime = cover->frameTime();
                    sendPose(testPose);

                    testPose.tran.x = testPose.tran.x + 10.0;
                    if (testPose.tran.x > 100)
                    {
                        testPose.tran.x = 0.0;
                    }
                    Pose newPose = testPose + origin;


                    errno_t ret = robot->linear_move(&newPose, MoveMode::ABS, false, 100);
                    if (ret != ERR_SUCC)
                    {
                        std::cout << "linear move failed !.\n";
                    }
                }
            }
            else
            {
                    Pose pose, newPose;
                    robot->get_tcp_position(&pose);
                    static Pose oldPose;
                    if (!(oldPose == pose))
                    {
                        oldPose = pose;
                        newPose = pose - origin;

                        sendPose(newPose);
                    }
                    RobotStatus rs;
                    robot->get_robot_status(&rs);
                    static int oldState = 0;
                    if (rs.tio_key[1] != oldState) // Point is pressed
                    {
                        oldState = rs.tio_key[1];
                        if (rs.tio_key[1])
                        {
                            newPose = pose - origin;
                            FILE* fp = fopen("/data/NCHC/trajectory.txt", "a");
                            fprintf(fp, "%s 0 100\n", newPose.to_string().c_str());
                            fprintf(stderr, "Saving %s 0 100\n", newPose.to_string().c_str());
                            fclose(fp);
                        }

                    }
            }
        }
        OpenThreads::Thread::microSleep(100);
    }
}
bool JAKAPlugin::update()
{
    
    
    return true; // request that scene be re-rendered
}
//------------------------------------------------------------------------------------------------------------------------------


bool JAKAPlugin::loadWRL(const char *path)				
{

    CartesianPose pose;
    robot->get_tcp_position(&pose);
    JointValue jpos;
    robot->get_joint_position(&jpos);
    for (int i = 0; i < 6; i++)
        std::cout << jpos.jVal[i];
    std::cout << std::endl;
    return true;
}

void JAKAPlugin::loadTrajectory(std::string filename)
{
    FILE* fp = fopen(filename.c_str(), "r");
    if (fp == NULL)
    {
        std::cout << "open file failed !.\n";
        return;
    }
    char line[256];
    fgets(line, sizeof(line), fp);
    Trajectory *trajectory = new Trajectory(line,robot,this);
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        std::cout << line;
        if (line[0] == '#')
            continue;
        Pose cp;
        double x, y, z, rx, ry, rz;
        int w, speed;
        sscanf(line, "%lf %lf %lf %lf %lf %lf %d %d", &x, &y, &z, &rx, &ry, &rz, &w,&speed);
        bool wait = w == 1 ? true : false;
        trajectory->addPose(x,y,z, rx /180.0*M_PI, ry / 180.0 * M_PI, rz / 180.0 * M_PI, wait,speed);
    }
    trajectories.push_back(trajectory);
}

void JAKAPlugin::startTrajectory(std::string name)
{
    for (int i = 0; i < trajectories.size(); i++)
    {
        if (trajectories[i]->name == name)
        {
            lastTrajectory = trajectories[i];
            currentTrajectory = trajectories[i];
            break;
        }
    }
}






//------------------------------------------------------------------------------------------------------------------------------
COVERPLUGIN(JAKAPlugin)
