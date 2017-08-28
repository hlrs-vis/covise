/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
 \brief main renderer class

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C) 2005-2007
         High Performance Computing Center Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   20.07.2004
 */
#ifndef OPEN_COVER_H
#define OPEN_COVER_H

#include <util/common.h>
#include <OpenVRUI/coInteractionManager.h>

#ifdef HAS_MPI
#include <mpi.h>
#endif

namespace covise
{
class VRBClient;
}

namespace opencover
{
class coHud;
class buttonSpecCell;
class coVRPlugin;

extern COVEREXPORT covise::VRBClient *vrbc;

class COVEREXPORT OpenCOVER
{
private:
    vrui::coInteractionManager interactionManager;
    bool exitFlag;
    void readConfigFile();
    void parseLine(char *line);
    int frameNum;
    char *vrbHost;
    int vrbPort;
    static OpenCOVER *s_instance;
    double fl_time, old_fl_time;
    float sum_time;
    bool printFPS;
    bool ignoreMouseEvents;

    void waitForWindowID();

    coVRPlugin *m_visPlugin;
    bool m_forceMpi;

public:
    OpenCOVER();
#ifdef HAS_MPI
    OpenCOVER(const MPI_Comm *comm);
#endif
#ifdef WIN32
    OpenCOVER(HWND parentWindow);
#else
    OpenCOVER(int parentWindow);
#endif
    bool run();
    bool init();
    bool initDone();
    ~OpenCOVER();
    void loop();
    bool frame();
    void doneRendering();
    void setExitFlag(bool flag);
    int getExitFlag()
    {
        return exitFlag;
    }
    static OpenCOVER *instance();
    void handleEvents(int type, int state, int code);
    coHud *hud;
    double beginAppTraversal;
    double endAppTraversal;
    void setIgnoreMouseEvents(bool ign)
    {
        ignoreMouseEvents = ign;
    }
#ifdef WIN32
    HWND parentWindow;
#else
    // should be Window from <X11/Xlib.h>
    int parentWindow;
#endif

    static void quitCallback(void *sceneGraph, buttonSpecCell *spec);
    coVRPlugin *visPlugin() const;
#ifdef HAS_MPI
    MPI_Comm m_comm;
#endif
    bool m_renderNext;
};
}
#endif
