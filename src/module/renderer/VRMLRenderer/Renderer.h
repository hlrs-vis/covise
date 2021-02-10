/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RENDERER_H
#define _RENDERER_H

/**************************************************************************\
**                                                           (C)1995 RUS  **
**                                                                        **
** Description: Framework class for COVISE renderer modules               **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author:                                                                **
**                                                                        **
**                             Dirk Rantzau                               **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
**                                                                        **
** Date:  11.09.95  V1.0                                                  **
\**************************************************************************/

#include <appl/RenderInterface.h>
#include "ObjectManager.h"

#ifndef PATH_MAX
#define PATH_MAX 1000
#endif

#include <stdlib.h>


namespace covise
{
class ApplicationProcess;
class ConnectionList;
class ClientConnection;
class ServerConnection;
class Host;
}

class Renderer
{

private:
    ObjectManager *om;
    int m_camera_update;
    int m_cam_needed;
    int m_obj_needed;
    //   RenderManager *rm;
    //   CommunicationManager *cm;

    static void quitCallback(void *userData, void *callbackData);
    static void addObjectCallback(void *userData, void *callbackData);
    static void deleteObjectCallback(void *userData, void *callbackData);
    static void renderCallback(void *userData, void *callbackData);
    static void masterSwitchCallback(void *userData, void *callbackData);
    static void paramCallback(bool inMapLoading, void *userData, void *callbackData);
    static void doCustomCallback(void *userData, void *callbackData);

    void quit(void *callbackData);
    void addObject(void *callbackData);
    void deleteObject(void *callbackData);
    void masterSwitch(void *callbackData);
    void render(void *callbackData);
    void param(const char *paraName);
    void doCustom(void *callbackData);

public:
    enum Mode
    {
        VRML,
        WEB
    };

    OutputMode outputMode;

    Mode rendererMode;
    ApplicationProcess *m_app;
    ConnectionList *m_connList;
    const ClientConnection *m_wconn;
    ServerConnection *m_open_conn;
    Host *m_host;
    int m_open_port;
    Host *m_webHost;
    int m_webport;
    int m_aws_wport;
    int m_aws_cport;

    char wrlFilename[PATH_MAX];

    Renderer(int argc, char *argv[]);
    void run()
    {
        CoviseRender::main_loop();
    }
    void start(void); // starting with a web_srv connection
    int check_aws(void);
    int start_aws(void);
    int register_vrml(void);
    int sendViewPoint(void); //sendig ViewPoint data to web server
    int sendObjectOK(void); //sendig end of Objects to web server

    ~Renderer()
    { /*  delete rm; delete cm; */
        delete om;
    }
};

#endif // _RENDERER_H
