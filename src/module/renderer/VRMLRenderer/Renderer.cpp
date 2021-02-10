/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
#include <covise/covise_appproc.h>
#include "Renderer.h"
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <net/covise_connect.h>

//
// static stub callback functions calling the real class
// member functions
//

void Renderer::quitCallback(void *userData, void *callbackData)
{
    Renderer *thisRenderer = (Renderer *)userData;
    thisRenderer->quit(callbackData);
}

void Renderer::addObjectCallback(void *userData, void *callbackData)
{
    Renderer *thisRenderer = (Renderer *)userData;
    thisRenderer->addObject(callbackData);
}

void Renderer::deleteObjectCallback(void *userData, void *callbackData)
{
    Renderer *thisRenderer = (Renderer *)userData;
    thisRenderer->deleteObject(callbackData);
}

void Renderer::masterSwitchCallback(void *userData, void *callbackData)
{
    Renderer *thisRenderer = (Renderer *)userData;
    thisRenderer->masterSwitch(callbackData);
}

void Renderer::renderCallback(void *userData, void *callbackData)
{
    Renderer *thisRenderer = (Renderer *)userData;
    thisRenderer->render(callbackData);
}

void Renderer::paramCallback(bool /*inMapLoading*/, void *userData, void *)
{
    Renderer *thisRenderer = (Renderer *)userData;
    thisRenderer->param(CoviseRender::get_reply_param_name());
}

void Renderer::doCustomCallback(void *userData, void *callbackData)
{
    Renderer *thisRenderer = (Renderer *)userData;
    thisRenderer->doCustom(callbackData);
}

//
//
//..........................................................................
//
Renderer::Renderer(int argc, char *argv[])
{

    CoviseRender::set_module_description("VRML Renderer");
    CoviseRender::add_port(INPUT_PORT, "RenderData", "Geometry|UnstructuredGrid|RectilinearGrid|StructuredGrid|Polygons|TriangleStrips|Lines|Points|Spheres", "render geometry");
    outputMode = X3DOM;
    // check type, use VRML as default if illegal name
    bool useDefault = false;
    if (strstr(argv[0], "VRML"))
        rendererMode = VRML;
    else if (strstr(argv[0], "WEB"))
        rendererMode = WEB;
    else
    {
        useDefault = true;
        rendererMode = VRML;
    }
    if (rendererMode == VRML)
    {
        CoviseRender::add_port(PARIN, "Filename", "Browser", "Output file name");
        CoviseRender::add_port(PARIN, "OutputMode", "Choice", "Vrml97 or X3DOM");
        CoviseRender::add_port(PARIN, "translate", "FloatVector", "(x, y, z) translation of the VRML model");
        CoviseRender::add_port(PARIN, "scale", "FloatVector", "scale of the VRML model");
        CoviseRender::add_port(PARIN, "axis_of_rotation", "FloatVector", "rotation of the VRML model");
        CoviseRender::add_port(PARIN, "angle_of_rotation", "FloatScalar", "rotation of the VRML model");
        static char defVal[256];
        snprintf(defVal, sizeof(defVal), "./%s_%s.wrl", argv[0], argv[4]);
        CoviseRender::set_port_default("Filename", defVal);
        CoviseRender::set_port_default("OutputMode", "2 Vrml97 X3DOM");

        //CoviseRender::update_choice_param(d_name, d_numChoices, d_choice, d_activeChoice);
        CoviseRender::set_port_default("translate", "0 0 0");
        CoviseRender::set_port_default("scale", "1 1 1");
        CoviseRender::set_port_default("axis_of_rotation", "0 0 1");
        CoviseRender::set_port_default("angle_of_rotation", "0");

        // don't overwrite defVal -- appl_lib doesn't copy
        char defaultValue[256];
        snprintf(defaultValue, sizeof(defaultValue), "./%s_%s.wrl", argv[0], argv[4]);
        CoviseRender::getname(wrlFilename, defaultValue); // try covise relative pathes
        if (!wrlFilename[0]) // if not: use given path
            strcpy(wrlFilename, defaultValue);
    }
    else
        strcpy(wrlFilename, "Web Connection");

    CoviseRender::init(argc, argv);

    m_app = CoviseRender::appmod;
    m_connList = m_app->getConnectionList();
    m_wconn = NULL;
    m_open_conn = NULL;
    m_host = NULL;
    m_open_port = 0;
    m_webHost = NULL;
    m_webport = 0;
    m_aws_wport = 0;
    m_aws_cport = 0;
    m_camera_update = 0;
    m_cam_needed = 1;
    m_obj_needed = 0;
    om = new ObjectManager();
    om->setFilename(wrlFilename);
    om->setOutputMode(outputMode);

    CoviseRender::set_render_callback(Renderer::renderCallback, this);
    CoviseRender::set_master_switch_callback(Renderer::masterSwitchCallback, this);
    CoviseRender::set_quit_callback(Renderer::quitCallback, this);
    CoviseRender::set_add_object_callback(Renderer::addObjectCallback, this);
    CoviseRender::set_delete_object_callback(Renderer::deleteObjectCallback, this);
    CoviseRender::set_param_callback(Renderer::paramCallback, this);
    CoviseRender::set_custom_callback(Renderer::doCustomCallback, this);

    // send warning when using default: must be after init() call
    if (useDefault)
        CoviseRender::sendError("Invalid name for VRML/WEB_Renderer: default to VRML");
}

//
//
//..........................................................................
//

void Renderer::quit(void *callbackData)
{
    (void)callbackData;
    //
    // ...... delete your data here .....
    //
    CoviseRender::sendInfo("Quitting now");
    //cerr << "\n---- Quitting !!!!\n";

    if (m_wconn != NULL)
    {
        int l = strlen(" ") + 1;
        char* d = new char[l];
        strcpy(d, " ");
        Message p_msg{ COVISE_MESSAGE_SOCKET_CLOSED , DataHandle(d, l)};

        m_wconn->sendMessage(&p_msg);

        m_connList->remove(m_wconn);
        delete m_wconn;
        m_wconn = NULL;
    }
}

void Renderer::param(const char *paraName)
{
    //
    // ...... do work here ........
    //

    if (strcmp("CAMERA", paraName) == 0)
    {
        CoviseRender::sendInfo("got camera position parameter");
    }
    else if (strcmp("OutputMode", paraName) == 0)
    {
        int val = 0;
        if (CoviseRender::get_reply_choice(&val))
        {
            outputMode = (OutputMode)val;
        }
    }
    else if (strcmp("Filename", paraName) == 0)
    {
        const char *reply;
        if (CoviseRender::get_reply_string(&reply))
        {
            CoviseRender::getname(wrlFilename, reply); // try covise relative pathes
            if (!wrlFilename[0]) // if not: use givemn path
                strcpy(wrlFilename, reply);
            CoviseRender::sendInfo("%s", wrlFilename);
            if (om)
                om->setFilename(wrlFilename);
        }
    }
    else if (!strcmp(paraName, "translate"))
    {

        float x, y, z;
        if (CoviseRender::get_reply_float_slider(&x, &y, &z) && objlist)
            objlist->setTranslation(x, y, z);
    }
    else if (!strcmp(paraName, "scale"))
    {

        float x, y, z;
        if (CoviseRender::get_reply_float_slider(&x, &y, &z) && objlist)
            objlist->setScale(x, y, z);
    }
    else if (!strcmp(paraName, "rotation"))
    {

        float x, y, z;
        if (CoviseRender::get_reply_float_slider(&x, &y, &z) && objlist)
            objlist->setRotation(x, y, z);
    }
    else if (!strcmp(paraName, "axis_of_rotation"))
    {

        float x, y, z;
        if (CoviseRender::get_reply_float_slider(&x, &y, &z) && objlist)
            objlist->setRotation(x, y, z);
    }
    else if (!strcmp(paraName, "angle_of_rotation"))
    {

        float d;
        if (CoviseRender::get_reply_float_scalar(&d) && objlist)
            objlist->setRotationAngle(d);
    }
}

void Renderer::addObject(void *callbackData)
{
    (void)callbackData;

    CoviseRender::sendInfo("Adding object %s to %s", CoviseRender::get_object_name(), wrlFilename);

    om->addObject(CoviseRender::get_object_name());
    m_obj_needed = 1;

    if (m_camera_update && objlist != NULL)
    {
        objlist->sendNewObjects(m_wconn);
        objlist->sendTimestep(m_wconn);
        m_obj_needed = 0;
        //m_cam_needed = 0;
    }
}

void Renderer::deleteObject(void *callbackData)
{
    (void)callbackData;
    char *obj_name;
    char print_buf[1000];

    CoviseRender::sendInfo("Removing object %s from %s", CoviseRender::get_object_name(), wrlFilename);
    m_obj_needed = 1;

    obj_name = CoviseRender::get_object_name();
    om->deleteObject(obj_name);

    if (m_camera_update && objlist != NULL)
    {
        snprintf(print_buf, sizeof(print_buf), " 9 %s", obj_name);
        objlist->send_obj(m_wconn, print_buf);
        objlist->sendTimestep(m_wconn);
        m_obj_needed = 0;
        //m_cam_needed = 0;
    }
}

void Renderer::render(void *callbackData)
{
    (void)callbackData;
    // char print_buf[1000];

    //cerr << "\n got msg " << CoviseRender::get_render_keyword() << endl << CoviseRender::get_render_data() << endl;

    if (strcmp((CoviseRender::get_render_keyword()), "VRMLCAMERA") == 0)
    {
        //cerr << "\n VRML_Renderer got msg " << CoviseRender::get_render_keyword() << endl <<
        //CoviseRender::get_render_data() << endl;
        int ret = objlist->setViewPoint(CoviseRender::get_render_data());
        if (ret)
            m_cam_needed = 1;
        if (m_camera_update && ret)
            sendViewPoint();
    }
    else if (strcmp((CoviseRender::get_render_keyword()), "VRML_TELEPOINTER") == 0)
    {
        //cerr << "\n VRML_Renderer got msg " << CoviseRender::get_render_keyword() << endl << CoviseRender::get_render_data() << endl;
        objlist->setTelepointer(CoviseRender::get_render_data());
        if (m_camera_update)
            objlist->sendTelepointer(m_wconn);
    }
    else if (strcmp((CoviseRender::get_render_keyword()), "SEQUENCER") == 0)
    {
        objlist->setSequencer(CoviseRender::get_render_data());
        if (m_camera_update)
            objlist->sendTimestep(m_wconn);
    }
    // CoviseRender::sendInfo("Got update message from master:");

    // sprintf(print_buf,"key: %s data: %s",CoviseRender::get_render_keyword(),
    //                                      CoviseRender::get_render_data() );
    // CoviseRender::sendInfo(print_buf);
}

void Renderer::masterSwitch(void *callbackData)
{
    (void)callbackData;
    // if(CoviseRender::is_master())
    //   CoviseRender::sendInfo("Changing state to master");
    // else
    //   CoviseRender::sendInfo("Changing state to slave");
}

void Renderer::doCustom(void *callbackData)
{
    Message *msg;
    char *tmp;
    int ret;

    //CoviseRender::sendInfo("doCustom callback called");
    if (callbackData)
    {
        msg = (Message *)callbackData;
        //cerr << endl << "  Custom Callback  ";
        //cerr << endl << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
        //cerr << " Message_type: " << msg->type << " from Sender: " << msg->sender << " Send_type: "  << msg->send_type << endl << "-------------------------------\n";
        //if(msg->data) cerr << msg->data;
        //else cerr << " NULL content ";
        //cerr << endl << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n";

        switch (msg->type)
        {
        case COVISE_MESSAGE_MAKE_DATA_CONNECTION:
            cerr << endl << "--- MAKE_DATA_CONNECTION message \n";
            ret = start_aws();
            if (ret)
                ; //register_vrml();
            else
                cerr << "\n Error starting aws !!!\n";
            break;
        case COVISE_MESSAGE_COMPLETE_DATA_CONNECTION:
            tmp = strtok(msg->data.accessData(), " ");
            m_aws_cport = atoi(tmp);
            tmp = strtok(NULL, " ");
            m_aws_wport = atoi(tmp);
            cerr << endl << "--- COMPLETE_DATA_CONNECTION : " << m_aws_cport << " " << m_aws_wport << endl;
            register_vrml();
            break;

        case COVISE_MESSAGE_PORT:
            tmp = strtok(msg->data.accessData(), " ");
            m_aws_cport = atoi(tmp);
            tmp = strtok(NULL, " ");
            m_aws_wport = atoi(tmp);
            cerr << endl << "--- PORT : " << m_aws_cport << " " << m_aws_wport << endl;
            register_vrml();
            break;

        case COVISE_MESSAGE_ASK_FOR_OBJECT:
            if (objlist != NULL)
            {
                objlist->sendObjects(msg->conn);
                objlist->sendTimestep(msg->conn);
                sendViewPoint();

                //sendObjectOK();
                m_obj_needed = 0;
                //m_cam_needed = 0;
            }
            break;
        case COVISE_MESSAGE_SET_ACCESS:
            m_camera_update = atoi(msg->data.data());
            if (m_camera_update && objlist != NULL)
            {
                if (m_obj_needed)
                {
                    objlist->sendNewObjects(msg->conn);
                    m_obj_needed = 0;
                    //m_cam_needed = 0;
                }
                if (m_cam_needed)
                {
                    sendViewPoint();
                    //m_cam_needed = 0;
                }
                objlist->sendTimestep(msg->conn);
            }
            //cerr << endl << "$$$$ - SET_ACCESS :" << m_camera_update << endl;
            break;
        default:
            break;

        } // end switch
    }
}

void Renderer::start(void)
{
    int id;
    std::string shost;
    std::string sport;
    char tmp_hostname[256];

    gethostname(tmp_hostname, 255); // fetching the local host
    Host thost(tmp_hostname);
    m_host = new Host(thost.getAddress());

    if (rendererMode == WEB)
    {
        // get HOSTSRV and COVISE_PORT from either shell var or CoviseConfig
        shost = getenv("HOSTSRV");
        if (shost.empty())
            shost = coCoviseConfig::getEntry("WEB_Renderer.HOSTSRV");

        sport = getenv("COVISE_PORT");
        if (sport.empty())
            sport = coCoviseConfig::getEntry("WEB_Renderer.COVISE_PORT");

        if ((!shost.empty()) || (!sport.empty()))
        {
            cerr << "\n HOSTSRV or COVISE_PORT environment variable not set !!!" << endl;
        }
        else
        {
            char *hostname;
            m_webHost = new Host(shost.c_str());
            hostname = (char *)m_webHost->getName();
            if (hostname != NULL)
            {
                m_webport = atoi(sport.c_str());
                id = (int)getpid();
                auto wconn = std::unique_ptr<ClientConnection>(new ClientConnection(m_webHost, m_webport, id, RENDERER));

                if (!wconn->is_connected())
                {
                    CoviseRender::sendError(" Error connecting to web_srv - terminating");
                    exit(0);
                }
                else
                {

                    // creating an open port for client connections
                    m_open_conn = new ServerConnection(&m_open_port, id, SIMPLECONTROLLER);

                    if (!m_open_conn->is_connected())
                    {
                        CoviseRender::sendError("VRML Renderer couldn't create an open socket - terminating");
                        delete m_open_conn;
                        m_open_conn = NULL;
                        m_open_port = 0;
                        exit(0);
                    }
                    else
                    {
                        //setting the open connection of the conn list;
                        m_connList->add_open_conn(m_open_conn);

                        //adding web connection to conn list;
                        m_wconn = dynamic_cast<const ClientConnection*>(m_connList->add(std::move(wconn)));

                        //disable the writing of the wrml file
                        om->set_write_file(0);

                        // registering vrml renderer
                        register_vrml();
                        //check_aws();
                    }
                }
            }
        }
    }

    CoviseRender::main_loop();
}

int Renderer::check_aws(void)
{
    cerr << "\n Check aws on the host " << m_host->getName() << endl;

    Message p_msg{ COVISE_MESSAGE_INIT , DataHandle((char*)m_host->getName(),strlen(m_host->getName()) + 1 , false)};
    m_wconn->sendMessage(&p_msg);

    return 1;
}

int Renderer::start_aws(void)
{

#ifndef WIN32
    char port_str[50];
    if (fork() == 0)
    {
        snprintf(port_str, sizeof(port_str), "%d", m_open_port);
        execlp("web_srv", "web_srv", port_str, NULL);
    }
#endif

    return 1;
}

int Renderer::register_vrml(void)
{

    DataHandle txt(250);

    // modulname_port(hostname).cgi-rnd#cport_wport
    snprintf(txt.accessData(), txt.length(), "%s_%d(%s).cgi-rnd#%d_%d", m_app->getName(), m_open_port, m_host->getName(), m_aws_cport, m_aws_wport);

    //cerr << "\n Registering the renderer : " << txt << endl;
    txt.setLength((int)strlen(txt.data()) + 1);
    Message p_msg{ COVISE_MESSAGE_START, txt };

    m_wconn->sendMessage(&p_msg);

    strcat(txt.accessData(), " has been registered !!!");

    CoviseRender::sendInfo("%s", txt.data());
    return 1;
}

int Renderer::sendObjectOK(void)
{
    if (m_wconn == NULL)
        return 0;

    Message p_msg{ COVISE_MESSAGE_OBJECT_OK, DataHandle{(char*)" ", strlen(" ") + 1, false} };
    //cerr << endl << "&&&& Sending OBJECT_OK " << endl;
    m_wconn->sendMessage(&p_msg);

    return 1;
}

int Renderer::sendViewPoint(void)
{
    if (m_wconn == NULL)
        return 0;
    else
    {
        char *camera = objlist->getViewPoint();
        if (camera != NULL)
        {
            Message p_msg{ COVISE_MESSAGE_PARINFO, DataHandle{camera, strlen(camera) + 1, false} };
            p_msg.type = COVISE_MESSAGE_PARINFO;
            //cerr << endl << "&&&& Sending CAMERA: " << camera << endl;
            m_wconn->sendMessage(&p_msg); // send ViewPoint

            //m_cam_needed = 0;
        }
        else
            return 0;
    }
    return 1;
}
