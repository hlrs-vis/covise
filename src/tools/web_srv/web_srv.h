/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:             ++
// ++                                                                     ++
// ++ Author:                  ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// ++**********************************************************************/

#ifndef WEB_SRV_H
#define WEB_SRV_H

#include <covise/covise_host.h>
#include "web_socket.h"
#include "web_connect.h"
#include "web_msg.h"
#include <covise/covise_signal.h>
#include "DaemonConn.h"

enum URI_type
{
    M_TEXT_HTML = 0, //  0  //  "text/html"
    M_TEXT_PLAIN, //  1  //  "text/plain"
    M_IMAGE_GIF, //  2  //  "image/gif"
    M_IMAGE_BITMAP, //  3  //  "image/x_bitmap"
    M_IMAGE_JPEG, //  4  //  "image/jpeg"
    M_AUDIO_BASIC, //  5  //  "audio/basic"
    M_VIDEO_MPEG, //  6  //  "video/mpeg"
    M_APPLICATION_OCTET, //  7  //  "application/octet-stream"
    M_APPLICATION_POSTSCRIPT, //  8  //  "application/postscript"
    M_APPLICATION_PROC, //  9  //  "application/process"
    M_XWORLD_XWRML, // 10  //  "x-world/x-vrml"
    M_XWRML_RENDERER, // 11  //  "x-world/x-vrml"
    M_REGISTERED_USERS, // 12  //  "text/plain"
    M_CACERT, // 13  //  "application/x-x509-ca-cert"
    M_SET_DYNAMIC_USR, // 14  //
    M_RST_DYNAMIC_USR, // 15  //
    M_SET_DYNAMIC_VIEW, // 16  //
    M_RST_DYNAMIC_VIEW, // 17  //
    M_AUX, // 18  //  "text/html"
    M_UNKNOWN, // 19  //  UNKNOWN
    M_START_COVISE, // 20  //  Send Launch Msg to Daemon
    M_QUIT_COVISE, // 21  //  Send Quit Msg to Daemon
    M_QUIT_COVISE_SESSION, // 22  //  Send Quit Msg to Master Daemon
    M_CLEAN_COVISE, // 23  //  Clean system
    M_START_DAEMON, // 24  //  Start local Daemon
    M_LAST // 25  //  M_LAST
};

#ifdef DEFINE_URI_TYPE

char *URI_type_array[] = {
    "text/html", //  0  //  M_TEXT_HTML
    "text/plain", //  1  //  M_TEXT_PLAIN
    "image/gif", //  2  //  M_IMAGE_GIF
    "image/x_bitmap", //  3  //  M_IMAGE_BITMAP
    "image/jpeg", //  4  //  M_IMAGE_JPEG
    "audio/basic", //  5  //  M_AUDIO_BASIC
    "video/mpeg", //  6  //  M_VIDEO_MPEG
    "application/octet-stream", //  7  //  M_APPLICATION_OCTET
    "application/postscript", //  8  //  M_APPLICATION_POSTSCRIPT
    "application/process", //  9  //  M_APPLICATION_PROC
    "x-world/x-vrml", // 10  //  M_XWORLD_XWRML
    "x-world/x-vrml", // 11  //  M_XWRML_RENDERER
    "text/plain", // 12  //  M_REGISTERED_USERS
    "application/x-x509-ca-cert", // 13  //  M_CACERT
    NULL, // 14  //  M_SET_DYNAMIC_USR
    NULL, // 15  //  M_RST_DYNAMIC_USR
    NULL, // 16  //  M_SET_DYNAMIC_VIEW
    NULL, // 17  //  M_RST_DYNAMIC_VIEW
    "text/html", // 18  //  M_AUX
    NULL, // 19  //  M_UNKNOWN
    NULL, // 20  //  M_START_COVISE
    NULL, // 21  //  M_QUIT_COVISE
    NULL, // 22  //  M_QUIT_COVISE_SESSION
    NULL, // 23  //  M_CLEAN_COVISE
    NULL, // 24  //  M_START_DAEMON
    NULL // 25  //  M_LAST
};

char *URI_ext_array[] = {
    "text", //  0  //  M_TEXT_PLAIN
    "txt", //  1  //  M_TEXT_PLAIN
    "html", //  2  //  M_TEXT_HTML
    "htm", //  3  //  M_TEXT_HTML
    "gif", //  4  //  M_IMAGE_GIF
    "bmp", //  5  //  M_IMAGE_BITMAP
    "jpeg", //  6  //  M_IMAGE_JPEG
    "mp3", //  7  //  M_AUDIO_BASIC
    "mpeg", //  8  //  M_VIDEO_MPEG
    "exe", //  9  //  M_APPLICATION_OCTET
    "ps", // 10  //  M_APPLICATION_POSTSCRIPT
    "wrl", // 11  //  M_XWORLD_XWRML
    "wrml", // 12  //  M_XWORLD_XWRML
    "cacert", // 13  //  M_CACERT
    "cgi-rnd", // 14  //  M_XWRML_RENDERER
    "cgi-reg", // 15  //  M_REGISTERED_USERS
    "cgi-set_dyn_usr", // 16  //  M_SET_DYNAMIC_USR
    "cgi-rst_dyn_usr", // 17  //  M_RST_DYNAMIC_USR
    "cgi-set_dyn_view", // 18  //  M_SET_DYNAMIC_VIEW
    "cgi-rst_dyn_view", // 19  //  M_RST_DYNAMIC_VIEW
    "cgi-aux", // 20  //  M_AUX
    NULL // 21  //  M_LAST
};

URI_type URI_ext_type[] = {
    M_TEXT_PLAIN, //  0  //  "text"
    M_TEXT_PLAIN, //  1  //  "txt"
    M_TEXT_HTML, //  2  //  "html"
    M_TEXT_HTML, //  3  //  "htm"
    M_IMAGE_GIF, //  4  //  "gif"
    M_IMAGE_BITMAP, //  5  //  "bmp"
    M_IMAGE_JPEG, //  6  //  "jpeg"
    M_AUDIO_BASIC, //  7  //  "mp3"
    M_VIDEO_MPEG, //  8  //  "mpeg"
    M_APPLICATION_OCTET, //  9  //  "exe"
    M_APPLICATION_POSTSCRIPT, // 10  //  "ps"
    M_XWORLD_XWRML, // 11  //  "wrl"
    M_XWORLD_XWRML, // 12  //  "wrml"
    M_CACERT, // 13  //  "cacert"
    M_XWRML_RENDERER, // 14  //  "cgi-rnd"
    M_REGISTERED_USERS, // 15  //  "cgi-reg"
    M_SET_DYNAMIC_USR, // 16  //  "cgi-set_dyn_usr"
    M_RST_DYNAMIC_USR, // 17  //  "cgi-rst_dyn_usr"
    M_SET_DYNAMIC_VIEW, // 18  //  "cgi-set_dyn_view"
    M_RST_DYNAMIC_VIEW, // 19  //  "cgi-rst_dyn_view"
    M_AUX, // 20  //  "cgi-aux"
    M_APPLICATION_OCTET // 21  //  NULL -  M_LAST
};

#else
extern char *URI_type_array[];
extern char *URI_ext_array[];
extern URI_type URI_ext_type[];
#endif

class InfoURI
{

public:
    int m_length;
    URI_type m_type;
    char *m_name; // complet URI
    char *m_URI; // URI received from client

    CConnection *m_conn;

    InfoURI();
    //InfoURI(char *URI_name);
    ~InfoURI();
};

class WHost
{
    char *m_address;
    int m_wport;
    int m_cport;
    int m_refs;

public:
    WHost();
    WHost(char *addr);
    ~WHost();
    char *getAddress(void)
    {
        return m_address;
    };
    void set_ports(int cport, int wport)
    {
        m_cport = cport;
        m_wport = wport;
    };
    int get_wport(void)
    {
        return m_wport;
    };
    int get_cport(void)
    {
        return m_cport;
    };
    int get_refs(void)
    {
        return m_refs;
    };
    int inc_refs(void)
    {
        return ++m_refs;
    };
    int dec_refs(void)
    {
        return --m_refs;
    };
};

class WebSrv
{
    int m_aws_port;
    CClientConnection *m_aws_conn;
    int m_id;
    SignalHandler sig_handler;
    ConnectionList *m_connList; // list of all connections
    Liste<WHost> *m_hostlist;
    char safe[200]; // for testing remember the last Renderer URI

    DaemonConn *dConnection;

public:
    WebSrv();
    ~WebSrv();
    void set_aws_port(int port)
    {
        m_aws_port = port;
    };
    int get_aws_port(void)
    {
        return m_aws_port;
    };
    void set_aws_conn(CClientConnection *conn)
    {
        m_aws_conn = conn;
    };
    int get_id(void)
    {
        return m_id;
    };
    WHost *add_host(char *name);
    int remove_host(char *name);
    WHost *get_host(char *name);
    int remove_aws(Host *host, int cport);
    int open_channel(int port, conn_type type);
    int open_channel(int *port, conn_type type);
    HMessage *wait_for_msg(void);
    void register_vrml(CMessage *msg);
    void mainLoop(void);
    InfoURI *get_infoURI(HMessage *p_Msg);
    char *get_file(char *file_name, int get_flag);
    void send_registered_users(HConnection *p_conn);
    void head_get(HMessage *p_msg, HConnection *p_conn, int get_flag);
    //void process_signal_handler(int sg, void *);

private:
    void startDaemon(bool master = false);
    void contactWeb_partner(const char *uri);
    void cleanCovise();

    void getPartnerHosts(char **partner_hosts, int *num_partners);
    FILE *openFile(const char *filename, char **file_buf, int *size);

    char *last_loaded_map_;
};
#endif
