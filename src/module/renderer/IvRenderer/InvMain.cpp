/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log:  $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */
#include <covise/covise.h>

//**************************************************************************
//
// * Description    : This is the main module for the renderer
//
//
// * Class(es)      : none
//
//
// * inherited from : none
//
//
// * Author  : Dirk Rantzau
//
//
// * History : 29.03.94 V 1.0
//
//
//
//**************************************************************************

//
// renderer stuff
//
#include "InvDefs.h"

//
// environment stuff
//
#include <util/coLog.h>
#include <net/covise_connect.h>
#include <covise/covise_msg.h>
#include <covise/covise_appproc.h>
#include "ModuleInfo.h"
#include <unistd.h>
#include <covise/Covise_Util.h>
#include <Xm/Xm.h>
#include "SoBillboard.h"
#ifndef WITHOUT_VIRVO
#include "SoVolume.h"
#include "SoVolumeDetail.h"
#endif

using namespace covise;
//
// external
//
extern void rm_startup(int argc, char *argv[]);

//
// some global stuff
//

char ModuleHead[1024];

ApplicationProcess *appmod;
int port;
char *host;
char *proc_name;
int proc_id;
int socket_id;
char *instance;
Widget MasterRequest;

enum appl_port_type
{
    DESCRIPTION = 0, //0
    INPUT_PORT, //1
    OUTPUT_PORT, //2
    PARIN, //3
    PAROUT //4
};
char *port_name[200] = { NULL, NULL };
char *m_name = NULL;
char *h_name = NULL;
char *port_description[200];
char *port_datatype[200];
char *port_dependency[200];
char *port_default[200];
int port_required[200];
int port_immediate[200];
enum appl_port_type port_type[200];
char *module_description = NULL;
char username[100];

void set_module_description(const char *descr)
{
    module_description = (char *)descr;
}

void add_port(enum appl_port_type type, char *name)
{
    if (type == OUTPUT_PORT || type == INPUT_PORT || type == PARIN || type == PAROUT)
    {
        int i = 0;
        while (port_name[i])
            i++;
        port_type[i] = type;
        port_name[i] = name;
        port_default[i] = NULL;
        port_datatype[i] = NULL;
        port_dependency[i] = NULL;
        port_required[i] = 1;
        port_immediate[i] = 0;
        port_description[i] = NULL;
        port_name[i + 1] = NULL;
    }
    else
    {
        cerr << "wrong description type in add_port " << name << "\n";
        return;
    }
}

void add_port(enum appl_port_type type, const char *name, const char *dt, const char *descr)
{
    if (type == OUTPUT_PORT || type == INPUT_PORT || type == PARIN || type == PAROUT)
    {
        int i = 0;
        while (port_name[i])
            i++;
        port_type[i] = type;
        port_name[i] = (char *)name;
        port_default[i] = NULL;
        port_datatype[i] = (char *)dt;
        port_dependency[i] = NULL;
        port_required[i] = 1;
        port_immediate[i] = 0;
        port_description[i] = (char *)descr;
        port_name[i + 1] = NULL;
    }
    else
    {
        cerr << "wrong description type in add_port " << name << "\n";
        return;
    }
}

void set_port_description(char *name, char *descr)
{
    int i = 0;
    while (port_name[i])
    {
        if (strcmp(port_name[i], name) == 0)
            break;
        i++;
    }
    if (port_name[i] == NULL)
    {
        cerr << "wrong portname " << name << " in set_port_description\n";
        return;
    }
    port_description[i] = descr;
}

void set_port_default(const char *name, const char *def)
{
    int i = 0;
    while (port_name[i])
    {
        if (strcmp(port_name[i], name) == 0)
            break;
        i++;
    }
    if (port_name[i] == NULL)
    {
        cerr << "wrong portname " << name << " in set_port_default\n";
        return;
    }

    if (port_type[i] != PARIN && port_type[i] != PAROUT)
    {
        cerr << "wrong port type in set_port_default " << name << "\n";
        return;
    }
    port_default[i] = (char *)def;
}

void set_port_datatype(char *name, char *dt)
{
    int i = 0;
    while (port_name[i])
    {
        if (strcmp(port_name[i], name) == 0)
            break;
        i++;
    }
    if (port_name[i] == NULL)
    {
        cerr << "wrong portname " << name << " in set_port_datatype\n";
        return;
    }
    port_datatype[i] = dt;
}

void set_port_required(char *name, int req)
{
    int i = 0;
    while (port_name[i])
    {
        if (strcmp(port_name[i], name) == 0)
            break;
        i++;
    }
    if (port_name[i] == NULL)
    {
        cerr << "wrong portname " << name << " in set_port_required\n";
        return;
    }
    if (port_type[i] != INPUT_PORT)
    {
        cerr << "wrong port type in set_port_required " << name << "\n";
        return;
    }
    port_required[i] = req;
}

void set_port_immediate(const char *name, int imm)
{
    int i = 0;
    while (port_name[i])
    {
        if (strcmp(port_name[i], name) == 0)
            break;
        i++;
    }
    if (port_name[i] == NULL)
    {
        cerr << "wrong portname " << name << " in set_port_immediate\n";
        return;
    }
    if (port_type[i] != PARIN)
    {
        cerr << "wrong port type in set_port_immediate " << name << "\n";
        return;
    }
    port_immediate[i] = imm;
}

char *get_description_message()
{
    CharBuffer msg(400);
    msg += "DESC\n";
    msg += m_name;
    msg += '\n';
    msg += h_name;
    msg += '\n';
    if (module_description)
        msg += module_description;
    else
        msg += m_name;
    msg += '\n';

    int i = 0, ninput = 0, noutput = 0, nparin = 0, nparout = 0;
    while (port_name[i])
    {
        switch (port_type[i])
        {
        case INPUT_PORT:
            ninput++;
            break;
        case OUTPUT_PORT:
            noutput++;
            break;
        case PARIN:
            nparin++;
            break;
        case PAROUT:
            nparout++;
            break;
        default:
            break;
        }
        i++;
    }
    msg += ninput; // number of parameters
    msg += '\n';
    msg += noutput;
    msg += '\n';
    msg += nparin;
    msg += '\n';
    msg += nparout;
    msg += '\n';
    i = 0; // INPUT ports
    while (port_name[i])
    {
        if (port_type[i] == INPUT_PORT)
        {
            msg += port_name[i];
            msg += '\n';
            if (port_datatype[i] == NULL)
            {
                cerr << "no datatype for port " << port_name[i] << "\n";
            }
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_required[i])
                msg += "req\n";
            else
                msg += "opt\n";
        }
        i++;
    }
    i = 0; // OUTPUT ports
    while (port_name[i])
    {
        if (port_type[i] == OUTPUT_PORT)
        {
            msg += port_name[i];
            msg += '\n';
            if (port_datatype[i] == NULL)
            {
                cerr << "no datatype for port " << port_name[i] << "\n";
            }
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_dependency[i])
            {
                msg += port_dependency[i];
                msg += '\n';
            }
            else
                msg += "default\n";
        }
        i++;
    }
    i = 0; // PARIN ports
    while (port_name[i])
    {
        if (port_type[i] == PARIN)
        {
            msg += port_name[i];
            msg += '\n';
            if (port_datatype[i] == NULL)
            {
                cerr << "no datatype for port " << port_name[i] << "\n";
            }
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_default[i] == NULL)
            {
                cerr << "no default value for parameter " << port_name[i] << "\n";
            }
            msg += port_default[i];
            msg += '\n';
            if (port_immediate[i])
                msg += "IMM\n";
            else
                msg += "START\n";
        }
        i++;
    }
    i = 0; // PAROUT ports
    while (port_name[i])
    {
        if (port_type[i] == PAROUT)
        {
            msg += port_name[i];
            msg += '\n';
            if (port_datatype[i] == NULL)
            {
                cerr << "no datatype for port " << port_name[i] << "\n";
            }
            msg += port_datatype[i];
            msg += '\n';
            if (port_description[i] == NULL)
                msg += port_name[i];
            else
                msg += port_description[i];
            msg += '\n';
            if (port_default[i] == NULL)
            {
                cerr << "no default value for parameter " << port_name[i] << "\n";
            }
            msg += port_default[i];
            msg += '\n';
        }
        i++;
    }
    return (msg.return_data());
}

void
printDesc(const char *callname)
{
    // strip leading path from module name
    const char *modName = strrchr(callname, '/');
    if (modName)
        modName++;
    else
        modName = callname;

    cout << "Module:      \"" << modName << "\"" << endl;
    cout << "Desc:        \"" << module_description << "\"" << endl;

    int i, numItems;

    // count parameters
    numItems = 0;
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == PARIN)
            numItems++;
    cout << "Parameters:   " << numItems << endl;

    // print parameters
    numItems = 0;
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == PARIN)
        {
            char immediate[10];
            switch (port_immediate[i])
            {
            case 0:
                strcpy(immediate, "START");
                break;
            case 1:
                strcpy(immediate, "IMM");
                break;
            default:
                strcpy(immediate, "START");
            }
            cout << "  \"" << port_name[i]
                 << "\" \"" << port_datatype[i]
                 << "\" \"" << port_default[i]
                 << "\" \"" << port_description[i]
                 << "\" \"" << immediate << '"' << endl;
        }

    // count OutPorts
    numItems = 0;
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == OUTPUT_PORT)
            numItems++;
    cout << "OutPorts:     " << numItems << endl;

    // print outPorts
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == OUTPUT_PORT)
        {
            char *dependency;
            if (port_dependency[i])
            {
                dependency = new char[1 + strlen(port_dependency[i])];
                strcpy(dependency, port_dependency[i]);
            }
            else
            {
                dependency = new char[10];
                strcpy(dependency, "default");
            }
            cout << "  \"" << port_name[i]
                 << "\" \"" << port_datatype[i]
                 << "\" \"" << port_description[i]
                 << "\" \"" << dependency << '"' << endl;
        }

    // count InPorts
    numItems = 0;
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == INPUT_PORT)
            numItems++;
    cout << "InPorts:      " << numItems << endl;

    // print InPorts
    for (i = 0; port_name[i]; i++)
        if (port_type[i] == INPUT_PORT)
        {
            char *required = new char[10];
            if (port_required[i] == 0)
            {
                strcpy(required, "opt");
            }
            else
            {
                strcpy(required, "req");
            }

            cout << "  \"" << port_name[i]
                 << "\" \"" << port_datatype[i]
                 << "\" \"" << port_description[i]
                 << "\" \"" << required << '"' << endl;
        }
}

//------------------------------------------------------------------------
//
//------------------------------------------------------------------------
int main(int argc, char *argv[])
{
    //  sleep(40);
    /*    char ch;
         cout << "Eingabe" << endl;
         cin.get(ch);
         cout << "Weiter" << endl;*/

    if ((argc < 7) || (argc > 8))
    {
        if (argc == 2 && 0 == strcmp(argv[1], "-d"))
        {
            set_module_description("OpenInventor 2.0 Renderer");
            add_port(INPUT_PORT,
                     "RenderData",
                     "Geometry|Points|"
                     "Text_Iv|UnstructuredGrid|RectilinearGrid|"
                     "StructuredGrid|Polygons|TriangleStrips|"
                     "Lines",
                     "render geometry or Inventor file");

            add_port(PARIN,
                     "AnnotationString",
                     "String",
                     "Annotation descr. string");

            set_port_default("AnnotationString", "empty");
            set_port_immediate("AnnotationString", 1);

            printDesc(argv[0]);
        }
        else
            cerr << "Application Module with inappropriate arguments called\n";
        exit(0);
    }

    //
    // parse arguments and store them in info class
    //
    m_name = proc_name = argv[0];
    port = atoi(argv[1]);
    h_name = host = argv[2];
    proc_id = atoi(argv[3]);
    instance = argv[4];
    //
    // contact controller
    //
    appmod = new ApplicationProcess(argv[0], argc, argv);
    //  appmod->contact_controller(port, host);

    set_module_description("OpenInventor 2.0 Renderer");

    // INPUT PORT
    add_port(INPUT_PORT,
             "RenderData",
             "Geometry|Points|"
             "Text_Iv|UnstructuredGrid|RectilinearGrid|"
             "StructuredGrid|Polygons|TriangleStrips|"
             "Lines",
             "render geometry or Inventor file");

    // annotation parameter
    add_port(PARIN,
             "AnnotationString",
             "String",
             "Annotation descr. string");

    set_port_default("AnnotationString", "empty");
    set_port_immediate("AnnotationString", 1);

    char* d = get_description_message();
    Message message{ COVISE_MESSAGE_PARINFO , DataHandle{d, strlen(d) + 1, false} };
    appmod->send_ctl_msg(&message);
    strcpy(username, "me");

    // get username from Controller
    strcpy(username, "me");
    char msg_buffer[300];
    sprintf(msg_buffer, "USERNAME\n%s\n%s\n%s\n%d\n", appmod->get_hostname(), m_name, instance, appmod->get_id());

    // set module information  ModuleInfo is a singleton
    std::string myHostname(appmod->get_hostname());
    int inst;
    if (sscanf(instance, "%d", &inst) != 1)
    {
        fprintf(stderr, "IvRenderer: main(): sscanf failed\n");
    }
    ModuleInfo->initialize(myHostname, inst);
    ModuleInfo->setExitOnUninit(); // better be strict

    //      cerr << "MAIN: Modinfo coMsg string: " << ModuleInfo->getCoMsgHeader() << endl;

    Message message2{ COVISE_MESSAGE_UI , DataHandle{msg_buffer, strlen(msg_buffer) + 1, false} };
    appmod->send_ctl_msg(&message2);

    sprintf(msg_buffer, "MODULE_DESC\n%s\n%s\n%s\n%s", m_name, instance, appmod->get_hostname(), module_description);
    message2.data.setLength(strlen(msg_buffer) + 1);
    appmod->send_ctl_msg(&message2);

    print_comment(__LINE__, __FILE__, "Renderer Process succeeded");

    /*    socket_id = appmod->get_socket_id(); in InvPort.h */

    // Initialized Inventor extensions
    SoDB::init();
#ifndef WITHOUT_VIRVO
    SoVolume::initClass();
    SoVolumeDetail::initClass();
#endif
    SoBillboard::initClass();

    //
    // build the render pieces and never return from here
    //
    rm_startup(argc, argv);
}
