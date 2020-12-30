/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <appl/CoviseBase.h>

#include <util/unixcompat.h>
#include <util/coFileUtil.h>
#include <net/covise_host.h>
#include <net/concrete_messages.h>
#include <util/coSpawnProgram.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/time.h>
#else
#include <stdio.h>
#include <process.h>
#include <io.h>
#include <direct.h>
#endif

#include <fcntl.h>
#include <stdlib.h>

#include <util/covise_version.h>
#include "CRB_Module.h"
#include <dmgr/dmgr.h>
#include <covise/Covise_Util.h>

#ifdef _NEC
#include <sys/socke.h>
#endif

#include <util/environment.h>
using namespace covise;

int proc_id;
char err_name[20];

bool rendererIsPossible = false;
bool rendererIsActive = false;
DataManagerProcess* datamgr;
Host* host;

int main(int argc, char* argv[])
{
    covise::setupEnvironment(argc, argv);

    int key;
    char* msg_key;
    moduleList mod;

    if (argc != 4 && argc != 5 && argc != 6)
    {
        cerr << "CRB (CoviseRequestBroker) with inappropriate arguments called\n";
        print_exit(__LINE__, __FILE__, 1);
    }

    //cerr << ".................  " << argc << "  " << argv[argc-1] << endl;

#if !defined(_WIN32) && !defined(__APPLE__)
  // set the right DISPLAY environment
    if (argc == 6)
    {
        setenv("DISPLAY", argv[argc - 1], true);
    }
    else
    {
        setenv("DISPLAY", ":0", false);
    }
#endif

    setenv("CO_MODULE_BACKEND", "covise", true);

#ifdef _WIN32
    WORD wVersionRequested;
    WSADATA wsaData;
    int err;
    wVersionRequested = MAKEWORD(1, 1);

    err = WSAStartup(wVersionRequested, &wsaData);
#endif

    int port = atoi(argv[1]);
    host = new Host(argv[2]);
    int id = atoi(argv[3]);
    proc_id = id;
    sprintf(err_name, "err%d", id);
    int send_back = 0;

    key = 2000 + (id << 24);
    datamgr = new DataManagerProcess((char*)"CRB", id, &key);

    datamgr->contact_controller(port, host);

    if (!datamgr->is_connected())
    {
        cerr << "CRB (CoviseRequestBroker) failed to connect to controller\n" << endl;
        print_exit(__LINE__, __FILE__, 1);
    }
    /*
       msg = new Message;
       msg->type = COVISE_MESSAGE_INIT;
       msg->data = (char*)CoviseVersion::shortVersion();
       msg->data.length() = strlen(msg->data)+1;
       datamgr->send_ctl_msg(msg);
       delete msg;
   */

    char* list_content = mod.get_list_message();
    const char* version = CoviseVersion::shortVersion();

    char* list_body = new char[strlen(list_content) + strlen(version) + 5];
    sprintf(list_body, "%s@%s", list_content, version);
    delete[] list_content;

    Message msg{ COVISE_MESSAGE_INIT, DataHandle{list_body, strlen(list_body) + 1 } };

    datamgr->send_ctl_msg(&msg);
    char* d = datamgr->get_list_of_interfaces();//  datamgr allocated data without delete
    msg.data = DataHandle{ d, strlen(d) + 1 };
    datamgr->send_ctl_msg(&msg);

    while (1)
    {
        msg = *datamgr->wait_for_msg();
        switch (msg.type)
        {

            /////  UI

        case COVISE_MESSAGE_QUIT:
            delete datamgr;
            exit(0);
            break;

        case COVISE_MESSAGE_UI: /* get Message-Keyword */
        {
            char bbuf[2000];
            if (msg.data.data())
            {
                strcpy(bbuf, msg.data.data());
            }
            else
            {
                break;
            }

            char* end = msg.data.accessData();
            msg_key = strsep(&end, "\n");

            if (strcmp(msg_key, "QUERY_IMBEDDED_RENDERER") == 0)
            {
                char* name = strsep(&end, " ");
                char* cat = strsep(&end, " ");
                mod.startRenderer(name, cat);
            }

            else if (strcmp(msg_key, "RENDERER_IMBEDDED_ACTIVE") == 0)
            {
                char* name = strsep(&end, "\n");
                (void)name;
                char* cat = strsep(&end, "\n");
                name = cat = NULL;
                char* state = strsep(&end, "\n");
                if (strcmp(state, "TRUE") == 0)
                    rendererIsActive = true;
                else
                    rendererIsActive = false;
            }

            else if (strcmp(msg_key, "RENDERER_IMBEDDED_POSSIBLE") == 0)
            {
                char* name = strsep(&end, "\n");
                (void)name;
                char* cat = strsep(&end, "\n");
                name = cat = NULL;
                char* state = strsep(&end, "\n");
                if (strcmp(state, "TRUE") == 0)
                    rendererIsPossible = true;
                else
                    rendererIsPossible = false;
            }

            else if (strcmp(msg_key, "FILE_SEARCH") == 0)
            {
                int i, num;
                char* tmp;
                char* hostname = strsep(&end, "\n");
                char* user = strsep(&end, "\n");
                char* mod = strsep(&end, "\n");
                char* inst = strsep(&end, "\n");
                char* port = strsep(&end, "\n");
                char* path = strsep(&end, "\n");
                char* sfilt = strsep(&end, "\n");

                //cerr << "____________FILE_SEARCH  " << path << endl;

                if (sfilt == NULL)
                {
                    sfilt = (char*)"*";
                }
                while (sfilt[0] && sfilt[0] == ' ')
                {
                    strcpy(sfilt, sfilt + 1); //get rid of leeding spaces
                }

                if (!path)
                {
                    break;
                }

                CharBuffer buf(1000);
                CharBuffer buf2(1000);
                CharBuffer buf3(1000);

                buf += "FILE_SEARCH_RESULT\n";
                buf += hostname;
                buf += '\n';
                buf += user;
                buf += '\n';
                buf += mod;
                buf += '\n';
                buf += inst;
                buf += '\n';
                buf += port;
                buf += '\n';

                num = 0;
#ifndef _WIN32

                if (path[0] == 0)
                {
                    path[0] = '/';
                    path[1] = 0;
                }
#else

                // check for available drives
                if (path[0] == '/' && path[1] != '\0' && path[2] == ':')
                    path++;

                if (strcmp(path, "") == 0)
                {
                    ULONG uDriveMask = _getdrives();
                    int num2 = -1;
                    if (uDriveMask == 0)
                    {
                        num = -1;
                    }

                    else
                    {
                        ULONG uDriveMask = _getdrives();
                        if (uDriveMask != 0)
                        {
                            char drive[10];
                            int currentDriveNumber = 0;
                            strcpy(drive, "C:");
                            while (uDriveMask)
                            {
                                if (uDriveMask & 1)
                                {
                                    drive[0] = 'A' + currentDriveNumber;
                                    buf2 += drive;
                                    buf2 += '\n';
                                    num++;
                                }
                                uDriveMask >>= 1;
                                currentDriveNumber++;
                            }
                        }
                        buf += num;
                        buf += '\n';
                        buf += (const char*)buf2;
                        buf += num2;
                        buf += '\n';
                        buf += (const char*)buf3;

                        char* b = buf.return_data();
                        Message retmsg{ COVISE_MESSAGE_UI , DataHandle{b, strlen(b) + 1} };
                        datamgr->send_ctl_msg(&retmsg);
                    }
                }

                else
                {
#endif
                    coDirectory* dir = coDirectory::open(path);
                    if (dir)
                    {
                        for (i = 0; i < dir->count(); i++)
                        {
                            tmp = dir->full_name(i);
                            if (dir->is_directory(i))
                            {
                                buf2 += tmp;
                                buf2 += '\n';
                                num++;
                            }
                            delete[] tmp;
                        }
                    }

                    buf += num;
                    buf += '\n';
                    buf += (const char*)buf2;

                    num = 0;
                    if (dir)
                    {
                        for (i = 0; i < dir->count(); i++)
                        {
                            tmp = (char*)dir->name(i);
                            if ((!dir->is_directory(i)) && (dir->match(tmp, sfilt)))
                            {
                                buf3 += tmp;
                                buf3 += '\n';
                                num++;
                            }
                            // the following line seems to be a bad bug
                            // at least it is not a bed bug
                            // and is therefore disabled. awi
                            //delete[] tmp;
                        }
                        delete dir;
                    }

                    buf += num;
                    buf += '\n';
                    buf += (const char*)buf3;
                    //delete[] path;

                    //cerr << "____________FILE_SEARCH  " << buf << endl;
                    char* b = buf.return_data();
                    Message retmsg{ COVISE_MESSAGE_UI , DataHandle{b, strlen(b) + 1} };
                    datamgr->send_ctl_msg(&retmsg);

#ifdef _WIN32
                }
#endif
            }
            else if (strcmp(msg_key, "FILE_LOOKUP") == 0)
            {

                char* hostname = strsep(&end, "\n");
                char* user = strsep(&end, "\n");
                char* mod = strsep(&end, "\n");
                char* inst = strsep(&end, "\n");
                char* port = strsep(&end, "\n");
                char* currpath = strsep(&end, "\n");
                char* filename = strsep(&end, "\n");

                CharBuffer buf(1000);
                buf += "FILE_LOOKUP_RESULT\n";
                buf += hostname;
                buf += '\n';
                buf += user;
                buf += '\n';
                buf += mod;
                buf += '\n';
                buf += inst;
                buf += '\n';
                buf += port;
                buf += '\n';
                buf += filename;
                buf += "\n";

                char slash[2] = "/";
                if (filename[0] == 0)
                {
                    filename = slash;
                }

                bool absPath = false;
                bool found = false;
                // look if filename is absolute or relative
                if (filename[0] == '/' || filename[0] == '~'
#ifdef _WIN32
                    || filename[0] == '\\' || filename[1] == ':'
#endif
                    )
                {
                    absPath = true;
                }

                else
                {
                    char* path = new char[strlen(currpath) + strlen(filename) + 2];
                    strcpy(path, currpath);
                    strcat(path, filename);

                    // make a normal fopen
                    char* returnPath = NULL;
                    FILE* fp = CoviseBase::fopen(path, "r", &returnPath);
                    if (fp)
                    {
                        buf += returnPath;
                        buf += "\n";
                        buf += "FILE";
                        found = true;
                        fclose(fp);
                    }
                    delete[] path;
                }

                if (!found)
                {
                    // make a normal fopen
                    char* returnPath = NULL;
                    FILE* fp = CoviseBase::fopen(filename, "r", &returnPath);
                    if (fp)
                    {
                        buf += returnPath;
                        buf += "\n";
                        buf += "FILE";
                        found = true;
                        fclose(fp);
                    }
                }

                if (!found)
                {
                    if (!absPath)
                    {
                        char* path = new char[strlen(currpath) + strlen(filename) + 2];
                        strcpy(path, currpath);
                        strcat(path, filename);

                        coDirectory* dir = coDirectory::open(path);
                        if (dir)
                        {
                            buf += dir->path();
                            if (buf[strlen(buf) - 1] != '/')
                            {
                                buf += "/";
                            }
                            buf += "\n";
                            buf += "DIRECTORY";
                            buf += "\n";
                            delete dir;
                            found = true;
                        }
                        delete[] path;
                    }

                    if (!found)
                    {
                        coDirectory* dir = coDirectory::open(filename);
                        if (dir)
                        {
                            buf += dir->path();
                            if (buf[strlen(buf) - 1] != '/')
                            {
                                buf += "/";
                            }
                            buf += "\n";
                            buf += "DIRECTORY";
                            buf += "\n";
                            delete dir;
                        }
                    }

                    if (!found)
                    {
                        char* request = new char[strlen(filename) + 1];
                        strcpy(request, filename);
#ifdef _WIN32
                        char* pp = request;
                        while (*pp)
                        {
                            if (*pp == '\\')
                                *pp = '/';
                            pp++;
                        }
#endif

                        char* name = NULL;
                        char* p = strrchr(request, '/');
                        if (p)
                        {
                            name = p + 1;
                            *p = '\0';
                        }
                        else
                        {
                            name = request;
                            request = (char*)"";
                        }

                        if (!absPath)
                        {
                            char* path = new char[strlen(currpath) + strlen(request) + 2];
                            strcpy(path, currpath);
                            strcat(path, request);
                            coDirectory* dir = coDirectory::open(path);
                            if (dir)
                            {
                                buf += dir->path();
                                buf += name;
                                buf += "\n";
                                buf += "NOENT";
                                delete dir;
                                found = true;
                            }
                        }

                        if (!found)
                        {
                            coDirectory* dir = coDirectory::open(request);
                            if (dir)
                            {
                                buf += dir->path();
                                buf += "/";
                                buf += name;
                                buf += "\n";
                                buf += "NOENT";
                                delete dir;
                                found = true;
                            }
                        }
                    }
                }

                if (!found)
                {
                    buf += filename;
                    buf += "\n";
                    buf += "NOENT";
                }

                //cerr << "____________FILE_LOOKUP  " << buf << endl;
                char* b = buf.return_data();
                Message retmsg{ COVISE_MESSAGE_UI , DataHandle{b, strlen(b) + 1} };
                datamgr->send_ctl_msg(&retmsg);
            }

            else
            {
                cerr << "CRB >> UNKNOWN UI MESSAGE" << msg_key << "  " << bbuf << "\n";
            }
        }

        break;
        case COVISE_MESSAGE_CRB_EXEC:
        {
            CRB_EXEC crbExec{ msg };
            if (strcmp(crbExec.category, "") != 0)
            {
                mod.start(crbExec);
                //old code: read name and category out of buffer and pass the rest
            }
            else
            {
                auto a = getCmdArgs(crbExec);
                auto args = cmdArgsToCharVec(a);
                const char* appName = crbExec.name;
                std::string execpath;
                const char* covisedir = getenv("COVISEDIR");
                const char* archsuffix = getenv("ARCHSUFFIX");
                if (covisedir && archsuffix)
                {
                    execpath += std::string(covisedir) + "/" + archsuffix + "/bin/";
                }
#if defined(__APPLE__) && !defined(__USE_WS_X11__)
                if (crbExec.name && !strcmp(crbExec.name, "mapeditor"))
                {
                    execpath += ("COVISE.app/Contents/MacOS/");
                    appName = "COVISE";
                    args[0] = (char*)appName;
                }
                if (crbExec.name && !strcmp(crbExec.name, "wsinterface"))
                {
                    execpath += ("wsinterface.app/Contents/MacOS/");
                }
#endif
                execpath += appName;
                args[0] = execpath.c_str();
                spawnProgram(args[0], args); 
            }
        }
        break;
        case COVISE_MESSAGE_QUERY_DATA_PATH:
        {
            msg.type = COVISE_MESSAGE_SEND_DATA_PATH;
            int len = 2;
            if (getenv("COVISE_PATH"))
            {
                len = (int)strlen(getenv("COVISE_PATH")) + 2;
            }
            msg.data = DataHandle(len);
#ifdef _WIN32
            msg.data.accessData()[0] = ';';
#else
            msg.data.accessData()[0] = ':';
#endif
            if (getenv("COVISE_PATH"))
            {
                strcpy(msg.data.accessData() + 1, getenv("COVISE_PATH"));
#ifdef _WIN32
                char* p = msg.data.accessData() + 1;
                while (*p)
                {
                    if (*p == '\\')
                        *p = '/';
                    p++;
                }
#endif
            }
            else
            {
                strcpy(msg.data.accessData() + 1, "");
            }
            msg.data.setLength(strlen(msg.data.data()) + 1);
            msg.conn->sendMessage(&msg);
        }
        break;

        default:

            send_back = datamgr->handle_msg(&msg);

            if (send_back == 3)
            {
                delete datamgr;
                exit(0);
                //print_exit(__LINE__, __FILE__, 0);
            }
            if ((send_back == 2) && (msg.type != COVISE_MESSAGE_EMPTY))
                msg.conn->sendMessage(&msg);

            break;
        }


        if (datamgr->getConnectionList()->count() <= 0)
        {
            delete datamgr;
            exit(0);
        }
#ifndef _WIN32
        int status;
        while (pid_t pid = waitpid(-1, &status, WNOHANG) > 0)
        {
            fprintf(stderr, "pid %ld exited: status=0x%x\n", (long)pid, status);
        }
#endif
    }
}
