/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// MODULE        scriptInterface.cpp
//
// Description:  This file contains interface functions to interface covise with
//               python. The method chosen requires SWIG (www.swig.org)
//               version 1.3.17 or higher.
//               The compilation result is a python modulue, which is dynamically
//               loaded when calling import covise in python.
//
// Initial version: 10.03.2003 (RM)
// Cleaned up; represents now a minimal interface 01.07.2005 (rm@visenso.de)
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2003 by VirCinity IT Consulting
// (C) 2005 by Visual Engieering Solutions GmbH, Stuttgart
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//

#include <covise/covise.h>
#include <net/covise_socket.h>
#include <net/dataHandle.h>
#include "config/CoviseConfig.h"
#include <util/coLog.h>
#include "scriptInterface.h"
#include "coMsgStruct.h"

using covise::Message;
using covise::DataHandle;
using covise::UserInterface;
using covise::COVISE_MESSAGE_UI;
using covise::COVISE_MESSAGE_RENDER;
using covise::USERINTERFACE;
using covise::print_exit;
using covise::print_comment;
using covise::COVISE_MESSAGE_QUIT;
using covise::COVISE_MESSAGE_MSG_OK;
using covise::coCoviseConfig;

Message *Ctrl_Msg; // message sending to controller
Message *Rend_Msg; // message sending to controller
UserInterface *UIF; // userinterface pointer
char *localHost, *localUser;
char Buffer[BUFSIZE]; // global text buffer
char *token[TOKMAX];
bool Master; // false, if not master userinterface

// initialize a connection to COVISE (controller)
int
run_xuif(int argc, char **argv)
{
    int nn;
    struct passwd *pwd;
    // check connection parameter
    if ((argc < 7) || (argc > 8))
    {
        fprintf(stderr, "COVISE python interface with inappropriate arguments called  %d\n", argc);
        print_exit(__LINE__, __FILE__, 0);
    }

    covise::Socket::initialize();

    // create userinterface process
    UIF = new UserInterface("AppModule", argc, argv);

    // get local host & username
    localHost = (char *)UIF->get_hostname();
//   pwd           = getpwuid(getuid());
//   localUser  = (char *)pwd->pw_name;
#ifndef _WIN32
    localUser = getlogin();
#else
    localUser = getenv("USERNAME");
#endif

    // init message for controller
    Ctrl_Msg = new Message();
    Ctrl_Msg->type = COVISE_MESSAGE_UI;
    Ctrl_Msg->send_type = USERINTERFACE;

    Rend_Msg = new Message();
    Rend_Msg->type = COVISE_MESSAGE_RENDER;
    Rend_Msg->send_type = USERINTERFACE;

    /**
   Error_Msg = new Message();
   Error_Msg->type = COVISE_MESSAGE_COVISE_ERROR;
   Error_Msg->send_type = USERINTERFACE;

   Info_Msg = new Message();
   Info_Msg->type = COVISE_MESSAGE_INFO;
   Info_Msg->send_type = USERINTERFACE;
**/
    print_comment(__LINE__, __FILE__, "scripting interface started");
    nn = argc;

    // wait for master
    while (1)
    {
        Message *msg = UIF->wait_for_msg();
        if (msg->type == COVISE_MESSAGE_QUIT)
        {
            delete UIF;
            print_exit(__LINE__, __FILE__, 0);
        }

        else if (msg->type == COVISE_MESSAGE_UI) // parse message
        {
            Msg_Parse(msg->data.accessData(), token, TOKMAX, "\n");
            if (strcmp(token[0], "MASTER") == 0)
            {
                Master = true;
                //fprintf(stderr, "xuif : MASTER \n");
                delete msg;
                break;
            }
            else if (strcmp(token[0], "SLAVE") == 0)
            {
                Master = false;
                //fprintf(stderr, "xuif : SLAVE \n");
                delete msg;
                break;
            }
        }
        delete msg;
    }

    Master = true;

    Message *msg = new Message(COVISE_MESSAGE_MSG_OK);
    UIF->send_ctl_msg(msg);
    delete msg;

    return 0;
}

// send a message to open a covise-map file
// note this is done by the controller
int
openMap(const char *fileName)
{
    char *buffer = new char[512];
    char newBuf[512];

    strcpy(newBuf, fileName);
    strcat(newBuf, "\n");

    strcpy(buffer, "NEW\n");
    Ctrl_Msg->data = covise::DataHandle(buffer, strlen(buffer) + 1);
    UIF->send_ctl_msg(Ctrl_Msg);

    strcpy(buffer, "OPEN\n");
    strcat(buffer, newBuf);
    Ctrl_Msg->data = covise::DataHandle(buffer, strlen(buffer) + 1);
    UIF->send_ctl_msg(Ctrl_Msg);

    return (0);
}

// send a exec-message to the controller
int
runMap()
{
    char* buffer = new char[512];

    strcpy(buffer, "EXEC\n");
    Ctrl_Msg->data = covise::DataHandle(buffer, strlen(buffer) + 1);
    UIF->send_ctl_msg(Ctrl_Msg);
    return (0);
}

// clean all
int
clean()
{
    strcpy(Buffer, "NEW\n");
    Ctrl_Msg->data = DataHandle{ Buffer, Ctrl_Msg->data.length(), false };
    UIF->send_ctl_msg(Ctrl_Msg);
    return (0);
}

// quit covise by sending a quit message
int
quit()
{

    Message msg{ COVISE_MESSAGE_QUIT, covise::DataHandle() };
    UIF->send_ctl_msg(&msg);

    return (0);
}

// checks for incoming messages and returns it as CoMsg struct
// this function is the base for receiving covise msgs.
CoMsg
getSingleMsg()
{
    CoMsg ret;
    ret.type = -1;
    ret.data = nullptr;
    Message *msg = UIF->check_for_msg();
    if (msg != nullptr)
    {
        ret.type = msg->type;
        ret.data = msg->data.accessData();
        ret.dh = msg->data;
        delete msg;
    }

    return ret;
}

/**
   @param msg -- Message-string
   @param aMessageDataNotSetYet -- A message-object
   @return 0 (Always succeed.)

   Fill @p msg into @p aMessageDataNotSetYet and send
   it to the UserInterface which may be the controller.

   This is the heart of matter for covise scripting!

   @todo
   @li Buffer-overflow is very likely here.  Change.
*/
static int
sendMsgToUIF(char *msg, Message *aMessageDataNotSetYet)
{
    char *buffer = new char[strlen(msg) + 1];
    strcpy(buffer, msg);
    aMessageDataNotSetYet->data = covise::DataHandle(buffer, strlen(buffer) + 1);
    if (UIF)
        UIF->send_ctl_msg(aMessageDataNotSetYet);
    return (0);
}

// sends a ctrl-msg. This is the heart of matter for covise scripting!
int
sendCtrlMsg(char *msg)
{
    return sendMsgToUIF(msg, Ctrl_Msg);
}

int
sendRendMsg(char *msg)
{
    return sendMsgToUIF(msg, Rend_Msg);
}

int
sendErrorMsg(char *msg)
{
    return sendMsgToUIF(msg, Ctrl_Msg);
}

int
sendInfoMsg(char *msg)
{
    return sendMsgToUIF(msg, Ctrl_Msg);
}

// helper -not wrapped -
int
Msg_Parse(char *line, char *token[], int tmax, const char *sep)
{
    char *tp;
    int count;

    count = 0;
    tp = strtok(line, sep);

    for (count = 0; count < tmax && tp != NULL;)
    {
        token[count] = tp;
        //delete [] tp;
        tp = strtok(NULL, sep);
        count++;
    }
    token[count] = NULL;
    return count;
}

char *
getCoConfigEntry(const char *entry)
{
    std::string value;
    bool entryExists;

    value = coCoviseConfig::getEntry(entry, &entryExists);

    if (entryExists)
    {
        char *v = strdup(value.c_str());
        return v;
    }
    else
    {
        return NULL;
    }
    //return (char *)coCoviseConfig::getEntry(entry).c_str();
}

char *
getCoConfigEntry(const char *entry, const char *variable)
{
    std::string value;
    bool entryExists;

    value = coCoviseConfig::getEntry(variable, entry, &entryExists);

    if (entryExists)
    {
        char *v = strdup(value.c_str());
        return v;
    }
    else
    {
        return NULL;
    }
    //return (char *)coCoviseConfig::getEntry(variable, entry).c_str();
}

char **
getCoConfigSubEntries(const char *entry)
{
    coCoviseConfig::ScopeEntries keysEntries = coCoviseConfig::getScopeEntries(entry);
    const char **keys = keysEntries.getValue();

    if (keys == NULL)
    {
        return NULL;
    }

    // create copy of keys and skip every second entry (is NULL)
    char **subKeys;
    int numSubKeys = 0;
    while (keys[2 * numSubKeys] != NULL)
    {
        numSubKeys++;
    }
    subKeys = new char *[numSubKeys + 1];
    subKeys[numSubKeys] = NULL;
    for (int i = 0; i < numSubKeys; i++)
    {
        int str_len = strlen(keys[2 * i]);
        subKeys[i] = new char[str_len + 1];
        strcpy(subKeys[i], keys[2 * i]);
    }

    return subKeys;
}

bool
coConfigIsOn(const char *entry)
{
    return coCoviseConfig::isOn(entry, false);
}

bool
coConfigIsOn(const char *entry, const bool &def)
{
    return coCoviseConfig::isOn(entry, def);
}

bool
coConfigIsOn(const char *entry, const char *variable)
{
    bool f = false;
    return coCoviseConfig::isOn(variable, entry, f);
}

bool
coConfigIsOn(const char *entry, const char *variable, const bool &def)
{
    return coCoviseConfig::isOn(variable, entry, def);
}
