/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                     (C)2005 Visenso ++
// ++ Description: Definition of some variables for the covise scripting  ++
// ++              for details see scriptInterface.cpp                    ++
// ++                                                                     ++
// ++                                                                     ++
// ++ Author: Ralf Mikulla (rm@visenso.de)                                ++
// ++                                                                     ++
// ++**********************************************************************/
#include <covise/covise.h>
#include <covise/covise_appproc.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <math.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <pwd.h>
#endif
#include "coMsgStruct.h"

const static int BUFSIZE = 6000000; // length of message buffer
const static int TOKMAX = 100000;

extern covise::Message *Ctrl_Msg, *Rend_Msg; // message sending to controller
//extern Message        *Info_Msg, *Error_Msg;                          // message sending to controller
extern covise::UserInterface *UIF; // userinterface pointer
extern char *localHost, *localUser;
extern char Buffer[BUFSIZE]; // global text buffer
extern char *token[TOKMAX];
extern bool Master; // false, if not master userinterface

#ifdef __alpha
extern "C" {
gethostname(char *, int);
}
#endif

int Msg_Parse(char *line, char *token[], int tmax, const char *sep);
CoMsg getSingleMsg();
int openMap(const char *fileName);
int runMap();
int clean();
int run_xuif(int argc, char **argv);
int quit();
int sendCtrlMsg(char *msg);
int sendRendMsg(char *msg);
int sendErrorMsg(char *msg);
int sendInfoMsg(char *msg);

// access to coviseConfig
char *getCoConfigEntry(const char *entry);
char *getCoConfigEntry(const char *entry, const char *variable);
char **getCoConfigSubEntries(const char *entry);
bool coConfigIsOn(const char *entry);
bool coConfigIsOn(const char *entry, const bool &def);
bool coConfigIsOn(const char *entry, const char *variable);
bool coConfigIsOn(const char *entry, const char *variable, const bool &def);
