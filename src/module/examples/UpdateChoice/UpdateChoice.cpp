/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1999 RUS  **
 ** Example module for Covise API 2.0 User-interface functions             **
 **                                                                        **
 ** Author:                                                                **
 **                             Andreas Werner                             **
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 ** Date:  23.09.99  V1.0                                                  **
\**************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>

#ifndef _WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#endif

#include <util/coviseCompat.h>
#include "UpdateChoice.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static const char *choiceVal0[] = { "This is", "my primary", "choice", "selection values" };
static const char *choiceVal1[] = { "Now only", "three", "different" };
static const char *choiceVal2[] = { "This", "are", "three", "othe", "ones" };
static const char **choiceVal[3];
static const int choiceLen[3] = { 4, 3, 5 };

UpdateChoice::UpdateChoice(int argc, char *argv[])
    : coModule(argc, argv, "UpdateChoice Program: Show all parameter types")
{
    choiceVal[0] = choiceVal0;
    choiceVal[1] = choiceVal1;
    choiceVal[2] = choiceVal2;

    // Immediate-mode String parameter
    stringImm = addStringParam("stringImm", "Immediate string");
    stringImm->setValue("This is an immediate String Parameter");

    // Now this is a choice : we have the choice between 6 values
    const char *selectChoice[] = { "Choice1", "Choice2", "Choice3" };
    choImm = addChoiceParam("select", "select the other choice's values");
    choImm->setValue(3, selectChoice, 0);

    // Choice to be modified
    choCom = addChoiceParam("choImm", "Nun auch noch Choices");
    choCom->setValue(choiceLen[0], choiceVal[0], 0);

    // Non-immediate String parameter: delivers new value only at compute()
    stringCom = addStringParam("stringCom", "Compute string");
}

void UpdateChoice::postInst()
{
    stringImm->show();
    choImm->show();
    choCom->show();
    stringCom->show();
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Parameter callback: This one is called whenever an immediate
// ++++                      mode parameter is changed, but NOT for
// ++++                      non-immediate ports
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void UpdateChoice::param(const char *name, bool /*inMapLoading*/)
{
    cerr << "\n ------- Parameter Callback for '"
         << name
         << "'" << endl;

    if (strcmp(name, choImm->getName()) == 0)
    {
        int sel = choImm->getValue();
        cerr << "Choice deliveres " << sel << endl;
        choCom->setValue(choiceLen[sel], choiceVal[sel], 1);
    }
}

MODULE_MAIN(Examples, UpdateChoice)
