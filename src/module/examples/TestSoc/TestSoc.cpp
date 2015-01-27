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

#include "TestSoc.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#endif

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TestSoc::TestSoc(int argc, char *argv[])
    : coModule(argc, argv, "TestSoc Program: Test socket handling")
{
    // no parameters, no ports...
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  postInst() is called once after we contacted Covise, but before
// ++++             getting into the main loop
// ++++
// ++++  We open our sockets here to prevent timeouts
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void TestSoc::postInst()
{
    // Open 3 sockets to some UNIX pipes

    // beware: ApplInterface->coModule->TestSoc inherits own 'open'
    fd1 = ::open("/var/tmp/SOC1", O_RDONLY);
    if (fd1 >= 0)
        addSocket(fd1);
    else
        cerr << "could not open /var/tmp/SOC1" << endl;

    fd2 = ::open("/var/tmp/SOC2", O_RDONLY);
    if (fd2 >= 0)
        addSocket(fd2);
    else
        cerr << "could not open /var/tmp/SOC2" << endl;

    fd3 = ::open("/var/tmp/SOC3", O_RDONLY);
    if (fd3 >= 0)
        addSocket(fd3);
    else
        cerr << "could not open /var/tmp/SOC3" << endl;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  sockData(int soc) is when data arrives on registered sockets
// ++++             getting into the main loop
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void TestSoc::sockData(int soc)
{
    cerr << "TestSoc::sockData called for socket number " << soc << endl;
    char buffer[65536];
    int len = read(soc, buffer, 65535);
    cerr << " read " << len << " bytes" << endl;
    if (len < 1)
    {
        cerr << "Ending socket " << soc << " monitoring" << endl;
        removeSocket(soc);
    }
    else
    {
        buffer[len] = '\0';
        cerr << buffer << endl;
    }
}

MODULE_MAIN(Examples, TestSoc)
