/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_FLUVIS_H
#define _READ_FLUVIS_H
/**************************************************************************\ 
 **                                                           (C)1999 RUS  **
 **                                                                        **
 ** Description: Read module for FLUVIS data format         	              **
 **                                                                        **
 ** Author: D. Rainer                                                      **
 **                                                                        **
 ** Date: August 99                                                        **
 **                                                                        **
 *\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>

class Application
{

private:
    int numCoord;

    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    void createGridObject();
    void createVelocityObject();
    void createScalarObject(int i);

public:
    Application(int argc, char *argv[]);
    void run()
    {
        Covise::main_loop();
    }
    ~Application()
    {
    }
};
#endif
