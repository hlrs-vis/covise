/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PERFORMER_SCENE_H
#define _PERFORMER_SCENE_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Fake read for Performer Models         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  12.10.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <do/coDoPoints.h>

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    static void paramCallback(bool inMapLoading, void *userData, void *callbackData);
    //  Parameter names
    char modelPath[1024];
    char *pointName;

    // Title of the module
    char *d_title;
    const char *getTitle()
    {
        return d_title;
    };
    void setTitle(const char *title);
    // initial Title of the module
    char *d_init_title;

    //  Shared memory data
    coDoPoints *point;

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
#endif // _READIHS_H
