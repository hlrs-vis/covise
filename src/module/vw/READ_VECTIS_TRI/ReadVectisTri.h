/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_VECTIS_TRI_H
#define _READ_VECTIS_TRI_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for VEXTIS Triangle Files                     **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Reiner Beller                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  18.07.97  V0.1                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);
    void paramChange(void *);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    static void paramCallback(void *userData, void *callbackData);

    //  Local data

    char *file_name;
    int num_points;
    int num_triangles;
    char vect_file;

public:
    Application(int argc, char *argv[]);
    ~Application();
    inline void run()
    {
        Covise::main_loop();
    }
};
#endif // _READ_VECTIS_TRI_H
