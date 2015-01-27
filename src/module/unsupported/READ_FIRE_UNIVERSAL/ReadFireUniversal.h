/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _PATRAN_H
#define _PATRAN_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for PATRAN Neutral and Results Files          **
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
#include "FireUniversalFile.h"
#include "ChoiceList.h"
#ifdef __hpux
#include <string.h>
#include <strings.h>
#endif
const int READ_DATA = 0;
const int DETERMINE_SIZE = 1;
const int MAX_NO_OF_DATA_SETS = 50;
const int DATA_SCALAR = 1;
const int DATA_VECTOR = 3;

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

    ChoiceList *choicelist;
    FireFile *firegridfile, *firedatafile;
    char *gridfile_name;
    char *datafile_name;
    int no_of_grid_points;
    int no_of_elements;
    int no_of_data_sets;
    long data_start[MAX_NO_OF_DATA_SETS];
    char *data_name[MAX_NO_OF_DATA_SETS];
    int data_type[MAX_NO_OF_DATA_SETS];
    int ident;
    int no1, no2, no3;

public:
    Application(int argc, char *argv[]);
    ~Application();
    inline void run()
    {
        Covise::main_loop();
    }
    int ReadGrid(int);
    int ReadData(int);
    int WriteFireMesh(){};
    int WriteFireData(int, int, int){};
    void ResetChoiceList(void);
    void UpdateChoiceList(void);
}
#endif // _PATRAN_H
