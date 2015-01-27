/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *                    Headerfile for COV_READ                           *
 *									*
 ************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdio.h>
#include <stdlib.h>
#include <util/DLinkList.h>
#define MAX_CELL_ZONES 1000

#define BUFSIZE 64000
#define PUTBACKSIZE 128
#define READ_CELLS 0
#define READ_FACES 1

class Application
{

private:
    //  Local data
    char tmpBuf[1000];
    char *dataFileName;
    int varTypes[1000];
    int varIsFace[1000];
    int numVars;
    int numCells, numFaces, numNodes, numVertices, numElements;
    float *x_coords;
    float *y_coords;
    float *z_coords;
    int *elementTypeList;
    int *vertices;

    int numFreeElements;
    int numFreeAlloc;

    //  member functions
    void paramChange(void *callbackData);
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void executeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    static void paramCallback(bool inMapLoading, void *userData, void *callbackData);

    void updateChoice();

public:
    Application(int argc, char *argv[]);
    ~Application();
    void run();

} *application;
