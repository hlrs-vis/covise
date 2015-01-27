/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _FENFLOSS_H
#define _FENFLOSS_H
/**************************************************************************\
**                                                                        **
**                                                                        **
** Description: URANUS Simulation                                         **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                             (C)1997 RUS                                **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
** Author:  D. Rantzau, U. Woessner                                       **
** Date:    04.03.97  V1.0                                                **
\**************************************************************************/
#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef __hpux
#define fmain_ FMAIN
#define readbeta_ READBETA
#define timeadv_ TIMEADV
#define mmain_ MMAIN
#endif
extern int computeit;

extern char *filename;
extern int num_timesteps;
extern int num_cells = 0;

extern "C" {

void covise_update_(
    int *n, int *dim);
void visco_test_(char *filename, int *);
void visco_get_vectors_(double *x_coord, double *y_coord, double *u, double *v, int *vl, int *num_c, int *num_conn);
}

class Application
{
private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);
    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    //  Member variables
    char *MeshName, *VelocName, *PressName, *KName, *EpsName, *DensName;
    coDoUnstructuredGrid *Grid;
    coDoVec3 *Veloc;
    coDoFloat *Press;
    coDoFloat *Dens;
    coDoFloat *K;
    coDoFloat *Eps;

public:
    Application(int argc, char *argv[]);
    void run();
    ~Application();
};
#endif // _URANUS_H
