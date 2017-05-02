/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DISPLACEUSG_H
#define _DISPLACEUSG_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Module to displace unstructured grids                     **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Reiner Beller                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  29.08.97  V0.1                                                  **
\**************************************************************************/

#include <util/coviseCompat.h>
#include <api/coSimpleModule.h>
using namespace covise;
#include <string>

class DisplaceUSG : public coSimpleModule
{

private:
    coInputPort *inMeshPort, *inDataPort;
    coOutputPort *outMeshPort;
    coFloatParam *paramScale;
    coBooleanParam *paramAbsolute;
    coChoiceParam *p_direction;
    //  member functions
    //void compute(void *callbackData);
    //void quit(void *callbackData);

    //  Static callback stubs
    //static void computeCallback(void *userData, void *callbackData);
    //static void quitCallback(void *userData, void *callbackData);

    //  Local data
    char buf[512];

    //  Shared memory data

    coDistributedObject *displaceNodes(const coDistributedObject *m,
                                       const coDistributedObject *d,
                                       float s,
                                       const char *meshName);

    virtual void preHandleObjects(coInputPort **);
    virtual void postHandleObjects(coOutputPort **);
    const coDistributedObject *p_original_grid_;
    int run_count;
    bool absolute;

public:
    DisplaceUSG(int argc, char *argv[]);
    virtual int compute(const char *port);
};
#endif // _DISPLACEUSG_H
