/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GENNORMALS_H
#define _GENNORMALS_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Generate Normals for Polygonal Data	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Uwe Woessner                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  25.08.97  V0.1                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;

#include <util/coviseCompat.h>

// Jetzt brauchen wir die Klasse Covise_Set_Handler nicht mehr

class GenNormals : public coSimpleModule
{
    static const char *s_defMapNames[];
    coChoiceParam *p_normalstyle;
    enum NormalSelectMap
    {
        BisecLargeAngle = 0,
        BisecSmallAngle = 1,
        Orthogonal = 2
    } normalstyle;

private:
    virtual int compute(const char *port);

    ////////// ports
    coInputPort *p_inPort;
    coOutputPort *p_outPort;

    //  Shared memory data
    coDistributedObject *gen_normals(const coDistributedObject *m,
                                     const char *objName);

public:
    GenNormals(int argc, char *argv[]);
};
#endif // _GENNORMALS_H
