/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INTERPOLATE_H
#define _INTERPOLATE_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Module to interpolate between unstructured data types     **
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
 ** Date:  10.10.97  V0.1                                                  **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <util/coviseCompat.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif // M_PI

class Interpolate : public coModule
{

private:
    //  member functions
    int compute(const char *port);

    //  Local data
    void interpolate(const coDistributedObject *d,
                     const int m,
                     const int s,
                     char *dataName,
                     int no,
                     coDistributedObject **outDataPtr);

    void interpolate_two_fields(
        const coDistributedObject *d,
        const coDistributedObject *f,
        const int m,
        const int s,
        char *dataName,
        int no,
        coDistributedObject **outDataPtr);

    void setupCycle(coDistributedObject **d,
                    const int s,
                    const int key,
                    const char *namebase,
                    coDistributedObject **outDataPtr1,
                    coDistributedObject **outDataPtr2,
                    coDistributedObject **outDataPtr3);

    // Ports and Parameters
    coInputPort *p_dataIn_1, *p_dataIn_2, *p_dataIn_3, *p_indexIn;
    coChoiceParam *p_type, *p_motion;
    coIntSliderParam *p_steps;
    coBooleanParam *p_abs;
    coBooleanParam *p_osci;
    coOutputPort *p_dataOut_1, *p_dataOut_2, *p_indexOut;

    float full_osci;
    int f_start, f_end;

public:
    virtual ~Interpolate()
    {
    }
    Interpolate(int argc, char *argv[]);
};
#endif // _INTERPOLATE_H
