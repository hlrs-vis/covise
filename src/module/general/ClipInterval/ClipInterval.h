/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CLIP_INTERVAL_H
#define _CLIP_INTERVAL_H
/**************************************************************************\ 
 **                                                     (C)2000 Vircinity  **
 **                                                                        **
 ** Description: Clip Interval Application                                 **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Ralph Bruckschen                            **
 **                            Vircinity GmbH                              **
 **                            Nobelstr. 15                                **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  30.0.00  V0.1                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>

class Clip_Interval : public coSimpleModule
{
public:
    enum
    {
        NUM_ADDITIONAL_PORTS = 5
    };

private:
    //////////  member functions
    virtual int compute(const char *port);
    virtual void preHandleObjects(coInputPort **);
    virtual void quit();
    virtual void postInst();

    ////////// the data in- and output ports
    coInputPort *Geo_In_Port, *Data_In_Port,
        *Data_Map_In_Port[NUM_ADDITIONAL_PORTS];
    coInputPort *p_minmax;
    coOutputPort *Geo_Out_Port, *Data_Out_Port,
        *Data_Map_Out_Port[NUM_ADDITIONAL_PORTS];
    coFloatSliderParam *Min_Slider, *Max_Slider;
    coBooleanParam *p_dummy;
    coBooleanParam *p_autominmax;

public:
    Clip_Interval(int argc, char *argv[]);
};
#endif // _CLIP_INTERVAL_H
