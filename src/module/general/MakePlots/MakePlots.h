/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __MAKE_PLOTS_H_
#define __MAKE_PLOTS_H_

/**************************************************************************\ 
 **                                                     (C)2001 Vircinity  **
 **                                                                        **
 ** Description: Make a set of coDoVec2 out of a few        **
 **              DO_Unstructured_S3D data. The object of the first input   **
 **              is the magnitude for the X-axis                           **
 ** Author:                                                                **
 **                                                                        **
 **                            Sergio Leseduarte                           **
 **                            Vircinity GmbH                              **
 **                            Nobelstr. 15                                **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  31.7.2001  (coding begins)                                      **
 **                   1.8.2001: Use properMinMax from COLLECT_TIMESTEPS    **
 **                             This routine is not perfect for all cases!!**
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

class MakePlots : public coModule
{
private:
    enum
    {
        NO_MAX_PLOTS = 6,
        VBIGTICK = 6,
        VSMALLTICK = 12,
        TBIGTICK = 6,
        TSMALLTICK = 12
    };
    float startTime_, minVal_, endTime_, maxVal_;
    float tBigTick_, tSmallTick_, vBigTick_, vSmallTick_;

    coStringParam *p_title, *p_xAxis, *p_yAxis;

    coFloatVectorParam *p_userXlimits;
    coFloatVectorParam *p_userYlimits;
    coBooleanParam *p_auto;
    coIntVectorParam *p_userXticks;
    coIntVectorParam *p_userYticks;
    //     coFloatParam *p_min,*p_max;
    // ports
    coInputPort *p_x;
    coInputPort *p_y[NO_MAX_PLOTS];
    coOutputPort *p_out;

    int noPlots_;
    int noPoints_;

    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    int Diagnose();
    void properMinMax(float &, float &, float &, float &, int);

public:
    MakePlots(int argc, char *argv[]);
};
#endif
