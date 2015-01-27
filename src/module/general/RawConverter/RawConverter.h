/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2008 HLRS **
 **                                                                        **
 ** Description: RawConverter .                                            **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:             Bruno Burbaum                                      **
 **                                                                        **
 **                                                                        **
 ** Cration Date: 9.10.2008                                               **
\***************************************************************************/

#ifndef _RAW_CONCERTER_H // OK, Verhindert mehrfach aufrufe
#define _RAW_CONCERTER_H //

#include <api/coModule.h>
using namespace covise; //

class RawConverter : public coModule
{

public:
    RawConverter(int argc, char **argv);
    virtual ~RawConverter();

private:
    static const int MAX_GRAY_VALUES = 65536;
    // Ports:
    coInputPort *poIVolume;
    coOutputPort *poOVolume;
    coOutputPort *p_Histo_In;
    coOutputPort *p_Histo_Out;

    // Parameters:
    coBooleanParam *pboAutoSize;
    coBooleanParam *pboLOGScale;
    //	    coIntSliderParam* piSequenceBegin;
    coIntScalarParam *piSequenceBegin;
    coIntScalarParam *piSequenceEnd;
    coFloatParam *pfsResultsBegin;
    coFloatParam *pfsResultsEnd;
    coBooleanParam *pboByteOrFloats;
    coFloatSliderParam *p_cutoff;
    coFloatSliderParam *p_pre_cutoff;
    coFloatSliderParam *p_logOffset;

    // Methods:
    virtual int compute(const char *port);
    void MakeHistogram(coOutputPort *myport, bool getfloats, float numColors, int numValues, int *intDataValues, float *floatDataValues);
    void MakeAutoScale();
};

#endif
