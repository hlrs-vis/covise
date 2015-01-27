/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _DATAHEIGHT_H
#define _DATAHEIGHT_H

/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Use scalar data array to displace polygon mesh              **
 **              in normal direction                                         **
 **                                                                          **
 ** Name:        DataHeight                                                  **                                                        **
 ** Category:    Mapper                                                      **
 **                                                                          **
 ** Author: M. Becker                                                        **
 **                                                                          **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <api/coModule.h>
using namespace covise;

class DataHeight : public coModule
{

private:
    //  member functions
    virtual int compute(const char *port);
    virtual void param(const char *name, bool inMapLoading);
    virtual void postInst();

    //  Ports

    coInputPort *p_polyIn;
    coInputPort *p_dataIn;
    coInputPort *p_normalsIn;

    // output port
    coOutputPort *p_polyOut;

    // parameters
    coFloatParam *p_scale;

public:
    DataHeight(int argc, char *argv[]);
    virtual ~DataHeight();
};
#endif
