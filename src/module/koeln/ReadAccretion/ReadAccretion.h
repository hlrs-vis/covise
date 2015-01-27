/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                          (C)2004 RRZK  **
**                                                                        **
** Description: Read XYZ files.                                           **
**                                                                        **
**                                                                        **
** Author:                                                                **
**                                                                        **
**                         Martin Aumueller                               **
**     High Performance Computing Center University of Stuttgart          **
**                         Allmandring 30                                 **
**                         70550 Stuttgart                                **
**                                                                        **
** Cration Date: 14.04.2004                                               **
\**************************************************************************/

#ifndef READ_XYZ_H
#define READ_XYZ_H

#include <api/coModule.h>
using namespace covise;
#include <map>
#include <string>

class coReadAccretion : public coModule
{
private:
    // Ports:
    coOutputPort *poPoint;
    coOutputPort *poVel;
    coOutputPort *poMass;
    coOutputPort *poInitialMass;
    coOutputPort *poMask;
    coOutputPort *poId;
    coOutputPort *poAttr;

    coOutputPort *poVStars, *poRStars, *poMassStars;

    // Parameters:
    coFileBrowserParam *pbrFilename; ///< name of first checkpoint file of a sequence
    coIntScalarParam *pStartTimestep;
    coIntScalarParam *pLimitTimesteps; ///< 0 = unlimited

    // Methods:
    virtual int compute(const char *port);

public:
    coReadAccretion(int argc, char *argv[]);
    bool displayWarnings();
};

#endif
