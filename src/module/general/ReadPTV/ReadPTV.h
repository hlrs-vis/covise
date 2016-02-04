/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                          (C)2004 HLRS  **
**                                                                        **
** Description: Read PTV files.                                           **
**                                                                        **
**                                                                        **
** Author:                                                                **
**                                                                        **
**                          Uwe Woessner                                  **
**     High Performance Computing Center University of Stuttgart          **
**                         Allmandring 30                                 **
**                         70550 Stuttgart                                **
**                                                                        **
** Cration Date: 14.04.2004                                               **
\**************************************************************************/

#ifndef READ_PTV_H
#define READ_PTV_H

#include <api/coModule.h>
using namespace covise;

class coReadPTV : public coModule
{
private:
    // Ports:
    coOutputPort *poPoints;
    coOutputPort *poVelos;
    coOutputPort *poTypes;

    // Parameters:
    coFileBrowserParam *pbrFilename; ///< name of first checkpoint file of a sequence
    coIntScalarParam *pLimitTimesteps; ///< 0 = unlimited

    // Methods:
    virtual int compute(const char *port);

public:
    coReadPTV(int argc, char *argv[]);
};

#endif
