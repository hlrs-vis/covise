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

#ifndef _READ_ZPR_H_
#define _READ_ZPR_H_

#include <api/coModule.h>
using namespace covise;

class coReadZPR : public coModule
{
private:
    // Ports:
    coOutputPort *poPoints;

    // Parameters:
    coFileBrowserParam *pbrFilename; ///< name of first checkpoint file of a sequence

    // Methods:
    virtual int compute(const char *port);

public:
    coReadZPR(int argc, char *argv[]);
};

#endif
