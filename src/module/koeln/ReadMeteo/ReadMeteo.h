/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                           (C)2002 RUS  **
**                                                                        **
** Description: Read IMD checkpoint files from ITAP.                      **
**                                                                        **
**                                                                        **
** Author:                                                                **
**                                                                        **
**                     Juergen Schulze-Doebold                            **
**     High Performance Computing Center University of Stuttgart          **
**                         Allmandring 30                                 **
**                         70550 Stuttgart                                **
**                                                                        **
** Cration Date: 03.09.2002                                               **
\**************************************************************************/

#ifndef _READ_HEIGHTFIELD_H_
#define _READ_HEIGHTFIELD_H_

#include <api/coModule.h>
using namespace covise;

class coReadMeteo : public coModule
{
private:
    // Ports:
    coOutputPort *poData;

    // Parameters:
    coFileBrowserParam *pbrFile; ///< name of first file of a sequence
    coIntScalarParam *pDimX; ///< dim X
    coIntScalarParam *pDimY; ///< dim Y
    coIntScalarParam *pDimZ; ///< dim Y
    coFloatParam *pScale; ///< Scale factor for data
    coBooleanParam *pboDataMode; ///< true = don't interpret data as height information
    coBooleanParam *pboWarnings; ///< true = display warnings when reading a file

    // members
    char *lineBuf;
    size_t lineBufSize;

    // Methods:
    virtual int compute(const char *port);
    float absVector(float, float, float);
    bool readLine(FILE *fp);
    bool readArray(FILE *fp, float **data, int numElems);

public:
    coReadMeteo(int argc, char *argv[]);
    bool displayWarnings();
};

#endif
