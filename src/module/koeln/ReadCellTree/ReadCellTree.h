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

class coReadCellTree : public coModule
{
private:
    // Ports:
    coOutputPort *poPoints;
    coOutputPort *poTreeLines;
    coOutputPort *poLines;
    coOutputPort *poDiameters;
    coOutputPort *poConnectivity;

    // Parameters:
    coFileBrowserParam *pFile; ///< name of data file containing adjacency matrix
    coFileBrowserParam *pCellDescFile; ///< name of file for cell descriptions
    coChoiceParam *pRootCell; ///< number of root cell type

    // members
    char *lineBuf;
    size_t lineBufSize;

    // Methods:
    virtual int compute(const char *port);
    virtual void param(const char *paramName, bool inMapLoading);
    float absVector(float, float, float);
    bool readLine(FILE *fp);
    bool readArray(FILE *fp, float **data, int numElems);

    vector<char *> cellTypes;

public:
    coReadCellTree(int argc, char *argv[]);
    bool displayWarnings();
};

#endif
