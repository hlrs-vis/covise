/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _WRITE_ENSIGHT_H
#define _WRITE_ENSIGHT_H

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <api/coSimpleModule.h>
using namespace covise;

// Write coDoLines in Ensight ASCII data format
class coWriteEnsight : public coModule
{
private:
    coFileBrowserParam *pa_filename; // file name of ASCII output file
    coInputPort *po_lines; // receive line data
    coInputPort *po_colors; // receive colors data
    int timesteps; // number of time steps

    int compute();
    void quit();
    void writeGeometryFile(FILE *);
    void writeVariablesFile(FILE *);
    void writeCaseFile(FILE *, char *, char *);
    void strcpyTail(char *, const char *, char);

public:
    coWriteEnsight();
    ~coWriteEnsight();
};
#endif
