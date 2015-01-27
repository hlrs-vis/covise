/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <api/coModule.h>
using namespace covise;

class ReadGromacs : public coModule
{
private:
    // Modul functions
    virtual int compute(const char *port);
    void readFileGro();
    void readFileXtc();

    // Ports
    coOutputPort *pointsOutput;
    coOutputPort *elementout;
    coOutputPort *pointsAnimationOutput;

    // Parameters
    coFileBrowserParam *groFileParam;
    coFileBrowserParam *xtcFileParam;

    // Data
    const char *fname;

public:
    ReadGromacs(int argc, char *argv[]);
};
