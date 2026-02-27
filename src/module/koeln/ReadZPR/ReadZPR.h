/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifndef _READ_ZPR_H_
#define _READ_ZPR_H_

#include <api/coModule.h>
using namespace covise;

// Read CRD files from ZPR.
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
