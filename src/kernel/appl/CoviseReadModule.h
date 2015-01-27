/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// basic ReaderModule - class for COVISE
// by Lars Frenzel
//
// history:   	01/16/1998  	started working
//

#if !defined(__COVISE_READ_MODULE_H)
#define __COVISE_READ_MODULE_H

#include "ApplInterface.h"
#include <util/coTypes.h>

class APPLEXPORT CoviseReadModule
{
private:
    // callbacks
    static void computeCallback(void *userData, void *);

public:
    // basic stuff
    CoviseReadModule();
    ~CoviseReadModule();

    // setting up the callback-functions
    void setCallbacks();

    // virtual stuff - should be provided by programmer
    virtual void compute();
};
#endif // __COVISE_READ_MODULE_H
