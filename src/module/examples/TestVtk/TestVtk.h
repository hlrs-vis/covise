/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TESTVTK_H
#define TESTVTK_H


#include <api/coSimpleModule.h>
using namespace covise;

class TestVtk : public coSimpleModule
{

private:
    //  member functions
    virtual int compute(const char *port);

    //  Ports
    coInputPort *input;
    coOutputPort *output;

public:
    TestVtk(int argc, char *argv[]);
    virtual ~TestVtk();
};
#endif
