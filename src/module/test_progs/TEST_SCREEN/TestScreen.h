/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __TEST_SCREEN_H_
#define __TEST_SCREEN_H_

// 09.05.01

#include <api/coModule.h>
using namespace covise;

/**
 * Class
 *
 */
class TestScreen : public coModule
{

private:
    /// Copy-Constructor: NOT  IMPLEMENTED
    TestScreen(const TestScreen &);

    /// Assignment operator: NOT  IMPLEMENTED
    TestScreen &operator=(const TestScreen &);

    /// this module has only the compute call-back
    virtual int compute();

    ////////// parameters
    coChoiceParam *p_whatOut;

    ////////// ports
    coOutputPort *p_outPort;
    coInt32SliderParam *p_numX, *p_numY, *p_numZ;

public:
    /// Default constructor
    TestScreen();
};
#endif
