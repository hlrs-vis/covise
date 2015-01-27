/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MIXCOLORS_H
#define _MIXCOLORS_H

#include <api/coSimpleModule.h>

using namespace covise;

class MixColors : public coSimpleModule
{

private:
    virtual int compute(const char *port);

    coInputPort *p_colors1, *p_colors2;
    coOutputPort *p_colorsOut;

public:
    MixColors(int argc, char *argv[]);
};
#endif
