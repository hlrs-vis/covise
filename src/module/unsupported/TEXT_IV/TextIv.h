/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __TEXT_IV
#define __TEXT_IV

#include <stdio.h>
#include <api/coModule.h>
using namespace covise;

class TextIv : public coModule
{
private:
    // port
    coOutputPort *p_textOut;

    // parameters
    coStringParam *param_string;
    coFloatVectorParam *param_translation;
    coFloatVectorParam *param_colour;
    coChoiceParam *param_def_colour;

    int compute();

public:
    TextIv();
};
#endif
