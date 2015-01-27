/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSetDocScaleMsg.h"
#include <iostream>
#include <sstream>

using namespace grmsg;

GRMSGEXPORT coGRSetDocScaleMsg::coGRSetDocScaleMsg(const char *document_name, float s)
    : coGRDocMsg(SET_DOCUMENT_SCALE, document_name)
{
    scale_ = s;

    ostringstream stream;
    stream << scale_;
    addToken(stream.str().c_str());
    is_valid_ = 1;
}

GRMSGEXPORT coGRSetDocScaleMsg::coGRSetDocScaleMsg(const char *msg)
    : coGRDocMsg(msg)
{
    string scale_str = extractFirstToken();
    istringstream stream(scale_str);
    stream >> scale_;
    is_valid_ = 1;
}

GRMSGEXPORT void coGRSetDocScaleMsg::getScale(float &s)
{
    s = scale_;
}
