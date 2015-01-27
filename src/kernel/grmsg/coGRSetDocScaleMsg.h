/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRSetDocScaleMsg - scale document in COVER            ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSETDOCSCALEMSG_H
#define COGRSETDOCSCALEMSG_H

#include "coGRDocMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRSetDocScaleMsg : public coGRDocMsg
{
public:
    // construct msg to send
    coGRSetDocScaleMsg(const char *document_name, float s);

    // reconstruct from received msg
    coGRSetDocScaleMsg(const char *msg);

    // specific functions
    void getScale(float &s);

private:
    float scale_;
};
}
#endif
