/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRSetDocPageMsg - set page of opened document in COVER            ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSETDOCPAGEMSG_H
#define COGRSETDOCPAGEMSG_H

#include "coGRDocMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRSetDocPageMsg : public coGRDocMsg
{
public:
    // construct msg to send
    coGRSetDocPageMsg(const char *document_name, int page);

    // reconstruct from received msg
    coGRSetDocPageMsg(const char *msg);

    // specific functions
    int getPage()
    {
        return page_;
    };

private:
    int page_;
};
}
#endif
