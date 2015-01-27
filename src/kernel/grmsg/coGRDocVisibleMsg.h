/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRDocVisibleMsg - sets visibility of opened document in COVER         ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRDOCVISIBLEMSG_H
#define COGRDOCVISIBLEMSG_H

#include "coGRDocMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRDocVisibleMsg : public coGRDocMsg
{
public:
    // construct msg to send
    coGRDocVisibleMsg(const char *document_name, int is_visible);

    // reconstruct from received msg
    coGRDocVisibleMsg(const char *msg);

    // specific functions
    int isVisible()
    {
        return is_visible_;
    };

private:
    int is_visible_;
};
}
#endif
