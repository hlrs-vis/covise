/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2008 VISENSO    ++
// ++ coGRSendDocNumbersMsg - sends min and may mumber of pages to GUI    ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRSETDOCPAGESIZE_H
#define COGRSETDOCPAGESIZE_H

#include "coGRDocMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRSetDocPageSizeMsg : public coGRDocMsg
{
public:
    // construct msg to send
    coGRSetDocPageSizeMsg(const char *document_name, int pageNo, float hsize, float vsize);

    // reconstruct from received msg
    coGRSetDocPageSizeMsg(const char *msg);

    // specific functions
    float getHSize()
    {
        return hSize_;
    };
    float getVSize()
    {
        return vSize_;
    };
    int getPageNo()
    {
        return pageNo_;
    };

private:
    float hSize_, vSize_;
    int pageNo_;
};
}
#endif
