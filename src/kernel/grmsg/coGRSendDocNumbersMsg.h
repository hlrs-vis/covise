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

#ifndef COGRSENDDOCNUMBERSMSG_H
#define COGRSENDDOCNUMBERSMSG_H

#include "coGRDocMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRSendDocNumbersMsg : public coGRDocMsg
{
public:
    // construct msg to send
    coGRSendDocNumbersMsg(const char *document_name, int minPage, int maxPage);

    // reconstruct from received msg
    coGRSendDocNumbersMsg(const char *msg);

    // specific functions
    int getMinPage()
    {
        return minPage_;
    };
    int getMaxPage()
    {
        return maxPage_;
    };

private:
    int minPage_, maxPage_;
};
}
#endif
