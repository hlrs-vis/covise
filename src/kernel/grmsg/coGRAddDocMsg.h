/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2007 VISENSO    ++
// ++ coGRAddDocMsg - show document in COVER                              ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRADDDOCMSG_H
#define COGRADDDOCMSG_H

#include "coGRDocMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRAddDocMsg : public coGRDocMsg
{
public:
    // construct msg to send
    coGRAddDocMsg(const char *document_name, const char *image_name);

    // reconstruct from received msg
    coGRAddDocMsg(const char *msg);

    // destructor
    ~coGRAddDocMsg();

    // specific functions
    const char *getImageName()
    {
        return imageName_;
    };

private:
    char *imageName_;
};
}
#endif
