/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2012 VISENSO    ++
// ++ coGRObjSetVariantMsg - sets the variant of a SceneObject            ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJSETVARIANTMSG_H
#define COGROBJSETVARIANTMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjSetVariantMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjSetVariantMsg(const char *obj_name, const char *groupName, const char *variantName);

    // reconstruct from received msg
    coGRObjSetVariantMsg(const char *msg);
    ~coGRObjSetVariantMsg();

    const char *getGroupName();
    const char *getVariantName();

private:
    char *groupName_;
    char *variantName_;
};
}
#endif
