/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2006 VISENSO    ++
// ++ coGRObjAttachedClipPlaneMsg - stores ClipPlane for an object        ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGROBJATTACHEDCLIPPLANEMSG_H
#define COGROBJATTACHEDCLIPPLANEMSG_H

#include "coGRObjMsg.h"
#include <util/coExport.h>

namespace grmsg
{

class GRMSGEXPORT coGRObjAttachedClipPlaneMsg : public coGRObjMsg
{
public:
    // construct msg to send
    coGRObjAttachedClipPlaneMsg(Mtype type, const char *obj_name, int index, float offset, bool flip);

    // reconstruct from received msg
    coGRObjAttachedClipPlaneMsg(const char *msg);

    // specific functions
    int getClipPlaneIndex()
    {
        return index_;
    };
    float getOffset()
    {
        return offset_;
    };
    bool isFlipped()
    {
        return flip_;
    };

private:
    int index_;
    float offset_;
    bool flip_;
};
}
#endif
