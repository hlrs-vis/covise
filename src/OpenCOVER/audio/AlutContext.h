/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COVER_AUDIO_ALUT_CONTEXT_H_
#define _COVER_AUDIO_ALUT_CONTEXT_H_

#include <util/coExport.h>

namespace opencover::audio
{

class COVRAUDIOEXPORT AlutContext
{
public:
    AlutContext();
    ~AlutContext();

    static bool is_initialized;
    static bool has_context;

private:
    static int _refcount;
};

}

#endif
