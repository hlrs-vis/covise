/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                                     ++
// ++ coGRSetViewpointFile - message to use a new viewpoint file          ++
// ++                                                                     ++
// ++                                                                     ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef COGRCHANGE_VIEWPOINT_FILE_H
#define COGRCHANGE_VIEWPOINT_FILE_H

#include "coGRMsg.h"
#include <util/coExport.h>
#include <string>
namespace grmsg
{

    class GRMSGEXPORT coGRSetViewpointFile : public coGRMsg
    {
    public:
        const char *getFileName() const;

        coGRSetViewpointFile(const char *name, int dummy);
        coGRSetViewpointFile(const char *msg);

    private:
        std::string m_name;
    };
}
#endif
