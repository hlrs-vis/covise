/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRSetViewpointFile.h"
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstring>

using namespace std;

namespace grmsg
{

    GRMSGEXPORT coGRSetViewpointFile::coGRSetViewpointFile(const char *msg)
        : coGRMsg(msg)
    {
        is_valid_ = 1;
        vector<string> tok = getAllTokens();

        if (!tok[0].empty())
        {
            m_name = tok[0];
        }
        else
        {
            m_name = "";
            is_valid_ = 0;
        }
    }

    GRMSGEXPORT coGRSetViewpointFile::coGRSetViewpointFile(const char *name, int dummy)
        : coGRMsg(SET_VIEWPOINT_FILE), m_name(name)
    {
        (void)dummy;
        is_valid_ = 1;
        ostringstream stream;
        addToken(m_name.c_str());
    }

    const char *coGRSetViewpointFile::getFileName() const
    {
        return m_name.c_str();
    }
}
