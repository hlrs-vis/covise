/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRChangeViewpointMsg.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>

using namespace grmsg;

GRMSGEXPORT coGRChangeViewpointMsg::coGRChangeViewpointMsg(const char *msg)
    : coGRMsg(msg)
{
    //plane_ = NULL;

    vector<string> tok = getAllTokens();
    if (!tok[0].empty())
    {
        sscanf(tok[0].c_str(), "%d", &id_);
    }
    else
    {
        id_ = -1;
        is_valid_ = 0;
    }
    /*if (!tok[1].empty() )
   {
      plane_ = strdup(tok[1].c_str());
   }
   else
   {
      //old message set no clipplane
      plane_ = strdup("");
   }
   */

    //fprintf(stderr,"coGRChangeViewpointMsg::coGRChangeViewpointMsg %s\n", msg );
}

GRMSGEXPORT coGRChangeViewpointMsg::coGRChangeViewpointMsg(int id /*, const char *clipplane */)
    : coGRMsg(CHANGE_VIEWPOINT)
{
    //fprintf(stderr,"coGRChangeViewpointMsg::coGRChangeViewpointMsg id=%d\n", id );

    is_valid_ = 1;

    id_ = id;
    //plane_ = strdup(clipplane);

    ostringstream stream;
    stream << id_;

    addToken(stream.str().c_str());
    //addToken( plane_ );
}

GRMSGEXPORT int coGRChangeViewpointMsg::getViewpointId()
{
    return id_;
}

/*
GRMSGEXPORT const char * coGRChangeViewpointMsg::getClipplane()
{
   return plane_;
}
*/
