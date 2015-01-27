/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coGRViewpointChangedMsg.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstdio>

namespace grmsg
{

GRMSGEXPORT coGRViewpointChangedMsg::coGRViewpointChangedMsg(const char *msg)
    : coGRMsg(msg)
{
    //fprintf(stderr,"coGRViewpointChangedMsg::coGRViewpointChangedMsg %s\n", msg );

    name_ = NULL;
    view_ = NULL;
    //plane_ = NULL;

    vector<string> tok = getAllTokens();

    if (tok.size() > 0 && !tok[0].empty())
    {
        sscanf(tok[0].c_str(), "%d", &id_);
    }
    else
    {
        id_ = -1;
        is_valid_ = 0;
    }
    if (tok.size() > 1 && !tok[1].empty())
    {
        name_ = strdup(tok[1].c_str());
    }
    else
    {
        name_ = NULL;
        is_valid_ = 0;
    }
    if (tok.size() > 2 && !tok[2].empty())
    {
        view_ = strdup(tok[2].c_str());
    }
    else
    {
        view_ = NULL;
        is_valid_ = 0;
    }
    /*
   if (!tok[3].empty() )
   {
   plane_ = strdup(tok[3].c_str());
}
   else
   {
      //old message set no clipplane
   plane_ = strdup("");
}
   */
}

GRMSGEXPORT coGRViewpointChangedMsg::coGRViewpointChangedMsg(int id, const char *name, const char *view /*, const char *clipplane */)
    : coGRMsg(VIEWPOINT_CHANGED)
{
    //fprintf(stderr,"coGRViewpointChangedMsg::coGRViewpointChangedMsg id=%d\n", id );

    is_valid_ = 1;

    id_ = id;
    name_ = strdup(name);
    view_ = strdup(view);
    //plane_ = strdup(clipplane);

    ostringstream stream;
    stream << id_;

    addToken(stream.str().c_str());
    addToken(name_);
    addToken(view_);
    //addToken( plane_ );
}

GRMSGEXPORT coGRViewpointChangedMsg::~coGRViewpointChangedMsg()
{
    COGRMSG_SAFEFREE(name_);
    COGRMSG_SAFEFREE(view_);
    //COGRMSG_SAFEFREE(plane_);
} //coGRViewpointChangedMsg::~coGRViewpointChangedMsg

GRMSGEXPORT int coGRViewpointChangedMsg::getViewpointId()
{
    return id_;
}

GRMSGEXPORT const char *coGRViewpointChangedMsg::getName()
{
    return name_;
}

GRMSGEXPORT const char *coGRViewpointChangedMsg::getView()
{
    return view_;
}

/*
GRMSGEXPORT const char * coGRViewpointChangedMsg::getClipplane()
{
   return plane_;
}
*/
}
