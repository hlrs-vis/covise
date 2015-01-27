/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cstring>
#include "coGRAddDocMsg.h"

using namespace grmsg;

GRMSGEXPORT coGRAddDocMsg::coGRAddDocMsg(const char *document_name, const char *image_name)
    : coGRDocMsg(ADD_DOCUMENT, document_name)
{
    imageName_ = NULL;

    if (document_name && image_name)
    {
        imageName_ = new char[strlen(image_name) + 1];
        strcpy(imageName_, image_name);
        addToken(image_name);

        is_valid_ = 1;
    }
    else
    {
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRAddDocMsg::coGRAddDocMsg(const char *msg)
    : coGRDocMsg(msg)
{
    string iName = extractFirstToken();

    if (!iName.empty())
    {
        imageName_ = new char[iName.size() + 1];
        strcpy(imageName_, iName.c_str());
    }
    else
    {
        imageName_ = NULL;
        is_valid_ = 0;
    }
}

GRMSGEXPORT coGRAddDocMsg::~coGRAddDocMsg()
{
    if (imageName_)
        delete[] imageName_;
}
