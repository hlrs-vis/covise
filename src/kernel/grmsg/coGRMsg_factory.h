/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef COGRMSG_FACTORY_H
#define COGRMSG_FACTORY_H

#include <memory>
#include <util/coExport.h>
namespace grmsg
{
class coGRMsg;
GRMSGEXPORT std::unique_ptr<coGRMsg> create(const char *msg);
}

#endif // COGRMSG_FACTORY_H