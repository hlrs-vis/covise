/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __MULTIGRIDMEMBER_DEFINED
#define __MULTIGRIDMEMBER_DEFINED
#include <util/coviseCompat.h>
#include <map>

class DxObject;

class MultiGridMember
{
    std::string name_;
    DxObject *object_;

public:
    MultiGridMember()
        : object_(NULL)
    {
    }
    MultiGridMember(const char *name, DxObject *object);
    MultiGridMember(int number, DxObject *object);
    ~MultiGridMember();
    std::string getName()
    {
        return name_;
    }
    DxObject *getObject()
    {
        return object_;
    }
};

typedef std::vector<MultiGridMember *> MemberList;
#endif
