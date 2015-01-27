/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __MULTIGRIDMEMBER_DEFINED
#define __MULTIGRIDMEMBER_DEFINED
#include <util/coviseCompat.h>
#include "DxObject.h"
#include <map>

class MultiGridMember
{
    char *name_;
    DxObject *object_;

public:
    MultiGridMember()
    {
        name_ = NULL;
        object_ = NULL;
    }
    MultiGridMember(const char *name, DxObject *object);
    MultiGridMember(int number, DxObject *object);
    ~MultiGridMember();
    const char *getName()
    {
        return name_;
    }
    DxObject *getObject()
    {
        return object_;
    }
};

//typedef list <MultiGridMember *> MemberList;
class MemberList
{
private:
    MultiGridMember **list_;
    int size_;

public:
    MemberList()
    {
        list_ = new MultiGridMember *[3000];
        size_ = 0;
    }
    ~MemberList()
    {
        delete[] list_;
    }
    void push_back(MultiGridMember *m)
    {
        list_[size_++] = m;
    }
    int size()
    {
        return size_;
    }
    MultiGridMember *get(int i)
    {
        return list_[i];
    }
};
#endif
