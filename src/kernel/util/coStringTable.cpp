/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS coStringTable
//
// a table for pairs of string and numbers
// where each number is assigned to a string
// and vice versa
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Initial version: 2001-17-12 cs
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes

#include "coExport.h"
#include "coStringTable.h"

using namespace covise;

coStringTable::coStringTable()
{
    cs_ = new std::map<const char *, int, ltstr>;
    ci_ = new std::map<int, const char *, ltint>;
}

void coStringTable::insert(int number, const char *str)
{
    (*cs_)[str] = number;
    (*ci_)[number] = str;
}

bool coStringTable::isElement(int x)
{
    return !(ci_->find(x) == ci_->end());
}

bool coStringTable::isElement(const char *str)
{
    return !(cs_->find(str) == cs_->end());
}

const char *coStringTable::operator[](int x)
{
    return (*ci_)[x];
}

int coStringTable::operator[](const char *str)
{
    return (*cs_)[str];
}
