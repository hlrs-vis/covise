/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SectionPoint.h"

SectionPoint::SectionPoint()
    : _secPointNumber(0)
    , _description("None")
{
}

SectionPoint::SectionPoint(int secPointNumber, const char *description)
    : _secPointNumber(secPointNumber)
    , _description(description)
{
    assert(secPointNumber > 0);
}

SectionPoint::~SectionPoint()
{
}

SectionPoint::SectionPoint(const SectionPoint &rhs)
    : _secPointNumber(rhs._secPointNumber)
    , _description(rhs._description)
{
}

bool
    SectionPoint::
    operator==(const SectionPoint &rhs) const
{
    return (_secPointNumber == rhs._secPointNumber
            && _description == rhs._description);
}

SectionPoint &
    SectionPoint::
    operator=(const SectionPoint &rhs)
{
    if (this == &rhs)
    {
        return *this;
    }
    _secPointNumber = rhs._secPointNumber;
    _description = rhs._description;
    return *this;
}

string
SectionPoint::str() const
{
    char buf[32];
    sprintf(buf, "%d: ", _secPointNumber);
    string ret(buf);
    ret += _description;
    return ret;
}
