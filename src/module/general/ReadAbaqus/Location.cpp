/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Location.h"
#include <algorithm>
using std::find;

Location::Location(odb_Enum::odb_ResultPositionEnum position)
    : _position(position)
{
}

Location::Location(const Location &rhs)
    : _position(rhs._position)
    , _sectionPoints(rhs._sectionPoints)
{
}

Location::~Location()
{
}

bool
    Location::
    operator==(const Location &rhs) const
{
    return (_position == rhs._position);
}

Location &
    Location::
    operator=(const Location &rhs)
{
    if (this == &rhs)
    {
        return *this;
    }

    _position = rhs._position;
    _sectionPoints = rhs._sectionPoints;
    return *this;
}

void
Location::AccumulateSectionPoint(const odb_SectionPoint &secP)
{
    SectionPoint testSectionPoint(secP.number(), secP.description().CStr());
    int i;
    for (i = 0; i < _sectionPoints.size(); ++i)
    {
        if (testSectionPoint == _sectionPoints[i])
        {
            break;
        }
    }
    if (i == _sectionPoints.size())
    {
        _sectionPoints.push_back(testSectionPoint);
    }
}

void
Location::AccumulateSectionPoint()
{
    SectionPoint testSectionPoint;
    vector<SectionPoint>::iterator it_sp = find(_sectionPoints.begin(), _sectionPoints.end(), testSectionPoint);
    if (it_sp == _sectionPoints.end())
    {
        vector<SectionPoint> newVec;
        newVec.push_back(testSectionPoint);
        int i;
        for (i = 0; i < _sectionPoints.size(); ++i)
        {
            newVec.push_back(_sectionPoints[i]);
        }
        _sectionPoints = newVec;
    }
}

bool
Location::NoDummy() const
{
    SectionPoint testSectionPoint;
    int i;
    for (i = 0; i < _sectionPoints.size(); ++i)
    {
        if (testSectionPoint == _sectionPoints[i])
        {
            break;
        }
    }
    if (i == _sectionPoints.size())
    {
        return true;
    }
    return false;
}

string
Location::ResultPositionEnumToString(odb_Enum::odb_ResultPositionEnum location)
{
    switch (location)
    {
    case odb_Enum::NODAL:
        return "NODAL";
    case odb_Enum::ELEMENT_NODAL:
        return "ELEMENT_NODAL";
    case odb_Enum::INTEGRATION_POINT:
        return "INTEGRATION_POINT";
    case odb_Enum::CENTROID:
        return "CENTROID";
    case odb_Enum::ELEMENT_FACE:
        return "ELEMENT_FACE";
    case odb_Enum::WHOLE_ELEMENT:
        return "WHOLE_ELEMENT";
    default:
        return "UNDEFINED_POSITION";
    }
    return "UNDEFINED_POSITION";
}

string
Location::str() const
{
    return ResultPositionEnumToString(_position);
}

void
Location::GetLists(vector<string> &sectionPoints) const
{
    int sp;
    for (sp = 0; sp < _sectionPoints.size(); ++sp)
    {
        sectionPoints.push_back(_sectionPoints[sp].str());
    }
}

void
Location::ReproduceVisualisation(OutputChoice &choice,
                                 const string &sectionPointOld) const
{
    choice._secPoint = 0;
    int sp;
    for (sp = 0; sp < _sectionPoints.size(); ++sp)
    {
        if (sectionPointOld == _sectionPoints[sp].str())
        {
            break;
        }
    }
    if (sp == _sectionPoints.size())
    {
        return;
    }
    choice._secPoint = sp;
}
