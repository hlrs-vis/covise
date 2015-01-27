/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Location
//
//  Abstraction for type of data attachment: per node, per element, ...
//  It also encapsulates available SectionPoints.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef _LOCATION_H_
#define _LOCATION_H_

#include "SectionPoint.h"
#include <util/coviseCompat.h>
typedef uint64_t uint64;
typedef int64_t int64;
#include "odb_API.h"

class Location
{
public:
    Location(odb_Enum::odb_ResultPositionEnum);
    Location(const Location &rhs);
    virtual ~Location();
    void AccumulateSectionPoint(const odb_SectionPoint &secP);
    void AccumulateSectionPoint();
    bool operator==(const Location &rhs) const;
    Location &operator=(const Location &rhs);
    bool NoDummy() const;
    static string ResultPositionEnumToString(odb_Enum::odb_ResultPositionEnum);
    string str() const;
    void GetLists(vector<string> &sectionPoints) const;
    void ReproduceVisualisation(OutputChoice &choice, const string &sectionPointOld) const;

protected:
private:
    odb_Enum::odb_ResultPositionEnum _position; // only this member is used
    // for equality
    vector<SectionPoint> _sectionPoints;
};
#endif
