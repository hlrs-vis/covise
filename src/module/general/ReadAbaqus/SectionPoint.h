/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS SectionPoint
//
//  This class describes a section point. See ABAQUS docu for
//  beams, shells and other exotic elements...
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _SECTION_POINT_H_
#define _SECTION_POINT_H_

#include <util/coviseCompat.h>

struct OutputChoice
{
    int _fieldLabel;
    int _component;
    int _invariant;
    int _location;
    int _secPoint;
};

class SectionPoint
{
public:
    SectionPoint(int secPointNumber, const char *description);
    SectionPoint(); // dummy section point
    virtual ~SectionPoint();
    SectionPoint(const SectionPoint &rhs);
    bool operator==(const SectionPoint &rhs) const;
    SectionPoint &operator=(const SectionPoint &rhs);
    string str() const;

protected:
private:
    int _secPointNumber;
    string _description;
};
#endif
