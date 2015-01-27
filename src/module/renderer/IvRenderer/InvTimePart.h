/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_TIME_PART_
#define _INV_TIME_PART_

// **************************************************************************
//
// Description    : utility class
//
// Class(es)      : TimePart
//
// Author  : Reiner Beller
//
// History :
//
// **************************************************************************

#include <assert.h>

class TimePart
{

public:
    TimePart(){};
    TimePart(int time, int part);

    // sets and gets
    void set(int t, int p);
    int getTime();
    int getPart();

    // other functions
    int equal(const TimePart &tiPa) const;

    // operators
    int &operator[](int i);
    const int &operator[](int i) const;
    int operator==(const TimePart &tiPa) const;
    TimePart &operator=(const TimePart &tiPa);

private:
    int tp[2];
};
#endif // _INV_TIME_PART_
