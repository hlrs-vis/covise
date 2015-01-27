/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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

#include "InvTimePart.h"

TimePart::TimePart(int time, int part)
{
    set(time, part);
}

void TimePart::set(int t, int p)
{
    tp[0] = t;
    tp[1] = p;
}

int TimePart::getTime()
{
    return tp[0];
}

int TimePart::getPart()
{
    return tp[1];
}

int TimePart::equal(const TimePart &tiPa) const
{
    return (tp[0] == tiPa[0] && tp[1] == tiPa[1]);
}

int &TimePart::operator[](int i)
{
    assert((i >= 0) && (i < 2));

    return tp[i];
}

const int &TimePart::operator[](int i) const
{
    assert((i >= 0) && (i < 2));

    return tp[i];
}

int TimePart::operator==(const TimePart &tiPa) const
{
    return equal(tiPa);
}

TimePart &TimePart::operator=(const TimePart &tiPa)
{
    tp[0] = tiPa[0];
    tp[1] = tiPa[1];

    return *this;
}

/*
zugehoerige hash-Funktionen:
unsigned long hash1(const TimePart &key ) const
{
  return ( (key[0]+1) % (key[1]+1));
}

unsigned long hash2(const TimePart &key ) const
{
  return( key[0]+key[1]+1 - key[0]%(key[0]*key[1]+1) );
}
*/
