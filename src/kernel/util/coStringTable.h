/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_STRING_TABLE
#define _CO_STRING_TABLE
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

#include <cstring>
#include <map>

namespace covise
{

struct ltint
{
    bool operator()(int s1, int s2) const
    {
        return (s1 < s2);
    }
};

struct ltstr
{
    bool operator()(const char *s1, const char *s2) const
    {
        return strcmp(s1, s2) < 0;
    }
};
class UTILEXPORT coStringTable
/**
   a coStringTable object may have a string or
   an integer as index. You can either search for
   the integer and get the string or search for
   the string and get the integer.
   Examples:
   coStringTable sample;
   sample.insert(177, "foo");

   // gives foo
   cout << sample[177] << endl;
// gives 177
cout << sample["foo"] << endl;
*/
{
private:
    std::map<const char *, int, ltstr> *cs_;
    std::map<int, const char *, ltint> *ci_;

public:
    /// default Constructor
    coStringTable();
    /** Insert a pair of integer/string
         @param number the number belonging to the string
         @param str the string to be inserted
       */
    void insert(int number, const char *str);
    /** check whether an integer occurs in the table
         @param x the integer to be found
       */
    bool isElement(int x);
    /** check whether a string occurs in the table
       */
    bool isElement(const char *str);
    /** get the string belonging to the integer
       * @param x the integer in question
       * @return the string belonging to the integer
       */
    const char *operator[](int x);

    /** get the integer belonging to the string
       * @param str
       */
    int operator[](const char *str);
};
}
#endif
