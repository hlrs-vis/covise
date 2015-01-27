/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Map1D
//
//  This class for integer mappings
//
//  Initial version: 2001-12-07 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _MAP_1D_H_
#define _MAP_1D_H_

#include <util/coIntHash.h>
#include <api/coModule.h>
using namespace covise;

class Map1D
{
private:
    int *mapping_;
    int min_;
    int max_;
    int length_;
    enum method
    {
        TRIVIAL,
        HASHING
    } method_;
    // label numbers are mapped into natural numbers
    // using a fast mapping if the set of spanned labels
    // (from the minimum to the maximum, including those
    // that are not used)
    // encompasses less than "TRIVIAL_LIMIT" numbers
    // the problem is that if there are too many unused
    // labels in this set;
    // otherwise, hashing is used, which may be especially
    // convienient if the first method could take up
    // too much memory in an inefficient way. Hashing
    // is slower, but less problems with memory usage
    // may be expected.
    const static int TRIVIAL_LIMIT = 1000000;
    coIntHash<int> labels_;

public:
    // list enthaelt labels
    Map1D &operator=(const Map1D &rhs);
    void setMap(int l = 0, const int *list = 0);
    Map1D(int l = 0, const int *list = 0);
    //   ~Map1D(){ delete [] mapping_;}
    ~Map1D()
    {
    }
    const int &operator[](int i) const;
};
#endif
