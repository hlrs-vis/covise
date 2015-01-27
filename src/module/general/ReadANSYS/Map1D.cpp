/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Map1D.h"
#include <limits.h>

void Map1D::setMap(int l, const int *list)
{
    delete[] mapping_;
    mapping_ = 0;
    min_ = INT_MAX;
    max_ = INT_MIN;
    if (l <= 0)
        return;
    int i;
    for (i = 0; i < l; ++i)
    {
        if (list[i] < min_)
            min_ = list[i];
        if (list[i] > max_)
            max_ = list[i];
    }
    length_ = l;
    if (max_ - min_ < TRIVIAL_LIMIT)
    {
        method_ = TRIVIAL;
        mapping_ = new int[max_ - min_ + 1];
        for (i = 0; i < l; ++i)
        {
            mapping_[list[i] - min_] = i;
        }
    }
    else
    {
        int i;
        method_ = HASHING;
        labels_.clear();
        for (i = 0; i < l; ++i)
        {
            labels_.insert(list[i], i);
        }
    }
}

Map1D::Map1D(int l, const int *list)
    : mapping_(0)
    , labels_(INT_MAX)
{
    setMap(l, list);
}

const int &Map1D::operator[](int i) const
{
    const int *ret;
    switch (method_)
    {
    case TRIVIAL:
        if (min_ > max_)
        {
            Covise::sendWarning("Unexpected error: Map1D[] may not be used");
            return min_;
        }
        if (i < min_)
        {
            cerr << "Unexpected error: Map1D[] with forbidden index: " << i << " < " << min_ << endl;
            i = min_;
            Covise::sendWarning("Unexpected error: Map1D[] with forbidden index");
        }
        if (i > max_)
        {
            Covise::sendWarning("Unexpected error: Map1D[] with forbidden index");
            cerr << "Unexpected error: Map1D[] with forbidden index: " << i << " > " << max_ << endl;
            i = max_;
        }
        ret = &mapping_[i - min_];
        break;
    case HASHING:
        ret = &labels_.find(i);
        break;
    /*
               ret=(int *)bsearch(&i,mapping_,length_,2*sizeof(int),compar);
               ret += 1;
               break;
         */
    default:
        Covise::sendWarning("Unexpected value for method_ in Map1D::operator[]. Aborting...");
        ret = NULL;
        break;
    }
    return *ret;
}
