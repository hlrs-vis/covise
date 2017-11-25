/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SmokeGeneratorSolutions.h"
#include <numeric>

using namespace osg;

SmokeGeneratorSolutions::SmokeGeneratorSolutions()
    : lengths_(NULL)
    , linepoints_(NULL)
{
}

SmokeGeneratorSolutions::~SmokeGeneratorSolutions()
{
    if (lengths_ != NULL)
        lengths_->unref();
    if (linepoints_ != NULL)
        linepoints_->unref();
}

void
SmokeGeneratorSolutions::clear()
{
    size_ = 0;
    if (lengths_ != NULL)
        lengths_->unref();
    if (linepoints_ != NULL)
        linepoints_->unref();
    lengths_ = NULL;
    linepoints_ = NULL;
}

void
SmokeGeneratorSolutions::set(const vector<vector<covise::coUniState> > &solus)
{
    clear();
    size_ = solus.size();
    lengths_ = new DrawArrayLengths(PrimitiveSet::LINE_STRIP);
    lengths_->ref();

    int line;
    for (line = 0; line < size_; ++line)
    {
        lengths_->push_back(solus[line].size());
    }
    //    int total_length = std::accumulate(lengths_->begin(),lengths_->end(),0);
    //    linepoints_ = (pfVec3*) pfCalloc(total_length, sizeof(pfVec3), pfGetSharedArena());
    //    linepoints_ = new Vec3Array(total_length);
    //    int count_point = 0;
    linepoints_ = new Vec3Array();
    linepoints_->ref();
    for (line = 0; line < size_; ++line)
    {
        Vec3 this_point;
        int point;
        const vector<covise::coUniState> &thisSolu = solus[line];
        for (point = 0; point < lengths_->at(line); ++point)
        {
            this_point[0] = thisSolu[point].point_[0];
            this_point[1] = thisSolu[point].point_[1];
            this_point[2] = thisSolu[point].point_[2];
            linepoints_->push_back(this_point);
            //          ++count_point;
        }
    }
}

DrawArrayLengths *
SmokeGeneratorSolutions::lengths() const
{
    return lengths_.get();
}

Vec3Array *
SmokeGeneratorSolutions::linepoints() const
{
    return linepoints_.get();
}

int
SmokeGeneratorSolutions::size() const
{
    return size_;
}
