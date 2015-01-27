/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BBOXES_H_
#define _BBOXES_H_

#include <util/coviseCompat.h>
#include <do/coDoSet.h>
#include <do/coDoPolygons.h>
#include <do/coDoData.h>

using namespace covise;

class BBoxes
{
public:
    typedef float BBox[4];
    enum
    {
        MIN_X = 0,
        MAX_X = 1,
        MIN_Y = 2,
        MAX_Y = 3
    };
    void clean();
    friend ostream &operator<<(ostream &outfile, const BBoxes &tree);
    void prepare(int no_times);
    const float *getBBox(int time) const;
    void FillBBox(const coDoSet *inObj, const coDoSet *inShift, int time);
    void FillBBox(const coDoPolygons *inObj, const coDoVec3 *inShift, int time);
    int getNoTimes() const
    {
        return no_times_;
    }
    BBoxes();
    ~BBoxes();

private:
    int no_times_;
    BBox *bboxes_;
};
#endif
