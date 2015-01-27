/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _EDGE_COLLAPSE_H_
#define _EDGE_COLLAPSE_H_
#include "EdgeCollapseBasis.h"

class EdgeCollapse : public EdgeCollapseBasis
{
public:
    EdgeCollapse(const vector<float> &x_c,
                 const vector<float> &y_c,
                 const vector<float> &z_c,
                 const vector<int> &conn_list,
                 const vector<float> &data_c,
                 const vector<float> &normals_c,
                 VertexContainer::TYPE vertCType,
                 TriangleContainer::TYPE triCType,
                 EdgeContainer::TYPE edgeCType);
    virtual int EdgeContraction(int num_max);
    virtual ~EdgeCollapse();
    //friend ostream& operator<<(ostream&,const EdgeCollapse&);
protected:
private:
};
#endif
