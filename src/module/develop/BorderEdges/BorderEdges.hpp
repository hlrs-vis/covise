/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef BORDEREDGES_HPP_THUDEC22153212CET2005
#define BORDEREDGES_HPP_THUDEC22153212CET2005

/*

   MODULE BorderEdges

   The output is the edges on the border of the input-grid.

*/

#include <api/coSimpleModule.h>

class BorderEdges : public coSimpleModule
{
    coInPort *p_in_geometry_;
    coOutPort *p_out_geometry_;

    virtual int compute();

public:
    BorderEdges();
};

// local variables:
// mode: c++
// end:

#endif
