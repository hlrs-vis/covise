/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COMPORESS_CONNECTIVITY_H_
#define _COMPORESS_CONNECTIVITY_H_
#include <api/coSimpleModule.h>
using namespace covise;

class CompressConnectivity : public coSimpleModule
{
public:
    CompressConnectivity();
    ~CompressConnectivity();

protected:
private:
    virtual int compute();
    coInputPort *p_grid_in_;
    coOutputPort *p_grid_out_, *p_data_out_;
};
#endif
