/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SPLIT_RESULTS_
#define _SPLIT_RESULTS_

#include <api/coSimpleModule.h>
using namespace covise;
#include <limits.h>

class SplitResults : public coSimpleModule
{
public:
    SplitResults();
    ~SplitResults();

protected:
private:
    virtual void copyAttributesToOutObj(coInputPort **, coOutputPort **, int);
    virtual int compute();
    coInputPort *p_indices_;
    coInputPort *p_in_data_;
    coOutputPort *p_out_data_;
};
#endif
