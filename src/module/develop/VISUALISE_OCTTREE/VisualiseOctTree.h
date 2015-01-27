/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VISUALISE_OCTTREE_H_
#define _VISUALISE_OCTTREE_H_

#include <api/coSimpleModule.h>
using namespace covise;
#include <covise/covise_octtree.h>
#include <util/coIA.h>

class Application : public coSimpleModule
{
private:
    virtual int compute();

    coInputPort *p_oct_;
    coOutputPort *p_lines_;

public:
    Application();
    virtual ~Application(void)
    {
    }
};
#endif
