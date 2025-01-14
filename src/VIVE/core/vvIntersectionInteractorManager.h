/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#pragma once

#include "vvIntersectionInteractor.h"
#include <list>

namespace vive
{

class vvIntersectionInteractor;

class VVCORE_EXPORT vvIntersectionInteractorManager
{
public:
    vvIntersectionInteractorManager();
    virtual ~vvIntersectionInteractorManager();

    void add(vvIntersectionInteractor *);
    void remove(vvIntersectionInteractor *);
    void enableCycleThroughInteractors();
    void disableCycleThroughInteractors();
    void cycleThroughInteractors(bool forward = true);

    static vvIntersectionInteractorManager *the();

    vvIntersectionInteractor *getCurrentIntersectionInteractor()
    {
        return currentIntersectionInteractor;
    }

private:
    // list of all interactions
    std::list<vvIntersectionInteractor *> allIntersectionInteractors;
    std::list<vvIntersectionInteractor *>::iterator allIntersectionInteractorsIt;
    vvIntersectionInteractor *currentIntersectionInteractor;
    bool wait_;
};
}
