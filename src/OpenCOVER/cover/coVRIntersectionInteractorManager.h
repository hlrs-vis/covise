/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_VR_INTERSECTION_INTERACTOR_MANAGER
#define CO_VR_INTERSECTION_INTERACTOR_MANAGER

#include "coVRIntersectionInteractorManager.h"
#include "coVRIntersectionInteractor.h"
#include <list>

namespace opencover
{

class coVRIntersectionInteractor;

class COVEREXPORT coVRIntersectionInteractorManager
{
public:
    coVRIntersectionInteractorManager();
    virtual ~coVRIntersectionInteractorManager();

    void add(coVRIntersectionInteractor *);
    void remove(coVRIntersectionInteractor *);
    void enableCycleThroughInteractors();
    void disableCycleThroughInteractors();
    void cycleThroughInteractors(bool forward = true);

    static coVRIntersectionInteractorManager *the();

    coVRIntersectionInteractor *getCurrentIntersectionInteractor()
    {
        return currentIntersectionInteractor;
    }

private:
    // list of all interactions
    std::list<coVRIntersectionInteractor *> allIntersectionInteractors;
    std::list<coVRIntersectionInteractor *>::iterator allIntersectionInteractorsIt;
    coVRIntersectionInteractor *currentIntersectionInteractor;
    bool wait_;
};
}
#endif
