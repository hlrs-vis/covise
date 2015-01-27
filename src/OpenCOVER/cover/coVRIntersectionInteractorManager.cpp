/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRIntersectionInteractorManager.h"
#include <config/CoviseConfig.h>
using namespace opencover;
using namespace covise;

coVRIntersectionInteractorManager::coVRIntersectionInteractorManager()
{
    allIntersectionInteractorsIt = allIntersectionInteractors.begin();
    currentIntersectionInteractor = NULL;

    if (!coCoviseConfig::isOn("COVER.Input.VisensoJoystick", false))
        wait_ = true; // wait heisst am ende der Interktorliste einmal warten, damit man bei nicht wii den zeigestrahl auch mal wieder aus den interaktoren raus bekommt
    else
        wait_ = false;
}

coVRIntersectionInteractorManager::~coVRIntersectionInteractorManager()
{
}

coVRIntersectionInteractorManager *coVRIntersectionInteractorManager::the()
{

    static coVRIntersectionInteractorManager *im = NULL;
    if (!im)
        im = new coVRIntersectionInteractorManager();
    return im;
}

void coVRIntersectionInteractorManager::enableCycleThroughInteractors()
{
    allIntersectionInteractorsIt = allIntersectionInteractors.begin();
    if (allIntersectionInteractors.begin() != allIntersectionInteractors.end())
    {
        currentIntersectionInteractor = (*allIntersectionInteractorsIt);
    }
    else
    {
        currentIntersectionInteractor = NULL;
    }
}

void coVRIntersectionInteractorManager::disableCycleThroughInteractors()
{
    currentIntersectionInteractor = NULL;
    //fprintf(stderr,"coVRIntersectionInteractorManager::disableCycleInteractors currentIntersectionInteractor = NULL\n");
}

void coVRIntersectionInteractorManager::cycleThroughInteractors(bool forward)
{

    // gibt es einen Interaction in der Liste
    if (allIntersectionInteractors.begin() != allIntersectionInteractors.end())
    {
        if (forward)
            allIntersectionInteractorsIt++;
        else
        {
            if (allIntersectionInteractorsIt == allIntersectionInteractors.begin())
                allIntersectionInteractorsIt = --(allIntersectionInteractors.end());
            else
                allIntersectionInteractorsIt--;
        }
        if (allIntersectionInteractorsIt == allIntersectionInteractors.end())
        {
            if (!wait_)
            {
                allIntersectionInteractorsIt = allIntersectionInteractors.begin();
                if (!coCoviseConfig::isOn("COVER.Input.VisensoJoystick", false))
                    wait_ = true;
                else
                    wait_ = false;
            }
            else
            {
                wait_ = false;
            }
        }
        else
        {
            if (!coCoviseConfig::isOn("COVER.Input.VisensoJoystick", false))
                wait_ = true;
            else
                wait_ = false;
        }
        if (allIntersectionInteractorsIt != allIntersectionInteractors.end())
        {
            currentIntersectionInteractor = (*allIntersectionInteractorsIt);
            //fprintf(stderr,"coVRIntersectionInteractorManager::cycleThroughInteractors currentIntersectionInteractor = %s\n", currentIntersectionInteractor->getInteractorName());
        }
        else
        {
            currentIntersectionInteractor = NULL;
        }
    }
    else
    {
        currentIntersectionInteractor = NULL;
        //fprintf(stderr,"coVRIntersectionInteractorManager::cycleThroughInteractors currentIntersectionInteractor because begin=end= NULL\n");
    }
}
void coVRIntersectionInteractorManager::add(coVRIntersectionInteractor *interactor)
{

    //fprintf(stderr,"coVRIntersectionInteractorManager::add interaction=%s\n", interactor->getInteractorName());
    allIntersectionInteractors.push_back(interactor);
    allIntersectionInteractorsIt = allIntersectionInteractors.begin();
    if (allIntersectionInteractors.begin() != allIntersectionInteractors.end())
    {
        currentIntersectionInteractor = (*allIntersectionInteractorsIt);
        //fprintf(stderr,"coVRIntersectionInteractorManager::add currentIntersectionInteractor = %s\n", currentIntersectionInteractor->getInteractorName());
    }
    else
    {
        currentIntersectionInteractor = NULL;
        //fprintf(stderr,"coVRIntersectionInteractorManager::add currentIntersectionInteractor = NULL - this is IMPOSSIBLE\n");
    }
}

void coVRIntersectionInteractorManager::remove(coVRIntersectionInteractor *interactor)
{
    //fprintf(stderr,"coVRIntersectionInteractorManager::remove interaction=%s\n",  interactor->getInteractorName());
    allIntersectionInteractors.remove(interactor);
    allIntersectionInteractorsIt = allIntersectionInteractors.begin();
    if (allIntersectionInteractors.begin() != allIntersectionInteractors.end())
    {
        currentIntersectionInteractor = (*allIntersectionInteractorsIt);
        //fprintf(stderr,"coVRIntersectionInteractorManager::remove currentIntersectionInteractor = %s\n", currentIntersectionInteractor->getInteractorName());
    }
    else
    {
        currentIntersectionInteractor = NULL;
        //fprintf(stderr,"coVRIntersectionInteractorManager::remove currentIntersectionInteractor = NULL\n");
    }
}
