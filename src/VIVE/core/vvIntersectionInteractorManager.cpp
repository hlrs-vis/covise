/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvIntersectionInteractorManager.h"
#include <config/CoviseConfig.h>
using namespace vive;
using namespace covise;

vvIntersectionInteractorManager::vvIntersectionInteractorManager()
{
    allIntersectionInteractorsIt = allIntersectionInteractors.begin();
    currentIntersectionInteractor = NULL;

    if (!coCoviseConfig::isOn("VIVE.Input.VisensoJoystick", false))
        wait_ = true; // wait heisst am ende der Interktorliste einmal warten, damit man bei nicht wii den zeigestrahl auch mal wieder aus den interaktoren raus bekommt
    else
        wait_ = false;
}

vvIntersectionInteractorManager::~vvIntersectionInteractorManager()
{
}

vvIntersectionInteractorManager *vvIntersectionInteractorManager::the()
{

    static vvIntersectionInteractorManager *im = NULL;
    if (!im)
        im = new vvIntersectionInteractorManager();
    return im;
}

void vvIntersectionInteractorManager::enableCycleThroughInteractors()
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

void vvIntersectionInteractorManager::disableCycleThroughInteractors()
{
    currentIntersectionInteractor = NULL;
    //fprintf(stderr,"vvIntersectionInteractorManager::disableCycleInteractors currentIntersectionInteractor = NULL\n");
}

void vvIntersectionInteractorManager::cycleThroughInteractors(bool forward)
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
                if (!coCoviseConfig::isOn("VIVE.Input.VisensoJoystick", false))
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
            if (!coCoviseConfig::isOn("VIVE.Input.VisensoJoystick", false))
                wait_ = true;
            else
                wait_ = false;
        }
        if (allIntersectionInteractorsIt != allIntersectionInteractors.end())
        {
            currentIntersectionInteractor = (*allIntersectionInteractorsIt);
            //fprintf(stderr,"vvIntersectionInteractorManager::cycleThroughInteractors currentIntersectionInteractor = %s\n", currentIntersectionInteractor->getInteractorName());
        }
        else
        {
            currentIntersectionInteractor = NULL;
        }
    }
    else
    {
        currentIntersectionInteractor = NULL;
        //fprintf(stderr,"vvIntersectionInteractorManager::cycleThroughInteractors currentIntersectionInteractor because begin=end= NULL\n");
    }
}
void vvIntersectionInteractorManager::add(vvIntersectionInteractor *interactor)
{

    //fprintf(stderr,"vvIntersectionInteractorManager::add interaction=%s\n", interactor->getInteractorName());
    allIntersectionInteractors.push_back(interactor);
    allIntersectionInteractorsIt = allIntersectionInteractors.begin();
    if (allIntersectionInteractors.begin() != allIntersectionInteractors.end())
    {
        currentIntersectionInteractor = (*allIntersectionInteractorsIt);
        //fprintf(stderr,"vvIntersectionInteractorManager::add currentIntersectionInteractor = %s\n", currentIntersectionInteractor->getInteractorName());
    }
    else
    {
        currentIntersectionInteractor = NULL;
        //fprintf(stderr,"vvIntersectionInteractorManager::add currentIntersectionInteractor = NULL - this is IMPOSSIBLE\n");
    }
}

void vvIntersectionInteractorManager::remove(vvIntersectionInteractor *interactor)
{
    //fprintf(stderr,"vvIntersectionInteractorManager::remove interaction=%s\n",  interactor->getInteractorName());
    allIntersectionInteractors.remove(interactor);
    allIntersectionInteractorsIt = allIntersectionInteractors.begin();
    if (allIntersectionInteractors.begin() != allIntersectionInteractors.end())
    {
        currentIntersectionInteractor = (*allIntersectionInteractorsIt);
        //fprintf(stderr,"vvIntersectionInteractorManager::remove currentIntersectionInteractor = %s\n", currentIntersectionInteractor->getInteractorName());
    }
    else
    {
        currentIntersectionInteractor = NULL;
        //fprintf(stderr,"vvIntersectionInteractorManager::remove currentIntersectionInteractor = NULL\n");
    }
}
