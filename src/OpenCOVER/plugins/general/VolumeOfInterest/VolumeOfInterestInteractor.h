/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VOLUME_OF_INTEREST_INTERACTOR_H
#define _VOLUME_OF_INTEREST_INTERACTOR_H

#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/StateSet>

#include <OpenVRUI/coTrackerButtonInteraction.h>

class VolumeOfInterestInteractor : public vrui::coTrackerButtonInteraction
{
public:
    VolumeOfInterestInteractor(vrui::coInteraction::InteractionType type, const char *name, vrui::coInteraction::InteractionPriority priority);

    ~VolumeOfInterestInteractor();

    // stop the interaction
    void stopInteraction();
    void registerInteractionFinishedCallback(void (*interactionFinished)());
    void unregisterInteractionFinishedCallback();

private:
    void (*m_interactionFinished)();
    template <class T>
    void swap(T &m, T &n)
    {
        T z = m;
        m = n;
        n = z;
    }
};
#endif
