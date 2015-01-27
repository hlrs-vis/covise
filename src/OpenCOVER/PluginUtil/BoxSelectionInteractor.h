/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BOX_SELECTION_INTERACTOR_H
#define _BOX_SELECTION_INTERACTOR_H

#include <util/coExport.h>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/StateSet>

#include "BoxSelection.h"

#include <OpenVRUI/coTrackerButtonInteraction.h>

namespace opencover
{

class PLUGIN_UTILEXPORT BoxSelectionInteractor : public vrui::coTrackerButtonInteraction
{
public:
    // create the scene graph for a movable, scalable wireframe cube
    BoxSelectionInteractor(vrui::coInteraction::InteractionType type, const char *name, vrui::coInteraction::InteractionPriority priority);

    // destroy the cube scene graph
    ~BoxSelectionInteractor();

    // start the interaction
    void startInteraction();

    // ongoing interaction
    void doInteraction();
    void registerInteractionRunningCallback(void (*interactionRunning)());
    void unregisterInteractionRunningCallback();

    // stop the interaction
    void stopInteraction();
    void registerInteractionFinishedCallback(void (*interactionFinished)());
    void unregisterInteractionFinishedCallback();

    void getBox(float &minX, float &minY, float &minZ, float &maxX, float &maxY, float &maxZ);

private:
    osg::Vec3 m_min, m_max;
    osg::Vec3 m_worldCoordMin, m_worldCoordMax;
    osg::Geode *m_cubeGeode;
    osg::ref_ptr<osg::Group> m_root;
    bool m_animationWasRunning;
    void (*m_interactionFinished)();
    void (*m_interactionRunning)();

    osg::Geometry *createWireframeBox(osg::Vec3 min, osg::Vec3 max);
    osg::StateSet *createBoxMaterial();

    void updateBox(osg::Vec3 min, osg::Vec3 max);
};
}
#endif
