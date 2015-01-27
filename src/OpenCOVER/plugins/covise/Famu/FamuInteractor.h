/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Famu_INTERACTOR_H
#define Famu_INTERACTOR_H

#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/StateSet>

//             objectsRoot
//                  |
//           objectsXformDCS
//                  |
//           objectsScaleDCS
//                  |
//              objectsRoot
//       |                    |
//    FamuGeode             xform
//    (covise)                |
//                          scale
//                            |
//                         wireFamu
//
#include <OpenVRUI/coTrackerButtonInteraction.h>

class FamuInteractor : public vrui::coTrackerButtonInteraction
{
private:
    float size;

    // scenegraph node
    osg::ref_ptr<osg::Group> root;
    osg::PositionAttitudeTransform *DCS;
    osg::Geode *FamuGeode;

    // functions to create the Famu
    osg::Geometry *createWireframeUnitFamu();
    osg::StateSet *createWireframeFamuMaterial();

    // needed for interaction
    osg::Vec3 initHandPos;

public:
    // create the scene graph for a movable, scalable wireframe Famu
    FamuInteractor(coInteraction::InteractionType type, const char *name, coInteraction::InteractionPriority priority);

    // destroy the Famu scene graph
    ~FamuInteractor();

    // ongoing interaction
    void doInteraction();

    // stop the interaction
    void stopInteraction();

    // start the interaction (grab pointer, set selected hl, store dcsmat)
    void startInteraction();

    // set the size in object coordinates
    void setSize(float s);

    // set the center in object coordinates
    void setCenter(osg::Vec3);

    // return the center
    osg::Vec3 getCenter();

    // show Famu
    void hide();

    // hide Famu
    void show();
};
#endif
