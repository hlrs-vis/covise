/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CUBE_INTERACTOR_H
#define CUBE_INTERACTOR_H

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
//    CubeGeode             xform
//    (covise)                |
//                          scale
//                            |
//                         wirecube
//
#include <OpenVRUI/coTrackerButtonInteraction.h>

using namespace vrui;
using namespace opencover;

class CubeInteractor : public coTrackerButtonInteraction
{
private:
    float size;

    // scenegraph node
    osg::ref_ptr<osg::Group> root;
    osg::PositionAttitudeTransform *DCS;
    osg::Geode *cubeGeode;

    // functions to create the cube
    osg::Geometry *createWireframeUnitCube();
    osg::StateSet *createWireframeCubeMaterial();

    // needed for interaction
    osg::Vec3 initHandPos;

public:
    // create the scene graph for a movable, scalable wireframe cube
    CubeInteractor(coInteraction::InteractionType type, const char *name, coInteraction::InteractionPriority priority);

    // destroy the cube scene graph
    ~CubeInteractor();

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

    // show cube
    void hide();

    // hide cube
    void show();
};
#endif
