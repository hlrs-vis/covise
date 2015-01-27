/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVR3DTransRotInteractor.h"
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/MatrixTransform>
#include <cover/coVRNavigationManager.h>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coVRPluginSupport.h>

using namespace opencover;

coVR3DTransRotInteractor::coVR3DTransRotInteractor(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium)
    : coVRIntersectionInteractor(s, type, iconName, interactorName, priority)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DTransRotInteractor(%s)\n", interactorName);
    }

    osg::Matrix sm;

    _interMat_o = m;
    ////interMat_o.print(0, 1,"interMat_o :", stderr);

    moveTransform->setMatrix(_interMat_o);

    createGeometry();
}

coVR3DTransRotInteractor::~coVR3DTransRotInteractor()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete ~coVR3DTransRotInteractor\n");
}

void
coVR3DTransRotInteractor::createGeometry()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "\ncoVR3DTransRotInteractor::createGeometry\n");

    osg::ShapeDrawable *xlConeDrawable, *xrConeDrawable, *ylConeDrawable, *yrConeDrawable, *zlConeDrawable, *zrConeDrawable, *sphereDrawable;
    //osg::Geometry *xaxisDrawable, *yaxisDrawable, *zaxisDrawable;

    osg::Vec3 origin(0, 0, 0), px(1, 0, 0), py(0, 1, 0), pz(0, 0, 1);
    osg::Vec4 red(1, 0, 0, 1), green(0, 1, 0, 1), blue(0, 0, 1, 1), color(0.5, 0.5, 0.5, 1);

    osg::Sphere *mySphere = new osg::Sphere(origin, 0.5);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereGeode = new osg::Geode();

    // old code
    //xaxisDrawable = createLine(origin, px, red);
    //yaxisDrawable = createLine(origin, py, green);
    //zaxisDrawable = createLine(origin, pz, blue);

    osg::Cone *myCone = new osg::Cone(origin, 0.25, 1.0);
    xlConeDrawable = new osg::ShapeDrawable(myCone, hint);
    xrConeDrawable = new osg::ShapeDrawable(myCone, hint);
    ylConeDrawable = new osg::ShapeDrawable(myCone, hint);
    yrConeDrawable = new osg::ShapeDrawable(myCone, hint);
    zlConeDrawable = new osg::ShapeDrawable(myCone, hint);
    zrConeDrawable = new osg::ShapeDrawable(myCone, hint);
    osg::Matrix m, m2;
    m.makeRotate(M_PI_2, 0.0, 1.0, 0.0);
    m2.makeTranslate(0.9, 0.0, 0.0);
    xrTransform = new osg::MatrixTransform();
    xrTransform->setMatrix(m * m2);
    m.makeRotate(-M_PI_2, 0.0, 1.0, 0.0);
    m2.makeTranslate(-0.9, 0.0, 0.0);
    xlTransform = new osg::MatrixTransform();
    xlTransform->setMatrix(m * m2);
    m.makeRotate(M_PI_2, 1.0, 0.0, 0.0);
    m2.makeTranslate(0.0, -0.9, 0.0);
    yrTransform = new osg::MatrixTransform();
    yrTransform->setMatrix(m * m2);
    m.makeRotate(-M_PI_2, 1.0, 0.0, 0.0);
    m2.makeTranslate(0.0, 0.9, 0.0);
    ylTransform = new osg::MatrixTransform();
    ylTransform->setMatrix(m * m2);
    m.makeRotate(0.0, 0.0, 1.0, 0.0);
    m2.makeTranslate(0.0, 0.0, 0.9);
    zrTransform = new osg::MatrixTransform();
    zrTransform->setMatrix(m * m2);
    m.makeRotate(M_PI, 0.0, 1.0, 0.0);
    m2.makeTranslate(0.0, 0.0, -0.9);
    zlTransform = new osg::MatrixTransform();
    zlTransform->setMatrix(m * m2);

    red.set(0.5, 0.2, 0.2, 1.0);
    green.set(0.2, 0.5, 0.2, 1.0);
    blue.set(0.2, 0.2, 0.5, 1.0);

    xlConeDrawable->setColor(red);
    xrConeDrawable->setColor(red);
    ylConeDrawable->setColor(green);
    yrConeDrawable->setColor(green);
    zlConeDrawable->setColor(blue);
    zrConeDrawable->setColor(blue);
    sphereDrawable->setColor(color);

    osg::Geode *tmpGeode;
    tmpGeode = new osg::Geode();
    tmpGeode->addDrawable(xrConeDrawable);
    xrTransform->addChild(tmpGeode);
    tmpGeode = new osg::Geode();
    tmpGeode->addDrawable(xlConeDrawable);
    xlTransform->addChild(tmpGeode);
    tmpGeode = new osg::Geode();
    tmpGeode->addDrawable(yrConeDrawable);
    yrTransform->addChild(tmpGeode);
    tmpGeode = new osg::Geode();
    tmpGeode->addDrawable(ylConeDrawable);
    ylTransform->addChild(tmpGeode);
    tmpGeode = new osg::Geode();
    tmpGeode->addDrawable(zrConeDrawable);
    zrTransform->addChild(tmpGeode);
    tmpGeode = new osg::Geode();
    tmpGeode->addDrawable(zlConeDrawable);
    zlTransform->addChild(tmpGeode);
    axisTransform = new osg::MatrixTransform();
    axisTransform->addChild(xrTransform.get());
    axisTransform->addChild(xlTransform.get());
    axisTransform->addChild(yrTransform.get());
    axisTransform->addChild(ylTransform.get());
    axisTransform->addChild(zrTransform.get());
    axisTransform->addChild(zlTransform.get());

    sphereGeode->addDrawable(sphereDrawable);
    axisTransform->addChild(sphereGeode.get());

    axisTransform->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
    geometryNode = axisTransform.get();
    scaleTransform->addChild(geometryNode.get());

    //_interPos = scaleTransform.getBound().center();
    //fprintf("coVR3DTransRotInteractor _interPos (%f %f %f) ", _interPos.x(), _interPos.y(), _interPos.z());///
}

osg::Geometry *
coVR3DTransRotInteractor::createLine(osg::Vec3 pos1, osg::Vec3 pos2, osg::Vec4 c)
{
    osg::Vec4Array *color;
    osg::Vec3Array *coord = new osg::Vec3Array(2);
    (*coord)[0].set(pos1);
    (*coord)[1].set(pos2);
    color = new osg::Vec4Array(1);
    (*color)[0].set(c[0], c[1], c[2], c[3]);
    osg::Geometry *geometry = new osg::Geometry();
    geometry->setVertexArray(coord);
    geometry->setColorArray(color);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2));
    geometry->setUseDisplayList(false);

    geometry->setStateSet(VRSceneGraph::instance()->loadUnlightedGeostate());
    return geometry;
}

void
coVR3DTransRotInteractor::startInteraction()
{
    osg::Matrix hm, hm_o, w_to_o, o_to_w, interMat; // hand matrix

    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DTransRotInteractor::startInteraction\n");

    w_to_o = cover->getInvBaseMat();

    hm = getPointerMat(); // hand matrix weltcoord
    hm_o = hm * w_to_o; // hand matrix objekt coord
    _oldHandMat = hm;
    _oldHandMat_o = hm_o;
    _invOldHandMat_o.invert(hm_o); // store the inv hand matrix
    _invOldHandMat.invert(hm);

    o_to_w = cover->getBaseMat();
    interMat = _interMat_o * o_to_w;

    _oldInteractorXformMat = interMat;
    _oldInteractorXformMat_o = _interMat_o;

    coVRIntersectionInteractor::startInteraction();
}

void
coVR3DTransRotInteractor::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DTransRotInteractor::move\n");

    osg::Matrix currHandMat, currHandMat_o, relHandMoveMat, relHandMoveMat_o, interactorXformMat, interactorXformMat_o, w_to_o, o_to_w;

    currHandMat = getPointerMat();
    // forbid translation in y-direction if traverseInteractors is on
    if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRConfig::instance()->useWiiNavigationVisenso())
    {
        osg::Vec3 trans = currHandMat.getTrans();
        trans[1] = _oldHandMat.getTrans()[1];
        currHandMat.setTrans(trans);
    }

    // get hand mat in object coords
    w_to_o = cover->getInvBaseMat();
    currHandMat_o = currHandMat * w_to_o;

    // translate from interactor to hand and back
    osg::Matrix transToHand_w, revTransToHand_w, transToHand_o, revTransToHand_o;

    transToHand_w.makeTranslate(currHandMat.getTrans() - _oldInteractorXformMat.getTrans());
    transToHand_o.makeTranslate(currHandMat_o.getTrans() - _oldInteractorXformMat_o.getTrans());
    revTransToHand_w.makeTranslate(_oldInteractorXformMat.getTrans() - currHandMat.getTrans());
    revTransToHand_o.makeTranslate(_oldInteractorXformMat_o.getTrans() - currHandMat_o.getTrans());

    relHandMoveMat_o = _invOldHandMat_o * currHandMat_o;
    relHandMoveMat = _invOldHandMat * currHandMat;

    if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors)
    {
        // move old mat to hand position, apply rel hand movement and move it back to
        interactorXformMat_o = _oldInteractorXformMat_o * transToHand_o * relHandMoveMat_o * revTransToHand_o;
        interactorXformMat = _oldInteractorXformMat * transToHand_w * relHandMoveMat * revTransToHand_w;
    }
    else
    {
        // apply rel hand movement
        interactorXformMat_o = _oldInteractorXformMat_o * relHandMoveMat_o;
        interactorXformMat = _oldInteractorXformMat * relHandMoveMat;
    }
    if (cover->restrictOn())
    {
        // restrict to visible scene
        osg::Vec3 pos_o, restrictedPos_o;
        pos_o = interactorXformMat_o.getTrans();
        restrictedPos_o = restrictToVisibleScene(pos_o);
        interactorXformMat_o.setTrans(restrictedPos_o);

        o_to_w = cover->getBaseMat();
        interactorXformMat = interactorXformMat_o * o_to_w;
    }

    // save old transformation
    _oldInteractorXformMat = interactorXformMat;
    _oldInteractorXformMat_o = interactorXformMat_o;

    _oldHandMat = currHandMat; // save current hand for rotation start
    _oldHandMat_o = currHandMat_o;
    _invOldHandMat.invert(currHandMat);
    _invOldHandMat_o.invert(currHandMat_o);

    if (coVRNavigationManager::instance()->isSnapping() && !coVRNavigationManager::instance()->isDegreeSnapping())
    {
        // snap orientation to 45 degree
        snapTo45Degrees(&interactorXformMat_o);
        o_to_w = cover->getBaseMat();
        interactorXformMat = interactorXformMat_o * o_to_w;
    }
    else if (coVRNavigationManager::instance()->isSnapping() && coVRNavigationManager::instance()->isDegreeSnapping())
    {
        // snap orientation
        snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &interactorXformMat_o);
        o_to_w = cover->getBaseMat();
        interactorXformMat = interactorXformMat_o * o_to_w;
    }

    // and now we apply it
    updateTransform(interactorXformMat_o);
}

void
coVR3DTransRotInteractor::updateTransform(osg::Matrix m)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coVR3DTransRotInteractor:setMatrix\n");

    _interMat_o = m;
    ////interMat_o.print(0, 1,"interMat_o :", stderr);

    moveTransform->setMatrix(m);
}
