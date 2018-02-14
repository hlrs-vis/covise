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
#include <osg/io_utils>

const float ArrowLength = 5.0f;

using namespace opencover;

coVR3DTransRotInteractor::coVR3DTransRotInteractor(osg::Matrix m, float s, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority = Medium)
    : coVRIntersectionInteractor(s, type, iconName, interactorName, priority)
{
    if (cover->debugLevel(2))
    {
        fprintf(stderr, "new coVR3DTransRotInteractor(%s)\n", interactorName);
    }

    createGeometry();

    coVR3DTransRotInteractor::updateTransform(m);
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

    bool restrictedInteraction = true;

    osg::ShapeDrawable *xlConeDrawable, *xrConeDrawable, *ylConeDrawable, *yrConeDrawable, *zlConeDrawable, *zrConeDrawable, *sphereDrawable;
    //osg::Geometry *xaxisDrawable, *yaxisDrawable, *zaxisDrawable;

    osg::Vec3 origin(0, 0, 0), px(1, 0, 0), py(0, 1, 0), pz(0, 0, 1);
    osg::Vec3 yaxis(0, 1, 0);
    osg::Vec3 normal(0, -ArrowLength, 0);
    osg::Vec4 red(1, 0, 0, 1), green(0, 1, 0, 1), blue(0, 0, 1, 1), color(0.5, 0.5, 0.5, 1);
    red.set(0.5, 0.2, 0.2, 1.0);
    green.set(0.2, 0.5, 0.2, 1.0);
    blue.set(0.2, 0.2, 0.5, 1.0);

    axisTransform = new osg::MatrixTransform();
    axisTransform->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));
    geometryNode = axisTransform.get();
    scaleTransform->addChild(geometryNode.get());

    osg::Sphere *mySphere = new osg::Sphere(origin, 0.5);
    osg::TessellationHints *hint = new osg::TessellationHints();
    hint->setDetailRatio(0.5);
    sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
    sphereDrawable->setColor(color);
    sphereGeode = new osg::Geode();
    sphereGeode->addDrawable(sphereDrawable);
    axisTransform->addChild(sphereGeode.get());

    if (restrictedInteraction)
    {
        auto myNormalShape = new osg::Cone(origin, 0.5, 2.0);
        osg::ShapeDrawable *normalSphereDrawable = new osg::ShapeDrawable(myNormalShape, hint);
        normalSphereDrawable->setColor(green);
        rotateGeode = new osg::Geode();
        rotateGeode->addDrawable(normalSphereDrawable);
        auto rotateTransform = new osg::MatrixTransform;
        rotateTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.), 1, 0, 0)*osg::Matrix::translate(normal));
        rotateTransform->addChild(rotateGeode);

        auto cyl = new osg::Cylinder(origin, 0.15, ArrowLength);
        auto cylDrawable = new osg::ShapeDrawable(cyl, hint);
        cylDrawable->setColor(color);
        translateGeode = new osg::Geode;
        translateGeode->addDrawable(cylDrawable);
        auto translateTransform = new osg::MatrixTransform;
        translateTransform->setMatrix(osg::Matrix::rotate(osg::inDegrees(90.), 1, 0, 0)*osg::Matrix::translate(normal*0.5));
        translateTransform->addChild(translateGeode);

        axisTransform->addChild(rotateTransform);
        axisTransform->addChild(translateTransform);
    }

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

    xlConeDrawable->setColor(red);
    xrConeDrawable->setColor(red);
    ylConeDrawable->setColor(green);
    yrConeDrawable->setColor(green);
    zlConeDrawable->setColor(blue);
    zrConeDrawable->setColor(blue);

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


    axisTransform->addChild(xrTransform.get());
    axisTransform->addChild(xlTransform.get());
    if (!restrictedInteraction)
        axisTransform->addChild(yrTransform.get());
    axisTransform->addChild(ylTransform.get());
    axisTransform->addChild(zrTransform.get());
    axisTransform->addChild(zlTransform.get());

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
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DTransRotInteractor::startInteraction\n");

    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix o_to_w = cover->getBaseMat();

    osg::Matrix hm = getPointerMat(); // hand matrix weltcoord
    osg::Matrix hm_o = hm * w_to_o; // hand matrix objekt coord
    _oldHandMat = hm;
    _invOldHandMat_o.invert(hm_o); // store the inv hand matrix

    osg::Matrix interMat = _interMat_o * o_to_w;

    _oldInteractorXformMat_o = _interMat_o;

    osg::Vec3 interPos = getMatrix().getTrans();
    // get diff between intersection point and sphere center
    _diff = interPos - _hitPos;
    _distance = (_hitPos - hm_o.getTrans()).length();

    _rotateOnly = _hitNode==rotateGeode;
    _translateOnly = _hitNode==translateGeode;
    if (!_rotateOnly && !_translateOnly)
    {
        _translateOnly = is2D();
    }

    coVRIntersectionInteractor::startInteraction();
}

void
coVR3DTransRotInteractor::doInteraction()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\ncoVR3DTransRotInteractor::move\n");

    osg::Vec3 origin(0, 0, 0);
    osg::Vec3 yaxis(0, 1, 0);

    osg::Matrix currHandMat = getPointerMat();
    // forbid translation in y-direction if traverseInteractors is on
    if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors && coVRConfig::instance()->useWiiNavigationVisenso())
    {
        osg::Vec3 trans = currHandMat.getTrans();
        trans[1] = _oldHandMat.getTrans()[1];
        currHandMat.setTrans(trans);
    }

    osg::Matrix o_to_w = cover->getBaseMat();
    // get hand mat in object coords
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;

    // translate from interactor to hand and back
    osg::Matrix transToHand_o, revTransToHand_o;

    transToHand_o.makeTranslate(currHandMat_o.getTrans() - _oldInteractorXformMat_o.getTrans());
    revTransToHand_o.makeTranslate(_oldInteractorXformMat_o.getTrans() - currHandMat_o.getTrans());

    osg::Matrix relHandMoveMat_o = _invOldHandMat_o * currHandMat_o;

    osg::Matrix interactorXformMat_o = _oldInteractorXformMat_o;
    if (_rotateOnly)
    {
        osg::Matrix i_to_o = scaleTransform->getMatrix()*moveTransform->getMatrix();
        osg::Matrix o_to_i = osg::Matrix::inverse(i_to_o);
        osg::Vec3 hand_i = origin * currHandMat * w_to_o * o_to_i;
        osg::Vec3 pos = hand_i;
        osg::Vec3 dir = yaxis * currHandMat * w_to_o * o_to_i;
        dir -= pos;
        dir.normalize();
        //std::cerr << "pos: " << pos << ", dir: " << dir << std::endl;
        double R = _diff.length() / getScale();
        double a = dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2];
        double b = 2.*(dir[0]*pos[0] + dir[1]*pos[1] + dir[2]*pos[2]);
        double c = pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2] - R*R;
        double D = b*b-4*a*c;
        //std::cerr << "scale=" << getScale() << ", a=" << a << ", b=" << b << ", c=" << c << ", disc=" << D << std::endl;
        double t = -1.;
        if (D >= 0)
        {
            double t1 = 0.5*(-b-sqrt(D))/a;
            double t2 = 0.5*(-b+sqrt(D))/a;
            if (t1 < 0)
            {
                t = t2;
            }
            else if (is2D())
            {
                t = t1;
            }
            else
            {
                double old = _distance / getScale();
                if (std::abs(old-t1) < std::abs(old-t2))
                    t = t1;
                else
                    t = t2;
            }
            //std::cerr << "solution: t1=" << t1 << ", t2=" << t2 << ", t=" << t << std::endl;
            //osg::Vec3 v1 = pos+dir*t1;
            //osg::Vec3 v2 = pos+dir*t2;
            //std::cerr << "    v1: " << v1 << ", v2: " << v2 << std::endl;
        }
        if (t < 0)
        {
            t = -dir * pos;
        }
        if (t >= 0)
        {
            _distance = t * getScale();
            osg::Vec3 isect = pos+dir*t;
            //std::cerr << "valid intersection: t=" << t << ", p=" << isect << ", dist=" << isect.length() << std::endl;
            osg::Matrix rot;
            rot.makeRotate(osg::Vec3(0,-1,0), isect);

            interactorXformMat_o = rot * getMatrix();
        }
        else
        {
            _distance = 0;
        }
    }
    else if (_translateOnly)
    {
        auto lp1_o = origin * currHandMat_o;
        auto lp2_o = yaxis * currHandMat_o;

        auto pointerDir_o = lp2_o - lp1_o;
        pointerDir_o.normalize();

        // get hand pos in object coords
        auto currHandPos_o = currHandMat_o.getTrans();

        auto interPos = currHandPos_o + pointerDir_o * _distance + _diff;
        interactorXformMat_o.setTrans(interPos);
    }
    else if (coVRNavigationManager::instance()->getMode() == coVRNavigationManager::TraverseInteractors)
    {
        // move old mat to hand position, apply rel hand movement and move it back to
        interactorXformMat_o = _oldInteractorXformMat_o * transToHand_o * relHandMoveMat_o * revTransToHand_o;
    }
    else
    {
        // apply rel hand movement
        interactorXformMat_o = _oldInteractorXformMat_o * relHandMoveMat_o;
    }

    // save old transformation
    _oldInteractorXformMat_o = interactorXformMat_o;

    _oldHandMat = currHandMat; // save current hand for rotation start
    _invOldHandMat_o.invert(currHandMat_o);

    if (cover->restrictOn())
    {
        // restrict to visible scene
        osg::Vec3 pos_o, restrictedPos_o;
        pos_o = interactorXformMat_o.getTrans();
        restrictedPos_o = restrictToVisibleScene(pos_o);
        interactorXformMat_o.setTrans(restrictedPos_o);
    }

    if (coVRNavigationManager::instance()->isSnapping())
    {
        if (coVRNavigationManager::instance()->isDegreeSnapping())
        {
            // snap orientation
            snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &interactorXformMat_o);
        }
        else
        {
            // snap orientation to 45 degree
            snapTo45Degrees(&interactorXformMat_o);
        }
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
