/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FlightPathVisualizer.h"

#include <math.h>

using namespace osg;
using covise::coCoviseConfig;
using namespace opencover;

BezierCurveVisualizer::Computation computation = BezierCurveVisualizer::CUBIC_APROXIMATION;

FlightPathVisualizer::FlightPathVisualizer(coVRPluginSupport *cover,
                                           std::vector<ViewDesc *> *_viewpoints)
    : viewpoints(_viewpoints)
{
    flightPathDCS = new MatrixTransform;
    lineDCS = NULL;

    eyepoint[0] = coCoviseConfig::getFloat("x", "COVER.ViewerPosition", 0.0f);
    eyepoint[1] = coCoviseConfig::getFloat("y", "COVER.ViewerPosition", -1500.0f);
    eyepoint[2] = coCoviseConfig::getFloat("z", "COVER.ViewerPosition", 0.0f);

    // Load initial Viewpoints (1, 10, 100, 1000 etc.)
    for (vector<ViewDesc *>::iterator it = viewpoints->begin(); it < viewpoints->end(); it++)
    {
        addViewpoint(*it);
    }

    cover->getObjectsRoot()->addChild(flightPathDCS.get());
}

FlightPathVisualizer::~FlightPathVisualizer()
{
    // delete nodes and scenegraph
    if (lineDCS)
        lineDCS = 0;

    if (flightPathDCS)
    {
        if (flightPathDCS->getNumParents())
        {
            flightPathDCS->getParent(0)->removeChild(flightPathDCS);
            //			pfDelete(flightPathDCS);                  // noch nötig ?
        }
    }
}

/*
 * Add Viewpoints from Covise Config
 * or manually in VR
 */
void FlightPathVisualizer::addViewpoint(ViewDesc *viewDesc)
{
    if (viewDesc->hasOrientation() == false
        || viewDesc->hasPosition() == false
        || viewDesc->hasScale() == false
        || viewDesc->isViewAll() == true)
    {
        return;
    }

    //add viewpoint to list
    vpList.push_back(viewDesc);

    //add curveVisualizer to list
    BezierCurveVisualizer *curve = new BezierCurveVisualizer(flightPathDCS, computation);
    bezierCurveVis.push_back(curve);

    updateDrawnCurve();
}

void FlightPathVisualizer::removeViewpoint(ViewDesc *viewDesc)
{
    if (viewDesc->hasOrientation() == false
        || viewDesc->hasPosition() == false
        || viewDesc->hasScale() == false)
    {
        return;
    }
    vpList.remove(viewDesc);
    updateDrawnCurve();
}

void FlightPathVisualizer::showFlightpath(bool state)
{
    if (flightPathDCS)
    {
        if (state == true)
        {
            std::vector<BezierCurveVisualizer *>::iterator iter;
            for (iter = bezierCurveVis.begin(); iter != bezierCurveVis.end(); iter++)
            {
                (*iter)->showCurve(true);
            }
        }
        else
        {
            std::vector<BezierCurveVisualizer *>::iterator iter;
            for (iter = bezierCurveVis.begin(); iter != bezierCurveVis.end(); iter++)
            {
                (*iter)->showCurve(false);
            }
        }
    }
}

void FlightPathVisualizer::updateCamera(Matrix cameraMatrix)
{
    Matrix objectSpace;
    objectSpace.invert(cameraMatrix);
    cameraDCS->setMatrix(objectSpace);
}

void FlightPathVisualizer::createCameraGeometry(ViewDesc *viewDesc)
{
    cameraGeode = new Geode;
    cameraPlaneGeoset = new osg::Geometry;
    cameraPlaneBorderGeoset = new osg::Geometry;
    cameraGeoset = new osg::Geometry;
    line1 = new osg::Geometry;
    line2 = new osg::Geometry;
    line3 = new osg::Geometry;
    line4 = new osg::Geometry;
    cameraDCS = new MatrixTransform;
    cameraSwitchNode = new Switch;

    cameraPlaneCoords = new Vec3Array(5);
    cameraBorderCoords = new Vec3Array(8);
    lineEyetoLeftDown = new Vec3Array(2);
    lineEyetoRightDown = new Vec3Array(2);
    lineEyetoRightUp = new Vec3Array(2);
    lineEyetoLeftUp = new Vec3Array(2);

    cameraPlaneCoords.get()->at(0) = Vec3(-800, 0, -600);
    cameraPlaneCoords.get()->at(1) = Vec3(800, 0, -600);
    cameraPlaneCoords.get()->at(2) = Vec3(800, 0, 600);
    cameraPlaneCoords.get()->at(3) = Vec3(-800, 0, 600);
    cameraPlaneCoords.get()->at(4) = eyepoint;

    cameraBorderCoords.get()->at(0) = Vec3(-800, 0, -600);
    cameraBorderCoords.get()->at(1) = Vec3(800, 0, -600);
    cameraBorderCoords.get()->at(2) = Vec3(800, 0, -600);
    cameraBorderCoords.get()->at(3) = Vec3(800, 0, 600);
    cameraBorderCoords.get()->at(4) = Vec3(800, 0, 600);
    cameraBorderCoords.get()->at(5) = Vec3(-800, 0, 600);
    cameraBorderCoords.get()->at(6) = Vec3(-800, 0, 600);
    cameraBorderCoords.get()->at(7) = Vec3(-800, 0, -600);

    Vec3 lu = Vec3(-800, 0, -600);
    Vec3 ru = Vec3(800, 0, -600);
    Vec3 ro = Vec3(800, 0, 600);
    Vec3 lo = Vec3(-800, 0, 600);
    Vec3 eye = eyepoint;

    lineEyetoLeftDown.get()->at(0) = eye;
    lineEyetoLeftDown.get()->at(1) = lu;

    lineEyetoRightDown.get()->at(0) = eye;
    lineEyetoRightDown.get()->at(1) = ru;

    lineEyetoRightUp.get()->at(0) = eye;
    lineEyetoRightUp.get()->at(1) = ro;

    lineEyetoLeftUp.get()->at(0) = eye;
    lineEyetoLeftUp.get()->at(1) = lo;

    line1->setVertexArray(lineEyetoLeftDown);
    line1->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line2->setVertexArray(lineEyetoRightDown);
    line2->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line3->setVertexArray(lineEyetoRightUp);
    line3->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line4->setVertexArray(lineEyetoLeftUp);
    line4->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));

    cameraGeoset->setVertexArray(cameraPlaneCoords);
    cameraPlaneGeoset->setVertexArray(cameraPlaneCoords);
    cameraPlaneBorderGeoset->setVertexArray(cameraBorderCoords);

    ref_ptr<Vec4Array> cameraColor = new Vec4Array;
    cameraColor->push_back(Vec4(1.0, 1.0, 0.0, 0.5)); // yellow
    ref_ptr<Vec4Array> cameraLineColor = new Vec4Array;
    cameraLineColor->push_back(Vec4(1.0, 1.0, 0.0, 1.0)); // yellow

    cameraPlaneGeoset->setVertexArray(cameraPlaneCoords);
    cameraPlaneGeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    cameraPlaneGeoset->setColorArray(cameraColor);
    cameraPlaneGeoset->setColorBinding(Geometry::BIND_OVERALL);
    cameraPlaneGeoset_state = cameraPlaneGeoset->getOrCreateStateSet();
    loadUnlightedGeostate(cameraPlaneGeoset_state);
    LineWidth *lw = new LineWidth(2.0);
    cameraPlaneGeoset_state->setAttribute(lw);

    cameraPlaneBorderGeoset->setVertexArray(cameraBorderCoords);
    cameraPlaneBorderGeoset->setColorArray(cameraLineColor);
    cameraPlaneBorderGeoset->setColorBinding(Geometry::BIND_OVERALL);
    cameraPlaneBorderGeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 8));
    cameraPlaneBorderGeoset_state = cameraPlaneBorderGeoset->getOrCreateStateSet();
    loadUnlightedGeostate(cameraPlaneBorderGeoset_state);
    cameraPlaneBorderGeoset_state->setAttribute(lw);

    line1->setVertexArray(lineEyetoLeftDown);
    line1->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line1->setColorArray(cameraLineColor.get());
    line1->setColorBinding(Geometry::BIND_OVERALL);
    line1_state = line1->getOrCreateStateSet();
    loadUnlightedGeostate(line1_state);
    line1_state->setAttribute(lw);

    line2->setVertexArray(lineEyetoRightDown);
    line2->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line2->setColorArray(cameraLineColor.get());
    line2->setColorBinding(Geometry::BIND_OVERALL);
    line2_state = line2->getOrCreateStateSet();
    loadUnlightedGeostate(line2_state);
    line2_state->setAttribute(lw);

    line3->setVertexArray(lineEyetoRightUp);
    line3->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line3->setColorArray(cameraLineColor.get());
    line3->setColorBinding(Geometry::BIND_OVERALL);
    line3_state = line3->getOrCreateStateSet();
    loadUnlightedGeostate(line3_state);
    line3_state->setAttribute(lw);

    line4->setVertexArray(lineEyetoLeftUp);
    line4->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line4->setColorArray(cameraLineColor.get());
    line4->setColorBinding(Geometry::BIND_OVERALL);
    line4_state = line4->getOrCreateStateSet();
    loadUnlightedGeostate(line4_state);
    line4_state->setAttribute(lw);

    cameraGeode->addDrawable(line1.get());
    cameraGeode->addDrawable(line2.get());
    cameraGeode->addDrawable(line3.get());
    cameraGeode->addDrawable(line4.get());
    cameraGeode->addDrawable(cameraPlaneGeoset);
    cameraGeode->addDrawable(cameraPlaneBorderGeoset);
    cameraGeode->addDrawable(cameraGeoset);

    Matrix m;
    float scale_ = viewDesc->getScale();

    if (viewDesc->hasMatrix())
    {
        m = viewDesc->getMatrix();
        m.preMultScale(Vec3(scale_, scale_, scale_));
        m.invert(m);
        cameraDCS->setMatrix(m);
    }
    else
    {
        viewDesc->coord.makeMat(m);
        m.preMultScale(Vec3(scale_, scale_, scale_));
        m.invert(m);
        cameraDCS->setMatrix(m);
    }

    cameraDCS->addChild(cameraGeode.get());
    cameraSwitchNode->addChild(cameraDCS.get(), true);
    cover->getObjectsRoot()->addChild(cameraSwitchNode.get());
}

void FlightPathVisualizer::deleteCameraGeometry()
{
    cameraSwitchNode->removeChild(cameraDCS.get());
    cameraDCS->removeChild(cameraGeode.get());
    cameraSwitchNode = NULL;
    cameraDCS = NULL;
}

void FlightPathVisualizer::updateDrawnCurve()
{
    //deactivate all BezierCurveVisualizer by removing their controlPoints
    for (size_t i = 0; i < bezierCurveVis.size(); i++)
    {
        bezierCurveVis[i]->removeAllControlPoints();
        bezierCurveVis[i]->updateGeometry();
    }

    //create List with active VP
    list<ViewDesc *> activeVpList;
    for (list<ViewDesc *>::const_iterator iter = vpList.begin(); iter != vpList.end(); iter++)
    {
        if ((*iter)->getFlightPathActivated() && (*iter)->hasGeometry())
        {
            activeVpList.push_back(*iter);
        }
    }

    // cancel if only one viewpoint
    if (activeVpList.size() < 2)
    {
        return;
    }

    list<ViewDesc *>::const_iterator iter;
    iter = activeVpList.begin();

    std::vector<BezierCurveVisualizer *>::iterator iter2;
    iter2 = bezierCurveVis.begin();

    // get first Viewpoint
    ViewDesc *v1 = *iter;
    MatrixTransform *v1dcs = v1->localDCS;

    Matrix v1dcsmat;
    v1dcsmat = v1dcs->getMatrix();
    iter++;

    for (; iter != activeVpList.end(); iter++)
    {
        // get second Viewpoint
        ViewDesc *v2 = *iter;
        MatrixTransform *v2dcs = v2->localDCS;
        Matrix v2dcsmat;
        v2dcsmat = v2dcs->getMatrix();

        // vectors to V1, V2
        Vec3 p1;
        Vec3 p4;

        p1 = v1dcsmat.getTrans();
        p4 = v2dcsmat.getTrans();

        // tangent vectors from V1, V2
        Vec3 p2 = v1->getScaledTangentOut();
        Vec3 p3 = v2->getScaledTangentIn();
        p2 = Matrix::transform3x3(p2, v1dcsmat);
        p3 = Matrix::transform3x3(p3, v2dcsmat);

        p2 += p1;
        p3 += p4;

        Vec3 shiftV1 = Vec3(0, 0, 0);
        Vec3 shiftV2 = Vec3(0, 0, 0);
        if (shiftFlightpathToEyePoint)
        {
            shiftV1 = -eyepoint / v1->getScale();
            shiftV2 = -eyepoint / v2->getScale();
        }

        shiftV1 = Matrix::transform3x3(shiftV1, v1dcsmat);
        shiftV2 = Matrix::transform3x3(shiftV2, v2dcsmat);
        p1 -= shiftV1;
        p2 -= shiftV1;
        p3 -= shiftV2;
        p4 -= shiftV2;

        // TODO dynamic stepsize?

        (*iter2)->removeAllControlPoints();

        (*iter2)->addControlPoint(p1);
        (*iter2)->addControlPoint(p2);
        (*iter2)->addControlPoint(p3);
        (*iter2)->addControlPoint(p4);

        (*iter2)->updateGeometry();

        v1 = v2;
        v1dcsmat = v2dcsmat;
        iter2++;
    }
}

void FlightPathVisualizer::loadUnlightedGeostate(ref_ptr<StateSet> state)
{
    ref_ptr<Material> mat = new Material;
    mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mat->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mat->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mat->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.f));
    mat->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    mat->setShininess(Material::FRONT_AND_BACK, 16.f);
    mat->setTransparency(Material::FRONT_AND_BACK, 1.f); // Noch Wert anpassen für Transparency

    state->setAttributeAndModes(mat, osg::StateAttribute::ON);
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
}

void FlightPathVisualizer::shiftFlightpath(bool state)
{
    shiftFlightpathToEyePoint = state;
}
