/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Tracer.h"
#include <alg/coUniTracer.h>
#include <cover/VRSceneGraph.h>

#include "../../covise/COVISE/VRCoviseGeometryManager.h"

#include <cover/coTranslator.h>

#include "ChargedObjectHandler.h"
#include "ElectricFieldPlugin.h"

const float SIZE_OF_PLANE = 0.75f;
const int STREAMLINES_PER_SIDE = 3; // total number of streamlines is (2*STREAMLINES_PER_SIDE+1)^2
const float STREAMLINES_WIDTH = 3.0f;

Tracer::Tracer()
    : GenericGuiObject("Tracer")
{
    p_visible = addGuiParamBool("Visible", false);
    p_matrix = addGuiParamMatrix("Matrix", osg::Matrix::translate(osg::Vec3(0.01, 0.0, 0.0)));

    interactor = new coVR3DTransRotInteractor(osg::Matrix::translate(osg::Vec3(0.01, 0.0, 0.0)), 0.1, coInteraction::ButtonA, "hand", "TracerInteractor", coInteraction::Medium);
    interactor->enableIntersection();

    menuItemVisible = new coCheckboxMenuItem(coTranslator::coTranslate("Zeige Feldlinien"), false, NULL);
    menuItemVisible->setMenuListener(this);
    ElectricFieldPlugin::plugin->getObjectsMenu()->insert(menuItemVisible, 0);
}

Tracer::~Tracer()
{
}

void Tracer::preFrame()
{
    if (interactor->isRunning())
    {
        osg::Matrix m = interactor->getMatrix();
        osg::Vec3 position = m.getTrans();
        float min = ChargedObjectHandler::Instance()->getGridMin();
        float max = ChargedObjectHandler::Instance()->getGridMax();
        if (position[0] > max)
            position[0] = max;
        else if (position[0] < min)
            position[0] = min;
        if (position[1] > max)
            position[1] = max;
        else if (position[1] < min)
            position[1] = min;
        if (position[2] > max)
            position[2] = max;
        else if (position[2] < min)
            position[2] = min;
        m.setTrans(position);

        interactor->updateTransform(m);
        update();
    }

    if (interactor->wasStopped())
    {
        p_matrix->setValue(interactor->getMatrix());
    }
}

void Tracer::guiParamChanged(GuiParam *guiParam)
{
    if (guiParam == p_matrix)
        interactor->updateTransform(p_matrix->getValue());
    if (guiParam == p_visible)
        menuItemVisible->setState(p_visible->getValue());
    // if any parameter was changed, just update
    update();
}

void Tracer::menuEvent(coMenuItem *menuItem)
{
    if (menuItem == menuItemVisible)
    {
        p_visible->setValue(menuItemVisible->getState());
    }
    // if any parameter was changed, just update
    update();
}

void
Tracer::update()
{
    if (p_visible->getValue())
    {
        interactor->show();
        interactor->enableIntersection();
    }
    else
    {
        interactor->hide();
        interactor->disableIntersection();
    }

    if (p_visible->getValue() && ChargedObjectHandler::Instance()->fieldIsValid())
    {
        osg::Matrix m_smoke = interactor->getMatrix();
        osg::Vec3 position = m_smoke.getTrans();
        osg::Vec3 dir(0.0, SIZE_OF_PLANE / float(2 * STREAMLINES_PER_SIDE), 0.0);
        osg::Vec3 dir2(0.0, 0.0, SIZE_OF_PLANE / float(2 * STREAMLINES_PER_SIDE));
        dir = m_smoke.getRotate() * dir;
        dir2 = m_smoke.getRotate() * dir2;
        // compute streamline
        float yarrini[6];
        std::vector<std::vector<coUniState> > solus;
        std::vector<coUniState> solution;

        float min = ChargedObjectHandler::Instance()->getGridMin();
        float max = ChargedObjectHandler::Instance()->getGridMax();
        int steps = ChargedObjectHandler::Instance()->getGridSteps();
        coUniTracer unitracer(min, max, min, max, min, max, steps, steps, steps,
                              ChargedObjectHandler::Instance()->getFieldU(), ChargedObjectHandler::Instance()->getFieldV(), ChargedObjectHandler::Instance()->getFieldW());

        for (int x = -STREAMLINES_PER_SIDE; x <= STREAMLINES_PER_SIDE; x++)
        {
            for (int z = -STREAMLINES_PER_SIDE; z <= STREAMLINES_PER_SIDE; z++)
            {
                yarrini[0] = position[0] + x * dir[0] + z * dir2[0];
                yarrini[1] = position[1] + x * dir[1] + z * dir2[1];
                yarrini[2] = position[2] + x * dir[2] + z * dir2[2];
                // Error must be small because we have problems with the quadratic field if the interactor is in a low region (jumps too far when tracing towards the charge) -> decreases performance!
                // Length doesnt really matter (should be high enough to follow a curve through our grid area)
                // MinVelocity must be small because of the quadratic field (seems to be relative to the maximum and the field inside a pointcharge is very high)
                unitracer.solve(yarrini, solution, 0.000003, 5.0, 0.001, 1);
                solus.push_back(solution);
                unitracer.solve(yarrini, solution, 0.000003, 5.0, 0.001, -1);
                solus.push_back(solution);
            }
        }

        solutions_.set(solus);
        displaySmoke();
    }
    else
    {
        if (smokeGeode_)
        {
            cover->getObjectsScale()->removeChild(geometryNode.get());
            cover->getObjectsRoot()->removeChild(smokeGeode_.get());
        }
    }
}

void
Tracer::displaySmoke()
{
    if (!smokeGeode_) //firsttime
    {
        smokeGeode_ = new osg::Geode();

        osg::Vec4Array *colorLine, *colorPoly;
        colorLine = new osg::Vec4Array(1);
        colorPoly = new osg::Vec4Array(1);
        coordLine_ = new osg::Vec3Array(5);
        coordPoly_ = new osg::Vec3Array(4);
        polyNormal_ = new osg::Vec3Array(1);

        (*colorLine)[0].set(1.0, 0.5, 0.5, 1.0);
        (*colorPoly)[0].set(1.0, 0.5, 0.5, 0.5);

        geometryLine_ = new osg::Geometry();
        geometryLine_->setColorArray(colorLine);
        geometryLine_->setColorBinding(osg::Geometry::BIND_OVERALL);
        geometryLine_->setVertexArray(coordLine_.get());
        geometryLine_->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 5));
        geometryLine_->setUseDisplayList(false);
        geometryLine_->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());

        geometryPoly_ = new osg::Geometry();
        geometryPoly_->setColorArray(colorPoly);
        geometryPoly_->setColorBinding(osg::Geometry::BIND_OVERALL);
        geometryPoly_->setVertexArray(coordPoly_.get());
        geometryPoly_->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));
        geometryPoly_->setUseDisplayList(false);
        geometryPoly_->setStateSet(VRSceneGraph::instance()->loadTransparentGeostate());

        geometryNode = new osg::Geode();
        geometryNode->addDrawable(geometryLine_.get());
        geometryNode->addDrawable(geometryPoly_.get());
        geometryNode->setName("TracerPlaneInteractor");
        geometryNode->setNodeMask(geometryNode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    }

    if (!smokeGeometry_)
    {
        smokeGeometry_ = new osg::Geometry();
        smokeGeometry_->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
        smokeGeometry_->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(STREAMLINES_WIDTH), osg::StateAttribute::ON);
        smokeGeode_->addDrawable(smokeGeometry_.get());
    }

    // remove last primitives
    for (unsigned int i = 0; i < smokeGeometry_->getNumPrimitiveSets(); i++)
        smokeGeometry_->removePrimitiveSet(i);

    // set color
    smokeColor_ = new osg::Vec4Array();
    smokeColor_->push_back(osg::Vec4(0.3, 1.0, 0.3, 1.0));
    smokeGeometry_->setColorArray(smokeColor_.get());
    smokeGeometry_->setColorBinding(osg::Geometry::BIND_OVERALL);
    smokeGeometry_->getStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    int numPoints = 0;
    //fprintf(stderr,"solutions_.size=%d\n", (int)solutions_.size());
    for (int i = 0; i < solutions_.size(); i++)
    {
        //fprintf(stderr,"line %d has %d points\n", i, solutions_.lengths()->at(i));
        numPoints += solutions_.lengths()->at(i);
    }

    if (numPoints != 0)
    {
        smokeGeometry_->addPrimitiveSet(solutions_.lengths());
        smokeGeometry_->setVertexArray(solutions_.linepoints());
    }
    else
    {
        // draw one point
        osg::Vec3 p = interactor->getMatrix().getTrans();
        osg::ref_ptr<osg::Vec3Array> points = new osg::Vec3Array();
        points->push_back(osg::Vec3(p[0], p[1], p[2]));
        smokeGeometry_->setVertexArray(points.get());
        osg::ref_ptr<osg::DrawArrayLengths> primitives = new osg::DrawArrayLengths(osg::PrimitiveSet::POINTS);
        primitives->push_back(1);
        smokeGeometry_->addPrimitiveSet(primitives.get());
    }

    osg::Vec3 pos1, pos2, pos3, pos4;
    osg::Matrix m_smoke = interactor->getMatrix();
    osg::Vec3 pos0 = m_smoke.getTrans();
    osg::Vec3 dir(0.0, SIZE_OF_PLANE / 2.0, 0.0);
    osg::Vec3 dir2(0.0, 0.0, SIZE_OF_PLANE / 2.0);
    dir = m_smoke.getRotate() * dir;
    dir2 = m_smoke.getRotate() * dir2;
    pos1 = pos0 - dir - dir2;
    pos2 = pos0 + dir + dir2;
    pos3 = pos0 - dir + dir2;
    pos4 = pos0 + dir - dir2;

    // draw rectangle
    (*coordLine_)[0].set(pos1[0], pos1[1], pos1[2]);
    (*coordLine_)[1].set(pos3[0], pos3[1], pos3[2]);
    (*coordLine_)[2].set(pos2[0], pos2[1], pos2[2]);
    (*coordLine_)[3].set(pos4[0], pos4[1], pos4[2]);
    (*coordLine_)[4].set(pos1[0], pos1[1], pos1[2]);

    (*coordPoly_)[0].set(pos1[0], pos1[1], pos1[2]);
    (*coordPoly_)[1].set(pos3[0], pos3[1], pos3[2]);
    (*coordPoly_)[2].set(pos2[0], pos2[1], pos2[2]);
    (*coordPoly_)[3].set(pos4[0], pos4[1], pos4[2]);

    osg::Vec3 v1 = pos1 - pos0;
    osg::Vec3 v2 = pos2 - pos1;
    osg::Vec3 vn = v1 ^ v2;
    vn.normalize();
    (*polyNormal_)[0].set(vn[0], vn[1], vn[2]);
    if (geometryLine_ != NULL)
        geometryLine_->dirtyBound();
    if (geometryPoly_ != NULL)
        geometryPoly_->dirtyBound();

    if (!cover->getObjectsScale()->containsNode(geometryNode.get()))
        cover->getObjectsScale()->addChild(geometryNode.get());
    if (!cover->getObjectsRoot()->containsNode(smokeGeode_.get()))
        cover->getObjectsRoot()->addChild(smokeGeode_.get());
}
