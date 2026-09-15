/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Trajectory.h"

#include <osg/Material>
#include <osg/PolygonOffset>

#include <cover/coVRPluginSupport.h>
#include <cover/VRSceneGraph.h>

#include "AuralRealityPlugin.h"

using namespace opencover;

osg::Matrix scale_offset = osg::Matrix::translate(osg::Vec3(1, 1, 1) * 0.2);

TrajectoryPoint::TrajectoryPoint(Trajectory *trajectory_)
    : trajectory(trajectory_)
    , anchorInteractor(osg::Matrix::identity(), 1000, vrui::coInteraction::ButtonA, "hand", "speakerInteractor", vrui::coInteraction::Medium)
    , controlPointInInteractor(osg::Matrix::identity(), 100, vrui::coInteraction::ButtonA, "hand", "speakerInteractor", vrui::coInteraction::Medium)
    , controlPointOutInteractor(osg::Matrix::identity(), 100, vrui::coInteraction::ButtonA, "hand", "speakerInteractor", vrui::coInteraction::Medium)
    , scaleInteractor(osg::Matrix::identity(), 100, vrui::coInteraction::ButtonA, "hand", "speakerInteractor", vrui::coInteraction::Medium)
{
    controlPointInInteractor.setModes(CustomTransformInteractor::TRANSLATE);
    controlPointOutInteractor.setModes(CustomTransformInteractor::TRANSLATE);
    scaleInteractor.setModes(CustomTransformInteractor::TRANSLATE);

    groupNode = new osg::Group;
    trajectory->getTransformNode()->addChild(groupNode);

    std::vector<osg::ref_ptr<osg::MatrixTransform> *> nodes = { &anchorNode, &controlPointInNode, &controlPointOutNode };
    int i = 0;
    for (auto n : nodes)
    {
        *n = new osg::MatrixTransform;
        groupNode->addChild(*n);

        osg::Sphere *mySphere = new osg::Sphere(osg::Vec3(0, 0, 0), i == 0 ? 0.08 : 0.04);
        osg::TessellationHints *hint = new osg::TessellationHints();
        hint->setDetailRatio(0.5);
        auto sphereDrawable = new osg::ShapeDrawable(mySphere, hint);
        sphereDrawable->setColor(i == 0 ? osg::Vec4(1, 1, 1, 1) : i == 1 ? osg::Vec4(0, 1, 0, 1)
                                                                         : osg::Vec4(1, 0, 0, 1));
        auto sphereGeode = new osg::Geode();
        sphereGeode->addDrawable(sphereDrawable);
        sphereGeode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));

        (*n)->addChild(sphereGeode);

        i++;
    }

    sensor = new SelectableSensor(&(AuralRealityPlugin::instance()->selection()), this, groupNode.get());

    updateSelection();
}

TrajectoryPoint::~TrajectoryPoint()
{
    delete sensor;
}

bool TrajectoryPoint::checkForChanges()
{
    bool changed = false;
    sensor->update();

    anchorInteractor.preFrame();
    if (anchorInteractor.isRunning())
    {
        // osg::Matrix before = anchor;
        anchor = anchorInteractor.getMatrix().getTrans();
        rotation = anchorInteractor.getMatrix().getRotate();

        // controlPointIn = (osg::Matrix::translate(controlPointIn) * osg::Matrix::inverse(before) * anchor).getTrans();
        // controlPointInNode->setMatrix(osg::Matrix::translate(controlPointIn));
        // controlPointInInteractor.updateTransform(osg::Matrix::translate(controlPointIn));
        //
        // controlPointOut = (osg::Matrix::translate(controlPointOut) * osg::Matrix::inverse(before) * anchor).getTrans();
        // controlPointOutNode->setMatrix(osg::Matrix::translate(controlPointOut));
        // controlPointOutInteractor.updateTransform(osg::Matrix::translate(controlPointOut));

        changed = true;
    }

    controlPointInInteractor.preFrame();
    if (controlPointInInteractor.isRunning())
    {
        controlPointIn = controlPointInInteractor.getMatrix().getTrans() - anchor;
        changed = true;
    }

    controlPointOutInteractor.preFrame();
    if (controlPointOutInteractor.isRunning())
    {
        controlPointOut = controlPointOutInteractor.getMatrix().getTrans() - anchor;
        changed = true;
    }

    scaleInteractor.preFrame();
    if (scaleInteractor.wasStarted())
    {
        scaleInteractorDistance = (scaleInteractor.getMatrix().getTrans() - anchor).length();
    }
    else if (scaleInteractor.isRunning())
    {
        float distance = (scaleInteractor.getMatrix().getTrans() - anchor).length();
        float scaleFactor = distance / scaleInteractorDistance;
        scaleInteractorDistance = distance;

        controlPointIn *= scaleFactor;
        controlPointOut *= scaleFactor;

        changed = true;
    }
    else if (scaleInteractor.wasStopped())
    {
        scaleInteractor.updateTransform(scale_offset * anchorNode->getMatrix());
    }

    if (changed)
    {
        updateNodeTransforms(!scaleInteractor.isRunning());
    }

    return changed;
}

void TrajectoryPoint::setTransforms(const osg::Vec3 &anchor_, const osg::Vec3 &controlPointIn_, const osg::Vec3 &controlPointOut_, const osg::Quat &rotation_)
{
    anchor = anchor_;
    controlPointIn = controlPointIn_;
    controlPointOut = controlPointOut_;
    rotation = rotation_;
    updateNodeTransforms();
}

void TrajectoryPoint::setTime(double time)
{
    m_time = time;
}

void TrajectoryPoint::updateNodeTransforms(bool includeScaleInteractor)
{
    auto anchorMatrix = osg::Matrix::rotate(rotation) * osg::Matrix::translate(anchor);
    anchorNode->setMatrix(anchorMatrix);
    controlPointInNode->setMatrix(osg::Matrix::translate(anchor + controlPointIn));
    controlPointOutNode->setMatrix(osg::Matrix::translate(anchor + controlPointOut));

    anchorInteractor.updateTransform(anchorMatrix);
    controlPointInInteractor.updateTransform(controlPointInNode->getMatrix());
    controlPointOutInteractor.updateTransform(controlPointOutNode->getMatrix());

    if (includeScaleInteractor)
    {
        scaleInteractor.updateTransform(scale_offset * anchorMatrix);
    }
}

void TrajectoryPoint::updateSelection()
{
    if (isSelected())
    {
        anchorInteractor.show();
        anchorInteractor.enableIntersection();

        controlPointInInteractor.show();
        controlPointInInteractor.enableIntersection();

        controlPointOutInteractor.show();
        controlPointOutInteractor.enableIntersection();

        scaleInteractor.show();
        scaleInteractor.enableIntersection();
    }
    else
    {
        anchorInteractor.hide();
        anchorInteractor.disableIntersection();

        controlPointInInteractor.hide();
        controlPointInInteractor.disableIntersection();

        controlPointOutInteractor.hide();
        controlPointOutInteractor.disableIntersection();

        scaleInteractor.hide();
        scaleInteractor.disableIntersection();
    }
}

Trajectory::Trajectory(const std::string &id)
    : TmtEntity(id)
    , TmtEntityTransformMixin()
{
    linesGeode = new osg::Geode();
    getTransformNode()->addChild(linesGeode);

    linesGeometry = new osg::Geometry();
    linesGeode->addDrawable(linesGeometry);

    bezierGeometry = new osg::Geometry();
    linesGeode->addDrawable(bezierGeometry);

    rebuildGeometry();
}
Trajectory::~Trajectory() { }

void Trajectory::preFrame()
{
    bool changed = false;

    for (auto &p : points)
    {
        changed |= p->checkForChanges();
    }

    changed |= checkTransformChanged();

    if (changed)
    {
        rebuildGeometry();
        AuralRealityPlugin::instance()->pushTrajectory(getId());
    }
}

osg::Vec4 GRAY(0.5, 0.5, 0.5, 0.5);
osg::Vec4 GREEN(0, 1, 0, 0.5);
osg::Vec4 RED(1, 0, 0, 0.5);
osg::Vec4 WHITE(1, 1, 1, 1.0);

void Trajectory::rebuildGeometry()
{
    // Geometry for grid lines
    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array();
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();

    osg::Vec3 prev;
    bool first = true;

    for (auto &p : points)
    {
        if (!first)
        {
            verts->push_back(prev);
            verts->push_back(p->anchor + p->controlPointIn);
            colors->push_back(GRAY);
            colors->push_back(GRAY);
        }

        verts->push_back(p->anchor + p->controlPointIn);
        verts->push_back(p->anchor);
        colors->push_back(GREEN);
        colors->push_back(GREEN);

        verts->push_back(p->anchor);
        verts->push_back(p->anchor + p->controlPointOut);
        colors->push_back(RED);
        colors->push_back(RED);

        first = false;
        prev = p->anchor + p->controlPointOut;
    }

    linesGeometry->setVertexArray(verts.get());
    linesGeometry->setColorArray(colors.get(), osg::Array::BIND_PER_VERTEX);
    linesGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    linesGeometry->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, verts->size()));
    osg::ref_ptr<osg::StateSet> ss = linesGeometry->getOrCreateStateSet();
    osg::ref_ptr<osg::LineWidth> lw = new osg::LineWidth(2.0);
    ss->setAttributeAndModes(lw, osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
    ss->setAttributeAndModes(new osg::PolygonOffset(-1.0f, -1.0f), osg::StateAttribute::ON);

    // Now the bezier curve
    verts = new osg::Vec3Array();
    colors = new osg::Vec4Array();

    for (int i = closed ? 0 : 1; i < points.size(); i++)
    {
        int i0 = (i == 0 ? points.size() : i) - 1;
        const auto &p0 = points[i0]->anchor;
        const auto &p3 = points[i]->anchor;
        const auto &p1 = p0 + points[i0]->controlPointOut;
        const auto &p2 = p3 + points[i]->controlPointIn;

        auto pP = p0;
        float dt = 1.0 / 32;
        for (float t = dt; t <= 1; t += dt)
        {
            float t_ = 1 - t;
            auto p = p0 * (t_ * t_ * t_) + p1 * (3 * t_ * t_ * t) + p2 * (3 * t_ * t * t) + p3 * (t * t * t);

            verts->push_back(pP);
            verts->push_back(p);
            colors->push_back(WHITE);
            colors->push_back(WHITE);
            pP = p;
        }
    }

    bezierGeometry->setVertexArray(verts.get());
    bezierGeometry->setColorArray(colors.get(), osg::Array::BIND_PER_VERTEX);
    bezierGeometry->setColorBinding(osg::Geometry::BIND_PER_VERTEX);
    bezierGeometry->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, verts->size()));
    ss = bezierGeometry->getOrCreateStateSet();
    lw = new osg::LineWidth(5.0);
    ss->setAttributeAndModes(lw, osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
    ss->setAttributeAndModes(new osg::PolygonOffset(-1.0f, -1.0f), osg::StateAttribute::ON);
}
