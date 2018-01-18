/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#define XK_MISCELLANY
#if !defined(_WIN32) && !defined(__APPLE__)
#include <X11/keysymdef.h>
#endif
#include "ViewPoint.h"
#include <util/covise_regexp.h>
#include <util/unixcompat.h>
#include <PluginUtil/PluginMessageTypes.h>
#include "ViewDesc.h"
#include <osg/ClipNode>
#include <config/CoviseConfig.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPluginSupport.h>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/ClipNode>
#include <osg/ClipNode>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Switch>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osg/LineWidth>
#include <osg/Material>
#include <cmath>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <OpenVRUI/coInteraction.h>

using namespace osg;
using covise::coCoviseConfig;
using namespace opencover;
using vrui::coInteraction;

ViewDesc::ViewDesc(const char *n, int id, const char *line,
                   ui::Menu *menu, ui::Menu *flightMenu, ui::Menu *editMenu,
                   ViewPoints *master, bool isChangeable)
: ui::Owner("Viewpoint"+std::to_string(id), cover->ui)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "----- ViewDesc::ViewDesc [id=%d][name=%s][line=%s]\n", id, n, line);

    id_ = id;
    isChangeable_ = isChangeable;
    isViewAll_ = false;
    isChangeableFromCover_ = coCoviseConfig::isOn("COVER.Plugin.ViewPoint.ChangeableFromCover", false);

    vp_line = line;
    scale_ = 1.0;
    name = n;
    flightState_ = true;
    hasGeometry_ = false;
    hasTangentOut_ = true;
    hasTangentIn_ = true;
    flightPathActivated = true;

    // viewpoint from covise7.0 project
    if (string2ViewDesc(vp_line,
                        &scale_,
                        &xformMat_(0, 0), &xformMat_(0, 1), &xformMat_(0, 2), &xformMat_(0, 3),
                        &xformMat_(1, 0), &xformMat_(1, 1), &xformMat_(1, 2), &xformMat_(1, 3),
                        &xformMat_(2, 0), &xformMat_(2, 1), &xformMat_(2, 2), &xformMat_(2, 3),
                        &xformMat_(3, 0), &xformMat_(3, 1), &xformMat_(3, 2), &xformMat_(3, 3),
                        &tangentIn[0], &tangentIn[1], &tangentIn[2], &tangentOut[0], &tangentOut[1], &tangentOut[2]))
    {
        hasScale_ = hasOrientation_ = hasPosition_ = hasMatrix_ = true;
        //fprintf(stderr, "7.0");
    }
    // viewpoint from file or from covise6.0 project
    else if (string2ViewDesc(vp_line,
                             &scale_,
                             &coord.xyz[0], &coord.xyz[1], &coord.xyz[2],
                             &coord.hpr[0], &coord.hpr[1], &coord.hpr[2],
                             &tangentIn[0], &tangentIn[1], &tangentIn[2], &tangentOut[0], &tangentOut[1], &tangentOut[2]))
    {
        hasScale_ = hasOrientation_ = hasPosition_ = true;
        hasMatrix_ = false;
        //fprintf(stderr, "6.0");
    }

    // line not correct
    else
    {
        coord.xyz[0] = coord.xyz[1] = coord.xyz[2] = 0.0;
        coord.hpr[0] = coord.hpr[1] = coord.hpr[2] = 0.0;
        hasScale_ = hasOrientation_ = hasPosition_ = true;
        hasMatrix_ = false;
        if (cover->debugLevel(3))
            cerr << "This viewpoint is not correct, assuming zero for all Values " << endl;
    }

    activated_ = false;

    createButtons(n, menu, flightMenu, editMenu, master);
    createGeometry();
}

ViewDesc::ViewDesc(const char *n, int id, float scale, Matrix m,
                   ui::Menu *menu, ui::Menu *flightMenu, ui::Menu *editMenu,
                   ViewPoints *master, bool isChangeable)
: ui::Owner("Viewpoint"+std::to_string(id), cover->ui)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "----- ViewDesc::ViewDesc [id=%d][name=%s][scale=%f][matrix=%f %f %f %f | %f %f %f %f | %f %f %f %f | %f %f %f %f]\n", id, n, scale, m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3), m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));

    id_ = id;
    isChangeable_ = isChangeable;
    scale_ = scale;
    name = n;
    xformMat_ = m;
    tangentIn = Vec3(0, -500, 0);
    tangentOut = Vec3(0, 500, 0);
    isViewAll_ = false;
    isChangeableFromCover_ = coCoviseConfig::isOn("COVER.Plugin.ViewPoint.ChangeableFromCover", false);

    flightState_ = true;
    hasScale_ = hasOrientation_ = hasPosition_ = true;
    hasMatrix_ = true;
    activated_ = false;
    hasGeometry_ = false;
    hasTangentOut_ = true;
    hasTangentIn_ = true;
    flightPathActivated = true;

    createButtons(n, menu, flightMenu, editMenu, master);

    createGeometry();
}

ViewDesc::ViewDesc(const char *n, int id,
                   ui::Menu *menu, ui::Menu *flightMenu, ui::Menu *editMenu,
                   ViewPoints *master, bool isChangeable)
: ui::Owner("Viewpoint"+std::to_string(id), cover->ui)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "----- new ViewDesc [%d] [%s]\n", id, n);

    id_ = id;
    isChangeable_ = isChangeable;
    isChangeableFromCover_ = coCoviseConfig::isOn("COVER.Plugin.ViewPoint.ChangeableFromCover", false);
    isViewAll_ = coCoviseConfig::isOn("COVER.Plugin.ViewPoint.ViewAll", false);
    hasScale_ = isViewAll_;

    createButtons(n, menu, flightMenu, editMenu, master);

    scale_ = 1.0;

    hasOrientation_ = hasPosition_ = false;
    hasMatrix_ = false;
    activated_ = false;
    hasGeometry_ = false;
    hasTangentOut_ = false;
    hasTangentIn_ = false;
    flightPathActivated = true;

    name = n;
    flightState_ = true;

    coord.xyz[0] = coord.xyz[1] = coord.xyz[2] = coord.hpr[0] = coord.hpr[1] = coord.hpr[2] = 0.0;

    createGeometry();
}

ViewDesc::ViewDesc(const char *n, int id, float scale,
                   ui::Menu *menu, ui::Menu *flightMenu, ui::Menu *editMenu,
                   ViewPoints *master, bool isChangeable)
: ui::Owner("Viewpoint"+std::to_string(id), cover->ui)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "----- new ViewDesc [id=%d][name=%s][scale=%f]\n", id, n, scale);

    id_ = id;
    isChangeable_ = isChangeable;
    isViewAll_ = false;
    isChangeableFromCover_ = coCoviseConfig::isOn("COVER.Plugin.ViewPoint.ChangeableFromCover", false);

    createButtons(n, menu, flightMenu, editMenu, master);

    scale_ = scale;

    hasScale_ = true;
    hasOrientation_ = hasPosition_ = false;
    hasMatrix_ = false;
    activated_ = false;
    hasGeometry_ = false;
    hasTangentOut_ = false;
    hasTangentIn_ = false;
    flightPathActivated = true;

    name = n;
    flightState_ = true;

    createGeometry();
}

ViewDesc::ViewDesc(const char *n, int id, Vec3 hpr,
                   ui::Menu *menu, ui::Menu *flightMenu, ui::Menu *editMenu,
                   ViewPoints *master, bool isChangeable)
: ui::Owner("Viewpoint"+std::to_string(id), cover->ui)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "----- new ViewDesc [id=%d][name=%s] hpr=[%f %f %f]\n", id, n, hpr[0], hpr[1], hpr[2]);

    id_ = id;
    isChangeable_ = isChangeable;

    createButtons(n, menu, flightMenu, editMenu, master);

    scale_ = -1.0;

    hasOrientation_ = true;
    hasMatrix_ = false;
    activated_ = false;
    hasGeometry_ = false;
    hasTangentOut_ = false;
    hasTangentIn_ = false;
    flightPathActivated = true;
    name = n;
    flightState_ = true;

    coord.xyz = osg::Vec3(0.0f, 0.0f, 0.0f);
    coord.hpr = hpr;

    // if not here, Viewpoints name is not read in correctly
    isChangeableFromCover_ = coCoviseConfig::isOn("COVER.Plugin.ViewPoint.ChangeableFromCover", false);

    isViewAll_ = coCoviseConfig::isOn("COVER.Plugin.ViewPoint.ViewAll", false);
    hasScale_ = isViewAll_;
    hasPosition_ = isViewAll_;

    createGeometry();
}

void ViewDesc::changeViewDesc()
{
    if (!isChangeable_)
        return;

    scale_ = cover->getScale();
    hasScale_ = true;

    xformMat_ = cover->getObjectsXform()->getMatrix();
    hasMatrix_ = true;

    if (cover->debugLevel(3))
        fprintf(stderr, "----- ViewDesc::changeViewDesc for [id=%d][name=%s] into [scale=%f][matrix=%f %f %f %f | %f %f %f %f | %f %f %f %f | %f %f %f %f]\n", id_, name, scale_, xformMat_(0, 0), xformMat_(0, 1), xformMat_(0, 2), xformMat_(0, 3), xformMat_(1, 0), xformMat_(1, 1), xformMat_(1, 2), xformMat_(1, 3), xformMat_(2, 0), xformMat_(2, 1), xformMat_(2, 2), xformMat_(2, 3), xformMat_(3, 0), xformMat_(3, 1), xformMat_(3, 2), xformMat_(3, 3));
}

void ViewDesc::changeViewDesc(float scale, Matrix m)
{
    if (!isChangeable_)
        return;

    if (cover->debugLevel(3))
        fprintf(stderr, "----- ViewDesc::changeViewDesc for [id=%d][name=%s] into [scale=%f][matrix=%f %f %f %f | %f %f %f %f | %f %f %f %f | %f %f %f %f]\n", id_, name, scale, m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3), m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));

    scale_ = scale;
    hasScale_ = true;

    xformMat_ = m;
    hasMatrix_ = true;
}

void ViewDesc::createButtons(const char *name,
                             ui::Menu *menu, ui::Menu *flightMenu, ui::Menu *editMenu,
                             ViewPoints *master)
{
    button_ = new ui::Action("Button", this);
    button_->setText(name);
    menu->add(button_);
    button_->setCallback([this](){
        //fprintf(stderr,"menuItem == button_");
        bool clip = ViewPoints::instance()->useClipPlanesCheck_->state();
        activate(clip);
        ViewPoints::instance()->activateViewpoint(this);
    });

    flightButton_ = new ui::Button("Flight", this);
    flightButton_->setText(name);
    flightButton_->setState(true);
    flightMenu->add(flightButton_);
    flightButton_->setCallback([this](bool state){
        flightState_ = state;
    });

    viewpointVisible = false;
    tangentVisible = false;
    editViewpoint = false;
    editTangent = false;

    // submenu for each viewpoint
    editVPMenu_ = new ui::Menu("Edit", this);
    editVPMenu_->setText(name);
    editMenu->add(editVPMenu_);

    showViewpointCheck_ = new ui::Button(editVPMenu_, "ShowHideViewpoint");
    showViewpointCheck_->setText("Show/hide viewpoint");
    showViewpointCheck_->setState(viewpointVisible);
    showViewpointCheck_->setCallback([this](bool state){
        viewpointVisible = state;
        flightPathActivated = viewpointVisible;

        if (viewpointVisible == false)
        {
            showMoveInteractors(false);
            showTangent(false);
            showTangentInteractors(false);
            showTangentInteractorsCheck_->setState(false);
        }
        showGeometry(viewpointVisible);
    });

    showTangentCheck_ = new ui::Button(editVPMenu_, "ShowHideTangent");
    showTangentCheck_->setText("Show/hide tangent");
    showTangentCheck_->setState(tangentVisible);
    showTangentCheck_->setCallback([this](bool state){
        tangentVisible = state;

        if (tangentVisible == false)
        {
            showTangentInteractors(false);
            showTangentInteractorsCheck_->setState(false);
        }

        showTangent(tangentVisible);
    });

    showMoveInteractorsCheck_ = new ui::Button(editVPMenu_, "MoveScaleViewpoint");
    showMoveInteractorsCheck_->setText("Move/scale viewpoint");
    showMoveInteractorsCheck_->setState(editViewpoint);
    showMoveInteractorsCheck_->setCallback([this](bool state){
        editViewpoint = state;
        if (editViewpoint)
        {
            // also show viewpoint
            viewpointVisible = true;
            flightPathActivated = true;
            showViewpointCheck_->setState(true);
            showGeometry(editViewpoint);
        }
        showMoveInteractors(editViewpoint);
    });

    showTangentInteractorsCheck_ = new ui::Button(editVPMenu_, "EditTangents");
    showTangentInteractorsCheck_->setText("Edit tangents");
    showTangentInteractorsCheck_->setState(editTangent);
    showTangentInteractorsCheck_->setCallback([this](bool state){
        editTangent = state;
        if (editTangent)
        {
            // also show tangent
            tangentVisible = true;
            showTangentCheck_->setState(true);
            showTangent(true);
        }
        showTangentInteractors(editTangent);
    });

    updateViewButton = new ui::Action(editVPMenu_, "UpdateToCurrentPosition");
    updateViewButton->setText("Update to current position");
    updateViewButton->setCallback([this](){
        osg::Matrix m = cover->getObjectsXform()->getMatrix();
        coord = m;
        ref_ptr<ClipNode> clipNode = cover->getObjectsRoot();
        if (ViewPoints::instance()->isClipPlaneChecked())
        {

            for (unsigned int i = 0; i < clipNode->getNumClipPlanes(); i++)
            {
                ClipPlane *cp = clipNode->getClipPlane(i);
                Vec4 plane = cp->getClipPlane();
                char planeString[1024];

                snprintf(planeString, 1024, "%d %f %f %f %f ", cp->getClipPlaneNum(), plane[0], plane[1], plane[2], plane[3]);
                // add to the current viewpoint
                addClipPlane(planeString);
            }
        }
        ViewPoints::instance()->saveAllViewPoints();
    });

    if (isChangeable_ && isChangeableFromCover_)
    {
        changeButton_ = new ui::Action("Change", this);
        string text = "Change ";
        text.append(name);
        changeButton_->setText(text);
        menu->add(changeButton_);
        changeButton_->setCallback([this](){
            ViewPoints::instance()->changeViewDesc(this);
        });
    }
}

ViewDesc::~ViewDesc()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "ViewDesc::~ViewDesc\n");

    deleteGeometry();
}

bool ViewDesc::string2ViewDesc(const std::string &line,
                               float *scale,
                               float *x, float *y, float *z,
                               float *h, float *p, float *r,
                               float *tanInX, float *tanInY, float *tanInZ,
                               float *tanOutX, float *tanOutY, float *tanOutZ)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "ViewDesc::string2ViewDesc euler\n");

    *scale = 1.0;
    *x = *y = *z = *h = *p = *r = 0.0;
    *tanInX = *tanInY = *tanInZ = *tanOutX = *tanOutY = *tanOutZ = 0.0;

    int numRes = sscanf(line.c_str(), "%f %f %f %f %f %f %f %f %f %f %f %f %f", scale, x, y, z, h, p, r,
                        tanInX, tanInY, tanInZ, tanOutX, tanOutY, tanOutZ);
    if (numRes == 7)
    {
        if (cover->debugLevel(4))
            cerr << "Viewpoints line was not complete: setting tangents to 0,0,0" << endl;
        return true;
    }
    else if (numRes != 13)
    {
        if (cover->debugLevel(4))
            cerr << "Viewpoints line was not correct: '" << line << "' read " << numRes << " elements, should be 13" << endl;
        return false;
    }
    else
        return true;
}

bool ViewDesc::string2ViewDesc(const std::string &line,
                               float *scale,
                               double *m00, double *m01, double *m02, double *m03,
                               double *m10, double *m11, double *m12, double *m13,
                               double *m20, double *m21, double *m22, double *m23,
                               double *m30, double *m31, double *m32, double *m33,
                               float *tanInX, float *tanInY, float *tanInZ,
                               float *tanOutX, float *tanOutY, float *tanOutZ)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "ViewDesc::string2ViewDesc matrix\n");

    *scale = 1.0;
    *m00 = *m11 = *m22 = *m33 = 1.0;
    *m01 = *m02 = *m03 = 0.0;
    *m10 = *m12 = *m13 = 0.0;
    *m20 = *m21 = *m23 = 0.0;
    *m30 = *m31 = *m32 = 0.0;
    *tanInX = *tanInY = *tanInZ = *tanOutX = *tanOutY = *tanOutZ = 0.0;

    int numRes = sscanf(line.c_str(), "%f %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %f %f %f %f %f %f ", scale, m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23, m30, m31, m32, m33, tanInX, tanInY, tanInZ, tanOutX, tanOutY, tanOutZ);
    //   fprintf(stderr, "%f\n %lf %lf %lf %lf \n %lf %lf %lf %lf \n  %lf %lf %lf %lf \n %lf %lf %lf %lf \n %f %f %f \n %f %f %f \n",*scale, *m00, *m01, *m02, *m03, *m10, *m11, *m12, *m13, *m20, *m21, *m22, *m23, *m30, *m31, *m32, *m33, *tanInX, *tanInY, *tanInZ, *tanOutX, *tanOutY, *tanOutZ);
    if (numRes == 17)
    {
        if (cover->debugLevel(4))
            cerr << "Viewpoints line was not complete: setting tangents to 0,0,0" << endl;
        return true;
    }
    else if (numRes != 23)
    {
        if (cover->debugLevel(4))
            cerr << "Viewpoints line was not correct: '" << line << "' read " << numRes << " elements, should be 13" << endl;
        return false;
    }
    else
        return true;
}

void
ViewDesc::setPosition(const char *posString)
{
    sscanf(posString, "%f %f %f", &(coord.xyz[0]), &(coord.xyz[1]), &(coord.xyz[2]));
    hasPosition_ = true;

    if (!hasGeometry_)
        createGeometry();
    updateGeometry();
}

void
ViewDesc::setScale(const char *scaleString)
{
    scale_ = atof(scaleString);
    hasScale_ = true;

    if (!hasGeometry_)
        createGeometry();
    updateGeometry();
}

void
ViewDesc::setScale(float scale)
{
    scale_ = scale;
    hasScale_ = true;

    if (!hasGeometry_)
        createGeometry();
    updateGeometry();
}

void
ViewDesc::setEuler(const char *eulerString)
{
    sscanf(eulerString, "%f %f %f", &(coord.hpr[0]), &(coord.hpr[1]), &(coord.hpr[2]));
    hasOrientation_ = true;

    if (!hasGeometry_)
        createGeometry();
    updateGeometry();
}

void
ViewDesc::setXFromMatrix(Matrix m)
{
    xformMat_ = m;
    hasMatrix_ = true;
}

void ViewDesc::setTangentOut(const char *tangentString)
{
    sscanf(tangentString, "%f %f %f", &(tangentOut[0]), &(tangentOut[1]), &(tangentOut[2]));
    hasTangentOut_ = true;
    if (!hasGeometry_)
        createGeometry();
    updateGeometry();
}

void ViewDesc::setTangentOut(Vec3 tanOut)
{
    tangentOut = tanOut;
    hasTangentOut_ = true;
    if (!hasGeometry_)
        createGeometry();
    updateGeometry();
}

void ViewDesc::setTangentIn(const char *tangentString)
{
    sscanf(tangentString, "%f %f %f", &(tangentIn[0]), &(tangentIn[1]), &(tangentIn[2]));
    hasTangentIn_ = true;
    if (!hasGeometry_)
        createGeometry();
    updateGeometry();
}

void ViewDesc::setTangentIn(Vec3 tanIn)
{
    tangentIn = tanIn;
    hasTangentIn_ = true;
    if (!hasGeometry_)
        createGeometry();
    updateGeometry();
}

bool ViewDesc::hasTangent()
{
    // true, if both tangents are set
    return hasTangentIn_ && hasTangentOut_;
}

Vec3 ViewDesc::getTangentIn()
{
    return tangentIn;
}

Vec3 ViewDesc::getTangentOut()
{
    return tangentOut;
}

Vec3 ViewDesc::getScaledTangentIn()
{
    Vec3 ret = tangentIn;
    ret *= (1 / scale_);
    return ret;
}

Vec3 ViewDesc::getScaledTangentOut()
{
    Vec3 ret = tangentOut;
    ret *= (1 / scale_);
    return ret;
}

void
ViewDesc::addClipPlane(const char *planeString)
{
    int id;
    float eqn[4];

    sscanf(planeString, "%d %f %f %f %f", &id, &eqn[0], &eqn[1], &eqn[2], &eqn[3]);

    // the ClipPlane plugin only handles 6 planes, so we'll
    // just discard every entry that's not in [0, 5]
    if (id >= 0 && id <= 5)
    {
        clipPlanes[id].enabled = true;
        clipPlanes[id].a = eqn[0];
        clipPlanes[id].b = eqn[1];
        clipPlanes[id].c = eqn[2];
        clipPlanes[id].d = eqn[3];
    }
}

const char *ViewDesc::getClipPlane(int i)
{
    stringstream ret;
    if (clipPlanes[i].enabled)
        ret << i << " " << clipPlanes[i].a << " " << clipPlanes[i].b << " " << clipPlanes[i].c << " " << clipPlanes[i].d;
    else
        ret << "";
    return ret.str().c_str();
}

void ViewDesc::activate(bool clipplane)
{
    if (cover->debugLevel(4))
        fprintf(stderr, "ViewDesc::activate(%s)\n", name.c_str());

    updateToViewAll();

    if (hasScale_ && scale_ > 0.0)
    {
        if (cover->debugLevel(4))
            fprintf(stderr, "vp has scale\n");
        cover->setScale(scale_);
    }
    else
    {
        if (cover->debugLevel(4))
            fprintf(stderr, "vp has no scale\n");
    }

    if (hasOrientation_ && !hasPosition_)
    {
        if (cover->debugLevel(4))
            fprintf(stderr, "vp has only orientation\n");
        osg::BoundingSphere sph;
        Matrix m, euler, initRot, invInitRot, diffRot, xform, trans1, trans2;
        Vec3 /*pos,*/ origin(0, 0, 0);

        sph = cover->getObjectsRoot()->getBound();
        xform = cover->getObjectsXform()->getMatrix();
        trans1.makeTranslate(-sph.center().x(), -sph.center().y(), -sph.center().z());
        MAKE_EULER_MAT(euler, coord.hpr[0], coord.hpr[1], coord.hpr[2]);
        initRot = xform;
        initRot.setTrans(origin);
        invInitRot.invert(initRot);
        diffRot.mult(invInitRot, euler);
        trans2.makeTranslate(sph.center().x(), sph.center().y(), sph.center().z());
        m.mult(xform, trans1);
        m.mult(m, diffRot);
        m.mult(m, trans2);
        cover->getObjectsXform()->setMatrix(m);
    }

    if (hasPosition_ && hasOrientation_ && (coord.hpr[0] < 360.0) && (coord.hpr[0] > -360.0))
    {
        if (hasMatrix_)
        {
            cover->getObjectsXform()->setMatrix(xformMat_);
            if (cover->debugLevel(4))
                fprintf(stderr, "vp has matrix pos is:[%f %f %f]\n", xformMat_(3, 0), xformMat_(3, 2), xformMat_(3, 2));
        }
        else
        {
            Matrix m;
            coord.makeMat(m);
            cover->getObjectsXform()->setMatrix(m);
            if (cover->debugLevel(4))
                fprintf(stderr, "vp has position %f %f %f and orientation %f %f %f\n", coord.xyz[0], coord.xyz[1], coord.xyz[2], coord.hpr[0], coord.hpr[1], coord.hpr[2]);
        }
    }

    if (!hasPosition_ && !hasOrientation_ && !hasScale_)
    {
        if (strncasecmp(name.c_str(), "center", 6) == 0)
        {
            //fprintf(stderr,"viewpoint center\n");
            osg::BoundingBox box;
            box = cover->getBBox(cover->getObjectsRoot());

            float lx = 0;
            float ly = 0;
            float lz = 0;
            if (box.valid())
            {

                lx = fabs(box.xMax() - box.xMin());
                ly = fabs(box.yMax() - box.yMin());
                lz = fabs(box.zMax() - box.zMin());
            }

            if ((lx == 0.0) || (ly == 0.0) || (lz == 0.0))
            {
                if (cover->debugLevel(4))
                    fprintf(stderr, "box is empty, we do not apply\n");
                return;
            }
            if ((lx == 2.0 * FLT_MAX) || (ly == 2.0 * FLT_MAX) || (lz == 2.0 * FLT_MAX))
            {
                if (cover->debugLevel(4))
                    fprintf(stderr, "box contains FLT_MAX, we do not apply\n");
                return;
            }
            Vec3 c_o; //center in object coordinates
            c_o[0] = (box.xMax() + box.xMin()) / 2.0;
            c_o[1] = (box.yMax() + box.yMin()) / 2.0;
            c_o[2] = (box.zMax() + box.zMin()) / 2.0;

            Vec3 c_w; // center in world coordinates
            Matrix o_to_w = cover->getBaseMat();
            c_w = c_o * o_to_w; //c_w.fullXformPt(c_o, o_to_w);

            Matrix m; // get old mat
            m = cover->getObjectsXform()->getMatrix();

            // translate it to center
            m.setTrans(-c_w[0], -c_w[1], -c_w[2]);

            // apply new matrix
            cover->getObjectsXform()->setMatrix(m);
        }

        if (strncasecmp(name.c_str(), "behind", 6) == 0)
        {
            //fprintf(stderr,"viewpoint behind\n");
            osg::BoundingBox box;
            box = cover->getBBox(cover->getObjectsXform());

            float lx = 0;
            float ly = 0;
            float lz = 0;
            if (box.valid())
            {
                lx = fabs(box.xMax() - box.xMin());
                ly = fabs(box.yMax() - box.yMin());
                lz = fabs(box.zMax() - box.zMin());
            }

            if ((lx == 2.0 * FLT_MAX) || (ly == 2.0 * FLT_MAX) || (lz == 2.0 * FLT_MAX))
            {
                if (cover->debugLevel(4))
                    fprintf(stderr, "box contains FLT_MAX, we do not apply\n");
                return;
            }
            Vec3 center; //center in object coordinates
            center[0] = (box.xMax() + box.xMin()) / 2.0;
            center[1] = (box.yMax() + box.yMin()) / 2.0;
            center[2] = (box.zMax() + box.zMin()) / 2.0;

            Matrix m; // get old mat
            m = cover->getObjectsXform()->getMatrix();

            // translate it to center
            m.setTrans(0, -center[1] + 0.5 * ly, 0);

            // apply new matrix
            cover->getObjectsXform()->setMatrix(m);
        }

        if (strncasecmp(name.c_str(), "onfloor", 6) == 0)
        {
            //fprintf(stderr,"viewpoint onfloor\n");
            osg::BoundingBox box;
            box = cover->getBBox(cover->getObjectsRoot());

            float lx = 0;
            float ly = 0;
            float lz = 0;
            if (box.valid())
            {
                lx = fabs(box.xMax() - box.xMin());
                ly = fabs(box.yMax() - box.yMin());
                lz = fabs(box.zMax() - box.zMin());
            }
            if ((lx == 0.0) || (ly == 0.0) || (lz == 0.0))
            {
                if (cover->debugLevel(4))
                    fprintf(stderr, "box is empty, we do not apply\n");
                return;
            }
            if ((lx == 2.0 * FLT_MAX) || (ly == 2.0 * FLT_MAX) || (lz == 2.0 * FLT_MAX))
            {
                if (cover->debugLevel(4))
                    fprintf(stderr, "box contains FLT_MAX, we do not apply\n");
                return;
            }

            Vec3 c_o; //center in object coordinates
            c_o[0] = (box.xMax() + box.xMin()) / 2.0;
            c_o[1] = (box.yMax() + box.yMin()) / 2.0;
            c_o[2] = (box.zMax() + box.zMin()) / 2.0;

            Matrix o_to_w = cover->getBaseMat();
            Vec3 c_w; // center in world coordinates
            c_w = c_o * o_to_w;

            Vec3 max;
            max = Vec3(box.xMax(), box.yMax(), box.zMax()) * o_to_w;
            Vec3 min;
            min = Vec3(box.xMin(), box.yMin(), box.zMin()) * o_to_w;

            float lz_w = fabs(max[2] - min[2]);

            Matrix t;
            // get old position
            t = cover->getObjectsXform()->getMatrix();

            t.setTrans(0, 0, -c_w[2] + lz_w * 0.5);
            cover->getObjectsXform()->setMatrix(t);
        }
    }

    // ****
    // only set clipping planes if the checkbox is checked
    if (clipplane)
    {
        for (int i = 0; i < 6; i++)
        {
            char msg[128];
            if (clipPlanes[i].enabled)
            {
                //fprintf(stderr, "..... send enable clipplane %d\n", i);
                // set clipping plane equation
                snprintf(msg, 128, "set %d %f %f %f %f",
                         i, clipPlanes[i].a, clipPlanes[i].b, clipPlanes[i].c, clipPlanes[i].d);
                cover->sendMessage(NULL, "ClipPlane", PluginMessageTypes::ClipPlaneMessage, strlen(msg), &msg);
                // enable it
                snprintf(msg, 9, "enable %d", i);
                cover->sendMessage(NULL, "ClipPlane", PluginMessageTypes::ClipPlaneMessage, 9, &msg);
            }
            else
            {
                // disable all others
                snprintf(msg, 10, "disable %d", i);
                cover->sendMessage(NULL, "ClipPlane", PluginMessageTypes::ClipPlaneMessage, 10, &msg);
            }
        }
    }

    // set flag, that viewpoint is reached
    // will send message to GUI in ViewPoint.cpp
    activated_ = true;
}

bool ViewDesc::isClipPlaneEnabled(int plane)
{
    if (plane < 0 || plane > 5)
        return false;

    return clipPlanes[plane].enabled;
}

bool ViewDesc::equalVP(Matrix m)
{
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (((m(i, j) - xformMat_(i, j)) * (m(i, j) - xformMat_(i, j))) > 0.0005)
                return false;
        }
    }
    return true;
}

bool ViewDesc::nearVP(Matrix m)
{
    Vec3 viewpoint;
    Vec3 destination;

    viewpoint = this->xformMat_.getTrans();
    destination = m.getTrans();

    float length;
    length = (viewpoint - destination).length();
    fprintf(stderr, "LENGTH: %f \n", length);
    if (length < 50.0)
        return true;
    else
        return false;
}

void ViewDesc::setFlightState(bool state)
{
    flightState_ = state;
    flightButton_->setState(state);
}

void ViewDesc::createGeometry()
{
    // return if geometry already created
    if (hasGeometry_)
        return;

    // return if not all data available
    if (!hasScale() || !hasPosition() || !hasOrientation() || !hasTangent())
    {
        //fprintf(stderr, "\n---ViewDesc::createGeometry failed: Not enough information\n");
        return;
    }

    if (cover->debugLevel(4))
        fprintf(stderr, "\nViewDesc::createGeometry\n");

    localDCS = new MatrixTransform();
    localDCS->setName("VPLoaclDCS");

    viewpointSwitchNode = new Switch();
    viewpointSwitchNode->setName("VPSwitch");
    tangentSwitchNode = new Switch();
    tangentSwitchNode->setName("VPTangentSwitch");
    tangentgeode = new Geode();
    tangentgeode->setName("VPTangentGeode");
    viewpointgeode = new Geode();
    viewpointgeode->setName("VPGeode");
    tangentlinesgeoset = new osg::Geometry();
    viewpointPlaneGeoset = new osg::Geometry();
    viewpointPlaneBorderGeoset = new osg::Geometry();
    viewpointGeoset = new osg::Geometry();

    line1 = new osg::Geometry();
    line2 = new osg::Geometry();
    line3 = new osg::Geometry();
    line4 = new osg::Geometry();

    // read eyepoint from Covise Config
    //initial view position
    eyepoint[0] = coCoviseConfig::getFloat("x", "COVER.ViewerPosition", 0.0f);
    eyepoint[1] = coCoviseConfig::getFloat("y", "COVER.ViewerPosition", -1500.0f);
    eyepoint[2] = coCoviseConfig::getFloat("z", "COVER.ViewerPosition", 0.0f);

    //create Interactors

    float _interSize = cover->getSceneSize() / 25;

    tanOutInteractor = new coVR3DTransInteractor(tangentOut, _interSize, coInteraction::ButtonA, "hand",
                                                 "vpTanOutInteractor", coInteraction::Medium);
    tanInInteractor = new coVR3DTransInteractor(tangentIn, _interSize, coInteraction::ButtonA, "hand",
                                                "vpTanInInteractor", coInteraction::Medium);
    scaleInteractor = new coVR3DTransInteractor(coord.xyz, _interSize, coInteraction::ButtonA, "hand",
                                                "vpScaleInteractor", coInteraction::Medium);
    //initially hide interactors
    tanOutInteractor->hide();
    tanInInteractor->hide();
    scaleInteractor->hide();

    Matrix localMat;
    localDCS->setMatrix(localMat);
    viewpointInteractor = new coVR3DTransRotInteractor(localMat, _interSize, coInteraction::ButtonA, "hand", "vpInteractor", coInteraction::Medium);

    // // create Label
    // float fontsize = 35 / scale_;
    // float lineLen = 20;
    // Vec4 fgColor(0.0, 1.0, 0.0, 1.0);
    // Vec4 bgColor(1.0, 1.0, 1.0, 0.0);
    // myLabel = new coVRLabel(" ", fontsize, lineLen, fgColor, bgColor);

    lineEyetoLeftDown = new Vec3Array(2);
    lineEyetoRightDown = new Vec3Array(2);
    lineEyetoRightUp = new Vec3Array(2);
    lineEyetoLeftUp = new Vec3Array(2);

    tangentlinescoords = new Vec3Array(4);
    viewpointCoords = new Vec3Array(5);
    viewpointBorderCoords = new Vec3Array(8);

    hasGeometry_ = true;
    updateGeometry();

    if (cover->debugLevel(4))
        fprintf(stderr, "\nViewDesc::createGeometry\n");

    ref_ptr<Vec4Array> tangentColors = new Vec4Array();
    tangentColors->push_back(Vec4(0.0, 0.0, 1.0, 1.0)); //azure
    tangentColors->push_back(Vec4(0.6, 0.0, 1.0, 1.0));

    tangentlinesgeoset->setVertexArray(tangentlinescoords.get());
    tangentlinesgeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 4));
    tangentlinesgeoset->setColorArray(tangentColors.get());
    tangentlinesgeoset->setColorBinding(Geometry::BIND_OVERALL);
    tangentlinesgeoset_state = tangentlinesgeoset->getOrCreateStateSet();
    loadUnlightedGeostate(tangentlinesgeoset_state);
    LineWidth *lw = new LineWidth(2.0);
    tangentlinesgeoset_state->setAttribute(lw);

    ref_ptr<Vec4Array> viewpointColor = new Vec4Array();
    viewpointColor->push_back(Vec4(0.0, 1.0, 0.0, 0.5)); // green
    ref_ptr<Vec4Array> viewpointboxLineColor = new Vec4Array();
    viewpointboxLineColor->push_back(Vec4(0.0, 1.0, 0.0, 1.0)); // green

    viewpointPlaneGeoset->setVertexArray(viewpointCoords.get());
    viewpointPlaneGeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    viewpointPlaneGeoset->setColorArray(viewpointColor.get());
    viewpointPlaneGeoset->setColorBinding(Geometry::BIND_OVERALL);
    viewpointPlaneGeoset_state = viewpointPlaneGeoset->getOrCreateStateSet();
    loadUnlightedGeostate(viewpointPlaneGeoset_state);

    viewpointPlaneBorderGeoset->setVertexArray(viewpointBorderCoords.get());
    viewpointPlaneBorderGeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 8));
    viewpointPlaneBorderGeoset->setColorArray(viewpointboxLineColor.get());
    viewpointPlaneBorderGeoset->setColorBinding(Geometry::BIND_OVERALL);
    viewpointPlaneBorderGeoset_state = viewpointPlaneBorderGeoset->getOrCreateStateSet();
    loadUnlightedGeostate(viewpointPlaneBorderGeoset_state);
    viewpointPlaneBorderGeoset_state->setAttribute(lw);

    line1->setVertexArray(lineEyetoLeftDown);
    line1->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line1->setColorArray(viewpointboxLineColor.get());
    line1->setColorBinding(Geometry::BIND_OVERALL);
    line1_state = line1->getOrCreateStateSet();
    loadUnlightedGeostate(line1_state);
    line1_state->setAttribute(lw);

    line2->setVertexArray(lineEyetoRightDown);
    line2->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line2->setColorArray(viewpointboxLineColor.get());
    line2->setColorBinding(Geometry::BIND_OVERALL);
    line2_state = line2->getOrCreateStateSet();
    loadUnlightedGeostate(line2_state);
    line2_state->setAttribute(lw);

    line3->setVertexArray(lineEyetoRightUp);
    line3->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line3->setColorArray(viewpointboxLineColor.get());
    line3->setColorBinding(Geometry::BIND_OVERALL);
    line3_state = line3->getOrCreateStateSet();
    loadUnlightedGeostate(line3_state);
    line3_state->setAttribute(lw);

    line4->setVertexArray(lineEyetoLeftUp);
    line4->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    line4->setColorArray(viewpointboxLineColor.get());
    line4->setColorBinding(Geometry::BIND_OVERALL);
    line4_state = line4->getOrCreateStateSet();
    loadUnlightedGeostate(line4_state);
    line4_state->setAttribute(lw);

    viewpointgeode->addDrawable(line1.get());
    viewpointgeode->addDrawable(line2.get());
    viewpointgeode->addDrawable(line3.get());
    viewpointgeode->addDrawable(line4.get());
    viewpointgeode->addDrawable(viewpointPlaneBorderGeoset.get());
    viewpointgeode->addDrawable(viewpointPlaneGeoset.get());
    viewpointgeode->setNodeMask(viewpointgeode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    tangentgeode->addDrawable(tangentlinesgeoset.get());

    viewpointSwitchNode->addChild(viewpointgeode.get(), false);
    tangentSwitchNode->addChild(tangentgeode.get(), false);

    // myLabel->reAttachTo(localDCS);

    localDCS->addChild(viewpointSwitchNode.get()); //Nicht vergessen bei deleteGeometry auch abaendern!
    localDCS->addChild(tangentSwitchNode.get());
    cover->getObjectsRoot()->addChild(localDCS.get());

    showGeometry(viewpointVisible);
    showTangent(tangentVisible);
    showTangentInteractors(editTangent);
    showMoveInteractors(editViewpoint);
}

void ViewDesc::loadUnlightedGeostate(ref_ptr<StateSet> state)
{
    ref_ptr<Material> mat = new Material;
    mat->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mat->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mat->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mat->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.f));
    mat->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    mat->setShininess(Material::FRONT_AND_BACK, 16.f);
    mat->setTransparency(Material::FRONT_AND_BACK, 1.0f); // Noch Wert anpassen fuer Transparency

    state->setAttributeAndModes(mat, osg::StateAttribute::ON);
    state->setMode(GL_BLEND, osg::StateAttribute::ON);
    state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
}

void ViewDesc::showGeometry(bool state)
{
    if (hasGeometry_)
    {
        if (state == true)
        {
            viewpointSwitchNode->setChildValue(this->viewpointgeode.get(), true);
            // myLabel->show();
            localDCS->setNodeMask(localDCS->getNodeMask() | (Isect::Visible));
        }
        else
        {
            viewpointSwitchNode->setChildValue(this->viewpointgeode.get(), false);
            // myLabel->hide();

            //Hide Interactors
            showMoveInteractorsCheck_->setState(false);
            editViewpoint = false;
        }
        showViewpointCheck_->setState(state);

        if ((showTangentCheck_->state() == false) && (showViewpointCheck_->state() == false))
            localDCS->setNodeMask(localDCS->getNodeMask() & (~(Isect::Visible | Isect::OsgEarthSecondary)));
    }
}

void ViewDesc::deleteGeometry()
{

    //fprintf(stderr, "hasGeometry _= %i", (int) hasGeometry_);
    if (hasGeometry_)
    {
        delete tanInInteractor;
        delete tanOutInteractor;
        delete viewpointInteractor;
        delete scaleInteractor;

        cover->getObjectsRoot()->removeChild(localDCS);
        localDCS = NULL;

        hasGeometry_ = false;
    }
}

void ViewDesc::showTangent(bool state)
{
    if (hasGeometry_)
    {
        if (state == true)
        {
            tangentSwitchNode->setChildValue(this->tangentgeode.get(), true);
            // myLabel->show();

            localDCS->setNodeMask(localDCS->getNodeMask() | (Isect::Visible));
        }
        else
        {
            tangentSwitchNode->setChildValue(this->tangentgeode.get(), false);
            // myLabel->hide();

            //Hide Interactors
            showTangentInteractorsCheck_->setState(false);
            editTangent = false;
        }
        showTangentCheck_->setState(state);

        if ((showTangentCheck_->state() == false) && (showViewpointCheck_->state() == false))
            localDCS->setNodeMask(localDCS->getNodeMask() & (~(Isect::Visible| Isect::OsgEarthSecondary)));
    }
}

void ViewDesc::showMoveInteractors(bool state)
{
    // fprintf(stderr, "\nViewDesc::showMoveInteractors_state %d\n ", (int)state);

    if (hasGeometry_)
    {
        if (state == true)
        {
            viewpointInteractor->show();
            viewpointInteractor->enableIntersection();
            scaleInteractor->show();
            scaleInteractor->enableIntersection();
        }
        else
        {
            viewpointInteractor->hide();
            viewpointInteractor->disableIntersection();
            scaleInteractor->hide();
            scaleInteractor->disableIntersection();
        }
        showMoveInteractorsCheck_->setState(state);
    }
}

void ViewDesc::showTangentInteractors(bool state)
{
    if (hasGeometry_)
    {
        if (state == true)
        {
            tanInInteractor->show();
            tanInInteractor->enableIntersection();
            tanOutInteractor->show();
            tanOutInteractor->enableIntersection();
        }
        else
        {
            tanInInteractor->hide();
            tanInInteractor->disableIntersection();
            tanOutInteractor->hide();
            tanOutInteractor->disableIntersection();
        }
        showTangentInteractorsCheck_->setState(state);
    }
}

void ViewDesc::updateGeometry()
{
    if (!hasGeometry_)
        return;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nViewDesc::updateGeometry\n");
    
    ViewPoints::instance()->dataChanged = true;
    tangentlinescoords->at(0) = Vec3(0, 0, 0);
    tangentlinescoords->at(1) = Vec3(tangentOut[0] / scale_, tangentOut[1] / scale_, tangentOut[2] / scale_);
    tangentlinescoords->at(2) = Vec3(0, 0, 0);
    tangentlinescoords->at(3) = Vec3(tangentIn[0] / scale_, tangentIn[1] / scale_, tangentIn[2] / scale_);

    if (shiftFlightpathToEyePoint)
    {
        tangentlinescoords->at(0) = eyepoint / scale_;
        tangentlinescoords->at(1) += eyepoint / scale_;
        tangentlinescoords->at(2) = eyepoint / scale_;
        tangentlinescoords->at(3) += eyepoint / scale_;
    }

    if (tangentlinesgeoset)
    {
        tangentlinesgeoset->setVertexArray(tangentlinescoords.get());
        tangentlinesgeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 4));
    }

    viewpointCoords.get()->at(0) = Vec3(-800 / scale_, 0, -600 / scale_);
    viewpointCoords.get()->at(1) = Vec3(800 / scale_, 0, -600 / scale_);
    viewpointCoords.get()->at(2) = Vec3(800 / scale_, 0, 600 / scale_);
    viewpointCoords.get()->at(3) = Vec3(-800 / scale_, 0, 600 / scale_);
    viewpointCoords.get()->at(4) = eyepoint / scale_;

    viewpointBorderCoords.get()->at(0) = Vec3(-800 / scale_, 0, -600 / scale_);
    viewpointBorderCoords.get()->at(1) = Vec3(800 / scale_, 0, -600 / scale_);
    viewpointBorderCoords.get()->at(2) = Vec3(800 / scale_, 0, -600 / scale_);
    viewpointBorderCoords.get()->at(3) = Vec3(800 / scale_, 0, 600 / scale_);
    viewpointBorderCoords.get()->at(4) = Vec3(800 / scale_, 0, 600 / scale_);
    viewpointBorderCoords.get()->at(5) = Vec3(-800 / scale_, 0, 600 / scale_);
    viewpointBorderCoords.get()->at(6) = Vec3(-800 / scale_, 0, 600 / scale_);
    viewpointBorderCoords.get()->at(7) = Vec3(-800 / scale_, 0, -600 / scale_);

    Vec3 lu = Vec3(-800 / scale_, 0, -600 / scale_);
    Vec3 ru = Vec3(800 / scale_, 0, -600 / scale_);
    Vec3 ro = Vec3(800 / scale_, 0, 600 / scale_);
    Vec3 lo = Vec3(-800 / scale_, 0, 600 / scale_);
    Vec3 eye = (eyepoint / scale_);

    lineEyetoLeftDown.get()->at(0) = eye;
    lineEyetoLeftDown.get()->at(1) = lu;

    lineEyetoRightDown.get()->at(0) = eye;
    lineEyetoRightDown.get()->at(1) = ru;

    lineEyetoRightUp.get()->at(0) = eye;
    lineEyetoRightUp.get()->at(1) = ro;

    lineEyetoLeftUp.get()->at(0) = eye;
    lineEyetoLeftUp.get()->at(1) = lo;

    if (viewpointPlaneGeoset && viewpointPlaneBorderGeoset && viewpointGeoset)
    {
        line1->setVertexArray(lineEyetoLeftDown);
        line1->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
        line2->setVertexArray(lineEyetoRightDown);
        line2->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
        line3->setVertexArray(lineEyetoRightUp);
        line3->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
        line4->setVertexArray(lineEyetoLeftUp);
        line4->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));

        viewpointPlaneGeoset->setVertexArray(viewpointCoords.get());
        viewpointPlaneGeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 3));

        viewpointPlaneBorderGeoset->setVertexArray(viewpointBorderCoords.get());
        viewpointPlaneBorderGeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 4));
    }

    Matrix rotMat;
    Matrix transMat;
    Matrix final;

    if (hasMatrix_)
    {
        transMat.makeTranslate(-xformMat_(3, 0) / scale_, -xformMat_(3, 1) / scale_, -xformMat_(3, 2) / scale_);
        Quat quat;
        quat = xformMat_.getRotate();
        double angle, x, y, z;
        quat.getRotate(angle, x, y, z);
        quat.makeRotate(-angle, x, y, z);
        rotMat.makeRotate(quat);

        final.makeIdentity();
        final.postMult(transMat);
        final.postMult(rotMat);
        localDCS->setMatrix(final);
    }
    else
    {
        //fprintf(stderr, "ohne Matrix");
        // rotMat.makeEuler(coord.hpr[0], coord.hpr[1], coord.hpr[2]);
        MAKE_EULER_MAT(rotMat, coord.hpr[0], coord.hpr[1], coord.hpr[2])
        transMat.makeTranslate(-coord.xyz[0] / scale_, -coord.xyz[1] / scale_, -coord.xyz[2] / scale_);
        Quat quat;
        // rotMat.getOrthoQuat(quat);
        quat = rotMat.getRotate();
        double angle, x, y, z;
        quat.getRotate(angle, x, y, z);
        quat.makeRotate(-angle, x, y, z);
        rotMat.makeRotate(quat);

        final.makeIdentity();
        final.postMult(transMat);
        final.postMult(rotMat);
        localDCS->setMatrix(final);
    }

    Vec3 tangent = tangentIn;
    tangent = Matrix::transform3x3(final, tangent);
    tangent *= (1 / scale_);
    tangent += Vec3(final(3, 0), final(3, 1), final(3, 2));

    // //update text
    // myLabel->setString(name);
    // myLabel->setPosition(Vec3(600 / scale_, 0, 600 / scale_));

    // Update Interactors
    //=========================================================================================

    Matrix localMat;
    localMat = localDCS->getMatrix();

    Vec3 localVec;
    localVec = localMat.getTrans();

    viewpointInteractor->updateTransform(localMat);

    Vec3 tangentInO;
    Vec3 shiftVec = Vec3(0, 0, 0);
    if (shiftFlightpathToEyePoint)
        shiftVec = -eyepoint;

    tangentInO = Matrix::transform3x3((tangentIn - shiftVec), localMat);
    tangentInO *= (1 / scale_);
    tangentInO += localVec;

    tanInInteractor->updateTransform(tangentInO);

    Vec3 tangentOutO;
    tangentOutO = Matrix::transform3x3((tangentOut - shiftVec), localMat);
    tangentOutO *= (1 / scale_);
    tangentOutO += localVec;
    tanOutInteractor->updateTransform(tangentOutO);

    scaleVec = Vec3(800 / scale_, 0, -600 / scale_); // urspruenglich 800 x -600
    scaleVec = Matrix::transform3x3(scaleVec, localMat);
    scaleVec += localVec;

    scaleInteractor->updateTransform(scaleVec);
}

void ViewDesc::shiftFlightpath(bool state)
{
    shiftFlightpathToEyePoint = state;
    updateGeometry();
}

void ViewDesc::preFrame(FlightPathVisualizer *vpVis)
{
    if (!hasGeometry_)
        return;

    tanInInteractor->preFrame();
    tanOutInteractor->preFrame();
    viewpointInteractor->preFrame();
    scaleInteractor->preFrame();

    if (tanInInteractor->isRunning())
    {
        Matrix local;
        local = localDCS->getMatrix();
        Vec3 localVec;
        localVec = local.getTrans();
        local.invert(local);

        Vec3 objectCoord;
        objectCoord = tanInInteractor->getPos();
        tangentIn = objectCoord - localVec;
        tangentIn = Matrix::transform3x3(tangentIn, local);

        if (shiftFlightpathToEyePoint)
            tangentIn -= eyepoint / scale_;
        tangentIn *= scale_;

        // C1 continuity
        //tangentOut = -tangentIn;

        // G1 continuity
        float outLen = tangentOut.length();
        tangentOut = -tangentIn;
        tangentOut.normalize();
        tangentOut *= outLen;

        updateGeometry();
    }
    if (tanOutInteractor->isRunning())
    {
        Matrix local;
        local = localDCS->getMatrix();
        Vec3 localVec;
        localVec = local.getTrans();
        local.invert(local);

        Vec3 objectCoord;
        objectCoord = tanOutInteractor->getPos();
        tangentOut = objectCoord - localVec;
        tangentOut = Matrix::transform3x3(tangentOut, local);

        if (shiftFlightpathToEyePoint)
            tangentOut -= eyepoint / scale_;
        tangentOut *= scale_;

        // C1 continuity
        //tangentIn = -tangentOut;

        // G1 continuity
        float inLen = tangentIn.length();
        tangentIn = -tangentOut;
        tangentIn.normalize();
        tangentIn *= inLen;

        updateGeometry();
    }
    if (scaleInteractor->isRunning())
    {
        Matrix local;
        local = localDCS->getMatrix();
        Vec3 localVec;
        localVec = local.getTrans();

        Vec3 tmp = scaleInteractor->getPos();

        float length = fabs((localVec - tmp).length());

        // restrict direction of scaleInteractor:
        scaleVec = Vec3(800, 0, -600); // urspruenglich 800 x -600
        scaleVec.normalize();
        scaleVec *= length;
        scaleVec = Matrix::transform3x3(scaleVec, local);

        scaleInteractor->updateTransform(scaleVec + localVec);
        // alten scale herausrechnen | | x  | | scale 2 --> x = 200
        // coord.xyz[0] /= _scale; // | x | scale 1 --> x = 100
        // coord.xyz[1] /= _scale;
        // coord.xyz[2] /= _scale;

        xformMat_(3, 0) /= scale_;
        xformMat_(3, 1) /= scale_;
        xformMat_(3, 2) /= scale_;

        scale_ = 1000 / scaleVec.length();

        xformMat_(3, 0) *= scale_;
        xformMat_(3, 1) *= scale_;
        xformMat_(3, 2) *= scale_;

        updateGeometry();
    }

    if (viewpointInteractor->isRunning())
    {
        Matrix m = viewpointInteractor->getMatrix();
        localDCS->setMatrix(m);

        // get world coordinates of viewpoint
        m.invert(m);

        xformMat_ = m;
        xformMat_(3, 0) *= scale_;
        xformMat_(3, 1) *= scale_;
        xformMat_(3, 2) *= scale_;

        updateGeometry();
    }
    vpVis->updateDrawnCurve();
    /*
    if (tanOutInteractor->wasStopped() || tanInInteractor->wasStopped() || viewpointInteractor->wasStopped() || scaleInteractor->wasStopped())
    {
    // Send Changes to GUI
    //=========================================================================================
    if(VRCoviseConnection::covconn && coVRMSController::msController->isMaster())
    {
    char newString[1000];
    sprintf(newString, "%f %f %f %f %f %f %f %f %f %f %f %f %f", scale, coord.xyz[0],
    coord.xyz[1], coord.xyz[2], coord.hpr[0], coord.hpr[1],
    coord.hpr[2], tangentOut[0], tangentOut[1], tangentOut[2], tangentIn[0],
    tangentIn[1], tangentIn[2]);

    // send to GUI
    coGRUpdateViewpointMsg vMsg(id_,newString);

    Message grmsg;
    grmsg.type = UI;
    grmsg.data = (char *)(vMsg.c_str());
    grmsg.length = strlen(grmsg.data)+1;
    CoviseRender::appmod->send_ctl_msg(&grmsg);
    }
    updateGeometry();
    // vpVis->updateDrawnCurve();
    }
    */
}

void ViewDesc::alignViewpoint(char alignment)
{
    if (!hasGeometry_)
        return;

    osg::Matrix m;
    m = localDCS->getMatrix();

    //get the heading, pitch and roll angle of the matrix
    Quat quat;
    quat = m.getRotate();

    double qx = quat.x();
    double qy = quat.y();
    double qz = quat.z();
    double qw = quat.w();

    double sqx = qx * qx;
    double sqy = qy * qy;
    double sqz = qz * qz;
    double sqw = qw * qw;

    double term1 = 2 * (qx * qy + qw * qz);
    double term2 = sqw + sqx - sqy - sqz;
    double term3 = -2 * (qx * qz - qw * qy);
    double term4 = 2 * (qw * qx + qy * qz);
    double term5 = sqw - sqx - sqy + sqz;

    double heading = atan2(term1, term2);
    double pitch = atan2(term4, term5);
    double roll = asin(term3);

    // fprintf(stderr, "hpr_start %f %f %f\n", heading, pitch, roll);

    if (alignment == 'z')
    {
        if (cover->debugLevel(4))
            printf("Aligning Viewpoints: Heading = 0\n");

        Quat q1, q2, q3;
        //set the heading angle to zero and turn the other angles into their old position
        q1.makeRotate(0, 0, 0, 1);
        q2.makeRotate(-pitch, 1, 0, 0);
        q3.makeRotate(-roll, 0, 1, 0);
        quat = q1 * q2 * q3;

        //translate viewpoint to old position (worldcoordinates!)
        osg::Matrix transMat;
        transMat.makeTranslate(-m(3, 0) * scale_, -m(3, 1) * scale_, -m(3, 2) * scale_);

        xformMat_.makeRotate(quat);
        xformMat_.preMult(transMat);
    }
    else if (alignment == 'x')
    {
        if (cover->debugLevel(4))
            printf("Aligning Viewpoints: pitch = 0\n");

        Quat q1, q2, q3;
        //set the pitch angle to zero and turn the other angles into their old position
        q1.makeRotate(-heading, 0, 0, 1);
        q2.makeRotate(0, 1, 0, 0);
        q3.makeRotate(-roll, 0, 1, 0);
        quat = q1 * q2 * q3;

        //translate viewpoint to old position (worldcoordinates!)
        osg::Matrix transMat;
        transMat.makeTranslate(-m(3, 0) * scale_, -m(3, 1) * scale_, -m(3, 2) * scale_);

        xformMat_.makeRotate(quat);
        xformMat_.preMult(transMat);
    }
    else if (alignment == 'y')
    {
        if (cover->debugLevel(4))
            printf("Aligning Viewpoints: roll = 0\n");

        Quat q1, q2, q3;
        //set the roll angle to zero and turn the other angles into their old position
        q1.makeRotate(-heading, 0, 0, 1);
        q2.makeRotate(-pitch, 1, 0, 0);
        q3.makeRotate(0, 0, 1, 0);
        quat = q1 * q2 * q3;

        //translate viewpoint to old position (worldcoordinates!)
        osg::Matrix transMat;
        transMat.makeTranslate(-m(3, 0) * scale_, -m(3, 1) * scale_, -m(3, 2) * scale_);

        xformMat_.makeRotate(quat);
        xformMat_.preMult(transMat);
    }

    updateGeometry();
}

void ViewDesc::updateToViewAll()
{
    if (!isViewAll_)
        return;
    // on ViewAll we disable viewerPosRotation
    coVRNavigationManager::instance()->enableViewerPosRotation(false);
    // get bsphere
    osg::BoundingSphere bsphere = VRSceneGraph::instance()->getBoundingSphere();
    if (bsphere.radius() == 0.f)
        bsphere.radius() = 1.f;
    float scaleFactor = 1.f;
    osg::Matrix matrix;
    VRSceneGraph::instance()->boundingSphereToMatrices(bsphere, true, &matrix, &scaleFactor);
    // set scale
    setScale(scaleFactor);
    // set position
    osg::Matrix euler;
    MAKE_EULER_MAT(euler, -coord.hpr[0], -coord.hpr[1], -coord.hpr[2]);
    coord.xyz = euler * matrix.getTrans();
}

void ViewDesc::setFlightPathActivated(bool active)
{
    flightPathActivated = active;
}

void ViewDesc::setName(const char *n)
{
    name = n;

    if (button_)
        button_->setText(name);
    if (flightButton_)
        flightButton_->setText(name);
    if (editVPMenu_)
        editVPMenu_->setText(name);
}
