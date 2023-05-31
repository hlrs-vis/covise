/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: ConicSectionPlugin                                          **
 **              for VR4Schule mathematics                                 **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include <PluginUtil/PluginMessageTypes.h>
#include <PluginUtil/coPlane.h>
#include "cover/VRSceneGraph.h"

#include "cover/coTranslator.h"

#include "ConicSectionPlugin.h"

using namespace osg;
using namespace opencover;
using namespace covise;

ConicSectionPlugin *ConicSectionPlugin::plugin = NULL;

//
// Constructor
//
ConicSectionPlugin::ConicSectionPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nConicSectionPlugin::ConicSectionPlugin\n");
}

//
// Destructor
//
ConicSectionPlugin::~ConicSectionPlugin()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nConicSectionPlugin::~ConicSectionPlugin\n");

    topCone_->unref();
    bottomCone_->unref();
    topConeTransform_->unref();
    topConeDraw_->unref();
    bottomConeDraw_->unref();
    topConeGeode_->unref();
    bottomConeGeode_->unref();
    Cone_->unref();

    plane_->unref();
    polyNormal_->unref();
    polyCoords_->unref();
    planeGeode_->unref();
    material_->unref();
    stateSet_->unref();

    delete showClipplane_;
    delete sectionPlaneEquation_;
    delete sectionType_;
    delete sectionEquation_;
    delete conicMenu_;
}

//
// INIT
//
bool ConicSectionPlugin::init()
{
    if (plugin)
        return false;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nConicSectionPlugin::ConicSectionPlugin\n");

    // set plugin
    ConicSectionPlugin::plugin = this;

    // rotate scene
    MatrixTransform *trafo = VRSceneGraph::instance()->getTransform();
    Matrix m;
    m.makeRotate(inDegrees(-90.0), 0.0, 0.0, 1.0);
    trafo->setMatrix(m);

    makeMenu();

    // create the cones
    topCone_ = new osg::Cone(Vec3(0, 0, 0), 0.5, 1);
    bottomCone_ = new osg::Cone(Vec3(0, 0, -1.5), 0.5, 1);
    topConeTransform_ = new MatrixTransform();
    topConeDraw_ = new ShapeDrawable();
    bottomConeDraw_ = new ShapeDrawable();
    topConeGeode_ = new Geode();
    bottomConeGeode_ = new Geode();
    Cone_ = new MatrixTransform();
    Cone_->setNodeMask(Cone_->getNodeMask() & (~Isect::Intersection));

    topCone_->ref();
    bottomCone_->ref();
    topConeTransform_->ref();
    topConeDraw_->ref();
    bottomConeDraw_->ref();
    topConeGeode_->ref();
    bottomConeGeode_->ref();
    Cone_->ref();

    // rotate the cone on top
    Matrix mCone;
    mCone.makeRotate(inDegrees(180.0), 1, 0, 0);
    topConeTransform_->setMatrix(mCone);

    topConeDraw_->setShape(topCone_.get());
    bottomConeDraw_->setShape(bottomCone_.get());
    topConeGeode_->addDrawable(topConeDraw_.get());
    bottomConeGeode_->addDrawable(bottomConeDraw_.get());
    topConeTransform_->addChild(topConeGeode_.get());
    Cone_->addChild(topConeTransform_.get());
    Cone_->addChild(bottomConeGeode_.get());

    cover->getObjectsRoot()->addChild(Cone_.get());

    // call viewall
    VRSceneGraph::instance()->viewAll();

    // geode for plane
    plane_ = new Geometry();
    plane_->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 6));
    plane_->setNormalBinding(Geometry::BIND_PER_VERTEX);
    plane_->setUseDisplayList(false);
    plane_->ref();

    planeGeode_ = new Geode();
    planeGeode_->ref();
    planeGeode_->addDrawable(plane_.get());
    planeGeode_->setNodeMask(planeGeode_->getNodeMask() & (Isect::Visible) & (~Isect::Intersection) & (~Isect::Pick));
    material_ = new Material();
    material_->ref();
    material_->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.545, 0.0, 0.545, 0.5));
    material_->setAmbient(Material::FRONT_AND_BACK, Vec4(0.545, 0.0, 0.545, 0.5));
    stateSet_ = VRSceneGraph::instance()->loadDefaultGeostate();
    stateSet_->ref();
    stateSet_->setMode(GL_BLEND, StateAttribute::ON);
    stateSet_->setAttributeAndModes(material_.get());
    stateSet_->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    planeGeode_->setStateSet(stateSet_.get());
    //cover->getObjectsRoot()->addChild(planeGeode_.get());

    oldPlane = Vec4(0, 0, 0, 0);

    // init plane and coordinates
    helperPlane_ = new coPlane(Vec3(0, 1, 0), Vec3(0, 0, 0));

    polyCoords_ = new osg::Vec3Array(6);
    polyCoords_->ref();
    (*polyCoords_)[0].set(-0.05, 0.0, -0.05);
    (*polyCoords_)[1].set(0.05, 0.0, -0.05);
    (*polyCoords_)[2].set(0.05, 0.0, 0.05);
    (*polyCoords_)[3].set(-0.05, 0.0, 0.05);
    (*polyCoords_)[4].set(-0.05, 0.0, -0.05);
    (*polyCoords_)[5].set(-0.05, 0.0, -0.05);
    plane_->setVertexArray(polyCoords_.get());

    polyNormal_ = new osg::Vec3Array(6);
    polyNormal_->ref();
    for (int i = 0; i < 6; i++)
    {
        (*polyNormal_)[i].set(0, 1, 0);
    }
    plane_->setNormalArray(polyNormal_.get());

    return true;
}

void ConicSectionPlugin::preFrame()
{
    // get ClipPlane No 0 and set equation to label
    ClipNode *clipNode = cover->getObjectsRoot();
    if (clipNode->getNumClipPlanes() > 0)
    {
        // get equation of section plane
        Vec4 v = clipNode->getClipPlane(0)->getClipPlane();
        // draw the plane
        drawPlane(v);
        if (oldPlane == v)
            return;
        stringstream plane;

        // print equation of plane in label
        plane << coTranslator::coTranslate("Ebene: ");

        plane << std::fixed << std::setprecision(2);

        //not translating mathematic equations
        plane << v.x() << "x ";
        if (v.y() > 0)
            plane << "+" << v.y() << "y ";
        else if (v.y() < 0)
            plane << v.y() << "y ";
        if (v.z() > 0)
            plane << "+" << v.z() << "z ";
        else if (v.z() < 0)
            plane << v.z() << "z ";
        if (v.w() > 0)
            plane << "+ " << v.w();
        else if (v.w() < 0)
            plane << v.w();
        plane << " = 0";
        sectionPlaneEquation_->setLabel(plane.str().c_str());

        // Equation of Cone and type of section
        std::string secType = calculateSection(v);
        if (secType == std::string("No Section"))
        {
            sectionEquation_->setLabel(sectionString(v));
        }
        else
        {
            sectionEquation_->setLabel(coTranslator::coTranslate("Gleichung: --- "));
        }
        sectionType_->setLabel(secType);

        oldPlane = v;
    }
    else
    {
        sectionPlaneEquation_->setLabel(coTranslator::coTranslate("Ebene: --- "));
        sectionType_->setLabel(coTranslator::coTranslate("Kegelschnitt: --- "));
        sectionEquation_->setLabel(coTranslator::coTranslate("Gleichung: --- "));
    }
}

void ConicSectionPlugin::makeMenu()
{
    Matrix dcsTransMat, dcsMat, preRot, preScale, tmp;
    OSGVruiMatrix menuMatrix;

    string text(coTranslator::coTranslate("Kegelschnitte"));

    conicMenu_ = new coRowMenu(text.c_str());
    conicMenu_->setVisible(true);
    conicMenu_->setAttachment(coUIElement::RIGHT);

    //position the menu
    //position the menu
    double px = (double)coCoviseConfig::getFloat("x", "COVER.Menu.Position", -1000);
    double py = (double)coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0);
    double pz = (double)coCoviseConfig::getFloat("z", "COVER.Menu.Position", 600);

    px = (double)coCoviseConfig::getFloat("x", "COVER.Plugin.ReadCollada.MenuPosition", px);
    py = (double)coCoviseConfig::getFloat("y", "COVER.Plugin.ReadCollada.MenuPosition", py);
    pz = (double)coCoviseConfig::getFloat("z", "COVER.Plugin.ReadCollada.MenuPosition", pz);

    // default is Mathematic.MenuSize then COVER.Menu.Size then 1.0
    float s = coCoviseConfig::getFloat("value", "COVER.Menu.Size", 1.0);
    s = coCoviseConfig::getFloat("value", "COVER.Plugin.ConicSection.MenuSize", s);

    preRot.makeRotate(inDegrees(90.0), 1.0, 0.0, 0.0);
    dcsTransMat.makeTranslate(px, py, pz);
    preScale.makeScale(s, s, s);

    tmp.mult(preScale, preRot);
    dcsMat.mult(tmp, dcsTransMat);

    menuMatrix.setMatrix(dcsMat);
    conicMenu_->setTransformMatrix(&menuMatrix);
    conicMenu_->setScale(cover->getSceneSize() / 2500);

    // menu items
    // enable and disable section plane
    text = coTranslator::coTranslate("Schnittebene");
    showClipplane_ = new coCheckboxMenuItem(text.c_str(), false);
    showClipplane_->setMenuListener(this);
    conicMenu_->add(showClipplane_);

    // label for section plane equation
    text = coTranslator::coTranslate("Ebene: --- ");
    sectionPlaneEquation_ = new coLabelMenuItem(text.c_str());
    conicMenu_->add(sectionPlaneEquation_);

    // label for section plane equation
    text = coTranslator::coTranslate("Kegelschnitt: --- ");
    sectionType_ = new coLabelMenuItem(text.c_str());
    conicMenu_->add(sectionType_);

    // label for section plane equation
    text = coTranslator::coTranslate("Gleichung: --- ");
    sectionEquation_ = new coLabelMenuItem(text.c_str());
    conicMenu_->add(sectionEquation_);

    conicMenu_->show();
}

/**
  * checks if a float is null 
  * numbers near null are also null
  */
bool ConicSectionPlugin::isNull(float f)
{
    if (f < 0.001 && f > -0.001)
        return true;
    return false;
}

/**
  * Calculate what the section is and return as string
  */
std::string ConicSectionPlugin::calculateSection(osg::Vec4 eq)
{
    double a, b, h;
    if (!isNull(eq.z()))
    {
        double z2 = eq.z() * eq.z();
        a = 1 - (eq.x() * eq.x() / z2);
        b = 1 - (eq.y() * eq.y() / z2);
        h = eq.x() * eq.y() / z2;
    }
    else if (!isNull(eq.y()))
    {
        double y2 = eq.y() * eq.y();
        a = 1 - (eq.x() * eq.x() / y2);
        b = 1 - (eq.z() * eq.z() / y2);
        h = eq.x() * eq.z() / y2;
    }
    else if (!isNull(eq.x()))
    {
        double x2 = eq.x() * eq.x();
        a = 1 - (eq.y() * eq.y() / x2);
        b = 1 - (eq.z() * eq.z() / x2);
        h = eq.y() * eq.z() / x2;
    }
    else
    {
        return coTranslator::coTranslate("Kegelschnitt: --- ");
    }

    double ab = a * b;
    double h2 = h * h;

    if (a == b && h == 0)
    {
        return coTranslator::coTranslate("Kegelschnitt: Kreis ");
    }
    if ((a == 0 && b != 0) || (a != 0 && b == 0))
    {
        return coTranslator::coTranslate("Kegelschnitt: Parabel / Hyperbel");
    }
    if (h2 < ab)
    {
        return coTranslator::coTranslate("Kegelschnitt: Ellipse");
    }
    if (h2 > ab)
    {
        // check if it only intersects one cone
        //       if (numIntersectTop <= 3 || numIntersectBottom <= 3)
        //          return "Section: Parabola";
        return coTranslator::coTranslate("Kegelschnitt: Parabel / Hyperbel");
    }

    return coTranslator::coTranslate("Kegelschnitt: --- ");
}

/**
  * calculate the equation of the section and return as string
  */

std::string ConicSectionPlugin::sectionString(osg::Vec4 eq)
{
    stringstream equation;
    equation << coTranslator::coTranslate("Gleichung: ");
    equation << std::fixed << std::setprecision(2);

    double ax, bx, cx, dx, ex, fx;

    if (!isNull(eq.z()))
    {
        double z2 = eq.z() * eq.z();
        ax = 1 - ((eq.x() * eq.x()) / z2);
        bx = 1 - (eq.y() * eq.y() / z2);
        cx = 2 * eq.x() * eq.y() / z2;
        dx = 2 * eq.x() * eq.w() / z2;
        ex = 2 * eq.y() * eq.w() / z2;
        fx = eq.w() * eq.w() / z2;

        if (!isNull(ax))
            equation << ax << "x^2 ";
        if (bx > 0)
            equation << "+" << bx << "y^2 ";
        else if (bx < 0)
            equation << bx << "y^2 ";
        if (cx > 0)
            equation << "+" << cx << "xy ";
        else if (cx < 0)
            equation << cx << "xy ";
        if (dx > 0)
            equation << "+" << dx << "x ";
        else if (dx < 0)
            equation << dx << "x ";
        if (ex > 0)
            equation << "+" << ex << "y ";
        else if (ex < 0)
            equation << ex << "y ";
        if (fx > 0)
            equation << "+" << fx << " = 0";
        else if (fx < 0)
            equation << fx << " = 0 ";
        else
            equation << " = 0 ";
    }
    else if (!isNull(eq.x()))
    {
        double a2 = eq.x() * eq.x();
        ax = 1 + ((eq.y() * eq.y()) / a2);
        bx = (eq.z() * eq.z() / a2) - 1;
        cx = 2 * eq.y() * eq.z() / a2;
        dx = 2 * eq.y() * eq.w() / a2;
        ex = 2 * eq.z() * eq.w() / a2;
        fx = eq.w() * eq.w() / a2;

        if (!isNull(ax))
            equation << ax << "y^2 ";
        if (bx > 0)
            equation << "+" << bx << "z^2 ";
        else if (bx < 0)
            equation << bx << "z^2 ";
        if (cx > 0)
            equation << "+" << cx << "yz ";
        else if (cx < 0)
            equation << cx << "yz ";
        if (dx > 0)
            equation << "+" << dx << "y ";
        else if (dx < 0)
            equation << dx << "y ";
        if (ex > 0)
            equation << "+" << ex << "z ";
        else if (ex < 0)
            equation << ex << "z ";
        if (fx > 0)
            equation << "+" << fx << " = 0";
        else if (fx < 0)
            equation << fx << " = 0 ";
        else
            equation << " = 0 ";
    }
    else if (!isNull(eq.y()))
    {
        double b2 = eq.y() * eq.y();
        ax = 1 + ((eq.x() * eq.x()) / b2);
        bx = (eq.z() * eq.z() / b2) - 1;
        cx = 2 * eq.x() * eq.z() / b2;
        dx = 2 * eq.x() * eq.w() / b2;
        ex = 2 * eq.z() * eq.w() / b2;
        fx = eq.w() * eq.w() / b2;

        if (!isNull(ax))
            equation << ax << "x^2 ";
        if (bx > 0)
            equation << "+" << bx << "z^2 ";
        else if (bx < 0)
            equation << bx << "z^2 ";
        if (cx > 0)
            equation << "+" << cx << "xz ";
        else if (cx < 0)
            equation << cx << "xz ";
        if (dx > 0)
            equation << "+" << dx << "x ";
        else if (dx < 0)
            equation << dx << "x ";
        if (ex > 0)
            equation << "+" << ex << "z ";
        else if (ex < 0)
            equation << ex << "z ";
        if (fx > 0)
            equation << "+" << fx << " = 0";
        else if (fx < 0)
            equation << fx << " = 0 ";
        else
            equation << " = 0 ";
    }

    return equation.str();
}

/**
 * show or hide sectionPlane (Clipplane)
 */
void ConicSectionPlugin::menuEvent(coMenuItem *menuItem)
{
    // send message to clipplane plugin to show or hide section plane
    if (menuItem == showClipplane_)
    {
        if (showClipplane_->getState())
        {
            cover->getObjectsRoot()->addChild(planeGeode_);
            cover->sendMessage(this, "ClipPlane", PluginMessageTypes::ClipPlaneMessage, 12, "enablePick 1");
        }
        else
        {
            cover->getObjectsRoot()->removeChild(planeGeode_);
            cover->sendMessage(this, "ClipPlane", PluginMessageTypes::ClipPlaneMessage, 13, "disablePick 1");
        }
    }
}

void ConicSectionPlugin::drawPlane(osg::Vec4 eq)
{
    //if (showClipplane_->getState())
    //{
    //helper plane
    Vec3 normal = Vec3(eq.x(), eq.y(), eq.z());
    normal.normalize();
    Vec3 point = normal * -eq.w();
    point = point + normal * 0.003;
    helperPlane_->update(normal, point);
    Vec3 intersect[6];
    osg::BoundingBox bboxCompl = cover->getBBox(Cone_.get());
    int numIntersect = helperPlane_->getBoxIntersectionPoints(bboxCompl, intersect);

    if (numIntersect > 0)
    {
        for (int i = 0; i < numIntersect; i++)
        {
            (*polyCoords_)[i] = intersect[i];
        }
        for (int i = numIntersect; i < 6; i++)
        {

            (*polyCoords_)[i] = (*polyCoords_)[numIntersect - 1];
        }
    }

    (*polyNormal_)[0].set(normal[0], normal[1], normal[2]);

    plane_->dirtyBound();
    //}
}

COVERPLUGIN(ConicSectionPlugin)
