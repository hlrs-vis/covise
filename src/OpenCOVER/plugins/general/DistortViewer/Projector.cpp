/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Projector.h"
#include "Scene.h"
#include "VisScene.h"
#include "XmlTools.h"
#include "HelpFuncs.h"

#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
using namespace covise;
using namespace opencover;

int Projector::num = 0;

Projector::Projector(bool load)
    : active(false)
    , stateFrust(true)
    , projRatio(1.0f)
    , aspectRw(16.0f)
    , aspectRh(9.0f)
    , near_c(0.1f)
    , far_c(10000.0f)
    , shiftx(0.0f)
    , shifty(0.0f)
    , position(osg::Vec3(0.0f, 0.0f, 0.0f))
    , projDirection(osg::Vec3(0.0f, 1.0f, 0.0f))
    , upDirection(osg::Vec3(0.0f, 0.0f, 1.0f))
    , autoCalc(true)
{
    //Gesmtzahl der Projektoren
    num++;
    setProjectorNum(num);

    //ggf. von XML laden
    if (load)
        loadFromXML();

    //Scene zur Visualisierung der Distortion erstellen
    visScene = new VisScene(this, load);

    //virt Proj. Screen berechnen
    projScreenPlane = calcScreenPlane();
    calcScreen();
}

Projector::~Projector(void)
{
    num--;

    delete visScene;
}

bool Projector::loadFromXML()
{
    std::string section;
    std::string var_str;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + ".Proj" + projNum_str + ".";
    //Projector
    section = "Projectors";
    position = XmlTools::getInstance()->loadVec3(path + section + ".PostionVec3", osg::Vec3(0.0f, 0.0f, 0.0f));
    projDirection = XmlTools::getInstance()->loadVec3(path + section + ".OriginVec3", osg::Vec3(0.0f, 1.0f, 0.0f));
    upDirection = XmlTools::getInstance()->loadVec3(path + section + ".RotationVec3", osg::Vec3(0.0f, 0.0f, 1.0f));
    aspectRh = XmlTools::getInstance()->loadFloatValue(path + section + ".AspectRatio", "Height", 9.0f);
    aspectRw = XmlTools::getInstance()->loadFloatValue(path + section + ".AspectRatio", "Width", 16.0f);
    projRatio = XmlTools::getInstance()->loadFloatValue(path + section, "ProjectRatio", 1.0f);
    shiftx = XmlTools::getInstance()->loadFloatValue(path + section + ".LensShift", "Horizontal");
    shifty = XmlTools::getInstance()->loadFloatValue(path + section + ".LensShift", "Vertical");
    far_c = XmlTools::getInstance()->loadFloatValue(path + section + ".Clipping", "Far", 0.1f);
    near_c = XmlTools::getInstance()->loadFloatValue(path + section + ".Clipping", "Near", 10000.0f);
    stateFrust = XmlTools::getInstance()->loadBoolValue(path + section, "FrustChk", true);

    //Screen
    section = "Screens";
    autoCalc = XmlTools::getInstance()->loadBoolValue(path + section, "AutoCalc", true);
    projScreenPlane = osg::Plane(XmlTools::getInstance()->loadVec4(path + section + ".PlaneVec4"));
    hprVec = XmlTools::getInstance()->loadVec3(path + section + ".EulerVec3");
    projScreenCenter = XmlTools::getInstance()->loadVec3(path + section + ".PositionVec3");
    projScreenHeight = XmlTools::getInstance()->loadFloatValue(path + section + ".Size", "Height");
    projScreenWidth = XmlTools::getInstance()->loadFloatValue(path + section + ".Size", "Width");

    return true;
}

bool Projector::saveToXML()
{
    std::string section;
    std::string var_str;
    std::string plugPath = XmlTools::getInstance()->getPlugPath();
    std::string path = plugPath + ".Proj" + projNum_str + ".";

    //Projector
    section = "Projector";
    XmlTools::getInstance()->saveVec3(position, path + section + ".PostionVec3");
    XmlTools::getInstance()->saveVec3(projDirection, path + section + ".OriginVec3");
    XmlTools::getInstance()->saveVec3(upDirection, path + section + ".RotationVec3");
    XmlTools::getInstance()->saveFloatValue(aspectRh, path + section + ".AspectRatio", "Height");
    XmlTools::getInstance()->saveFloatValue(aspectRw, path + section + ".AspectRatio", "Width");
    XmlTools::getInstance()->saveFloatValue(projRatio, path + section, "ProjectRatio");
    XmlTools::getInstance()->saveFloatValue(shiftx, path + section + ".LensShift", "Horizontal");
    XmlTools::getInstance()->saveFloatValue(shifty, path + section + ".LensShift", "Vertical");
    XmlTools::getInstance()->saveFloatValue(far_c, path + section + ".Clipping", "Far");
    XmlTools::getInstance()->saveFloatValue(near_c, path + section + ".Clipping", "Near");
    XmlTools::getInstance()->saveBoolValue(stateFrust, path + section, "FrustChk");

    //Screen -> in Unter ScreenConfig
    section = "Screens";
    XmlTools::getInstance()->saveBoolValue(autoCalc, path + section, "AutoCalc");
    XmlTools::getInstance()->saveVec4(projScreenPlane.asVec4(), path + section + ".PlaneVec4");
    XmlTools::getInstance()->saveVec3(hprVec, path + section + ".EulerVec3");
    XmlTools::getInstance()->saveVec3(projScreenCenter, path + section + ".PositionVec3");
    XmlTools::getInstance()->saveFloatValue(projScreenHeight, path + section + ".Size", "Height");
    XmlTools::getInstance()->saveFloatValue(projScreenWidth, path + section + ".Size", "Width");

    //-> Werte zum Auslesen von Opencover in Cover.ScreenConfig
    XmlTools::getInstance()->saveStrValue(projNum_str, "COVER.ScreenConfig.Screen", "name");
    XmlTools::getInstance()->saveStrValue(projNum_str, "COVER.ScreenConfig.Screen:" + projNum_str, "screen");
    XmlTools::getInstance()->saveStrValue("Projector" + projNum_str, "COVER.ScreenConfig.Screen:" + projNum_str, "comment");
    XmlTools::getInstance()->saveFloatValue(hprVec.x(), "COVER.ScreenConfig.Screen:" + projNum_str, "h");
    XmlTools::getInstance()->saveFloatValue(hprVec.y(), "COVER.ScreenConfig.Screen:" + projNum_str, "p");
    XmlTools::getInstance()->saveFloatValue(hprVec.z(), "COVER.ScreenConfig.Screen:" + projNum_str, "r");
    XmlTools::getInstance()->saveIntValue(projScreenHeight, "COVER.ScreenConfig.Screen:" + projNum_str, "height");
    XmlTools::getInstance()->saveIntValue(projScreenWidth, "COVER.ScreenConfig.Screen:" + projNum_str, "width");
    XmlTools::getInstance()->saveIntValue(projScreenCenter.x(), "COVER.ScreenConfig.Screen:" + projNum_str, "originX");
    XmlTools::getInstance()->saveIntValue(projScreenCenter.y(), "COVER.ScreenConfig.Screen:" + projNum_str, "originY");
    XmlTools::getInstance()->saveIntValue(projScreenCenter.z(), "COVER.ScreenConfig.Screen:" + projNum_str, "originZ");

    //Einträge in xml schreiben
    XmlTools::getInstance()->saveToXml();

    visScene->saveToXML();

    return true;
}

void Projector::setProjectorNum(int new_projNum)
{
    projNum = new_projNum;
    HelpFuncs::IntToString(projNum, projNum_str); //Integer to String
}

void Projector::setProjRatio(float ratio)
{
    projRatio = ratio;
    update();
}

float Projector::getFovY()
{
    //Seitenlängen des Projektor-Frustums im Abstand der near-Clippingebene
    osg::Vec4 frustNear = getFrustumSizeNear();

    float fovy = atan(frustNear.z() / near_c) + atan(frustNear.w() / near_c);
    return fovy;
}

float Projector::getFovX()
{
    //Seitenlängen des Projektor-Frustums im Abstand der near-Clippingebene
    osg::Vec4 frustNear = getFrustumSizeNear();

    float fovx = atan(frustNear.x() / near_c) + atan(frustNear.y() / near_c);
    return fovx;
}

void Projector::setAspectRatioH(float aspectRatio)
{
    aspectRh = aspectRatio;
    update();
}

void Projector::setAspectRatioW(float aspectRatio)
{
    aspectRw = aspectRatio;
    update();
}

void Projector::setLensShiftH(float lensShift)
{
    shiftx = lensShift;
    update();
}

void Projector::setLensShiftV(float lensShift)
{
    shifty = lensShift;
    update();
}

void Projector::setNearClipping(float nearClipping)
{
    near_c = nearClipping;
    update();
}

void Projector::setFarClipping(float farClipping)
{
    far_c = farClipping;
    update();
}

void Projector::setStateFrust(bool new_state)
{
    stateFrust = new_state;
}

osg::Matrix Projector::getTransMat()
{
    osg::Matrix transMat;
    transMat = osg::Matrix::inverse(getViewMat());
    return transMat;
}

osg::Matrix Projector::getProjMat()
{
    //Definiert über Seitenlängen des Projektor-Frustums im Abstand der near-Clippingebene
    osg::Vec4 frustNear = getFrustumSizeNear();

    osg::Matrix pProjMat;
    pProjMat.makeFrustum(-frustNear.y(), frustNear.x(), -frustNear.w(), frustNear.z(), near_c, far_c);
    return pProjMat;
}

osg::Matrix Projector::getViewMat()
{
    osg::Matrix viewMat;
    viewMat.makeLookAt(position, projDirection, upDirection);
    return viewMat;
}

void Projector::setPosition(osg::Vec3 pos)
{
    position = pos;
    update();
}

void Projector::setProjDirection(osg::Vec3 projDir)
{
    projDirection = projDir;
    update();
}

void Projector::setUpDirection(osg::Vec3 upDir)
{
    upDirection = upDir;
    update();
}

void Projector::rotate(osg::Matrix rotMat)
{
    projDirection = rotMat * projDirection;
    upDirection = rotMat * upDirection;
    update();
}

void Projector::translate(osg::Vec3f translation)
{
    position += translation;
    update();
}

osg::Vec4 Projector::getFrustumSizeNear()
{
    float b_left = (near_c / projRatio) * (1.0f - shiftx);
    float b_right = (near_c / projRatio) * (1.0f + shiftx);
    float h_top = (near_c / (projRatio * (aspectRw / aspectRh))) * (1.0f + shifty);
    float h_bottom = (near_c / (projRatio * (aspectRw / aspectRh))) * (1.0f - shifty);

    return osg::Vec4(b_right, b_left, h_top, h_bottom);
}

osg::Vec4 Projector::getFrustumSizeFar()
{
    float b_left = (far_c / projRatio) * (1.0f - shiftx);
    float b_right = (far_c / projRatio) * (1.0f + shiftx);
    float h_top = (far_c / (projRatio * (aspectRw / aspectRh))) * (1.0f + shifty);
    float h_bottom = (far_c / (projRatio * (aspectRw / aspectRh))) * (1.0f - shifty);

    return osg::Vec4(b_right, b_left, h_top, h_bottom);
}

osg::Geometry *Projector::drawFrustum()
{
    osg::ref_ptr<osg::Geometry> frustumGeom = new osg::Geometry();

    osg::ref_ptr<osg::StateSet> stateSet = new osg::StateSet();
    frustumGeom->setStateSet(stateSet.get());

    //Linienstärke des Frustums setzen
    osg::ref_ptr<osg::LineWidth> lineWidth = new osg::LineWidth();
    lineWidth->setWidth(4.0f);
    stateSet->setAttributeAndModes(lineWidth.get(), osg::StateAttribute::ON);

    //Seitenlängen des Projektor-Frustums im Abstand der near-Clippingebene
    osg::Vec4 frustNear = getFrustumSizeNear();

    //und im Abstand der far-Clippingebene
    osg::Vec4 frustFar = getFrustumSizeFar();

    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array(9);
    (*vertices)[0].set(0.0f, 0.0f, 0.0f);
    //Ecken near
    (*vertices)[1].set(-frustNear.y(), frustNear.z(), -near_c); //left,top
    (*vertices)[2].set(frustNear.x(), frustNear.z(), -near_c); //right,top
    (*vertices)[3].set(frustNear.x(), -frustNear.w(), -near_c); //right,bottom
    (*vertices)[4].set(-frustNear.y(), -frustNear.w(), -near_c); //left,bottom
    //Ecken far
    (*vertices)[5].set(-frustFar.y(), frustFar.z(), -far_c); //left,top
    (*vertices)[6].set(frustFar.x(), frustFar.z(), -far_c); //right,top
    (*vertices)[7].set(frustFar.x(), -frustFar.w(), -far_c); //right,bottom
    (*vertices)[8].set(-frustFar.y(), -frustFar.w(), -far_c); //left,bottom
    frustumGeom->setVertexArray(vertices.get());

    osg::ref_ptr<osg::DrawElementsUInt> indices = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 32);
    //US bis near
    (*indices)[0] = 0;
    (*indices)[1] = 1;
    (*indices)[2] = 0;
    (*indices)[3] = 2;
    (*indices)[4] = 0;
    (*indices)[5] = 3;
    (*indices)[6] = 0;
    (*indices)[7] = 4;
    //near-Rahmen
    (*indices)[8] = 1;
    (*indices)[9] = 2;
    (*indices)[10] = 2;
    (*indices)[11] = 3;
    (*indices)[12] = 3;
    (*indices)[13] = 4;
    (*indices)[14] = 4;
    (*indices)[15] = 1;
    //near bis far
    (*indices)[16] = 1;
    (*indices)[17] = 5;
    (*indices)[18] = 2;
    (*indices)[19] = 6;
    (*indices)[20] = 3;
    (*indices)[21] = 7;
    (*indices)[22] = 4;
    (*indices)[23] = 8;
    //far-Rahmen
    (*indices)[24] = 5;
    (*indices)[25] = 6;
    (*indices)[26] = 6;
    (*indices)[27] = 7;
    (*indices)[28] = 7;
    (*indices)[29] = 8;
    (*indices)[30] = 8;
    (*indices)[31] = 5;
    frustumGeom->addPrimitiveSet(indices.get());

    //Farbe festlegen
    osg::ref_ptr<osg::Vec4Array> color = new osg::Vec4Array();
    color->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 0.7f));
    frustumGeom->setColorArray(color.get());
    frustumGeom->setColorBinding(osg::Geometry::BIND_OVERALL);

    //normalen automatisch berechnen
    osgUtil::SmoothingVisitor::smooth(*frustumGeom);

    return frustumGeom.release();
}

osg::Group *Projector::draw()
{
    osg::Vec4 color; //Alpha-Wert der dargestellten Geometrien

    //Wenn Projektor im editor aktiv, dann hervorheben
    if (active)
        color = osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f);
    else
        color = osg::Vec4(0.0f, 1.0f, 0.0f, 0.5f);

    //virt. Proj-Screen in Endlage
    osg::ref_ptr<osg::Geode> geodeScreen = new osg::Geode();
    geodeScreen->setName("virtProjScreen");
    geodeScreen->addDrawable(drawScreen(color));

    //Geometrie-Node der später Transformiert wird
    osg::ref_ptr<osg::Geode> geodeProjector = new osg::Geode();
    geodeProjector->setName("ProjectorWithFrust");
    geodeProjector->addDrawable(drawFrustum());

    //geodeTrans von Standardlage in Blickrichtung transformieren
    osg::ref_ptr<osg::MatrixTransform> transProjector = new osg::MatrixTransform();
    transProjector->setMatrix(getTransMat());
    transProjector->addChild(geodeProjector.get());

    osg::ref_ptr<osg::Group> groupProjector = new osg::Group();
    groupProjector->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    groupProjector->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
    groupProjector->addChild(geodeScreen.get());
    if (active)
        groupProjector->addChild(transProjector.get());

    return groupProjector.release();
}

osg::Camera *Projector::getProjCam()
{
    //---------------
    // Projektorkamera erstellen
    osg::ref_ptr<osg::Camera> projCam = new osg::Camera();

    //Lage und Projektion festlegen
    projCam->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    projCam->setViewMatrix(getViewMat());
    projCam->setProjectionMatrix(getProjMat());

    // set up the background color and clear mask.
    projCam->setClearColor(osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    projCam->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set up projection.
    projCam->setComputeNearFarMode(osg::Camera::DO_NOT_COMPUTE_NEAR_FAR);

    // Camera vor Haupt-Kamera des viewers rendern lassen
    projCam->setRenderOrder(osg::Camera::PRE_RENDER);

    return projCam.release();
}

//---------------------------------------
// PROJEKTOR SCREEN
//---------------------------------------

void Projector::setAutoCalc(bool enabled)
{
    autoCalc = enabled;
}

void Projector::setScreenPlane(osg::Plane plane)
{
    projScreenPlane = plane;
    update();
}

void Projector::setScreenCenter(osg::Vec3 centerPos)
{
    projScreenCenter = centerPos;
}

void Projector::setScreenHeight(float height)
{
    projScreenHeight = height;
}

void Projector::setScreenWidth(float width)
{
    projScreenWidth = width;
}

osg::Matrix Projector::getScreenTransMat()
{
    //Matrix die Transofrmation des Screens vom US zur endgültigen Lage beschreibt
    osg::Matrix sTransMat;

    //bestimme neues Koordinatensystem
    osg::Vec3 new_z = projScreenPlane.getNormal(); //normale der projScreen-Ebene neue z-Achse
    osg::Vec3 new_x = (*getPlaneIntersecPnts())[1] - (*getPlaneIntersecPnts())[0]; //Vektor der Durchstoßpunkte des Frustums (oben)
    osg::Vec3 new_y = new_z ^ new_x; //Kreuzprodukt -> Vektor der Durchstoßpunkte des Frustums (linke Seite)

    //Vektoren normieren, damit daraus Rotationsmatrix erstellt werden kann
    new_z.normalize();
    new_y.normalize();
    new_x.normalize();

    //Rotations, bzw. Drehmatrix vom (alten) WeltKS ins neue KS
    osg::Matrix rotMat = osg::Matrix(new_x.x(), new_x.y(), new_x.z(), 0.0f,
                                     new_y.x(), new_y.y(), new_y.z(), 0.0f,
                                     new_z.x(), new_z.y(), new_z.z(), 0.0f,
                                     0.0f, 0.0f, 0.0f, 1.0f);

    //TransformationsMatrix des virt. ProjScreens
    sTransMat = rotMat * osg::Matrix::translate(projScreenCenter); //Verschiebung US-> neue Screenmitte

    return sTransMat;
}

osg::Vec3Array *Projector::getFarCorners()
{
    //Vektoren vom US des Projektor-Frustums zu den Eckpunkten im Abstand der far-clipping Ebene
    //-----------------------------------------------------------------

    //Seitenlängen des Projektor-Frustums im Abstand der far-Clippingebene
    osg::Vec4 frustFar = getFrustumSizeFar();

    osg::ref_ptr<osg::Vec3Array> cornersFar = new osg::Vec3Array;
    cornersFar->push_back(osg::Vec3(-frustFar.y(), frustFar.z(), -far_c) * getTransMat()); //links-oben
    cornersFar->push_back(osg::Vec3(frustFar.x(), frustFar.z(), -far_c) * getTransMat()); //rechts-oben
    cornersFar->push_back(osg::Vec3(frustFar.x(), -frustFar.w(), -far_c) * getTransMat()); //rechts-unten
    cornersFar->push_back(osg::Vec3(-frustFar.y(), -frustFar.w(), -far_c) * getTransMat()); //links-unten

    return cornersFar.release();
}

osg::Vec3Array *Projector::getScreenIntersecPnts()
{
    //Array mit Koordinaten der Schnittpunkte
    osg::ref_ptr<osg::Vec3Array> intersections = new osg::Vec3Array;

    //der vier Kanten der Sichtpyramide
    osg::ref_ptr<osg::Vec3Array> cornersFar = new osg::Vec3Array;
    cornersFar = getFarCorners();
    for (unsigned int i = 0; i < (cornersFar.get()->size()); i++)
    {
        osg::Vec3 corner = (*cornersFar)[i];

        //Geradenabschnitt und Geometrie, die gegeneinander auf Schnittpunkte hin geprüft werden
        osg::ref_ptr<osgUtil::LineSegmentIntersector> lineSeg = new osgUtil::LineSegmentIntersector(corner, getPosition());
        osgUtil::IntersectionVisitor findIntersections(lineSeg.get());
        osg::ref_ptr<osg::MatrixTransform> screenGeo = Scene::getScreen()->draw(false);
        screenGeo->accept(findIntersections);
        osgUtil::LineSegmentIntersector::Intersection currHit = lineSeg->getFirstIntersection();

        // Schnittpunkt
        osg::Vec3d intersecPnt = currHit.getWorldIntersectPoint();
        intersections->push_back(intersecPnt);
    }
    return intersections.release(); //(li-o,re-o,re-u,li-u)
}

osg::Plane Projector::calcScreenPlane()
{
    //Hole Schnittpunkte des Frustums und der ProjGeometrie
    osg::ref_ptr<osg::Vec3Array> intersecPnts = getScreenIntersecPnts();

    //Berechne ABstand zwischen Schnittpunkt und Betrachter
    std::vector<float> distance;
    for (unsigned int i = 0; i < (intersecPnts.get()->size()); i++)
    {
        //Abstand der Schnittpunkte zum Betrachter
        float dist = osg::Vec3((*intersecPnts)[i] - getPosition()).length();
        distance.push_back(dist);
    }

    //Suche die 3 Nächsten Punkte zum Betrachter
    //-> lösche größte Punkt mit größter Distanz aus Array
    float largest = distance[0];
    unsigned int largest_index = 0;
    osg::ref_ptr<osg::Vec3Array> planePnts = new osg::Vec3Array;

    for (unsigned int j = 1; j < (distance.size()); j++)
    {
        if (distance[j] > largest)
        {
            planePnts->push_back((*intersecPnts)[largest_index]);
            largest = distance[j];
            largest_index = j;
        }
        else
        {
            planePnts->push_back((*intersecPnts)[j]);
        }
    }

    //Spanne Ebene zwischen den drei nächsten Punkten zum Betrachter auf
    osg::Plane intersecPlane;
    intersecPlane = osg::Plane((*planePnts)[0], (*planePnts)[1], (*planePnts)[2]);

    return intersecPlane;
}

osg::Vec3Array *Projector::getPlaneIntersecPnts()
{
    //Lage des Ursprungs der Sichtpyramide
    osg::Vec3 origin = getPosition();

    //Ebene (Ax + By +Cz +D =0) des Screens als Vektor (A,B,C,D)
    osg::Vec4 plane = projScreenPlane.asVec4();

    //Eckpunkte der Sichtpyramide in far-Clippingebene
    osg::ref_ptr<osg::Vec3Array> cornersFar = new osg::Vec3Array;
    cornersFar = getFarCorners();

    osg::ref_ptr<osg::Vec3Array> intersecPoints = new osg::Vec3Array;

    //Schnittpunkte zwischen der Screenebene und den Kanten der Sichtpyramide
    for (unsigned int i = 0; i < (cornersFar.get()->size()); i++)
    {
        osg::Vec3 corner = (*cornersFar)[i];
        float ortho = projScreenPlane.getNormal() * corner;

        //Punkt auf g: origin+skalar*corner
        float skalar1 = (-plane.w() - plane.x() * origin.x() - plane.y() * origin.y() - plane.z() * origin.z())
                        / (plane.x() * (corner.x() - origin.x()) + plane.y() * (corner.y() - origin.y()) + plane.z() * (corner.z() - origin.z()));

        intersecPoints->push_back(origin + (corner - origin) * skalar1);
    }

    //Seitenlängen des Projektor-Frustums im Abstand der far-Clippingebene
    osg::Vec4 frustFar = getFrustumSizeFar();

    //Mittelpunkt in far-Clippingebene
    osg::Vec3 centerFar = osg::Vec3(frustFar.x() - frustFar.y(), frustFar.z() - frustFar.w(), -far_c) * getTransMat();

    //Schnittpunkte zwischen Screenebene und Gerade von Projektor zu Mittelpunkt in ClippingEbene
    float skalar2 = (-plane.w() - plane.x() * origin.x() - plane.y() * origin.y() - plane.z() * origin.z())
                    / (plane.x() * (centerFar.x() - origin.x()) + plane.y() * (centerFar.y() - origin.y()) + plane.z() * (centerFar.z() - origin.z()));

    intersecPoints->push_back(origin + (centerFar - origin) * skalar2);

    return intersecPoints.release(); //(li-o,re-o,re-u,li-u,centerFrust)
}

void Projector::calcScreen()
{
    //Voraussetzung: virt. ProjScreen soll so groß wie nötig (intersecPnts umfassen)
    //und so klein wie möglich sein (möglichst wenig Auflösung der DistMap verschwenden)

    //Schnittpunkt des Projektorfrustums mit der Screenebene
    osg::ref_ptr<osg::Vec3Array> intersecPnts = getPlaneIntersecPnts();

    //ScreenCenter beliebig festsetzen (hier Schnittpunkt Ebene zu Far.Mittelpunkt)
    projScreenCenter = (*intersecPnts)[4];

    //Intersec Points auf x-y Ebene Projezieren
    osg::ref_ptr<osg::Vec3Array> intersecPntsXY = new osg::Vec3Array();
    for (unsigned int i = 0; i < ((intersecPnts.get()->size() - 1)); i++) //letztes Element (mittelpunkt) wird entfernt
    {
        intersecPntsXY->push_back((*intersecPnts)[i] * osg::Matrix::inverse(getScreenTransMat()));
    }

    //Suche nach größtem und kleinstem x-Wert, bzw. y-Wert der Projezierten Schnittpunkte
    float largestX = 0;
    float largestY = 0;
    float smallestX = 0;
    float smallestY = 0;
    for (unsigned int j = 0; j < (intersecPntsXY.get()->size()); j++)
    {
        if ((*intersecPntsXY)[j].x() > largestX)
            largestX = (*intersecPntsXY)[j].x();
        if ((*intersecPntsXY)[j].x() < smallestX)
            smallestX = (*intersecPntsXY)[j].x();
        if ((*intersecPntsXY)[j].y() > largestY)
            largestY = (*intersecPntsXY)[j].y();
        if ((*intersecPntsXY)[j].y() < smallestY)
            smallestY = (*intersecPntsXY)[j].y();
    }

    //Intersec Points auf x-y Ebene Projezieren
    osg::ref_ptr<osg::Vec3Array> cornerPntsXY = new osg::Vec3Array();
    cornerPntsXY->push_back(osg::Vec3(smallestX, largestY, 0.0f)); //li,o
    cornerPntsXY->push_back(osg::Vec3(largestX, largestY, 0.0f)); //re,o
    cornerPntsXY->push_back(osg::Vec3(largestX, smallestY, 0.0f)); //re,u
    cornerPntsXY->push_back(osg::Vec3(smallestX, smallestY, 0.0f)); //li,u

    //Intersec Points zurück in Screen Ebene Projezieren
    cornerPnts = new osg::Vec3Array();
    for (unsigned int k = 0; k < (cornerPntsXY.get()->size()); k++)
    {
        cornerPnts->push_back((*cornerPntsXY)[k] * getScreenTransMat());
    }

    //Screengröße berechnen und in Variable speichern
    projScreenWidth = osg::Vec3((*cornerPnts)[0] - (*cornerPnts)[1]).length();
    projScreenHeight = osg::Vec3((*cornerPnts)[1] - (*cornerPnts)[2]).length();

    //Screenmittelpunkt berechnen und in Variable speichern
    projScreenCenter = (*cornerPnts)[0] //Vec US -> li-o
                       + ((*cornerPnts)[1] - (*cornerPnts)[0]) / 2 //Vec (li-o -> re-o)/2
                       + ((*cornerPnts)[3] - (*cornerPnts)[0]) / 2; //Vec (li-o -> li-u)/2

    //Eulerwinkel updaten
    hprVec = getHPR();
}

void Projector::update()
{
    calcScreen();
}

osg::Geometry *Projector::drawScreen(osg::Vec4 color)
{
    osg::ref_ptr<osg::Geometry> geomProjScreen = new osg::Geometry();

    // Eckpunkte setzen
    geomProjScreen->setVertexArray(cornerPnts.get());
    osg::ref_ptr<osg::DrawArrays> drawArrayScreen = new osg::DrawArrays(GL_QUADS, 0, 4);
    geomProjScreen->addPrimitiveSet(drawArrayScreen.get());

    //Farbe festlegen
    osg::ref_ptr<osg::Vec4Array> colorArray = new osg::Vec4Array();
    colorArray->push_back(color);
    geomProjScreen->setColorArray(colorArray.get());
    geomProjScreen->setColorBinding(osg::Geometry::BIND_OVERALL);

    //Normalen
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array();
    normals->push_back(osg::Vec3(0.0f, 1.0f, 0.0f) * getScreenTransMat());
    geomProjScreen->setNormalArray(normals.get());
    geomProjScreen->setNormalBinding(osg::Geometry::BIND_OVERALL);

    return geomProjScreen.release();
}

//#define GET_HPR(m,h,p,r)          { float cp; p= asin(m(1,2)); cp = cos(p); r = acos(m(2,2)/cp); h = -asin(m(1,0)/cp);  }
osg::Vec3 Projector::getHPR()
{
    osg::Matrix m = getScreenTransMat();
    float cp;
    float h, p, r; //Eulerwinkel

    osg::Vec3 v1(m(0, 0), m(0, 1), m(0, 2));
    osg::Vec3 v2(m(1, 0), m(1, 1), m(1, 2));
    osg::Vec3 v3(m(2, 0), m(2, 1), m(2, 2));

    v1.normalize();
    v2.normalize();
    v3.normalize();

    m(0, 0) = v1[0];
    m(0, 1) = v1[1];
    m(0, 2) = v1[2];
    m(1, 0) = v2[0];
    m(1, 1) = v2[1];
    m(1, 2) = v2[2];
    m(2, 0) = v3[0];
    m(2, 1) = v3[1];
    m(2, 2) = v3[2];

    p = asin(m(1, 2));
    cp = cos(p);
    float d = m(1, 0) / cp;
    if (d > 1.0)
    {
        h = -M_PI_2;
    }
    else if (d < -1.0)
    {
        h = M_PI_2;
    }
    else
        h = -asin(d);
    float diff = cos(h) * cp - m(1, 1);
    if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other Heading angle */
    {
        h = M_PI_2 - h;
        diff = cos(h) * cp - m(1, 1);
        if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other pitch angle */
        {
            p = M_PI - p;
            cp = cos(p);
            d = m(1, 0) / cp;
            if (d > 1.0)
            {
                h = -M_PI_2;
            }
            else if (d < -1.0)
            {
                h = M_PI_2;
            }
            else
                h = -asin(d);
            diff = cos(h) * cp - m(1, 1);
            if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other Heading angle */
            {
                h = M_PI - h;
            }
        }
    }
    d = m(2, 2) / cp;
    if (d > 1.0)
        r = 0;
    else if (d > 1.0)
        r = M_PI;
    else
        r = acos(d);

    diff = -sin(r) * cp - m(0, 2);
    if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other roll angle */
        r = -r;

    //Result
    h = h / M_PI * 180.0;
    p = p / M_PI * 180.0;
    r = r / M_PI * 180.0;

    return osg::Vec3(h, p, r);
}
