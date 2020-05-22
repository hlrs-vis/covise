/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include <cstring>
#include <cmath>

#include <config/CoviseConfig.h>
#include "cover/coTranslator.h"
#include "cover/coVRPluginSupport.h"

#include <osg/ShapeDrawable>
#include <cover/coIntersection.h>
#include <osgText/Text>

#include "HfT_string.h"
#include "HfT_osg_Plugin01_NormalschnittAnimation.h"

using namespace osg;

HfT_osg_Plugin01_NormalschnittAnimation::HfT_osg_Plugin01_NormalschnittAnimation()
{
    m_pfeil_exist = false;
    m_normalebene_exist = false;
    m_schiefebene_exist = false;
    m_schnittkurve_exist = false;
    m_schiefschnittkurve_exist = false;
    m_punkt = -1;
    m_normalwinkel = 0;
    m_normalwinkel_schritt = 45;
    m_schiefwinkel = 0;
    m_schiefwinkel_schritt = 22.5;
    m_hauptkruemmungsrichtung_exist = false;
    m_hauptkruemmung_exist = false;
    m_schmiegtangente_exist = false;
    m_kruemmungskreise_exist = false;
    m_meusnierkugel_exist = false;
}

void HfT_osg_Plugin01_NormalschnittAnimation::quicksort(Vec2Array *arr, int left, int right)
{
    int i = left, j = right;
    float tmp;
    float pivot = (*arr)[(left + right) / 2].x();

    /* partition */
    while (i <= j)
    {
        while ((*arr)[i].x() < pivot)
            i++;
        while ((*arr)[j].x() > pivot)
            j--;
        if (i <= j)
        {

            tmp = (*arr)[i].x();
            (*arr)[i].x() = (*arr)[j].x();
            (*arr)[j].x() = tmp;

            tmp = (*arr)[i].y();
            (*arr)[i].y() = (*arr)[j].y();
            (*arr)[j].y() = tmp;

            i++;
            j--;
        }
    }
    /* recursion */
    if (left < j)
        quicksort(arr, left, j);
    if (i < right)
        quicksort(arr, i, right);
}

void HfT_osg_Plugin01_NormalschnittAnimation::Create_Pfeil(osg::Vec3 startPickPos, HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf, MatrixTransform *mts_plane)
{
    if (mts_surf)
    {
        if (mts_plane)
        {
            BoundingBox bs = surf->getBoundingBox();
            float Radius = bs.radius();
            //Pfeil:
            ref_ptr<ShapeDrawable> kugel = new ShapeDrawable;
            kugel->setShape(new Sphere(Vec3(0.0f, 0.0f, 0.0f), 0.005 * Radius));
            //kugel->setColor(Vec4(1.0f, 0.0f, 0.0f, 1.0f));

            osg::Material *pMat1 = new osg::Material;
            pMat1->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
            pMat1->setShininess(osg::Material::FRONT, 5.0f);
            kugel->getOrCreateStateSet()->setAttributeAndModes(pMat1, osg::StateAttribute::ON);

            ref_ptr<ShapeDrawable> zylinder = new ShapeDrawable;
            zylinder->setShape(new Cylinder(Vec3(0.0f, 0.0f, 0.05 * Radius), 0.002 * Radius, 0.1 * Radius));
            //zylinder->setColor(Vec4(1.0f, 0.0f, 0.0f, 1.0f));

            osg::Material *pMat2 = new osg::Material;
            pMat2->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
            pMat2->setShininess(osg::Material::FRONT, 5.0f);
            zylinder->getOrCreateStateSet()->setAttributeAndModes(pMat2, osg::StateAttribute::ON);

            ref_ptr<osg::ShapeDrawable> kegel = new ShapeDrawable;
            kegel->setShape(new Cone(Vec3(0.0f, 0.0f, 0.1 * Radius), 0.008 * Radius, 0.02 * Radius));
            //kegel->setColor(Vec4(1.0f, 0.0f, 0.0f, 1.0f));

            osg::Material *pMat3 = new osg::Material;
            pMat3->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
            pMat3->setShininess(osg::Material::FRONT, 5.0f);
            kegel->getOrCreateStateSet()->setAttributeAndModes(pMat3, osg::StateAttribute::ON);

            ref_ptr<Geode> pfeil = new Geode;
            pfeil->addDrawable(kugel.get());
            pfeil->addDrawable(zylinder.get());
            pfeil->addDrawable(kegel.get());

            if ((startPickPos.x() >= bs.xMin() && startPickPos.x() <= bs.xMax()) && (startPickPos.y() >= bs.yMin() && startPickPos.y() <= bs.yMax()) && (startPickPos.z() >= bs.zMin() && startPickPos.z() <= bs.zMax()) /*bs.contains(startPickPos)*/)
            {

                surf->insertArray2(startPickPos);

                // Punkte auf der Fläche/Ebene
                ref_ptr<Vec3Array> points = surf->getPointArray();
                ref_ptr<Vec2Array> points_2 = surf->getPointArray2();
                ref_ptr<Vec3Array> points2 = plane->getPointArray();

                int rechts = (*points_2).size() - 1;
                quicksort(points_2, 0, rechts);

                //// Normalen auf der Fläche/Ebene
                ref_ptr<Vec3Array> normals = surf->getNormalArray();
                ref_ptr<Vec3Array> normals2 = plane->getNormalArray();

                //Stelle im Ursprungsarray des gewählten Punktes
                m_punkt = (int)(*points_2)[0].y();

                if (m_punkt >= 0)
                {

                    if ((*normals)[m_punkt] != Vec3(0, 0, 0))
                    {
                        // Verschiebung des Pfeils auf der Fläche
                        ref_ptr<MatrixTransform> verschiebung = new MatrixTransform();
                        Matrix m;
                        m.makeTranslate((*points)[m_punkt]);
                        verschiebung->setMatrix(m);

                        // Verschiebung des Pfeils auf der Ebene
                        ref_ptr<MatrixTransform> verschiebung2 = new MatrixTransform();
                        Matrix m2;
                        m2.makeTranslate((*points2)[m_punkt]);
                        verschiebung2->setMatrix(m2);

                        // Drehung des Pfeils in Richtung der Normalen
                        ref_ptr<MatrixTransform> drehung = new MatrixTransform();
                        Vec3 normal = Vec3((*normals)[m_punkt]);
                        Quat rotation;
                        rotation.makeRotate(Vec3(0.0f, 0.0f, 1.0f), normal);
                        drehung->setMatrix(Matrix(rotation));

                        if (m_pfeil_exist == false)
                        {
                            ref_ptr<Switch> sw_pfeil = new Switch;
                            m_Pfeil = sw_pfeil;
                            ref_ptr<Switch> sw_pfeil2 = new Switch;
                            m_Pfeil_in_Ebene = sw_pfeil2;
                        }

                        mts_surf->addChild(m_Pfeil);
                        m_Pfeil->addChild(verschiebung);
                        verschiebung->addChild(drehung);
                        drehung->addChild(pfeil);

                        mts_plane->addChild(m_Pfeil_in_Ebene);
                        m_Pfeil_in_Ebene->addChild(verschiebung2);
                        verschiebung2->addChild(pfeil);

                        m_pfeil_exist = true;
                        //return true;
                    }
                    else
                    {
                        m_pfeil_exist = false;
                        //return false;
                    }
                }
                else
                {
                    m_pfeil_exist = false;
                }
                surf->getPointArray2()->clear();
            }
        }
    }
}
bool HfT_osg_Plugin01_NormalschnittAnimation::Normalschnitt_Animation(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf, MatrixTransform *mts_plane)
{
    if (!m_pfeil_exist)
        return false;
    if (!mts_surf)
        return false;
    if (!mts_plane)
        return false;

    // Punkte/Normalen auf der Flaeche
    ref_ptr<Vec3Array> points = surf->getPointArray();
    ref_ptr<Vec3Array> normals = surf->getNormalArray();

    // Verschiebung in den ausgewählten Punkt der Fläche
    ref_ptr<MatrixTransform> verschiebung = new MatrixTransform();
    Matrix m;
    m.makeTranslate((*points)[m_punkt]);
    verschiebung->setMatrix(m);

    // Drehung der Ebene um die Normale
    ref_ptr<MatrixTransform> normal_drehung = new MatrixTransform();
    Vec3 normal = Vec3((*normals)[m_punkt]);
    Quat rotation;
    rotation.makeRotate(DegreesToRadians(m_normalwinkel), normal);
    normal_drehung->setMatrix(Matrix(rotation));

    // Drehung in Richtung der Normalen
    ref_ptr<MatrixTransform> drehung = new MatrixTransform();
    Vec3 normal2 = Vec3((*normals)[m_punkt]);
    Quat rotation2;
    rotation2.makeRotate(Vec3(0.0f, 0.0f, 1.0f), normal2);
    drehung->setMatrix(Matrix(rotation2));

    if (m_normalebene_exist == false)
    {
        ref_ptr<Switch> sw_ebenen = new Switch;
        m_normalEbenen = sw_ebenen;
        ref_ptr<Switch> sw_schnittkurven = new Switch;
        m_Schnittkurven = sw_schnittkurven;
        ref_ptr<Switch> sw_schnittkurven2 = new Switch;
        m_Schnittkurven_in_Ebene = sw_schnittkurven2;
    }

    ref_ptr<Geode> ebene = new Geode;
    ebene->addDrawable(Create_Plane(surf, verschiebung, normal_drehung, drehung));
    drehung->addChild(ebene);
    normal_drehung->addChild(drehung);
    verschiebung->addChild(normal_drehung);
    m_normalEbenen->addChild(verschiebung);
    if (m_normalebene_exist == false)
        mts_surf->addChild(m_normalEbenen);

    m_normalebene_exist = true;

    if (!m_schnittkurve_exist)
        Change_Mode(surf, plane, 6);

    ref_ptr<Geode> schnittkurve = new Geode;
    schnittkurve->addDrawable(Schnittkurvenberechnung_auf_Flaeche(surf)); // -> mit Intersect Visitor
    m_Schnittkurven->addChild(schnittkurve);
    if (m_schnittkurve_exist == false)
        mts_surf->addChild(m_Schnittkurven);

    ref_ptr<Geode> schnittkurve_in_ebene = new Geode;
    schnittkurve_in_ebene->addDrawable(Schnittkurvenberechnung_in_Ebene(surf, plane)); // -> mit Newton
    m_Schnittkurven_in_Ebene->addChild(schnittkurve_in_ebene);
    if (m_schnittkurve_exist == false)
        mts_plane->addChild(m_Schnittkurven_in_Ebene);

    m_schnittkurve_exist = true;
    m_normalwinkel = m_normalwinkel + m_normalwinkel_schritt;
    return true;
}

ref_ptr<Drawable> HfT_osg_Plugin01_NormalschnittAnimation::Create_Plane(HfT_osg_Plugin01_ParametricSurface *surf, ref_ptr<MatrixTransform> verschiebung, ref_ptr<MatrixTransform> drehung2, ref_ptr<MatrixTransform> drehung1)
{
    // Ebene erstellen
    BoundingBox bs = surf->getBoundingBox();
    float Radius = bs.radius();
    ref_ptr<Vec3Array> vertices = new Vec3Array;
    ref_ptr<Vec3dArray> geradenPunkte = new Vec3dArray;

    //Rechtecke
    float hoehe = 0.125 * Radius;
    float breite = (int)(0.2f * Radius);
    float schritt = breite / 20;
    int zaehler = 0;

    for (float x = -breite / 2; x < breite / 2; x = x + schritt)
    {
        vertices->push_back(Vec3d(x, 0.0f, -hoehe * 2 / 3));
        vertices->push_back(Vec3d(x + schritt, 0.0f, -hoehe * 2 / 3));
        vertices->push_back(Vec3d(x + schritt, 0.0f, hoehe));
        vertices->push_back(Vec3(x, 0.0f, hoehe));

        //1. und 4. Punkt speichern für Geraden
        geradenPunkte->push_back(Vec3d(x, 0.0f, -hoehe * 2 / 3));
        geradenPunkte->push_back(Vec3d(x, 0.0f, hoehe));

        zaehler++;
        //Beim letzten Rechteck 2. und 3. Punkt speichern für letzte Gerade
        if (zaehler * schritt >= breite)
        {
            geradenPunkte->push_back(Vec3d(x + schritt, 0.0f, -hoehe * 2 / 3));
            geradenPunkte->push_back(Vec3d(x + schritt, 0.0f, hoehe));
        }
    }

    //Normale
    ref_ptr<Vec3Array> normal = new Vec3Array;
    normal->push_back(Vec3(0.0f, -1.0f, 0.0f));

    //Farbe
    ref_ptr<Vec4Array> colors = new Vec4Array;
    //colors->push_back(Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    //colors->push_back(Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    //colors->push_back(Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    colors->push_back(Vec4(0.0f, 0.0f, 1.0f, 1.0f));

    ref_ptr<Geometry> ebene = new Geometry;

    ebene->setNormalArray(normal.get());
    ebene->setNormalBinding(Geometry::BIND_OVERALL);
    ebene->setColorArray(colors.get());
    ebene->setColorBinding(Geometry::BIND_PER_VERTEX);
    ebene->setColorBinding(Geometry::BIND_OVERALL);
    ebene->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    //osg::Material *pMat = new osg::Material;
    //pMat->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 1.0f, 1.0f));
    //pMat->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    //ebene->getOrCreateStateSet()->setAttributeAndModes(pMat,osg::StateAttribute::ON);
    ebene->setVertexArray(vertices.get());
    ebene->addPrimitiveSet(new DrawArrays(GL_QUADS, 0, vertices->size()));

    ref_ptr<Vec3dArray> geradenPunkteNeu = new Vec3dArray;
    Matrix ma, mb, mc;

    //Punkte neue Koordinaten zuordnen -> nach Drehungen und Verschiebung
    for (unsigned int i = 0; i < geradenPunkte->size(); i++)
    {
        Vec3d pos = Vec3d((*geradenPunkte)[i]);
        ma = drehung1->getMatrix();
        mb = drehung2->getMatrix();
        mc = verschiebung->getMatrix();
        pos = pos * ma * mb * mc;
        geradenPunkteNeu->push_back(pos);
    }

    m_geradenPunkteNeu = geradenPunkteNeu;
    return ebene.get();
}

ref_ptr<Drawable> HfT_osg_Plugin01_NormalschnittAnimation::Schnittkurvenberechnung_auf_Flaeche(HfT_osg_Plugin01_ParametricSurface *surf)
{
    ref_ptr<Vec3Array> vertices = new Vec3Array;

    int z = 0;
    //Schleife ueber alle Geraden
    for (unsigned int i = 0; i < m_geradenPunkteNeu->size(); i = i + 2)
    {
        z++;
        Vec3d A = Vec3((*m_geradenPunkteNeu)[i]);
        Vec3d B = Vec3((*m_geradenPunkteNeu)[i + 1]);


        opencover::coIntersector* isect = opencover::coIntersection::instance()->newIntersector(A, B);
        osgUtil::IntersectionVisitor visitor(isect);
        visitor.setTraversalMask(opencover::Isect::Pick);

        surf->accept(visitor);

        if (isect->containsIntersections())
        {
            auto results = isect->getFirstIntersection();

            Vec3d intersecPnt = results.getWorldIntersectPoint();
            vertices->push_back(intersecPnt);
        }
    }

    m_schnittpunkte = vertices;
    ref_ptr<Geometry> schnittkurve = new Geometry;

    ref_ptr<Vec4Array> colors = new Vec4Array;
    /*colors->push_back(Vec4(1.0f, 0.64f, 0.0f, 1.0f));*/
    colors->push_back(Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    schnittkurve->setColorArray(colors.get());
    schnittkurve->setColorBinding(Geometry::BIND_OVERALL);
    schnittkurve->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    //osg::Material *pMat1 = new osg::Material;
    //pMat1->setEmission(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    //pMat1->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    //schnittkurve->getOrCreateStateSet()->setAttributeAndModes(pMat1,osg::StateAttribute::ON);
    schnittkurve->setVertexArray(vertices.get());
    schnittkurve->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, vertices->size()));
    osg::StateSet *stateset = schnittkurve->getOrCreateStateSet();
    osg::LineWidth *lw = new osg::LineWidth(3.0f);
    stateset->setAttribute(lw);
    return schnittkurve;
}

ref_ptr<Drawable> HfT_osg_Plugin01_NormalschnittAnimation::Schnittkurvenberechnung_in_Ebene(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane)
{
    Vec2dArray *parameterSchnittkurve = new Vec2dArray(m_schnittpunkte->size()); // mit Nullen
    ref_ptr<Vec2dArray> PARAMETERSCHNITTKURVE = new Vec2dArray(); // Feld mit Parameter
    ref_ptr<Vec3Array> vertices = new Vec3Array(); // Raumpunkte in der Ebene

    Vec2Array *parameter = surf->getParameterValues();
    Vec2 startParameter = (*parameter)[m_punkt];
    Vec3Array *koordinaten = surf->getPointArray();
    Vec3 startKoordinaten = (*koordinaten)[m_punkt];

    double ua = surf->getLowerBoundU();
    double ue = surf->getUpperBoundU();
    double va = surf->getLowerBoundV();
    double ve = surf->getUpperBoundV();

    double minabstand = 100;
    int zaehler = 0;
    for (unsigned int i = 0; i < m_schnittpunkte->size(); i++)
    {
        double abstand = Abstand3(m_schnittpunkte->at(i), startKoordinaten);
        if (abstand < minabstand)
        {
            minabstand = abstand;
            zaehler = i;
        }
    }

    // Startpunkt ist ausgewählter Punkt
    for (int i = zaehler; i >= 0; i--)
    {
        Vec2d param = NewtonVerfahren((*m_schnittpunkte)[i], startParameter, surf);
        if (param[0] > ua && param[0] < ue && param[1] > va && param[1] < ve)
        {
            (*parameterSchnittkurve)[i] = param;
            startParameter = param;
        }
    }

    startParameter = (*parameter)[m_punkt];
    for (unsigned int i = zaehler + 1; i < m_schnittpunkte->size(); i++)
    {
        Vec2d param = NewtonVerfahren((*m_schnittpunkte)[i], startParameter, surf);
        if (param[0] > ua && param[0] < ue && param[1] > va && param[1] < ve)
        {
            (*parameterSchnittkurve)[i] = param;
            startParameter = param;
        }
    }

    for (unsigned int i = 0; i < parameterSchnittkurve->size(); i++)
    {
        if ((*parameterSchnittkurve)[i] != Vec2d(0, 0))
            PARAMETERSCHNITTKURVE->push_back((*parameterSchnittkurve)[i]);
    }

    // Schleife ueber alle Schnittpunkte
    for (unsigned int i = 0; i < PARAMETERSCHNITTKURVE->size(); i++)
    {
        double u = PARAMETERSCHNITTKURVE->at(i)[0];
        double v = PARAMETERSCHNITTKURVE->at(i)[1];
        Vec3dArray *pkte = plane->computeSchnittkurve(u, v, 1);
        Vec3d p = (*pkte)[0];
        vertices->push_back(p);
    }

    ref_ptr<Geometry> schnittkurve_in_ebene = new Geometry;

    ref_ptr<Vec4Array> colors = new Vec4Array;
    colors->push_back(Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    schnittkurve_in_ebene->setColorArray(colors.get());
    schnittkurve_in_ebene->setColorBinding(Geometry::BIND_OVERALL);
    schnittkurve_in_ebene->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    //   osg::Material *pMat1 = new osg::Material;
    //pMat1->setEmission(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    ////pMat1->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    //schnittkurve_in_ebene->getOrCreateStateSet()->setAttributeAndModes(pMat1,osg::StateAttribute::ON);
    schnittkurve_in_ebene->setVertexArray(vertices.get());
    schnittkurve_in_ebene->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, vertices->size()));
    osg::StateSet *stateset = schnittkurve_in_ebene->getOrCreateStateSet();
    osg::LineWidth *lw = new osg::LineWidth(3.0f);
    stateset->setAttribute(lw);
    return schnittkurve_in_ebene;
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Hauptkruemmungsrichtungen(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf)
{
    BoundingBox bs = surf->getBoundingBox();
    float Radius = bs.radius();

    // Punkte/Normalen auf der Fläche
    ref_ptr<Vec3Array> points = surf->getPointArray();
    ref_ptr<Vec3Array> normals = surf->getNormalArray();

    m_hauptkruemmungsrichtung_exist = true;
    Change_Mode(surf, plane, 6);

    Vec2Array *parameter = surf->getParameterValues();
    Vec2 p = (*parameter)[m_punkt];
    Vec3dArray *pkte = surf->computeSchnittkurve(p[0], p[1], 1);
    Vec3d fu = (*pkte)[1];
    Vec3d fv = (*pkte)[2];
    Vec3dArray *pkte2 = surf->computeSchnittkurve(p[0], p[1], 2);
    Vec3d punkt2 = (*pkte2)[0];
    double K = punkt2[0];
    double H = punkt2[1];
    double E = punkt2[2];
    Vec3d punkt3 = (*pkte2)[1];
    double F = punkt3[0];
    double G = punkt3[1];
    double L = punkt3[2];
    Vec3d punkt4 = (*pkte2)[2];
    double M = punkt4[0];
    double N = punkt4[1];

    double HptKr1;
    double HptKr2;
    Vec3 hptKrR1;
    Vec3 hptKrR2;

    if (F == 0 && M == 0)
    {
        hptKrR1 = fu / (sqrt(fu[0] * fu[0] + fu[1] * fu[1] + fu[2] * fu[2]));
        hptKrR2 = fv / (sqrt(fv[0] * fv[0] + fv[1] * fv[1] + fv[2] * fv[2]));
        HptKr1 = L / E;
        HptKr2 = N / G;
    }
    else
    {
        double q = sqrt((E * N - G * L) * (E * N - G * L) - 4 * (E * M - F * L) * (F * N - G * M));
        double lambda1 = (-(E * N - G * L) + q) / (2 * (F * N - G * M));
        double lambda2 = (-(E * N - G * L) - q) / (2 * (F * N - G * M));
        hptKrR1 = (fu + fv * lambda1) / (sqrt(E + 2 * F * lambda1 + G * lambda1 * lambda1));
        hptKrR2 = (fu + fv * lambda2) / (sqrt(E + 2 * F * lambda2 + G * lambda2 * lambda2));
        //Diskriminante manchmal negativ -> Betrag
        HptKr1 = H - sqrt(double(abs(H * H - K)));
        HptKr2 = H + sqrt(double(abs(H * H - K)));
    }

    // keine Hauptkrümmungsrichtungen in einem Nabel- oder Flachpunkt
    if (abs(HptKr2 - HptKr1) < 0.00001)
    {
        m_hauptkruemmungsrichtung_exist = false;
        if (!KruemmungsMethode_aktiv())
            Change_Mode(surf, plane, 5);
        return false;
    }

    ref_ptr<ShapeDrawable> kugel = new ShapeDrawable;
    kugel->setShape(new Sphere(Vec3(0.0f, 0.0f, 0.0f), 0.005 * Radius));
    //kugel->setColor(Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    osg::Material *pMat1 = new osg::Material;
    pMat1->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    pMat1->setShininess(osg::Material::FRONT, 5.0f);
    kugel->getOrCreateStateSet()->setAttributeAndModes(pMat1, osg::StateAttribute::ON);

    ref_ptr<ShapeDrawable> zylinder = new ShapeDrawable;
    zylinder->setShape(new Cylinder(Vec3(0.0f, 0.0f, 0.05 * Radius), 0.002 * Radius, 0.1 * Radius));
    //zylinder->setColor(Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    osg::Material *pMat2 = new osg::Material;
    pMat2->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    pMat2->setShininess(osg::Material::FRONT, 5.0f);
    zylinder->getOrCreateStateSet()->setAttributeAndModes(pMat2, osg::StateAttribute::ON);

    ref_ptr<osg::ShapeDrawable> kegel = new ShapeDrawable;
    kegel->setShape(new Cone(Vec3(0.0f, 0.0f, 0.1 * Radius), 0.008 * Radius, 0.02 * Radius));
    //kegel->setColor(Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    osg::Material *pMat3 = new osg::Material;
    pMat3->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    pMat3->setShininess(osg::Material::FRONT, 5.0f);
    kegel->getOrCreateStateSet()->setAttributeAndModes(pMat3, osg::StateAttribute::ON);

    ref_ptr<Geode> pfeil = new Geode;
    pfeil->addDrawable(kugel.get());
    pfeil->addDrawable(zylinder.get());
    pfeil->addDrawable(kegel.get());

    // Verschiebung des Pfeils auf der Fläche
    ref_ptr<MatrixTransform> verschiebung = new MatrixTransform();
    Matrix m;
    m.makeTranslate((*points)[m_punkt]);
    verschiebung->setMatrix(m);

    // Drehung des Pfeils in Richtung der 1. HptKrR
    ref_ptr<MatrixTransform> drehung1 = new MatrixTransform();
    Quat rotation;
    rotation.makeRotate(Vec3(0.0f, 0.0f, 1.0f), hptKrR1);
    drehung1->setMatrix(Matrix(rotation));

    // Drehung des Pfeils in Richtung der 2. HptKrR
    ref_ptr<MatrixTransform> drehung2 = new MatrixTransform();
    Quat rotation2;
    rotation2.makeRotate(Vec3(0.0f, 0.0f, 1.0f), hptKrR2);
    drehung2->setMatrix(Matrix(rotation2));

    ref_ptr<Switch> sw_pfeil = new Switch;
    m_HptKrPfeile = sw_pfeil;

    m_HptKrPfeile->addChild(verschiebung);
    verschiebung->addChild(drehung1);
    verschiebung->addChild(drehung2);
    drehung1->addChild(pfeil);
    drehung2->addChild(pfeil);
    mts_surf->addChild(m_HptKrPfeile);
    return true;
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Hauptkruemmungen(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf, MatrixTransform *mts_plane)
{
    BoundingBox bs = surf->getBoundingBox();
    float Radius = bs.radius();

    m_hauptkruemmung_exist = true;
    Change_Mode(surf, plane, 6);

    Vec2Array *parameter = surf->getParameterValues();
    Vec2 p = (*parameter)[m_punkt];
    Vec3dArray *pkte = surf->computeSchnittkurve(p[0], p[1], 1);
    Vec3d punkt1 = (*pkte)[0];
    Vec3d fu = (*pkte)[1];
    Vec3d fv = (*pkte)[2];
    Vec3d n1 = (*pkte)[6];
    Vec3d n = n1 / (sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]));
    Vec3dArray *pkte2 = surf->computeSchnittkurve(p[0], p[1], 2);
    Vec3d punkt2 = (*pkte2)[0];
    double K = punkt2[0];
    double H = punkt2[1];
    double E = punkt2[2];
    Vec3d punkt3 = (*pkte2)[1];
    double F = punkt3[0];
    double G = punkt3[1];
    double L = punkt3[2];
    Vec3d punkt4 = (*pkte2)[2];
    double M = punkt4[0];
    double N = punkt4[1];

    double HptKr1;
    double HptKr2;
    Vec3d hptKrR1;
    Vec3d hptKrR2;
    bool k1max = false;
    bool k2max = false;

    if (F == 0 && M == 0)
    {
        hptKrR1 = fu / (sqrt(fu[0] * fu[0] + fu[1] * fu[1] + fu[2] * fu[2]));
        hptKrR2 = fv / (sqrt(fv[0] * fv[0] + fv[1] * fv[1] + fv[2] * fv[2]));
        HptKr1 = L / E;
        HptKr2 = N / G;
    }
    else
    {
        double q = sqrt((E * N - G * L) * (E * N - G * L) - 4 * (E * M - F * L) * (F * N - G * M));
        double lambda1 = (-(E * N - G * L) + q) / (2 * (F * N - G * M));
        double lambda2 = (-(E * N - G * L) - q) / (2 * (F * N - G * M));
        hptKrR1 = (fu + fv * lambda1) / (sqrt(E + 2 * F * lambda1 + G * lambda1 * lambda1));
        hptKrR2 = (fu + fv * lambda2) / (sqrt(E + 2 * F * lambda2 + G * lambda2 * lambda2));
        //Diskriminante manchmal negativ -> Betrag
        HptKr1 = H - sqrt(abs(H * H - K));
        HptKr2 = H + sqrt(abs(H * H - K));
    }

    if (abs(HptKr1) > abs(HptKr2))
        k1max = true;
    if (abs(HptKr2) > abs(HptKr1))
        k2max = true;
    if (abs(abs(HptKr2) - abs(HptKr1)) < 0.00001)
    {
        k1max = false;
        k2max = false;
    }

    // keine Hauptkrümmungen in einem Nabel- oder Flachpunkt
    if (abs(HptKr2 - HptKr1) < 0.00001)
    {
        m_hauptkruemmung_exist = false;
        if (!KruemmungsMethode_aktiv())
            Change_Mode(surf, plane, 5);
        return false;
    }

    ref_ptr<Vec3Array> punkteH1o = new Vec3Array;
    for (int i = -10; i <= 10; i++)
    {
        Vec3 punktImRaum = punkt1 + n * Radius * 0.12 + hptKrR1 * i * 0.0095 * Radius;
        punkteH1o->push_back(punktImRaum);
    }
    ref_ptr<Vec3Array> punkteH1u = new Vec3Array;
    for (int i = -10; i <= 10; i++)
    {
        Vec3 punktImRaum = punkt1 - n * Radius * 0.078 + hptKrR1 * i * 0.0095 * Radius;
        punkteH1u->push_back(punktImRaum);
    }

    ref_ptr<Vec3Array> punkteH2o = new Vec3Array;
    for (int i = -10; i <= 10; i++)
    {
        Vec3 punktImRaum = punkt1 + n * Radius * 0.12 + hptKrR2 * i * 0.0095 * Radius;
        punkteH2o->push_back(punktImRaum);
    }
    ref_ptr<Vec3Array> punkteH2u = new Vec3Array;
    for (int i = -10; i <= 10; i++)
    {
        Vec3 punktImRaum = punkt1 - n * Radius * 0.078 + hptKrR2 * i * 0.0095 * Radius;
        punkteH2u->push_back(punktImRaum);
    }

    ref_ptr<Vec3Array> schnittpunkte1 = new Vec3Array();
    for (unsigned int i = 0; i < punkteH1o->size(); i++)
    {
        Vec3d A = Vec3((*punkteH1o)[i]);
        Vec3d B = Vec3((*punkteH1u)[i]);


        opencover::coIntersector* isect = opencover::coIntersection::instance()->newIntersector(A, B);
        osgUtil::IntersectionVisitor visitor(isect);
        visitor.setTraversalMask(opencover::Isect::Pick);

        surf->accept(visitor);

        if (isect->containsIntersections())
        {
            auto results = isect->getFirstIntersection();
            Vec3d intersecPnt = results.getWorldIntersectPoint();
            schnittpunkte1->push_back(intersecPnt);
        }
    }

    ref_ptr<Vec3Array> schnittpunkte2 = new Vec3Array();
    for (unsigned int i = 0; i < punkteH2o->size(); i++)
    {
        Vec3d A = Vec3((*punkteH2o)[i]);
        Vec3d B = Vec3((*punkteH2u)[i]);



        opencover::coIntersector* isect = opencover::coIntersection::instance()->newIntersector(A, B);
        osgUtil::IntersectionVisitor visitor(isect);
        visitor.setTraversalMask(opencover::Isect::Pick);

        surf->accept(visitor);

        if (isect->containsIntersections())
        {
            auto results = isect->getFirstIntersection();
            Vec3d intersecPnt = results.getWorldIntersectPoint();
            schnittpunkte2->push_back(intersecPnt);
        }
    }
    ref_ptr<Geometry> HptKr1_auf_Flaeche = new Geometry;
    ref_ptr<Vec4Array> colors1 = new Vec4Array;
    //osg::Material *pMat1 = new osg::Material;
    if (k1max == true || (k1max == false && k2max == false))
    {
        colors1->push_back(Vec4(1.0f, 1.0f, 0.0f, 1.0f)); //gelb
        //pMat1->setEmission(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 0.0f, 1.0f));
        //pMat1->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    }
    else
    {
        colors1->push_back(Vec4(0.4196f, 1.0f, 0.952f, 1.0f));
        //pMat1->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 1.0f, 0.2f, 1.0f));
        //pMat1->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    }

    HptKr1_auf_Flaeche->setColorArray(colors1.get());
    HptKr1_auf_Flaeche->setColorBinding(Geometry::BIND_OVERALL);
    HptKr1_auf_Flaeche->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    //HptKr1_auf_Flaeche->getOrCreateStateSet()->setAttributeAndModes(pMat1,osg::StateAttribute::ON);
    HptKr1_auf_Flaeche->setVertexArray(schnittpunkte1.get());
    HptKr1_auf_Flaeche->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, schnittpunkte1->size()));
    osg::StateSet *stateset1 = HptKr1_auf_Flaeche->getOrCreateStateSet();
    osg::LineWidth *lw = new osg::LineWidth(3.0f);
    lw = new osg::LineWidth(3.0f);
    stateset1->setAttribute(lw);

    ref_ptr<Geometry> HptKr2_auf_Flaeche = new Geometry;
    ref_ptr<Vec4Array> colors2 = new Vec4Array;
    //osg::Material *pMat2 = new osg::Material;
    if (k2max == true || (k1max == false && k2max == false))
    {
        colors2->push_back(Vec4(1.0f, 1.0f, 0.0f, 1.0f)); //gelb	1.0f, 1.0f, 0.0f
        //pMat2->setEmission(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 1.0f, 0.0f, 1.0f));
        //pMat2->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    }
    else
    {
        colors2->push_back(Vec4(0.4196f, 1.0f, 0.952f, 1.0f)); //0.4196f, 1.0f, 0.839fVec4(0.0f, 1.0f, 0.2f, 1.0f)//gruen
        //pMat2->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 1.0f, 0.2f, 1.0f));
        //pMat2->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    }

    HptKr2_auf_Flaeche->setColorArray(colors2.get());
    HptKr2_auf_Flaeche->setColorBinding(Geometry::BIND_OVERALL);
    HptKr2_auf_Flaeche->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    //HptKr2_auf_Flaeche->getOrCreateStateSet()->setAttributeAndModes(pMat2,osg::StateAttribute::ON);
    HptKr2_auf_Flaeche->setVertexArray(schnittpunkte2.get());
    HptKr2_auf_Flaeche->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, schnittpunkte2->size()));
    osg::StateSet *stateset2 = HptKr2_auf_Flaeche->getOrCreateStateSet();
    lw = new osg::LineWidth(3.0f);
    stateset2->setAttribute(lw);

    color1 = (*colors1)[0];
    fontSize = Radius * 0.0225;
    position1 = (*schnittpunkte1)[0];
    HptKr1 = HptKr1 * 100000;
    HptKr1 = double((int)(float(HptKr1)));
    HptKr1 = HptKr1 / 100000;
    string txt1 = coTranslator::coTranslate("Hauptkruemmung 1: ");
    string txt2 = HfT_double_to_string(HptKr1);
    text_1 = txt1 + txt2;

    color2 = (*colors2)[0];
    position2 = (*schnittpunkte2)[0];
    HptKr2 = HptKr2 * 100000;
    HptKr2 = double((int)(float(HptKr2)));
    HptKr2 = HptKr2 / 100000;
    string txt3 = coTranslator::coTranslate("Hauptkruemmung 2: ");
    string txt4 = HfT_double_to_string(HptKr2);
    text_2 = txt3 + txt4;

    ref_ptr<Geode> HptKrR_auf_Flaeche = new Geode;
    mp_Hauptkruemmungen_auf_Flaeche = HptKrR_auf_Flaeche;
    HptKrR_auf_Flaeche->addDrawable(HptKr1_auf_Flaeche);
    HptKrR_auf_Flaeche->addDrawable(HptKr2_auf_Flaeche);

    mts_surf->addChild(HptKrR_auf_Flaeche);

    // Hauptkrümmungen in der Ebene
    ref_ptr<Vec3Array> vertices = new Vec3Array();
    Vec2dArray *parameterSchnittkurve = new Vec2dArray(schnittpunkte1->size()); // mit Nullen
    ref_ptr<Vec2dArray> PARAMETERSCHNITTKURVE = new Vec2dArray();

    Vec2 startParameter = (*parameter)[m_punkt];
    Vec3Array *koordinaten = surf->getPointArray();
    Vec3 startKoordinaten = (*koordinaten)[m_punkt];

    double ua = surf->getLowerBoundU();
    double ue = surf->getUpperBoundU();
    double va = surf->getLowerBoundV();
    double ve = surf->getUpperBoundV();

    double minabstand = 100;
    int zaehler = 0;
    for (unsigned int i = 0; i < schnittpunkte1->size(); i++)
    {
        double abstand = Abstand3(schnittpunkte1->at(i), startKoordinaten);
        if (abstand < minabstand)
        {
            minabstand = abstand;
            zaehler = i;
        }
    }

    for (int i = zaehler; i >= 0; i--)
    {
        Vec2d param = NewtonVerfahren((*schnittpunkte1)[i], startParameter, surf);
        if (param[0] > ua && param[0] < ue && param[1] > va && param[1] < ve)
        {
            (*parameterSchnittkurve)[i] = param;
            startParameter = param;
        }
    }
    startParameter = (*parameter)[m_punkt];
    for (unsigned int i = zaehler + 1; i < schnittpunkte1->size(); i++)
    {
        Vec2d param = NewtonVerfahren((*schnittpunkte1)[i], startParameter, surf);
        if (param[0] > ua && param[0] < ue && param[1] > va && param[1] < ve)
        {
            (*parameterSchnittkurve)[i] = param;
            startParameter = param;
        }
    }

    for (unsigned int i = 0; i < parameterSchnittkurve->size(); i++)
    {
        if ((*parameterSchnittkurve)[i] != Vec2d(0, 0))
            PARAMETERSCHNITTKURVE->push_back((*parameterSchnittkurve)[i]);
    }

    // Schleife ueber alle Schnittpunkte
    for (unsigned int i = 0; i < PARAMETERSCHNITTKURVE->size(); i++)
    {
        double u = PARAMETERSCHNITTKURVE->at(i)[0];
        double v = PARAMETERSCHNITTKURVE->at(i)[1];
        Vec3dArray *pkte = plane->computeSchnittkurve(u, v, 1);
        Vec3d p = (*pkte)[0];
        vertices->push_back(p);
    }

    ref_ptr<Vec3Array> vertices2 = new Vec3Array();
    Vec2dArray *parameterSchnittkurve2 = new Vec2dArray(schnittpunkte2->size()); // mit Nullen
    ref_ptr<Vec2dArray> PARAMETERSCHNITTKURVE2 = new Vec2dArray();

    startParameter = (*parameter)[m_punkt];
    startKoordinaten = (*koordinaten)[m_punkt];

    minabstand = 100;
    zaehler = 0;
    for (unsigned int i = 0; i < schnittpunkte2->size(); i++)
    {
        double abstand = Abstand3(schnittpunkte2->at(i), startKoordinaten);
        if (abstand < minabstand)
        {
            minabstand = abstand;
            zaehler = i;
        }
    }

    for (int i = zaehler; i >= 0; i--)
    {
        Vec2d param = NewtonVerfahren((*schnittpunkte2)[i], startParameter, surf);
        if (param[0] > ua && param[0] < ue && param[1] > va && param[1] < ve)
        {
            (*parameterSchnittkurve2)[i] = param;
            startParameter = param;
        }
    }
    startParameter = (*parameter)[m_punkt];
    for (unsigned int i = zaehler + 1; i < schnittpunkte2->size(); i++)
    {
        Vec2d param = NewtonVerfahren((*schnittpunkte2)[i], startParameter, surf);
        if (param[0] > ua && param[0] < ue && param[1] > va && param[1] < ve)
        {
            (*parameterSchnittkurve2)[i] = param;
            startParameter = param;
        }
    }

    for (unsigned int i = 0; i < parameterSchnittkurve2->size(); i++)
    {
        if ((*parameterSchnittkurve2)[i] != Vec2d(0, 0))
            PARAMETERSCHNITTKURVE2->push_back((*parameterSchnittkurve2)[i]);
    }

    // Schleife ueber alle Schnittpunkte
    for (unsigned int i = 0; i < PARAMETERSCHNITTKURVE2->size(); i++)
    {
        double u = PARAMETERSCHNITTKURVE2->at(i)[0];
        double v = PARAMETERSCHNITTKURVE2->at(i)[1];
        Vec3dArray *pkte = plane->computeSchnittkurve(u, v, 1);
        Vec3d p = (*pkte)[0];
        vertices2->push_back(p);
    }

    ref_ptr<Geode> HptKrR_in_Ebene = new Geode;
    mp_Hauptkruemmungen_in_Ebene = HptKrR_in_Ebene;
    ref_ptr<Geometry> Hptschnittkurve1_in_ebene = new Geometry;
    Hptschnittkurve1_in_ebene->setVertexArray(vertices);
    ref_ptr<Geometry> Hptschnittkurve2_in_ebene = new Geometry;
    Hptschnittkurve2_in_ebene->setVertexArray(vertices2);
    Hptschnittkurve1_in_ebene->setColorArray(colors1.get());
    //Hptschnittkurve1_in_ebene->getOrCreateStateSet()->setAttributeAndModes(pMat1,osg::StateAttribute::ON);
    Hptschnittkurve2_in_ebene->setColorArray(colors2.get());
    //Hptschnittkurve2_in_ebene->getOrCreateStateSet()->setAttributeAndModes(pMat2,osg::StateAttribute::ON);
    Hptschnittkurve1_in_ebene->setColorBinding(Geometry::BIND_OVERALL);
    Hptschnittkurve2_in_ebene->setColorBinding(Geometry::BIND_OVERALL);
    Hptschnittkurve1_in_ebene->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    Hptschnittkurve2_in_ebene->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    Hptschnittkurve1_in_ebene->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, vertices->size()));
    Hptschnittkurve2_in_ebene->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, vertices2->size()));
    osg::StateSet *stateset3 = Hptschnittkurve1_in_ebene->getOrCreateStateSet();
    osg::StateSet *stateset4 = Hptschnittkurve2_in_ebene->getOrCreateStateSet();
    stateset3->setAttribute(lw);
    stateset4->setAttribute(lw);

    HptKrR_in_Ebene->addDrawable(Hptschnittkurve1_in_ebene);
    HptKrR_in_Ebene->addDrawable(Hptschnittkurve2_in_ebene);
    mts_plane->addChild(HptKrR_in_Ebene);
    return true;
}

void HfT_osg_Plugin01_NormalschnittAnimation::Schmiegtangenten(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf)
{
    BoundingBox bs = surf->getBoundingBox();
    float Radius = bs.radius();

    m_schmiegtangente_exist = true;
    Change_Mode(surf, plane, 6);

    Vec2Array *parameter = surf->getParameterValues();
    Vec2 p = (*parameter)[m_punkt];
    Vec3dArray *pkte = surf->computeSchnittkurve(p[0], p[1], 1);
    Vec3d punkt1 = (*pkte)[0];
    Vec3d fu = (*pkte)[1];
    Vec3d fv = (*pkte)[2];
    Vec3dArray *pkte2 = surf->computeSchnittkurve(p[0], p[1], 2);
    Vec3d punkt2 = (*pkte2)[1];
    double L = punkt2[2];
    Vec3d punkt3 = (*pkte2)[2];
    double M = punkt3[0];
    double N = punkt3[1];

    Vec3d richtung1;
    Vec3d richtung2;
    double wurzel = M * M - L * N;
    std::cerr << "wurzel= " << wurzel << std::endl;
    if (wurzel < 0)
    {
        //elliptischer Punkt -> keine Schmiegtangente
        richtung1 = Vec3d(0, 0, 0);
        richtung2 = Vec3d(0, 0, 0);
        m_schmiegtangente_exist = false;
        if (!KruemmungsMethode_aktiv())
            Change_Mode(surf, plane, 5);
    }
    else if (wurzel == 0)
    {
        std::cerr << "wurzel=0 if... " << std::endl;
        //parabolischer Punkt -> Schmiegtangenten fallen zusammen
        richtung1 = Vec3d(0, 0, 0);
        richtung2 = fu * (-M) + fv * L;
        richtung1 = richtung1 / (sqrt(richtung1[0] * richtung1[0] + richtung1[1] * richtung1[1] + richtung1[2] * richtung1[2]));
        richtung2 = richtung2 / (sqrt(richtung2[0] * richtung2[0] + richtung2[1] * richtung2[1] + richtung2[2] * richtung2[2]));
    }
    else if (wurzel > 0)
    {
        //hyperbolischer Punkt -> zwei Schmiegtangenten
        richtung1 = fu * (-M + sqrt(M * M - L * N)) + fv * L;
        richtung2 = fu * (-M - sqrt(M * M - L * N)) + fv * L;
        richtung1 = richtung1 / (sqrt(richtung1[0] * richtung1[0] + richtung1[1] * richtung1[1] + richtung1[2] * richtung1[2]));
        richtung2 = richtung2 / (sqrt(richtung2[0] * richtung2[0] + richtung2[1] * richtung2[1] + richtung2[2] * richtung2[2]));
    }

    ref_ptr<Vec3Array> punkteR1 = new Vec3Array;
    for (int i = -1; i <= 1; i++)
    {
        Vec3 punktImRaum = punkt1 + richtung1 * i * 0.1 * Radius;
        punkteR1->push_back(punktImRaum);
    }
    ref_ptr<Vec3Array> punkteR2 = new Vec3Array;
    for (int i = -1; i <= 1; i++)
    {
        Vec3 punktImRaum = punkt1 + richtung2 * i * 0.1 * Radius;
        punkteR2->push_back(punktImRaum);
    }

    ref_ptr<Geometry> Schmiegtangente1 = new Geometry;
    ref_ptr<Geometry> Schmiegtangente2 = new Geometry;

    ref_ptr<Vec4Array> colors1 = new Vec4Array;
    colors1->push_back(Vec4(1.0f, 1.0f, 1.0f, 1.0f)); //1.0, 0.0, 0.9rosa//1.0f, 0.4588f, 0.094f, 1.0forange//1.0f, 0.631f, 0.9647f//0.8f, 0.0f, 1.0f, 1.0f
    Schmiegtangente1->setColorArray(colors1.get());
    Schmiegtangente2->setColorArray(colors1.get());
    Schmiegtangente1->setColorBinding(Geometry::BIND_OVERALL);
    Schmiegtangente2->setColorBinding(Geometry::BIND_OVERALL);

    //osg::Material *pMat = new osg::Material;
    //pMat->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 1.0f, 1.0f));
    //pMat->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    //Schmiegtangente1->getOrCreateStateSet()->setAttributeAndModes(pMat,osg::StateAttribute::ON);
    //Schmiegtangente2->getOrCreateStateSet()->setAttributeAndModes(pMat,osg::StateAttribute::ON);
    Schmiegtangente1->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    Schmiegtangente2->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    Schmiegtangente1->setVertexArray(punkteR1.get());
    Schmiegtangente2->setVertexArray(punkteR2.get());
    Schmiegtangente1->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, punkteR1->size()));
    Schmiegtangente2->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, punkteR2->size()));
    osg::StateSet *stateset1 = Schmiegtangente1->getOrCreateStateSet();
    osg::StateSet *stateset2 = Schmiegtangente2->getOrCreateStateSet();
    osg::LineWidth *lw = new osg::LineWidth(3.0f);
    lw = new osg::LineWidth(3.0f);
    stateset1->setAttribute(lw);
    stateset2->setAttribute(lw);

    ref_ptr<Geode> Schmiegtangente_auf_Flaeche = new Geode;
    mp_Schmiegtangente_auf_Flaeche = Schmiegtangente_auf_Flaeche;
    Schmiegtangente_auf_Flaeche->addDrawable(Schmiegtangente1);
    Schmiegtangente_auf_Flaeche->addDrawable(Schmiegtangente2);
    mts_surf->addChild(Schmiegtangente_auf_Flaeche);
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Schiefschnitt_Animation(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf, MatrixTransform *mts_plane)
{
    if (!m_pfeil_exist)
        return false;
    if (!mts_surf)
        return false;
    if (!mts_plane)
        return false;

    // Punkte/Normalen auf der Flaeche
    ref_ptr<Vec3Array> points = surf->getPointArray();
    ref_ptr<Vec3Array> normals = surf->getNormalArray();

    Vec2Array *parameter = surf->getParameterValues();
    Vec2 p = (*parameter)[m_punkt];
    Vec3dArray *pkte = surf->computeSchnittkurve(p[0], p[1], 1);
    Vec3d fu = (*pkte)[1];
    Vec3d fv = (*pkte)[2];
    Vec3dArray *pkte2 = surf->computeSchnittkurve(p[0], p[1], 2);
    Vec3d punkt2 = (*pkte2)[0];
    double K = punkt2[0];
    double H = punkt2[1];
    double E = punkt2[2];
    Vec3d punkt3 = (*pkte2)[1];
    double F = punkt3[0];
    double G = punkt3[1];
    double L = punkt3[2];
    Vec3d punkt4 = (*pkte2)[2];
    double M = punkt4[0];
    double N = punkt4[1];

    Vec3d hptKrR1;
    double K1;
    if (F == 0 && M == 0)
    {
        hptKrR1 = fu / (sqrt(fu[0] * fu[0] + fu[1] * fu[1] + fu[2] * fu[2]));
        K1 = L / E;
    }
    else
    {
        double q = sqrt((E * N - G * L) * (E * N - G * L) - 4 * (E * M - F * L) * (F * N - G * M));
        double lambda1 = (-(E * N - G * L) + q) / (2 * (F * N - G * M));
        hptKrR1 = (fu + fv * lambda1) / (sqrt(E + 2 * F * lambda1 + G * lambda1 * lambda1));
        K1 = H - sqrt(abs(H * H - K));
    }

    // Drehung der Ebene in Richtung der 1. HptKrR
    ref_ptr<MatrixTransform> drehung0 = new MatrixTransform();
    ref_ptr<MatrixTransform> drehung00 = new MatrixTransform();
    Quat rotation1;
    rotation1.makeRotate(Vec3d(1.0f, 0.0f, 0.0f), hptKrR1);
    drehung0->setMatrix(Matrix(rotation1));
    drehung00->setMatrix(Matrix(rotation1));

    Vec3 zachse = Vec3(0, 0, 1);
    Vec3 zachseneu = zachse * drehung0->getMatrix();

    // Drehung in Richtung der Normalen
    ref_ptr<MatrixTransform> drehung1 = new MatrixTransform();
    ref_ptr<MatrixTransform> drehung11 = new MatrixTransform();
    Vec3 normal = Vec3((*normals)[m_punkt]);
    Quat rotation2;
    rotation2.makeRotate(zachseneu, normal);
    drehung1->setMatrix(Matrix(rotation2));
    drehung11->setMatrix(Matrix(rotation2));

    // Schief-Drehung
    ref_ptr<MatrixTransform> drehung2 = new MatrixTransform();
    ref_ptr<MatrixTransform> drehung22 = new MatrixTransform();
    Quat rotation3;
    rotation3.makeRotate(DegreesToRadians(m_schiefwinkel), hptKrR1);
    drehung2->setMatrix(Matrix(rotation3));
    drehung22->setMatrix(Matrix(rotation3));

    // Drehung zusammen
    ref_ptr<MatrixTransform> drehung3 = new MatrixTransform();
    ref_ptr<MatrixTransform> drehung33 = new MatrixTransform();
    Matrix d = drehung2->getMatrix() * drehung1->getMatrix();
    drehung3->setMatrix(d);
    drehung33->setMatrix(d);

    // Verschiebung in den ausgewählten Punkt der Fläche
    ref_ptr<MatrixTransform> verschiebung1 = new MatrixTransform();
    ref_ptr<MatrixTransform> verschiebung11 = new MatrixTransform();
    Matrix m;
    m.makeTranslate((*points)[m_punkt]);
    verschiebung1->setMatrix(m);
    verschiebung11->setMatrix(m);

    if (!m_schiefebene_exist)
    {
        ref_ptr<Switch> sw_schiefebenen = new Switch;
        m_schiefEbenen = sw_schiefebenen;
        ref_ptr<Switch> sw_schiefschnittkurven = new Switch;
        m_SchiefSchnittkurven = sw_schiefschnittkurven;
        ref_ptr<Switch> sw_schiefschnittkurven2 = new Switch;
        m_SchiefSchnittkurven_in_Ebene = sw_schiefschnittkurven2;
        ref_ptr<Switch> sw_kreise = new Switch;
        m_Kruemmungskreise = sw_kreise;
    }

    if (m_schiefebene_exist == false)
        mts_surf->addChild(m_schiefEbenen);
    if (m_schiefebene_exist == false)
        mts_surf->addChild(m_Kruemmungskreise);

    ref_ptr<Geode> ebene = new Geode;
    ebene->addDrawable(Create_Plane(surf, verschiebung1, drehung3, drehung0));
    drehung0->addChild(ebene);
    drehung3->addChild(drehung0);
    verschiebung1->addChild(drehung3);
    m_schiefEbenen->addChild(verschiebung1);

    ref_ptr<Geode> kreise = new Geode;
    kreise->addDrawable(Create_Kruemmungskreis(K1));
    drehung00->addChild(kreise);
    drehung33->addChild(drehung00);
    verschiebung11->addChild(drehung33);
    m_Kruemmungskreise->addChild(verschiebung11);

    m_schiefebene_exist = true;
    if (!m_schiefschnittkurve_exist)
    {
        Change_Mode(surf, plane, 6);
    }

    ref_ptr<Geode> schnittkurve = new Geode;
    schnittkurve->addDrawable(Schnittkurvenberechnung_auf_Flaeche(surf)); // -> mit Intersection Visitor
    m_SchiefSchnittkurven->addChild(schnittkurve);
    if (m_schiefschnittkurve_exist == false)
        mts_surf->addChild(m_SchiefSchnittkurven);

    ref_ptr<Geode> schnittkurve_in_ebene = new Geode;
    schnittkurve_in_ebene->addDrawable(Schnittkurvenberechnung_in_Ebene(surf, plane)); // -> mit Newton
    m_SchiefSchnittkurven_in_Ebene->addChild(schnittkurve_in_ebene);
    if (m_schiefschnittkurve_exist == false)
        mts_plane->addChild(m_SchiefSchnittkurven_in_Ebene);

    m_schiefschnittkurve_exist = true;
    return true;
}

ref_ptr<Drawable> HfT_osg_Plugin01_NormalschnittAnimation::Create_Kruemmungskreis(double Kruemmung)
{
    double Kneu = Kruemmung / cos(DegreesToRadians(m_schiefwinkel));
    double radius = abs(1 / Kneu);

    // Kreis erstellen
    ref_ptr<Vec3Array> vertices = new Vec3Array;
    for (double t = 0; t <= 2 * PI; t = t + 2 * PI / 60)
    {
        double x = radius * cos(t);
        double z = -radius + radius * sin(t);
        vertices->push_back(Vec3f(x, 0, z));
    }

    ref_ptr<Geometry> kreis = new Geometry;

    ref_ptr<Vec4Array> colors = new Vec4Array;
    colors->push_back(Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    kreis->setColorArray(colors.get());
    kreis->setColorBinding(Geometry::BIND_OVERALL);
    kreis->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    //osg::Material *pMat = new osg::Material;
    //pMat->setEmission(osg::Material::FRONT_AND_BACK, Vec4(1.0f, 0.0f, 0.0f, 1.0f));
    //pMat->setShininess(osg::Material::FRONT_AND_BACK, 5.0f);
    //kreis->getOrCreateStateSet()->setAttributeAndModes(pMat,osg::StateAttribute::ON);
    kreis->setVertexArray(vertices.get());
    kreis->addPrimitiveSet(new DrawArrays(GL_LINE_STRIP, 0, vertices->size()));
    osg::StateSet *stateset = kreis->getOrCreateStateSet();
    osg::LineWidth *lw = new osg::LineWidth(3.0f);
    lw = new osg::LineWidth(3.0f);
    stateset->setAttribute(lw);
    m_kruemmungskreise_exist = true;
    return kreis.get();
}

void HfT_osg_Plugin01_NormalschnittAnimation::Meusnier_Kugel(HfT_osg_Plugin01_ParametricSurface *surf, int i, MatrixTransform *mts_surf)
{
    // Punkte/Normalen auf der Flaeche
    ref_ptr<Vec3Array> points = surf->getPointArray();
    ref_ptr<Vec3Array> normals = surf->getNormalArray();

    Vec2Array *parameter = surf->getParameterValues();
    Vec2 p = (*parameter)[m_punkt];
    Vec3dArray *pkte = surf->computeSchnittkurve(p[0], p[1], 1);
    Vec3d fu = (*pkte)[1];
    Vec3d fv = (*pkte)[2];
    Vec3dArray *pkte2 = surf->computeSchnittkurve(p[0], p[1], 2);
    Vec3d punkt2 = (*pkte2)[0];
    double K = punkt2[0];
    double H = punkt2[1];
    double E = punkt2[2];
    Vec3d punkt3 = (*pkte2)[1];
    double F = punkt3[0];
    double G = punkt3[1];
    double L = punkt3[2];
    Vec3d punkt4 = (*pkte2)[2];
    double M = punkt4[0];
    double N = punkt4[1];

    Vec3d hptKrR1;
    double K1;
    if (F == 0 && M == 0)
    {
        hptKrR1 = fu / (sqrt(fu[0] * fu[0] + fu[1] * fu[1] + fu[2] * fu[2]));
        K1 = L / E;
    }
    else
    {
        double q = sqrt((E * N - G * L) * (E * N - G * L) - 4 * (E * M - F * L) * (F * N - G * M));
        double lambda1 = (-(E * N - G * L) + q) / (2 * (F * N - G * M));
        hptKrR1 = (fu + fv * lambda1) / (sqrt(E + 2 * F * lambda1 + G * lambda1 * lambda1));
        K1 = H - sqrt(abs(H * H - K));
    }

    // Drehung der Ebene in Richtung der 1. HptKrR
    ref_ptr<MatrixTransform> drehung0 = new MatrixTransform();
    Quat rotation1;
    rotation1.makeRotate(Vec3d(1.0f, 0.0f, 0.0f), hptKrR1);
    drehung0->setMatrix(Matrix(rotation1));

    Vec3 zachse = Vec3(0, 0, 1);
    Vec3 zachseneu = zachse * drehung0->getMatrix();

    // Drehung in Richtung der Normalen
    ref_ptr<MatrixTransform> drehung1 = new MatrixTransform();
    Vec3 normal = Vec3((*normals)[m_punkt]);
    Quat rotation2;
    rotation2.makeRotate(zachseneu, normal);
    drehung1->setMatrix(Matrix(rotation2));

    // Schief-Drehung
    ref_ptr<MatrixTransform> drehung2 = new MatrixTransform();
    Quat rotation3;
    rotation3.makeRotate(DegreesToRadians(0.0), hptKrR1);
    drehung2->setMatrix(Matrix(rotation3));

    // Verschiebung in den ausgewählten Punkt der Fläche
    ref_ptr<MatrixTransform> verschiebung = new MatrixTransform();
    Matrix m;
    m.makeTranslate((*points)[m_punkt]);
    verschiebung->setMatrix(m);

    m_meusnierkugel_exist = true;
    ref_ptr<Switch> ms = new Switch;
    m_MeusnierKugel = ms;
    ref_ptr<Geode> meusnierkugel = new Geode;
    double radius = abs(1 / K1);
    ref_ptr<ShapeDrawable> kugel = new ShapeDrawable;
    kugel->setShape(new Sphere(Vec3(0.0f, 0.0f, -radius), radius));

    if (i == 1)
    {
        osg::Material *pMat = new osg::Material;
        pMat->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 1.0f, 1.0f)); //1.0, 0.976, 0.7412, 1.0//0.8, 0.0, 1.0, 1.0
        pMat->setShininess(osg::Material::FRONT, 5.0f);
        kugel->getOrCreateStateSet()->setAttributeAndModes(pMat, osg::StateAttribute::ON);
    }
    if (i == 2)
    {
        osg::StateSet *pStateSet = kugel->getOrCreateStateSet();
        pStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
        pStateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

        pStateSet->setMode(GL_CULL_FACE, osg::StateAttribute::ON);
        pStateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);
        pStateSet->setRenderBinDetails(1, "RenderBin");

        osg::Material *pMat2 = new osg::Material;
        pMat2->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 1.0f, 0.5f)); //1.0, 0.0, 1.0, 0.5
        pMat2->setShininess(osg::Material::FRONT, 5.0f);
        kugel->getOrCreateStateSet()->setAttributeAndModes(pMat2, osg::StateAttribute::ON);
    }

    meusnierkugel->addDrawable(kugel.get());
    drehung0->addChild(meusnierkugel);
    drehung1->addChild(drehung0);
    drehung2->addChild(drehung1);
    verschiebung->addChild(drehung2);
    m_MeusnierKugel->addChild(verschiebung);
    mts_surf->addChild(m_MeusnierKugel);
}

void HfT_osg_Plugin01_NormalschnittAnimation::Remove_Pfeil(MatrixTransform *mts_surf, MatrixTransform *mts_plane)
{
    mts_surf->removeChild(m_Pfeil);
    mts_plane->removeChild(m_Pfeil_in_Ebene);
    m_pfeil_exist = false;
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Remove_NormalEbenen(MatrixTransform *mts_surf)
{
    mts_surf->removeChild(m_normalEbenen);
    m_normalebene_exist = false;
    return true;
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Remove_Schnittkurve(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf, MatrixTransform *mts_plane)
{
    mts_surf->removeChild(m_Schnittkurven);
    mts_plane->removeChild(m_Schnittkurven_in_Ebene);
    m_schnittkurve_exist = false;
    m_normalwinkel = 0;
    if (!KruemmungsMethode_aktiv())
        Change_Mode(surf, plane, 5);
    return true;
}

void HfT_osg_Plugin01_NormalschnittAnimation::Remove_Hauptkruemmungsrichtungen(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf)
{
    m_hauptkruemmungsrichtung_exist = false;
    if (!KruemmungsMethode_aktiv())
        Change_Mode(surf, plane, 5);
    mts_surf->removeChild(m_HptKrPfeile);
}

void HfT_osg_Plugin01_NormalschnittAnimation::Remove_Hauptkruemmungen(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf, MatrixTransform *mts_plane)
{
    m_hauptkruemmung_exist = false;
    if (!KruemmungsMethode_aktiv())
        Change_Mode(surf, plane, 5);
    mts_surf->removeChild(mp_Hauptkruemmungen_auf_Flaeche);
    mts_plane->removeChild(mp_Hauptkruemmungen_in_Ebene);
}

void HfT_osg_Plugin01_NormalschnittAnimation::Remove_Schmiegtangenten(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf)
{
    m_schmiegtangente_exist = false;
    if (!KruemmungsMethode_aktiv())
        Change_Mode(surf, plane, 5);
    mts_surf->removeChild(mp_Schmiegtangente_auf_Flaeche);
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Remove_SchiefEbenen(MatrixTransform *mts_surf)
{
    mts_surf->removeChild(m_schiefEbenen);
    m_schiefebene_exist = false;
    return true;
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Remove_SchiefSchnittkurve(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf, MatrixTransform *mts_plane)
{
    mts_surf->removeChild(m_SchiefSchnittkurven);
    mts_plane->removeChild(m_SchiefSchnittkurven_in_Ebene);
    m_schiefschnittkurve_exist = false;
    m_schiefwinkel = 0;
    if (!KruemmungsMethode_aktiv())
        Change_Mode(surf, plane, 5);
    return true;
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Remove_MeusnierKugel(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf)
{
    mts_surf->removeChild(m_MeusnierKugel);
    m_meusnierkugel_exist = false;
    if (!KruemmungsMethode_aktiv())
        Change_Mode(surf, plane, 5);
    return true;
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Remove_Kruemmungskreise(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf)
{
    mts_surf->removeChild(m_Kruemmungskreise);
    m_kruemmungskreise_exist = false;
    if (!KruemmungsMethode_aktiv())
        Change_Mode(surf, plane, 5);
    return true;
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Clear(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, MatrixTransform *mts_surf, MatrixTransform *mts_plane)
{
    if (m_pfeil_exist)
    {
        Remove_Pfeil(mts_surf, mts_plane);
        m_punkt = -1;
    }
    if (m_normalebene_exist)
        Remove_NormalEbenen(mts_surf);
    if (m_schiefebene_exist)
        Remove_SchiefEbenen(mts_surf);
    if (m_schnittkurve_exist)
        Remove_Schnittkurve(surf, plane, mts_surf, mts_plane);
    if (m_schiefschnittkurve_exist)
        Remove_SchiefSchnittkurve(surf, plane, mts_surf, mts_plane);
    if (m_hauptkruemmung_exist)
        Remove_Hauptkruemmungen(surf, plane, mts_surf, mts_plane);
    if (m_hauptkruemmungsrichtung_exist)
        Remove_Hauptkruemmungsrichtungen(surf, plane, mts_surf);
    if (m_schmiegtangente_exist)
        Remove_Schmiegtangenten(surf, plane, mts_surf);
    if (m_kruemmungskreise_exist)
        Remove_Kruemmungskreise(surf, plane, mts_surf);
    if (m_meusnierkugel_exist)
        Remove_MeusnierKugel(surf, plane, mts_surf);
    m_schiefwinkel = 0;
    m_normalwinkel = 0;
    return true;
}

bool HfT_osg_Plugin01_NormalschnittAnimation::KruemmungsMethode_aktiv()
{
    if (m_normalebene_exist)
        return true;
    else if (m_schiefebene_exist)
        return true;
    else if (m_schnittkurve_exist)
        return true;
    else if (m_schiefschnittkurve_exist)
        return true;
    else if (m_hauptkruemmung_exist)
        return true;
    else if (m_hauptkruemmungsrichtung_exist)
        return true;
    else if (m_schmiegtangente_exist)
        return true;
    else if (m_kruemmungskreise_exist)
        return true;
    else if (m_meusnierkugel_exist)
        return true;
    else
        return false;
}

Vec2d HfT_osg_Plugin01_NormalschnittAnimation::NewtonVerfahren(Vec3 schnittpunkt, Vec2 w1, HfT_osg_Plugin01_ParametricSurface *surf)
{
    Vec2d w2;
    Vec2d wopt;
    Vec3dArray *pkte = new Vec3dArray();
    double d2 = 100;
    double d3 = 100;
    double abstand;

    while (d2 > 0.01)
    {
        pkte = surf->computeSchnittkurve(w1[0], w1[1], 1);
        Vec3d f = (*pkte)[0];
        Vec3d fu = (*pkte)[1];
        Vec3d fv = (*pkte)[2];
        Vec3d fuv = (*pkte)[3];
        Vec3d fuu = (*pkte)[4];
        Vec3d fvv = (*pkte)[5];

        abstand = Abstand3(schnittpunkt, f);
        if (abstand < d3)
        {
            d3 = abstand;
            wopt = w1;
        }

        double H1u = Skalarprodukt(-fu, -fu) + Skalarprodukt(schnittpunkt - f, -fuu);
        double H2u = Skalarprodukt(-fu, -fv) + Skalarprodukt(schnittpunkt - f, -fuv);
        double H1v = Skalarprodukt(-fv, -fu) + Skalarprodukt(schnittpunkt - f, -fuv);
        double H2v = Skalarprodukt(-fv, -fv) + Skalarprodukt(schnittpunkt - f, -fvv);

        double det = (H1u * H2v) - (H1v * H2u);

        double H1 = Skalarprodukt(schnittpunkt - f, -fu);
        double H2 = Skalarprodukt(schnittpunkt - f, -fv);

        double v1 = (H2v / det * H1) + (-H1v / det * H2);
        double v2 = (-H2u / det * H1) + (H1u / det * H2);

        Vec2d D = Vec2d(v1, v2);
        w2 = w1 - D;

        pkte = surf->computeSchnittkurve(w2[0], w2[1], 1);
        Vec3d f2 = (*pkte)[0];

        abstand = Abstand3(schnittpunkt, f2);
        if (abstand < d3)
        {
            d3 = abstand;
            wopt = w2;
        }

        d2 = Abstand2(w1, w2);
        w1 = w2;
    }

    if (wopt[0] != w1[0] && wopt[1] != w1[1])
    {
        w2 = wopt;
    }
    return Vec2d(w2[0], w2[1]);
}

bool HfT_osg_Plugin01_NormalschnittAnimation::Change_Mode(HfT_osg_Plugin01_ParametricSurface *surf, HfT_osg_Plugin01_ParametricSurface *plane, int mode1)
{
    surf->recomputeMode((SurfMode)mode1);
    plane->recomputeMode((SurfMode)mode1);
    return true;
}

double HfT_osg_Plugin01_NormalschnittAnimation::Abstand2(Vec2d p1, Vec2d p2)
{
    double x1 = p1[0];
    double y1 = p1[1];
    double x2 = p2[0];
    double y2 = p2[1];
    double abstand = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    return abstand;
}

float HfT_osg_Plugin01_NormalschnittAnimation::Abstand3(Vec3 p1, Vec3 p2)
{
    float x1 = p1[0];
    float y1 = p1[1];
    float z1 = p1[2];
    float x2 = p2[0];
    float y2 = p2[1];
    float z2 = p2[2];
    float abstand = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2));
    return abstand;
}

double HfT_osg_Plugin01_NormalschnittAnimation::Skalarprodukt(Vec3d p1, Vec3d p2)
{
    float x1 = p1[0];
    float y1 = p1[1];
    float z1 = p1[2];
    float x2 = p2[0];
    float y2 = p2[1];
    float z2 = p2[2];
    double s = (x1 * x2) + (y1 * y2) + (z1 * z2);
    return s;
}
