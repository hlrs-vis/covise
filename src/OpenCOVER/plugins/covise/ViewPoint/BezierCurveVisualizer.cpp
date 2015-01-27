/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * BezierPlugin.cpp
 *
 *  Created on: Dec 3, 2010
 *      Author: tm_te
 */
#include "BezierCurveVisualizer.h"
#include <math.h>

using namespace osg;
using namespace opencover;

BezierCurveVisualizer::BezierCurveVisualizer(ref_ptr<MatrixTransform> master, Computation computation)
{
    bezierDCS = new MatrixTransform;
    lineDCS = NULL;

    this->computation = computation;

    showCasteljauGeom = false;
    showCurveGeom = false;
    showControlPolygonGeom = false;
    showTangentBeginGeom = false;
    showTangentEndGeom = false;
    t = 0;

    master->addChild(bezierDCS.get());
}

BezierCurveVisualizer::~BezierCurveVisualizer()
{
    // delete nodes and scenegraph
    if (lineDCS)
        lineDCS = 0;

    if (bezierDCS)
    {
        if (bezierDCS->getNumParents())
        {
            bezierDCS->getParent(0)->removeChild(bezierDCS);
            //			pfDelete(flightPathDCS);                  // noch nötig ?
        }
    }
}

void BezierCurveVisualizer::showCurve(bool state)
{
    showCurveGeom = state;
}

void BezierCurveVisualizer::showCasteljau(bool state)
{
    showCasteljauGeom = state;
}

void BezierCurveVisualizer::showControlPolygon(bool state)
{
    showControlPolygonGeom = state;
}

void BezierCurveVisualizer::showTangentBegin(bool state)
{
    showTangentBeginGeom = state;
}

void BezierCurveVisualizer::showTangentEnd(bool state)
{
    showTangentEndGeom = state;
}

void BezierCurveVisualizer::addControlPoint(Vec3 newPoint)
{
    controlPoints.push_back(newPoint);
}

void BezierCurveVisualizer::addVectorOfControlPoints(std::vector<Vec3> newPoints)
{
    for (int i = 0; i < newPoints.size(); i++)
    {
        addControlPoint(newPoints[i]);
    }
}

void BezierCurveVisualizer::removeControlPoint()
{
    controlPoints.pop_back();
}

void BezierCurveVisualizer::removeAllControlPoints()
{
    controlPoints.clear();
}

void BezierCurveVisualizer::updateControlPoints(std::vector<Vec3> newPointList)
{
    removeAllControlPoints();
    for (int i = 0; i < newPointList.size(); i++)
    {
        addControlPoint(newPointList[i]);
    }
}

void BezierCurveVisualizer::ChangeControlPoint(Vec3 changePoint, int index)
{
    controlPoints[index] = changePoint;
}

void BezierCurveVisualizer::drawLine(ref_ptr<MatrixTransform> master, Vec3 point1, Vec3 point2, Vec4 color)
{
    drawLine(master, point1, point2, color, 2);
}

void BezierCurveVisualizer::drawLine(ref_ptr<MatrixTransform> master, Vec3 point1, Vec3 point2, Vec4 color, float linewidth)
{
    ref_ptr<Vec3Array> lineCoordList = new Vec3Array(2);
    lineCoordList.get()->at(0) = point1;
    lineCoordList.get()->at(1) = point2;

    ref_ptr<Vec4Array> lineColorList = new Vec4Array;
    lineColorList->push_back(color);

    ref_ptr<Geometry> lineGeoset = new Geometry;
    ref_ptr<StateSet> lineGeoset_state = lineGeoset->getOrCreateStateSet();
    lineGeoset->setVertexArray(lineCoordList);
    lineGeoset->setColorArray(lineColorList);
    lineGeoset->setColorBinding(Geometry::BIND_OVERALL);
    lineGeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    ref_ptr<LineWidth> lw = new LineWidth(linewidth);
    lineGeoset_state->setAttribute(lw);

    ref_ptr<Material> mtl = new Material;
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.f));
    mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    mtl->setShininess(Material::FRONT_AND_BACK, 16.f);

    //	lineGeoState->setAttributeAndModes(material, osg::StateAttribute::ON);
    lineGeoset_state->setAttribute(mtl.get());

    ref_ptr<Geode> lineGeode = new Geode;
    lineGeode->addDrawable(lineGeoset);

    master->addChild(lineGeode);
}

void BezierCurveVisualizer::updateGeometry()
{
    // delete old curve
    if (lineDCS)
    {
        bezierDCS->removeChild(lineDCS.get());
        //		pfDelete(lineDCS);                 // nötig ?
    }

    lineDCS = new MatrixTransform;
    bezierDCS->addChild(lineDCS.get());

    // cancel if only one ControlPoint
    if (controlPoints.size() < 2)
    {
        return;
    }

    if (showCurveGeom)
    {
        drawCurve(lineDCS);
    }

    if (showCasteljauGeom)
    {
        visualizeCasteljau(lineDCS, t);
    }

    if (showControlPolygonGeom)
    {
        Vec4 colorBlue = Vec4(0.0, 0.0, 1.0, 1.0);
        Vec4 colorPurple = Vec4(0.7, 0, 0.7, 1.0);
        float linewidth = 4.0;

        if (showTangentBeginGeom)
        {
            drawLine(lineDCS, controlPoints[0], controlPoints[1], colorPurple, linewidth);
        }
        else
        {
            drawLine(lineDCS, controlPoints[0], controlPoints[1], colorBlue);
        }

        if (showTangentEndGeom)
        {
            drawLine(lineDCS, controlPoints[controlPoints.size() - 2], controlPoints[controlPoints.size() - 1], colorPurple, linewidth);
        }
        else
        {
            drawLine(lineDCS, controlPoints[controlPoints.size() - 2], controlPoints[controlPoints.size() - 1], colorBlue);
        }

        if (controlPoints.size() > 2)
        {
            for (int i = 1; i < controlPoints.size() - 2; i++)
            {
                drawLine(lineDCS, controlPoints[i], controlPoints[i + 1], colorBlue);
            }
        }
    }
}

void BezierCurveVisualizer::drawCurve(ref_ptr<MatrixTransform> master)
{
    Vec4 colorRed = Vec4(1.0, 0.0, 0.0, 1.0);
    double t;

    switch (computation)
    {
    case APROXIMATION:
    {
        std::vector<Vec3> points = controlPoints;
        Vec3 p1 = casteljauAproximation(controlPoints, 0.0);
        Vec3 p2 = casteljauAproximation(controlPoints, 0.005);
        drawLine(master, p1, p2, colorRed);
        for (t = 0.01; t <= 1.0; t += 0.005)
        {
            p1 = p2;
            p2 = casteljauAproximation(controlPoints, t);
            drawLine(master.get(), p1, p2, colorRed);
        }
        break;
    }
    case EXACT_COMPUTATION:
    {
        Vec3 p1 = computePointOnCurve(controlPoints, 0.0);
        Vec3 p2 = computePointOnCurve(controlPoints, 0.001);
        drawLine(master, p1, p2, colorRed);
        for (t = 0.002; t <= 1.0; t += 0.001)
        {
            p1 = p2;
            p2 = computePointOnCurve(controlPoints, t);
            drawLine(master.get(), p1, p2, colorRed);
        }
        break;
    }
    //takes just the first 4 points of controlPoints to compute the curve
    case CUBIC_APROXIMATION:
    {
        if (controlPoints.size() < 4)
        {
            if (cover->debugLevel(3))
                fprintf(stderr, "Nicht genügend Kontrollpunkte vorhanden, um eine kubische Bezierkurve zu erzeugen.\n");
        }
        else
        {
            Vec3 p1 = controlPoints[0];
            Vec3 p2 = controlPoints[1];
            Vec3 p3 = controlPoints[2];
            Vec3 p4 = controlPoints[3];

            cubicCasteljauAproximation(master, p1, p2, p3, p4, 4);
        }

        break;
    }
    default:
    {
    }
    }
}

void BezierCurveVisualizer::visualizeCasteljau(ref_ptr<MatrixTransform> master, double t)
{
    Vec4 colors[9];
    colors[0] = Vec4(0.0, 1.0, 0.0, 1.0); //green
    colors[1] = Vec4(0.93, 0.71, 0.13, 1.0); //gold
    colors[2] = Vec4(0.0, 0.0, 0.5, 1.0); // steel blue
    colors[3] = Vec4(0.6, 0.2, 0.8, 1.0); //dark orchid
    colors[4] = Vec4(0.0, 0.75, 1.0, 1.0); //light blue
    colors[5] = Vec4(1.0, 0.39, 0.28, 1.0); //light red
    colors[6] = Vec4(0.0, 0.55, 0.0, 1.0); //dark green
    colors[7] = Vec4(1.0, 1.0, 0.0, 1.0); //yellow
    colors[8] = Vec4(1.0, 0.0, 1.0, 1.0); //purple

    std::vector<Vec3> points = controlPoints;
    int steps = points.size() - 1;

    for (int n = 0; n < steps; n++)
    {
        for (int i = 0; i < points.size() - 1 - n; i++)
        {
            points[i] *= (1 - t);
            Vec3 tmp = points[i + 1];
            tmp *= t;
            points[i] += tmp;
        }
        for (int j = 0; j < points.size() - 2 - n; j++)
        {
            drawLine(master, points[j], points[j + 1], colors[n]);
        }
    }

    ref_ptr<Vec4Array> lineColorList = new Vec4Array;
    lineColorList->push_back(colors[2]);

    osg::Sphere *unitSphere = new osg::Sphere(points[0], 0.3);
    osg::ShapeDrawable *unitSphereDrawable = new osg::ShapeDrawable(unitSphere);
    ref_ptr<StateSet> sphere_state = unitSphereDrawable->getOrCreateStateSet();
    unitSphereDrawable->setColor(colors[controlPoints.size() - 3]);

    ref_ptr<Material> mtl = new Material;
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.f));
    mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    mtl->setShininess(Material::FRONT_AND_BACK, 16.f);
    sphere_state->setAttribute(mtl.get());

    ref_ptr<Geode> sphereGeode = new Geode;
    sphereGeode->addDrawable(unitSphereDrawable);

    master->addChild(sphereGeode.get());
}

//Aproximation der Bezierkurve über den Casteljau-Algorithmus
Vec3 BezierCurveVisualizer::casteljauAproximation(std::vector<Vec3> points, double t)
{
    int steps = points.size() - 1;
    for (int n = 0; n < steps; n++)
    {
        for (int i = 0; i < points.size() - 1 - n; i++)
        {
            points[i] *= (1 - t);
            Vec3 tmp = points[i + 1];
            tmp *= t;
            points[i] += tmp;
        }
    }
    return points[0];
}

//Exakte Berechnung der Bezierkurve
Vec3 BezierCurveVisualizer::computePointOnCurve(std::vector<Vec3> controlPoints, double t)
{
    int grad = controlPoints.size() - 1;
    Vec3 erg = Vec3(0, 0, 0);
    for (int i = 0; i <= grad; i++)
    {
        Vec3 tmp = controlPoints[i];
        tmp *= bernstein(grad, i, t);
        erg += tmp;
    }
    return erg;
}

double BezierCurveVisualizer::bernstein(double n, double i, double t)
{
    double erg;
    erg = binomKoeff(n, i) * pow(t, i) * pow((1 - t), (n - i));
    return erg;
}

double BezierCurveVisualizer::binomKoeff(double n, double k)
{
    double zaehler = factorial(n);
    double nenner = factorial(k) * factorial(n - k);
    return zaehler / nenner;
}

double BezierCurveVisualizer::factorial(double k)
{
    double n = k;
    double erg = 1;
    while (n != 0)
    {
        erg *= n;
        n--;
    }
    return erg;
}

//Es werden aus der Liste "controlPoints" nur die ersten 4 Punkte benutzt, um eine kubische Bezierkurve zu erzeugen
//speziell für das Plugin "Viewpoints" vonnöten!
void BezierCurveVisualizer::cubicCasteljauAproximation(ref_ptr<MatrixTransform> master, Vec3 p1, Vec3 p2,
                                                       Vec3 p3, Vec3 p4, int steps)
{

    if (steps == 0)
    {
        Vec4 colorRed = Vec4(1.0, 0.0, 0.0, 1.0);
        drawLine(master, p1, p2, colorRed);
        drawLine(master, p2, p3, colorRed);
        drawLine(master, p3, p4, colorRed);
        return;
    }
    else
    {
        steps--;
        Vec3 p12 = (p1 + p2) / 2;
        Vec3 p23 = (p2 + p3) / 2;
        Vec3 p34 = (p3 + p4) / 2;
        Vec3 p123 = (p12 + p23) / 2;
        Vec3 p234 = (p23 + p34) / 2;
        Vec3 p1234 = (p123 + p234) / 2;

        cubicCasteljauAproximation(master, p1, p12, p123, p1234, steps);
        cubicCasteljauAproximation(master, p1234, p234, p34, p4, steps);
    }
}

void BezierCurveVisualizer::setT(double newT)
{
    if (t >= 0 && t <= 1)
    {
        t = newT;
    }
    else
    {
        t = 0;
        fprintf(stderr, "t liegt nicht zwischen 0 und 1! t wird auf 0 gesetzt.");
    }
}

std::vector<Vec3> BezierCurveVisualizer::getAllControlPoints()
{
    return controlPoints;
}

void BezierCurveVisualizer::degreeElevation()
{
    if (controlPoints.size() < 3)
        return;

    std::vector<Vec3> newControlPoints;
    int n = controlPoints.size() - 1; //Grad der alten Kurve

    newControlPoints.push_back(controlPoints[0]);
    for (int i = 1; i <= n + 1; i++)
    {
        double alpha = (double)i / (n + 1);

        Vec3 tmp1;
        tmp1 = controlPoints[i - 1];
        tmp1 *= alpha;

        Vec3 tmp2;
        tmp2 = controlPoints[i];
        tmp2 *= (1 - alpha);

        tmp1 += tmp2;
        newControlPoints.push_back(tmp1);
    }

    updateControlPoints(newControlPoints);
}

std::vector<Vec3> BezierCurveVisualizer::degreeReductionForward()
{

    //Gradreduktion nach Forest vorwaerts
    std::vector<Vec3> newControlPointsForward;
    int n = controlPoints.size() - 2; //Grad der neuen Kurve
    Vec3 tmp1, tmp2;
    double alpha, beta;

    newControlPointsForward.push_back(controlPoints[0]);
    for (int i = 1; i <= n; i++)
    {
        alpha = (double)(n + 1) / (n + 1 - i);
        beta = (double)i / (n + 1 - i);

        tmp1 = controlPoints[i];
        tmp1 *= alpha;

        tmp2 = newControlPointsForward[i - 1];
        tmp2 *= beta;

        tmp1 -= tmp2;

        newControlPointsForward.push_back(tmp1);
    }

    return newControlPointsForward;
}

std::vector<Vec3> BezierCurveVisualizer::degreeReductionBackward()
{

    //Gradreduktion nach Forest rueckwaerts
    std::vector<Vec3> newControlPointsBackwardPrep;

    int n = controlPoints.size() - 2; //Grad der neuen Kurve
    Vec3 tmp1, tmp2;
    double alpha, beta;

    newControlPointsBackwardPrep.push_back(controlPoints[controlPoints.size() - 1]);
    for (int i = 1; i <= n; i++)
    {
        alpha = (double)(n + 1) / (n + 1 - i);
        beta = (double)i / (n + 1 - i);

        tmp1 = controlPoints[n + 1 - i];
        tmp1 *= alpha;

        tmp2 = newControlPointsBackwardPrep[i - 1];
        tmp2 *= beta;

        tmp1 -= tmp2;

        newControlPointsBackwardPrep.push_back(tmp1);
    }

    std::vector<Vec3> newControlPointsBackward;
    for (int i = newControlPointsBackwardPrep.size() - 1; i >= 0; i--)
    {
        newControlPointsBackward.push_back(newControlPointsBackwardPrep[i]);
    }

    return newControlPointsBackward;
}

void BezierCurveVisualizer::degreeReductionForest()
{
    if (controlPoints.size() < 3)
        return;

    int n = controlPoints.size() - 2; //Grad der neuen Kurve
    Vec3 tmp1, tmp2;

    std::vector<Vec3> newControlPointsForward = degreeReductionForward();
    std::vector<Vec3> newControlPointsBackward = degreeReductionBackward();
    std::vector<Vec3> newControlPointsFinal;

    //Berechne Kombination aus beiden obigen Kurven
    //--------------------------------------------------------------------------------------------------------------
    //Neue Kurve hat ungeraden Grad
    if (n % 2 == 1)
    {
        //uebernehme erste Haelfte der vorwaerts Gradreduktion
        for (int i = 0; i < (n + 1) / 2; i++)
        {
            newControlPointsFinal.push_back(newControlPointsForward[i]);
        }

        //uebernehme zweite Haelfte der rueckwaerts Gradreduktion
        for (int i = (n + 1) / 2; i < n + 1; i++)
        {
            newControlPointsFinal.push_back(newControlPointsBackward[i]);
        }
    }

    //neue Kurve hat geraden Grad
    else
    {
        //uebernehme erste Haelfte der vorwaerts Gradreduktion
        for (int i = 0; i < n / 2; i++)
        {
            newControlPointsFinal.push_back(newControlPointsForward[i]);
        }

        //mittle den Punkt, der nicht zugeordnet werden kann
        tmp1 = newControlPointsForward[n / 2];
        tmp1 *= 0.5;

        tmp2 = newControlPointsBackward[n / 2];
        tmp2 *= 0.5;

        tmp1 += tmp2;
        newControlPointsFinal.push_back(tmp1);

        //uebernehme zweite Haelfte der rueckwaerts Gradreduktion
        for (int i = (n / 2) + 1; i < n + 1; i++)
        {
            newControlPointsFinal.push_back(newControlPointsBackward[i]);
        }
    }

    updateControlPoints(newControlPointsFinal);
}

void BezierCurveVisualizer::degreeReductionFarin()
{
    if (controlPoints.size() < 3)
        return;

    int n = controlPoints.size() - 2; //Grad der neuen Kurve
    Vec3 tmp1, tmp2;
    double alpha;

    std::vector<Vec3> newControlPointsForward = degreeReductionForward();
    std::vector<Vec3> newControlPointsBackward = degreeReductionBackward();
    std::vector<Vec3> newControlPointsFinal;

    for (int i = 0; i < n + 1; i++)
    {
        alpha = (double)i / n;

        tmp1 = newControlPointsForward[i];
        tmp1 *= (1 - alpha);

        tmp2 = newControlPointsBackward[i];
        tmp2 *= alpha;

        tmp1 += tmp2;
        newControlPointsFinal.push_back(tmp1);
    }

    updateControlPoints(newControlPointsFinal);
}
