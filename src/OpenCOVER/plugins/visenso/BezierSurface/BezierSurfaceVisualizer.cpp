/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * BezierSurfaceVisualizer.cpp
 *
 *  Created on: Dec 13, 2010
 *      Author: tm_te
 */

#include "BezierSurfaceVisualizer.h"
#include <math.h>

using namespace osg;

BezierSurfaceVisualizer::BezierSurfaceVisualizer(ref_ptr<MatrixTransform> master)
{
    bezierDCS = new MatrixTransform;
    lineDCS = NULL;

    showCasteljauGeom = false;
    showSurfaceGeom = false;
    showControlPolygonGeom = false;
    n = 0;
    m = 0;
    parameterU = 0;
    parameterV = 0;

    master->addChild(bezierDCS.get());
}

BezierSurfaceVisualizer::~BezierSurfaceVisualizer()
{
    // delete nodes and scenegraph
    if (lineDCS)
        lineDCS = 0;

    if (bezierDCS)
    {
        if (bezierDCS->getNumParents())
        {
            bezierDCS->getParent(0)->removeChild(bezierDCS);
        }
    }
}

void BezierSurfaceVisualizer::showSurface(bool state)
{
    showSurfaceGeom = state;
}

void BezierSurfaceVisualizer::showCasteljau(bool state)
{
    showCasteljauGeom = state;
}

void BezierSurfaceVisualizer::showControlPolygon(bool state)
{
    showControlPolygonGeom = state;
}

void BezierSurfaceVisualizer::addControlPoint(Vec3 newPoint)
{
    controlPoints.push_back(newPoint);
}

void BezierSurfaceVisualizer::updateControlPoints(std::vector<Vec3> newPointList, int newN, int newM)
{
    n = newN;
    m = newM;
    removeAllControlPoints();
    for (size_t i = 0; i < newPointList.size(); i++)
    {
        addControlPoint(newPointList[i]);
    }
}

void BezierSurfaceVisualizer::setN(int newN)
{
    n = newN;
}

void BezierSurfaceVisualizer::setM(int newM)
{
    m = newM;
}

void BezierSurfaceVisualizer::removeAllControlPoints()
{
    controlPoints.clear();
}

void BezierSurfaceVisualizer::changeControlPoint(Vec3 changePoint, int index)
{
    controlPoints[index] = changePoint;
}

void BezierSurfaceVisualizer::drawLine(ref_ptr<MatrixTransform> master, Vec3 point1, Vec3 point2, Vec4 color)
{
    drawLine(master, point1, point2, color, 2);
}

void BezierSurfaceVisualizer::drawLine(ref_ptr<MatrixTransform> master, Vec3 point1, Vec3 point2, Vec4 color, float linewidth)
{
    ref_ptr<Vec3Array> lineCoordList = new Vec3Array(2);
    lineCoordList.get()->at(0) = point1;
    lineCoordList.get()->at(1) = point2;

    ref_ptr<Geometry> lineGeoset = new Geometry;
    lineGeoset->setVertexArray(lineCoordList);
    lineGeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::LINES, 0, 2));
    lineGeoset->setNormalBinding(Geometry::BIND_OFF);

    ref_ptr<Vec4Array> lineColorList = new Vec4Array;
    lineColorList->push_back(color);
    lineGeoset->setColorArray(lineColorList);
    lineGeoset->setColorBinding(Geometry::BIND_OVERALL);

    ref_ptr<StateSet> lineGeoset_state = lineGeoset->getOrCreateStateSet();
    ref_ptr<LineWidth> lw = new LineWidth(linewidth);
    lineGeoset_state->setAttribute(lw);
    lineGeoset_state->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF); //disable lightning for lines

    ref_ptr<Material> mtl = new Material;
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.f));
    mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.9f, 0.9f, 0.9f, 1.f));
    mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    mtl->setShininess(Material::FRONT_AND_BACK, 16.f);

    lineGeoset_state->setAttribute(mtl.get());

    ref_ptr<Geode> lineGeode = new Geode;
    lineGeode->addDrawable(lineGeoset);

    master->addChild(lineGeode);
}

Vec3 BezierSurfaceVisualizer::computeNormal(Vec3 p1, Vec3 p2, Vec3 p3)
{
    Vec3 v = p2;
    v -= p1;

    Vec3 u = p3;
    u -= p1;

    Vec3 normal;
    normal[0] = v[1] * u[2] - v[2] * u[1];
    normal[1] = v[2] * u[0] - v[0] * u[2];
    normal[2] = v[0] * u[1] - v[1] * u[0];

    normal /= normal.length();

    return normal;
}

void BezierSurfaceVisualizer::drawPatch(ref_ptr<MatrixTransform> master, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4, Vec4 color)
{
    ref_ptr<Vec3Array> patchCoordList = new Vec3Array(4);
    patchCoordList.get()->at(0) = p1;
    patchCoordList.get()->at(1) = p2;
    patchCoordList.get()->at(2) = p3;
    patchCoordList.get()->at(3) = p4;

    ref_ptr<Geometry> patchGeoset = new Geometry;
    patchGeoset->setVertexArray(patchCoordList);
    patchGeoset->addPrimitiveSet(new DrawArrays(PrimitiveSet::POLYGON, 0, 4));

    ref_ptr<Vec4Array> patchColorList = new Vec4Array(1);
    patchColorList.get()->at(0) = color;
    patchGeoset->setColorArray(patchColorList);
    patchGeoset->setColorBinding(osg::Geometry::BIND_OVERALL);

    ref_ptr<Vec3Array> normals = new osg::Vec3Array(1);
    Vec3 normal = computeNormal(p1, p2, p3);
    normals.get()->at(0) = normal;
    patchGeoset->setNormalArray(normals);
    patchGeoset->setNormalBinding(osg::Geometry::BIND_OVERALL);

    ref_ptr<StateSet> patchGeoset_state = patchGeoset->getOrCreateStateSet();
    ref_ptr<Material> mtl = new Material;
    mtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.1f, 0.1f, 0.1f, 1.f));
    mtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.1f, 0.1f, 0.1f, 1.f));
    mtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    mtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.f));
    mtl->setShininess(Material::FRONT_AND_BACK, 0.1f);

    patchGeoset_state->setAttribute(mtl.get());
    patchGeoset_state->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    patchGeoset_state->setMode(GL_BLEND, osg::StateAttribute::ON);
    patchGeoset_state->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    ref_ptr<Geode> patchGeode = new Geode;
    patchGeode->addDrawable(patchGeoset);

    master->addChild(patchGeode);
}

int BezierSurfaceVisualizer::getN()
{
    return n;
}

int BezierSurfaceVisualizer::getM()
{
    return m;
}

void BezierSurfaceVisualizer::updateGeometry()
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
    //	if (controlPoints.size() < 2)
    //	{
    //		return;
    //	}

    if (showSurfaceGeom)
    {
        drawSurface(lineDCS);
    }

    if (showCasteljauGeom)
    {
        visualizeCasteljau(lineDCS);
    }

    if (showControlPolygonGeom)
    {
        //Vec4 colorBlue = Vec4(0.0, 0.0, 1.0, 1.0);
        //Vec4 colorPurple = Vec4(0.7, 0, 0.7, 1.0);
        //Vec4 colorYellow = Vec4(1.0, 1.0, 0.0, 1.0);
        //Vec4 colorGreen = Vec4(0.0, 0.55, 0.0, 1.0);
        Vec4 colorGold = Vec4(0.93, 0.71, 0.13, 1.0);

        //Verbindungslinien in v Richtung
        if (n >= 2)
        {
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n - 1; j++)
                {
                    drawLine(lineDCS, controlPoints[j + (i * n)], controlPoints[(j + 1) + (i * n)], colorGold);
                }
            }
        }

        //Verbindungslinien in u Richtung
        if (m >= 2)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m - 1; j++)
                {
                    drawLine(lineDCS, controlPoints[i + (j * n)], controlPoints[i + ((j + 1) * n)], colorGold);
                }
            }
        }
    }
}

void BezierSurfaceVisualizer::drawSurface(ref_ptr<MatrixTransform> master)
{
    Vec4 colorBlue = Vec4(0.0, 0.6, 0.6, 0.8);
    double u, v;

    std::vector<Vec3> preCasteljauPoints;
    std::vector<Vec3> lowerCasteljauU;
    std::vector<Vec3> upperCasteljauU;
    Vec3 p1, p2, p3, p4;
    bool newLine = true;

    for (v = 0; v < 1; v += 0.05)
    {
        for (u = 0; u < 1; u += 0.05)
        {

            int counter = 0;
            for (size_t i = 0; i < controlPoints.size(); i++)
            {
                preCasteljauPoints.push_back(controlPoints[i]);
                counter++;

                if (counter == n)
                {
                    counter = 0;
                    upperCasteljauU.push_back(casteljauAproximation(preCasteljauPoints, (v + 0.05)));
                    lowerCasteljauU.push_back(casteljauAproximation(preCasteljauPoints, v));
                    preCasteljauPoints.clear();
                }
            }

            if (newLine)
            {
                p1 = casteljauAproximation(lowerCasteljauU, u);
                p2 = casteljauAproximation(lowerCasteljauU, (u + 0.05));
                p3 = casteljauAproximation(upperCasteljauU, (u + 0.05));
                p4 = casteljauAproximation(upperCasteljauU, u);
                newLine = false;
            }
            else
            {
                p1 = p2;
                p4 = p3;
                p2 = casteljauAproximation(lowerCasteljauU, (u + 0.05));
                p3 = casteljauAproximation(upperCasteljauU, (u + 0.05));
            }

            drawPatch(master, p1, p2, p3, p4, colorBlue);
            upperCasteljauU.clear();
            lowerCasteljauU.clear();
        }
        newLine = true;
    }
}

void BezierSurfaceVisualizer::setParameterU(double newU)
{
    if (newU >= 0 && newU <= 1)
    {
        parameterU = newU;
    }
    else
    {
        parameterU = 0;
        fprintf(stderr, "u liegt nicht zwischen 0 und 1! u wird auf 0 gesetzt.");
    }
}

void BezierSurfaceVisualizer::setParameterV(double newV)
{
    if (newV >= 0 && newV <= 1)
    {
        parameterV = newV;
    }
    else
    {
        parameterV = 0;
        fprintf(stderr, "v liegt nicht zwischen 0 und 1! v wird auf 0 gesetzt.");
    }
}

void BezierSurfaceVisualizer::visualizeCasteljau(ref_ptr<MatrixTransform> master)
{
    std::vector<Vec3> preCasteljauPoints;
    std::vector<Vec3> casteljauU;

    Vec4 colors[9];
    colors[0] = Vec4(0.0, 0.0, 0.5, 1.0); // steel blue
    colors[1] = Vec4(0.0, 1.0, 0.0, 1.0); //green
    colors[2] = Vec4(1.0, 0.39, 0.28, 1.0); //light red
    colors[3] = Vec4(1.0, 1.0, 0.0, 1.0); //yellow
    colors[4] = Vec4(1.0, 0.0, 1.0, 1.0); //purple
    colors[5] = Vec4(0.0, 0.55, 0.0, 1.0); //dark green
    colors[6] = Vec4(0.0, 0.75, 1.0, 1.0); //light blue
    colors[7] = Vec4(0.6, 0.2, 0.8, 1.0); //dark orchid
    colors[8] = Vec4(0.93, 0.71, 0.13, 1.0); //gold

    int counter = 0;
    for (size_t i = 0; i < controlPoints.size(); i++)
    {
        preCasteljauPoints.push_back(controlPoints[i]);
        counter++;

        if (counter == n)
        {
            counter = 0;
            casteljauU.push_back(visualizeCasteljauCurve(master, preCasteljauPoints, parameterV));
            preCasteljauPoints.clear();
        }
    }

    int colorOffset = casteljauU.size() - 3 + (n - m);

    for (size_t i = 0; i < casteljauU.size() - 1; i++)
    {
        drawLine(master, casteljauU[i], casteljauU[i + 1], colors[colorOffset]);
    }

    Vec3 casteljauPoint = visualizeCasteljauCurve(master, casteljauU, parameterU, colorOffset + 1);

    osg::Sphere *unitSphere = new osg::Sphere(casteljauPoint, 10);
    osg::ShapeDrawable *unitSphereDrawable = new osg::ShapeDrawable(unitSphere);
    ref_ptr<StateSet> sphere_state = unitSphereDrawable->getOrCreateStateSet();
    unitSphereDrawable->setColor(colors[casteljauU.size() - 3 + colorOffset + 1]);

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

Vec3 BezierSurfaceVisualizer::visualizeCasteljauCurve(ref_ptr<MatrixTransform> master, std::vector<Vec3> casteljauPoints, double t)
{
    return visualizeCasteljauCurve(master, casteljauPoints, t, 0);
}

Vec3 BezierSurfaceVisualizer::visualizeCasteljauCurve(ref_ptr<MatrixTransform> master, std::vector<Vec3> casteljauPoints, double t, int colorOffset)
{
    Vec4 colors[9];
    colors[0] = Vec4(0.0, 0.0, 0.5, 1.0); // steel blue
    colors[1] = Vec4(0.0, 1.0, 0.0, 1.0); //green
    colors[2] = Vec4(1.0, 0.39, 0.28, 1.0); //light red
    colors[3] = Vec4(1.0, 1.0, 0.0, 1.0); //yellow
    colors[4] = Vec4(1.0, 0.0, 1.0, 1.0); //purple
    colors[5] = Vec4(0.0, 0.55, 0.0, 1.0); //dark green
    colors[6] = Vec4(0.0, 0.75, 1.0, 1.0); //light blue
    colors[7] = Vec4(0.6, 0.2, 0.8, 1.0); //dark orchid
    colors[8] = Vec4(0.93, 0.71, 0.13, 1.0); //gold

    std::vector<Vec3> points = casteljauPoints;
    int steps = points.size() - 1;

    for (size_t n = 0; ssize_t(n) < steps; n++)
    {
        for (size_t i = 0; i < points.size() - 1 - n; i++)
        {
            points[i] *= (1 - t);
            Vec3 tmp = points[i + 1];
            tmp *= t;
            points[i] += tmp;
        }
        for (size_t j = 0; j < points.size() - 2 - n; j++)
        {
            drawLine(master, points[j], points[j + 1], colors[n + colorOffset]);
        }
    }

    osg::Sphere *unitSphere = new osg::Sphere(points[0], 7);
    osg::ShapeDrawable *unitSphereDrawable = new osg::ShapeDrawable(unitSphere);
    ref_ptr<StateSet> sphere_state = unitSphereDrawable->getOrCreateStateSet();
    unitSphereDrawable->setColor(colors[casteljauPoints.size() - 3 + colorOffset]);

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

    return points[0];
}

//Aproximation einer Bezierkurve mit dem Casteljau-Algorithmus
Vec3 BezierSurfaceVisualizer::casteljauAproximation(std::vector<Vec3> points, double t)
{
    int steps = points.size() - 1;
    for (size_t n = 0; ssize_t(n) < steps; n++)
    {
        for (size_t i = 0; i < points.size() - 1 - n; i++)
        {
            points[i] *= (1 - t);
            Vec3 tmp = points[i + 1];
            tmp *= t;
            points[i] += tmp;
        }
    }
    return points[0];
}

//Exakte Berechnung eines Flaechenpunktes
Vec3 BezierSurfaceVisualizer::computePointOnCurve(double u, double v)
{
    Vec3 tmp;
    Vec3 erg = Vec3(0, 0, 0);

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            tmp = controlPoints[j + (i * n)];
            tmp *= bernstein(m - 1, i, u);
            tmp *= bernstein(n - 1, j, v);
            erg += tmp;
        }
    }

    return erg;
}

double BezierSurfaceVisualizer::bernstein(double n, double i, double t)
{
    double erg;
    erg = binomKoeff(n, i) * pow(t, i) * pow((1 - t), (n - i));
    return erg;
}

double BezierSurfaceVisualizer::binomKoeff(double n, double k)
{
    double zaehler = factorial(n);
    double nenner = factorial(k) * factorial(n - k);
    return zaehler / nenner;
}

double BezierSurfaceVisualizer::factorial(double k)
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

std::vector<Vec3> BezierSurfaceVisualizer::getAllControlPoints()
{
    return controlPoints;
}

void BezierSurfaceVisualizer::degreeElevation(char direction)
{
    switch (direction)
    {
    case 'u': // Graderhöhung in Richtung des Parameters u
    {
        if (m < 3)
        {
            return;
        }

        std::vector<Vec3> newControlPoints(n * (m + 1), Vec3(0, 0, 0));
        for (int i = 0; i < n; i++)
        {

            newControlPoints[i] = controlPoints[i];
            for (int j = 1; j < m + 1; j++)
            {

                double alpha = (double)j / m;

                Vec3 tmp1;
                tmp1 = controlPoints[i + ((j - 1) * n)];
                tmp1 *= alpha;

                if (j != m)
                {
                    Vec3 tmp2;
                    tmp2 = controlPoints[i + (j * n)];
                    tmp2 *= (1 - alpha);

                    tmp1 += tmp2;
                }

                newControlPoints[i + (j * n)] = tmp1;
            }
        }
        updateControlPoints(newControlPoints, n, m + 1);
        break;
    }

    case 'v': //Graderhöhung in Richtung des Parameters v
    {
        if (n < 3)
        {
            return;
        }

        std::vector<Vec3> newControlPoints((n + 1) * m, Vec3(0, 0, 0));
        for (int i = 0; i < m; i++)
        {

            newControlPoints[i * (n + 1)] = controlPoints[i * n];
            for (int j = 1; j < n + 1; j++)
            {

                double alpha = (double)j / n;

                Vec3 tmp1;
                tmp1 = controlPoints[(j - 1) + i * n];
                tmp1 *= alpha;

                if (j != n)
                {
                    Vec3 tmp2;
                    tmp2 = controlPoints[j + (i * n)];
                    tmp2 *= (1 - alpha);

                    tmp1 += tmp2;
                }

                newControlPoints[j + (i * (n + 1))] = tmp1;
            }
        }
        updateControlPoints(newControlPoints, n + 1, m);
        break;
    }

    default:
    {
    }
    }
}

std::vector<Vec3> BezierSurfaceVisualizer::degreeReductionForward(char direction)
{
    Vec3 tmp1, tmp2;
    double alpha, beta;

    switch (direction)
    {
    case 'u': //Gradreduktion nach Forest vorwaerts in Richtung des Parameters u
    {
        std::vector<Vec3> newControlPointsForward(n * (m - 1), Vec3(0, 0, 0));
        for (int i = 0; i < n; i++)
        {

            newControlPointsForward[i] = controlPoints[i];
            for (int j = 1; j < m - 1; j++)
            {

                alpha = (double)(m - 1) / (m - 1 - j);
                beta = (double)j / (m - 1 - j);

                tmp1 = controlPoints[i + (j * n)];
                tmp1 *= alpha;

                tmp2 = newControlPointsForward[i + ((j - 1) * n)];
                tmp2 *= beta;

                tmp1 -= tmp2;
                newControlPointsForward[i + (j * n)] = tmp1;
            }
        }

        return newControlPointsForward;
    }

    case 'v': //Gradreduktion nach Forest vorwaerts in Richtung des Parameters v
    {
        std::vector<Vec3> newControlPointsForward((n - 1) * m, Vec3(0, 0, 0));
        for (int i = 0; i < m; i++)
        {

            newControlPointsForward[i * (n - 1)] = controlPoints[i * n];
            for (int j = 1; j < n - 1; j++)
            {

                alpha = (double)(n - 1) / (n - 1 - j);
                beta = (double)j / (n - 1 - j);

                tmp1 = controlPoints[j + (i * n)];
                tmp1 *= alpha;

                tmp2 = newControlPointsForward[(j - 1) + i * (n - 1)];
                tmp2 *= beta;

                tmp1 -= tmp2;
                newControlPointsForward[j + i * (n - 1)] = tmp1;
            }
        }

        return newControlPointsForward;
    }

    default:
    {
        std::vector<Vec3> newControlPointsForward((n - 1) * m, Vec3(0, 0, 0));
        return newControlPointsForward;
    }
    }
}

std::vector<Vec3> BezierSurfaceVisualizer::degreeReductionBackward(char direction)
{
    Vec3 tmp1, tmp2;
    double alpha, beta;

    switch (direction)
    {
    case 'u': //Gradreduktion nach Forest rueckwaerts in Richtung u
    {
        std::vector<Vec3> newControlPointsBackward(n * (m - 1));
        for (int i = 0; i < n; i++)
        {

            newControlPointsBackward[controlPoints.size() - 1 - n - (n - 1 - i)] = controlPoints[controlPoints.size() - 1 - (n - 1 - i)];
            for (int j = 1; j < m - 1; j++)
            {

                alpha = (double)(m - 1) / (m - 1 - j);
                beta = (double)j / (m - 1 - j);

                tmp1 = controlPoints[(n * (m - j - 1)) + i];
                tmp1 *= alpha;

                tmp2 = newControlPointsBackward[(n * (m - j - 1)) + i];
                tmp2 *= beta;

                tmp1 -= tmp2;

                newControlPointsBackward[i + (m - j - 2) * n] = tmp1;
            }
        }

        return newControlPointsBackward;
        ;
    }

    case 'v': //Gradreduktion nach Forest rueckwaerts in Richtung v
    {
        std::vector<Vec3> newControlPointsBackward((n - 1) * m);
        for (int i = 0; i < m; i++)
        {

            newControlPointsBackward[controlPoints.size() - 1 - m - (m - 1 - i) * (n - 1)] = controlPoints[controlPoints.size() - 1 - (m - 1 - i) * n];
            for (int j = 1; j < n - 1; j++)
            {

                alpha = (double)(n - 1) / (n - 1 - j);
                beta = (double)j / (n - 1 - j);

                tmp1 = controlPoints[(n - 1) - j + i * n];
                tmp1 *= alpha;

                tmp2 = newControlPointsBackward[(n - 1) - j + i * (n - 1)];
                tmp2 *= beta;

                tmp1 -= tmp2;

                newControlPointsBackward[(n - 2) - j + i * (n - 1)] = tmp1;
            }
        }

        return newControlPointsBackward;
    }

    default:
    {
        std::vector<Vec3> newControlPointsBackward((n - 1) * m);
        return newControlPointsBackward;
    }
    }
}

void BezierSurfaceVisualizer::degreeReductionForest(char direction)
{
    std::vector<Vec3> newControlPointsForward = degreeReductionForward(direction);
    std::vector<Vec3> newControlPointsBackward = degreeReductionBackward(direction);
    Vec3 tmp1, tmp2;

    switch (direction)
    {
    case 'u': //Gradreduktion vorwaerts und rueckwaerts kombiniert in Richtung u
    {
        if (m < 4)
            return;

        m--;
        std::vector<Vec3> newControlPointsFinal(n * m, Vec3(0, 0, 0));

        //Berechne Kombination aus beiden obigen Kurven
        //--------------------------------------------------------------------------------------------------------------
        //Neue Kurve hat ungeraden Grad
        if ((m - 1) % 2 == 1)
        {
            //uebernehme erste Haelfte der vorwaerts Gradreduktion
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m / 2; j++)
                {
                    newControlPointsFinal[i + (n * j)] = newControlPointsForward[i + (n * j)];
                }
            }

            //uebernehme zweite Haelfte der rueckwaerts Gradreduktion
            for (int i = 0; i < n; i++)
            {
                for (int j = m / 2; j < m; j++)
                {
                    newControlPointsFinal[i + (n * j)] = newControlPointsBackward[i + (n * j)];
                }
            }
        }

        //neue Kurve hat geraden Grad
        else
        {
            //uebernehme erste Haelfte der vorwaerts Gradreduktion
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < (m - 1) / 2; j++)
                {
                    newControlPointsFinal[i + (j * n)] = newControlPointsForward[i + (j * n)];
                }
            }

            //mittle die Punkte, die nicht zugeordnet werden können
            for (int i = 0; i < n; i++)
            {
                tmp1 = newControlPointsForward[(((m - 1) / 2) * n) + i];
                tmp1 *= 0.5;

                tmp2 = newControlPointsBackward[(((m - 1) / 2) * n) + i];
                tmp2 *= 0.5;

                tmp1 += tmp2;
                newControlPointsFinal[(((m - 1) / 2) * n) + i] = tmp1;
            }

            //uebernehme zweite Haelfte der rueckwaerts Gradreduktion
            for (int i = 0; i < n; i++)
            {
                for (int j = (m - 1) / 2 + 1; j < m; j++)
                {
                    newControlPointsFinal[i + (j * n)] = newControlPointsBackward[i + (j * n)];
                }
            }
        }

        updateControlPoints(newControlPointsFinal, n, m);
        break;
    }

    case 'v': //Gradreduktion vorwaerts und rueckwaerts kombiniert in Richtung v
    {
        if (n < 4)
            return;

        n--;
        std::vector<Vec3> newControlPointsFinal(n * m, Vec3(0, 0, 0));

        //Berechne Kombination aus beiden obigen Kurven
        //--------------------------------------------------------------------------------------------------------------
        //Neue Kurve hat ungeraden Grad
        if ((n - 1) % 2 == 1)
        {
            //uebernehme erste Haelfte der vorwaerts Gradreduktion
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n / 2; j++)
                {
                    newControlPointsFinal[j + (n * i)] = newControlPointsForward[j + (n * i)];
                }
            }

            //uebernehme zweite Haelfte der rueckwaerts Gradreduktion
            for (int i = 0; i < m; i++)
            {
                for (int j = n / 2; j < n; j++)
                {
                    newControlPointsFinal[j + (n * i)] = newControlPointsBackward[j + (n * i)];
                }
            }
        }

        //neue Kurve hat geraden Grad
        else
        {
            //uebernehme erste Haelfte der vorwaerts Gradreduktion
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < (n - 1) / 2; j++)
                {
                    newControlPointsFinal[j + (i * n)] = newControlPointsForward[j + (i * n)];
                }
            }

            //mittle die Punkte, die nicht zugeordnet werden können
            for (int i = 0; i < m; i++)
            {
                tmp1 = newControlPointsForward[((n - 1) / 2) + n * i];
                tmp1 *= 0.5;

                tmp2 = newControlPointsBackward[((n - 1) / 2) + n * i];
                tmp2 *= 0.5;

                tmp1 += tmp2;
                newControlPointsFinal[((n - 1) / 2) + n * i] = tmp1;
            }

            //uebernehme zweite Haelfte der rueckwaerts Gradreduktion
            for (int i = 0; i < m; i++)
            {
                for (int j = (n - 1) / 2 + 1; j < n; j++)
                {
                    newControlPointsFinal[j + (i * n)] = newControlPointsBackward[j + (i * n)];
                }
            }
        }

        updateControlPoints(newControlPointsFinal, n, m);
    }
    }
}

void BezierSurfaceVisualizer::degreeReductionFarin(char direction)
{
    std::vector<Vec3> newControlPointsForward = degreeReductionForward(direction);
    std::vector<Vec3> newControlPointsBackward = degreeReductionBackward(direction);
    Vec3 tmp1, tmp2;
    double alpha;

    switch (direction)
    {
    case 'u':
    {
        if (m < 4)
            return;

        m--;
        std::vector<Vec3> newControlPointsFinal(n * m, Vec3(0, 0, 0));

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {

                alpha = (double)j / n;

                tmp1 = newControlPointsForward[i + (j * n)];
                tmp1 *= (1 - alpha);

                tmp2 = newControlPointsBackward[i + (j * n)];
                tmp2 *= alpha;

                tmp1 += tmp2;
                newControlPointsFinal[i + (j * n)] = tmp1;
            }
        }

        updateControlPoints(newControlPointsFinal, n, m);
        break;
    }

    case 'v':
    {
        if (n < 4)
            return;

        n--;
        std::vector<Vec3> newControlPointsFinal(n * m, Vec3(0, 0, 0));

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {

                alpha = (double)j / n;

                tmp1 = newControlPointsForward[j + (i * n)];
                tmp1 *= (1 - alpha);

                tmp2 = newControlPointsBackward[j + (i * n)];
                tmp2 *= alpha;

                tmp1 += tmp2;
                newControlPointsFinal[j + (i * n)] = tmp1;
            }
        }

        updateControlPoints(newControlPointsFinal, n, m);
        break;
    }

    default:
    {
    }
    }
}
