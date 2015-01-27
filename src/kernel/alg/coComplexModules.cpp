/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coVector.h>
#include <util/coMatrix.h>

#include "coComplexModules.h"
#include <do/coDoColormap.h>
#include <do/coDoData.h>
#include <do/coDoPoints.h>
#include <do/coDoTriangleStrips.h>
#include "coSphere.h"

using namespace covise;

coDistributedObject *
ComplexModules::DataTexture(const string &color_name,
                            const coDistributedObject *dataOut,
                            const coDistributedObject *colorMap,
                            bool texture,
                            int repeat, float *min, float *max)
{
    if (!dynamic_cast<const coDoColormap *>(colorMap))
    {
        colorMap = NULL;
        // return NULL;
    }
    if (!dataOut)
    {
        return NULL;
    }
    coColors co_colors(dataOut, (const coDoColormap *)colorMap, false);
    return co_colors.getColors(coObjInfo(color_name.c_str()), texture, true, repeat, min, max);
}

coDistributedObject *
ComplexModules::DataTextureLineCropped(const string &color_name,
                                       coDistributedObject *dataOut,
                                       coDistributedObject *lines,
                                       const coDistributedObject *colorMap,
                                       bool texture,
                                       int repeat, int croppedLength,
                                       float min, float max)
{
    if (!colorMap || !colorMap->isType("COLMAP"))
    {
        colorMap = NULL;
        // return NULL;
    }
    if (!dataOut)
    {
        return NULL;
    }

    coColors co_colors(dataOut, (const coDoColormap *)colorMap, false);
    coDistributedObject *uncroppedColors = co_colors.getColors(color_name.c_str(), texture, true, repeat, &min, &max);
    coDistributedObject *croppedColors;
    if (croppedLength == 0)
    {
        return uncroppedColors;
    }
    else
    {
        // farben beschneiden
        int numElements, dummy;
        coDistributedObject **croppedColorsList;
        const coDistributedObject *const *setListColors = ((coDoSet *)uncroppedColors)->getAllElements(&numElements);
        const coDistributedObject *const *setListLines = ((coDoSet *)lines)->getAllElements(&dummy); // dummy == numElements

        croppedColorsList = new coDistributedObject *[numElements + 1];
        croppedColorsList[numElements] = NULL;

        for (int curSetEle = 0; curSetEle < numElements; curSetEle++)
        {
            int *cornerList, *lineList;
            float *xPoints, *yPoints, *zPoints;
            ((coDoLines *)setListLines[curSetEle])->getAddresses(&xPoints, &yPoints, &zPoints, &cornerList, &lineList);
            int numLines = ((coDoLines *)setListLines[curSetEle])->getNumLines();
            int numCorners = ((coDoLines *)setListLines[curSetEle])->getNumVertices();
            int numCroppedColors = 0;

            // zaehlen, wieviele vertices der linien benutzt werden (noetig fuer die anzahl der farb samples) und entsprechend speicher reservieren
            int totalVertices = 0;
            for (int curLine = 0; curLine < numLines; curLine++)
            {
                int lineLength = (curLine != numLines - 1) ? (lineList[curLine + 1] - lineList[curLine]) : (numCorners - lineList[numLines - 1]); //num of corners for current line
                totalVertices += (lineLength > croppedLength) ? croppedLength : lineLength;
            }
            croppedColorsList[curSetEle] = new coDoRGBA("Cropped_Colors", totalVertices * repeat);
            for (int curLine = 0; curLine < numLines; curLine++)
            {
                int lineLength = (curLine != numLines - 1) ? (lineList[curLine + 1] - lineList[curLine]) : (numCorners - lineList[numLines - 1]); //num of corners for current line
                int l = (lineLength > croppedLength) ? croppedLength : lineLength;

                int lastCorner = lineList[curLine] + lineLength;
                for (int curCorner = lastCorner - l; curCorner < lastCorner; curCorner++)
                {
                    int samplingPos;
                    samplingPos = cornerList[curCorner];

                    float r, g, b, a;
                    ((coDoRGBA *)setListColors[curSetEle])->getFloatRGBA(samplingPos * repeat, &r, &g, &b, &a);
                    for (int i = 0; i < repeat; i++)
                    {
                        ((coDoRGBA *)croppedColorsList[curSetEle])->setFloatRGBA(numCroppedColors++, r, g, b, a);
                    }
                }
            }
        }

        croppedColors = new coDoSet("Cropped_Colors", croppedColorsList);

        // uncroppedColors loeschen, da nicht nicht ausgegeben wird
        delete uncroppedColors;

        return croppedColors;
    }
}

coDistributedObject *
ComplexModules::MakeArrows(const char *name,
                           const coDistributedObject *geo, const coDistributedObject *data,
                           const char *nameColor, coDistributedObject **colorSurf,
                           coDistributedObject **colorLines, float /*factor*/,
                           const coDistributedObject *colorMap,
                           bool ColorMapAttrib,
                           const ScalarContainer *SCont,
                           float scale, int lineChoice, int numsectors, int project_lines,
                           int vect_option)
{
    /// safeguard input params
    if (numsectors < 0)
        numsectors = 0;

    // timesteps are a special case
    //   if(!(geo->isType("SETELE") && geo->getAttribute("TIMESTEP"))){
    coDistrVectField *p_VecField = NULL;
    if (dynamic_cast<const coDoColormap *>(colorMap))
    {
        p_VecField = new coDistrVectField(geo, data, (const coDoColormap *)colorMap,
                                          scale, lineChoice, numsectors, project_lines);
    }
    else
    {
        p_VecField = new coDistrVectField(geo, data, NULL,
                                          scale, lineChoice, numsectors, project_lines);
    }
    coDistributedObject *linesOut = NULL;
    p_VecField->Execute(&linesOut, colorSurf, colorLines,
                        name, nameColor,
                        ColorMapAttrib, SCont, vect_option);
    delete p_VecField;
    return linesOut;
}

// like previous MakeArrows, but with params for arrow head
coDistributedObject *
ComplexModules::MakeArrows(const char *name,
                           const coDistributedObject *geo, const coDistributedObject *data,
                           const char *nameColor, coDistributedObject **colorSurf,
                           coDistributedObject **colorLines, float /*factor*/,
                           const coDistributedObject *colorMap,
                           bool ColorMapAttrib,
                           const ScalarContainer *SCont,
                           float scale, int lineChoice, int numsectors, float arrow_factor, float angle,
                           int project_lines, int vect_option)
{
    /// safeguard input params
    if (numsectors < 0)
        numsectors = 0;

    // timesteps are a special case
    //   if(!(geo->isType("SETELE") && geo->getAttribute("TIMESTEP"))){
    coDistrVectField *p_VecField = NULL;
    if (colorMap && colorMap->isType("COLMAP"))
    {
        p_VecField = new coDistrVectField(geo, data, (coDoColormap *)colorMap,
                                          scale, lineChoice, numsectors, arrow_factor, angle, project_lines);
    }
    else
    {
        p_VecField = new coDistrVectField(geo, data, NULL, scale, lineChoice,
                                          numsectors, arrow_factor, angle,
                                          project_lines);
    }
    coDistributedObject *linesOut = NULL;
    p_VecField->Execute(&linesOut, colorSurf, colorLines,
                        name, nameColor,
                        ColorMapAttrib, SCont, vect_option);
    delete p_VecField;
    return linesOut;
}

#include "coSphere.h"

coDistributedObject *
ComplexModules::Spheres(const char *name, const coDistributedObject *points, float radius,
                        const char *name_norm,
                        coDistributedObject **normals)
{
    if (dynamic_cast<const coDoSet *>(points))
    {
        int no_e;
        const coDistributedObject *const *setList = ((const coDoSet *)points)->getAllElements(&no_e);
        coDistributedObject **outList = new coDistributedObject *[no_e + 1];
        coDistributedObject **normList = new coDistributedObject *[no_e + 1];
        outList[no_e] = NULL;
        normList[no_e] = NULL;
        int i;
        for (i = 0; i < no_e; ++i)
        {
            string name_i = name;
            string name_norm_i = name_norm;
            char buf[16];
            sprintf(buf, "_%d", i);
            name_i += buf;
            name_norm_i += buf;
            outList[i] = ComplexModules::Spheres(name_i.c_str(), setList[i], radius,
                                                 name_norm_i.c_str(), &normList[i]);
        }
        coDistributedObject *outSet = new coDoSet(coObjInfo(name), outList);
        *normals = new coDoSet(coObjInfo(name_norm), normList);
        outSet->copyAllAttributes(points);
        for (i = 0; i < no_e; ++i)
        {
            delete outList[i];
            delete setList[i];
            delete normList[i];
        }
        delete[] setList;
        delete[] outList;
        delete[] normList;
        return outSet;
    }
    else if (const coDoPoints *Points = dynamic_cast<const coDoPoints *>(points))
    {
        int numPoints = Points->getNumPoints();
        coDoTriangleStrips *spheres = new coDoTriangleStrips(coObjInfo(name),
                                                             17 * numPoints, 42 * numPoints, 6 * numPoints);
        float *xSpheres, *ySpheres, *zSpheres;
        int *vl, *tsl;
        spheres->getAddresses(&xSpheres, &ySpheres, &zSpheres, &vl, &tsl);

        *normals = new coDoVec3(coObjInfo(name_norm),
                                17 * numPoints);
        float *normalsOut[3];
        coDoVec3 *Normals = (coDoVec3 *)(*normals);
        Normals->getAddresses(&normalsOut[0], &normalsOut[1], &normalsOut[2]);
        coSphere mySphere;
        float *dataIn[3];
        dataIn[0] = dataIn[1] = dataIn[2] = NULL;
        float *xPoints, *yPoints, *zPoints;
        Points->getAddresses(&xPoints, &yPoints, &zPoints);
        mySphere.computeSpheres(1.0, radius, dataIn, xSpheres, ySpheres, zSpheres,
                                xPoints, yPoints, zPoints,
                                normalsOut,
                                tsl, vl, numPoints,
                                NULL, NULL, 0, 0);

        spheres->copyAllAttributes(points);
        return spheres;
    }
    return (new coDoTriangleStrips(coObjInfo(name), 0, 0, 0));
}

coDistributedObject *
ComplexModules::Tubelines(const char *name, const coDistributedObject *lines, const coDistributedObject *tangents, float tubeSize, float radius, int trailLength, const char *headType, coDistributedObject **colors)
{
    int numHeadCoords;
    int numHeadVertices;
    int numHeadPolygons;
    enum _headTypeEnum
    {
        NONE,
        SPHERE,
        BAR,
        BAR_MAGNET,
        COMPASS
    };
    _headTypeEnum headTypeEnum;

    if ((headType == NULL) || (strcmp(headType, "NONE") == 0))
    {
        // nur roehren, keine kopfstuecke!
        headTypeEnum = NONE;
        numHeadCoords = 0;
        numHeadVertices = 0;
        numHeadPolygons = 0;
    }
    else
    {
        if (strcmp(headType, "COMPASS") == 0)
        {
            headTypeEnum = COMPASS;
            numHeadCoords = 10;
            numHeadVertices = 24;
            numHeadPolygons = 8;
        }
        else if (strcmp(headType, "BAR_MAGNET") == 0)
        {
            headTypeEnum = BAR_MAGNET;
            numHeadCoords = 16;
            numHeadVertices = 40;
            numHeadPolygons = 10;
        }
        else
        {
            // nicht unterstuetztes kopfstueck
            headTypeEnum = NONE;
            numHeadCoords = 0;
            numHeadVertices = 0;
            numHeadPolygons = 0;
        }
    }

    if (lines && lines->isType("SETELE"))
    {
        int no_e, dummy;
        const coDistributedObject *const *setListLines = ((coDoSet *)lines)->getAllElements(&no_e);
        const coDistributedObject *const *setListTangents = ((coDoSet *)tangents)->getAllElements(&dummy);
        coDistributedObject **outList = new coDistributedObject *[no_e + 1];
        coDistributedObject **outListColors = NULL;

        if (colors && (headTypeEnum != NONE))
        {
            outListColors = new coDistributedObject *[no_e + 1];
        }

        if (colors && (headTypeEnum != NONE))
        {
            outListColors[no_e] = NULL;
        }
        outList[no_e] = NULL;

        for (int currentSetEle = 0; currentSetEle < no_e; currentSetEle++)
        {
            string name_i = name;
            char buf[16];
            sprintf(buf, "_%d", currentSetEle);
            name_i += buf;
            if (!setListTangents[currentSetEle]->isType("USTVDT"))
            {
                std::cout << "No vector data in tangents!" << std::endl;
            }

            coDoPolygons *polygonObj;
            int numLines = ((coDoLines *)setListLines[currentSetEle])->getNumLines();
            int numCorners = ((coDoLines *)setListLines[currentSetEle])->getNumVertices();
            int numPoints = ((coDoLines *)setListLines[currentSetEle])->getNumPoints();
            //int numTangents = ((coDoUnstructured_V3D_Data *)setListTangents[currentSetEle])->getNumPoints();    // matches numPoints
            float *xCoords = new float[2 * numPoints * 4 + numLines * numHeadCoords];
            float *yCoords = new float[2 * numPoints * 4 + numLines * numHeadCoords];
            float *zCoords = new float[2 * numPoints * 4 + numLines * numHeadCoords];
            int *vertexList = new int[numPoints * 16 + numLines * numHeadVertices];
            int *polygonList = new int[numPoints * 4 + numLines * numHeadPolygons];
            //float tubeSize = 0.1*radius;
            float *xPoints, *yPoints, *zPoints;
            int *cornerList, *lineList;
            float *xTangents, *yTangents, *zTangents;

            ((coDoLines *)setListLines[currentSetEle])->getAddresses(&xPoints, &yPoints, &zPoints, &cornerList, &lineList);
            ((coDoVec3 *)setListTangents[currentSetEle])->getAddresses(&xTangents, &yTangents, &zTangents);

            // create connectivity, vertices and so on
            int numCoords = 0;
            int numVertices = 0;
            int numPolygons = 0;
            for (int curLine = 0; curLine < numLines; curLine++)
            {
                int upperEnd; // fuer die letzte linie muss die grenze in der corner list anders ermittelt werden
                int lowerEnd; // abhaengig von schweiflaenge

                upperEnd = (curLine != numLines - 1) ? lineList[curLine + 1] : numCorners;
                lowerEnd = lineList[curLine];
                // schweiflaenge mittels lowerEnd anpassen
                if ((trailLength > 0) && (upperEnd - lowerEnd > trailLength + 1))
                {
                    lowerEnd = upperEnd - trailLength - 1;
                }

                // hole den up und right vector
                coVector up(0.0, 0.0, 1.0);
                coVector tangent;
                coVector right;

                tangent[0] = xTangents[cornerList[lowerEnd]];
                tangent[1] = yTangents[cornerList[lowerEnd]];
                tangent[2] = zTangents[cornerList[lowerEnd]];

                right = tangent.cross(up);
                right.normalize();
                up = right.cross(tangent);
                up.normalize();
                right = right * tubeSize;
                up = up * tubeSize;

                // vorlauf
                xCoords[numCoords] = (float)(xPoints[cornerList[lowerEnd]] + right[0] - up[0]);
                yCoords[numCoords] = (float)(yPoints[cornerList[lowerEnd]] + right[1] - up[1]);
                zCoords[numCoords] = (float)(zPoints[cornerList[lowerEnd]] + right[2] - up[2]);
                numCoords++;
                xCoords[numCoords] = (float)(xPoints[cornerList[lowerEnd]] + right[0] + up[0]);
                yCoords[numCoords] = (float)(yPoints[cornerList[lowerEnd]] + right[1] + up[1]);
                zCoords[numCoords] = (float)(zPoints[cornerList[lowerEnd]] + right[2] + up[2]);
                numCoords++;
                xCoords[numCoords] = (float)(xPoints[cornerList[lowerEnd]] - right[0] + up[0]);
                yCoords[numCoords] = (float)(yPoints[cornerList[lowerEnd]] - right[1] + up[1]);
                zCoords[numCoords] = (float)(zPoints[cornerList[lowerEnd]] - right[2] + up[2]);
                numCoords++;
                xCoords[numCoords] = (float)(xPoints[cornerList[lowerEnd]] - right[0] - up[0]);
                yCoords[numCoords] = (float)(yPoints[cornerList[lowerEnd]] - right[1] - up[1]);
                zCoords[numCoords] = (float)(zPoints[cornerList[lowerEnd]] - right[2] - up[2]);
                numCoords++;

                for (int curCorner = lowerEnd + 1; curCorner < upperEnd; curCorner++)
                {
                    // hole den up und right vector
                    coVector up(0.0, 0.0, 1.0);
                    coVector tangent;
                    coVector right;

                    tangent[0] = xTangents[cornerList[curCorner]];
                    tangent[1] = yTangents[cornerList[curCorner]];
                    tangent[2] = zTangents[cornerList[curCorner]];

                    right = tangent.cross(up);
                    right.normalize();
                    up = right.cross(tangent);
                    up.normalize();
                    right = right * tubeSize;
                    up = up * tubeSize;

                    xCoords[numCoords] = (float)(xPoints[cornerList[curCorner]] + right[0] - up[0]);
                    yCoords[numCoords] = (float)(yPoints[cornerList[curCorner]] + right[1] - up[1]);
                    zCoords[numCoords] = (float)(zPoints[cornerList[curCorner]] + right[2] - up[2]);
                    numCoords++;
                    xCoords[numCoords] = (float)(xPoints[cornerList[curCorner]] + right[0] + up[0]);
                    yCoords[numCoords] = (float)(yPoints[cornerList[curCorner]] + right[1] + up[1]);
                    zCoords[numCoords] = (float)(zPoints[cornerList[curCorner]] + right[2] + up[2]);
                    numCoords++;
                    xCoords[numCoords] = (float)(xPoints[cornerList[curCorner]] - right[0] + up[0]);
                    yCoords[numCoords] = (float)(yPoints[cornerList[curCorner]] - right[1] + up[1]);
                    zCoords[numCoords] = (float)(zPoints[cornerList[curCorner]] - right[2] + up[2]);
                    numCoords++;
                    xCoords[numCoords] = (float)(xPoints[cornerList[curCorner]] - right[0] - up[0]);
                    yCoords[numCoords] = (float)(yPoints[cornerList[curCorner]] - right[1] - up[1]);
                    zCoords[numCoords] = (float)(zPoints[cornerList[curCorner]] - right[2] - up[2]);
                    numCoords++;

                    vertexList[numVertices++] = numCoords - 8;
                    vertexList[numVertices++] = numCoords - 5;
                    vertexList[numVertices++] = numCoords - 1;
                    vertexList[numVertices++] = numCoords - 4;
                    polygonList[numPolygons] = numPolygons * 4;
                    numPolygons++;

                    vertexList[numVertices++] = numCoords - 5;
                    vertexList[numVertices++] = numCoords - 6;
                    vertexList[numVertices++] = numCoords - 2;
                    vertexList[numVertices++] = numCoords - 1;
                    polygonList[numPolygons] = numPolygons * 4;
                    numPolygons++;

                    vertexList[numVertices++] = numCoords - 6;
                    vertexList[numVertices++] = numCoords - 7;
                    vertexList[numVertices++] = numCoords - 3;
                    vertexList[numVertices++] = numCoords - 2;
                    polygonList[numPolygons] = numPolygons * 4;
                    numPolygons++;

                    vertexList[numVertices++] = numCoords - 7;
                    vertexList[numVertices++] = numCoords - 8;
                    vertexList[numVertices++] = numCoords - 4;
                    vertexList[numVertices++] = numCoords - 3;
                    polygonList[numPolygons] = numPolygons * 4;
                    numPolygons++;
                }
            }

            // Kopf jeder Linie anhaengen (mit u.u. eigener einfaerbung)
            if (headTypeEnum != NONE)
            {
                for (int curLine = 0; curLine < numLines; curLine++)
                {
                    int lastCorner; // fuer die letzte linie muss die grenze in der corner list anders ermittelt werden
                    lastCorner = (curLine != numLines - 1) ? lineList[curLine + 1] - 1 : numCorners - 1;

                    // hole den up und right vector
                    coVector up(0.0, 0.0, 1.0);
                    coVector tangent;
                    coVector right;
                    coMatrix transform;

                    tangent[0] = xTangents[cornerList[lastCorner]];
                    tangent[1] = yTangents[cornerList[lastCorner]];
                    tangent[2] = zTangents[cornerList[lastCorner]];

                    tangent.normalize();
                    right = tangent.cross(up);
                    right.normalize();
                    up = right.cross(tangent);
                    up.normalize();

                    transform.set(0, 0, tangent[0]);
                    transform.set(1, 0, tangent[1]);
                    transform.set(2, 0, tangent[2]);

                    transform.set(0, 1, right[0]);
                    transform.set(1, 1, right[1]);
                    transform.set(2, 1, right[2]);

                    transform.set(0, 2, up[0]);
                    transform.set(1, 2, up[1]);
                    transform.set(2, 2, up[2]);

                    transform.set(0, 3, xPoints[cornerList[lastCorner]]);
                    transform.set(1, 3, yPoints[cornerList[lastCorner]]);
                    transform.set(2, 3, zPoints[cornerList[lastCorner]]);

                    transform.set(3, 3, 1.0);

                    // initialisiere koordinaten vom kopf
                    switch (headTypeEnum)
                    {
                    case COMPASS:
                        xCoords[numCoords] = -0.0;
                        yCoords[numCoords] = +0.15f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = -0.0;
                        yCoords[numCoords] = -0.15f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = -0.0;
                        yCoords[numCoords] = -0.15f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = -0.0;
                        yCoords[numCoords] = +0.15f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = -0.95f * radius;
                        yCoords[numCoords] = -0.0;
                        zCoords[numCoords] = -0.0;
                        numCoords++;
                        xCoords[numCoords] = +0.0;
                        yCoords[numCoords] = +0.15f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = +0.0;
                        yCoords[numCoords] = -0.15f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = +0.0;
                        yCoords[numCoords] = -0.15f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = +0.0;
                        yCoords[numCoords] = +0.15f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = +0.95f * radius;
                        yCoords[numCoords] = +0.0;
                        zCoords[numCoords] = +0.0;
                        numCoords++;
                        break;
                    case BAR_MAGNET:
                        // green half
                        xCoords[numCoords] = 0.0;
                        yCoords[numCoords] = +0.5f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = 0.0;
                        yCoords[numCoords] = -0.5f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = 0.0;
                        yCoords[numCoords] = -0.5f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = 0.0;
                        yCoords[numCoords] = +0.5f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = -0.95f * radius;
                        yCoords[numCoords] = +0.5f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = -0.95f * radius;
                        yCoords[numCoords] = -0.5f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = -0.95f * radius;
                        yCoords[numCoords] = -0.5f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = -0.95f * radius;
                        yCoords[numCoords] = +0.5f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;

                        // red half
                        xCoords[numCoords] = 0.0;
                        yCoords[numCoords] = +0.5f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = 0.0;
                        yCoords[numCoords] = -0.5f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = 0.0;
                        yCoords[numCoords] = -0.5f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = 0.0;
                        yCoords[numCoords] = +0.5f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = +0.95f * radius;
                        yCoords[numCoords] = +0.5f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = +0.95f * radius;
                        yCoords[numCoords] = -0.5f * radius;
                        zCoords[numCoords] = +0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = +0.95f * radius;
                        yCoords[numCoords] = -0.5f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        xCoords[numCoords] = +0.95f * radius;
                        yCoords[numCoords] = +0.5f * radius;
                        zCoords[numCoords] = -0.15f * radius;
                        numCoords++;
                        break;
                    default:
                        break;
                    }

                    // transformiere koordinaten vom kopf
                    for (int i = 1; i <= numHeadCoords; i++)
                    {
                        coVector _v(xCoords[numCoords - i], yCoords[numCoords - i], zCoords[numCoords - i]);

                        _v = transform * _v;

                        xCoords[numCoords - i] = (float)_v[0];
                        yCoords[numCoords - i] = (float)_v[1];
                        zCoords[numCoords - i] = (float)_v[2];
                    }

                    // set up connectivity
                    switch (headTypeEnum)
                    {
                    case COMPASS:
                        vertexList[numVertices++] = numCoords - 10;
                        vertexList[numVertices++] = numCoords - 9;
                        vertexList[numVertices++] = numCoords - 6;
                        polygonList[numPolygons++] = numVertices - 3;
                        vertexList[numVertices++] = numCoords - 9;
                        vertexList[numVertices++] = numCoords - 8;
                        vertexList[numVertices++] = numCoords - 6;
                        polygonList[numPolygons++] = numVertices - 3;
                        vertexList[numVertices++] = numCoords - 8;
                        vertexList[numVertices++] = numCoords - 7;
                        vertexList[numVertices++] = numCoords - 6;
                        polygonList[numPolygons++] = numVertices - 3;
                        vertexList[numVertices++] = numCoords - 7;
                        vertexList[numVertices++] = numCoords - 10;
                        vertexList[numVertices++] = numCoords - 6;
                        polygonList[numPolygons++] = numVertices - 3;
                        vertexList[numVertices++] = numCoords - 5;
                        vertexList[numVertices++] = numCoords - 2;
                        vertexList[numVertices++] = numCoords - 1;
                        polygonList[numPolygons++] = numVertices - 3;
                        vertexList[numVertices++] = numCoords - 2;
                        vertexList[numVertices++] = numCoords - 3;
                        vertexList[numVertices++] = numCoords - 1;
                        polygonList[numPolygons++] = numVertices - 3;
                        vertexList[numVertices++] = numCoords - 3;
                        vertexList[numVertices++] = numCoords - 4;
                        vertexList[numVertices++] = numCoords - 1;
                        polygonList[numPolygons++] = numVertices - 3;
                        vertexList[numVertices++] = numCoords - 4;
                        vertexList[numVertices++] = numCoords - 5;
                        vertexList[numVertices++] = numCoords - 1;
                        polygonList[numPolygons++] = numVertices - 3;
                        break;
                    case BAR_MAGNET:
                        // green half of bar magnet
                        vertexList[numVertices++] = numCoords - 16;
                        vertexList[numVertices++] = numCoords - 15;
                        vertexList[numVertices++] = numCoords - 11;
                        vertexList[numVertices++] = numCoords - 12;
                        polygonList[numPolygons++] = numVertices - 4;
                        vertexList[numVertices++] = numCoords - 15;
                        vertexList[numVertices++] = numCoords - 14;
                        vertexList[numVertices++] = numCoords - 10;
                        vertexList[numVertices++] = numCoords - 11;
                        polygonList[numPolygons++] = numVertices - 4;
                        vertexList[numVertices++] = numCoords - 14;
                        vertexList[numVertices++] = numCoords - 13;
                        vertexList[numVertices++] = numCoords - 9;
                        vertexList[numVertices++] = numCoords - 10;
                        polygonList[numPolygons++] = numVertices - 4;
                        vertexList[numVertices++] = numCoords - 13;
                        vertexList[numVertices++] = numCoords - 16;
                        vertexList[numVertices++] = numCoords - 12;
                        vertexList[numVertices++] = numCoords - 9;
                        polygonList[numPolygons++] = numVertices - 4;
                        vertexList[numVertices++] = numCoords - 12;
                        vertexList[numVertices++] = numCoords - 11;
                        vertexList[numVertices++] = numCoords - 10;
                        vertexList[numVertices++] = numCoords - 9;
                        polygonList[numPolygons++] = numVertices - 4;

                        // red half of bar magnet
                        vertexList[numVertices++] = numCoords - 5;
                        vertexList[numVertices++] = numCoords - 6;
                        vertexList[numVertices++] = numCoords - 2;
                        vertexList[numVertices++] = numCoords - 1;
                        polygonList[numPolygons++] = numVertices - 4;
                        vertexList[numVertices++] = numCoords - 6;
                        vertexList[numVertices++] = numCoords - 7;
                        vertexList[numVertices++] = numCoords - 3;
                        vertexList[numVertices++] = numCoords - 2;
                        polygonList[numPolygons++] = numVertices - 4;
                        vertexList[numVertices++] = numCoords - 7;
                        vertexList[numVertices++] = numCoords - 8;
                        vertexList[numVertices++] = numCoords - 4;
                        vertexList[numVertices++] = numCoords - 3;
                        polygonList[numPolygons++] = numVertices - 4;
                        vertexList[numVertices++] = numCoords - 8;
                        vertexList[numVertices++] = numCoords - 5;
                        vertexList[numVertices++] = numCoords - 1;
                        vertexList[numVertices++] = numCoords - 4;
                        polygonList[numPolygons++] = numVertices - 4;
                        vertexList[numVertices++] = numCoords - 1;
                        vertexList[numVertices++] = numCoords - 2;
                        vertexList[numVertices++] = numCoords - 3;
                        vertexList[numVertices++] = numCoords - 4;
                        polygonList[numPolygons++] = numVertices - 4;
                        break;
                    default:
                        break;
                    }
                }
            }

            // assign custom colors to every vertex
            if (colors && (headTypeEnum != NONE))
            {
                switch (headTypeEnum)
                {
                case COMPASS:
                    outListColors[currentSetEle] = new coDoRGBA("Tubelines_Colors", numLines * numHeadCoords);
                    for (int i = 0; i < numLines; i++)
                    {
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 0, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 1, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 2, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 3, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 4, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 5, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 6, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 7, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 8, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 9, 1.0, 0.0, 0.0);
                    }
                    break;
                case BAR_MAGNET:
                    outListColors[currentSetEle] = new coDoRGBA("Tubelines_Colors", numLines * numHeadCoords);
                    for (int i = 0; i < numLines; i++)
                    {
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 0, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 1, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 2, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 3, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 4, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 5, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 6, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 7, 0.0, 1.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 8, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 9, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 10, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 11, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 12, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 13, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 14, 1.0, 0.0, 0.0);
                        ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * numHeadCoords + 15, 1.0, 0.0, 0.0);
                    }
                    break;
                default:
                    break;
                }
            }

            // create the polygons data object
            polygonObj = new coDoPolygons(name_i.c_str(),
                                          numCoords, xCoords, yCoords, zCoords,
                                          numVertices, vertexList,
                                          numPolygons, polygonList);
            delete[] xCoords;
            delete[] yCoords;
            delete[] zCoords;
            delete[] vertexList;
            delete[] polygonList;

            //polygonObj->copyAllAttributes(lines);
            outList[currentSetEle] = polygonObj;
        }

        coDistributedObject *outSet = new coDoSet(name, outList);
        if (colors && (headTypeEnum != NONE))
        {
            *colors = new coDoSet("TubelinesColors", outListColors);
        }

        outSet->copyAllAttributes(lines);

        for (int i = 0; i < no_e; i++)
        {
            delete outList[i];
            delete setListLines[i]; // should be done
            if (colors && (headTypeEnum != NONE))
            {
                delete outListColors[i];
            }
        }
        delete[] setListLines; // should be done
        delete[] outList;
        if (colors && (headTypeEnum != NONE))
        {
            delete[] outListColors;
        }

        return outSet;
    }
    else
    {
        std::cout << "Tubelines: No SETELE found! Assuming non-coDoSet lines..." << std::endl;

        return ThickStreamlines(name, lines, tangents, tubeSize, trailLength);
        //return NULL;
    }
}

coDistributedObject *
ComplexModules::ThickStreamlines(const char *name, const coDistributedObject *lines, const coDistributedObject *tangents, float tubeSize, int trailLength)
{

    if (lines && lines->isType("LINES"))
    {
        //       string name_i = name;
        //       char buf[16];
        //       sprintf(buf,"_%d",currentSetEle);
        //       name_i += buf;

        if (!tangents->isType("USTVDT"))
        {
            std::cout << "No vector data in tangents!" << std::endl;
        }

        coDoPolygons *polygonObj;
        int numLines = ((coDoLines *)lines)->getNumLines();
        int numCorners = ((coDoLines *)lines)->getNumVertices();
        int numPoints = ((coDoLines *)lines)->getNumPoints();
        //int numTangents = ((coDoVec3 *)setListTangents[currentSetEle])->getNumPoints();    // matches numPoints
        float *xCoords = new float[2 * numPoints * 4];
        float *yCoords = new float[2 * numPoints * 4];
        float *zCoords = new float[2 * numPoints * 4];
        int *vertexList = new int[numPoints * 16];
        int *polygonList = new int[numPoints * 4];
        //float tubeSize = 0.1*radius;
        float *xPoints, *yPoints, *zPoints;
        int *cornerList, *lineList;
        float *xTangents, *yTangents, *zTangents;

        ((coDoLines *)lines)->getAddresses(&xPoints, &yPoints, &zPoints, &cornerList, &lineList);
        ((coDoVec3 *)tangents)->getAddresses(&xTangents, &yTangents, &zTangents);

        // create connectivity, vertices and so on
        int numCoords = 0;
        int numVertices = 0;
        int numPolygons = 0;

        for (int curLine = 0; curLine < numLines; curLine++)
        {
            int upperEnd; // fuer die letzte linie muss die grenze in der corner list anders ermittelt werden
            int lowerEnd; // abhaengig von schweiflaenge

            upperEnd = (curLine != numLines - 1) ? lineList[curLine + 1] : numCorners;
            lowerEnd = lineList[curLine];
            // schweiflaenge mittels lowerEnd anpassen
            if ((trailLength > 0) && (upperEnd - lowerEnd > trailLength + 1))
            {
                lowerEnd = upperEnd - trailLength - 1;
            }

            // hole den up und right vector
            coVector up(0.0, 0.0, 1.0);
            coVector tangent;
            coVector right;

            // why? because streamlines from a plane with initial points out of domain will return a degenerated line
            // use only valid entries and skip the rest
            if (upperEnd == lowerEnd)
            {
                continue;
            }

            tangent[0] = xTangents[cornerList[lowerEnd]];
            tangent[1] = yTangents[cornerList[lowerEnd]];
            tangent[2] = zTangents[cornerList[lowerEnd]];

            right = tangent.cross(up);
            right.normalize();
            up = right.cross(tangent);
            up.normalize();
            right = right * tubeSize;
            up = up * tubeSize;

            // vorlauf
            xCoords[numCoords] = (float)(xPoints[cornerList[lowerEnd]] + right[0] - up[0]);
            yCoords[numCoords] = (float)(yPoints[cornerList[lowerEnd]] + right[1] - up[1]);
            zCoords[numCoords] = (float)(zPoints[cornerList[lowerEnd]] + right[2] - up[2]);
            numCoords++;
            xCoords[numCoords] = (float)(xPoints[cornerList[lowerEnd]] + right[0] + up[0]);
            yCoords[numCoords] = (float)(yPoints[cornerList[lowerEnd]] + right[1] + up[1]);
            zCoords[numCoords] = (float)(zPoints[cornerList[lowerEnd]] + right[2] + up[2]);
            numCoords++;
            xCoords[numCoords] = (float)(xPoints[cornerList[lowerEnd]] - right[0] + up[0]);
            yCoords[numCoords] = (float)(yPoints[cornerList[lowerEnd]] - right[1] + up[1]);
            zCoords[numCoords] = (float)(zPoints[cornerList[lowerEnd]] - right[2] + up[2]);
            numCoords++;
            xCoords[numCoords] = (float)(xPoints[cornerList[lowerEnd]] - right[0] - up[0]);
            yCoords[numCoords] = (float)(yPoints[cornerList[lowerEnd]] - right[1] - up[1]);
            zCoords[numCoords] = (float)(zPoints[cornerList[lowerEnd]] - right[2] - up[2]);
            numCoords++;

            for (int curCorner = lowerEnd + 1; curCorner < upperEnd; curCorner++)
            {
                // hole den up und right vector
                coVector up(0.0, 0.0, 1.0);
                coVector tangent;
                coVector right;

                tangent[0] = xTangents[cornerList[curCorner]];
                tangent[1] = yTangents[cornerList[curCorner]];
                tangent[2] = zTangents[cornerList[curCorner]];

                right = tangent.cross(up);
                right.normalize();
                up = right.cross(tangent);
                up.normalize();
                right = right * tubeSize;
                up = up * tubeSize;

                xCoords[numCoords] = (float)(xPoints[cornerList[curCorner]] + right[0] - up[0]);
                yCoords[numCoords] = (float)(yPoints[cornerList[curCorner]] + right[1] - up[1]);
                zCoords[numCoords] = (float)(zPoints[cornerList[curCorner]] + right[2] - up[2]);
                numCoords++;
                xCoords[numCoords] = (float)(xPoints[cornerList[curCorner]] + right[0] + up[0]);
                yCoords[numCoords] = (float)(yPoints[cornerList[curCorner]] + right[1] + up[1]);
                zCoords[numCoords] = (float)(zPoints[cornerList[curCorner]] + right[2] + up[2]);
                numCoords++;
                xCoords[numCoords] = (float)(xPoints[cornerList[curCorner]] - right[0] + up[0]);
                yCoords[numCoords] = (float)(yPoints[cornerList[curCorner]] - right[1] + up[1]);
                zCoords[numCoords] = (float)(zPoints[cornerList[curCorner]] - right[2] + up[2]);
                numCoords++;
                xCoords[numCoords] = (float)(xPoints[cornerList[curCorner]] - right[0] - up[0]);
                yCoords[numCoords] = (float)(yPoints[cornerList[curCorner]] - right[1] - up[1]);
                zCoords[numCoords] = (float)(zPoints[cornerList[curCorner]] - right[2] - up[2]);
                numCoords++;

                vertexList[numVertices++] = numCoords - 8;
                vertexList[numVertices++] = numCoords - 5;
                vertexList[numVertices++] = numCoords - 1;
                vertexList[numVertices++] = numCoords - 4;
                polygonList[numPolygons] = numPolygons * 4;
                numPolygons++;

                vertexList[numVertices++] = numCoords - 5;
                vertexList[numVertices++] = numCoords - 6;
                vertexList[numVertices++] = numCoords - 2;
                vertexList[numVertices++] = numCoords - 1;
                polygonList[numPolygons] = numPolygons * 4;
                numPolygons++;

                vertexList[numVertices++] = numCoords - 6;
                vertexList[numVertices++] = numCoords - 7;
                vertexList[numVertices++] = numCoords - 3;
                vertexList[numVertices++] = numCoords - 2;
                polygonList[numPolygons] = numPolygons * 4;
                numPolygons++;

                vertexList[numVertices++] = numCoords - 7;
                vertexList[numVertices++] = numCoords - 8;
                vertexList[numVertices++] = numCoords - 4;
                vertexList[numVertices++] = numCoords - 3;
                polygonList[numPolygons] = numPolygons * 4;
                numPolygons++;
            }
        }
        // create the polygons data object
        polygonObj = new coDoPolygons(name,
                                      numCoords, xCoords, yCoords, zCoords,
                                      numVertices, vertexList,
                                      numPolygons, polygonList);

        delete[] xCoords;
        delete[] yCoords;
        delete[] zCoords;
        delete[] vertexList;
        delete[] polygonList;
        polygonObj->copyAllAttributes(lines);

        return polygonObj;
    }
    else
    {
        std::cout << "ThickStreamlines: No LINES found!" << std::endl;
        return NULL;
    }
}

coDistributedObject *
ComplexModules::Bars(const char *name, coDistributedObject *points, float radius, coDistributedObject *tangents, const char *name_norm, coDistributedObject **normals)
{
    (void)normals;

    if (points && points->isType("SETELE"))
    {
        int no_e, dummy;
        const coDistributedObject *const *setListPoints = ((coDoSet *)points)->getAllElements(&no_e);
        const coDistributedObject *const *setListTangents = ((coDoSet *)tangents)->getAllElements(&dummy);
        coDistributedObject **outList = new coDistributedObject *[no_e + 1];
        //DistributedObject **normList = new DistributedObject*[no_e+1];
        outList[no_e] = NULL;
        //normList[no_e] = NULL;

        for (int currentSetEle = 0; currentSetEle < no_e; currentSetEle++)
        {
            string name_i = name;
            string name_norm_i = name_norm;
            char buf[16];
            sprintf(buf, "_%d", currentSetEle);
            name_i += buf;
            name_norm_i += buf;

            //coDoPoints * Points = (coDoPoints *)setListPoints[currentSetEle];
            coDoPolygons *polygonObj;
            int numPoints = ((coDoPoints *)setListPoints[currentSetEle])->getNumPoints();
            int trailLength = 20; // maximal trail length. might be shorter
            int realTrailLength = 0;
            float *xCoords = new float[numPoints * (8 + trailLength * 3)];
            float *yCoords = new float[numPoints * (8 + trailLength * 3)];
            float *zCoords = new float[numPoints * (8 + trailLength * 3)];
            int *vertexList = new int[numPoints * (24 + trailLength * 3)];
            int *polygonList = new int[numPoints * (6 + trailLength)];
            float sVal = radius / 2.0f;
            float *xPoints, *yPoints, *zPoints;
            float *v_x = NULL, *v_y = NULL, *v_z = NULL;

            ((coDoPoints *)setListPoints[currentSetEle])->getAddresses(&xPoints, &yPoints, &zPoints);
            if (setListTangents[currentSetEle]->isType("USTVDT"))
            {
                ((coDoVec3 *)setListTangents[currentSetEle])->getAddresses(&v_x, &v_y, &v_z);
            }

            // set up connectivity of bars
            for (int i = 0; i < numPoints; i++)
            {
                vertexList[24 * i + 0] = 8 * i + 0;
                vertexList[24 * i + 1] = 8 * i + 3;
                vertexList[24 * i + 2] = 8 * i + 7;
                vertexList[24 * i + 3] = 8 * i + 4;

                vertexList[24 * i + 4] = 8 * i + 3;
                vertexList[24 * i + 5] = 8 * i + 2;
                vertexList[24 * i + 6] = 8 * i + 6;
                vertexList[24 * i + 7] = 8 * i + 7;

                vertexList[24 * i + 8] = 8 * i + 0;
                vertexList[24 * i + 9] = 8 * i + 1;
                vertexList[24 * i + 10] = 8 * i + 2;
                vertexList[24 * i + 11] = 8 * i + 3;

                vertexList[24 * i + 12] = 8 * i + 0;
                vertexList[24 * i + 13] = 8 * i + 4;
                vertexList[24 * i + 14] = 8 * i + 5;
                vertexList[24 * i + 15] = 8 * i + 1;

                vertexList[24 * i + 16] = 8 * i + 1;
                vertexList[24 * i + 17] = 8 * i + 5;
                vertexList[24 * i + 18] = 8 * i + 6;
                vertexList[24 * i + 19] = 8 * i + 2;

                vertexList[24 * i + 20] = 8 * i + 7;
                vertexList[24 * i + 21] = 8 * i + 6;
                vertexList[24 * i + 22] = 8 * i + 5;
                vertexList[24 * i + 23] = 8 * i + 4;

                polygonList[6 * i + 0] = 24 * i + 0;
                polygonList[6 * i + 1] = 24 * i + 4;
                polygonList[6 * i + 2] = 24 * i + 8;
                polygonList[6 * i + 3] = 24 * i + 12;
                polygonList[6 * i + 4] = 24 * i + 16;
                polygonList[6 * i + 5] = 24 * i + 20;
            }

            // set up vertices of all bars (including rotation and translation)
            for (int i = 0; i < numPoints; i++)
            {
                coVector tangent(v_x ? v_x[i] : 0., v_y ? v_y[i] : 1., v_z ? v_z[i] : 0.);
                coVector up(0.0, 0.0, 1.0);
                coVector right;
                coMatrix transform;

                tangent.normalize();
                right = tangent.cross(up);
                up = right.cross(tangent);
                right.normalize();
                up.normalize();

                //transform.unity();

                transform.set(0, 0, tangent[0]);
                transform.set(1, 0, tangent[1]);
                transform.set(2, 0, tangent[2]);

                transform.set(0, 1, right[0]);
                transform.set(1, 1, right[1]);
                transform.set(2, 1, right[2]);

                transform.set(0, 2, up[0]);
                transform.set(1, 2, up[1]);
                transform.set(2, 2, up[2]);

                transform.set(0, 3, xPoints[i]);
                transform.set(1, 3, yPoints[i]);
                transform.set(2, 3, zPoints[i]);

                transform.set(3, 3, 1.0);

                // initialize untransformed points
                xCoords[8 * i + 0] = -0.95f * sVal;
                yCoords[8 * i + 0] = -0.5f * sVal;
                zCoords[8 * i + 0] = -0.15f * sVal;

                xCoords[8 * i + 1] = -0.95f * sVal;
                yCoords[8 * i + 1] = +0.5f * sVal;
                zCoords[8 * i + 1] = -0.15f * sVal;

                xCoords[8 * i + 2] = +0.95f * sVal;
                yCoords[8 * i + 2] = +0.5f * sVal;
                zCoords[8 * i + 2] = -0.15f * sVal;

                xCoords[8 * i + 3] = +0.95f * sVal;
                yCoords[8 * i + 3] = -0.5f * sVal;
                zCoords[8 * i + 3] = -0.15f * sVal;

                xCoords[8 * i + 4] = -0.95f * sVal;
                yCoords[8 * i + 4] = -0.5f * sVal;
                zCoords[8 * i + 4] = +0.15f * sVal;

                xCoords[8 * i + 5] = -0.95f * sVal;
                yCoords[8 * i + 5] = +0.5f * sVal;
                zCoords[8 * i + 5] = +0.15f * sVal;

                xCoords[8 * i + 6] = +0.95f * sVal;
                yCoords[8 * i + 6] = +0.5f * sVal;
                zCoords[8 * i + 6] = +0.15f * sVal;

                xCoords[8 * i + 7] = +0.95f * sVal;
                yCoords[8 * i + 7] = -0.5f * sVal;
                zCoords[8 * i + 7] = +0.15f * sVal;

                // transform initialized points
                for (int j = 0; j < 8; j++)
                {
                    coVector v_(xCoords[8 * i + j], yCoords[8 * i + j], zCoords[8 * i + j]);

                    v_ = transform * v_;

                    xCoords[8 * i + j] = (float)v_[0];
                    yCoords[8 * i + j] = (float)v_[1];
                    zCoords[8 * i + j] = (float)v_[2];
                }
            }

            // create the polygons data object
            //polygonObj = new coDoPolygons("Poly", numPoints*8, xCoords, yCoords, zCoords, numPoints*24, vertexList, numPoints*6, polygonList);
            polygonObj = new coDoPolygons(name_i.c_str(),
                                          numPoints * (8 + realTrailLength * 3), xCoords, yCoords, zCoords,
                                          numPoints * (24 + realTrailLength * 3), vertexList,
                                          numPoints * (6 + realTrailLength), polygonList);

            delete[] xCoords;
            delete[] yCoords;
            delete[] zCoords;
            delete[] vertexList;
            delete[] polygonList;
            //polygonObj->copyAllAttributes(points);
            //return polygonObj;
            outList[currentSetEle] = polygonObj;
        }

        coDistributedObject *outSet = new coDoSet(name, outList);
        //*normals = new coDoSet(name_norm,normList);
        outSet->copyAllAttributes(points);
        for (int i = 0; i < no_e; i++)
        {
            delete outList[i];
            delete setListPoints[i]; // should be done
            //delete normList[i];
        }
        delete[] setListPoints; // should be done
        delete[] outList;
        //delete [] normList;
        return outSet;
    }
    else if (points && points->isType("POINTS"))
    {
        std::cout << "POINTS found, but SETELE needed!" << std::endl;
    }
    std::cout << "SHOULD NEVER BE REACHED!" << std::endl;
    return (new coDoTriangleStrips(name, 0, 0, 0)); // should never be reached
}

coDistributedObject *
ComplexModules::Compass(const char *name, const coDistributedObject *points, float radius, const coDistributedObject *tangents, const char *name_norm, coDistributedObject **normals, coDistributedObject **colors)
{
    static int currentSetEle = 0; // used to determine the current set element later on in this function
    if (points && points->isType("SETELE"))
    {
        int no_e;
        const coDistributedObject *const *setList = ((coDoSet *)points)->getAllElements(&no_e);
        coDistributedObject **outList = new coDistributedObject *[no_e + 1];
        //coDistributedObject **normList = new coDistributedObject*[no_e+1];
        coDistributedObject **outListColors = NULL;

        if (colors)
        {
            outListColors = new coDistributedObject *[no_e + 1];
        }

        // make endings for coDoSet
        outList[no_e] = NULL;
        //normList[no_e] = NULL;
        if (colors)
        {
            outListColors[no_e] = NULL;
        }

        for (currentSetEle = 0; currentSetEle < no_e; ++currentSetEle)
        {
            string name_i = name;
            string name_norm_i = name_norm;
            string name_colors_i = "CompassColors";
            char buf[16];
            sprintf(buf, "_%d", currentSetEle);
            name_i += buf;
            name_norm_i += buf;
            name_colors_i += buf;

            outList[currentSetEle] = Compass(name_i.c_str(), setList[currentSetEle], radius, tangents, name_norm_i.c_str(), normals, colors); //&normList[i]);

            // assign colors to every vertex
            if (colors)
            {
                int no_p = ((coDoPoints *)(setList[currentSetEle]))->getNumPoints(); // anzahl der punkte in diesem set element
                outListColors[currentSetEle] = new coDoRGBA(name_colors_i.c_str(), no_p * 10);
                for (int i = 0; i < no_p; i++)
                {
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 0, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 1, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 2, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 3, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 4, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 5, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 6, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 7, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 8, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 10 + 9, 1.0, 0.0, 0.0);
                }
            }
        }

        coDistributedObject *outSet = new coDoSet(name, outList);
        //*normals = new coDoSet(name_norm,normList);
        if (colors)
        {
            *colors = new coDoSet("CompassColors", outListColors);
        }

        outSet->copyAllAttributes(points);

        for (int i = 0; i < no_e; i++)
        {
            delete outList[i];
            delete setList[i];
            //delete normList[i];
            if (colors)
            {
                delete outListColors[i];
            }
        }
        delete[] setList;
        delete[] outList;
        //delete [] normList;
        if (colors)
        {
            delete[] outListColors;
        }

        return outSet;
    }
    else if (points && points->isType("POINTS"))
    {
        const coDoPoints *Points = (coDoPoints *)points;
        int numPoints = Points->getNumPoints();
        coDoPolygons *polygonObj;
        float *xCoords = new float[numPoints * 10];
        float *yCoords = new float[numPoints * 10];
        float *zCoords = new float[numPoints * 10];
        int *vertexList = new int[numPoints * 24];
        int *polygonList = new int[numPoints * 8];
        float sVal = radius / 2.0f;
        const coDistributedObject *const *setList = NULL;
        float *xPoints, *yPoints, *zPoints;
        float *v_x, *v_y, *v_z;

        Points->getAddresses(&xPoints, &yPoints, &zPoints);

        if (tangents->isType("SETELE"))
        {
            int num_setele;
            setList = ((coDoSet *)tangents)->getAllElements(&num_setele);
        }

        // set up connectivity
        for (int i = 0; i < numPoints; i++)
        {
            vertexList[24 * i + 0] = 10 * i + 0;
            vertexList[24 * i + 1] = 10 * i + 1;
            vertexList[24 * i + 2] = 10 * i + 4;

            vertexList[24 * i + 3] = 10 * i + 1;
            vertexList[24 * i + 4] = 10 * i + 2;
            vertexList[24 * i + 5] = 10 * i + 4;

            vertexList[24 * i + 6] = 10 * i + 2;
            vertexList[24 * i + 7] = 10 * i + 3;
            vertexList[24 * i + 8] = 10 * i + 4;

            vertexList[24 * i + 9] = 10 * i + 3;
            vertexList[24 * i + 10] = 10 * i + 0;
            vertexList[24 * i + 11] = 10 * i + 4;

            vertexList[24 * i + 12] = 10 * i + 5;
            vertexList[24 * i + 13] = 10 * i + 8;
            vertexList[24 * i + 14] = 10 * i + 9;

            vertexList[24 * i + 15] = 10 * i + 8;
            vertexList[24 * i + 16] = 10 * i + 7;
            vertexList[24 * i + 17] = 10 * i + 9;

            vertexList[24 * i + 18] = 10 * i + 7;
            vertexList[24 * i + 19] = 10 * i + 6;
            vertexList[24 * i + 20] = 10 * i + 9;

            vertexList[24 * i + 21] = 10 * i + 6;
            vertexList[24 * i + 22] = 10 * i + 5;
            vertexList[24 * i + 23] = 10 * i + 9;

            polygonList[8 * i + 0] = 24 * i + 0;
            polygonList[8 * i + 1] = 24 * i + 3;
            polygonList[8 * i + 2] = 24 * i + 6;
            polygonList[8 * i + 3] = 24 * i + 9;
            polygonList[8 * i + 4] = 24 * i + 12;
            polygonList[8 * i + 5] = 24 * i + 15;
            polygonList[8 * i + 6] = 24 * i + 18;
            polygonList[8 * i + 7] = 24 * i + 21;
        }

        // set up vertices of all cubes (including rotation and translation)
        for (int i = 0; i < numPoints; i++)
        {
            coVector tangent(0.0, 1.0, 0.0);
            if (setList && setList[currentSetEle]->isType("USTVDT"))
            {
                ((coDoVec3 *)setList[currentSetEle])->getAddresses(&v_x, &v_y, &v_z);
                //int num = ((coDoVec3 *)setList[currentSetEle])->getNumPoints();  // unused
                tangent[0] = v_x[i];
                tangent[1] = v_y[i];
                tangent[2] = v_z[i];
            }

            coVector up(0.0, 0.0, 1.0);
            coVector right;
            coMatrix transform;

            tangent.normalize();
            right = tangent.cross(up);
            up = right.cross(tangent);
            right.normalize();
            up.normalize();

            //transform.unity();

            transform.set(0, 0, tangent[0]);
            transform.set(1, 0, tangent[1]);
            transform.set(2, 0, tangent[2]);

            transform.set(0, 1, right[0]);
            transform.set(1, 1, right[1]);
            transform.set(2, 1, right[2]);

            transform.set(0, 2, up[0]);
            transform.set(1, 2, up[1]);
            transform.set(2, 2, up[2]);

            transform.set(0, 3, xPoints[i]);
            transform.set(1, 3, yPoints[i]);
            transform.set(2, 3, zPoints[i]);

            transform.set(3, 3, 1.0);

            // initialize untransformed points
            xCoords[10 * i + 0] = 0.0;
            yCoords[10 * i + 0] = +0.15f * sVal;
            zCoords[10 * i + 0] = +0.15f * sVal;

            xCoords[10 * i + 1] = 0.0;
            yCoords[10 * i + 1] = -0.15f * sVal;
            zCoords[10 * i + 1] = +0.15f * sVal;

            xCoords[10 * i + 2] = 0.0;
            yCoords[10 * i + 2] = -0.15f * sVal;
            zCoords[10 * i + 2] = -0.15f * sVal;

            xCoords[10 * i + 3] = 0.0;
            yCoords[10 * i + 3] = +0.15f * sVal;
            zCoords[10 * i + 3] = -0.15f * sVal;

            xCoords[10 * i + 4] = -0.95f * sVal;
            yCoords[10 * i + 4] = 0.0;
            zCoords[10 * i + 4] = 0.0;

            xCoords[10 * i + 5] = 0.0;
            yCoords[10 * i + 5] = +0.15f * sVal;
            zCoords[10 * i + 5] = +0.15f * sVal;

            xCoords[10 * i + 6] = 0.0;
            yCoords[10 * i + 6] = -0.15f * sVal;
            zCoords[10 * i + 6] = +0.15f * sVal;

            xCoords[10 * i + 7] = 0.0;
            yCoords[10 * i + 7] = -0.15f * sVal;
            zCoords[10 * i + 7] = -0.15f * sVal;

            xCoords[10 * i + 8] = 0.0;
            yCoords[10 * i + 8] = +0.15f * sVal;
            zCoords[10 * i + 8] = -0.15f * sVal;

            xCoords[10 * i + 9] = +0.95f * sVal;
            yCoords[10 * i + 9] = 0.0;
            zCoords[10 * i + 9] = 0.0;

            // transform initialized points
            for (int j = 0; j < 10; j++)
            {
                coVector v_(xCoords[10 * i + j], yCoords[10 * i + j], zCoords[10 * i + j]);

                v_ = transform * v_;

                xCoords[10 * i + j] = (float)v_[0];
                yCoords[10 * i + j] = (float)v_[1];
                zCoords[10 * i + j] = (float)v_[2];
            }
        }

        // create the polygons data object
        polygonObj = new coDoPolygons(name, numPoints * 10, xCoords, yCoords, zCoords, numPoints * 24, vertexList, numPoints * 8, polygonList);

        delete[] xCoords;
        delete[] yCoords;
        delete[] zCoords;
        delete[] vertexList;
        delete[] polygonList;
        //polygonObj->copyAllAttributes(points);
        return polygonObj;
    }
    return (new coDoTriangleStrips(name, 0, 0, 0)); // should never be reached
}

coDistributedObject *
ComplexModules::BarMagnets(const char *name, const coDistributedObject *points, float radius, const coDistributedObject *tangents, const char *name_norm, coDistributedObject **normals, coDistributedObject **colors)
{
    static int currentSetEle = 0; // used to determine the current set element later on in this function
    if (points && points->isType("SETELE"))
    {
        int no_e;
        const coDistributedObject *const *setList = ((coDoSet *)points)->getAllElements(&no_e);
        coDistributedObject **outList = new coDistributedObject *[no_e + 1];
        //coDistributedObject **normList = new coDistributedObject*[no_e+1];
        coDistributedObject **outListColors = NULL;

        if (colors)
        {
            outListColors = new coDistributedObject *[no_e + 1];
        }

        // make endings for coDoSet
        outList[no_e] = NULL;
        //normList[no_e] = NULL;
        if (colors)
        {
            outListColors[no_e] = NULL;
        }

        for (currentSetEle = 0; currentSetEle < no_e; ++currentSetEle)
        {
            string name_i = name;
            string name_norm_i = name_norm;
            string name_colors_i = "BarMagnetColors";
            char buf[16];
            sprintf(buf, "_%d", currentSetEle);
            name_i += buf;
            name_norm_i += buf;
            name_colors_i += buf;

            outList[currentSetEle] = BarMagnets(name_i.c_str(), setList[currentSetEle], radius, tangents, name_norm_i.c_str(), normals, colors); //&normList[i]);

            // assign colors to every vertex
            if (colors)
            {
                int no_p = ((coDoPoints *)(setList[currentSetEle]))->getNumPoints(); // anzahl der punkte in diesem set element
                outListColors[currentSetEle] = new coDoRGBA(name_colors_i.c_str(), no_p * 16);
                for (int i = 0; i < no_p; i++)
                {
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 0, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 1, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 2, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 3, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 4, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 5, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 6, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 7, 0.0, 1.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 8, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 9, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 10, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 11, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 12, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 13, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 14, 1.0, 0.0, 0.0);
                    ((coDoRGBA *)outListColors[currentSetEle])->setFloatRGBA(i * 16 + 15, 1.0, 0.0, 0.0);
                }
            }
        }

        coDistributedObject *outSet = new coDoSet(name, outList);
        //*normals = new coDoSet(name_norm,normList);
        if (colors)
        {
            *colors = new coDoSet("BarMagnetColors", outListColors);
        }

        outSet->copyAllAttributes(points);

        for (int i = 0; i < no_e; i++)
        {
            delete outList[i];
            delete setList[i];
            //delete normList[i];
            if (colors)
            {
                delete outListColors[i];
            }
        }
        delete[] setList;
        delete[] outList;
        //delete [] normList;
        if (colors)
        {
            delete[] outListColors;
        }

        return outSet;
    }
    else if (points && points->isType("POINTS"))
    {
        coDoPoints *Points = (coDoPoints *)points;
        int numPoints = Points->getNumPoints();
        coDoPolygons *polygonObj;
        float *xCoords = new float[numPoints * 16];
        float *yCoords = new float[numPoints * 16];
        float *zCoords = new float[numPoints * 16];
        int *vertexList = new int[numPoints * 40];
        int *polygonList = new int[numPoints * 10];
        float sVal = radius / 2.0f;
        const coDistributedObject *const *setList = NULL;
        float *xPoints, *yPoints, *zPoints;
        float *v_x = NULL, *v_y = NULL, *v_z = NULL;

        Points->getAddresses(&xPoints, &yPoints, &zPoints);

        if (tangents->isType("SETELE"))
        {
            int num_setele;
            setList = ((coDoSet *)tangents)->getAllElements(&num_setele);
        }

        // set up connectivity
        for (int i = 0; i < numPoints; i++)
        {
            // green half of bar magnet
            vertexList[40 * i + 0] = 16 * i + 0;
            vertexList[40 * i + 1] = 16 * i + 1;
            vertexList[40 * i + 2] = 16 * i + 5;
            vertexList[40 * i + 3] = 16 * i + 4;

            vertexList[40 * i + 4] = 16 * i + 1;
            vertexList[40 * i + 5] = 16 * i + 2;
            vertexList[40 * i + 6] = 16 * i + 6;
            vertexList[40 * i + 7] = 16 * i + 5;

            vertexList[40 * i + 8] = 16 * i + 2;
            vertexList[40 * i + 9] = 16 * i + 3;
            vertexList[40 * i + 10] = 16 * i + 7;
            vertexList[40 * i + 11] = 16 * i + 6;

            vertexList[40 * i + 12] = 16 * i + 3;
            vertexList[40 * i + 13] = 16 * i + 0;
            vertexList[40 * i + 14] = 16 * i + 4;
            vertexList[40 * i + 15] = 16 * i + 7;

            vertexList[40 * i + 16] = 16 * i + 4;
            vertexList[40 * i + 17] = 16 * i + 5;
            vertexList[40 * i + 18] = 16 * i + 6;
            vertexList[40 * i + 19] = 16 * i + 7;

            // red half of bar magnet
            vertexList[40 * i + 20] = 16 * i + 9;
            vertexList[40 * i + 21] = 16 * i + 8;
            vertexList[40 * i + 22] = 16 * i + 12;
            vertexList[40 * i + 23] = 16 * i + 13;

            vertexList[40 * i + 24] = 16 * i + 8;
            vertexList[40 * i + 25] = 16 * i + 11;
            vertexList[40 * i + 26] = 16 * i + 15;
            vertexList[40 * i + 27] = 16 * i + 12;

            vertexList[40 * i + 28] = 16 * i + 11;
            vertexList[40 * i + 29] = 16 * i + 10;
            vertexList[40 * i + 30] = 16 * i + 14;
            vertexList[40 * i + 31] = 16 * i + 15;

            vertexList[40 * i + 32] = 16 * i + 10;
            vertexList[40 * i + 33] = 16 * i + 9;
            vertexList[40 * i + 34] = 16 * i + 13;
            vertexList[40 * i + 35] = 16 * i + 14;

            vertexList[40 * i + 36] = 16 * i + 15;
            vertexList[40 * i + 37] = 16 * i + 14;
            vertexList[40 * i + 38] = 16 * i + 13;
            vertexList[40 * i + 39] = 16 * i + 12;

            polygonList[10 * i + 0] = 40 * i + 0;
            polygonList[10 * i + 1] = 40 * i + 4;
            polygonList[10 * i + 2] = 40 * i + 8;
            polygonList[10 * i + 3] = 40 * i + 12;
            polygonList[10 * i + 4] = 40 * i + 16;
            polygonList[10 * i + 5] = 40 * i + 20;
            polygonList[10 * i + 6] = 40 * i + 24;
            polygonList[10 * i + 7] = 40 * i + 28;
            polygonList[10 * i + 8] = 40 * i + 32;
            polygonList[10 * i + 9] = 40 * i + 36;
        }

        // set up vertices of all bar magnets (including rotation and translation)
        for (int i = 0; i < numPoints; i++)
        {
            coVector tangent(0., 1., 0.);
            if (setList && setList[currentSetEle]->isType("USTVDT"))
            {
                ((coDoVec3 *)setList[currentSetEle])->getAddresses(&v_x, &v_y, &v_z);
                //int num = ((coDoVec3 *)setList[currentSetEle])->getNumPoints();  // unused
                tangent[0] = v_x[i];
                tangent[1] = v_y[i];
                tangent[2] = v_z[i];
            }

            coVector up(0.0, 0.0, 1.0);
            coVector right;
            coMatrix transform;

            tangent.normalize();
            right = tangent.cross(up);
            up = right.cross(tangent);
            right.normalize();
            up.normalize();

            transform.set(0, 0, tangent[0]);
            transform.set(1, 0, tangent[1]);
            transform.set(2, 0, tangent[2]);

            transform.set(0, 1, right[0]);
            transform.set(1, 1, right[1]);
            transform.set(2, 1, right[2]);

            transform.set(0, 2, up[0]);
            transform.set(1, 2, up[1]);
            transform.set(2, 2, up[2]);

            transform.set(0, 3, xPoints[i]);
            transform.set(1, 3, yPoints[i]);
            transform.set(2, 3, zPoints[i]);

            transform.set(3, 3, 1.0);

            // initialize untransformed points
            // green half
            xCoords[16 * i + 0] = 0.0;
            yCoords[16 * i + 0] = +0.5f * sVal;
            zCoords[16 * i + 0] = +0.15f * sVal;

            xCoords[16 * i + 1] = 0.0;
            yCoords[16 * i + 1] = -0.5f * sVal;
            zCoords[16 * i + 1] = +0.15f * sVal;

            xCoords[16 * i + 2] = 0.0;
            yCoords[16 * i + 2] = -0.5f * sVal;
            zCoords[16 * i + 2] = -0.15f * sVal;

            xCoords[16 * i + 3] = 0.0;
            yCoords[16 * i + 3] = +0.5f * sVal;
            zCoords[16 * i + 3] = -0.15f * sVal;

            xCoords[16 * i + 4] = -0.95f * sVal;
            yCoords[16 * i + 4] = +0.5f * sVal;
            zCoords[16 * i + 4] = +0.15f * sVal;

            xCoords[16 * i + 5] = -0.95f * sVal;
            yCoords[16 * i + 5] = -0.5f * sVal;
            zCoords[16 * i + 5] = +0.15f * sVal;

            xCoords[16 * i + 6] = -0.95f * sVal;
            yCoords[16 * i + 6] = -0.5f * sVal;
            zCoords[16 * i + 6] = -0.15f * sVal;

            xCoords[16 * i + 7] = -0.95f * sVal;
            yCoords[16 * i + 7] = +0.5f * sVal;
            zCoords[16 * i + 7] = -0.15f * sVal;

            // red half
            xCoords[16 * i + 8] = 0.0;
            yCoords[16 * i + 8] = +0.5f * sVal;
            zCoords[16 * i + 8] = +0.15f * sVal;

            xCoords[16 * i + 9] = 0.0;
            yCoords[16 * i + 9] = -0.5f * sVal;
            zCoords[16 * i + 9] = +0.15f * sVal;

            xCoords[16 * i + 10] = 0.0;
            yCoords[16 * i + 10] = -0.5f * sVal;
            zCoords[16 * i + 10] = -0.15f * sVal;

            xCoords[16 * i + 11] = 0.0;
            yCoords[16 * i + 11] = +0.5f * sVal;
            zCoords[16 * i + 11] = -0.15f * sVal;

            xCoords[16 * i + 12] = +0.95f * sVal;
            yCoords[16 * i + 12] = +0.5f * sVal;
            zCoords[16 * i + 12] = +0.15f * sVal;

            xCoords[16 * i + 13] = +0.95f * sVal;
            yCoords[16 * i + 13] = -0.5f * sVal;
            zCoords[16 * i + 13] = +0.15f * sVal;

            xCoords[16 * i + 14] = +0.95f * sVal;
            yCoords[16 * i + 14] = -0.5f * sVal;
            zCoords[16 * i + 14] = -0.15f * sVal;

            xCoords[16 * i + 15] = +0.95f * sVal;
            yCoords[16 * i + 15] = +0.5f * sVal;
            zCoords[16 * i + 15] = -0.15f * sVal;

            // transform initialized points
            for (int j = 0; j < 16; j++)
            {
                coVector v_(xCoords[16 * i + j], yCoords[16 * i + j], zCoords[16 * i + j]);

                v_ = transform * v_;

                xCoords[16 * i + j] = (float)v_[0];
                yCoords[16 * i + j] = (float)v_[1];
                zCoords[16 * i + j] = (float)v_[2];
            }
        }

        // create the polygons data object
        polygonObj = new coDoPolygons(name, numPoints * 16, xCoords, yCoords, zCoords, numPoints * 40, vertexList, numPoints * 10, polygonList);

        delete[] xCoords;
        delete[] yCoords;
        delete[] zCoords;
        delete[] vertexList;
        delete[] polygonList;
        //polygonObj->copyAllAttributes(points);
        return polygonObj;
    }
    return (new coDoTriangleStrips(name, 0, 0, 0)); // should never be reached
}

coDistributedObject *
ComplexModules::croppedLinesSet(const coDistributedObject *lines, int croppedLength)
{
    if (!lines->isType("SETELE"))
    {
        Covise::sendError("Need a coDoSet of coDoLines!");
        return NULL;
    }

    int numSetEle;
    const coDistributedObject *const *setListLines = ((coDoSet *)lines)->getAllElements(&numSetEle);
    coDistributedObject **croppedLinesList = new coDistributedObject *[numSetEle + 1];
    croppedLinesList[numSetEle] = NULL;

    for (int curSetEle = 0; curSetEle < numSetEle; curSetEle++)
    {
        if (!((coDoLines *)setListLines[curSetEle])->isType("LINES"))
        {
            Covise::sendError("Element of coDoSet is not a coDoLines!");
            return NULL;
        }

        int *lineList, *cornerList;
        float *x_coords, *y_coords, *z_coords;
        int *croppedLineList, *croppedCornerList;
        float *cropped_x_coords, *cropped_y_coords, *cropped_z_coords;

        int numLines = ((coDoLines *)setListLines[curSetEle])->getNumLines();
        int numCorners = ((coDoLines *)setListLines[curSetEle])->getNumVertices();

        ((coDoLines *)setListLines[curSetEle])->getAddresses(&x_coords, &y_coords, &z_coords, &cornerList, &lineList);

        // zaehlen, wieviele vertices der linien benutzt werden (noetig fuer die anzahl der vertices der croppedLines) und entsprechend speicher reservieren
        int totalVertices = 0;
        for (int curLine = 0; curLine < numLines; curLine++)
        {
            int lineLength = (curLine != numLines - 1) ? (lineList[curLine + 1] - lineList[curLine]) : (numCorners - lineList[numLines - 1]); //num of corners for current line
            totalVertices += (lineLength > croppedLength + 1) ? croppedLength + 1 : lineLength;
        }
        croppedLinesList[curSetEle] = new coDoLines("Cropped_Lines", totalVertices, totalVertices, numLines);
        ((coDoLines *)croppedLinesList[curSetEle])->getAddresses(&cropped_x_coords, &cropped_y_coords, &cropped_z_coords, &croppedCornerList, &croppedLineList);

        //int numCroppedLines = 0;
        int numCroppedCorners = 0;
        int numCroppedCoords = 0;

        // werte in cropped line kopieren
        for (int curLine = 0; curLine < numLines; curLine++)
        {
            int lineLength = (curLine != numLines - 1) ? (lineList[curLine + 1] - lineList[curLine]) : (numCorners - lineList[numLines - 1]); //num of corners for current line
            int l = (lineLength > croppedLength + 1) ? croppedLength + 1 : lineLength;
            int lastCorner = lineList[curLine] + lineLength;

            croppedLineList[curLine] = numCroppedCorners;
            for (int curCorner = lastCorner - l; curCorner < lastCorner; curCorner++)
            {
                croppedCornerList[numCroppedCorners] = numCroppedCoords;
                cropped_x_coords[numCroppedCoords] = x_coords[cornerList[curCorner]];
                cropped_y_coords[numCroppedCoords] = y_coords[cornerList[curCorner]];
                cropped_z_coords[numCroppedCoords] = z_coords[cornerList[curCorner]];
                numCroppedCorners++;
                numCroppedCoords++;
            }
        }
    }

    coDistributedObject *outSet = new coDoSet("CroppedLinesSet", croppedLinesList);
    outSet->copyAllAttributes(lines);
    return outSet;
}
