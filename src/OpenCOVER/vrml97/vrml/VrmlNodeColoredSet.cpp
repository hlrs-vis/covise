/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeColoredSet.cpp

#include <cfloat>

#include "VrmlNodeColoredSet.h"

#include "VrmlNodeType.h"
#include "VrmlNodeColor.h"
#include "VrmlNodeColorRGBA.h"
#include "VrmlNodeCoordinate.h"
#include "VrmlNodeNormal.h"
#include "VrmlNodeTextureCoordinate.h"
#include "VrmlNodeMultiTextureCoordinate.h"
#include "VrmlNodeTextureCoordinateGenerator.h"

#include "System.h"
#include "Viewer.h"

using namespace vrml;

void VrmlNodeColoredSet::initFields(VrmlNodeColoredSet *node, VrmlNodeType *t)
{
    VrmlNodeGeometry::initFields(node, t); // Parent class

    initFieldsHelper(node, t,
                     exposedField("color", node->d_color),
                     field("colorPerVertex", node->d_colorPerVertex),
                     exposedField("coord", node->d_coord));
}

VrmlNodeColoredSet::VrmlNodeColoredSet(VrmlScene *scene, const std::string &name)
    : VrmlNodeGeometry(scene, name)
    , d_colorPerVertex(true)
{
}

bool VrmlNodeColoredSet::isModified() const
{
    return (d_modified || (d_color.get() && d_color.get()->isModified()) || (d_coord.get() && d_coord.get()->isModified()));
}

void VrmlNodeColoredSet::clearFlags()
{
    VrmlNode::clearFlags();
    if (d_color.get())
        d_color.get()->clearFlags();
    if (d_coord.get())
        d_coord.get()->clearFlags();
}

void VrmlNodeColoredSet::addToScene(VrmlScene *s, const char *rel)
{
    nodeStack.push_front(this);
    d_scene = s;
    if (d_color.get())
        d_color.get()->addToScene(s, rel);
    if (d_coord.get())
        d_coord.get()->addToScene(s, rel);
    nodeStack.pop_front();
}

void VrmlNodeColoredSet::copyRoutes(VrmlNamespace *ns)
{
    nodeStack.push_front(this);
    VrmlNode::copyRoutes(ns);
    if (d_color.get())
        d_color.get()->copyRoutes(ns);
    if (d_coord.get())
        d_coord.get()->copyRoutes(ns);
    nodeStack.pop_front();
}

VrmlNodeColor *VrmlNodeColoredSet::color()
{
    return d_color.get() ? d_color.get()->as<VrmlNodeColor>() : 0;
}

VrmlNode *VrmlNodeColoredSet::getCoordinate()
{
    return d_coord.get();
}

static void generateDefaultTextureCoordinates(float **tc, int numberTexture,
                                              VrmlMFVec3f &coord, VrmlMFInt &ci)
{
    int nvert = coord.size();
    int *coordIndex = ci.get();
    int nci = ci.size();

    // find min/max
    float min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
    float max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    float width[3];
    int i, n, s, t;
    if (nvert > 0)
    {
        for (i = 0; i < nci; i++)
        {
            if (coordIndex[i] > 0)
            {
                for (n = 0; n < 3; n++)
                {
                    if (min[n] > coord[coordIndex[i]][n])
                        min[n] = coord[coordIndex[i]][n];
                    if (max[n] < coord[coordIndex[i]][n])
                        max[n] = coord[coordIndex[i]][n];
                }
            }
        }
    }
    for (n = 0; n < 3; n++)
        width[n] = max[n] - min[n];
    s = 0;
    t = 1;

    max[0] = max[1] = max[2] = -1.0;
    for (n = 0; n < 3; n++)
        if (width[n] > max[0])
        {
            max[0] = width[n];
            s = n;
        }
    for (n = 0; n < 3; n++)
        if ((n != s) && (width[n] > max[1]))
        {
            max[1] = width[n];
            t = n;
        }
    tc[numberTexture] = new float[nvert * 2];

    for (i = 0; i < nvert; i++)
    {
        tc[numberTexture][2 * i] = (coord[i][s] - min[s]) / width[s];
        tc[numberTexture][2 * i + 1] = (coord[i][t] - min[t]) / width[s];
    }
}

static int getTexCoordGeneratorName(VrmlNodeTextureCoordinateGenerator *generator)
{
    int ret = 0;
    if (strcmp(generator->getField("mode")->toSFString()->get(), "SPHERE") == 0)
        ret = Viewer::TEXTURE_COORDINATE_GENERATOR_MODE_SPHERE;
    else if (strcmp(generator->getField("mode")->toSFString()->get(), "CAMERASPACEREFLECTIONVECTOR") == 0)
        ret = Viewer::TEXTURE_COORDINATE_GENERATOR_MODE_CAMERASPACEREFLECTIONVECTOR;
    else if (strcmp(generator->getField("mode")->toSFString()->get(), "COORD") == 0)
        ret = Viewer::TEXTURE_COORDINATE_GENERATOR_MODE_COORD;
    else if (strcmp(generator->getField("mode")->toSFString()->get(), "CAMERASPACENORMAL") == 0)
        ret = Viewer::TEXTURE_COORDINATE_GENERATOR_MODE_CAMERASPACENORMAL;
    else if (strcmp(generator->getField("mode")->toSFString()->get(), "COORD-EYE") == 0)
        ret = Viewer::TEXTURE_COORDINATE_GENERATOR_MODE_COORD_EYE;
    return ret;
}

// TO DO: stripify, crease angle, generate normals ...

Viewer::Object VrmlNodeColoredSet::insertGeometry(Viewer *viewer,
                                                  unsigned int optMask,
                                                  VrmlMFInt &coordIndex,
                                                  VrmlMFInt &colorIndex,
                                                  VrmlSFFloat &creaseAngle,
                                                  VrmlSFNode &normal,
                                                  VrmlMFInt &normalIndex,
                                                  VrmlSFNode &texCoord,
                                                  VrmlMFInt &texCoordIndex,
                                                  VrmlSFNode &texCoord2,
                                                  VrmlMFInt &texCoordIndex2,
                                                  VrmlSFNode &texCoord3,
                                                  VrmlMFInt &texCoordIndex3,
                                                  VrmlSFNode &texCoord4,
                                                  VrmlMFInt &texCoordIndex4)
{
    //cerr << "VrmlNodeColoredSet::insertGeometry" << endl;

    Viewer::Object obj = 0;

    if (d_coord.get() && coordIndex.size() > 0)
    {
        VrmlMFVec3f &coord = d_coord.get()->as<VrmlNodeCoordinate>()->coordinate();
        int nvert = coord.size();
        float *tc[Viewer::NUM_TEXUNITS], *color = 0, *norm = 0;
        int ntc[Viewer::NUM_TEXUNITS];
        int ntci[Viewer::NUM_TEXUNITS], *tci[Viewer::NUM_TEXUNITS]; // texture coordinate indices
        int nci = 0, *ci = 0; // color indices
        int nni = 0, *ni = 0; // normal indices
        int *localci = NULL;
        float *localtc = NULL;
        float *localvn = NULL; // per vertex normals
        int *localni = NULL;

        int texCoordGeneratorNames[Viewer::NUM_TEXUNITS];
        float *texCoordGeneratorParameter[Viewer::NUM_TEXUNITS];
        int numTexCoordGeneratorParameter[Viewer::NUM_TEXUNITS];
        for (int i = 0; i < Viewer::NUM_TEXUNITS; i++)
        {
            texCoordGeneratorNames[i] = 0;
            texCoordGeneratorParameter[i] = NULL;
            numTexCoordGeneratorParameter[i] = 0;
        }
        int *texCoordGeneratorName = NULL;

        float **multi_tc = tc;
        int *multi_ntci = ntci;
        int **multi_tci = tci;

        for (int i = 0; i < Viewer::NUM_TEXUNITS; i++)
        {
            tc[i] = NULL;
            ntc[i] = 0;
            tci[i] = NULL;
            ntci[i] = 0;
        }

        // Get texture coordinate Index for texture 1
        ntci[0] = texCoordIndex.size();
        if (ntci[0])
            tci[0] = texCoordIndex.get();
        else
        {
            ntci[0] = coordIndex.size();
            tci[0] = coordIndex.get();
        }

        // Get texture coordinate Index for texture 2
        ntci[1] = texCoordIndex2.size();
        if (ntci[1])
            tci[1] = texCoordIndex2.get();
        // Get texture coordinate Index for texture 3
        ntci[2] = texCoordIndex3.size();
        if (ntci[2])
            tci[2] = texCoordIndex3.get();
        // Get texture coordinate Index for texture 4
        ntci[3] = texCoordIndex4.size();
        if (ntci[3])
            tci[3] = texCoordIndex4.get();

        bool hasTextureCoordinate = false;
        VrmlNode *stdTexCoord = texCoord.get();
        if (stdTexCoord && (strcmp(stdTexCoord->nodeType()->getName(), "MultiTextureCoordinate") == 0))
        {
            VrmlNodeMultiTextureCoordinate *multiTexCoord = stdTexCoord->as<VrmlNodeMultiTextureCoordinate>();
            if (multiTexCoord)
            {
                VrmlMFNode mTextureCoordinate = multiTexCoord->texCoord();
                int numberTexCoords = mTextureCoordinate.size();
                multi_tc = new float *[Viewer::NUM_TEXUNITS];
                multi_ntci = new int[Viewer::NUM_TEXUNITS];
                multi_tci = new int *[Viewer::NUM_TEXUNITS];
                for (int i = 0; i < Viewer::NUM_TEXUNITS; i++)
                {
                    multi_tc[i] = NULL;
                    multi_ntci[i] = 0;
                    multi_tci[i] = NULL;
                }
                for (int i = 0; i < numberTexCoords; i++)
                {
                    bool hasMultiTextureCoordinate = false;
                    VrmlNode *node = mTextureCoordinate.get(i);
                    if (node && (strcmp(node->nodeType()->getName(),
                                        "TextureCoordinateGenerator") == 0))
                    {
                        VrmlNodeTextureCoordinateGenerator *generator = node->as<VrmlNodeTextureCoordinateGenerator>();
                        texCoordGeneratorNames[i] = getTexCoordGeneratorName(generator);
                        if (texCoordGeneratorNames[i] != 0)
                        {
                            texCoordGeneratorName = texCoordGeneratorNames;
                            const VrmlMFFloat *parameter = generator->getField("parameter")->toMFFloat();
                            texCoordGeneratorParameter[i] = parameter->get();
                            numTexCoordGeneratorParameter[i] = parameter->size();
                            continue;
                        }
                        else
                            std::cerr << "Sorry, " << generator->getField("mode")->toSFString()->get() << " is currently not supported yet as TextureCoordinateGenerator.mode" << std::endl;
                    }
                    else if (node && (strcmp(node->nodeType()->getName(),
                                             "TextureCoordinate") == 0))
                    {
                        VrmlNodeTextureCoordinate *texCoordN = node->as<VrmlNodeTextureCoordinate>();
                        VrmlMFVec2f &texcoord = texCoordN->as<VrmlNodeTextureCoordinate>()->coordinate();
                        if (texcoord.get())
                        {
                            multi_tc[i] = &texcoord[0][0];
                            ;
                            hasMultiTextureCoordinate = true;
                        }
                    }
                    if (!hasMultiTextureCoordinate)
                        generateDefaultTextureCoordinates(multi_tc, i, coord, coordIndex);

                    multi_ntci[i] = texCoordIndex.size();
                    if (multi_ntci[i])
                        multi_tci[i] = texCoordIndex.get();
                    else
                    {
                        multi_tci[i] = coordIndex.get();
                        multi_ntci[i] = coordIndex.size();
                    }
                }
            }
        }
        else if (stdTexCoord && (strcmp(stdTexCoord->nodeType()->getName(),
                                        "TextureCoordinateGenerator") == 0))
        {
            VrmlNodeTextureCoordinateGenerator *generator = stdTexCoord->as<VrmlNodeTextureCoordinateGenerator>();
            texCoordGeneratorNames[0] = getTexCoordGeneratorName(generator);
            if (texCoordGeneratorNames[0] != 0)
            {
                texCoordGeneratorName = texCoordGeneratorNames;
                const VrmlMFFloat *parameter = generator->getField("parameter")->toMFFloat();
                texCoordGeneratorParameter[0] = parameter->get();
                numTexCoordGeneratorParameter[0] = parameter->size();
            }
            else
                std::cerr << "Sorry, " << generator->getField("mode")->toSFString()->get() << " is currently not supported yet as TextureCoordinateGenerator.mode" << std::endl;
        }
        else if (texCoord.get())
        {
            // Get texture coordinates for texture 1
            hasTextureCoordinate = true;
            VrmlMFVec2f &texcoord = texCoord.get()->as<VrmlNodeTextureCoordinate>()->coordinate();
            tc[0] = &texcoord[0][0];
            ntc[0] = texcoord.size();
        }
        if (!hasTextureCoordinate) // compute default texture coordinates
        {
            generateDefaultTextureCoordinates(tc, 0, coord, coordIndex);
            localtc = tc[0];
        }

        // Get texture coordinates for texture 2
        if (texCoord2.get())
        {
            VrmlMFVec2f &texcoord2 = texCoord2.get()->as<VrmlNodeTextureCoordinate>()->coordinate();
            tc[1] = &texcoord2[0][0];
            ntc[1] = texcoord2.size();
        }
        // Get texture coordinates for texture 3
        if (texCoord3.get())
        {
            VrmlMFVec2f &texcoord3 = texCoord3.get()->as<VrmlNodeTextureCoordinate>()->coordinate();
            tc[2] = &texcoord3[0][0];
            ntc[2] = texcoord3.size();
        }
        // Get texture coordinates for texture 4
        if (texCoord4.get())
        {
            VrmlMFVec2f &texcoord4 = texCoord4.get()->as<VrmlNodeTextureCoordinate>()->coordinate();
            tc[3] = &texcoord4[0][0];
            ntc[3] = texcoord4.size();
        }

        for (int i = 4; i < Viewer::NUM_TEXUNITS; i++)
        {
            tc[i] = NULL;
            ntc[i] = 0;
        }

        for (int i = 0; i < Viewer::NUM_TEXUNITS; i++)
        {
            // check #tc is consistent with #coords/max texCoordIndex...
            if (tci[i] && ntci[i] < coordIndex.size())
            {
                char coordString[10] = "";
                if (i > 0)
                {
                    sprintf(coordString, "%d", i);
                }
                System::the->error("IndexedFaceSet: not enough texCoordIndex%s values (there should be at least as many as coordIndex values).\n", coordString);
                System::the->error("IndexedFaceSet: #coord %d, #coordIndex %d, #texCoord %d, #texCoordIndex %d\n", nvert, coordIndex.size(), ntc[i], ntci[i]);
                tci[i] = 0;
                ntci[i] = 0;
            }
        }

        // check #colors is consistent with colorPerVtx, colorIndex...
        int cSize = -1;
        VrmlNode *colorNode = d_color.get();
        if (colorNode && (strcmp(colorNode->nodeType()->getName(), "ColorRGBA") == 0))
        {
            VrmlMFColorRGBA &c = d_color.get()->as<VrmlNodeColorRGBA>()->color();
            color = &c[0][0];
            cSize = c.size();
            optMask |= Viewer::MASK_COLOR_RGBA;
        }
        else if (colorNode)
        {
            VrmlMFColor &c = d_color.get()->as<VrmlNodeColor>()->color();
            color = &c[0][0];
            cSize = c.size();
        }
        if (cSize > -1)
        {
            nci = colorIndex.size();
            if (nci)
                ci = colorIndex.get();
            else
            {
                if (nvert == cSize)
                {
                    // color per Vertex
                    nci = coordIndex.size();
                    if (nci)
                        ci = coordIndex.get();
                }
                else
                {
                    nci = cSize;
                    ci = localci = new int[nci];
                    for (int i = 0; i < nci; i++)
                    {
                        ci[i] = i;
                    }
                }
            }
        }

        // check #normals is consistent with normalPerVtx, normalIndex...
        if (normal.get())
        {
            VrmlMFVec3f &n = normal.get()->as<VrmlNodeNormal>()->normal();
            norm = &n[0][0];
            nni = normalIndex.size();
            if (nni)
                ni = normalIndex.get();
            else
            {
                if (nvert == n.size())
                {
                    // normal per Vertex
                    nni = coordIndex.size();
                    ni = coordIndex.get();
                }
            }
        }
        else
        {
            // set up normal indices
            nni = coordIndex.size();
            localni = new int[nni];
            ni = localni;

            optMask |= Viewer::MASK_NORMAL_PER_VERTEX;

            localvn = new float[3 * nni];
            norm = localvn;
            bool normalPerVertex = (optMask & Viewer::MASK_NORMAL_PER_VERTEX)!=0;
            bool ccw = optMask & Viewer::MASK_CCW;
            viewer->computeNormals(coord.get(), coordIndex.size(),
                                   coordIndex.get(), localvn, localni,
                                   normalPerVertex ? creaseAngle.get() : 0.f, ccw);
        }

        if (nvert)
        {
            obj = viewer->insertShell(optMask,
                                      nvert, &coord[0][0], coordIndex.size(), &coordIndex[0],
                                      multi_tc, multi_ntci, multi_tci,
                                      norm, nni, ni,
                                      color, nci, ci,
                                      name(),
                                      texCoordGeneratorName,
                                      texCoordGeneratorParameter,
                                      numTexCoordGeneratorParameter);
        }
        if (multi_tc != tc)
            delete[] multi_tc;
        if (multi_ntci != ntci)
            delete[] multi_ntci;
        if (multi_tci != tci)
            delete[] multi_tci;
        if (localci)
            delete[] localci;
        if (localtc)
            delete[] localtc;
        if (localni)
            delete[] localni;
        if (localvn)
            delete[] localvn;
    }

    return obj;
}

// empty field substitutes
VrmlSFNode texCoord0;
VrmlMFInt texCoordIndex0;

Viewer::Object VrmlNodeColoredSet::insertGeometry(Viewer *viewer,
                                                  unsigned int optMask,
                                                  VrmlMFInt &coordIndex,
                                                  VrmlMFInt &colorIndex,
                                                  VrmlSFFloat &creaseAngle,
                                                  VrmlSFNode &normal,
                                                  VrmlMFInt &normalIndex,
                                                  VrmlSFNode &texCoord,
                                                  VrmlMFInt &texCoordIndex)
{
    return insertGeometry(viewer, optMask, coordIndex, colorIndex, creaseAngle,
                          normal, normalIndex, texCoord, texCoordIndex,
                          texCoord0, texCoordIndex0, texCoord0, texCoordIndex0,
                          texCoord0, texCoordIndex0);
}
