/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  Viewer.cpp
//  Abstract base class for display of VRML models
//

#include "config.h"
#include "Viewer.h"
#include "Player.h"

#include "MathUtils.h"
#include "VrmlScene.h"

#include <iostream>
using std::cerr;
using std::endl;

#include <vector>
using std::vector;
using namespace vrml;

Viewer::Viewer()
{
    setNumTextures(NUM_TEXUNITS);
}

Viewer::Viewer(VrmlScene *scene)
    : d_scene(scene)
    , d_player(0)
{
    setNumTextures(NUM_TEXUNITS);
}

//  Empty destructor for derived classes to call.

Viewer::~Viewer() {}

void Viewer::setNameModes(const char *, const char *)
{
}
void
Viewer::setNumTextures(int num)
{
    if (num > NUM_TEXUNITS)
    {
        cerr << "Viewer::setNumTextures: too large" << endl;
        numTextures = NUM_TEXUNITS;
    }
    else
    {
        numTextures = num;
    }
}

void
Viewer::getPosition(float *x, float *y, float *z)
{
    VrmlMat4 currentTransform;
    getCurrentTransform(currentTransform);

    VrmlMat4 vrmlBaseMat;
    getVrmlBaseMat(vrmlBaseMat);

    VrmlMat4 viewerMat;
    getViewerMat(viewerMat);

    VrmlVec3 pos;
    MgetTrans(pos, viewerMat);
    //fprintf(stderr, "getPosition: pos=(%f %f %f)\n", pos[0], pos[1], pos[2]);

    MM(currentTransform, vrmlBaseMat);

    VrmlMat4 inv;
    Minvert(inv, currentTransform);

    VrmlVec3 v;
    VM(v, inv, pos);

    //fprintf(stderr, "getPosition: (%f %f %f)\n", v[0], v[1], v[2]);
    *x = v[0];
    *y = v[1];
    *z = v[2];
}

void
Viewer::getPositionWC(float *x, float *y, float *z)
{
    VrmlMat4 viewerMat;
    getViewerMat(viewerMat);
    VrmlVec3 pos;
    MgetTrans(pos, viewerMat);
    *x = pos[0];
    *y = pos[1];
    *z = pos[2];
}

void
Viewer::getWC(float px, float py, float pz, float *x, float *y, float *z)
{
    VrmlVec3 pos = { px, py, pz };
    VrmlMat4 currentTransform;
    getCurrentTransform(currentTransform);
    VrmlMat4 vrmlBaseMat;
    getVrmlBaseMat(vrmlBaseMat);
    MM(currentTransform, vrmlBaseMat);
    VrmlVec3 v;
    VM(v, currentTransform, pos);
    *x = v[0];
    *y = v[1];
    *z = v[2];
}

void
Viewer::getOrientation(float *orientation)
{
    VrmlMat4 currentTransform;
    getCurrentTransform(currentTransform);
    VrmlVec3 vorig = { 0.0, 0.0, -1 };
    VrmlMat4 viewerMat;
    getViewerMat(viewerMat);
    VrmlVec3 pos = // cave direction
        {
          0.0, 1.0, 0.0
        };
    VrmlMat4 vrmlBaseMat;
    getVrmlBaseMat(vrmlBaseMat);
    MM(currentTransform, vrmlBaseMat);
    VrmlMat4 inv;
    Minvert(inv, currentTransform);
    VrmlVec3 v;
    VM(v, inv, pos);
    Vnorm(v);
    Vnorm(vorig);
    orientation[3] = acos(Vdot(v, vorig));
    Vcross(orientation, v, vorig);
}

static float boxVert[24] = {
    -1.0, -1.0, -1.0,
    -1.0, -1.0, 1.0,
    -1.0, 1.0, -1.0,
    -1.0, 1.0, 1.0,
    1.0, -1.0, -1.0,
    1.0, -1.0, 1.0,
    1.0, 1.0, -1.0,
    1.0, 1.0, 1.0
};

static int boxVertInd[30] = {
    2, 0, 1, 3, -1,
    1, 0, 4, 5, -1,
    3, 1, 5, 7, -1,
    2, 3, 7, 6, -1,
    6, 4, 0, 2, -1,
    7, 5, 4, 6, -1
};

static float boxNormals[18] = {
    -1.0, 0.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, 0.0, 1.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, -1.0,
    1.0, 0.0, 0.0
};

static float boxTexCoords[8] = {
    0.0, 1.0,
    0.0, 0.0,
    1.0, 0.0,
    1.0, 1.0
};

Viewer::Object Viewer::insertBox(float x, float y, float z)
{
    //fprintf(stderr, "insertBox(x=%f, y=%f, z=%f)\n", x,y,z);

    float vert[8 * 3];
    for (int i = 0; i < 8; i++)
    {
        vert[i * 3 + 0] = boxVert[i * 3 + 0] * x * 0.5f;
        vert[i * 3 + 1] = boxVert[i * 3 + 1] * y * 0.5f;
        vert[i * 3 + 2] = boxVert[i * 3 + 2] * z * 0.5f;
    }

    int tcind[30];
    for (int i = 0; i < 6; i++)
    {
        tcind[i * 5 + 0] = 0;
        tcind[i * 5 + 1] = 1;
        tcind[i * 5 + 2] = 2;
        tcind[i * 5 + 3] = 3;
        tcind[i * 5 + 4] = -1;
    }

    float **tcoords = new float *[numTextures + 1];
    int *ntcinds = new int[numTextures + 1];
    int **tcinds = new int *[numTextures + 1];
    for (int i = 0; i < numTextures; i++)
    {
        tcoords[i] = boxTexCoords;
        ntcinds[i] = 30;
        tcinds[i] = tcind;
    }
    tcoords[numTextures] = NULL;
    ntcinds[numTextures] = -1;
    tcinds[numTextures] = NULL;

    int nind[30];
    for (int i = 0; i < 6; i++)
    {
        nind[i * 5 + 0] = i;
        nind[i * 5 + 1] = i;
        nind[i * 5 + 2] = i;
        nind[i * 5 + 3] = i;
        nind[i * 5 + 4] = -1;
    }
    int mask = Viewer::MASK_CONVEX | Viewer::MASK_NORMAL_PER_VERTEX;
    Object obj = insertShell(mask,
                             8, vert, 30, boxVertInd,
                             tcoords, ntcinds, tcinds,
                             boxNormals, 30, nind,
                             NULL, 0, NULL,
                             "Box");

    delete[] tcoords;
    delete[] ntcinds;
    delete[] tcinds;

    return obj;
}

//  Build a cylinder object. It might be smarter to do just one, and reference
//  it with scaling (but the world creator could just as easily do that with
//  DEF/USE ...).

Viewer::Object Viewer::insertCylinder(float h, float r, bool bottom, bool side, bool top)
{
    const int segments = 20;
    //fprintf(stderr, "insertCylinder(h=%f, r=%f, bottom=%d, side=%d, top=%d)\n", h, r, int(bottom), int(side), int(top));

    int nvert = segments * 2;
    float *vert = new float[nvert * 3];
    float *normal = new float[(segments + 2) * 3];
    float *tcoord = new float[(segments + 1) * 4 * 2 + segments * 2];

    for (int i = 0; i < segments; i++)
    {
        float angle = (float)(i * M_PI * 2.0f / segments);
        vert[i * 6 + 0] = vert[i * 6 + 3] = r * cos(angle);
        vert[i * 6 + 2] = vert[i * 6 + 5] = r * sin(angle);
        vert[i * 6 + 1] = 0.5f * h;
        vert[i * 6 + 4] = -vert[i * 6 + 1];

        // tex coords for side
        tcoord[i * 4 + 3] = 0.0;
        tcoord[i * 4 + 1] = 1.0;
        tcoord[i * 4 + 0] = tcoord[i * 4 + 2] = 1.0f - float(i) / float(segments);

        // tex coords for top and bottom
        tcoord[(segments + 1) * 4 + i * 2 + 0] = (float)sin(angle + M_PI) / 2.0f + 0.5f;
        tcoord[(segments + 1) * 4 + i * 2 + 1] = (float)cos(angle + M_PI) / 2.0f + 0.5f;
        tcoord[(segments + 1) * 8 + i * 2 + 0] = (float)cos(angle - M_PI_2) / 2.0f + 0.5f;
        tcoord[(segments + 1) * 8 + i * 2 + 1] = (float)sin(angle - M_PI_2) / 2.0f + 0.5f;

        // normals
        normal[i * 3 + 0] = -cos(angle);
        normal[i * 3 + 1] = 0.0;
        normal[i * 3 + 2] = -sin(angle);
    }
    // normals for top and bottom
    normal[segments * 3 + 0] = normal[segments * 3 + 2] = normal[segments * 3 + 3] = normal[segments * 3 + 5] = 0.0;
    normal[segments * 3 + 1] = -1.0;
    normal[segments * 3 + 4] = 1.0;

    // texture coords at seam
    tcoord[segments * 4 + 3] = 0.0;
    tcoord[segments * 4 + 1] = 1.0;
    tcoord[segments * 4 + 0] = tcoord[segments * 4 + 2] = 0.0;

    // vertex indices
    int numind = 0;
    if (side)
        numind += 5 * segments;
    if (top)
        numind += segments + 1;
    if (bottom)
        numind += segments + 1;
    int *vind = new int[numind];
    int *nind = new int[numind];
    int *tcind = new int[numind];
    int off = 0;
    if (side)
    {
        for (int i = 0; i < segments; i++)
        {
            tcind[off + i * 5 + 0] = vind[off + i * 5 + 0] = (i % segments) * 2;
            tcind[off + i * 5 + 1] = vind[off + i * 5 + 1] = (i % segments) * 2 + 1;
            vind[off + i * 5 + 2] = ((i + 1) % segments) * 2 + 1;
            vind[off + i * 5 + 3] = ((i + 1) % segments) * 2;
            tcind[off + i * 5 + 2] = (i + 1) * 2 + 1;
            tcind[off + i * 5 + 3] = (i + 1) * 2;

            nind[off + i * 5 + 0] = i % segments;
            nind[off + i * 5 + 1] = i % segments;
            nind[off + i * 5 + 2] = (i + 1) % segments;
            nind[off + i * 5 + 3] = (i + 1) % segments;

            tcind[off + i * 5 + 4] = nind[off + i * 5 + 4] = vind[off + i * 5 + 4] = -1;
        }
        off += segments * 5;
    }
    if (top)
    {
        for (int i = 0; i < segments; i++)
        {
            vind[off + i] = (i % segments) * 2;
            nind[off + i] = segments;
            tcind[off + i] = (segments + 1) * 2 + i % segments;
        }
        tcind[off + segments] = nind[off + segments] = vind[off + segments] = -1;
        off += segments + 1;
    }
    if (bottom)
    {
        for (int i = 0; i < segments; i++)
        {
            vind[off + i] = ((segments - i) % segments) * 2 + 1;
            nind[off + i] = segments + 1;
            tcind[off + i] = (segments + 1) * 4 + (segments - i) % segments;
        }
        tcind[off + segments] = nind[off + segments] = vind[off + segments] = -1;
        off += segments + 1;
    }

    float **tcoords = new float *[numTextures + 1];
    int **tcinds = new int *[numTextures + 1];
    int *ntcinds = new int[numTextures + 1];
    for (int i = 0; i < numTextures; i++)
    {
        tcoords[i] = tcoord;
        tcinds[i] = tcind;
        ntcinds[i] = numind;
    }
    tcoords[numTextures] = NULL;
    tcinds[numTextures] = NULL;
    ntcinds[numTextures] = 0;

    int mask = Viewer::MASK_NORMAL_PER_VERTEX | MASK_CONVEX;
    Object obj = insertShell(mask,
                             nvert, vert, numind, vind,
                             tcoords, ntcinds, tcinds,
                             normal, numind, nind,
                             NULL, 0, NULL,
                             "Cylinder");

    delete[] tcoords;
    delete[] tcinds;
    delete[] ntcinds;
    delete[] tcoord;
    delete[] tcind;
    delete[] normal;
    delete[] nind;
    delete[] vert;
    delete[] vind;

    return obj;
}

Viewer::Object Viewer::insertSphere(float r)
{
    const int segments = 20; // has to be even

    int nvert = (segments + 1) * (segments / 2 + 1);
    float *vert = new float[nvert * 3];
    float *normal = new float[nvert * 3];
    float *tcoord = new float[nvert * 2];

    int l = segments / 2 + 1;
    for (int i = 0; i < segments + 1; i++)
    {
        float theta = (float)(2.0f * M_PI * i / segments);
        for (int j = 0; j <= segments / 2; j++)
        {
            float phi = (float)(2.0f * M_PI * j / segments);
            normal[(i * l + j) * 3 + 0] = (float)(-cos(theta) * cos(M_PI / 2.0 - phi));
            normal[(i * l + j) * 3 + 1] = (float)-sin(M_PI / 2.0 - phi);
            normal[(i * l + j) * 3 + 2] = (float)(-sin(theta) * cos(M_PI / 2.0 - phi));

            tcoord[(i * l + j) * 2 + 0] = 1.0f - (float)((theta - M_PI_2) / (2.0 * M_PI));
            tcoord[(i * l + j) * 2 + 1] = (float)((phi) / M_PI);
        }
    }
    for (int i = 0; i < nvert * 3; i++)
    {
        vert[i] = r * normal[i];
    }

    int numind = segments * segments / 2 * 5;
    int *vind = new int[numind];
    int *nind = new int[numind];
    int *tcind = new int[numind];
    for (int i = 0; i < segments; i++)
    {
        for (int j = 0; j < segments / 2; j++)
        {
            tcind[(i * segments / 2 + j) * 5 + 0] = i * l + j;
            tcind[(i * segments / 2 + j) * 5 + 1] = i * l + j + 1;
            tcind[(i * segments / 2 + j) * 5 + 2] = (i + 1) * l + j + 1;
            tcind[(i * segments / 2 + j) * 5 + 3] = (i + 1) * l + j;
            tcind[(i * segments / 2 + j) * 5 + 4] = -1;

            vind[(i * segments / 2 + j) * 5 + 0] = i * l + j;
            vind[(i * segments / 2 + j) * 5 + 1] = i * l + j + 1;
            vind[(i * segments / 2 + j) * 5 + 2] = ((i + 1) % segments) * l + j + 1;
            vind[(i * segments / 2 + j) * 5 + 3] = ((i + 1) % segments) * l + j;
            vind[(i * segments / 2 + j) * 5 + 4] = -1;
        }
    }

    for (int i = 0; i < numind; i++)
    {
        nind[i] = vind[i];
    }

    float **tcoords = new float *[numTextures + 1];
    int **tcinds = new int *[numTextures + 1];
    int *ntcinds = new int[numTextures + 1];
    for (int i = 0; i < numTextures; i++)
    {
        tcoords[i] = tcoord;
        tcinds[i] = tcind;
        ntcinds[i] = numind;
    }
    tcoords[numTextures] = NULL;
    tcinds[numTextures] = NULL;
    ntcinds[numTextures] = 0;

    int mask = Viewer::MASK_NORMAL_PER_VERTEX | MASK_CONVEX | MASK_SOLID | MASK_CCW;
    Object obj = insertShell(mask,
                             nvert, vert, numind, vind,
                             tcoords, ntcinds, tcinds,
                             normal, numind, nind,
                             NULL, 0, NULL,
                             "Sphere");

    delete[] tcoords;
    delete[] tcinds;
    delete[] ntcinds;
    delete[] tcoord;
    delete[] tcind;
    delete[] normal;
    delete[] nind;
    delete[] vert;
    delete[] vind;

    return obj;
}

Viewer::Object Viewer::insertCone(float h, float r, bool bottom, bool side)
{
    const int segments = 20;
    //fprintf(stderr, "insertCone(h=%f, r=%f, bottom=%d, side=%d)\n", h, r, int(bottom), int(side));

    int nvert = segments + 1;
    float *vert = new float[nvert * 3];
    float *normal = new float[(segments * 2 + 1) * 3];
    float *tcoord = new float[segments * 4 + segments * 2];

    for (int i = 0; i < segments; i++)
    {
        float angle = (float)(i * M_PI * 2.0f / segments);
        vert[i * 3 + 0] = r * cos(angle);
        vert[i * 3 + 1] = -0.5f * h;
        vert[i * 3 + 2] = r * sin(angle);

        // tex coords for side
        tcoord[i * 4 + 1] = 0.0;
        tcoord[i * 4 + 3] = 1.0;
        tcoord[i * 4 + 0] = tcoord[i * 4 + 2] = 1.0f - 1.0f * (float)i / (float)segments;

        // tex coords for bottom
        tcoord[segments * 4 + i * 2 + 0] = (float)sin(angle + M_PI) / 2.0f + 0.5f;
        tcoord[segments * 4 + i * 2 + 1] = (float)cos(angle + M_PI) / 2.0f + 0.5f;

        // normals
        float l = sqrt(r * r + h * h);
        normal[i * 3 + 0] = -cos(angle) * h / l;
        normal[i * 3 + 1] = -r / l;
        normal[i * 3 + 2] = -sin(angle) * h / l;

        normal[(segments + i) * 3 + 0] = (float)-cos(angle + M_PI / segments) * h / l;
        normal[(segments + i) * 3 + 1] = -r / l;
        normal[(segments + i) * 3 + 2] = (float)-sin(angle + M_PI / segments) * h / l;
    }
    // vertex at top
    vert[segments * 3 + 0] = vert[segments * 3 + 2] = 0.0;
    vert[segments * 3 + 1] = h * 0.5f;

    // normal for bottom
    normal[segments * 6 + 0] = normal[segments * 6 + 2] = 0.0;
    normal[segments * 6 + 1] = 1.0;

    // texture coords at seam
    tcoord[segments * 4 + 1] = 0.0;
    tcoord[segments * 4 + 3] = 1.0;
    tcoord[segments * 4 + 0] = tcoord[segments * 4 + 2] = 0.0;

    // vertex indices
    int numind = 0;
    if (side)
        numind += 5 * segments;
    if (bottom)
        numind += segments + 1;
    int *vind = new int[numind];
    int *nind = new int[numind];
    int *tcind = new int[numind];
    int off = 0;
    if (side)
    {
        for (int i = 0; i < segments; i++)
        {
            tcind[off + i * 5 + 0] = i * 2;
            tcind[off + i * 5 + 1] = (i + 1) * 2;
            tcind[off + i * 5 + 2] = (i + 1) * 2 + 1;
            tcind[off + i * 5 + 3] = i * 2 + 1;

            vind[off + i * 5 + 0] = i % segments;
            vind[off + i * 5 + 1] = (i + 1) % segments;
            vind[off + i * 5 + 2] = segments;
            vind[off + i * 5 + 3] = segments;

            nind[off + i * 5 + 0] = i % segments;
            nind[off + i * 5 + 1] = (i + 1) % segments;
            nind[off + i * 5 + 2] = i % segments + segments;
            nind[off + i * 5 + 3] = i % segments + segments;

            tcind[off + i * 5 + 4] = nind[off + i * 5 + 4] = vind[off + i * 5 + 4] = -1;
        }
        off += segments * 5;
    }
    if (bottom)
    {
        for (int i = 0; i < segments; i++)
        {
            vind[off + i] = (segments - i) % segments;
            nind[off + i] = 2 * segments;
            tcind[off + i] = segments * 2 + (segments - i) % segments;
        }
        tcind[off + segments] = nind[off + segments] = vind[off + segments] = -1;
        off += segments + 1;
    }

    float **tcoords = new float *[numTextures + 1];
    int **tcinds = new int *[numTextures + 1];
    int *ntcinds = new int[numTextures + 1];
    for (int i = 0; i < numTextures; i++)
    {
        tcoords[i] = tcoord;
        tcinds[i] = tcind;
        ntcinds[i] = numind;
    }
    tcoords[numTextures] = NULL;
    tcinds[numTextures] = NULL;
    ntcinds[numTextures] = 0;

    int mask = Viewer::MASK_NORMAL_PER_VERTEX | MASK_CONVEX;
    Object obj = insertShell(mask,
                             nvert, vert, numind, vind,
                             tcoords, ntcinds, tcinds,
                             normal, numind, nind,
                             NULL, 0, NULL,
                             "Cone");

    delete[] tcoords;
    delete[] tcinds;
    delete[] ntcinds;
    delete[] tcoord;
    delete[] tcind;
    delete[] normal;
    delete[] nind;
    delete[] vert;
    delete[] vind;

    return obj;
}

Viewer::Object Viewer::insertExtrusion(unsigned int mask,
                                       int nOrientation,
                                       float *orientation,
                                       int nScale,
                                       float *scale,
                                       int nCrossSection,
                                       float *crossSection,
                                       int nSpine,
                                       float *spine,
                                       float creaseAngle)
{
    int nvert = nCrossSection * nSpine;
    float *vert = new float[nvert * 3];

    float *tcoord = new float[nvert * 2];

    // Xscp, Yscp, Zscp- columns of xform matrix to align cross section
    // with spine segments.
    float Xscp[3] = { 1.0, 0.0, 0.0 };
    float Yscp[3] = { 0.0, 1.0, 0.0 };
    float Zscp[3] = { 0.0, 0.0, 1.0 };
    float lastZ[3];

    // Is the spine a closed curve (last pt == first pt)?
    bool spineClosed = (FPZERO(spine[3 * (nSpine - 1) + 0] - spine[0]) && FPZERO(spine[3 * (nSpine - 1) + 1] - spine[1]) && FPZERO(spine[3 * (nSpine - 1) + 2] - spine[2]));

    bool crossSectionClosed = (FPZERO(crossSection[2 * (nCrossSection - 1) + 0] - crossSection[0]) && FPZERO(crossSection[2 * (nCrossSection - 1) + 1] - crossSection[1]));

    /*
   fprintf(stderr, "insertExtrusion: spine closed=%d, cross section closed=%d\n",
      (int)spineClosed, (int)crossSectionClosed);
      */

    // Is the spine a straight line?
    bool spineStraight = true;
    for (int i = 1; i < nSpine - 1; ++i)
    {
        float v1[3], v2[3];
        v1[0] = spine[3 * (i - 1) + 0] - spine[3 * (i) + 0];
        v1[1] = spine[3 * (i - 1) + 1] - spine[3 * (i) + 1];
        v1[2] = spine[3 * (i - 1) + 2] - spine[3 * (i) + 2];
        v2[0] = spine[3 * (i + 1) + 0] - spine[3 * (i) + 0];
        v2[1] = spine[3 * (i + 1) + 1] - spine[3 * (i) + 1];
        v2[2] = spine[3 * (i + 1) + 2] - spine[3 * (i) + 2];
        Vcross(v1, v2, v1);
        if (Vlength(v1) != 0.0)
        {
            spineStraight = false;
            Vnorm(v1);
            Vset(lastZ, v1);
            break;
        }
    }

    // If the spine is a straight line, compute a constant SCP xform
    if (spineStraight)
    {
        float V1[3] = { 0.0, 1.0, 0.0 },
              V2[3], V3[3];
        V2[0] = spine[3 * (nSpine - 1) + 0] - spine[0];
        V2[1] = spine[3 * (nSpine - 1) + 1] - spine[1];
        V2[2] = spine[3 * (nSpine - 1) + 2] - spine[2];
        Vcross(V3, V2, V1);
        double len = Vlength(V3);
        if (len != 0.0) // Not aligned with Y axis
        {
            Vscale(V3, 1.0f / (float)len);

            float orient[4]; // Axis/angle
            Vset(orient, V3);
            orient[3] = acos(Vdot(V1, V2));
            double scp[16]; // xform matrix
            Mrotation(scp, orient);
            for (int k = 0; k < 3; ++k)
            {
                Xscp[k] = (float)scp[0 * 4 + k];
                Yscp[k] = (float)scp[1 * 4 + k];
                Zscp[k] = (float)scp[2 * 4 + k];
            }
        }
    }

    // Orientation matrix
    double om[16];
    if (nOrientation == 1 && !FPZERO(orientation[3]))
        Mrotation(om, orientation);

    // Compute coordinates, texture coordinates:
    for (int i = 0, ci = 0; i < nSpine; ++i, ci += nCrossSection)
    {

        // Scale cross section
        for (int j = 0; j < nCrossSection; ++j)
        {
            vert[3 * (ci + j) + 0] = scale[0] * crossSection[2 * j];
            vert[3 * (ci + j) + 1] = 0.0;
            vert[3 * (ci + j) + 2] = scale[1] * crossSection[2 * j + 1];
        }

        // Compute Spine-aligned Cross-section Plane (SCP)
        if (!spineStraight)
        {
            float S1[3], S2[3]; // Spine vectors [i,i-1] and [i,i+1]
            int yi1, yi2, si1, s1i2, s2i2;

            if (spineClosed && (i == 0 || i == nSpine - 1))
            {
                yi1 = 3 * (nSpine - 2);
                yi2 = 3;
                si1 = 0;
                s1i2 = 3 * (nSpine - 2);
                s2i2 = 3;
            }
            else if (i == 0)
            {
                yi1 = 0;
                yi2 = 3;
                si1 = 3;
                s1i2 = 0;
                s2i2 = 6;
            }
            else if (i == nSpine - 1)
            {
                yi1 = 3 * (nSpine - 2);
                yi2 = 3 * (nSpine - 1);
                si1 = 3 * (nSpine - 2);
                s1i2 = 3 * (nSpine - 3);
                s2i2 = 3 * (nSpine - 1);
            }
            else
            {
                yi1 = 3 * (i - 1);
                yi2 = 3 * (i + 1);
                si1 = 3 * i;
                s1i2 = 3 * (i - 1);
                s2i2 = 3 * (i + 1);
            }

            Vdiff(Yscp, &spine[yi2], &spine[yi1]);
            Vdiff(S1, &spine[s1i2], &spine[si1]);
            Vdiff(S2, &spine[s2i2], &spine[si1]);

            Vnorm(Yscp);
            Vset(lastZ, Zscp); // Save last Zscp
            Vcross(Zscp, S2, S1);

            float VlenZ = (float)Vlength(Zscp);
            if (VlenZ == 0.0)
                Vset(Zscp, lastZ);
            else
                Vscale(Zscp, 1.0f / VlenZ);

            if ((i > 0) && (Vdot(Zscp, lastZ) < 0.0))
                Vscale(Zscp, -1.0);

            Vcross(Xscp, Yscp, Zscp);
        }

        // Rotate cross section into SCP
        for (int j = 0; j < nCrossSection; ++j)
        {
            float cx, cy, cz;
            cx = vert[3 * (ci + j) + 0] * Xscp[0] + vert[3 * (ci + j) + 1] * Yscp[0] + vert[3 * (ci + j) + 2] * Zscp[0];
            cy = vert[3 * (ci + j) + 0] * Xscp[1] + vert[3 * (ci + j) + 1] * Yscp[1] + vert[3 * (ci + j) + 2] * Zscp[1];
            cz = vert[3 * (ci + j) + 0] * Xscp[2] + vert[3 * (ci + j) + 1] * Yscp[2] + vert[3 * (ci + j) + 2] * Zscp[2];
            vert[3 * (ci + j) + 0] = cx;
            vert[3 * (ci + j) + 1] = cy;
            vert[3 * (ci + j) + 2] = cz;
        }

        // Apply orientation
        if (orientation != NULL && !FPZERO(orientation[3]))
        {
            if (nOrientation > 1)
                Mrotation(om, orientation);

            for (int j = 0; j < nCrossSection; ++j)
            {
                float cx, cy, cz;
                cx = (float)(vert[3 * (ci + j) + 0] * om[0 * 4 + 0] + vert[3 * (ci + j) + 1] * om[1 * 4 + 0] + vert[3 * (ci + j) + 2] * om[2 * 4 + 0]);
                cy = (float)(vert[3 * (ci + j) + 0] * om[0 * 4 + 1] + vert[3 * (ci + j) + 1] * om[1 * 4 + 1] + vert[3 * (ci + j) + 2] * om[2 * 4 + 1]);
                cz = (float)(vert[3 * (ci + j) + 0] * om[0 * 4 + 2] + vert[3 * (ci + j) + 1] * om[1 * 4 + 2] + vert[3 * (ci + j) + 2] * om[2 * 4 + 2]);
                vert[3 * (ci + j) + 0] = cx;
                vert[3 * (ci + j) + 1] = cy;
                vert[3 * (ci + j) + 2] = cz;
            }
        }

        // Translate cross section
        for (int j = 0; j < nCrossSection; ++j)
        {
            vert[3 * (ci + j) + 0] += spine[3 * i + 0];
            vert[3 * (ci + j) + 1] += spine[3 * i + 1];
            vert[3 * (ci + j) + 2] += spine[3 * i + 2];

            // Texture coords
            tcoord[2 * (ci + j) + 0] = ((float)j) / (nCrossSection - 1);
            tcoord[2 * (ci + j) + 1] = 1.0f - ((float)i) / (nSpine - 1);
        }

        if (nScale > 1)
            scale += 2;
        if (nOrientation > 1)
            orientation += 4;
    }

    int numind = (nCrossSection - 1) * (nSpine - 1) * 5;
    if (mask & MASK_BOTTOM)
    {
        numind += nCrossSection;
        if (!crossSectionClosed)
            numind++;
    }
    if (mask & MASK_TOP)
    {
        numind += nCrossSection;
        if (!crossSectionClosed)
            numind++;
    }
    int *faces = new int[numind];

    // And compute face indices:
    int polyIndex = 0;
    for (int i = 0, ci = 0; i < nSpine - 1; ++i, ci += nCrossSection)
    {
        for (int j = 0; j < nCrossSection - 1; ++j)
        {
            faces[polyIndex + 0] = ci + j;
            if (crossSectionClosed && j == nCrossSection - 2)
            {
                faces[polyIndex + 1] = ci;
                faces[polyIndex + 2] = ci + nCrossSection;
            }
            else
            {
                faces[polyIndex + 1] = ci + j + 1;
                faces[polyIndex + 2] = ci + j + 1 + nCrossSection;
            }
            faces[polyIndex + 3] = ci + j + nCrossSection;
            faces[polyIndex + 4] = -1;
            polyIndex += 5;
        }
    }

    if (mask & MASK_TOP)
    {
        for (int j = 0; j < nCrossSection - 1; j++)
        {
            faces[polyIndex++] = (nSpine - 1) * nCrossSection + j;
        }
        if (!crossSectionClosed)
        {
            faces[polyIndex++] = nSpine * nCrossSection - 1;
        }
        faces[polyIndex++] = -1;
    }

    if (mask & MASK_BOTTOM)
    {
        if (!crossSectionClosed)
            faces[polyIndex++] = nCrossSection - 1;
        for (int j = nCrossSection - 2; j >= 0; j--)
        {
            faces[polyIndex++] = j;
        }
        faces[polyIndex++] = -1;
    }

    float **tcoords = new float *[numTextures + 1];
    int **tcinds = new int *[numTextures + 1];
    int *ntcinds = new int[numTextures + 1];
    for (int i = 0; i < numTextures; i++)
    {
        tcoords[i] = tcoord;
        tcinds[i] = faces;
        ntcinds[i] = numind;
    }
    tcoords[numTextures] = NULL;
    tcinds[numTextures] = NULL;
    ntcinds[numTextures] = 0;

    /*
   fprintf(stderr, "insertExtrusion(nSpine=%d, nCrossSection=%d, polyIndex=%d\n",
      nSpine, nCrossSection, polyIndex);
   fprintf(stderr, "insertExtrusion(mask=0x%x, nvert=%d, numind=%d, creaseAngle=%f\n",
      mask, nvert, numind, creaseAngle);
      */

    float *localvn = new float[3 * numind];
    int *localni = new int[numind];
    computeNormals(vert, numind, faces, localvn, localni, creaseAngle, true);

    mask |= Viewer::MASK_NORMAL_PER_VERTEX;
    Object obj = insertShell(mask,
                             nvert, vert, numind, faces,
                             tcoords, ntcinds, tcinds,
                             localvn, numind, localni, //normal, numind, nind,
                             NULL, 0, NULL,
                             "Extrusion");

    delete[] tcoords;
    delete[] tcinds;
    delete[] ntcinds;
    delete[] tcoord;
    delete[] vert;
    delete[] faces;
    delete[] localni;
    delete[] localvn;

    return obj;
}

Viewer::Object Viewer::insertElevationGrid(unsigned int mask,
                                           int nx,
                                           int nz,
                                           float *height,
                                           float dx,
                                           float dz,
                                           float *tc,
                                           float *normals,
                                           float *colors,
                                           float creaseAngle)
{
    float *localtc = NULL;
    int ncoord = nx * nz;
    float *coord = new float[ncoord * 3];
    float *localcolors = NULL;
    int colorComponents = 3;
    if (mask & MASK_COLOR_RGBA)
        colorComponents = 4;
    if (colors)
    {
        if (mask & MASK_COLOR_PER_VERTEX)
            localcolors = colors;
        else
            localcolors = new float[colorComponents * 2 * (nz - 1) * (nx - 1)];
    }
    int ind = 0, i = 0;
    for (int z = 0; z < nz; z++)
    {
        float zz = dz * z;
        for (int x = 0; x < nx; x++)
        {
            float xx = dx * x;
            coord[ind++] = xx;
            coord[ind++] = height[i++];
            coord[ind++] = zz;
        }
    }

    int ncind = (nx - 1) * (nz - 1) * 2 * 4;
    //int ncind = (nx-1)*(nz-1)*5;
    int *cind = new int[ncind];
    ind = 0;
    for (int z = 0; z < nz - 1; z++)
    {
        for (int x = 0; x < nx - 1; x++)
        {
            if ((x + z) % 2)
            {
                cind[ind++] = (z + 1) * nx + x;
                cind[ind++] = z * nx + x;
                cind[ind++] = z * nx + x + 1;
                cind[ind++] = -1;
                cind[ind++] = (z + 1) * nx + x + 1;
                cind[ind++] = (z + 1) * nx + x;
                cind[ind++] = z * nx + x + 1;
                cind[ind++] = -1;
            }
            else
            {
                cind[ind++] = (z + 1) * nx + x;
                cind[ind++] = z * nx + x;
                cind[ind++] = (z + 1) * nx + x + 1;
                cind[ind++] = -1;
                cind[ind++] = z * nx + x + 1;
                cind[ind++] = (z + 1) * nx + x + 1;
                cind[ind++] = z * nx + x;
                cind[ind++] = -1;
            }
        }
    }

    int numColors = 0;
    int *localColorIndex = NULL;
    if (colors && ((mask & MASK_COLOR_PER_VERTEX) == 0))
    {
        localColorIndex = new int[2 * (nx - 1) * (nz - 1)];
        for (int z = 0; z < nz - 1; z++)
        {
            for (int x = 0; x < nx - 1; x++)
            {
                for (int j = 0; j < colorComponents; j++)
                {
                    localcolors[numColors * 2 * colorComponents + j] = colors[(z * (nx - 1) + x) * colorComponents + j];
                    localcolors[numColors * 2 * colorComponents + j + colorComponents] = colors[(z * (nx - 1) + x) * colorComponents + j];
                }
                localColorIndex[numColors * 2] = numColors * 2;
                localColorIndex[numColors * 2 + 1] = numColors * 2 + 1;
                numColors++;
            }
        }
    }

    if (tc == NULL)
    {
        localtc = new float[ncoord * 2];
        tc = localtc;
        ind = 0;
        for (int x = 0; x < nx; x++)
        {
            for (int z = 0; z < nz; z++)
            {
                localtc[ind++] = float(x) / float(nx);
                localtc[ind++] = float(z) / float(nz);
            }
        }
    }
    float **tcoords = new float *[numTextures + 1];
    int *ntcinds = new int[numTextures + 1];
    int **tcinds = new int *[numTextures + 1];
    for (int i = 0; i < numTextures; i++)
    {
        tcoords[i] = tc;
        ntcinds[i] = ncind;
        tcinds[i] = cind;
    }
    tcoords[numTextures] = NULL;
    ntcinds[numTextures] = -1;
    tcinds[numTextures] = NULL;

    float *localnormals = NULL;
    int *localnind = NULL;
    int *nind = cind;
    if (!normals)
    {
        localnormals = new float[ncind * 3];
        normals = localnormals;
        localnind = new int[ncind];
        nind = localnind;
        computeNormals(coord, ncind, cind, localnormals, nind, creaseAngle, true);
    }

    unsigned int elevationGridMask = mask | MASK_CONVEX;
    if (mask & MASK_CCW)
        elevationGridMask = mask & (~MASK_CCW);
    else
        elevationGridMask = mask | MASK_CCW;

    (void)colors;
    Object obj = insertShell(elevationGridMask,
                             ncoord, coord, ncind, cind,
                             tcoords, ntcinds, tcinds,
                             normals, ncind, nind,
                             localcolors,
                             (mask & MASK_COLOR_PER_VERTEX) ? (colors ? ncind : 0) : (colors ? numColors * 2 : 0),
                             (mask & MASK_COLOR_PER_VERTEX) ? (colors ? cind : NULL) : (colors ? localColorIndex : NULL),
                             "ElevationGrid");

    delete[] localnormals;
    delete[] cind;
    delete[] localnind;
    delete[] localtc;
    delete[] coord;
    if (!(mask & MASK_COLOR_PER_VERTEX))
    {
        delete[] localcolors;
        delete[] localColorIndex;
    }
    return obj;
}

void Viewer::computeNormals(const float *coord,
                            int numInd, const int *coordIndex,
                            float *normal, int *nind,
                            float creaseAngle, bool /*ccw*/)
{
    if (coord == NULL)
        return;
    // determine number of faces and their start indices
    vector<int> faceStart;
    faceStart.push_back(0);
    int maxInd = -1;
    for (int i = 0; i < numInd; i++)
    {
        if (coordIndex[i] > maxInd)
            maxInd = coordIndex[i];
        if (coordIndex[i] == -1)
        {
            faceStart.push_back(i + 1);
        }
    }
    if (numInd <= 0 || coordIndex[numInd - 1] != -1)
    {
        faceStart.push_back(numInd);
    }
    int nFaces = (int)faceStart.size() - 1;

    // compute normal for each face
    float *faceNormals = new float[3 * nFaces];
    for (int f = 0; f < nFaces; f++)
    {
        VrmlVec3 n = { 0.0, 0.0, 0.0 };
        VrmlVec3 p;
        Vset(p, &coord[coordIndex[faceStart[f]] * 3]);
        VrmlVec3 p1;
        Vset(p1, &coord[coordIndex[faceStart[f] + 1] * 3]);
        Vsub(p1, p);
        for (int i = faceStart[f] + 2; i < faceStart[f + 1] - 1; i++)
        {
            VrmlVec3 p2;
            Vset(p2, &coord[coordIndex[i] * 3]);
            Vsub(p2, p);
            VrmlVec3 cross;
            Vcross(cross, p1, p2);
            Vadd(n, cross);
        }
        Vnorm(n);
        /* if(!ccw)
		  Vscale(n,-1);*/
        memcpy(&faceNormals[3 * f], n, sizeof(float) * 3);
    }

    // for each vertex determine neighbouring faces
    vector<int> *neighbourFaces = new vector<int>[maxInd + 1];
    int face = 0;
    for (int i = 0; i < numInd; i++)
    {
        if (coordIndex[i] == -1)
        {
            face++;
            continue;
        }
        neighbourFaces[coordIndex[i]].push_back(face);
    }

    // compute per vertex normals
    float cosCrease = cos(creaseAngle);
    face = 0;
    for (int i = 0; i < numInd; i++)
    {
        int ind = coordIndex[i];
        if (ind == -1)
        {
            face++;
            continue;
        }
        VrmlVec3 n;
        Vset(n, &faceNormals[3 * face]);
        for (unsigned int f = 0; f < neighbourFaces[ind].size(); f++)
        {
            int otherFace = neighbourFaces[ind][f];
            if (otherFace == face)
                continue;
            float cosAngle = Vdot(&faceNormals[3 * face], &faceNormals[3 * otherFace]);

            if (cosAngle >= cosCrease)
            {
                Vadd(n, &faceNormals[3 * otherFace]);
            }
        }

        Vnorm(n);
        Vset(&normal[3 * i], n);
    }
    delete[] faceNormals;
    delete[] neighbourFaces;

    // set up normal indices
    for (int i = 0; i < numInd; i++)
    {
        if (coordIndex[i] < 0)
            nind[i] = -1;
        else
            nind[i] = i;
    }
}

// Scale an image to make sizes powers of two. This puts the data back
// into the memory pointed to by pixels, so there better be enough.

void Viewer::scaleTexture(int w, int h,
                          int newW, int newH,
                          int nc,
                          unsigned char *pixels)
{
    //fprintf(stderr,"\nViewerOsg::scaleTexture: Scale %d components from %d x %d to %d x %d\n\n", nc,w,h,newW,newH);
    // correct ugly hack by Uwe, cf. vrml97/vrml/tifread.c
    // 40 means it is not transparent but comes in 4 components
    if (nc == 40)
        nc = 4;

    ////////////////////////////////////////////////////////////
    // Bi-Linar interpolation between old and new image
    // Line weights calculated in loop, column weights
    // pre-calculated.
    ////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////
    // Pre-calculate interpolation index and ratios for columns
    // Row Interpolation is between Pixels srcColIdx[i] and
    // srcColIdx[i+1] with factors (1-srcColFct[i]) and srcColFct[i]
    int *colIdx = new int[newW];
    float *colFct0 = new float[newW];
    float *colFct1 = new float[newW];

    float srcW = (float)w;
    float dstW = (float)newW;
    for (int col = 0; col < newW; col++)
    {
        // float "index" of row in src image
        float srcColFl = (col + 0.5f) * srcW / dstW - 0.5f;
        register int idx = (int)srcColFl;
        colIdx[col] = idx;

        colFct1[col] = srcColFl - idx;
        colFct0[col] = 1.0f - colFct1[col];
    }

    // Intermediate array for Pixels -
    unsigned char *newpix = new unsigned char[nc * newW * newH];

    /// Line-by-line calculations of result image
    float srcH = (float)h;
    float dstH = (float)newH;
    for (int row = 0; row < newH; row++)
    {
        // calculate per-line invariants
        // 'line index' in src img
        double srcRowFl = (row + 0.5) * srcH / dstH - 0.5;

        int rowIdx = (int)srcRowFl; // line index for interpolation

        float rowFct1 = (float)srcRowFl - rowIdx; // interpolation factors for both lines
        float rowFct0 = 1.0f - rowFct1;

        unsigned char *srcRow0 = pixels + nc * w * rowIdx;
        if (rowIdx + 1 >= h)
            rowIdx = h - 2;
        unsigned char *srcRow1 = pixels + nc * w * (rowIdx + 1);
        unsigned char *dstLine = newpix + nc * newW * row;

        for (int col = 0; col < newW; col++)
        {
            // calculate per-pixel invariants
            register unsigned char *pix00 = srcRow0 + nc * colIdx[col];
            register unsigned char *pix01 = pix00 + (colIdx[col] < w - 1 ? nc : 0);
            register unsigned char *pix10 = srcRow1 + nc * colIdx[col];
            register unsigned char *pix11 = pix10 + (colIdx[col] < w - 1 ? nc : 0);

            // loop for all colors
            for (int i = 0; i < nc; i++)
            {
                *dstLine = (unsigned char)(*pix00 * rowFct0 * colFct0[col]
                                           + *pix01 * rowFct0 * colFct1[col]
                                           + *pix10 * rowFct1 * colFct0[col]
                                           + *pix11 * rowFct1 * colFct1[col]);

                ++dstLine;
                ++pix00;
                ++pix01;
                ++pix10;
                ++pix11;
            }
        }
    }

    memcpy(pixels, newpix, nc * newW * newH);

    delete[] newpix;

    // free intermediate storage
    delete[] colIdx;
    delete[] colFct0;
    delete[] colFct1;
}
