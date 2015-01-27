/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GridBlock.h"

#include "LTracer.h"

GridBlock::GridBlock(GridStep *parent, Grid *owner, int b)
{
    step = parent;
    grid = owner;
    myBlock = b;

    gbiCache = NULL;

    numSteps = 1;
    isRotatingFlag = false;
    return;
}

GridBlock::~GridBlock()
{
    if (gbiCache)
        delete gbiCache;
    return;
}

void GridBlock::setRotation(float speed, float rax, float ray, float raz)
{
    rotSpeed = speed;
    rotAxis[0] = rax;
    rotAxis[1] = ray;
    rotAxis[2] = raz;
    isRotatingFlag = true;
}

int GridBlock::intersectionTest(float *A, float *B, float *C, float *O,
                                float *P, float *alpha, float *beta, float *t)
{
    // based on Graphics Gems I, 390pp "An Efficient Ray-Polygon Intersection"

    // NOTE: See TetraGrid.cpp from TETRA_TRACE for some precautions to prevent
    //   numerical problems. They are intentionally left out here but we might have
    //   to add them later depending on our results.

    // =====================================================
    // First Step: Intersecting the Embedding Plane

    float N[3], a[3], b[3], d;

    // a = B-A
    a[0] = B[0] - A[0];
    a[1] = B[1] - A[1];
    a[2] = B[2] - A[2];

    // b = C-A
    b[0] = C[0] - A[0];
    b[1] = C[1] - A[1];
    b[2] = C[2] - A[2];

    // N = a x b
    N[0] = a[1] * b[2] - a[2] * b[1];
    N[1] = a[2] * b[0] - a[0] * b[2];
    N[2] = a[0] * b[1] - a[1] * b[0];

    // d = -(A*N)
    d = -(A[0] * N[0] + A[1] * N[1] + A[2] * N[2]);

    // parametric representation of the ray:  r = O+D*t, where D = P-O
    float D[3];
    D[0] = P[0] - O[0];
    D[1] = P[1] - O[1];
    D[2] = P[2] - O[2];

    // evaluate the intersection point with   t = -((d+N*O)/(N*D))
    //  Note that it could be possible for the polygon and the ray to be parallel (N*D=0)
    //  so we have to check this first.   q = N*D
    float q = N[0] * D[0] + N[1] * D[1] + N[2] * D[2];
    if (q == 0.0)
        return (1);

    // not parallel, so compute t now
    *t = -(d + (N[0] * O[0] + N[1] * O[1] + N[2] * O[2])) / q;

    // check if the intersection is behind the ray
    if (*t < 0.0)
    {
        // numerical problem: allow the intersection to be slightly behind the ray
        if (*t > -5.0e-8)
            *t = 0.0;
        else
        {
            //cerr << "b{" << *t << "}";
            return (2);
        }
    }
    //cerr << "|" << *t << "|";

    // the point of intersection p is then: p = O + D*t
    float p[3];
    p[0] = O[0] + D[0] * (*t);
    p[1] = O[1] + D[1] * (*t);
    p[2] = O[2] + D[2] * (*t);

    // find the dominant axis of the normal vector
    float n, n_max;
    int i1, i2;

    n_max = fabsf(N[0]);
    i1 = 1;
    i2 = 2;

    n = fabsf(N[1]);
    if (n > n_max)
    {
        n_max = n;
        i1 = 0;
        i2 = 2;
    }

    n = fabsf(N[2]);
    if (n > n_max)
    {
        i1 = 0;
        i2 = 1;
    }

    // compute projected points u, v (as shown on page 392/393)
    float u0, u1, u2, v0, v1, v2;

    u0 = p[i1] - A[i1];
    u1 = B[i1] - A[i1];
    u2 = C[i1] - A[i1];

    v0 = p[i2] - A[i2];
    v1 = B[i2] - A[i2];
    v2 = C[i2] - A[i2];

    // compute alpha and beta
    q = (u1 * v2) - (u2 * v1);
    *alpha = (u0 * v2 - u2 * v0) / q;
    *beta = (u1 * v0 - u0 * v1) / q;
    if (*alpha > -1.0e-6 && *beta > -1.0e-6 && (*alpha + *beta) < 1.0 + 1.0e-6)
        return (0);
    return (3);

    /*
   int i = 3;
   if( u1==0.0 )
   {
      *beta = u0/u2;
      //cerr << "1";
      if( *beta>=0.0 && *beta<=1.0 )
      {
         //cerr << ".";
         *alpha = (v0-(*beta)*v2)/v1;

   //cerr << "(" << *alpha << "," << *beta << ")";

   //i = ((*alpha>=0.0) && (*alpha+*beta<=1.0))?0:3;
   // numerical problem: allow for the intersection to be slightly outside the triangle
   i = ((*alpha>=-1.0e-6) && (*alpha+*beta<=1.0+1.0e-4))?0:3;
   }
   //else
   //   cerr << "[" << *beta << "]";
   }
   else
   {
   *beta = (v0*u1 - u0*v1)/(v2*u1 - u2*v1);
   //cerr << "2";
   if( *beta>=0.0 && *beta<=1.0 )
   {
   //cerr << ".";
   *alpha = (u0-(*beta)*u2)/u1;

   //cerr << "(" << *alpha << "," << *beta << ")";

   //i = ((*alpha>=0.0) && (*alpha+*beta<=1.0))?0:3;
   // numerical problem: allow for the intersection to be slightly outside the triangle
   i = ((*alpha>=-1.0e-6) && (*alpha+*beta<=1.0+1.0e-4))?0:3;
   }
   //else
   //   cerr << "[" << *beta << "]";
   }
   */

    //return( i );
}

float GridBlock::tetraVolume(float *p0, float *p1, float *p2, float *p3)
{
    // compute the volume of the tetrahedra defined through the given points

    float v;
    float a, b, c, d, e, f;

    a = p2[1] - p0[1];
    b = p2[2] - p0[2];
    c = p2[0] - p0[0];
    d = p3[2] - p0[2];
    e = p3[0] - p0[0];
    f = p3[1] - p0[1];

    v = ((a * d - f * b) * (p1[0] - p0[0]) + (b * e - d * c) * (p1[1] - p0[1]) + (c * f - e * a) * (p1[2] - p0[2])) / 6.0f;

    return (v);
}

int GridBlock::tetraTrace(float *p0, float *p1, float *p2, float *p3, float *px)
{
    float alpha, beta;
    float pc[3];
    int r, s = -1;
    float tMin = 1e+10, t;

    // 1. calculate volumetric center of the tetrahedra
    pc[0] = (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0f;
    pc[1] = (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0f;
    pc[2] = (p0[2] + p1[2] + p2[2] + p3[2]) / 4.0f;

    // Side 0 (0-1-2)
    if (!(r = this->intersectionTest(p0, p2, p1, pc, px, &alpha, &beta, &t)))
        return (0);
    if (r == 3 && alpha >= 0.0 && beta >= 0.0)
    {
        s = 0;
        tMin = t;
    }

    // Side 1 (0-1-3)
    if (!(r = this->intersectionTest(p0, p1, p3, pc, px, &alpha, &beta, &t)))
        return (1);
    if (r == 3 && t < tMin && alpha >= 0.0 && beta >= 0.0)
    {
        s = 1;
        tMin = t;
    }

    // Side 2 (0-2-3)
    if (!(r = this->intersectionTest(p0, p3, p2, pc, px, &alpha, &beta, &t)))
        return (2);
    if (r == 3 && t < tMin && alpha >= 0.0 && beta >= 0.0)
    {
        s = 2;
        tMin = t;
    }

    // Side 3 (1-2-3)
    if (!(r = this->intersectionTest(p1, p2, p3, pc, px, &alpha, &beta, &t)))
        return (3);
    if (r == 3 && t < tMin && alpha >= 0.0 && beta >= 0.0)
    {
        s = 3;
        tMin = t;
    }

    //cerr << "best fit" << endl;

    // just use the "best fitting" side if we found one
    if (s != -1)
        return (s);

    // hmm seems we are passing exactly via an edge or corner...
    // ...so let's role the dice
    cerr << "GridBlock::tetraTrace=> passed via edge or corner" << endl;

    cerr << "------------>" << endl;
    cerr << "p0: " << p0[0] << " " << p0[1] << " " << p0[2] << endl;
    cerr << "p1: " << p1[0] << " " << p1[1] << " " << p1[2] << endl;
    cerr << "p2: " << p2[0] << " " << p2[1] << " " << p2[2] << endl;
    cerr << "p3: " << p3[0] << " " << p3[1] << " " << p3[2] << endl;
    cerr << "px: " << px[0] << " " << px[1] << " " << px[2] << endl << "..." << endl;
    cerr << "S0: " << this->intersectionTest(p0, p2, p1, pc, px, &alpha, &beta, &t) << "  a: " << alpha << "  b: " << beta << "  t: " << t << endl;
    cerr << "S1: " << this->intersectionTest(p0, p1, p3, pc, px, &alpha, &beta, &t) << "  a: " << alpha << "  b: " << beta << "  t: " << t << endl;
    cerr << "S2: " << this->intersectionTest(p0, p3, p2, pc, px, &alpha, &beta, &t) << "  a: " << alpha << "  b: " << beta << "  t: " << t << endl;
    cerr << "S3: " << this->intersectionTest(p1, p2, p3, pc, px, &alpha, &beta, &t) << "  a: " << alpha << "  b: " << beta << "  t: " << t << endl;

    cerr << "<------------" << endl;

    debugTetra(p0, p1, p2, p3);
    debugTetra(px[0], px[1], px[2], 0.001f);
    debugTetra(pc[0], pc[1], pc[2], 0.0005f);

    debugTetra(p0[0], p0[1], p0[2], 0.0001f);
    debugTetra(p1[0], p1[1], p1[2], 0.0002f);
    debugTetra(p2[0], p2[1], p2[2], 0.0003f);
    debugTetra(p3[0], p3[1], p3[2], 0.0004f);

    debugTetra(pc, pc, pc, px);

    return (rand() % 4);
}

void GridBlock::debugTetra(float p0[3], float p1[3], float p2[3], float p3[3])
{
    if (g_numElem >= G_ELEMMAX)
        return;

    g_elemList[g_numElem] = g_numConn;
    g_connList[g_numConn] = g_numPoints;
    g_numConn++;
    g_connList[g_numConn] = g_numPoints + 1;
    g_numConn++;
    g_connList[g_numConn] = g_numPoints + 2;
    g_numConn++;
    g_connList[g_numConn] = g_numPoints + 3;
    g_numConn++;
    g_typeList[g_numElem] = TYPE_TETRAHEDER;
    g_dOut[g_numElem] = (float)g_numElem;
    g_numElem++;

    g_xCoord[g_numPoints] = p0[0];
    g_yCoord[g_numPoints] = p0[1];
    g_zCoord[g_numPoints] = p0[2];
    g_numPoints++;

    g_xCoord[g_numPoints] = p1[0];
    g_yCoord[g_numPoints] = p1[1];
    g_zCoord[g_numPoints] = p1[2];
    g_numPoints++;

    g_xCoord[g_numPoints] = p2[0];
    g_yCoord[g_numPoints] = p2[1];
    g_zCoord[g_numPoints] = p2[2];
    g_numPoints++;

    g_xCoord[g_numPoints] = p3[0];
    g_yCoord[g_numPoints] = p3[1];
    g_zCoord[g_numPoints] = p3[2];
    g_numPoints++;

    return;
}

void GridBlock::debugTetra(float x, float y, float z, float dl)
{
    float p0[3], p1[3], p2[3], p3[3];

    p0[0] = x - dl;
    p0[1] = y - dl;
    p0[2] = z - dl;
    p1[0] = x + dl;
    p1[1] = y - dl;
    p1[2] = z - dl;
    p2[0] = x;
    p2[1] = y + dl;
    p2[2] = z - dl;
    p3[0] = x;
    p3[1] = y;
    p3[2] = z + dl;
    this->debugTetra(p0, p1, p2, p3);
    g_dOut[g_numElem - 1] = (float)(g_numElem > 2 ? g_numElem - 2 : 0);

    return;
}
