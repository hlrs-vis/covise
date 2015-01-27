/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <math.h>
#include "MathUtils.h"
#include "VrmlLinMath.h"

VrmlMatrix::VrmlMatrix(float a00, float a01, float a02, float a03,
                       float a10, float a11, float a12, float a13,
                       float a20, float a21, float a22, float a23,
                       float a30, float a31, float a32, float a33)
{
    mat[0][0] = a00;
    mat[0][1] = a01;
    mat[0][2] = a02;
    mat[0][3] = a03;
    mat[1][0] = a10;
    mat[1][1] = a11;
    mat[1][2] = a12;
    mat[1][3] = a13;
    mat[2][0] = a20;
    mat[2][1] = a21;
    mat[2][2] = a22;
    mat[2][3] = a23;
    mat[3][0] = a30;
    mat[3][1] = a31;
    mat[3][2] = a32;
    mat[3][3] = a33;
}

void VrmlMatrix::makeIdent()
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            mat[i][j] = i == j ? 1.0 : 0.0;
}

void VrmlMatrix::makeTrans(float x, float y, float z)
{
    makeIdent();
    mat[3][0] = x;
    mat[3][1] = y;
    mat[3][2] = z;
}

void VrmlMatrix::makeScale(float x, float y, float z)
{
    makeIdent();
    mat[0][0] = x;
    mat[1][1] = y;
    mat[2][2] = z;
}

void VrmlMatrix::makeRot(float x, float y, float z, float angle)
{
    float len = sqrt(x * x + y * y + z * z);
    if (!FPZERO(len))
    {
        x /= len;
        y /= len;
        z /= len;
    }
    float s = sin(angle);
    float c = cos(angle);
    float t = 1.0 - c;

    mat[0][0] = t * x * x + c;
    mat[0][1] = t * x * y + s * z;
    mat[0][2] = t * x * z - s * y;
    mat[0][3] = 0.0;
    mat[1][0] = t * x * y - s * z;
    mat[1][1] = t * y * y + c;
    mat[1][2] = t * y * z + s * x;
    mat[1][3] = 0.0;
    mat[2][0] = t * x * z + s * y;
    mat[2][1] = t * y * z - s * x;
    mat[2][2] = t * z * z + c;
    mat[2][3] = 0.0;
    mat[3][0] = mat[3][1] = mat[3][2] = 0.0;
    mat[3][3] = 1.0;
}

bool VrmlMatrix::invertFull(const VrmlMatrix &m)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            mat[i][j] = m.mat[i][j];

    int p[4];
    for (int k = 0; k < 4; k++)
    {
        p[k] = 0;
        float sup = 0.0;
        for (int i = k; i < 4; i++)
        {
            float s = 0.0;
            for (int j = k; j < 4; j++)
                s += fabs(mat[i][j]);
            float q = fabs(mat[i][k]) / s;
            if (sup < q)
            {
                sup = q;
                p[k] = i;
            }
        }
        if (FPZERO(sup))
            return false;
        if (p[k] != k)
            for (int j = 0; j < 4; j++)
            {
                float h = mat[k][j];
                mat[k][j] = mat[p[k]][j];
                mat[p[k]][j] = h;
            }
        float pivot = mat[k][k];
        for (int j = 0; j < 4; j++)
            if (j != k)
            {
                mat[k][j] = -mat[k][j] / pivot;
                for (int i = 0; i < 4; i++)
                    if (i != k)
                        mat[i][j] += mat[i][k] * mat[k][j];
            }
        for (int i = 0; i < 4; i++)
            mat[i][k] = mat[i][k] / pivot;
        mat[k][k] = 1.0 / pivot;
    }

    for (int k = 4 - 1; k >= 0; k--)
        if (p[k] != k)
            for (int i = 0; i < 4; i++)
            {
                float h = mat[i][k];
                mat[i][k] = mat[i][p[k]];
                mat[i][p[k]] = h;
            }

    return true;
}

void VrmlMatrix::invertAff(const VrmlMatrix &m)
{
    invertFull(m);
}

void VrmlMatrix::transpose(const VrmlMatrix &m)
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            mat[i][j] = m.mat[j][i];
}

void VrmlMatrix::mult(const VrmlMatrix &m1, const VrmlMatrix &m2)
{
    float b[4][4];
    float c[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            b[i][j] = m1.mat[i][j];
            c[i][j] = m2.mat[i][j];
        }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            mat[i][j] = b[0][j] * c[i][0] + b[1][j] * c[i][1] + b[2][j] * c[i][2] + b[3][j] * c[i][3];
}

VrmlMatrix &VrmlMatrix::operator*=(VrmlMatrix &m)
{
    float b[4][4];
    float c[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
        {
            b[i][j] = mat[i][j];
            c[i][j] = m.mat[i][j];
        }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            mat[i][j] = b[0][j] * c[i][0] + b[1][j] * c[i][1] + b[2][j] * c[i][2] + b[3][j] * c[i][3];

    return *this;
}

VrmlVec::VrmlVec(float x, float y, float z)
{
    this->x[0] = x;
    this->x[1] = y;
    this->x[2] = z;
}

void VrmlVec::xformVec(const VrmlVec &v, const VrmlMatrix &m)
{
    float u[3] = { v.x[0], v.x[1], v.x[2] };
    for (int i = 0; i < 3; i++)
        x[i] = u[0] * m.mat[i][0] + u[1] * m.mat[i][1] + u[2] * m.mat[i][2];
}

void VrmlVec::xformPt(const VrmlVec &v, const VrmlMatrix &m)
{
    float u[3] = { v.x[0], v.x[1], v.x[2] };
    for (int i = 0; i < 3; i++)
        x[i] = u[0] * m.mat[i][0] + u[1] * m.mat[i][1] + u[2] * m.mat[i][2] + m.mat[i][3];
}

void VrmlVec::fullXformPt(const VrmlVec &v, const VrmlMatrix &m)
{
    float u[3] = { v.x[0], v.x[1], v.x[2] };
    float w = u[0] * m.mat[3][0] + u[1] * m.mat[3][1] + u[2] * m.mat[3][2] + m.mat[3][3];
    for (int i = 0; i < 3; i++)
        x[i] = (u[0] * m.mat[i][0] + u[1] * m.mat[i][1] + u[2] * m.mat[i][2] + m.mat[i][3]) / w;
}
