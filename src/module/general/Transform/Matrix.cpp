/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Matrix.h"
#include <vector>
#include <appl/ApplInterface.h>
#include <do/coDoUnstructuredGrid.h>
using namespace covise;

Matrix::Matrix()
{
    this->rotation_[XX] = 1.0;
    this->rotation_[XY] = 0.0;
    this->rotation_[XZ] = 0.0;
    this->rotation_[YX] = 0.0;
    this->rotation_[YY] = 1.0;
    this->rotation_[YZ] = 0.0;
    this->rotation_[ZX] = 0.0;
    this->rotation_[ZY] = 0.0;
    this->rotation_[ZZ] = 1.0;
    IdentityMatrix_ = true;
    type_ = TRIVIAL;
    this->translation_[X] = 0.0;
    this->translation_[Y] = 0.0;
    this->translation_[Z] = 0.0;
    this->jacobian_ = POS_JACOBIAN;
}

Matrix::Matrix(const Matrix &rhs)
{
    std::copy(rhs.rotation_, rhs.rotation_ + 9, rotation_);
    std::copy(rhs.translation_, rhs.translation_ + 3, translation_);
    IdentityMatrix_ = rhs.IdentityMatrix_;
    type_ = rhs.type_;
    jacobian_ = rhs.jacobian_;
}

Matrix &
    Matrix::
    operator=(const Matrix &rhs)
{
    if (this != &rhs)
    {
        std::copy(rhs.rotation_, rhs.rotation_ + 9, rotation_);
        std::copy(rhs.translation_, rhs.translation_ + 3, translation_);
        IdentityMatrix_ = rhs.IdentityMatrix_;
        type_ = rhs.type_;
        jacobian_ = rhs.jacobian_;
    }
    return *this;
}

void
Matrix::setFlags()
{
    IdentityMatrix_ = false;
    type_ = ROTATION;
    jacobian_ = (Determinant() > 0.0) ? POS_JACOBIAN : NEG_JACOBIAN;
    /*
      cout<<"+++++++++++++"<<endl;
      std::copy(rotation_,rotation_+9,
                std::ostream_iterator<float>(cout, " "));
      cout<<endl;
      std::copy(translation_,translation_+3,
                std::ostream_iterator<float>(cout, " "));
      cout<<endl;
   */
}

float
Matrix::Determinant() const
{
    return (rotation_[XX] * rotation_[YY] * rotation_[ZZ] + rotation_[XY] * rotation_[YZ] * rotation_[ZX] + rotation_[XZ] * rotation_[YX] * rotation_[ZY] - rotation_[ZX] * rotation_[YY] * rotation_[XZ] - rotation_[ZY] * rotation_[YZ] * rotation_[XX] - rotation_[ZZ] * rotation_[YX] * rotation_[XY]);
}

void
Matrix::setMatrix(const float *v, int column)
{
    assert(column >= 0 && column < 4);
    switch (column)
    {
    case 0:
        this->rotation_[XX] = v[0];
        this->rotation_[YX] = v[1];
        this->rotation_[ZX] = v[2];
        break;
    case 1:
        this->rotation_[XY] = v[0];
        this->rotation_[YY] = v[1];
        this->rotation_[ZY] = v[2];
        break;
    case 2:
        this->rotation_[XZ] = v[0];
        this->rotation_[YZ] = v[1];
        this->rotation_[ZZ] = v[2];
        break;
    case 3:
        this->translation_[X] = v[0];
        this->translation_[Y] = v[1];
        this->translation_[Z] = v[2];
        break;
    }
}

Matrix &
    Matrix::
    operator*=(const Matrix &rhs)
{
    type_ = OTHER;
    float tmp[9];
    memcpy(tmp, this->rotation_, sizeof(float) * 9);
    if (!(IdentityMatrix_ && rhs.IdentityMatrix_))
    {
        this->rotation_[XX] = tmp[XX] * rhs.rotation_[XX] + tmp[XY] * rhs.rotation_[YX] + tmp[XZ] * rhs.rotation_[ZX];
        this->rotation_[XY] = tmp[XX] * rhs.rotation_[XY] + tmp[XY] * rhs.rotation_[YY] + tmp[XZ] * rhs.rotation_[ZY];
        this->rotation_[XZ] = tmp[XX] * rhs.rotation_[XZ] + tmp[XY] * rhs.rotation_[YZ] + tmp[XZ] * rhs.rotation_[ZZ];
        this->rotation_[YX] = tmp[YX] * rhs.rotation_[XX] + tmp[YY] * rhs.rotation_[YX] + tmp[YZ] * rhs.rotation_[ZX];
        this->rotation_[YY] = tmp[YX] * rhs.rotation_[XY] + tmp[YY] * rhs.rotation_[YY] + tmp[YZ] * rhs.rotation_[ZY];
        this->rotation_[YZ] = tmp[YX] * rhs.rotation_[XZ] + tmp[YY] * rhs.rotation_[YZ] + tmp[YZ] * rhs.rotation_[ZZ];
        this->rotation_[ZX] = tmp[ZX] * rhs.rotation_[XX] + tmp[ZY] * rhs.rotation_[YX] + tmp[ZZ] * rhs.rotation_[ZX];
        this->rotation_[ZY] = tmp[ZX] * rhs.rotation_[XY] + tmp[ZY] * rhs.rotation_[YY] + tmp[ZZ] * rhs.rotation_[ZY];
        this->rotation_[ZZ] = tmp[ZX] * rhs.rotation_[XZ] + tmp[ZY] * rhs.rotation_[YZ] + tmp[ZZ] * rhs.rotation_[ZZ];
        IdentityMatrix_ = false;
    }

    this->translation_[X] += tmp[XX] * rhs.translation_[X] + tmp[XY] * rhs.translation_[Y] + tmp[XZ] * rhs.translation_[Z];
    this->translation_[Y] += tmp[YX] * rhs.translation_[X] + tmp[YY] * rhs.translation_[Y] + tmp[YZ] * rhs.translation_[Z];
    this->translation_[Z] += tmp[ZX] * rhs.translation_[X] + tmp[ZY] * rhs.translation_[Y] + tmp[ZZ] * rhs.translation_[Z];

    if (this->jacobian_ == rhs.jacobian_)
    {
        this->jacobian_ = POS_JACOBIAN;
    }
    else
    {
        this->jacobian_ = NEG_JACOBIAN;
    }

    return *this;
}

// keep the correct orientation in polygons, elements, triangle strips...
// used for polygons
void
Matrix::transformLists(int nCorners, int nPolygons,
                       int *outCl, const int *outPl) const
{
    if (jacobian_ == POS_JACOBIAN)
    {
        return;
    }
    // loop over elements
    std::vector<int> cellConn;
    for (int elem = 0; elem < nPolygons - 1; ++elem)
    {
        int first = outPl[elem];
        int last = outPl[elem + 1];
        int vert;
        for (vert = last - 1; vert >= first; --vert)
        {
            cellConn.push_back(outCl[vert]);
        }
        for (vert = 0; vert < last - first; ++vert)
        {
            outCl[vert + first] = cellConn[vert];
        }
        cellConn.clear();
    }
    int first = outPl[nPolygons - 1];
    int last = nCorners;
    int vert;
    for (vert = last - 1; vert >= first; --vert)
    {
        cellConn.push_back(outCl[vert]);
    }
    for (vert = 0; vert < last - first; ++vert)
    {
        outCl[vert + first] = cellConn[vert];
    }
}

// version for unstructured grid elements
void
Matrix::transformLists(int nCorners, int nPolygons,
                       int *outCl, const int *outPl, const int *type) const
{
    (void)nCorners;

    if (jacobian_ == POS_JACOBIAN)
    {
        return;
    }
    // loop over elements
    std::vector<int> cellConn;
    for (int elem = 0; elem < nPolygons; ++elem)
    {
        int *base = outCl + outPl[elem];
        int tmpCl[8];
        switch (type[elem])
        {
        case TYPE_HEXAEDER:
            tmpCl[0] = base[1];
            tmpCl[1] = base[0];
            tmpCl[2] = base[3];
            tmpCl[3] = base[2];
            tmpCl[4] = base[5];
            tmpCl[5] = base[4];
            tmpCl[6] = base[7];
            tmpCl[7] = base[6];
            memcpy(base, tmpCl, 8 * sizeof(int));
            break;
        case TYPE_PRISM:
            tmpCl[0] = base[1];
            tmpCl[1] = base[0];
            memcpy(base, tmpCl, 2 * sizeof(int));
            tmpCl[3] = base[4];
            tmpCl[4] = base[3];
            memcpy(base + 3, tmpCl + 3, 2 * sizeof(int));
            break;
        case TYPE_PYRAMID:
            tmpCl[0] = base[1];
            tmpCl[1] = base[0];
            tmpCl[2] = base[3];
            tmpCl[3] = base[2];
            memcpy(base, tmpCl, 4 * sizeof(int));
            break;
        case TYPE_TETRAHEDER:
            tmpCl[0] = base[1];
            tmpCl[1] = base[0];
            memcpy(base, tmpCl, 2 * sizeof(int));
            break;
        case TYPE_QUAD:
            tmpCl[0] = base[1];
            tmpCl[1] = base[0];
            tmpCl[2] = base[3];
            tmpCl[3] = base[2];
            memcpy(base, tmpCl, 4 * sizeof(int));
            break;
        case TYPE_TRIANGLE:
            tmpCl[0] = base[1];
            tmpCl[1] = base[0];
            memcpy(base, tmpCl, 2 * sizeof(int));
            break;
        case TYPE_BAR:
            break;
        case TYPE_POINT:
            break;
        default:
            break;
        }
    }
    return;
}

void Matrix::transformCoordinates(int nPoints, float *outX, float *outY, float *outZ, float *inX, float *inY, float *inZ) const
{
    int point;
    if (IdentityMatrix_)
    {
        for (point = 0; point < nPoints; ++point)
        {
            outX[point] = inX[point] + translation_[X];
            outY[point] = inY[point] + translation_[Y];
            outZ[point] = inZ[point] + translation_[Z];
        }
    }
    else
    {
        if (false && type_ == OTHER)
        {
            for (point = 0; point < nPoints; ++point)
            {
                outX[point] = inX[point] * rotation_[XX] + translation_[X];
                outY[point] = inY[point] * rotation_[YY] + translation_[Y];
                outZ[point] = inZ[point] * rotation_[ZZ] + translation_[Z];
            }
        }
        else
        {
            for (point = 0; point < nPoints; ++point)
            {
                outX[point] = rotation_[XX] * inX[point] + rotation_[XY] * inY[point] + rotation_[XZ] * inZ[point] + translation_[X];
                outY[point] = rotation_[YX] * inX[point] + rotation_[YY] * inY[point] + rotation_[YZ] * inZ[point] + translation_[Y];
                outZ[point] = rotation_[ZX] * inX[point] + rotation_[ZY] * inY[point] + rotation_[ZZ] * inZ[point] + translation_[Z];
            }
        }
    }
}

void
Matrix::transformCoordinates(int nPoints, float *xc, float *yc, float *zc) const
{
    int point;
    if (IdentityMatrix_)
    {
        for (point = 0; point < nPoints; ++point)
        {
            xc[point] += translation_[X];
            yc[point] += translation_[Y];
            zc[point] += translation_[Z];
        }
    }
    else
    {
        if (false && type_ == OTHER)
        {
            for (point = 0; point < nPoints; ++point)
            {
                xc[point] *= rotation_[XX];
                yc[point] *= rotation_[YY];
                zc[point] *= rotation_[ZZ];
                xc[point] += translation_[X];
                yc[point] += translation_[Y];
                zc[point] += translation_[Z];
            }
        }
        else
        {
            float out[3];
            for (point = 0; point < nPoints; ++point)
            {
                out[X] = rotation_[XX] * xc[point] + rotation_[XY] * yc[point] + rotation_[XZ] * zc[point];
                out[Y] = rotation_[YX] * xc[point] + rotation_[YY] * yc[point] + rotation_[YZ] * zc[point];
                out[Z] = rotation_[ZX] * xc[point] + rotation_[ZY] * yc[point] + rotation_[ZZ] * zc[point];
                xc[point] = out[X] + translation_[X];
                yc[point] = out[Y] + translation_[Y];
                zc[point] = out[Z] + translation_[Z];
            }
        }
    }
}

void
Matrix::ScaleMatrix(float scaleFactor, float x, float y, float z, int type)
{
    // translation_[X] = (1.0 - scaleFactor)*x;
    // translation_[Y] = (1.0 - scaleFactor)*y;
    // translation_[Z] = (1.0 - scaleFactor)*z;
    // the lines above are comented out for compatibility
    translation_[X] = x;
    translation_[Y] = y;
    translation_[Z] = z;
    if (type == 1 || type == 2)
        rotation_[XX] *= scaleFactor;
    if (type == 1 || type == 3)
        rotation_[YY] *= scaleFactor;
    if (type == 1 || type == 4)
        rotation_[ZZ] *= scaleFactor;
    if (scaleFactor < 0.0)
    {
        jacobian_ = NEG_JACOBIAN;
    }
    IdentityMatrix_ = false;
    type_ = SCALE;
}

void
Matrix::TranslateMatrix(float x, float y, float z)
{
    translation_[X] = x;
    translation_[Y] = y;
    translation_[Z] = z;
    type_ = TRANSLATION;
}

void
Matrix::RotateMatrix(float angleDEG,
                     const float *vertex,
                     float *normal)
{
    if (Normalise(normal) != 0)
    {
        return;
    }

    float angle = (float)(angleDEG * M_PI / 180.0);
    float ca = cos(angle);
    float sa = sin(angle);
    float angleFactor = 1 - ca;

    // new initialization as identity matrix
    rotation_[XX] = 1.;
    rotation_[XY] = 0.;
    rotation_[XZ] = 0.;
    rotation_[YX] = 0.;
    rotation_[YY] = 1.;
    rotation_[YZ] = 0.;
    rotation_[ZX] = 0.;
    rotation_[ZY] = 0.;
    rotation_[ZZ] = 1.;

    // translation
    translation_[X] = angleFactor * vertex[X];
    translation_[Y] = angleFactor * vertex[Y];
    translation_[Z] = angleFactor * vertex[Z];
    float scalNOV = normal[X] * vertex[X] + normal[Y] * vertex[Y] + normal[Z] * vertex[Z];
    translation_[X] -= angleFactor * scalNOV * normal[X];
    translation_[Y] -= angleFactor * scalNOV * normal[Y];
    translation_[Z] -= angleFactor * scalNOV * normal[Z];
    translation_[X] -= sa * (normal[Y] * vertex[Z] - normal[Z] * vertex[Y]);
    translation_[Y] -= sa * (normal[Z] * vertex[X] - normal[X] * vertex[Z]);
    translation_[Z] -= sa * (normal[X] * vertex[Y] - normal[Y] * vertex[X]);

    // rotation
    rotation_[XX] *= ca;
    rotation_[YY] *= ca;
    rotation_[ZZ] *= ca;

    rotation_[XX] += normal[X] * normal[X] * angleFactor;
    rotation_[XY] += normal[X] * normal[Y] * angleFactor;
    rotation_[XZ] += normal[X] * normal[Z] * angleFactor;
    rotation_[YX] += normal[Y] * normal[X] * angleFactor;
    rotation_[YY] += normal[Y] * normal[Y] * angleFactor;
    rotation_[YZ] += normal[Y] * normal[Z] * angleFactor;
    rotation_[ZX] += normal[Z] * normal[X] * angleFactor;
    rotation_[ZY] += normal[Z] * normal[Y] * angleFactor;
    rotation_[ZZ] += normal[Z] * normal[Z] * angleFactor;

    rotation_[XY] -= sa * normal[Z];
    rotation_[XZ] += sa * normal[Y];
    rotation_[YZ] -= sa * normal[X];
    rotation_[YX] += sa * normal[Z];
    rotation_[ZX] -= sa * normal[Y];
    rotation_[ZY] += sa * normal[X];
    IdentityMatrix_ = false;
    type_ = ROTATION;
}

void
Matrix::MirrorMatrix(float distance, float *normal)
{
    if (Normalise(normal) != 0)
    {
        return;
    }

    // set the translation part
    translation_[X] = 2 * distance * normal[X];
    translation_[Y] = 2 * distance * normal[Y];
    translation_[Z] = 2 * distance * normal[Z];

    // and now the rotation
    rotation_[XX] -= 2.0f * normal[X] * normal[X];
    rotation_[XY] -= 2.0f * normal[X] * normal[Y];
    rotation_[XZ] -= 2.0f * normal[X] * normal[Z];
    rotation_[YX] -= 2.0f * normal[Y] * normal[X];
    rotation_[YY] -= 2.0f * normal[Y] * normal[Y];
    rotation_[YZ] -= 2.0f * normal[Y] * normal[Z];
    rotation_[ZX] -= 2.0f * normal[Z] * normal[X];
    rotation_[ZY] -= 2.0f * normal[Z] * normal[Y];
    rotation_[ZZ] -= 2.0f * normal[Z] * normal[Z];

    // mark jacobian sign
    jacobian_ = NEG_JACOBIAN;
    IdentityMatrix_ = false;
    type_ = MIRROR;
}

float
Matrix::get(int i, int j) const
{
    if (i < 0 || j < 0 || i > 3 || j > 3)
    {
        return 0.;
    }

    if (i < 3 && j < 3)
    {
        if (j < 3)
            return rotation_[i * 3 + j];
        else
            return translation_[j];
    }

    if (i == j)
        return 1.;

    return 0.;
}

int
Matrix::Normalise(float *normal)
{
    // normalise
    float nlen = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (nlen == 0.0)
    {
        Covise::sendWarning("Degenerate normal vector: ignoring operation");
        return -1;
    }
    normal[0] /= nlen;
    normal[1] /= nlen;
    normal[2] /= nlen;
    return 0;
}

Matrix::Type
Matrix::type() const
{
    return type_;
}

void
Matrix::transformVector(int nPoints, float *vector[3]) const
{
    // we do not scale vectors for compatibility!!!!! Use Calc if necessary
    if (IdentityMatrix_ || type_ == TRIVIAL || type_ == SCALE)
    {
        return;
    }
    int point;
    // this may be optimised
    if (false && type_ == OTHER)
    {
        for (point = 0; point < nPoints; ++point)
        {
            vector[X][point] *= rotation_[XX];
            vector[Y][point] *= rotation_[YY];
            vector[Z][point] *= rotation_[ZZ];
        }
    }
    else
    {
        float tmp[3];
        for (point = 0; point < nPoints; ++point)
        {
            tmp[X] = rotation_[XX] * vector[X][point] + rotation_[XY] * vector[Y][point] + rotation_[XZ] * vector[Z][point];
            tmp[Y] = rotation_[YX] * vector[X][point] + rotation_[YY] * vector[Y][point] + rotation_[YZ] * vector[Z][point];
            tmp[Z] = rotation_[ZX] * vector[X][point] + rotation_[ZY] * vector[Y][point] + rotation_[ZZ] * vector[Z][point];
            vector[X][point] = tmp[X];
            vector[Y][point] = tmp[Y];
            vector[Z][point] = tmp[Z];
        }
    }
}

int
Matrix::WhichAxis(const float *vector)
{
    double x = fabs(vector[0]);
    double y = fabs(vector[1]);
    double z = fabs(vector[2]);
    int ret = 2;
    if (x > y && x > z)
    {
        ret = 0;
    }
    else if (y > x && y > z)
    {
        ret = 1;
    }
    return ret;
}

// get the permutation of coordinate axes when rotating them
void
Matrix::getOrder(int newOrder[3]) const
{
    float vector[3][3];
    vector[0][0] = rotation_[XX];
    vector[0][1] = rotation_[XY];
    vector[0][2] = rotation_[XZ];
    vector[1][0] = rotation_[YX];
    vector[1][1] = rotation_[YY];
    vector[1][2] = rotation_[YZ];
    vector[2][0] = rotation_[ZX];
    vector[2][1] = rotation_[ZY];
    vector[2][2] = rotation_[ZZ];
    int dim;
    for (dim = 0; dim < 3; ++dim)
    {
        newOrder[dim] = WhichAxis(vector[dim]);
    }
}

// ... the number of points per dimension in uniform
// and rectlinear grids is accordingly permutated
void
Matrix::reOrder(int nx, int ny, int nz, int *nnx, int *nny, int *nnz) const
{
    int ns[3];
    ns[0] = nx;
    ns[1] = ny;
    ns[2] = nz;
    int *output[3];
    output[0] = nnx;
    output[1] = nny;
    output[2] = nnz;
    int dim;
    int newOrder[3];
    getOrder(newOrder);
    for (dim = 0; dim < 3; ++dim)
    {
        *output[newOrder[dim]] = ns[dim];
    }
}

// in this case we permit also a scalar field
void
Matrix::reOrder(const float *u, float *uout, int nx, int ny, int nz) const
{
    int nnx, nny, nnz;

    int newOrder[3];
    getOrder(newOrder);

    reOrder(nx, ny, nz, &nnx, &nny, &nnz);

    // up(ip,jp,kp) = u(i,j,k) where ip = j (for instance if WhichAxis(v[1])==0
    int coordp[3];
    int &ip = coordp[0];
    int &jp = coordp[1];
    int &kp = coordp[2];
    int coord[3];
    for (ip = 0; ip < nnx; ++ip)
    {
        int bxp = ip * nny;
        for (jp = 0; jp < nny; ++jp)
        {
            int byp = (bxp + jp) * nnz;
            for (kp = 0; kp < nnz; ++kp)
            {
                coord[0] = coordp[newOrder[0]];
                coord[1] = coordp[newOrder[1]];
                coord[2] = coordp[newOrder[2]];
                uout[byp + kp] = u[coord[0] * ny * nz + coord[1] * nz + coord[2]];
            }
        }
    }
}

// and this is for vector fields
void
Matrix::reOrderAndTransform(float *u[3], float *uout[3], int nx, int ny, int nz) const
{
    int nnx, nny, nnz;

    int newOrder[3];
    getOrder(newOrder);

    reOrder(nx, ny, nz, &nnx, &nny, &nnz);

    // up(ip,jp,kp) = u(i,j,k) where ip = j (for instance if WhichAxis(v[1])==0
    int coordp[3];
    int &ip = coordp[0];
    int &jp = coordp[1];
    int &kp = coordp[2];
    int coord[3];
    for (ip = 0; ip < nnx; ++ip)
    {
        int bxp = ip * nny;
        for (jp = 0; jp < nny; ++jp)
        {
            int byp = (bxp + jp) * nnz;
            for (kp = 0; kp < nnz; ++kp)
            {
                coord[0] = coordp[newOrder[0]];
                coord[1] = coordp[newOrder[1]];
                coord[2] = coordp[newOrder[2]];
                uout[0][byp + kp] = u[0][coord[0] * ny * nz + coord[1] * nz + coord[2]];
                uout[1][byp + kp] = u[1][coord[0] * ny * nz + coord[1] * nz + coord[2]];
                uout[2][byp + kp] = u[2][coord[0] * ny * nz + coord[1] * nz + coord[2]];
            }
        }
    }
    // and now transform
    transformVector(nnx * nny * nnz, uout);
}
