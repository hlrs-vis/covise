/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiMatrix.h>

#include <OpenSG/OSGQuaternion.h>

OSG_USING_NAMESPACE

SGVruiMatrix::SGVruiMatrix()
{
}

SGVruiMatrix::~SGVruiMatrix()
{
}

vruiMatrix *SGVruiMatrix::makeIdentity()
{
    this->matrix.setIdentity();
    return this;
}

vruiMatrix *SGVruiMatrix::preTranslated(double x, double y, double z, vruiMatrix *matrix)
{
    Matrix4d translate;
    translate.setIdentity();
    translate.setTranslate(x, y, z);
    SGVruiMatrix *osgVruiMatrix = dynamic_cast<SGVruiMatrix *>(matrix);
    translate.mult(osgVruiMatrix->matrix);
    this->matrix = translate;
    return this;
}

vruiMatrix *SGVruiMatrix::makeTranslate(double x, double y, double z)
{
    this->matrix.setIdentity();
    this->matrix.setTranslate(x, y, z);
    return this;
}

vruiMatrix *SGVruiMatrix::makeRotate(double degrees, double xAxis, double yAxis, double zAxis)
{
    this->matrix.setIdentity();
    this->matrix.setRotate(QuaternionBase<Real64>(Vec3d(xAxis, yAxis, zAxis), osgdegree2rad(degrees)));
    return this;
}

vruiMatrix *SGVruiMatrix::makeScale(double x, double y, double z)
{
    this->matrix.setIdentity();
    this->matrix.setScale(x, y, z);
    return this;
}

vruiMatrix *SGVruiMatrix::makeEuler(double h, double p, double r)
{
    QuaternionBase<Real64> q;
    q.setValue(osgdegree2rad(h), osgdegree2rad(p), osgdegree2rad(r));
    this->matrix.setIdentity();
    this->matrix.setRotate(q);
    return this;
}

vruiMatrix *SGVruiMatrix::mult(const vruiMatrix *matrix)
{
    const SGVruiMatrix *osgVruiMatrix = dynamic_cast<const SGVruiMatrix *>(matrix);
    this->matrix.mult(const_cast<SGVruiMatrix *>(osgVruiMatrix)->matrix);
    return this;
}

vruiMatrix *SGVruiMatrix::makeInverse(const vruiMatrix *source)
{
    const SGVruiMatrix *sourceMat = dynamic_cast<const SGVruiMatrix *>(source);
    this->matrix.invertFrom(sourceMat->matrix);
    return this;
}

double &SGVruiMatrix::operator()(int row, int column)
{
    return this->matrix[column][row];
}

double SGVruiMatrix::operator()(int row, int column) const
{
    return (double)this->matrix[column][row];
}

coVector SGVruiMatrix::getFullXformPt(const coVector &point) const
{
    Vec3d pnt(point[0], point[1], point[2]);
    this->matrix.multVecMatrix(pnt);
    //transformedPnt /= transformedPnt[3];

    return coVector(pnt[0], pnt[1], pnt[2]);
}

coVector SGVruiMatrix::getHPR() const
{
    QuaternionBase<Real64> rotation, scaleOrientation;
    Vec3d translation, scaleFactor;
    this->matrix.getTransform(translation, rotation, scaleFactor, scaleOrientation);

    coVector hpr;
    Matrix4d m = this->matrix;

    double cp;
    hpr[1] = asin(m[2][1]);
    cp = cos(hpr[1]);
    hpr[0] = -asin(m[0][1] / cp);
    double diff = cos(hpr[0]) * cp - m[1][1];
    if (diff < -0.01 || diff > 0.01)
    { /* oops, not correct, use other Heading angle */
        hpr[0] = M_PI - hpr[0];
        diff = cos(hpr[0]) * cp - m[1][1];
        if (diff < -0.01 || diff > 0.01)
        { /* oops, not correct, use other pitch angle */
            hpr[1] = M_PI - hpr[1];
            cp = cos(hpr[1]);
            hpr[0] = -asin(m[0][1] / cp);
            diff = cos(hpr[0]) * cp - m[1][1];
            if (diff < -0.01 || diff > 0.01)
            { /* oops, not correct, use other Heading angle */
                hpr[0] = M_PI - hpr[0];
            }
        }
    }
    hpr[2] = acos(m[2][2] / cp);
    diff = -sin(hpr[2]) * cp - m[2][0];
    if (diff < -0.01 || diff > 0.01) /* oops, not correct, use other roll angle */
        hpr[2] = -hpr[2];
    hpr[0] = hpr[0] / M_PI * 180.0;
    hpr[1] = hpr[1] / M_PI * 180.0;
    hpr[2] = hpr[2] / M_PI * 180.0;

    return hpr;
}

void SGVruiMatrix::setMatrix(const Matrix4d &matrix)
{
    this->matrix = matrix;
}

void SGVruiMatrix::setMatrix(const Matrix4f &matrix)
{
    const Real32 *m = matrix.getValues();
    this->matrix.setValue(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
}

Matrix4d SGVruiMatrix::getMatrix()
{
    return this->matrix;
}

Matrix4f SGVruiMatrix::getFloatMatrix()
{
    const Real64 *m = this->matrix.getValues();
    this->matrixFloat.setValue((float)m[0], (float)m[1], (float)m[2], (float)m[3],
                               (float)m[4], (float)m[5], (float)m[6], (float)m[7],
                               (float)m[8], (float)m[9], (float)m[10], (float)m[11],
                               (float)m[12], (float)m[13], (float)m[14], (float)m[15]);
    return matrixFloat;
}
