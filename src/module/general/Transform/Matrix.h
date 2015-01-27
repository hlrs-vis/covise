/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Matrix
//
// Euclidean transformation
//
//  Initial version: 2002-05-?? Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _TRANSFORM_MATRIX_H
#define _TRANSFORM_MATRIX_H

class Matrix
{
public:
    enum Type // OTHER only appears when tiling as combiantion of mirroring and translations
    {
        TRIVIAL,
        ROTATION,
        TRANSLATION,
        SCALE,
        MIRROR,
        OTHER
    };
    /// constructor
    Matrix();
    Matrix(const Matrix &rhs);
    Matrix &operator=(const Matrix &rhs);
    /// set components, column=0,1,2 -> rotation, column=3 -> translation
    void setMatrix(const float *v, int column);
    /// set auxiliary flags based on matrix components
    void setFlags();
    /// multiplication, only used when tiling
    Matrix &operator*=(const Matrix &rhs);
    /** change polygon connectivity if the jacobian is negative
       * @param nCorners number of vertices
       * @param nPolygons number of polygons (or elements or lines...
       * @param outCl connectivity to be modified if necessary
       * @param outPl polygon list
       */
    void transformLists(int nCorners, int nPolygons, int *outCl, const int *outPl) const;
    /** change unstructured grid connectivity if the jacobian is negative
       * @param nCorners number of vertices
       * @param nPolygons number of polygons (or elements or lines...
       * @param outCl connectivity to be modified if necessary
       * @param outPl polygon list
       */
    void transformLists(int nCorners, int nPolygons, int *outCl, const int *outPl, const int *type) const;
    /// transform coordinate points
    void transformCoordinates(int nPoints, float *, float *, float *) const;
    void transformCoordinates(int nPoints, float *outX, float *outY, float *outZ, float *inX, float *inY, float *inZ) const;
    /// transform vector field
    void transformVector(int nPoints, float *vector[3]) const;

    /** set scaling matrix
       * @param scale scaling factor
       * @param x X-coordinate of the image of the origin
       * @param y Y-coordinate of the image of the origin
       * @param z Z-coordinate of the image of the origin
       */
    void ScaleMatrix(float scale, float x, float y, float z, int type = 1);
    /// set a translation
    void TranslateMatrix(float, float, float);
    /** set a rotation
       * @param angleDEG angle in degrees
       * @param vertex a point of the axis
       * @param axis direction
       */
    void RotateMatrix(float angleDEG, const float *vertex, float *normal);
    /** set a mirroring transformation
       * @param distance distance from the origin to the mirror
       * @param mirror mirror normal
       */
    void MirrorMatrix(float distance, float *normal);

    /// reordering number of points of RCTGRD an UNIRD when rotating
    void reOrder(int nx, int ny, int nz, int *nnx, int *nny, int *nnz) const;
    /** reordering scalar data on RCTGRD an UNIRD when rotating
       * @param u input scalar field
       * @param u outpur scalar field
       * @param nx number of points in the X-direction for the input geometry
       * @param ny number of points in the Y-direction for the input geometry
       * @param nz number of points in the Z-direction for the input geometry
       */
    void reOrder(const float *u, float *uout, int nx, int ny, int nz) const;
    /** reordering vector data on RCTGRD an UNIRD when rotating
       * @param u input vector field
       * @param u outpur vector field
       * @param nx number of points in the X-direction for the input geometry
       * @param ny number of points in the Y-direction for the input geometry
       * @param nz number of points in the Z-direction for the input geometry
       */
    void reOrderAndTransform(float *u[3], float *uout[3], int nx, int ny, int nz) const;

    float get(int i, int j) const;

    /// normalise a 3-component vector and return 0, if vector is 0, return -1
    static int Normalise(float *);
    /// return transformation type
    Type type() const;
    enum Jacobian
    {
        NEG_JACOBIAN,
        POS_JACOBIAN
    };
    /// return jacobian sign
    Jacobian getJacobian() const
    {
        return jacobian_;
    }
    /// return 0, 1 or 2, meaning X, Y, Z, according to the maximum of the absolute values of the three components of a vector
    static int WhichAxis(const float *vector);

private:
    float Determinant() const;
    // write the permutation of coordinate direction when rotating RCTGRD or UNIGD
    void getOrder(int newOrder[3]) const;
    enum
    {
        XX = 0,
        XY = 1,
        XZ = 2,
        YX = 3,
        YY = 4,
        YZ = 5,
        ZX = 6,
        ZY = 7,
        ZZ = 8
    };
    enum
    {
        X = 0,
        Y = 1,
        Z = 2
    };
    float rotation_[9]; // the matrix
    float translation_[3]; // the translation
    Jacobian jacobian_; // jacobian sign
    bool IdentityMatrix_; // flag if rotation is assuredly the identity
    Type type_; // transformation type
};
#endif
