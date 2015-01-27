/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "Interpolator.h"

Interpolator::Interpolator()
{
}

Interpolator::~Interpolator()
{
}

/*
 * Interpolate Scale
 */
float Interpolator::interpolateScale(float startScale,
                                     ViewDesc *destination, float lambda)
{
    float scale;

    //   if (destination->_scale <0)
    if (destination->getScale() < 0)
    {
        scale = startScale;
    }
    else
    {
        //      scale = destination->_scale + (startScale - destination->_scale)*(1
        //            -lambda);
        scale = destination->getScale() + (startScale - destination->getScale()) * (1 - lambda);
    }
    return scale;
}

/*
 * Interpolate Rotation
 */
Matrix Interpolator::interpolateRotation(Matrix startMatrix, Vec3 tangentOut, double startScale, double destScale, Matrix destMat, Vec3 tangentIn, float lambda, RotationMode rotationMode)
{
    Matrix matrix_I;
    Matrix matrix_F;
    Quat rotQuat_I;
    Quat rotQuat_F;
    switch (rotationMode)
    {
    case QUATERNION:
    {
        //   matrix_I.makeCoord(&startCoord);
        //      startCoord.makeMat(matrix_I);
        matrix_I = startMatrix;
        rotQuat_I.set(matrix_I);

        //   matrix_F.makeCoord(&(destination->coord));
        //      destination->coord.makeMat(matrix_F);
        matrix_F = destMat;
        rotQuat_F.set(matrix_F);

        // and interpolate rotation Quaternions using slerp
        Quat rotQuat_Interpolated;
        //      rotQuat_Interpolated.slerp(lambda, rotQuat_I, rotQuat_F);
        rotQuat_Interpolated.slerp(lambda, rotQuat_I, rotQuat_F);

        // get the interpolated rotation-matrix
        Matrix matrix_Interpolated;
        matrix_Interpolated.makeIdentity();
        //      matrix_Interpolated.makeQuat(rotQuat_Interpolated);
        matrix_Interpolated.makeRotate(rotQuat_Interpolated);

        //returnMatrix.postMult(matrix_Interpolated);
        return matrix_Interpolated;
        break;
    }
    case FOLLOWPATH:
    {
        //      matrix_I.makeCoord(&startCoord);
        //      startCoord.makeMat(matrix_I);
        matrix_I = startMatrix;
        //      matrix_F.makeCoord(&(destination->coord));
        //      destination->coord.makeMat(matrix_F);
        matrix_F = destMat;

        //get P1, P4 in object coordinates
        matrix_I.invert(matrix_I);
        matrix_F.invert(matrix_F);
        Vec3 p1;
        Vec3 p4;
        //      matrix_I.getRow(3, p1);                                                                  // ?
        //      matrix_F.getRow(3, p4);
        p1 = matrix_I.getTrans();
        p4 = matrix_F.getTrans();

        //get P2, P3
        Vec3 p2;
        Vec3 p3;
        //      p2.xformVec(tangentOut, matrix_I);
        //      p3.xformVec(tangentIn, matrix_F);
        p2 = Matrix::transform3x3(tangentOut, matrix_I);
        p3 = Matrix::transform3x3(tangentIn, matrix_F);

        //      p2.add(p1, p2);
        //      p3.add(p3, p4);

        p2 += p1;
        p3 += p4;

        // exclude scale form calculation!
        p1 *= (1 / startScale);
        p2 *= (1 / startScale);
        //      p3 *= (1 / destination->_scale);
        //      p4 *= (1 / destination->_scale);
        p3 *= (1 / destScale);
        p4 *= (1 / destScale);

        Vec3 vecArray[4] = { p1, p2, p3, p4 };
        //      Vec3Array *vecArray = new Vec3Array;
        //      vecArray->push_back(p1);
        //      vecArray->push_back(p2);
        //      vecArray->push_back(p3);
        //      vecArray->push_back(p4);

        // get current tangent
        Vec3 currentHeading = casteljau(vecArray, lambda, 3);
        Vec3 z = Vec3(0, 0, 1);
        Vec3 x;

        //      x.cross(currentHeading, z);
        //      x.normalize();
        //      z.cross(x, currentHeading);
        //      z.normalize();

        //      x = currentHeading^z;
        x[0] = ((currentHeading[1] * z[2]) - (currentHeading[2] * z[1]));
        x[1] = ((currentHeading[2] * z[0]) - (currentHeading[0] * z[2]));
        x[2] = ((currentHeading[0] * z[1]) - (currentHeading[1] * z[0]));

        x.normalize();
        //      z = x^currentHeading;
        z[0] = ((x[1] * currentHeading[2]) - (x[2] * currentHeading[1]));
        z[1] = ((x[2] * currentHeading[0]) - (x[0] * currentHeading[2]));
        z[2] = ((x[0] * currentHeading[1]) - (x[1] * currentHeading[0]));

        z.normalize();

        Matrix ret;
        ret.makeIdentity();
        //      ret.setRow(0, x);
        //      ret.setRow(1, currentHeading);
        //      ret.setRow(2, z);

        ret(0, 0) = x[0];
        ret(0, 1) = x[1];
        ret(0, 2) = x[2];
        ret(1, 0) = currentHeading[0];
        ret(1, 1) = currentHeading[1];
        ret(1, 2) = currentHeading[2];
        ret(2, 0) = z[0];
        ret(2, 1) = z[1];
        ret(2, 2) = z[2];

        ret.invert(ret);

        return ret;
        break;
    }
    default:
        Matrix ret;
        ret.makeIdentity();
        return ret;
    }

    // this part should never be reached
    return Matrix();
}

Matrix Interpolator::interpolateRotation(coCoord startCoord, Vec3 tangentOut, double startScale, ViewDesc *destination, Vec3 tangentIn, float lambda, RotationMode rotationMode)
{
    Matrix matrix_I;
    Matrix matrix_F;
    Quat rotQuat_I;
    Quat rotQuat_F;
    switch (rotationMode)
    {
    case QUATERNION:
    {
        //   matrix_I.makeCoord(&startCoord);
        startCoord.makeMat(matrix_I);
        //      matrix_I = startMatrix;
        rotQuat_I.set(matrix_I);

        //   matrix_F.makeCoord(&(destination->coord));
        destination->coord.makeMat(matrix_F);
        //      matrix_F = destination->getMatrix();
        rotQuat_F.set(matrix_F);

        // and interpolate rotation Quaternions using slerp
        Quat rotQuat_Interpolated;
        //      rotQuat_Interpolated.slerp(lambda, rotQuat_I, rotQuat_F);
        rotQuat_Interpolated.slerp(lambda, rotQuat_I, rotQuat_F);

        // get the interpolated rotation-matrix
        Matrix matrix_Interpolated;
        matrix_Interpolated.makeIdentity();
        //      matrix_Interpolated.makeQuat(rotQuat_Interpolated);
        matrix_Interpolated.makeRotate(rotQuat_Interpolated);

        //returnMatrix.postMult(matrix_Interpolated);
        return matrix_Interpolated;
        break;
    }
    case FOLLOWPATH:
    {
        //      matrix_I.makeCoord(&startCoord);
        startCoord.makeMat(matrix_I);
        //      matrix_I = startMatrix;
        //      matrix_F.makeCoord(&(destination->coord));
        destination->coord.makeMat(matrix_F);
        //      matrix_F = destination->getMatrix();

        //get P1, P4 in object coordinates
        matrix_I.invert(matrix_I);
        matrix_F.invert(matrix_F);
        Vec3 p1;
        Vec3 p4;
        //      matrix_I.getRow(3, p1);                                                                  // ?
        //      matrix_F.getRow(3, p4);
        p1 = matrix_I.getTrans();
        p4 = matrix_F.getTrans();

        //get P2, P3
        Vec3 p2;
        Vec3 p3;
        //      p2.xformVec(tangentOut, matrix_I);
        //      p3.xformVec(tangentIn, matrix_F);
        p2 = Matrix::transform3x3(tangentOut, matrix_I);
        p3 = Matrix::transform3x3(tangentIn, matrix_F);

        //      p2.add(p1, p2);
        //      p3.add(p3, p4);

        p2 += p1;
        p3 += p4;

        // exclude scale form calculation!
        p1 *= (1 / startScale);
        p2 *= (1 / startScale);
        //      p3 *= (1 / destination->_scale);
        //      p4 *= (1 / destination->_scale);
        p3 *= (1 / destination->getScale());
        p4 *= (1 / destination->getScale());

        Vec3 vecArray[4] = { p1, p2, p3, p4 };
        //      Vec3Array *vecArray = new Vec3Array;
        //      vecArray->push_back(p1);
        //      vecArray->push_back(p2);
        //      vecArray->push_back(p3);
        //      vecArray->push_back(p4);

        // get current tangent
        Vec3 currentHeading = casteljau(vecArray, lambda, 3);
        Vec3 z = Vec3(0, 0, 1);
        Vec3 x;

        //      x.cross(currentHeading, z);
        //      x.normalize();
        //      z.cross(x, currentHeading);
        //      z.normalize();

        //      x = currentHeading^z;
        x[0] = ((currentHeading[1] * z[2]) - (currentHeading[2] * z[1]));
        x[1] = ((currentHeading[2] * z[0]) - (currentHeading[0] * z[2]));
        x[2] = ((currentHeading[0] * z[1]) - (currentHeading[1] * z[0]));

        x.normalize();
        //      z = x^currentHeading;
        z[0] = ((x[1] * currentHeading[2]) - (x[2] * currentHeading[1]));
        z[1] = ((x[2] * currentHeading[0]) - (x[0] * currentHeading[2]));
        z[2] = ((x[0] * currentHeading[1]) - (x[1] * currentHeading[0]));

        z.normalize();

        Matrix ret;
        ret.makeIdentity();
        //      ret.setRow(0, x);
        //      ret.setRow(1, currentHeading);
        //      ret.setRow(2, z);

        ret(0, 0) = x[0];
        ret(0, 1) = x[1];
        ret(0, 2) = x[2];
        ret(1, 0) = currentHeading[0];
        ret(1, 1) = currentHeading[1];
        ret(1, 2) = currentHeading[2];
        ret(2, 0) = z[0];
        ret(2, 1) = z[1];
        ret(2, 2) = z[2];

        ret.invert(ret);

        return ret;
        break;
    }
    default:
        Matrix ret;
        ret.makeIdentity();
        return ret;
    }

    // this part should never be reached
    return Matrix();
}

/*
 * Interpolate Translation
 */
Matrix Interpolator::interpolateTranslation(
    Matrix startMatrix, Vec3 tangentOut, double startScale, Matrix destinationMatrix, double destinationScale, Vec3 tangentIn,
    float lambda, float interpolatedScale, TranslationMode translationMode)
{
    Matrix returnMatrix;

    Matrix matrix_I;
    //   matrix_I.makeCoord(&startCoord);
    //   startCoord.makeMat(matrix_I);
    matrix_I = startMatrix;

    Matrix matrix_F;
    //   matrix_F.makeCoord(&destinationCoord);
    //   destinationCoord.makeMat(matrix_F);
    matrix_F = destinationMatrix;

    //get P1, P4 in object coordinates
    matrix_I.invert(matrix_I);
    matrix_F.invert(matrix_F);
    Vec3 p1;
    Vec3 p4;
    p1 = matrix_I.getTrans();
    p4 = matrix_F.getTrans();

    //get P2, P3
    Vec3 p2;
    Vec3 p3;
    //   p2.xformVec(tangentOut, matrix_I);
    //   p3.xformVec(tangentIn, matrix_F);
    p2 = Matrix::transform3x3(tangentOut, matrix_I);
    p3 = Matrix::transform3x3(tangentIn, matrix_F);

    //   p2.add(p1, p2);
    //   p3.add(p3, p4);

    p2 += p1; // ?
    p3 += p4;

    // exclude scale form calculation!
    p1 *= (1 / startScale);
    p2 *= (1 / startScale);
    p3 *= (1 / destinationScale);
    p4 *= (1 / destinationScale);

    Vec3 vecArray[4] = { p1, p2, p3, p4 };

    Vec3 translationTemp;

    switch (translationMode)
    {
    case LINEAR_TRANSLATION:
        translationTemp = interpolateTranslationLinear(vecArray, lambda);
        break;
    case BEZIER:
        translationTemp = interpolateTranslationBezier(vecArray, lambda);
        break;
    }

    // create return matrix, scaled with interpolatedScale
    returnMatrix.makeTranslate(Vec3(translationTemp[0] * interpolatedScale,
                                    translationTemp[1] * interpolatedScale, translationTemp[2] * interpolatedScale)); // ?
    returnMatrix.invert(returnMatrix);

    return returnMatrix;
}

Matrix Interpolator::interpolateTranslation(
    coCoord startCoord, Vec3 tangentOut, double startScale, coCoord destinationCoord, double destinationScale, Vec3 tangentIn,
    float lambda, float interpolatedScale, TranslationMode translationMode)
{
    Matrix returnMatrix;

    Matrix matrix_I;
    //   matrix_I.makeCoord(&startCoord);
    startCoord.makeMat(matrix_I);
    //   matrix_I = startMatrix;

    Matrix matrix_F;
    //   matrix_F.makeCoord(&destinationCoord);
    destinationCoord.makeMat(matrix_F);
    //   matrix_F = destinationMatrix;

    //get P1, P4 in object coordinates
    matrix_I.invert(matrix_I);
    matrix_F.invert(matrix_F);
    Vec3 p1;
    Vec3 p4;
    p1 = matrix_I.getTrans();
    p4 = matrix_F.getTrans();

    //get P2, P3
    Vec3 p2;
    Vec3 p3;
    //   p2.xformVec(tangentOut, matrix_I);
    //   p3.xformVec(tangentIn, matrix_F);
    p2 = Matrix::transform3x3(tangentOut, matrix_I);
    p3 = Matrix::transform3x3(tangentIn, matrix_F);

    //   p2.add(p1, p2);
    //   p3.add(p3, p4);

    p2 += p1; // ?
    p3 += p4;

    // exclude scale form calculation!
    p1 *= (1 / startScale);
    p2 *= (1 / startScale);
    p3 *= (1 / destinationScale);
    p4 *= (1 / destinationScale);

    Vec3 vecArray[4] = { p1, p2, p3, p4 };

    Vec3 translationTemp;

    switch (translationMode)
    {
    case LINEAR_TRANSLATION:
        translationTemp = interpolateTranslationLinear(vecArray, lambda);
        break;
    case BEZIER:
        translationTemp = interpolateTranslationBezier(vecArray, lambda);
        break;
    }

    // create return matrix, scaled with interpolatedScale
    returnMatrix.makeTranslate(Vec3(translationTemp[0] * interpolatedScale,
                                    translationTemp[1] * interpolatedScale, translationTemp[2] * interpolatedScale)); // ?
    returnMatrix.invert(returnMatrix);

    return returnMatrix;
}

Vec3 Interpolator::interpolateTranslationLinear(Vec3 points[], float lambda)
{
    Vec3 translation;

    for (int i = 0; i < 3; i++)
    {
        // linear interpolation of translation
        translation[i] = points[3][i] + (1 - lambda) * (points[0][i] - points[3][i]);
    }
    return translation;
}

Vec3 Interpolator::interpolateTranslationBezier(Vec3 points[], float lambda)
{
    float arcLength[401][2]; // [parametricValue][arcLength]
    float arcLengthSum = 0.0;
    float d = 0.0025;
    int i = 0;
    Vec3 tmp;
    Vec3 last = points[0];
    Vec3 pointsCopy[4] = { points[0], points[1], points[2], points[3] };

    for (float parametricValue = 0; parametricValue < 1.0; parametricValue += d)
    {
        tmp = casteljau(pointsCopy, parametricValue, 4);
        Vec3 arc;
        //      arc.sub(last, tmp);                                                                           ?
        arc = last - tmp;
        arcLengthSum += arc.length();
        arcLength[i][0] = parametricValue;
        arcLength[i][1] = arcLengthSum;
        i++;
        // restore array
        pointsCopy[0] = points[0];
        pointsCopy[1] = points[1];
        pointsCopy[2] = points[2];
        pointsCopy[3] = points[3];
        last = tmp;
    }

    // find closest table entry
    i = 0;
    while (arcLength[i][1] < lambda * arcLengthSum)
    {
        i++;
    }

    // get low and high parametric values
    float low = arcLength[i - 1][0];
    float high = arcLength[i][0];

    // interpolate linear between low and high
    float newlambda;
    newlambda = low + ((lambda - low) / (high - low) * (high - low));

    // calculate final point
    return casteljau(points, newlambda, 4);
}

Vec3 Interpolator::casteljau(Vec3 points[], float lambda, int depth)
{
    for (int k = 1; k < depth; k++)
    { //    k1  k2  k3
        for (int i = 0; i < 4 - k; i++) //i0  p0
        { //        p01

            //         Vec3 t1s;                        //i1  p1      p012
            //         t1s.scale(1-lambda, points[i]);         //        p12      p0123
            //         Vec3 t2s;                        //i2  p2     p123
            //         t2s.scale(lambda, points[i+1]);         //        p23
            //         points[i].add(t1s, t2s);            //i3  p3                                                   ?

            Vec3 t1s = points[i]; //i1  p1      p012
            t1s *= (1 - lambda); //        p12      p0123
            Vec3 t2s = points[i + 1]; //i2  p2     p123
            t2s *= lambda; //        p23
            points[i] = t1s + t2s; //i3  p3
        }
    }
    // return tangent if depth = 3
    if (depth < 4)
    {
        Vec3 ret = points[1] - points[0];
        ret.normalize();
        return ret;
    }

    // return point if depth = 4
    return points[0];
}

/* called from ViewPoint::preFrame()
 *
 * t = current time    lambda
 * b = start value      0
 * c = change in value   1
 * d = duration         1
 */
float Interpolator::easingFunc(EasingFunction e, float t, float b, float c, float d)
{
    switch (e)
    {
    case LINEAR_EASING:
        return t;
    case QUADRIC_IN_OUT:
        t /= d / 2;
        if (t < 1)
            return c / 2 * t * t + b;
        t--;
        return -c / 2 * (t * (t - 2) - 1) + b;
    case QUADRIC_IN:
        t /= d;
        return c * t * t + b;
    case QUADRIC_OUT:
        t /= d;
        return -c * t * (t - 2) + b;

    case CUBIC_IN_OUT:
        t /= d / 2;
        if (t < 1)
            return c / 2 * t * t * t + b;
        t -= 2;
        return c / 2 * (t * t * t + 2) + b;
    case CUBIC_IN:
        t /= d;
        return c * t * t * t + b;
    case CUBIC_OUT:
        t /= d;
        t--;
        return c * (t * t * t + 1) + b;

    case QUARTIC_IN_OUT:
        t /= d / 2;
        if (t < 1)
            return c / 2 * t * t * t * t + b;
        t -= 2;
        return -c / 2 * (t * t * t * t - 2) + b;
    case QUARTIC_IN:
        t /= d;
        return c * t * t * t * t + b;
    case QUARTIC_OUT:
        t /= d;
        t--;
        return -c * (t * t * t * t - 1) + b;

    default:
        return t;
    }
}
