/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// modeled after Performer pfMatrix
// BUT:
//      - angles are in rad
//      - rotations are axis followed by angle

#include "vrmlexport.h"

namespace vrml
{

class VRMLEXPORT VrmlMatrix
{
    friend class VrmlVec;

public:
    VrmlMatrix(float a00, float a01, float a02, float a03,
               float a10, float a11, float a12, float a13,
               float a20, float a21, float a22, float a23,
               float a30, float a31, float a32, float a33);
    void makeIdent();
    void makeTrans(float x, float y, float z);
    void makeScale(float x, float y, float z);
    void makeRot(float degrees, float x, float y, float z);

    void transpose(const VrmlMatrix &m);

    bool invertFull(const VrmlMatrix &m);
    void invertAff(const VrmlMatrix &m);

    VrmlMatrix &operator*=(VrmlMatrix &m);
    void mult(const VrmlMatrix &m1, const VrmlMatrix &m2);
    VrmlMatrix &preMult(VrmlMatrix &m);
    VrmlMatrix &postMult(VrmlMatrix &m);

private:
    float mat[4][4];
};

class VRMLEXPORT VrmlVec
{
public:
    VrmlVec(float x, float y, float z);
    void xformVec(const VrmlVec &v, const VrmlMatrix &m);
    void xformPt(const VrmlVec &v, const VrmlMatrix &m);
    void fullXformPt(const VrmlVec &v, const VrmlMatrix &m);

private:
    float x[3];
};
}
