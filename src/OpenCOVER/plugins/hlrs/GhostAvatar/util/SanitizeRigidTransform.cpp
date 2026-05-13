#include <algorithm>
#include <cmath>

#include <osg/Vec3>

#include "SanitizeRigidTransform.h"

double determinant3x3(const osg::Matrix &m)
{
    return m(0,0)*(m(1,1)*m(2,2)-m(1,2)*m(2,1))
         - m(0,1)*(m(1,0)*m(2,2)-m(1,2)*m(2,0))
         + m(0,2)*(m(1,0)*m(2,1)-m(1,1)*m(2,0));
}

osg::Matrix extractRotation(const osg::Matrix &m)
{
    
    auto noTrans = m;
    noTrans.setTrans(osg::Vec3(0,0,0));
    // Extract 3x3 block

    // Quick check: if already orthonormal (rotation) then return early
    {
        auto noTranstransposed = noTrans;
        noTranstransposed.transpose(noTrans);
        auto ATA = noTranstransposed * noTrans;

        // check deviation from identity
        double maxDev = 0.0;
        for (int r=0;r<3;++r)
            for (int c=0;c<3;++c)
                maxDev = std::max(maxDev, fabs(ATA(r,c) - (r==c ? 1.0 : 0.0)));
        if (maxDev < 1e-6) // already rotation
            return noTrans;
    }

    // Newton iteration for polar decomposition: X_{k+1} = 0.5 * (X_k + inv(transpose(X_k)))
    auto x = noTrans;

    const int maxIter = 12;
    for (int iter = 0; iter < maxIter; ++iter)
    {
        auto invX = osg::Matrix::inverse(x);
        // transpose(invX)
        auto Tinv = invX;
        Tinv.transpose(invX);

        double maxDiff = 0.0;
        for (int r=0;r<3;++r)
            for (int c=0;c<3;++c)
            {
                double next = 0.5 * (x(r,c) + Tinv(r,c));
                maxDiff = std::max(maxDiff, fabs(next - x(r,c)));
                x(r,c) = next;
            }
        if (maxDiff < 1e-9) break;
    }

    // Fix handedness if necessary
    double d = determinant3x3(x);
    if (d < 0.0)
        for (int c=0;c<3;++c) x(0,c) = -x(0,c);
    return x;
}

osg::Matrix sanitizeRigidTransform(const osg::Matrix &m)
{
    auto rot = extractRotation(m);
    rot.setTrans(m.getTrans());
    return rot;
}