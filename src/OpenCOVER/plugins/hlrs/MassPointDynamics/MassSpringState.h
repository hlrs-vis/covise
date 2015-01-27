/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MassSpringState_h
#define MassSpringState_h

#include <vector>

#include <cmath>

#include "LinAlgSupport.h"

struct MassSpringParameters
{
    MassSpringParameters(double kset, double dset, double mset, double lset)
        : k(kset)
        , d(dset)
        , m(mset)
        , l(lset)
    {
    }

    double k, d, m, l;
};

class MassSpringState
{
public:
    MassSpringState();
    MassSpringState(const Vec3d &, const Vec3d &, const MassSpringParameters &);

    const Vec3d &r() const;
    const Vec3d &v() const;
    const MassSpringParameters &params() const;

    MassSpringState operator+(const MassSpringState &) const;
    const MassSpringState &operator+=(const MassSpringState &);

    MassSpringState operator*(double) const;

    MassSpringState dstate(double) const;

    void addLink(MassSpringState &);

protected:
    Vec3d r_;
    Vec3d v_;
    const MassSpringParameters &params_;

    std::vector<MassSpringState *> linkVector;
};

#endif
