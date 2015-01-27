/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MZ_LIB_CPP
#define MZ_LIB_CPP

#include <math.h>

bool Mz_is_positive_definite(UniSys *us, mat3 S, mat3 M)
{
    // according to Appendix A of "obj. def. of a vortex"

    // get eigenvalues of S
    vec3 eigenvalsS;
    bool allReal = (mat3eigenvalues(S, eigenvalsS) == 3);
    if (!allReal)
    {
        us->error("got complex eigenvalues for S");
        return false;
    }

    // sort eigenvalues according to Haller:
    // sign(s1) = sign(s2) != sign(s3)   and   |s1| >= |s2|
    if (eigenvalsS[0] * eigenvalsS[1] < 0)
    {
        if (!(eigenvalsS[0] * eigenvalsS[2] < 0))
        {
            double w = eigenvalsS[1];
            eigenvalsS[1] = eigenvalsS[2];
            eigenvalsS[2] = w;
        }
        else
        {
            double w = eigenvalsS[0];
            eigenvalsS[0] = eigenvalsS[2];
            eigenvalsS[2] = w;
        }
    }
    if (fabs(eigenvalsS[0]) < fabs(eigenvalsS[1]))
    {
        double w = eigenvalsS[0];
        eigenvalsS[0] = eigenvalsS[1];
        eigenvalsS[1] = w;
    }

    // get eigenvectors of S
    vec3 eigenvectsS[3];
    mat3realEigenvector(S, eigenvalsS[0], eigenvectsS[0]);
    mat3realEigenvector(S, eigenvalsS[1], eigenvectsS[1]);
    mat3realEigenvector(S, eigenvalsS[2], eigenvectsS[2]);
    vec3nrm(eigenvectsS[0], eigenvectsS[0]);
    vec3nrm(eigenvectsS[1], eigenvectsS[1]);
    vec3nrm(eigenvectsS[2], eigenvectsS[2]);

    mat3 Mh; // M in strain basis
    {
        mat3 E, ET, W;
        mat3setcols(E, eigenvectsS[0], eigenvectsS[1], eigenvectsS[2]);
        mat3trp(E, ET);
        mat3mul(M, E, W);
        mat3mul(ET, W, Mh);
    }

    // a
    double a = -eigenvalsS[0] / eigenvalsS[2];

    // S0
    double S0 = (Mh[0][0] * (1 - a) - Mh[1][1] * a);
    S0 = S0 * S0;
    S0 += 4 * Mh[0][1] * Mh[0][1] * a * (1 - a);

    double Mh02_2, Mh12_2, Mh01_2;
    Mh02_2 = Mh[0][2] * Mh[0][2];
    Mh12_2 = Mh[1][2] * Mh[1][2];
    Mh01_2 = Mh[0][1] * Mh[0][1];

    // A
    double A = 4 * sqrt(a) * (1 - a) * (Mh[0][2] * (Mh[0][0] * (1 - a) - Mh[1][1] * a) + 2 * a * Mh[0][1] * Mh[1][2]);

    double B = 4 * a * (Mh02_2 * (1 - a) * (1 - a) + (1 - a) * (a * Mh12_2 - Mh01_2))
               + 2 * a * ((Mh[0][0] * (1 - a) - Mh[1][1] * a) * (Mh[2][2] * (1 - a) + Mh[1][1]));

    double C = 4 * sqrt(a * a * a) * (1 - a) * (Mh[0][2] * (Mh[2][2] * (1 - a) + Mh[1][1]) - 2 * Mh[0][1] * Mh[1][2]);

    double D = a * a * ((Mh[2][2] * (1 - a) + M[1][1]) * (Mh[2][2] * (1 - a) + M[1][1]) - 4 * (1 - a) * Mh12_2);

    A /= S0;
    B /= S0;
    C /= S0;
    D /= S0;

    // poly := p^4 + Ap^3 + Bp^2 + Cp + D = 0
    // must not have real roots in [-1,1] for Mz being positive definite

    // scheme:
    // if sign(poly(-1)) != sign(poly(1)) -> zero inside [-1,1]
    // else {
    //   get extrema inside [-1,1]
    //   if there is an extremum with sign(extremum) != sign(poly(-1)) -> zero inside [-1,1]
    //   TODO: this does not handle the case when the extremum lies on p-axis
    // }

    double poly1neg, poly1;
    poly1neg = pow(-1.0, 4) + A * pow(-1.0, 3) + B * pow(-1.0, 2) + C * (-1) + D;
    poly1 = pow(1.0, 4) + A * pow(1.0, 3) + B * pow(1.0, 2) + C * (1) + D;

    if (poly1neg * poly1 < 0.0)
    {
        // different signs at -1 and 1 -> zero inside [-1,1]
        return false;
    }
    else
    {
        // get extrema inside [-1,1]

        // derivative:
        // 4p^3 + 3Ap^2 + 2Bp + C = 0

        vec3 a, r;
        a[2] = 3 * A / 4.0;
        a[1] = 2 * B / 4.0;
        a[0] = C / 4.0;

        int realRootNb = vec3cubicroots(a, r);

        // roots inside [-1,1]
        for (int i = 0; i < realRootNb; i++)
        {
            if ((r[i] >= -1) && (r[i] <= 1))
            {
                double ro = r[i];
                double polyRoot;
                polyRoot = pow(ro, 4) + A * pow(ro, 3) + B * pow(ro, 2) + C * ro + D;
                if (polyRoot * poly1 < 0.0)
                {
                    // different signs -> zero inside [-1,1]
                    return false;
                }
            }
        }

        return true;
    }
}

#endif
