/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///----------------------------------
///Author: Florian Seybold, 2009
///www.hlrs.de
//----------------------------------
///
///TODO:
///

#ifndef __PacejkaMagicFormula_h
#define __PacejkaMagicFormula_h

#include "GeometricAlgebra.h"

namespace delfttyre97
{
static const double a0 = 1.799;
static const double a1 = 0.0;
static const double a2 = 1688.0; //Lateral coefficients
static const double a3 = 4140.0;
static const double a4 = 6.026;
static const double a5 = 0.0;
static const double a6 = -0.3589;
static const double a7 = 1.0;
static const double a8 = 0.0;
static const double a9 = -0.006111;
static const double a10 = -0.03224;
static const double a11 = 0.0;
static const double a12 = 0.0;
static const double a13 = 0.0;
static const double a14 = 0.0;
static const double b0 = 1.65;
static const double b1 = 0.0;
static const double b2 = 1688.0; //Longitudinal coefficients
static const double b3 = 0.0;
static const double b4 = 229.0;
static const double b5 = 0.0;
static const double b6 = 0.0;
static const double b7 = 0.0;
static const double b8 = -10.0;
static const double b9 = 0.0;
static const double b10 = 0.0;
static const double c0 = 2.068;
static const double c1 = -6.49;
static const double c2 = -21.85;
static const double c3 = 0.416;
static const double c4 = -21.31;
static const double c5 = 0.02942;
static const double c6 = 0.0;
static const double c7 = -1.197;
static const double c8 = 5.228;
static const double c9 = -14.84;
static const double c10 = 0.0;
static const double c11 = 0.0;
static const double c12 = -0.003736;
static const double c13 = 0.03891;
static const double c14 = 0.0;
static const double c15 = 0.0;
static const double c16 = 0.639;
static const double c17 = 1.693;

template <class ELOAD, class ECENTER, class ESLIP, class ECAMBER>
struct ContactWrench
{
    typedef typename ELOAD::MultivectorType MLOAD;
    typedef typename ECENTER::MultivectorType MCENTER;
    typedef typename ESLIP::MultivectorType MSLIP;
    typedef typename ECAMBER::MultivectorType MCAMBER;

    typedef gealg::Multivector<3, 0x030201> MultivectorType;

    ContactWrench(const ELOAD &l, const ECENTER &m, const ESLIP &h, const ECAMBER &c)
        : l_(l)
        , m_(m)
        , h_(h)
        , c_(c)
    {
    }

    template <uint8_t EB, class T>
    double element(const T &vars) const
    {
        return r_.template element<EB>(vars);
    }

    template <class T>
    void evaluate(const T &vars)
    {
        MLOAD l = l_(vars).e_;
        MCENTER m = m_(vars).e_;
        MSLIP h = h_(vars).e_;
        MCAMBER c = c_(vars).e_;

        static const uint8_t IFz = gealg::MultivectorElementBitmapSearch<MLOAD, 0x04>::index;
        double Fz = (IFz == MLOAD::num) ? 0.0 : l[IFz] / 1000.0;

        //(w_wfl*r_w*(~element<0x01>(dr_wfl))-cardyn::one)*sgn(element<0x01>(dr_wfl)),
        //atan2(element<0x02>(dr_wfl), element<0x01>(dr_wfl))
        double kau;
        double alpha;
        if (fabs(m[0]) > 0.0001)
        {
            kau = -h[0] / m[0] * (m[0] >= 0.0 ? 1.0 : -1.0) * 100;
            alpha = atan2(-m[1], m[0]) * 360 / (2.0 * M_PI);
        }
        else
        {
            kau = 0.0;
            alpha = 0.0;
        }

        /*double kau = (MSLIP::num==0 || (uint8_t)MSLIP::bitmap!=0) ? std::numeric_limits<double>::signaling_NaN() : m[0]*100.0;
         double alpha = (MSLIPANGLE::num==0 || (uint8_t)MSLIPANGLE::bitmap!=0) ? std::numeric_limits<double>::signaling_NaN() : h[0]*360/(2.0*M_PI);
         if(std::isnan(kau) || fabs(kau)==std::numeric_limits<double>::infinity()) kau=0.0; //kau=NaN most likely a singularity -> tyre stands still;
         if(std::isnan(alpha) || fabs(alpha)==std::numeric_limits<double>::infinity()) alpha=0.0; //kau=NaN most likely a singularity -> tyre is drifting laterally;*/
        double camber = (MCAMBER::num == 0 || (uint8_t)MCAMBER::bitmap != 0) ? std::numeric_limits<double>::signaling_NaN() : c[0];

        //longitudinal force
        double Cx = b0;
        double Dx = (b1 * Fz + b2) * Fz;
        double Bx = ((b3 * Fz * Fz + b4 * Fz) * exp(-b5 * Fz)) / (Cx * Dx);
        if (Bx != Bx)
        {
            Bx = 1e9;
        } //Applies when Fz=0 -> no load
        double Ex = b6 * Fz * Fz + b7 * Fz + b8;
        double Shx = b9 * Fz + b10;
        double Svx = 0.0;
        r_[0] = Dx * sin(Cx * atan(Bx * (1 - Ex) * (kau + Shx) + Ex * atan(Bx * (kau + Shx)))) + Svx;

        //lateral force
        double Cy = a0;
        double Dy = (a1 * Fz + a2) * Fz;
        double By = a3 * sin(2.0 * atan(Fz / a4)) * (1.0 - a5 * fabs(camber)) / (Cy * Dy);
        if (By != By)
        {
            By = 1e9;
        } //Applies when Fz=0 or a4=0
        double Ey = a6 * Fz + a7;
        double Shy = a8 * camber + a9 * Fz + a10;
        //double Svy=(a111*Fz+a112)*camber*Fz+a12*Fz+a13;
        double Svy = a12 * Fz + a13;
        r_[1] = Dy * sin(Cy * atan(By * (1.0 - Ey) * (alpha + Shy) + Ey * atan(By * (alpha + Shy)))) + Svy;

        //combining slip: cutting lateral force
        /*double Fy = r_[1]*sqrt(1-pow(r_[0]/(Dx+Svx),2));
         if(Fy==Fy) {
            r_[1] = Fy;
         }*/
        //combining slip: scaling Fx and Fy
        double Fxy = sqrt(r_[0] * r_[0] + r_[1] * r_[1]);
        double Fxy_max = sqrt(pow(Dx + Svx, 2) + pow(Dy + Svy, 2));
        if (Fxy > Fxy_max)
        {
            double r_xy = Fxy_max / Fxy;
            r_[0] *= r_xy;
            r_[1] *= r_xy;
        }

        //aligning moment
        double Cm = c0;
        double Dm = c1 * Fz * Fz + c2 * Fz;
        double Em = (c7 * Fz * Fz + c8 * Fz + c9) * (1 - c10 * fabs(camber));
        double Bm = ((c3 * Fz * Fz + c4 * Fz) * (1 - c6 * fabs(camber)) * exp(-c5 * Fz)) / (Cm * Dm);
        if (Bm != Bm)
        {
            Bm = 1e9;
        } //Applies when Fz=0
        double Shm = c11 * camber + c12 * Fz + c13;
        double Svm = (c14 * Fz * Fz + c15 * Fz) * camber + c16 * Fz + c17;

        r_[2] = Dm * sin(Cm * atan(Bm * (1.0 - Em) * (alpha + Shm) + Em * atan(Bm * (alpha + Shm)))) + Svm;
    }

    ELOAD l_;
    ECENTER m_;
    ESLIP h_;
    ECAMBER c_;

    MultivectorType r_;
};

template <class L, class M, class H, class C>
gealg::Expression<ContactWrench<gealg::Expression<L>, gealg::Expression<M>, gealg::Expression<H>, gealg::Expression<C> > >
wrench(const gealg::Expression<L> &l, const gealg::Expression<M> &m, const gealg::Expression<H> &h, const gealg::Expression<C> &c)
{
    typedef ContactWrench<gealg::Expression<L>, gealg::Expression<M>, gealg::Expression<H>, gealg::Expression<C> > ContactWrenchType;
    return gealg::Expression<ContactWrenchType>(ContactWrenchType(l, m, h, c));
}

template <class EDIS, class EVEL> //K=tyre spring constant, MASS=wheel mass, RDR=radial damping ratio, SI-units
struct NormalForce
{
    typedef typename EDIS::MultivectorType MDIS;
    typedef typename EVEL::MultivectorType MVEL;
    typedef gealg::Multivector<1, 0x04> MultivectorType;

    NormalForce(const EDIS &ed, const EVEL &ev, double k, double mass, double rdr)
        : ed_(ed)
        , ev_(ev)
        , k_(k)
        , d_(2.0 * rdr * sqrt(k * mass))
    {
    }

    template <uint8_t EB, class T>
    double element(const T &vars) const
    {
        return r_.template element<EB>(vars);
    }

    template <class T>
    void evaluate(const T &vars)
    {
        MDIS md = ed_(vars).e_;
        MVEL mv = ev_(vars).e_;

        static const uint8_t ID = gealg::MultivectorElementBitmapSearch<MDIS, 0x04>::index;
        static const uint8_t IV = gealg::MultivectorElementBitmapSearch<MVEL, 0x04>::index;

        double Dz = (ID == MDIS::num) ? 0.0 : md[ID]; //Displacement: positiv value -> tyre compressed
        double Vz = (IV == MDIS::num) ? 0.0 : mv[IV]; //Velocity of tyre: positiv value -> downwards

        if (Dz > 0.0)
        {
            r_[0] = -Dz * k_ - Vz * d_;
        }
        else
        {
            r_[0] = 0.0;
        }
    }

    EDIS ed_;
    EVEL ev_;
    double k_;
    double d_;

    MultivectorType r_;
};

template <class L, class H>
gealg::Expression<NormalForce<gealg::Expression<L>, gealg::Expression<H> > >
Fz(const gealg::Expression<L> &l, const gealg::Expression<H> &h, double k, double mass, double rdr)
{
    typedef NormalForce<gealg::Expression<L>, gealg::Expression<H> > NormalForceExpressionType;
    return gealg::Expression<NormalForceExpressionType>(NormalForceExpressionType(l, h, k, mass, rdr));
}
}
#endif
