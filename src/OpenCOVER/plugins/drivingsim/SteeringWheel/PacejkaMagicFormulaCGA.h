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

#ifndef __PacejkaMagicFormulaCGA_h
#define __PacejkaMagicFormulaCGA_h

#include "gaalet.h"

namespace delfttyre97CGA
{
typedef gaalet::algebra<gaalet::signature<4, 1> > cm;

typedef cm::mv<1, 2, 4>::type Vector;
typedef cm::mv<1, 2, 4, 8, 0x10>::type Point;
typedef Point Sphere;
typedef cm::mv<1, 2, 4, 8, 0x10>::type Plane;

typedef cm::mv<0, 3, 5, 6>::type Rotor;

typedef cm::mv<0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type S_type;
typedef cm::mv<0x00, 0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x0f, 0x11, 0x12, 0x14, 0x17>::type D_type;

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

struct ContactWrench
{
    typedef cm::mv<1, 2, 3, 4, 5, 6>::type result_wrench_t;

    ContactWrench(const double &R0, const double &k, const double &mass, const double &rdr)
        : R0_(R0)
        , k_(k)
        , d_(2.0 * rdr * sqrt(k * mass))
    {
    }

    S_type operator()(const Plane &P, const S_type &V) const
    {
        cm::mv<0x01>::type e1 = { 1.0 };
        cm::mv<0x02>::type e2 = { 1.0 };
        cm::mv<0x04>::type e3 = { 1.0 };
        cm::mv<0x08>::type ep = { 1.0 };
        cm::mv<0x10>::type em = { 1.0 };

        cm::mv<0x00>::type one = { 1.0 };

        cm::mv<0x08, 0x10>::type e0 = 0.5 * (em - ep);
        cm::mv<0x08, 0x10>::type einf = em + ep;

        cm::mv<0x18>::type E = ep * em;

        cm::mv<0x1f>::type Ic = e1 * e2 * e3 * ep * em;
        cm::mv<0x07>::type Ie = e1 * e2 * e3;

        Plane Pn = eval(P * (1.0 / sqrt(eval(P & P))));

        double r = eval((1.0) * (Pn & e0));
        double Dz = R0_ - r; //Displacement: positiv value -> tyre compressed
        double Vz = V.element<4>(); //Velocity of tyre: positiv value -> downwards

        double Fz = 0.0;
        if (Dz > 0.0)
        {
            //Fz = (-Dz*k_ - Vz*d_)/1000.0;
            Fz = (Dz * k_ + Vz * d_);
        }

        auto Vc = eval(V & e0);
        auto p_R0 = R0_ * e3 + 0.5 * R0_ * R0_ * einf + e0;
        auto Vs = eval(V & p_R0);

        //double kau = -Vs.element<1>()/(Vc.element<1>()+0.01)*(Vc.element<1>()>=0.0?1.0:-1.0)*100.0;
        double kau = -Vs.element<1>() / (Vc.element<1>() + 0.1) * (Vc.element<1>() >= 0.0 ? 1.0 : -1.0) * 0.001;
        double alpha = atan2(-Vc.element<2>(), Vc.element<1>()) * 360.0 / (2.0 * M_PI);
        /*if(fabs(m[0])>0.0001) {
            kau = -h[0]/m[0]*(m[0]>=0.0?1.0:-1.0)*100;
            alpha = atan2(-m[1], m[0])*360/(2.0*M_PI);
         }
         else {
            kau = 0.0;
            alpha = 0.0;
         }*/

        //double camber = (MCAMBER::num==0 || (uint8_t)MCAMBER::bitmap!=0) ? std::numeric_limits<double>::signaling_NaN() : c[0];
        //double camber = atan2(Pn.element<2>(), Pn.element<4>())*180.0/M_PI;
        double camber = 0.0;

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
        double F_x = Dx * sin(Cx * atan(Bx * (1.0 - Ex) * (kau + Shx) + Ex * atan(Bx * (kau + Shx)))) + Svx;

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
        double F_y = Dy * sin(Cy * atan(By * (1.0 - Ey) * (alpha + Shy) + Ey * atan(By * (alpha + Shy)))) + Svy;

        //combining slip: cutting lateral force
        /*double Fy = r_[1]*sqrt(1-pow(r_[0]/(Dx+Svx),2));
         if(Fy==Fy) {
            r_[1] = Fy;
         }*/
        //combining slip: scaling Fx and Fy
        double Fxy = sqrt(F_x * F_x + F_y * F_y);
        double Fxy_max = sqrt(pow(Dx + Svx, 2.0) + pow(Dy + Svy, 2.0));
        if (Fxy > Fxy_max)
        {
            double r_xy = Fxy_max / Fxy;
            F_x *= r_xy;
            F_y *= r_xy;
        }

        //aligning moment
        double Cm = c0;
        double Dm = c1 * Fz * Fz + c2 * Fz;
        double Em = (c7 * Fz * Fz + c8 * Fz + c9) * (1.0 - c10 * fabs(camber));
        double Bm = ((c3 * Fz * Fz + c4 * Fz) * (1.0 - c6 * fabs(camber)) * exp(-c5 * Fz)) / (Cm * Dm + 0.1);
        //if(Bm!=Bm) { Bm = 1e9; }   //Applies when Fz=0
        double Shm = c11 * camber + c12 * Fz + c13;
        double Svm = (c14 * Fz * Fz + c15 * Fz) * camber + c16 * Fz + c17;

        double M_z = Dm * sin(Cm * atan(Bm * (1.0 - Em) * (alpha + Shm) + Em * atan(Bm * (alpha + Shm)))) + Svm;

        double M_x = 0.0;
        double M_y = 0.0;

        S_type F = (F_x * e1 + F_y * e2 + Fz * e3) * e0; // - (M_x*e1+M_y*e2+M_z*e3)*Ie;
        //std::cout << "F_t: " << (F&einf) << ", F_r: " << (F&Ie) << std::endl;
        return std::move(F);
    }

    double R0_;
    double k_;
    double d_;
};
}
#endif
