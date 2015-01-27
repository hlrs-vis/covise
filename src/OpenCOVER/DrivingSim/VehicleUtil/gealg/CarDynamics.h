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
/// - Steering: same angle at front wheels -> use ackermann angles (A-Steering),
///             no static righting moment -> add inclination to steering axis (pay attention to steering offset)
/// - Geometry: position vector CoM to Wheel joint use common distances -> add real distances

#ifndef __CarDynamics_h
#define __CarDynamics_h

#include "RungeKutta.h"

#include "RigidBody.h"
#include "PacejkaMagicFormula.h"

#include <vector>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#define typeof decltype
#endif

namespace cardyn
{

static gealg::mv<3, 0x040201>::type g;
static gealg::mv<3, 0x040201>::type x;
static gealg::mv<3, 0x040201>::type y;
static gealg::mv<3, 0x040201>::type z;
static gealg::mv<1, 0x00>::type F_aull;
static gealg::mv<1, 0x00>::type one;
static gealg::mv<1, 0x07>::type i;

//Wheel positions in car body frame
static gealg::mv<3, 0x040201>::type r_wfl;
static gealg::mv<3, 0x040201>::type r_wfr;
static gealg::mv<3, 0x040201>::type r_wrl;
static gealg::mv<3, 0x040201>::type r_wrr;

static gealg::mv<1, 0x04>::type d_wfl;
static gealg::mv<1, 0x04>::type v_wfl;

static gealg::mv<2, 0x0600>::type q_w;

//Carbody
static double m_b = 1450.0;
static double r_b = 3.0;
//Sphere
static double I_b = 2.0 / 5.0 * m_b * r_b * r_b;

//Wheel
static double m_w = 20;
static double r_w = 0.325;
//static double I_w = m_w*r_w*r_w;
static double I_w = 2.3;
static double u_wn = 0.4;
static double v_wn = 1.3;
static double w_wn = 0.7;
//static double k_wf = 100000.0;
//static double k_wf = 17400.0;
//static double k_wr = 26100.0;
//static double d_wf = 2000.0;
//static double d_wf = 2600.0;
//static double d_wr = 2600.0;

//Braking system
static double mu_b = 0.135;
static double d_b = 0.01;

//Anti roll bar
static double k_arb = 50000;

//Clutch
static double k_cn = 1.5;

//Engine
static double a_e = -0.000862;
static double b_e = 0.83;
static double c_e = 400;
static double I_e = 0.5;
static double d_e = 0.5;

//Transmission
static std::vector<double> i_g(7);
static double i_a = 3.5;

//Ground
//static double k_g = 240000.0;
static double k_gf = 140000.0;
static double k_gr = 210000.0;
static double rdr = 1.0;
//static double d_g = 2.0*rdr*sqrt(m_w*k_g);

//following plane
static gealg::mv<1, 0x03>::type P_xy;

typedef tuple<
    //State:
    gealg::mv<3, 0x040201>::type, //Body position	//0
    gealg::mv<3, 0x040201>::type,
    gealg::mv<4, 0x06050300>::type, //Body rotation
    gealg::mv<3, 0x060503>::type,
    gealg::mv<1, 0x04>::type, //Spring damper compression, wheel front left
    gealg::mv<1, 0x04>::type,
    gealg::mv<1, 0x04>::type, //Spring damper compression, wheel front right
    gealg::mv<1, 0x04>::type,
    gealg::mv<1, 0x04>::type, //Spring damper compression, wheel rear left
    gealg::mv<1, 0x04>::type,
    gealg::mv<1, 0x04>::type, //Spring damper compression, wheel rear right //10
    gealg::mv<1, 0x04>::type,
    gealg::mv<1, 0x00>::type, //Wheel angular velocity, wheel front left
    gealg::mv<1, 0x00>::type, //Wheel angular velocity, wheel front right
    gealg::mv<1, 0x00>::type, //Wheel angular velocity, wheel rear left
    gealg::mv<1, 0x00>::type, //Wheel angular velocity, wheel rear right
    gealg::mv<1, 0x00>::type, //Engine speed
    //Helpers:
    gealg::mv<1, 0x00>::type, //Engine moment
    gealg::mv<1, 0x00>::type, //empty
    gealg::mv<3, 0x040201>::type, //wheel velocity in body frame
    gealg::mv<3, 0x040201>::type, //20
    gealg::mv<3, 0x040201>::type,
    gealg::mv<3, 0x040201>::type,
    gealg::mv<1, 0x04>::type, //ground normal force on wheel, front left
    gealg::mv<1, 0x04>::type, //ground normal force on wheel, front right
    gealg::mv<1, 0x04>::type, //ground normal force on wheel, rear left
    gealg::mv<1, 0x04>::type, //ground normal force on wheel, rear right
    gealg::mv<1, 0x04>::type, //Vertical spring damper force on wheel, front left
    gealg::mv<1, 0x04>::type, //Vertical spring damper force on wheel, front right
    gealg::mv<1, 0x04>::type, //Vertical spring damper force on wheel, rear left
    gealg::mv<1, 0x04>::type, //Vertical spring damper force on wheel, rear right //30
    gealg::mv<3, 0x030201>::type, //contact wrench (complex force: Fx, Fy, Mz) on wheel, front left
    gealg::mv<3, 0x030201>::type, //contact wrench (complex force: Fx, Fy, Mz) on wheel, front right
    gealg::mv<3, 0x030201>::type, //contact wrench (complex force: Fx, Fy, Mz) on wheel, rear left
    gealg::mv<3, 0x030201>::type, //contact wrench (complex force: Fx, Fy, Mz) on wheel, rear right
    //Input:
    gealg::mv<1, 0x04>::type, //Vertical wheel distance to ground, front left
    gealg::mv<1, 0x04>::type, //Vertical wheel distance to ground, front right
    gealg::mv<1, 0x04>::type, //Vertical wheel distance to ground, rear left
    gealg::mv<1, 0x04>::type, //Vertical wheel distance to ground, rear right
    gealg::mv<2, 0x0300>::type, //Vertical wheel rotor, front left
    gealg::mv<2, 0x0300>::type, //Vertical wheel rotor, front rear //40
    gealg::mv<1, 0x00>::type, //Transmission
    gealg::mv<1, 0x00>::type, //Gas pedal position [0.0-1.0]
    gealg::mv<1, 0x00>::type, //Brake shoe force
    gealg::mv<1, 0x00>::type, //Clutch coefficient
    //Following plane coeffs:
    gealg::mv<1, 0x00>::type,
    gealg::mv<1, 0x00>::type,
    gealg::mv<1, 0x00>::type,
    gealg::mv<1, 0x00>::type,
    gealg::mv<1, 0x00>::type,
    gealg::mv<1, 0x00>::type, //50
    gealg::mv<1, 0x00>::type,
    gealg::mv<1, 0x00>::type> StateVectorType;

static gealg::var<0, StateVectorType>::type p_b;
static gealg::var<1, StateVectorType>::type dp_b;
static gealg::var<2, StateVectorType>::type q_b;
static gealg::var<3, StateVectorType>::type w_b;
static gealg::var<4, StateVectorType>::type u_wfl;
static gealg::var<5, StateVectorType>::type du_wfl;
static gealg::var<6, StateVectorType>::type u_wfr;
static gealg::var<7, StateVectorType>::type du_wfr;
static gealg::var<8, StateVectorType>::type u_wrl;
static gealg::var<9, StateVectorType>::type du_wrl;
static gealg::var<10, StateVectorType>::type u_wrr;
static gealg::var<11, StateVectorType>::type du_wrr;
static gealg::var<12, StateVectorType>::type w_wfl;
static gealg::var<13, StateVectorType>::type w_wfr;
static gealg::var<14, StateVectorType>::type w_wrl;
static gealg::var<15, StateVectorType>::type w_wrr;
static gealg::var<16, StateVectorType>::type w_e;
//static gealg::var<17, StateVectorType>::type P_b;
//Helpers:
static gealg::var<17, StateVectorType>::type Me_e;
static gealg::var<18, StateVectorType>::type Mw_e;
static gealg::var<19, StateVectorType>::type dr_wfl;
static gealg::var<20, StateVectorType>::type dr_wfr;
static gealg::var<21, StateVectorType>::type dr_wrl;
static gealg::var<22, StateVectorType>::type dr_wrr;
static gealg::var<23, StateVectorType>::type Fz_wfl;
static gealg::var<24, StateVectorType>::type Fz_wfr;
static gealg::var<25, StateVectorType>::type Fz_wrl;
static gealg::var<26, StateVectorType>::type Fz_wrr;
static gealg::var<27, StateVectorType>::type Fsd_wfl;
static gealg::var<28, StateVectorType>::type Fsd_wfr;
static gealg::var<29, StateVectorType>::type Fsd_wrl;
static gealg::var<30, StateVectorType>::type Fsd_wrr;
static gealg::var<31, StateVectorType>::type W_wfl;
static gealg::var<32, StateVectorType>::type W_wfr;
static gealg::var<33, StateVectorType>::type W_wrl;
static gealg::var<34, StateVectorType>::type W_wrr;
//Input:
static gealg::var<35, StateVectorType>::type Dv_wfl;
static gealg::var<36, StateVectorType>::type Dv_wfr;
static gealg::var<37, StateVectorType>::type Dv_wrl;
static gealg::var<38, StateVectorType>::type Dv_wrr;
static gealg::var<39, StateVectorType>::type q_wfl;
static gealg::var<40, StateVectorType>::type q_wfr;
static gealg::var<41, StateVectorType>::type i_pt;
static gealg::var<42, StateVectorType>::type s_gp;
static gealg::var<43, StateVectorType>::type F_b;
static gealg::var<44, StateVectorType>::type k_c;
//Following plane coeffs:
static gealg::var<45, StateVectorType>::type k_Pp;
static gealg::var<46, StateVectorType>::type d_Pp;
static gealg::var<47, StateVectorType>::type k_Pq;
static gealg::var<48, StateVectorType>::type d_Pq;
static gealg::var<49, StateVectorType>::type k_wf;
static gealg::var<50, StateVectorType>::type k_wr;
static gealg::var<51, StateVectorType>::type d_wf;
static gealg::var<52, StateVectorType>::type d_wr;

typedef tuple<typeof(dp_b),
              typeof(grade<1>(((!q_b) * ((grade<1>((!q_wfl) * W_wfl * q_wfl + (!q_wfr) * W_wfr * q_wfr + W_wrl + W_wrr) + (Fsd_wfl + Fsd_wfr + Fsd_wrl + Fsd_wrr)) * (1.0 / m_b)) * q_b)) + g),
              typeof(q_b *w_b * 0.5),
              /*typeof(grade<2>(((!q_b)*( r_wfl*(Fsd_wfl+grade<1>((!a_s)*W_wfl*a_s)) + r_wfr*(Fsd_wfr+grade<1>((!a_s)*W_wfr*a_s))
                               + r_wrl*(Fsd_wrl+grade<1>(W_wrl)) + r_wrr*(Fsd_wrr+grade<1>(W_wrr))  )*(1.0/I_b))*q_b)),*/
              typeof(grade<2>((!q_b) * rigidbody::euler<590, 1730, 1950>(q_b * w_b * (!q_b),
                                                                         r_wfl * (Fsd_wfl + grade<1>((!q_wfl) * W_wfl * q_wfl) - (u_wfl - u_wfr) * k_arb) + r_wfr * (Fsd_wfr + grade<1>((!q_wfr) * W_wfr * q_wfr) + (u_wfl - u_wfr) * k_arb)
                                                                         + r_wrl * (Fsd_wrl + grade<1>(W_wrl)) + r_wrr * (Fsd_wrr + grade<1>(W_wrr))) * q_b)),
              typeof(du_wfl),
              typeof((Fsd_wfl + Fz_wfl) * (1.0 / m_w)),
              typeof(du_wfr),
              typeof((Fsd_wfr + Fz_wfr) * (1.0 / m_w)),
              typeof(du_wrl),
              typeof((Fsd_wrl + Fz_wrl) * (1.0 / m_w)),
              typeof(du_wrr),
              typeof((Fsd_wrr + Fz_wrr) * (1.0 / m_w)),
              typeof((element<0x01>(W_wfl) * (-r_w) - tanh(w_wfl * d_b) * mu_b * F_b) * (1.0 / I_w)),
              typeof((element<0x01>(W_wfr) * (-r_w) - tanh(w_wfl * d_b) * mu_b * F_b) * (1.0 / I_w)),
              typeof((element<0x01>(W_wrl) * (-r_w) - tanh(w_wfl * d_b) * mu_b * F_b + i_pt * (w_e - (w_wrl + w_wrr) * i_pt * 0.5) * (k_c * 0.5)) * (1.0 / I_w)),
              typeof((element<0x01>(W_wrr) * (-r_w) - tanh(w_wfl * d_b) * mu_b * F_b + i_pt * (w_e - (w_wrl + w_wrr) * i_pt * 0.5) * (k_c * 0.5)) * (1.0 / I_w)),
              typeof(Mw_e *(1.0 / I_e)),
              //Helpers:
              typeof(s_gp *(w_e *w_e *a_e + w_e * b_e + one * c_e)),
              typeof(Me_e - (w_e - (w_wrl + w_wrr) * i_pt * 0.5) * k_c - w_e * d_e),
              typeof(grade<1>(q_wfl *(q_b *dp_b *(!q_b) + r_wfl * q_b * w_b * (!q_b) - du_wfl) * (!q_wfl))),
              typeof(grade<1>(q_wfr *(q_b *dp_b *(!q_b) + r_wfr * q_b * w_b * (!q_b) - du_wfr) * (!q_wfr))),
              typeof(grade<1>(q_b *dp_b *(!q_b) + r_wrl * q_b * w_b * (!q_b) - du_wrl)),
              typeof(grade<1>(q_b *dp_b *(!q_b) + r_wrr * q_b * w_b * (!q_b) - du_wrr)),
              typeof(delfttyre97::Fz(Dv_wfl *(-1.0), part<1, 0x04>(dr_wfl) * (-1.0), k_gf, m_w, rdr)),
              typeof(delfttyre97::Fz(Dv_wfr *(-1.0), part<1, 0x04>(dr_wfr) * (-1.0), k_gf, m_w, rdr)),
              typeof(delfttyre97::Fz(Dv_wrl *(-1.0), part<1, 0x04>(dr_wrl) * (-1.0), k_gr, m_w, rdr)),
              typeof(delfttyre97::Fz(Dv_wrr *(-1.0), part<1, 0x04>(dr_wrr) * (-1.0), k_gr, m_w, rdr)),
              typeof((u_wfl * k_wf + du_wfl * d_wf) * (-1.0)),
              typeof((u_wfr * k_wf + du_wfr * d_wf) * (-1.0)),
              typeof((u_wrl * k_wr + du_wrl * d_wr) * (-1.0)),
              typeof((u_wrr * k_wr + du_wrr * d_wr) * (-1.0)),
              typeof(part<3, 0x030201>((!q_w) * delfttyre97::wrench(q_w * Fz_wfl * (!q_w),
                                                                    part<2, 0x0201>(q_w * dr_wfl * (!q_w)),
                                                                    part<2, 0x0201>(q_w * (dr_wfl - cardyn::x * w_wfl * r_w) * (!q_w)),
                                                                    one * (2.0)) * q_w)),
              typeof(part<3, 0x030201>((!q_w) * delfttyre97::wrench(q_w * Fz_wfr * (!q_w),
                                                                    part<2, 0x0201>(q_w * dr_wfr * (!q_w)),
                                                                    part<2, 0x0201>(q_w * (dr_wfr - cardyn::x * w_wfr * r_w) * (!q_w)),
                                                                    one * (2.0)) * q_w)),
              typeof(part<3, 0x030201>((!q_w) * delfttyre97::wrench(q_w * Fz_wrl * (!q_w),
                                                                    part<2, 0x0201>(q_w * dr_wrl * (!q_w)),
                                                                    part<2, 0x0201>(q_w * (dr_wrl - cardyn::x * w_wrl * r_w) * (!q_w)),
                                                                    one * (2.0)) * q_w)),
              typeof(part<3, 0x030201>((!q_w) * delfttyre97::wrench(q_w * Fz_wrr * (!q_w),
                                                                    part<2, 0x0201>(q_w * dr_wrr * (!q_w)),
                                                                    part<2, 0x0201>(q_w * (dr_wrr - cardyn::x * w_wrr * r_w) * (!q_w)),
                                                                    one * (2.0)) * q_w))> ExpressionVectorType;

static ExpressionVectorType getExpressionVector()
{
    g[2] = -9.81;
    x[0] = 1.0;
    y[1] = 1.0;
    z[2] = 1.0;
    i[0] = 1.0;
    one[0] = 1.0;

    /*r_wfl[0] = v_wn; r_wfl[1] = w_wn; r_wfl[2] = -u_wn;
      r_wfr[0] = v_wn; r_wfr[1] = -w_wn; r_wfr[2] = -u_wn;
      r_wrl[0] = -v_wn; r_wrl[1] = w_wn; r_wrl[2] = -u_wn;
      r_wrr[0] = -v_wn; r_wrr[1] = -w_wn; r_wrr[2] = -u_wn;*/

    r_wfl[0] = 1.410;
    r_wfl[1] = 0.747;
    r_wfl[2] = -0.4;
    r_wfr[0] = 1.410;
    r_wfr[1] = -0.747;
    r_wfr[2] = -0.4;
    r_wrl[0] = -0.940;
    r_wrl[1] = 0.812;
    r_wrl[2] = -0.4;
    r_wrr[0] = -0.940;
    r_wrr[1] = -0.812;
    r_wrr[2] = -0.4;

    q_w[0] = cos(0.5 * M_PI);
    q_w[1] = sin(0.5 * M_PI);

    i_g[0] = -3.6;
    i_g[1] = 0.0;
    i_g[2] = 3.6;
    i_g[3] = 2.19;
    i_g[4] = 1.41;
    i_g[5] = 1.0;
    i_g[6] = 0.83;

    P_xy[0] = 1.0;

    ExpressionVectorType exprVec(
        dp_b,
        grade<1>(((!q_b) * ((grade<1>((!q_wfl) * W_wfl * q_wfl + (!q_wfr) * W_wfr * q_wfr + W_wrl + W_wrr) + (Fsd_wfl + Fsd_wfr + Fsd_wrl + Fsd_wrr)) * (1.0 / m_b)) * q_b)) + g,
        q_b * w_b * 0.5,
        //grade<2>(((!q_b)*( r_wfl*(Fsd_wfl+grade<1>((!a_s)*W_wfl*a_s)) + r_wfr*(Fsd_wfr+grade<1>((!a_s)*W_wfr*a_s))
        //                            + r_wrl*(Fsd_wrl+grade<1>(W_wrl)) + r_wrr*(Fsd_wrr+grade<1>(W_wrr))  )*(1.0/I_b))*q_b),
        grade<2>((!q_b) * rigidbody::euler<590, 1730, 1950>(q_b * w_b * (!q_b),
                                                            r_wfl * (Fsd_wfl + grade<1>((!q_wfl) * W_wfl * q_wfl) - (u_wfl - u_wfr) * k_arb) + r_wfr * (Fsd_wfr + grade<1>((!q_wfr) * W_wfr * q_wfr) + (u_wfl - u_wfr) * k_arb)
                                                            + r_wrl * (Fsd_wrl + grade<1>(W_wrl)) + r_wrr * (Fsd_wrr + grade<1>(W_wrr))) * q_b),
        du_wfl,
        (Fsd_wfl + Fz_wfl) * (1.0 / m_w),
        du_wfr,
        (Fsd_wfr + Fz_wfr) * (1.0 / m_w),
        du_wrl,
        (Fsd_wrl + Fz_wrl) * (1.0 / m_w),
        du_wrr,
        (Fsd_wrr + Fz_wrr) * (1.0 / m_w),
        (element<0x01>(W_wfl) * (-r_w) - tanh(w_wfl * d_b) * mu_b * F_b) * (1.0 / I_w),
        (element<0x01>(W_wfr) * (-r_w) - tanh(w_wfl * d_b) * mu_b * F_b) * (1.0 / I_w),
        (element<0x01>(W_wrl) * (-r_w) - tanh(w_wfl * d_b) * mu_b * F_b + i_pt * (w_e - (w_wrl + w_wrr) * i_pt * 0.5) * (k_c * 0.5)) * (1.0 / I_w),
        (element<0x01>(W_wrr) * (-r_w) - tanh(w_wfl * d_b) * mu_b * F_b + i_pt * (w_e - (w_wrl + w_wrr) * i_pt * 0.5) * (k_c * 0.5)) * (1.0 / I_w),
        Mw_e * (1.0 / I_e),
        //Helpers:
        s_gp * (w_e * w_e * a_e + w_e * b_e + one * c_e),
        (Me_e - (w_e - (w_wrl + w_wrr) * i_pt * 0.5) * k_c - w_e * d_e),
        grade<1>(q_wfl * (q_b * dp_b * (!q_b) + r_wfl * q_b * w_b * (!q_b) - du_wfl) * (!q_wfl)),
        grade<1>(q_wfr * (q_b * dp_b * (!q_b) + r_wfr * q_b * w_b * (!q_b) - du_wfr) * (!q_wfr)),
        grade<1>(q_b * dp_b * (!q_b) + r_wrl * q_b * w_b * (!q_b) - du_wrl),
        grade<1>(q_b * dp_b * (!q_b) + r_wrr * q_b * w_b * (!q_b) - du_wrr),
        delfttyre97::Fz(Dv_wfl * (-1.0), part<1, 0x04>(dr_wfl) * (-1.0), 240000.0, 10.0, 1.0),
        delfttyre97::Fz(Dv_wfr * (-1.0), part<1, 0x04>(dr_wfr) * (-1.0), 240000.0, 10.0, 1.0),
        delfttyre97::Fz(Dv_wrl * (-1.0), part<1, 0x04>(dr_wrl) * (-1.0), 240000.0, 10.0, 1.0),
        delfttyre97::Fz(Dv_wrr * (-1.0), part<1, 0x04>(dr_wrr) * (-1.0), 240000.0, 10.0, 1.0),
        (u_wfl * k_wf + du_wfl * d_wf) * (-1.0),
        (u_wfr * k_wf + du_wfr * d_wf) * (-1.0),
        (u_wrl * k_wr + du_wrl * d_wr) * (-1.0),
        (u_wrr * k_wr + du_wrr * d_wr) * (-1.0),
        part<3, 0x030201>((!q_w) * delfttyre97::wrench(q_w * Fz_wfl * (!q_w),
                                                       part<2, 0x0201>(q_w * dr_wfl * (!q_w)),
                                                       part<2, 0x0201>(q_w * (dr_wfl - cardyn::x * w_wfl * r_w) * (!q_w)),
                                                       one * (2.0)) * q_w),
        part<3, 0x030201>((!q_w) * delfttyre97::wrench(q_w * Fz_wfr * (!q_w),
                                                       part<2, 0x0201>(q_w * dr_wfr * (!q_w)),
                                                       part<2, 0x0201>(q_w * (dr_wfr - cardyn::x * w_wfr * r_w) * (!q_w)),
                                                       one * (2.0)) * q_w),
        part<3, 0x030201>((!q_w) * delfttyre97::wrench(q_w * Fz_wrl * (!q_w),
                                                       part<2, 0x0201>(q_w * dr_wrl * (!q_w)),
                                                       part<2, 0x0201>(q_w * (dr_wrl - cardyn::x * w_wrl * r_w) * (!q_w)),
                                                       one * (2.0)) * q_w),
        part<3, 0x030201>((!q_w) * delfttyre97::wrench(q_w * Fz_wrr * (!q_w),
                                                       part<2, 0x0201>(q_w * dr_wrr * (!q_w)),
                                                       part<2, 0x0201>(q_w * (dr_wrr - cardyn::x * w_wrr * r_w) * (!q_w)),
                                                       one * (2.0)) * q_w));

    return exprVec;
}
}
#endif
