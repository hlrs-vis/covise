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
#include "MagicFormula2004.h"

#include <vector>

namespace cardyn
{

typedef typeof(gealg::mv<6, 0x171412110f0c, 0x10>::type() + gealg::mv<6, 0x0a0906050300, 0x10>::type()) D_expr_t;
typedef D_expr_t::result_type D_type;
typedef typeof(gealg::mv<5, 0x1c1a191615, 0x10>::type() + gealg::mv<5, 0x130e0d0b07, 0x10>::type()) L_expr_t;
typedef L_expr_t::result_type L_type;
typedef std::pair<D_type, L_type> DL_type;

static gealg::mv<3, 0x040201>::type g;
static gealg::mv<1, 0x01>::type x;
static gealg::mv<1, 0x02>::type y;
static gealg::mv<1, 0x04>::type z;
static gealg::mv<1, 0x00>::type F_aull;
static gealg::mv<1, 0x00>::type one;
static gealg::mv<1, 0x07>::type i;

//Wheel positions in car body frame
static gealg::mv<3, 0x040201>::type r_wfl2;
static gealg::mv<3, 0x040201>::type r_wfr2;
static gealg::mv<3, 0x040201>::type r_wrl;
static gealg::mv<3, 0x040201>::type r_wrr;

static gealg::mv<1, 0x04>::type d_wfl;
static gealg::mv<1, 0x04>::type v_wfl;

static gealg::mv<2, 0x0600>::type q_w;

//Spring-Damper positions in car body frame
static gealg::mv<3, 0x040201>::type v_sfl;
static gealg::mv<3, 0x040201>::type r_sfl;
static gealg::mv<3, 0x040201>::type v_sfr;
static gealg::mv<3, 0x040201>::type r_sfr;

//Carbody
static double m_b = 253.0;
static double r_b = 3.0;
//Sphere
static double I_b = 2.0 / 5.0 * m_b * r_b * r_b;

//Wheel
static double m_w = 10;
static double r_w = 0.255;
//static double I_w = m_w*r_w*r_w;
static double I_w = 0.65025;
static double u_wn = 0.4;
static double v_wn = 1.3;
static double w_wn = 0.7;
//static double k_wf = 100000.0;
static double k_wf = 17400.0;
static double k_wr = 26100.0;
//static double d_wf = 2000.0;
static double d_wf = 2600.0;
static double d_wr = 2600.0;

static gealg::mv<4, 0x06050300>::type R_n_wfl;
static gealg::mv<4, 0x06050300>::type R_n_wfr;
static gealg::mv<4, 0x06050300>::type R_n_wrl;
static gealg::mv<4, 0x06050300>::type R_n_wrr;

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
static double k_Pb = 10.0;
static double d_Pb = 10.0;
static gealg::mv<1, 0x03>::type P_xy;

typedef tuple<
    //State:
    gealg::mv<3, 0x040201>::type, //Body position
    gealg::mv<3, 0x040201>::type,
    gealg::mv<4, 0x06050300>::type, //Body rotation
    gealg::mv<3, 0x060503>::type,
    gealg::mv<1, 0x00>::type, //Spring damper compression, wheel front left
    gealg::mv<1, 0x00>::type,
    gealg::mv<1, 0x00>::type, //Spring damper compression, wheel front right
    gealg::mv<1, 0x00>::type,
    gealg::mv<1, 0x04>::type, //Spring damper compression, wheel rear left
    gealg::mv<1, 0x04>::type,
    gealg::mv<1, 0x04>::type, //Spring damper compression, wheel rear right
    gealg::mv<1, 0x04>::type,
    gealg::mv<1, 0x05>::type, //Wheel angular velocity, wheel front left
    gealg::mv<1, 0x05>::type, //Wheel angular velocity, wheel front right
    gealg::mv<1, 0x05>::type, //Wheel angular velocity, wheel rear left
    gealg::mv<1, 0x05>::type, //Wheel angular velocity, wheel rear right
    gealg::mv<1, 0x00>::type, //Engine speed
    gealg::mv<4, 0x07060503>::type,
    gealg::mv<4, 0x07060503>::type,
    //Helpers:
    gealg::mv<3, 0x040201>::type, //wheel velocity in body frame
    gealg::mv<3, 0x040201>::type,
    gealg::mv<3, 0x040201>::type,
    gealg::mv<3, 0x040201>::type,
    gealg::mv<4, 0x06050300>::type, //wheel rotor, front left
    gealg::mv<4, 0x06050300>::type, //wheel rotor, front right
    gealg::mv<4, 0x06050300>::type, //wheel rotor, rear left
    gealg::mv<4, 0x06050300>::type, //wheel rotor, rear right
    gealg::mv<1, 0x00>::type, //Vertical spring damper force on wheel, front left
    gealg::mv<1, 0x00>::type, //Vertical spring damper force on wheel, front right
    gealg::mv<1, 0x04>::type, //Vertical spring damper force on wheel, rear left
    gealg::mv<1, 0x04>::type, //Vertical spring damper force on wheel, rear right
    gealg::mv<4, 0x04030201>::type, //contact wrench (complex force: Fx, Fy, Fz, Mz) on wheel, front left
    gealg::mv<4, 0x04030201>::type, //contact wrench (complex force: Fx, Fy, Fz, Mz) on wheel, front right
    gealg::mv<4, 0x04030201>::type, //contact wrench (complex force: Fx, Fy, Fz, Mz) on wheel, rear left
    gealg::mv<4, 0x04030201>::type, //contact wrench (complex force: Fx, Fy, Fz, Mz) on wheel, rear right
    //Input:
    gealg::mv<1, 0x04>::type, //Vertical wheel distance to ground, front left
    gealg::mv<1, 0x04>::type, //Vertical wheel distance to ground, front right
    gealg::mv<1, 0x04>::type, //Vertical wheel distance to ground, rear left
    gealg::mv<1, 0x04>::type, //Vertical wheel distance to ground, rear right
    gealg::mv<2, 0x0300>::type, //Vertical wheel rotor, front left
    gealg::mv<2, 0x0300>::type, //Vertical wheel rotor, front rear
    gealg::mv<1, 0x00>::type, //Transmission
    gealg::mv<1, 0x00>::type, //Gas pedal position [0.0-1.0]
    gealg::mv<1, 0x00>::type, //Brake shoe force
    gealg::mv<1, 0x00>::type, //Clutch coefficient
    D_type, //Displacement wheel front left
    gealg::mv<5, 0x1008040201>::type, //Force Plane front left
    gealg::mv<3, 0x040201>::type, //wheel position front left
    gealg::mv<3, 0x040201>::type, //normalenvektor force plane left
    D_type, //Displacement wheel front right
    gealg::mv<5, 0x1008040201>::type, //Force Plane wheel front right
    gealg::mv<3, 0x040201>::type, //wheel position front right
    gealg::mv<3, 0x040201>::type //normalenvektor force plane right
    > StateVectorType;
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
static gealg::var<17, StateVectorType>::type P_b;
static gealg::var<18, StateVectorType>::type dP_b;
//Helpers:
static gealg::var<19, StateVectorType>::type dr_wfl;
static gealg::var<20, StateVectorType>::type dr_wfr;
static gealg::var<21, StateVectorType>::type dr_wrl;
static gealg::var<22, StateVectorType>::type dr_wrr;
static gealg::var<23, StateVectorType>::type R_wfl;
static gealg::var<24, StateVectorType>::type R_wfr;
static gealg::var<25, StateVectorType>::type R_wrl;
static gealg::var<26, StateVectorType>::type R_wrr;
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
static gealg::var<45, StateVectorType>::type D_wfl;
static gealg::var<46, StateVectorType>::type P_wfl;
static gealg::var<47, StateVectorType>::type r_wfl;
static gealg::var<48, StateVectorType>::type n_nfl;
static gealg::var<49, StateVectorType>::type D_wfr;
static gealg::var<50, StateVectorType>::type P_wfr;
static gealg::var<51, StateVectorType>::type r_wfr;
static gealg::var<52, StateVectorType>::type n_nfr;

typedef tuple<typeof(dp_b),
              typeof(grade<1>(((!q_b) * ((grade<1>((!q_wfl) * (part<3, 0x040201>((W_wfl ^ n_nfl) * (~(n_nfl)))) * q_wfl + (!q_wfr) * (part<3, 0x040201>((W_wfr ^ n_nfr) * (~(n_nfr)))) * q_wfr + part<2, 0x0201>(W_wrl) + part<2, 0x0201>(W_wrr)) + (z * Fsd_wfl * 2.0 + (z * Fsd_wfr * 2.0) + Fsd_wrl + Fsd_wrr)) * (1.0 / m_b)) * q_b)) + g), //f_a=(W_wfl^P_wfl)*(~(P_wfl)) Kraftkomponente auf den Aufbau
              typeof(q_b *w_b * 0.5),
              typeof(grade<2>((!q_b) * rigidbody::euler<590, 1730, 1950>(q_b * w_b * (!q_b),
                                                                         ((z * Fsd_wfl * 2.0 * (v_sfl * (~(sqrt(v_sfl % v_sfl))))) * r_sfl) + (r_wfl * (grade<1>((!q_wfl) * (part<3, 0x040201>((W_wfl ^ n_nfl) * (~(n_nfl)))) * q_wfl) - (z * u_wfl * 2.0 - z * u_wfr * 2.0) * k_arb)) + ((z * Fsd_wfr * 2.0 * (v_sfr * (~(sqrt(v_sfr % v_sfr))))) * r_sfr) + (r_wfr * (grade<1>((!q_wfr) * (part<3, 0x040201>((W_wfr ^ n_nfr) * (~(n_nfr)))) * q_wfr) - (z * u_wfl * 2.0 - z * u_wfr * 2.0) * k_arb)) + r_wrl * (Fsd_wrl + part<2, 0x0201>(W_wrl)) + r_wrr * (Fsd_wrr + part<2, 0x0201>(W_wrr))) * q_b)),
              typeof(du_wfl),
              typeof((Fsd_wfl - grade<0>((((P_wfl % W_wfl) * (~(P_wfl))) * 0.5) % n_nfl)) * (1.0 / m_w)), //Kraftkomponente in Feder-Dämpfer Element
              typeof(du_wfr),
              typeof((Fsd_wfr - grade<0>((((P_wfr % W_wfr) * (~(P_wfr))) * 0.5) % n_nfr)) * (1.0 / m_w)),
              typeof(du_wrl),
              typeof((Fsd_wrl - part<1, 0x04>(W_wrl)) * (1.0 / m_w)),
              typeof(du_wrr),
              typeof((Fsd_wrr - part<1, 0x04>(W_wrr)) * (1.0 / m_w)),
              typeof((element<0x01>(W_wfl) * (-r_w) - tanh(element<0x05>(w_wfl) * d_b) * mu_b * F_b) * (1.0 / I_w) * gealg::mv<1, 0x05>::type(1.0)),
              typeof((element<0x01>(W_wfr) * (-r_w) - tanh(element<0x05>(w_wfl) * d_b) * mu_b * F_b) * (1.0 / I_w) * gealg::mv<1, 0x05>::type(1.0)),
              typeof((element<0x01>(W_wrl) * (-r_w) - tanh(element<0x05>(w_wfl) * d_b) * mu_b * F_b + i_pt * (w_e - element<0x05>(w_wrl + w_wrr) * i_pt * 0.5) * (k_c * 0.5)) * (1.0 / I_w) * gealg::mv<1, 0x05>::type(1.0)),
              typeof((element<0x01>(W_wrr) * (-r_w) - tanh(element<0x05>(w_wfl) * d_b) * mu_b * F_b + i_pt * (w_e - element<0x05>(w_wrl + w_wrr) * i_pt * 0.5) * (k_c * 0.5)) * (1.0 / I_w) * gealg::mv<1, 0x05>::type(1.0)),
              typeof((s_gp * (w_e * w_e * a_e + w_e * b_e + one * c_e) - (w_e - element<0x05>(w_wrl + w_wrr) * i_pt * 0.5) * k_c - w_e * d_e) * (1.0 / I_e)),
              typeof(dP_b),
              typeof((((one + p_b) ^ (grade<2>((!q_b) * P_xy * q_b))) - P_b) * magnitude(((one + p_b) ^ (grade<2>((!q_b) * P_xy * q_b))) - P_b) * k_Pb - dP_b * d_Pb),
              //Helpers:
              typeof(grade<1>(q_wfl *(q_b *dp_b *(!q_b) + r_wfl * q_b * w_b * (!q_b) - z * du_wfl * 2.0) * (!q_wfl))),
              typeof(grade<1>(q_wfr *(q_b *dp_b *(!q_b) + r_wfr * q_b * w_b * (!q_b) - z * du_wfr * 2.0) * (!q_wfr))),
              typeof(grade<1>(q_b *dp_b *(!q_b) + r_wrl * q_b * w_b * (!q_b) - du_wrl)),
              typeof(grade<1>(q_b *dp_b *(!q_b) + r_wrr * q_b * w_b * (!q_b) - du_wrr)),
              typeof(R_n_wfl),
              typeof(R_n_wfr),
              typeof(R_n_wrl *exp(y *u_wrl *(-0.5))),
              typeof(R_n_wrr *exp(y *u_wrr *(0.5))),
              typeof((u_wfl * k_wf + du_wfl * d_wf) * (-1.0)),
              typeof((u_wfr * k_wf + du_wfr * d_wf) * (-1.0)),
              typeof((u_wrl * k_wr + du_wrl * d_wr) * (-1.0)),
              typeof((u_wrr * k_wr + du_wrr * d_wr) * (-1.0)),
              typeof(part<4, 0x04030201>((!q_w) * magicformula2004::wrench(Dv_wfl, //distance difference with respect to camber angle?
                                                                           (!q_b) * part<4, 0x06050300>(D_wfl),
                                                                           part<4, 0x05040201>(q_w * (dr_wfl + w_wfl) * (!q_w)),
                                                                           magicformula2004::TyrePropertyPack(magicformula2004::TyrePropertyPack::TYRE_LEFT)) * q_w)),
              typeof(part<4, 0x04030201>((!q_w) * magicformula2004::wrench(Dv_wfr, //distance difference with respect to camber angle?
                                                                           (!q_b) * part<4, 0x06050300>(D_wfr),
                                                                           part<4, 0x05040201>(q_w * (dr_wfr + w_wfr) * (!q_w)),
                                                                           magicformula2004::TyrePropertyPack(magicformula2004::TyrePropertyPack::TYRE_RIGHT)) * q_w)),
              typeof(part<4, 0x04030201>((!q_w) * magicformula2004::wrench(Dv_wrl, //distance difference with respect to camber angle?
                                                                           R_wrl,
                                                                           part<4, 0x05040201>(q_w * (dr_wrl + w_wrl) * (!q_w)),
                                                                           magicformula2004::TyrePropertyPack(magicformula2004::TyrePropertyPack::TYRE_LEFT)) * q_w)),
              typeof(part<4, 0x04030201>((!q_w) * magicformula2004::wrench(Dv_wrr, //distance difference with respect to camber angle?
                                                                           R_wrr,
                                                                           part<4, 0x05040201>(q_w * (dr_wrr + w_wrr) * (!q_w)),
                                                                           magicformula2004::TyrePropertyPack(magicformula2004::TyrePropertyPack::TYRE_RIGHT)) * q_w))> ExpressionVectorType;

static ExpressionVectorType getExpressionVector()
{
    g[2] = -9.81;
    x[0] = 1.0;
    y[0] = 1.0;
    z[0] = 1.0;
    i[0] = 1.0;
    one[0] = 1.0;

    r_wfl2[0] = 0.8711;
    r_wfl2[1] = 0.607;
    r_wfl2[2] = -0.05032;
    r_wfr2[0] = 0.8711;
    r_wfr2[1] = -0.607;
    r_wfr2[2] = -0.05032;
    r_wrl[0] = -0.7639;
    r_wrl[1] = 0.571;
    r_wrl[2] = -0.05032;
    r_wrr[0] = -0.7639;
    r_wrr[1] = -0.571;
    r_wrr[2] = -0.05032;

    q_w[0] = cos(0.5 * M_PI);
    q_w[1] = sin(0.5 * M_PI);

    v_sfl[0] = 0.0;
    v_sfl[1] = -0.24141;
    v_sfl[2] = 0.09155;
    r_sfl[0] = 0.8711;
    r_sfl[1] = 0.05159;
    r_sfl[2] = 0.28023;
    v_sfr[0] = 0.0;
    v_sfr[1] = 0.24141;
    v_sfr[2] = 0.09155;
    r_sfr[0] = 0.8711;
    r_sfr[1] = -0.05159;
    r_sfr[2] = 0.28023;

    i_g[0] = -3.6;
    i_g[1] = 0.0;
    i_g[2] = 3.6;
    i_g[3] = 2.19;
    i_g[4] = 1.41;
    i_g[5] = 1.0;
    i_g[6] = 0.83;

    P_xy[0] = 1.0;

    /*R_n_wfl[0] = 1.0; R_n_wfl[1] = 0.0; R_n_wfl[2] = 0.0; R_n_wfl[3] = 0.0;
      R_n_wfr[0] = 1.0; R_n_wfr[1] = 0.0; R_n_wfr[2] = 0.0; R_n_wfr[3] = 0.0;
      R_n_wrl[0] = 1.0; R_n_wrl[1] = 0.0; R_n_wrl[2] = 0.0; R_n_wrl[3] = 0.0;
      R_n_wrr[0] = 1.0; R_n_wrr[1] = 0.0; R_n_wrr[2] = 0.0; R_n_wrr[3] = 0.0;*/
    R_n_wfl = exp((-0.5) * i * (M_PI * 0.05 * x + M_PI * 0.1 * y + 0.0 * z));
    R_n_wfr = exp((-0.5) * i * (-M_PI * 0.05 * x + M_PI * 0.1 * y + 0.0 * z));
    R_n_wrl = exp((-0.5) * i * (M_PI * 0.05 * x + 0.0 * y + 0.0 * z));
    R_n_wrr = exp((-0.5) * i * (-M_PI * 0.05 * x + 0.0 * y + 0.0 * z));
    /*R_n_wfl[0] = 0.9992; R_n_wfl[1] = 0.0; R_n_wfl[2] = 0.0; R_n_wfl[3] = 0.0399893;
      R_n_wfr[0] = 0.9992; R_n_wfr[1] = 0.0; R_n_wfr[2] = 0.0; R_n_wfr[3] = -0.0399893;
      R_n_wrl[0] = 0.9992; R_n_wrl[1] = 0.0; R_n_wrl[2] = 0.0; R_n_wrl[3] = 0.0399893;
      R_n_wrr[0] = 0.9992; R_n_wrr[1] = 0.0; R_n_wrr[2] = 0.0; R_n_wrr[3] = -0.0399893;*/

    ExpressionVectorType exprVec(
        dp_b,
        grade<1>(((!q_b) * ((grade<1>((!q_wfl) * (part<3, 0x040201>((W_wfl ^ n_nfl) * (~(n_nfl)))) * q_wfl + (!q_wfr) * (part<3, 0x040201>((W_wfr ^ n_nfr) * (~(n_nfr)))) * q_wfr + part<2, 0x0201>(W_wrl) + part<2, 0x0201>(W_wrr)) + (z * Fsd_wfl * 2.0 + (z * Fsd_wfr * 2.0) + Fsd_wrl + Fsd_wrr)) * (1.0 / m_b)) * q_b)) + g, //f_a=(W_wfl^P_wfl)*(~(P_wfl)) Kraftkomponente auf den Aufbau
        q_b * w_b * 0.5,
        grade<2>((!q_b) * rigidbody::euler<590, 1730, 1950>(q_b * w_b * (!q_b),
                                                            ((z * Fsd_wfl * 2.0 * (v_sfl * (~(sqrt(v_sfl % v_sfl))))) * r_sfl) + (r_wfl * (grade<1>((!q_wfl) * (part<3, 0x040201>((W_wfl ^ n_nfl) * (~(n_nfl)))) * q_wfl) - (z * u_wfl * 2.0 - z * u_wfr * 2.0) * k_arb)) + ((z * Fsd_wfr * 2.0 * (v_sfr * (~(sqrt(v_sfr % v_sfr))))) * r_sfr) + (r_wfr * (grade<1>((!q_wfr) * (part<3, 0x040201>((W_wfr ^ n_nfr) * (~(n_nfr)))) * q_wfr) - (z * u_wfl * 2.0 - z * u_wfr * 2.0) * k_arb)) + r_wrl * (Fsd_wrl + part<2, 0x0201>(W_wrl)) + r_wrr * (Fsd_wrr + part<2, 0x0201>(W_wrr))) * q_b),
        du_wfl,
        (Fsd_wfl - grade<0>((((P_wfl % W_wfl) * (~(P_wfl))) * 0.5) % n_nfl)) * (1.0 / m_w), //Kraftkomponente in Feder-Dämpfer Element
        du_wfr,
        (Fsd_wfr - grade<0>((((P_wfr % W_wfr) * (~(P_wfr))) * 0.5) % n_nfr)) * (1.0 / m_w),
        du_wrl,
        (Fsd_wrl - part<1, 0x04>(W_wrl)) * (1.0 / m_w),
        du_wrr,
        (Fsd_wrr - part<1, 0x04>(W_wrr)) * (1.0 / m_w),
        (element<0x01>(W_wfl) * (-r_w) - tanh(element<0x05>(w_wfl) * d_b) * mu_b * F_b) * (1.0 / I_w) * gealg::mv<1, 0x05>::type(1.0),
        (element<0x01>(W_wfr) * (-r_w) - tanh(element<0x05>(w_wfl) * d_b) * mu_b * F_b) * (1.0 / I_w) * gealg::mv<1, 0x05>::type(1.0),
        (element<0x01>(W_wrl) * (-r_w) - tanh(element<0x05>(w_wfl) * d_b) * mu_b * F_b + i_pt * (w_e - element<0x05>(w_wrl + w_wrr) * i_pt * 0.5) * (k_c * 0.5)) * (1.0 / I_w) * gealg::mv<1, 0x05>::type(1.0),
        (element<0x01>(W_wrr) * (-r_w) - tanh(element<0x05>(w_wfl) * d_b) * mu_b * F_b + i_pt * (w_e - element<0x05>(w_wrl + w_wrr) * i_pt * 0.5) * (k_c * 0.5)) * (1.0 / I_w) * gealg::mv<1, 0x05>::type(1.0),
        (s_gp * (w_e * w_e * a_e + w_e * b_e + one * c_e) - (w_e - element<0x05>(w_wrl + w_wrr) * i_pt * 0.5) * k_c - w_e * d_e) * (1.0 / I_e),
        dP_b,
        (((one + p_b) ^ (grade<2>((!q_b) * P_xy * q_b))) - P_b) * magnitude(((one + p_b) ^ (grade<2>((!q_b) * P_xy * q_b))) - P_b) * k_Pb - dP_b * d_Pb,
        //Helpers:
        grade<1>(q_wfl * (q_b * dp_b * (!q_b) + r_wfl * q_b * w_b * (!q_b) - z * du_wfl * 2.0) * (!q_wfl)),
        grade<1>(q_wfr * (q_b * dp_b * (!q_b) + r_wfr * q_b * w_b * (!q_b) - z * du_wfr * 2.0) * (!q_wfr)),
        grade<1>(q_b * dp_b * (!q_b) + r_wrl * q_b * w_b * (!q_b) - du_wrl),
        grade<1>(q_b * dp_b * (!q_b) + r_wrr * q_b * w_b * (!q_b) - du_wrr),
        R_n_wfl,
        R_n_wfr,
        R_n_wrl * exp(y * u_wrl * (-0.5)),
        R_n_wrr * exp(y * u_wrr * (0.5)),
        (u_wfl * k_wf + du_wfl * d_wf) * (-1.0),
        (u_wfr * k_wf + du_wfr * d_wf) * (-1.0),
        (u_wrl * k_wr + du_wrl * d_wr) * (-1.0),
        (u_wrr * k_wr + du_wrr * d_wr) * (-1.0),
        part<4, 0x04030201>((!q_w) * magicformula2004::wrench(Dv_wfl, //distance difference with respect to camber angle?
                                                              (!q_b) * part<4, 0x06050300>(D_wfl),
                                                              part<4, 0x05040201>(q_w * (dr_wfl + w_wfl) * (!q_w)),
                                                              magicformula2004::TyrePropertyPack(1)) * q_w),
        part<4, 0x04030201>((!q_w) * magicformula2004::wrench(Dv_wfr, //distance difference with respect to camber angle?
                                                              (!q_b) * part<4, 0x06050300>(D_wfr),
                                                              part<4, 0x05040201>(q_w * (dr_wfr + w_wfr) * (!q_w)),
                                                              magicformula2004::TyrePropertyPack(-1)) * q_w),
        part<4, 0x04030201>((!q_w) * magicformula2004::wrench(Dv_wrl, //distance difference with respect to camber angle?
                                                              R_wrl,
                                                              part<4, 0x05040201>(q_w * (dr_wrl + w_wrl) * (!q_w)),
                                                              magicformula2004::TyrePropertyPack(1)) * q_w),
        part<4, 0x04030201>((!q_w) * magicformula2004::wrench(Dv_wrr, //distance difference with respect to camber angle?
                                                              R_wrr,
                                                              part<4, 0x05040201>(q_w * (dr_wrr + w_wrr) * (!q_w)),
                                                              magicformula2004::TyrePropertyPack(-1)) * q_w));

    return exprVec;
}
}
#endif
