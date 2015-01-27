/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CAR_DYNAMICS_H
#define __CAR_DYNAMICS_H

#include "gaalet.h"
#include "MagicFormula2004.h"
#include <tuple>

namespace cardyn
{

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

typedef std::tuple<double, //Vertical wheel distance to ground, front lef
                   double, //Vertical wheel distance to ground, front right
                   double, //Vertical wheel distance to ground, rear left
                   double, //Vertical wheel distance to ground, rear right
                   double, //Steering angle
                   double, //Transmission
                   double, //Gas pedal position [0.0-1.0]
                   double, //Brake shoe force
                   double //Clutch coefficient
                   > InputVector;

typedef std::tuple<em::mv<1, 2, 4>::type, //Body position
                   em::mv<1, 2, 4>::type,
                   em::mv<0, 3, 5, 6>::type, //Body rotation
                   em::mv<3, 5, 6>::type,
                   double, //Spring damper compression, wheel front left
                   double,
                   double, //Spring damper compression, wheel front right
                   double,
                   double, //Spring damper compression, wheel rear left
                   double,
                   double, //Spring damper compression, wheel rear right
                   double,
                   double, //Wheel angular velocity, wheel front left
                   double, //Wheel angular velocity, wheel front right
                   double, //Wheel angular velocity, wheel rear left
                   double, //Wheel angular velocity, wheel rear right
                   double //Engine speed
                   > StateVector;

struct StateEquation
{
    StateEquation(const InputVector &input_,
                  const magicformula2004::ContactWrench &tyre_fl_,
                  const magicformula2004::ContactWrench &tyre_fr_,
                  const magicformula2004::ContactWrench &tyre_rl_,
                  const magicformula2004::ContactWrench &tyre_rr_)
        : input(input_)
        , tyre_fl(tyre_fl_)
        , tyre_fr(tyre_fr_)
        , tyre_rl(tyre_rl_)
        , tyre_rr(tyre_rr_)
    {
        g[2] = -9.81;
        x[0] = 1.0;
        y[0] = 1.0;
        z[0] = 1.0;
        I[0] = 1.0;
        one[0] = 1.0;

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

        i_g.resize(7);
        i_g[0] = -3.6;
        i_g[1] = 0.0;
        i_g[2] = 3.6;
        i_g[3] = 2.19;
        i_g[4] = 1.41;
        i_g[5] = 1.0;
        i_g[6] = 0.83;

        R_n_wfl = exp((-0.5) * I * (M_PI * 0.05 * x + M_PI * 0.1 * y + 0.0 * z));
        R_n_wfr = exp((-0.5) * I * (-M_PI * 0.05 * x + M_PI * 0.1 * y + 0.0 * z));
        R_n_wrl = exp((-0.5) * I * (M_PI * 0.05 * x + 0.0 * y + 0.0 * z));
        R_n_wrr = exp((-0.5) * I * (-M_PI * 0.05 * x + 0.0 * y + 0.0 * z));
    }

    StateVector operator()(const double &t, const StateVector &oldState) const
    {
        const auto &p_b = std::get<0>(oldState);
        const auto &dp_b = std::get<1>(oldState);
        const auto &q_b = std::get<2>(oldState);
        const auto &w_b = std::get<3>(oldState);
        const auto &u_wfl = std::get<4>(oldState);
        const auto &du_wfl = std::get<5>(oldState);
        const auto &u_wfr = std::get<6>(oldState);
        const auto &du_wfr = std::get<7>(oldState);
        const auto &u_wrl = std::get<8>(oldState);
        const auto &du_wrl = std::get<9>(oldState);
        const auto &u_wrr = std::get<10>(oldState);
        const auto &du_wrr = std::get<11>(oldState);
        const auto &w_wfl = std::get<12>(oldState);
        const auto &w_wfr = std::get<13>(oldState);
        const auto &w_wrl = std::get<14>(oldState);
        const auto &w_wrr = std::get<15>(oldState);
        const auto &w_e = std::get<16>(oldState);

        //Ackermann steering
        const double &steerAngle = std::get<4>(input);

        double cotSteerAngle = (r_wfl[0] - r_wrl[0]) * (1.0 / tan(steerAngle));
        double angleFL = atan(1.0 / (cotSteerAngle - w_wn / (v_wn * 2.0)));
        double angleFR = atan(1.0 / (cotSteerAngle + w_wn / (v_wn * 2.0)));
        gaalet::mv<0, 3>::type q_wfl = { cos(angleFL * 0.5), sin(angleFL * 0.5) };
        gaalet::mv<0, 3>::type q_wfr = { cos(angleFR * 0.5), sin(angleFR * 0.5) };

        //wheel velocity in body frame:
        auto dr_wfl = grade<1>(q_wfl * (q_b * dp_b * (!q_b) + r_wfl * q_b * w_b * (!q_b) - du_wfl * z) * (!q_wfl));
        auto dr_wfr = grade<1>(q_wfr * (q_b * dp_b * (!q_b) + r_wfr * q_b * w_b * (!q_b) - du_wfr * z) * (!q_wfr));
        auto dr_wrl = grade<1>(q_b * dp_b * (!q_b) + r_wrl * q_b * w_b * (!q_b) - du_wrl * z);
        auto dr_wrr = grade<1>(q_b * dp_b * (!q_b) + r_wrr * q_b * w_b * (!q_b) - du_wrr * z);

        //wheel rotors:
        auto R_wfl = R_n_wfl * exp(y * z * u_wfl * (-0.5));
        auto R_wfr = R_n_wfr * exp(y * z * u_wfr * (0.5));
        auto R_wrl = R_n_wrl * exp(y * z * u_wrl * (-0.5));
        auto R_wrr = R_n_wrr * exp(y * z * u_wrr * (0.5));

        //Suspension spring damper force:
        auto Fsd_wfl = (u_wfl * k_wf + du_wfl * d_wf) * (-1.0);
        auto Fsd_wfr = (u_wfr * k_wf + du_wfr * d_wf) * (-1.0);
        auto Fsd_wrl = (u_wrl * k_wr + du_wrl * d_wr) * (-1.0);
        auto Fsd_wrr = (u_wrr * k_wr + du_wrr * d_wr) * (-1.0);

        const double &Dv_wfl = std::get<0>(input);
        const double &Dv_wfr = std::get<1>(input);
        const double &Dv_wrl = std::get<2>(input);
        const double &Dv_wrr = std::get<3>(input);
        //Tyre forces and moments:
        auto W_wfl = ((!q_w) * tyre_fl(Dv_wfl, //distance difference with respect to camber angle?
                                       R_wfl,
                                       part<1, 2, 4, 5>(q_w * (dr_wfl + w_wfl * x * z) * (!q_w))) * q_w);
        auto W_wfr = ((!q_w) * tyre_fr(Dv_wfr, //distance difference with respect to camber angle?
                                       R_wfr,
                                       part<1, 2, 4, 5>(q_w * (dr_wfr + w_wfr * x * z) * (!q_w))) * q_w);
        auto W_wrl = ((!q_w) * tyre_rl(Dv_wrl, //distance difference with respect to camber angle?
                                       R_wrl,
                                       part<1, 2, 4, 5>(q_w * (dr_wrl + w_wrl * x * z) * (!q_w))) * q_w);
        auto W_wrr = ((!q_w) * tyre_rr(Dv_wrr, //distance difference with respect to camber angle?
                                       R_wrr,
                                       part<1, 2, 4, 5>(q_w * (dr_wrr + w_wrr * x * z) * (!q_w))) * q_w);

        //Body acceleration:
        auto ddp_b = grade<1>(((!q_b) * ((grade<1>((!q_wfl) * part<1, 2>(W_wfl) * q_wfl + (!q_wfr) * part<1, 2>(W_wfr) * q_wfr + part<1, 2>(W_wrl) + part<1, 2>(W_wrr)) + (Fsd_wfl + Fsd_wfr + Fsd_wrl + Fsd_wrr) * z) * (1.0 / m_b)) * q_b)) + g;
        auto w_b_I = eval(q_b * w_b * (!q_b));
        double k_arb = this->k_arb;
        em::mv<1, 2, 4>::type t_b_I = (-1.0) * I * (r_wfl * (Fsd_wfl * z + grade<1>((!q_wfl) * part<1, 2>(W_wfl) * q_wfl) - (u_wfl - u_wfr) * z * k_arb) + r_wfr * (Fsd_wfr * z + grade<1>((!q_wfr) * part<1, 2>(W_wfr) * q_wfr) + (u_wfl - u_wfr) * z * k_arb) + r_wrl * (Fsd_wrl * z + part<1, 2>(W_wrl)) + r_wrr * (Fsd_wrr * z + part<1, 2>(W_wrr)));
        em::mv<1, 2, 4>::type dw_b_I;
        double In_1 = 590.0, In_2 = 1730.0, In_3 = 1950.0;
        dw_b_I[0] = (t_b_I[0] - (In_3 - In_2) * w_b_I[1] * w_b_I[2]) / In_1;
        dw_b_I[1] = (t_b_I[1] - (In_1 - In_3) * w_b_I[2] * w_b_I[0]) / In_2;
        dw_b_I[2] = (t_b_I[2] - (In_2 - In_1) * w_b_I[0] * w_b_I[1]) / In_3;

        auto dw_b = grade<2>((!q_b) * (I * dw_b_I) * q_b);

        const double &i_pt = std::get<5>(input);
        const double &s_gp = std::get<6>(input);
        const double &F_b = std::get<7>(input);

        StateVector newState(
            dp_b,
            ddp_b,
            q_b * w_b * 0.5,
            dw_b,
            du_wfl,
            (Fsd_wfl - W_wfl.element<4>()) * (1.0 / m_w),
            du_wfr,
            (Fsd_wfr - W_wfr.element<4>()) * (1.0 / m_w),
            du_wrl,
            (Fsd_wrl - W_wrl.element<4>()) * (1.0 / m_w),
            du_wrr,
            (Fsd_wrr - W_wrr.element<4>()) * (1.0 / m_w),
            (W_wfl.element<1>() * (-r_w) - tanh(w_wfl * d_b) * mu_b * F_b) * (1.0 / I_w),
            (W_wfr.element<1>() * (-r_w) - tanh(w_wfr * d_b) * mu_b * F_b) * (1.0 / I_w),
            (W_wrl.element<1>() * (-r_w) - tanh(w_wrl * d_b) * mu_b * F_b + i_pt * (w_e - (w_wrl + w_wrr) * i_pt * 0.5)) * (1.0 / I_w),
            (W_wrr.element<1>() * (-r_w) - tanh(w_wrr * d_b) * mu_b * F_b + i_pt * (w_e - (w_wrl + w_wrr) * i_pt * 0.5)) * (1.0 / I_w),
            (s_gp * (w_e * w_e * a_e + w_e * b_e + c_e) - (w_e - (w_wrl + w_wrr) * i_pt * 0.5) - w_e * d_e) * (1.0 / I_e));

        return std::move(newState);
    }

    const InputVector &input;
    magicformula2004::ContactWrench tyre_fl;
    magicformula2004::ContactWrench tyre_fr;
    magicformula2004::ContactWrench tyre_rl;
    magicformula2004::ContactWrench tyre_rr;

    em::mv<1, 2, 4>::type g;
    em::mv<1>::type x;
    em::mv<2>::type y;
    em::mv<4>::type z;
    em::mv<0>::type F_aull;
    em::mv<0>::type one;
    em::mv<7>::type I;

    //Wheel positions in car body frame
    em::mv<1, 2, 4>::type r_wfl;
    em::mv<1, 2, 4>::type r_wfr;
    em::mv<1, 2, 4>::type r_wrl;
    em::mv<1, 2, 4>::type r_wrr;

    em::mv<0, 6>::type q_w;

    //Carbody
    static constexpr double m_b = 1450.0;
    static constexpr double r_b = 3.0;
    //Sphere
    //static constexpr double I_b = 2.0/5.0*m_b*r_b*r_b;

    //Wheel
    static constexpr double m_w = 20;
    static constexpr double r_w = 0.325;
    static constexpr double I_w = 2.3;
    static constexpr double u_wn = 0.4;
    static constexpr double v_wn = 1.3;
    static constexpr double w_wn = 0.7;
    static constexpr double k_wf = 17400.0;
    static constexpr double k_wr = 26100.0;
    static constexpr double d_wf = 2600.0;
    static constexpr double d_wr = 2600.0;
    gaalet::mv<0, 3, 5, 6>::type R_n_wfl;
    gaalet::mv<0, 3, 5, 6>::type R_n_wfr;
    gaalet::mv<0, 3, 5, 6>::type R_n_wrl;
    gaalet::mv<0, 3, 5, 6>::type R_n_wrr;

    //Braking system
    static constexpr double mu_b = 0.135;
    static constexpr double d_b = 0.01;

    //Anti roll bar
    //static constexpr double k_arb = 50000;
    static constexpr double k_arb = 100000;

    //Clutch
    static constexpr double k_cn = 1.5;

    //Engine
    static constexpr double a_e = -0.000862;
    static constexpr double b_e = 0.83;
    static constexpr double c_e = 400;
    static constexpr double I_e = 0.5;
    static constexpr double d_e = 0.5;

    //Transmission
    std::vector<double> i_g;
    static constexpr double i_a = 3.5;
};

} //end namespace cardyn

#endif
