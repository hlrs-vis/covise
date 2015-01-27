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

//defintion of basisvectors, null basis, pseudoscalars, helper unit scalar
typedef gaalet::algebra<gaalet::signature<4, 1> > cm;
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

typedef cm::mv<1, 2, 4>::type Vector;
typedef cm::mv<1, 2, 4, 8, 0x10>::type Point;
typedef Point Sphere;
typedef cm::mv<1, 2, 4, 8, 0x10>::type Plane;

typedef cm::mv<0, 3, 5, 6>::type Rotor;

typedef cm::mv<0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type S_type;
typedef cm::mv<0x00, 0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x0f, 0x11, 0x12, 0x14, 0x17>::type D_type;

typedef std::tuple<Plane, //Ground plane, front lef
                   Plane, //Ground plane, front right
                   Plane, //Ground plane, rear left
                   Plane, //Ground plane, rear right
                   double, //Steering angle
                   double, //Transmission
                   double, //Gas pedal position [0.0-1.0]
                   double, //Brake shoe force
                   double //Clutch coefficient
                   > InputVector;

typedef std::tuple<D_type, //Displacement versor wheel hub front left
                   D_type, //Displacement versor wheel hub front right
                   D_type, //Displacement versor wheel hub rear left
                   D_type //Displacement versor wheel hub rear right
                   > OutputVector;

typedef std::tuple<D_type, //Body displacement
                   S_type, //Screw velocity
                   double,
                   double,
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
                  OutputVector &output_,
                  const magicformula2004::ContactWrench &tyre_fl_,
                  const magicformula2004::ContactWrench &tyre_fr_,
                  const magicformula2004::ContactWrench &tyre_rl_,
                  const magicformula2004::ContactWrench &tyre_rr_)
        : input(input_)
        , output(output_)
        , tyre_fl(tyre_fl_)
        , tyre_fr(tyre_fr_)
        , tyre_rl(tyre_rl_)
        , tyre_rr(tyre_rr_)
    {
        g[0] = -9.81;

        cm::mv<1, 2, 4>::type x_wfl = { 1.410, 0.747, -0.4 };
        r_wfl = x_wfl + 0.5 * (x_wfl & x_wfl) * einf + e0;
        cm::mv<1, 2, 4>::type x_wfr = { 1.410, -0.747, -0.4 };
        r_wfr = x_wfr + 0.5 * (x_wfr & x_wfr) * einf + e0;
        cm::mv<1, 2, 4>::type x_wrl = { -0.940, 0.812, -0.4 };
        r_wrl = x_wrl + 0.5 * (x_wrl & x_wrl) * einf + e0;
        cm::mv<1, 2, 4>::type x_wrr = { -0.940, -0.812, -0.4 };
        r_wrr = x_wrr + 0.5 * (x_wrr & x_wrr) * einf + e0;

        //q_w[0] = cos(0.5*M_PI); q_w[1] = sin(0.5*M_PI);
        R_w = exp(-0.5 * M_PI * e2 * e3);

        i_g.resize(7);
        i_g[0] = -3.6;
        i_g[1] = 0.0;
        i_g[2] = 3.6;
        i_g[3] = 2.19;
        i_g[4] = 1.41;
        i_g[5] = 1.0;
        i_g[6] = 0.83;

        R_n_wfl = exp((-0.5) * Ie * (M_PI * 0.05 * e1 + M_PI * 0.1 * e2 + 0.0 * e3));
        R_n_wfr = exp((-0.5) * Ie * (-M_PI * 0.05 * e1 + M_PI * 0.1 * e2 + 0.0 * e3));
        R_n_wrl = exp((-0.5) * Ie * (M_PI * 0.05 * e1 + 0.0 * e2 + 0.0 * e3));
        R_n_wrr = exp((-0.5) * Ie * (-M_PI * 0.05 * e1 + 0.0 * e2 + 0.0 * e3));

        //car body wishbone joints
        auto x_wbf_fl = 1.638 * e1 + 0.333 * e2 + (-0.488) * e3;
        p_wbf_fl = x_wbf_fl + 0.5 * (x_wbf_fl & x_wbf_fl) * einf + e0;
        auto x_wbf_fr = 1.638 * e1 - 0.333 * e2 + (-0.488) * e3;
        p_wbf_fr = x_wbf_fr + 0.5 * (x_wbf_fr & x_wbf_fr) * einf + e0;
        auto x_wbf_rl = (-0.712) * e1 + 0.333 * e2 + (-0.488) * e3;
        p_wbf_rl = x_wbf_rl + 0.5 * (x_wbf_rl & x_wbf_rl) * einf + e0;
        auto x_wbf_rr = (-0.712) * e1 - 0.333 * e2 + (-0.488) * e3;
        p_wbf_rr = x_wbf_rr + 0.5 * (x_wbf_rr & x_wbf_rr) * einf + e0;

        auto x_wbr_fl = 1.4 * e1 + 0.333 * e2 + (-0.488) * e3;
        p_wbr_fl = x_wbr_fl + 0.5 * (x_wbr_fl & x_wbr_fl) * einf + e0;
        auto x_wbr_fr = 1.4 * e1 - 0.333 * e2 + (-0.488) * e3;
        p_wbr_fr = x_wbr_fr + 0.5 * (x_wbr_fr & x_wbr_fr) * einf + e0;
        auto x_wbr_rl = (-0.95) * e1 + 0.333 * e2 + (-0.488) * e3;
        p_wbr_rl = x_wbr_rl + 0.5 * (x_wbr_rl & x_wbr_rl) * einf + e0;
        auto x_wbr_rr = (-0.95) * e1 - 0.333 * e2 + (-0.488) * e3;
        p_wbr_rr = x_wbr_rr + 0.5 * (x_wbr_rr & x_wbr_rr) * einf + e0;

        //dome of mcpherson strut
        auto x_mps_fl = (1.335) * e1 + 0.473 * e2 + (-0.043) * e3;
        p_mps_fl = x_mps_fl + 0.5 * (x_mps_fl & x_mps_fl) * einf + e0;
        auto x_mps_fr = (1.335) * e1 - 0.473 * e2 + (-0.043) * e3;
        p_mps_fr = x_mps_fr + 0.5 * (x_mps_fr & x_mps_fr) * einf + e0;
        auto x_mps_rl = (-1.015) * e1 + 0.473 * e2 + (-0.043) * e3;
        p_mps_rl = x_mps_rl + 0.5 * (x_mps_rl & x_mps_rl) * einf + e0;
        auto x_mps_rr = (-1.015) * e1 - 0.473 * e2 + (-0.043) * e3;
        p_mps_rr = x_mps_rr + 0.5 * (x_mps_rr & x_mps_rr) * einf + e0;

        //steering link initial position
        auto x_steer0_fl = 1.3 * e1 + 0.141 * e2 + (-0.388) * e3;
        p_steer0_fl = x_steer0_fl + 0.5 * (x_steer0_fl & x_steer0_fl) * einf + e0;
        auto x_steer0_fr = 1.3 * e1 - 0.141 * e2 + (-0.388) * e3;
        p_steer0_fr = x_steer0_fr + 0.5 * (x_steer0_fr & x_steer0_fr) * einf + e0;
        auto x_steer0_rl = (-1.05) * e1 + 0.141 * e2 + (-0.388) * e3;
        p_steer0_rl = x_steer0_rl + 0.5 * (x_steer0_rl & x_steer0_rl) * einf + e0;
        auto x_steer0_rr = (-1.05) * e1 - 0.141 * e2 + (-0.388) * e3;
        p_steer0_rr = x_steer0_rr + 0.5 * (x_steer0_rr & x_steer0_rr) * einf + e0;
    }

    StateVector operator()(const double &t, const StateVector &oldState) const
    {
        //state
        const auto &D_b = std::get<0>(oldState);
        //const auto& p_b = std::get<0>(oldState);
        //const auto& dp_b = std::get<1>(oldState);
        const auto &V_b = std::get<1>(oldState);
        //const auto& q_b = std::get<2>(oldState);
        //const auto& w_b = std::get<3>(oldState);
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

        //input
        const auto &P_wfl = std::get<0>(input);
        const auto &P_wfr = std::get<1>(input);
        const auto &P_wrl = std::get<2>(input);
        const auto &P_wrr = std::get<3>(input);
        const double &steerAngle = std::get<4>(input);
        const double &i_pt = std::get<5>(input);
        const double &s_gp = std::get<6>(input);
        const double &f_b = std::get<7>(input);

        //output
        auto &D_wfl = std::get<0>(output);
        auto &D_wfr = std::get<1>(output);
        auto &D_wrl = std::get<2>(output);
        auto &D_wrr = std::get<3>(output);

        //Axle kinematics
        D_wfl = wheelVersor(u_wfl, steerAngle * 0.1, p_wbf_fl, p_wbr_fl, p_mps_fl, p_steer0_fl, wheel_left);
        D_wfr = wheelVersor(u_wfr, steerAngle * 0.1, p_wbf_fr, p_wbr_fr, p_mps_fr, p_steer0_fr, wheel_right);
        D_wrl = wheelVersor(u_wrl, 0.0, p_wbf_rl, p_wbr_rl, p_mps_rl, p_steer0_rl, wheel_left);
        D_wrr = wheelVersor(u_wrr, 0.0, p_wbf_rr, p_wbr_rr, p_mps_rr, p_steer0_rr, wheel_right);

        auto q_wfl = part<0, 3, 5, 6>(D_wfl);
        auto q_wfr = part<0, 3, 5, 6>(D_wfr);

        //Ackermann steering
        /*double cotSteerAngle = (r_wfl[0]-r_wrl[0])*(1.0/tan(steerAngle));
      double angleFL = atan(1.0/(cotSteerAngle - w_wn/(v_wn*2.0)));
      double angleFR = atan(1.0/(cotSteerAngle + w_wn/(v_wn*2.0)));
      gaalet::mv<0,3>::type q_wfl = {cos(angleFL*0.5), sin(angleFL*0.5)};
      gaalet::mv<0,3>::type q_wfr = {cos(angleFR*0.5), sin(angleFR*0.5)};

      //wheel velocity in body frame:
      auto dr_wfl = grade<1>(q_wfl*((V_b&r_wfl)-du_wfl*e3)*(!q_wfl));
      auto dr_wfr = grade<1>(q_wfr*((V_b&r_wfr)-du_wfr*e3)*(!q_wfr));
      auto dr_wrl = grade<1>((V_b&r_wrl)-du_wrl*e3);
      auto dr_wrr = grade<1>((V_b&r_wrr)-du_wrr*e3);*/

        //wheel rotors:
        /*auto R_wfl = R_n_wfl*exp(e2*e3*u_wfl*(-0.5));
      auto R_wfr = R_n_wfr*exp(e2*e3*u_wfr*(0.5));
      auto R_wrl = R_n_wrl*exp(e2*e3*u_wrl*(-0.5));
      auto R_wrr = R_n_wrr*exp(e2*e3*u_wrr*(0.5));*/

        //Suspension spring damper force:
        auto Fsd_wfl = (u_wfl * k_wf + du_wfl * d_wf) * (-1.0);
        auto Fsd_wfr = (u_wfr * k_wf + du_wfr * d_wf) * (-1.0);
        auto Fsd_wrl = (u_wrl * k_wr + du_wrl * d_wr) * (-1.0);
        auto Fsd_wrr = (u_wrr * k_wr + du_wrr * d_wr) * (-1.0);

        /*double Dv_wfl = eval((-1.0) * (P_wfl&grade<1>(D_b*D_wfl*e0*~(D_b*D_wfl))) * (1.0/sqrt(eval(P_wfl&P_wfl))));
      double Dv_wfr = eval((-1.0) * (P_wfr&grade<1>(D_b*D_wfr*e0*~(D_b*D_wfr))) * (1.0/sqrt(eval(P_wfr&P_wfr))));
      double Dv_wrl = eval((-1.0) * (P_wrl&grade<1>(D_b*D_wrl*e0*~(D_b*D_wrl))) * (1.0/sqrt(eval(P_wrl&P_wrl))));
      double Dv_wrr = eval((-1.0) * (P_wrr&grade<1>(D_b*D_wrr*e0*~(D_b*D_wrr))) * (1.0/sqrt(eval(P_wrr&P_wrr))));
      //Tyre forces and moments:
      auto W_wfl = ((!q_w)*tyre_fl( Dv_wfl, //distance difference with respect to camber angle?
            R_wfl,
            part<1,2,4,5>(q_w*(dr_wfl + w_wfl*e1*e3)*(!q_w)))*q_w);
      auto W_wfr = ((!q_w)*tyre_fr( Dv_wfr, //distance difference with respect to camber angle?
            R_wfr,
            part<1,2,4,5>(q_w*(dr_wfr + w_wfr*e1*e3)*(!q_w)))*q_w);
      auto W_wrl = ((!q_w)*tyre_rl( Dv_wrl, //distance difference with respect to camber angle?
            R_wrl,
            part<1,2,4,5>(q_w*(dr_wrl + w_wrl*e1*e3)*(!q_w)))*q_w);
      auto W_wrr = ((!q_w)*tyre_rr( Dv_wrr, //distance difference with respect to camber angle?
            R_wrr,
            part<1,2,4,5>(q_w*(dr_wrr + w_wrr*e1*e3)*(!q_w)))*q_w);*/
        //std::cout << "v_b: " << (V_b&e0) << ", v_b_wfl: " << grade<1>((~D_wfl*V_b*D_wfl)&e0) << std::endl;
        //std::cout << "V_b: " << (V_b) << ", V_b_wfl: " << grade<2>((~D_wfl*V_b*D_wfl)) << std::endl;
        //std::cout << "D_wfl: " << grade<1>(~D_wfl*e1*D_wfl) << std::endl;
        //std::cout << "R_wfl: " << grade<1>(~part<0,3,5,6>(D_wfl)*e1*part<0,3,5,6>(D_wfl)) << std::endl;
        auto V_wfl = grade<2>((~R_w) * ((~D_wfl) * V_b * D_wfl - Ie * (w_wfl)*e2 + einf * (-du_wfl) * e3) * R_w);
        auto V_wfr = grade<2>((~R_w) * ((~D_wfr) * V_b * D_wfr - Ie * (w_wfr)*e2 + einf * (-du_wfr) * e3) * R_w);
        auto V_wrl = grade<2>((~R_w) * ((~D_wrl) * V_b * D_wrl - Ie * (w_wrl)*e2 + einf * (-du_wrl) * e3) * R_w);
        auto V_wrr = grade<2>((~R_w) * ((~D_wrr) * V_b * D_wrr - Ie * (w_wrr)*e2 + einf * (-du_wrr) * e3) * R_w);
        auto F_wfl = grade<2>(R_w * tyre_fl(~(D_b * D_wfl * R_w) * P_wfl * (D_b * D_wfl * R_w), V_wfl) * ~R_w);
        auto F_wfr = grade<2>(R_w * tyre_fr(~(D_b * D_wfr * R_w) * P_wfr * (D_b * D_wfr * R_w), V_wfr) * ~R_w);
        auto F_wrl = grade<2>(R_w * tyre_rl(~(D_b * D_wrl * R_w) * P_wrl * (D_b * D_wrl * R_w), V_wrl) * ~R_w);
        auto F_wrr = grade<2>(R_w * tyre_rr(~(D_b * D_wrr * R_w) * P_wrr * (D_b * D_wrr * R_w), V_wrr) * ~R_w);
        /*auto W_wfl = (R_w * tyre_fl(~(D_b*D_wfl*R_w)*P_wfl*(D_b*D_wfl*R_w), V_wfl) * ~R_w);
      auto W_wfr = (R_w * tyre_fr(~(D_b*D_wfr*R_w)*P_wfr*(D_b*D_wfr*R_w), V_wfr) * ~R_w);
      auto W_wrl = (R_w * tyre_rl(~(D_b*D_wrl*R_w)*P_wrl*(D_b*D_wrl*R_w), V_wrl) * ~R_w);
      auto W_wrr = (R_w * tyre_rr(~(D_b*D_wrr*R_w)*P_wrr*(D_b*D_wrr*R_w), V_wrr) * ~R_w);*/
        /*auto W_wfl = (R_w * tyre_fl(~(D_b*D_wfl*R_w)*P_wfl*(D_b*D_wfl*R_w), part<1,2,4,5>(~R_w*(dr_wfl + w_wfl*e1*e3)*R_w)) * ~R_w);
      auto W_wfr = (R_w * tyre_fl(~(D_b*D_wfr*R_w)*P_wfr*(D_b*D_wfr*R_w), part<1,2,4,5>(~R_w*(dr_wfr + w_wfr*e1*e3)*R_w)) * ~R_w);
      auto W_wrl = (R_w * tyre_fl(~(D_b*D_wrl*R_w)*P_wrl*(D_b*D_wrl*R_w), part<1,2,4,5>(~R_w*(dr_wrl + w_wrl*e1*e3)*R_w)) * ~R_w);
      auto W_wrr = (R_w * tyre_fl(~(D_b*D_wrr*R_w)*P_wrr*(D_b*D_wrr*R_w), part<1,2,4,5>(~R_w*(dr_wrr + w_wrr*e1*e3)*R_w)) * ~R_w);*/

        auto f_u_wfl = ((einf & F_wfl) & e3) * (~e3);
        auto f_u_wfr = ((einf & F_wfr) & e3) * (~e3);
        auto f_u_wrl = ((einf & F_wrl) & e3) * (~e3);
        auto f_u_wrr = ((einf & F_wrr) & e3) * (~e3);

        auto F_wfl_b = part_type<S_type>(D_wfl * (F_wfl + (Fsd_wfl * e3 - f_u_wfl) * e0) * (~D_wfl));
        auto F_wfr_b = part_type<S_type>(D_wfr * (F_wfr + (Fsd_wfr * e3 - f_u_wfr) * e0) * (~D_wfr));
        auto F_wrl_b = part_type<S_type>(D_wrl * (F_wrl + (Fsd_wrl * e3 - f_u_wrl) * e0) * (~D_wrl));
        auto F_wrr_b = part_type<S_type>(D_wrr * (F_wrr + (Fsd_wrr * e3 - f_u_wrr) * e0) * (~D_wrr));
        auto F_b = F_wfl_b + F_wfr_b + F_wrl_b + F_wrr_b;

        auto w_b_b = eval(Ie & V_b);
        auto v_b_b = eval(V_b & e0);

        //Body acceleration:
        //auto ddp_b_b = eval(grade<1>((((grade<1>((!q_wfl)*part<1, 2>(W_wfl)*q_wfl+(!q_wfr)*part<1, 2>(W_wfr)*q_wfr+part<1, 2>(W_wrl)+part<1, 2>(W_wrr))+(Fsd_wfl+Fsd_wfr+Fsd_wrl+Fsd_wrr)*e3)*(1.0/m_b)))) + grade<1>((!part<0,3,5,6>(D_b))*g*part<0,3,5,6>(D_b)));
        auto R_b = part<0, 3, 5, 6>(D_b);
        auto ddp_b_b = eval(grade<1>((einf & F_b) * (1.0 / m_b) + grade<1>((~R_b) * g * part<0, 3, 5, 6>(R_b)) + 0.5 * Ie * w_b_b * v_b_b + (-0.5) * v_b_b * R_b * Ie * w_b_b * (~R_b)));

        double k_arb = this->k_arb;
        //cm::mv<1,2,4>::type t_b_b = (-1.0)*Ie*((part<1,2,4>(r_wfl)^(Fsd_wfl*e3+grade<1>((!q_wfl)*part<1,2>(W_wfl)*q_wfl)-(u_wfl-u_wfr)*e3*k_arb)) + (part<1,2,4>(r_wfr)^(Fsd_wfr*e3+grade<1>((!q_wfr)*part<1,2>(W_wfr)*q_wfr)+(u_wfl-u_wfr)*e3*k_arb)) + (part<1,2,4>(r_wrl)^(Fsd_wrl*e3+part<1,2>(W_wrl))) + (part<1,2,4>(r_wrr)^(Fsd_wrr*e3+part<1,2>(W_wrr))));
        auto t_b_b = eval(Ie & F_b);
        //cm::mv<1,2,4>::type t_b_b = (-1.0)*Ie*((part<1,2,4>(r_wfl)^(Fsd_wfl*z-(u_wfl-u_wfr)*z*k_arb)) + (part<1,2,4>(r_wfr)^(Fsd_wfr*z+(u_wfl-u_wfr)*z*k_arb)) + (part<1,2,4>(r_wrl)^(Fsd_wrl*z)) + (part<1,2,4>(r_wrr)^(Fsd_wrr*z+part<1,2>(W_wrr))));
        //cm::mv<1,2,4>::type t_b_b = (-1.0)*Ie*((part<1,2,4>(r_wfl)^(Fsd_wfl*z)) + (part<1,2,4>(r_wfr)^(Fsd_wfr*z)) + (part<1,2,4>(r_wrl)^(Fsd_wrl*z)) + (part<1,2,4>(r_wrr)^(Fsd_wrr*z)));
        cm::mv<1, 2, 4>::type dw_b_b;
        double In_1 = 590.0, In_2 = 1730.0, In_3 = 1950.0;
        dw_b_b[0] = (t_b_b[0] - (In_3 - In_2) * w_b_b[1] * w_b_b[2]) / In_1;
        dw_b_b[1] = (t_b_b[1] - (In_1 - In_3) * w_b_b[2] * w_b_b[0]) / In_2;
        dw_b_b[2] = (t_b_b[2] - (In_2 - In_1) * w_b_b[0] * w_b_b[1]) / In_3;

        auto dV_b = (-1.0) * Ie * dw_b_b + einf * ddp_b_b;

        StateVector newState(
            part_type<D_type>(D_b * V_b * 0.5),
            dV_b,
            0.0,
            0.0,
            du_wfl,
            (Fsd_wfl + (einf & F_wfl).element<0x04>()) * (1.0 / m_w),
            du_wfr,
            (Fsd_wfr + (einf & F_wfr).element<0x04>()) * (1.0 / m_w),
            du_wrl,
            (Fsd_wrl + (einf & F_wrl).element<0x04>()) * (1.0 / m_w),
            du_wrr,
            (Fsd_wrr + (einf & F_wrr).element<0x04>()) * (1.0 / m_w),
            ((einf & F_wfl).element<1>() * (-r_w) - tanh(w_wfl * d_b) * mu_b * f_b) * (1.0 / I_w),
            ((einf & F_wfr).element<1>() * (-r_w) - tanh(w_wfr * d_b) * mu_b * f_b) * (1.0 / I_w),
            ((einf & F_wrl).element<1>() * (-r_w) - tanh(w_wrl * d_b) * mu_b * f_b + i_pt * (w_e - (w_wrl + w_wrr) * i_pt * 0.5)) * (1.0 / I_w),
            ((einf & F_wrr).element<1>() * (-r_w) - tanh(w_wrr * d_b) * mu_b * f_b + i_pt * (w_e - (w_wrl + w_wrr) * i_pt * 0.5)) * (1.0 / I_w),
            (s_gp * (w_e * w_e * a_e + w_e * b_e + c_e) - (w_e - (w_wrl + w_wrr) * i_pt * 0.5) - w_e * d_e) * (1.0 / I_e));

        return std::move(newState);
    }

    D_type wheelVersor(double stroke, double steer, const Point &p_wbf, const Point &p_wbr, const Point &p_mps, const Point &p_steer0, const double side) const
    {
        stroke = std::max(-0.24, std::min(0.16, stroke));
        //steer = std::max(-0.09, std::min(0.05, steer));
        steer = std::max(-0.05, std::min(0.05, steer));
        //?dstroke = -(Mouse(2, 1, 1)-Pi)/Pi;
        //rod lengthes:
        //mcpherson strut: spring and wheel carrier
        double r_mps = 0.486 + stroke;
        //?dr_mps = dstroke;
        //wishbone
        double r_wbf = 0.365;
        double r_wbr = 0.23;

        //wishbone circle
        auto s_wbf = p_wbf - 0.5 * r_wbf * r_wbf * einf;
        auto s_wbr = p_wbr - 0.5 * r_wbr * r_wbr * einf;
        auto c_wb = s_wbf ^ s_wbr;

        //sphere of mcpherson strut: spring and wheel carrier
        Sphere s_mps = p_mps - 0.5 * r_mps * r_mps * einf;
        //?ds_mps = -dr_mps*r_mps*e;

        //wheel carrier lower joint
        auto Pp_wc = Ic * (c_wb ^ s_mps);
        //?dPp_wc = c_wb^ds_mps * I;
        Point p_wc = (Pp_wc + one * side * sqrt(eval(Pp_wc & Pp_wc))) * !(Pp_wc & einf);
        //?p_wc;
        //?dp_wc = (dPp_wc + 0.5*(1.0/sqrt(Pp_wc*Pp_wc)*(dPp_wc*Pp_wc+Pp_wc*dPp_wc)))*(1.0/(Pp_wc.einf)) - (Pp_wc + sqrt(Pp_wc.Pp_wc))*(dPp_wc.einf)*(1.0/((Pp_wc.einf)*(Pp_wc.einf)));
        //?dp_wc = (dPp_wc*(p_wc))°1;
        //?dp_wc = (dPp_wc.(p_wc));
        //:dp_wc;
        //?((Pp_wc.einf)*(Pp_wc.einf));
        //?dp_wc_R3 = (dp_wc^E)*E;
        //?dp_wc_R3.dp_wc_R3;
        //?Pp_wc.Pp_wc;
        //?Pp_wc*Pp_wc;

        //steering link
        double r_sl = 0.4;
        //steering arm:
        //from mcpherson struct dome
        double r_samps = sqrt(pow(r_mps - 0.15, 2) + pow(0.12, 2));
        //from wheel carrier lower joint
        double r_sawc = sqrt(pow(0.15, 2) + pow(0.12, 2));

        //Translation induced to steering link inner joint by steering wheel (e.g. via a cograil)
        auto T_steer = one - einf * (steer * e2) * 0.5;
        auto p_steer = grade<1>(T_steer * p_steer0 * (~T_steer));

        auto s_samps = p_mps - 0.5 * r_samps * r_samps * einf;

        auto s_sawc = p_wc - 0.5 * r_sawc * r_sawc * einf;

        auto s_steer = p_steer - 0.5 * r_sl * r_sl * einf;

        //steering arm
        auto Pp_sa = (s_sawc ^ s_samps ^ s_steer) * Ic;
        Point p_sa = (Pp_sa - one * side * sqrt(eval(Pp_sa & Pp_sa))) * !(Pp_sa & einf);

        //plane of wheel
        auto pi_w = eval((p_sa ^ p_mps ^ p_wc ^ einf) * Ic);
        auto mag_pi_w = magnitude(pi_w);
        pi_w = pi_w * !mag_pi_w;

        auto T_wb = one + 0.5 * einf * (p_wc - e0);
        auto n_wb = e2;
        auto pi_wb = e0 & (einf ^ (n_wb + pi_w));
        Rotor R_wb = pi_wb * n_wb;
        auto T_w = one + 0.5 * einf * (side * 0.2 * e2 + 0.1 * e3);
        Rotor R_wb0 = exp((-0.5) * side * (-M_PI * 0.06) * e2 * e3);
        D_type D_w = T_wb * R_wb * T_w * R_wb0;
        D_w = D_w * !magnitude(D_w);

        return std::move(D_w);
    }

    const InputVector &input;
    OutputVector &output;
    magicformula2004::ContactWrench tyre_fl;
    magicformula2004::ContactWrench tyre_fr;
    magicformula2004::ContactWrench tyre_rl;
    magicformula2004::ContactWrench tyre_rr;

    cm::mv<4>::type g;

    //Wheel positions in car body frame
    cm::mv<1, 2, 4, 8, 0x10>::type r_wfl;
    cm::mv<1, 2, 4, 8, 0x10>::type r_wfr;
    cm::mv<1, 2, 4, 8, 0x10>::type r_wrl;
    cm::mv<1, 2, 4, 8, 0x10>::type r_wrr;

    cm::mv<0, 6>::type R_w;

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
    //static constexpr double k_wf = 17400.0;
    static constexpr double k_wf = 30000.0;
    //static constexpr double k_wr = 26100.0;
    static constexpr double k_wr = 30000.0;
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

    //Wheel kinematics
    static constexpr double wheel_left = 1.0;
    static constexpr double wheel_right = -1.0;

    Point p_wbf_fl;
    Point p_wbf_fr;
    Point p_wbf_rl;
    Point p_wbf_rr;
    Point p_wbr_fl;
    Point p_wbr_fr;
    Point p_wbr_rl;
    Point p_wbr_rr;
    Point p_mps_fl;
    Point p_mps_fr;
    Point p_mps_rl;
    Point p_mps_rr;
    Point p_steer0_fl;
    Point p_steer0_fr;
    Point p_steer0_rl;
    Point p_steer0_rr;
};

} //end namespace cardyn

#endif
