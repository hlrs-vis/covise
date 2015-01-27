/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CAR_DYNAMICS_RTUS_H
#define __CAR_DYNAMICS_RTUS_H

#include "gaalet.h"
#include "MagicFormula2004.h"
#include "RungeKuttaClassic.h"
#include <tuple>

#include "Vehicle.h"
#include "VehicleDynamics.h"

#include "RoadSystem/RoadSystem.h"

namespace cardyn_rtus
{

//defintion of basisvectors, null basis, pseudoscalars, helper unit scalar
typedef gaalet::algebra<gaalet::signature<4, 1> > cm;
extern cm::mv<0x01>::type e1;
extern cm::mv<0x02>::type e2;
extern cm::mv<0x04>::type e3;
extern cm::mv<0x08>::type ep;
extern cm::mv<0x10>::type em;

extern cm::mv<0x00>::type one;

extern cm::mv<0x08, 0x10>::type e0;
extern cm::mv<0x08, 0x10>::type einf;

extern cm::mv<0x18>::type E;

extern cm::mv<0x1f>::type Ic;
extern cm::mv<0x07>::type Ie;

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
                   D_type, //Displacement versor wheel hub rear right
                   Vector, //Plane defined by roll center axis and tyre contact point wheel front left
                   Vector //wheel front right
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

        cm::mv<1, 2, 4>::type x_wfl = { 0.8711, 0.607, -0.05032 };
        r_wfl = x_wfl + 0.5 * (x_wfl & x_wfl) * einf + e0;
        cm::mv<1, 2, 4>::type x_wfr = { 0.8711, -0.607, -0.05032 };
        r_wfr = x_wfr + 0.5 * (x_wfr & x_wfr) * einf + e0;
        cm::mv<1, 2, 4>::type x_wrl = { -0.940, 0.812, -0.05032 };
        r_wrl = x_wrl + 0.5 * (x_wrl & x_wrl) * einf + e0;
        cm::mv<1, 2, 4>::type x_wrr = { -0.940, -0.812, -0.05032 };
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
        /*auto x_wbf_fl = 1.638*e1 + 0.333*e2 + (-0.488)*e3;
      p_wbf_fl = x_wbf_fl + 0.5*(x_wbf_fl&x_wbf_fl)*einf + e0;
      auto x_wbf_fr = 1.638*e1 - 0.333*e2 + (-0.488)*e3;
      p_wbf_fr = x_wbf_fr + 0.5*(x_wbf_fr&x_wbf_fr)*einf + e0;*/
        auto x_wbf_rl = (-0.712) * e1 + 0.333 * e2 + (-0.488) * e3;
        p_wbf_rl = x_wbf_rl + 0.5 * (x_wbf_rl & x_wbf_rl) * einf + e0;
        auto x_wbf_rr = (-0.712) * e1 - 0.333 * e2 + (-0.488) * e3;
        p_wbf_rr = x_wbf_rr + 0.5 * (x_wbf_rr & x_wbf_rr) * einf + e0;

        /*auto x_wbr_fl = 1.4*e1 + 0.333*e2 + (-0.488)*e3;
      p_wbr_fl = x_wbr_fl + 0.5*(x_wbr_fl&x_wbr_fl)*einf + e0;
      auto x_wbr_fr = 1.4*e1 - 0.333*e2 + (-0.488)*e3;
      p_wbr_fr = x_wbr_fr + 0.5*(x_wbr_fr&x_wbr_fr)*einf + e0;*/
        auto x_wbr_rl = (-0.95) * e1 + 0.333 * e2 + (-0.488) * e3;
        p_wbr_rl = x_wbr_rl + 0.5 * (x_wbr_rl & x_wbr_rl) * einf + e0;
        auto x_wbr_rr = (-0.95) * e1 - 0.333 * e2 + (-0.488) * e3;
        p_wbr_rr = x_wbr_rr + 0.5 * (x_wbr_rr & x_wbr_rr) * einf + e0;

        //dome of mcpherson strut
        /*auto x_mps_fl = (1.335)*e1 + 0.473*e2 + (-0.043)*e3;
      p_mps_fl = x_mps_fl + 0.5*(x_mps_fl&x_mps_fl)*einf + e0;
      auto x_mps_fr = (1.335)*e1 - 0.473*e2 + (-0.043)*e3;
      p_mps_fr = x_mps_fr + 0.5*(x_mps_fr&x_mps_fr)*einf + e0;*/
        auto x_mps_rl = (-1.015) * e1 + 0.473 * e2 + (-0.043) * e3;
        p_mps_rl = x_mps_rl + 0.5 * (x_mps_rl & x_mps_rl) * einf + e0;
        auto x_mps_rr = (-1.015) * e1 - 0.473 * e2 + (-0.043) * e3;
        p_mps_rr = x_mps_rr + 0.5 * (x_mps_rr & x_mps_rr) * einf + e0;

        //steering link initial position
        /*auto x_steer0_fl = 1.3*e1 + 0.141*e2 + (-0.388)*e3;
      p_steer0_fl = x_steer0_fl + 0.5*(x_steer0_fl&x_steer0_fl)*einf + e0;
      auto x_steer0_fr = 1.3*e1 - 0.141*e2 + (-0.388)*e3;
      p_steer0_fr = x_steer0_fr + 0.5*(x_steer0_fr&x_steer0_fr)*einf + e0;*/
        auto x_steer0_rl = (-1.05) * e1 + 0.141 * e2 + (-0.388) * e3;
        p_steer0_rl = x_steer0_rl + 0.5 * (x_steer0_rl & x_steer0_rl) * einf + e0;
        auto x_steer0_rr = (-1.05) * e1 - 0.141 * e2 + (-0.388) * e3;
        p_steer0_rr = x_steer0_rr + 0.5 * (x_steer0_rr & x_steer0_rr) * einf + e0;

        //position of spring-damper element
        auto v_sfl = 0.0 * e1 + 0.24141 * e2 + 0.09155 * e3;
        nv_sfl = v_sfl * (1.0 / (sqrt(eval(v_sfl & v_sfl))));
        auto v_sfr = 0.0 * e1 + (-0.24141) * e2 + 0.09155 * e3;
        nv_sfr = v_sfr * (1.0 / (sqrt(eval(v_sfr & v_sfr))));
        auto r_sfl = 0.8711 * e1 + 0.05159 * e2 + 0.28023 * e3;
        nr_sfl = r_sfl * (1.0 / (sqrt(eval(r_sfl & r_sfl))));
        auto r_sfr = 0.8711 * e1 + (-0.05159) * e2 + 0.28023 * e3;
        nr_sfr = r_sfr * (1.0 / (sqrt(eval(r_sfr & r_sfr))));
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
        auto &nrc_wfl = std::get<4>(output);
        auto &nrc_wfr = std::get<5>(output);

        //Axle kinematics
        Radaufhaengung_wfl(u_wfl, steerAngle * 0.1, D_wfl, nrc_wfl);
        Radaufhaengung_wfr(u_wfr, steerAngle * 0.1, D_wfr, nrc_wfr);

        D_wrl = wheelVersor(u_wrl, 0.0, p_wbf_rl, p_wbr_rl, p_mps_rl, p_steer0_rl, wheel_left);
        D_wrr = wheelVersor(u_wrr, 0.0, p_wbf_rr, p_wbr_rr, p_mps_rr, p_steer0_rr, wheel_right);

        auto q_wfl = part<0, 3, 5, 6>(D_wfl);
        auto q_wfr = part<0, 3, 5, 6>(D_wfr);

        std::cout << "D_wfl: " << D_wfl << ", nrc_wfl: " << nrc_wfl << std::endl;
        std::cout << "D_wrl: " << D_wrl << std::endl;

        std::cout << "P_wfl: " << ~(D_b * D_wfl * R_w) * P_wfl * (D_b * D_wfl * R_w) << std::endl;
        std::cout << "P_wrl: " << ~(D_b * D_wrl * R_w) * P_wrl * (D_b * D_wrl * R_w) << std::endl;
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

        //auto f_u_wfl = ((einf&F_wfl)&e3)*(~e3);
        //auto f_u_wfl = ((einf&F_wfl)&nrc_wfl)*(~nrc_wfl);
        auto f_u_wfl = ((einf & F_wfl) & nrc_wfl);
        //auto f_u_wfr = ((einf&F_wfr)&e3)*(~e3);
        auto f_u_wfr = ((einf & F_wfr) & nrc_wfr) * (~nrc_wfr);
        auto f_u_wrl = ((einf & F_wrl) & e3) * (~e3);
        auto f_u_wrr = ((einf & F_wrr) & e3) * (~e3);

        std::cout << "F_wfl: " << F_wfl << std::endl;
        std::cout << "nrc_wfl: " << nrc_wfl << std::endl;
        std::cout << "f_u_wfl: " << f_u_wfl << std::endl;
        std::cout << "(einf&F_wrl).element<0x04>(): " << (einf & F_wrl).element<0x04>() << std::endl;
        std::cout << "u_wfl: " << u_wfl << std::endl;

        //auto F_wfl_b = part_type<S_type>(D_wfl*(F_wfl + (Fsd_wfl*e3-f_u_wfl)*e0)*(~D_wfl));
        //auto F_wfl_b = part_type<S_type>(D_wfl*(F_wfl - f_u_wfl*e0)*(~D_wfl));
        auto F_wfl_b = ((einf & F_wfl) ^ nrc_wfl) * (~nrc_wfl);
        //auto F_wfr_b = part_type<S_type>(D_wfr*(F_wfr + (Fsd_wfr*e3-f_u_wfr)*e0)*(~D_wfr));
        auto F_wfr_b = ((einf & F_wfr) ^ nrc_wfr) * (~nrc_wfr);
        auto F_wrl_b = part_type<S_type>(D_wrl * (F_wrl + (Fsd_wrl * e3 - f_u_wrl) * e0) * (~D_wrl));
        auto F_wrr_b = part_type<S_type>(D_wrr * (F_wrr + (Fsd_wrr * e3 - f_u_wrr) * e0) * (~D_wrr));
        auto F_b = F_wfl_b + F_wfr_b + F_wrl_b + F_wrr_b;

        auto w_b_b = eval(Ie & V_b);
        auto v_b_b = eval(V_b & e0);

        //Body acceleration:
        //auto ddp_b_b = eval(grade<1>((((grade<1>((!q_wfl)*part<1, 2>(W_wfl)*q_wfl+(!q_wfr)*part<1, 2>(W_wfr)*q_wfr+part<1, 2>(W_wrl)+part<1, 2>(W_wrr))+(Fsd_wfl+Fsd_wfr+Fsd_wrl+Fsd_wrr)*e3)*(1.0/m_b)))) + grade<1>((!part<0,3,5,6>(D_b))*g*part<0,3,5,6>(D_b)));
        auto R_b = part<0, 3, 5, 6>(D_b);
        auto ddp_b_b = eval(grade<1>(((einf & F_b) + Fsd_wfl * nv_sfl + Fsd_wfr * nv_sfr) * (1.0 / m_b)) + grade<1>((~R_b) * g * part<0, 3, 5, 6>(R_b)) + 0.5 * Ie * w_b_b * v_b_b + (-0.5) * v_b_b * R_b * Ie * w_b_b * (~R_b));

        double k_arb = this->k_arb;
        //cm::mv<1,2,4>::type t_b_b = (-1.0)*Ie*((part<1,2,4>(r_wfl)^(Fsd_wfl*e3+grade<1>((!q_wfl)*part<1,2>(W_wfl)*q_wfl)-(u_wfl-u_wfr)*e3*k_arb)) + (part<1,2,4>(r_wfr)^(Fsd_wfr*e3+grade<1>((!q_wfr)*part<1,2>(W_wfr)*q_wfr)+(u_wfl-u_wfr)*e3*k_arb)) + (part<1,2,4>(r_wrl)^(Fsd_wrl*e3+part<1,2>(W_wrl))) + (part<1,2,4>(r_wrr)^(Fsd_wrr*e3+part<1,2>(W_wrr))));
        //auto t_b_b = eval(Ie&F_b);
        cm::mv<1, 2, 4>::type t_b_b = eval((Fsd_wfl * nv_sfl * nr_sfl) + (r_wfl * (F_wfl_b - (e3 * u_wfl * 2.0 - e3 * u_wfr * 2.0) * k_arb)) + (Fsd_wfr * nv_sfr * nr_sfr) + (r_wfr * (F_wfr_b - (e3 * u_wfl * 2.0 - e3 * u_wfr * 2.0) * k_arb)) + (Ie & (F_wrl_b + F_wrr_b)));
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
            (Fsd_wfl + f_u_wfl.element<0>() * 0.5) * (1.0 / m_w),
            du_wfr,
            (Fsd_wfr + f_u_wfr.element<0>() * 0.5) * (1.0 / m_w),
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

    void Radaufhaengung_wfl(double u_wfl, double steerAngle, D_type &D_wfl, Vector &nrc_wfl) const
    {
        u_wfl = std::max(-0.02, std::min(0.02, u_wfl));

        //Feder-Daempfer System
        auto r_fsl = e1 * 0.004 + (e2 * 0.05159 * (-1.0)) + e3 * 0.58555;
        auto p_fsl = eval(grade<1>(r_fsl + (r_fsl & r_fsl) * einf * 0.5 + e0));
        auto r_fbl = e1 * 0.004 + (e2 * 0.266 * (-1.0)) + e3 * 0.456;
        auto p_fbl = eval(grade<1>(r_fbl + (r_fbl & r_fbl) * einf * 0.5 + e0));
        //Querlenkerpunkt front lower frame left
        auto r_fll = e1 * 0.004 + (e2 * 0.195 * (-1.0)) + e3 * 0.097;
        auto p_fll = eval(grade<1>(r_fll + (r_fll & r_fll) * einf * 0.5 + e0));
        auto r_fll2 = e1 * 0.280 + (e2 * 0.195 * (-1.0)) + e3 * 0.097;
        auto p_fll2 = eval(grade<1>(r_fll2 + (r_fll2 & r_fll2) * einf * 0.5 + e0));

        double r_fsb = 0.04633;
        double r_fsd = 0.25772 - u_wfl;

        auto s_fsl = eval(grade<1>(p_fsl - einf * r_fsd * r_fsd * 0.5));
        auto s_fbsl = eval(grade<1>(p_fbl - einf * r_fsb * r_fsb * 0.5));
        auto c_fsbl = (s_fsl ^ s_fbsl);
        auto phi_fsd = (p_fll ^ p_fsl ^ p_fbl ^ einf) * Ic;
        auto Pp_fsbl = (phi_fsd ^ c_fsbl) * Ic;
        auto p_fsbl = eval(grade<1>((Pp_fsbl + one * (sqrt((Pp_fsbl & Pp_fsbl).element<0>()) * (-1.0))) * (!(Pp_fsbl & einf))));

        double r_fpb = 0.0764;
        double r_fpsb = 0.05116;

        auto s_fsbl = grade<1>(p_fsbl - einf * r_fpsb * r_fpsb * 0.5);
        auto s_fbl = grade<1>(p_fbl - einf * r_fpb * r_fpb * 0.5);
        auto c_fpbl = (s_fsbl ^ s_fbl);
        auto Pp_fpbl = (phi_fsd ^ c_fpbl) * Ic;
        auto p_fpbl = eval(grade<1>((Pp_fpbl + one * (sqrt((Pp_fpbl & Pp_fpbl).element<0>()) * (-1.0))) * (!(Pp_fpbl & einf))));

        //Querlenker

        double r_fp = 0.38418;
        double r_fl = 0.35726;

        auto s_fll = grade<1>(p_fll - einf * r_fl * r_fl * 0.5);
        auto s_fpbl = grade<1>(p_fpbl - einf * r_fp * r_fp * 0.5);
        auto c_flol = (s_fpbl ^ s_fll);
        auto Pp_flol = (phi_fsd ^ c_flol) * Ic;
        auto p_flol = eval(grade<1>((Pp_flol + one * (sqrt((Pp_flol & Pp_flol).element<0>()) * (-1.0))) * (!(Pp_flol & einf))));

        auto r_ful = e1 * 0.037 + (e2 * 0.288 * (-1.0)) + e3 * 0.261;
        auto p_ful = eval(grade<1>(r_ful + (r_ful & r_ful) * einf * 0.5 + e0));
        auto r_ful2 = e1 * 0.210 + (e2 * 0.288 * (-1.0)) + e3 * 0.261;
        auto p_ful2 = eval(grade<1>(r_ful2 + (r_ful2 & r_ful2) * einf * 0.5 + e0));

        double r_fo = 0.21921;
        double r_fu = 0.26086;

        //Punkte fuer Ebene oberer Querlenker
        auto r_phi1 = e1 * 0.037 + e2 * 0.0 + e3 * 0.0;
        auto p_phi1 = eval(grade<1>(r_phi1 + (r_phi1 & r_phi1) * einf * 0.5 + e0));
        auto r_phi2 = e1 * 0.037 + (e2 * (-1.0)) + e3 * 0.0;
        auto p_phi2 = eval(grade<1>(r_phi2 + (r_phi2 & r_phi2) * einf * 0.5 + e0));

        auto s_ful = grade<1>(p_ful - einf * r_fu * r_fu * 0.5);
        auto s_flol = grade<1>(p_flol - einf * r_fo * r_fo * 0.5);
        auto c_flul = (s_flol ^ s_ful);
        auto phi_fuo = (p_ful ^ p_phi1 ^ p_phi2 ^ einf) * Ic;
        auto Pp_fuol = (phi_fuo ^ c_flul) * Ic;
        auto p_fuol = eval(grade<1>((Pp_fuol + one * (sqrt((Pp_fuol & Pp_fuol).element<0>()) * (-1.0))) * (!(Pp_fuol & einf))));

        //Spurstange
        double steering = steerAngle * 11.4592; //Umrechnung vom Lenkwinkel auf Lenkgetriebeweg
        auto r_ftl = e1 * (-0.055) + e2 * ((0.204 * (-1.0)) + steering) + e3 * 0.101; //Anbindungspunkt tie rod an Rahmen
        auto p_ftl = eval(grade<1>(r_ftl + (r_ftl & r_ftl) * einf * 0.5 + e0));

        double r_ft = 0.39760; //Länge tie rod
        double r_fto = 0.08377; //Abstand p_flol zu p_ftol
        double r_fuo = 0.23717; //Abstand p_fuol zu p_ftol

        auto s_ftol = grade<1>(p_flol - einf * r_fto * r_fto * 0.5);
        auto s_ftl = grade<1>(p_ftl - einf * r_ft * r_ft * 0.5);
        auto s_fuol = grade<1>(p_fuol - einf * r_fuo * r_fuo * 0.5);
        auto Pp_ftol = (s_ftol ^ s_ftl ^ s_fuol) * Ic;
        auto p_ftol = eval(grade<1>((Pp_ftol + one * sqrt((Pp_ftol & Pp_ftol).element<0>())) * (!(Pp_ftol & einf))));

        //Bestimmung Radaufstandspunkt

        double r_wheel = 0.255; //Reifenradius
        auto phi_fpol = (p_flol ^ p_fuol ^ p_ftol ^ einf) * Ic; //Ebene front points outer left
        auto phi_fpoln = eval(phi_fpol * (!(magnitude(phi_fpol))) * (-1.0));
        auto T_fwrl = eval(one + einf * (p_flol - e0) * 0.5); //Definition Translator
        auto phi_fwrl = eval(e0 & (einf ^ (e2 * sqrt(eval(phi_fpoln & phi_fpoln)) + phi_fpoln))); //Ebene front wheel reference left
        auto R_fwrl_raw = part<0, 3, 5, 6>(phi_fwrl * e2); //Definition Rotor
        auto R_fwrl = R_fwrl_raw * !magnitude(R_fwrl_raw);
        //auto R_frwl1 = one*(0.5*(-1.0)) + ((e2^e3)*0.00187*(-1.0)) + ((e1^e2)*0.161*(-1.0));
        auto R_frwl1 = one * (0.986952634679) + ((e2 ^ e3) * 0.00187 * (-1.0)) + ((e1 ^ e2) * 0.161 * (-1.0));
        //auto R_frwl2 = eval(exp((~(sqrt(e1*e1)))*0.5*e1*e1*e2*e3*(2.0*3.141/180.0)*(-1.0)));
        auto R_frwl2 = eval(exp(0.5 * e1 * e1 * e2 * e3 * (2.0 * 3.141 / 180.0) * (-1.0)));
        //auto R_frwl3 = eval(exp((~(sqrt(e3*e3)))*0.5*e3*e1*e2*e3*(0.5*3.141/180.0)));
        auto R_frwl3 = eval(exp(0.5 * e3 * e1 * e2 * e3 * (0.5 * 3.141 / 180.0)));
        auto T_fwp = eval(one + einf * (e1 * (-0.004) + (e2 * 0.050 * (-1.0)) + e3 * 0.1028) * 0.5);
        auto D_wfl_ks = eval(T_fwrl * R_fwrl * R_frwl1 * T_fwp * R_frwl3 * R_frwl2);
        //cm::mv<0,3,5,6>::type R_ks;
        //R_ks[0] = cos(0.5*M_PI); R_ks[1] = sin(-0.5*M_PI); R_ks[2] = 0.0; R_ks[3] = 0.0;
        //D_wfl = R_ks*D_wfl_ks;
        //auto R_ks = exp(0.5*M_PI*e1*e2);
        D_wfl = D_wfl_ks;

        std::cout << "D_wfl_ks: " << D_wfl_ks << std::endl;
        std::cout << "R_fwrl: " << R_fwrl << std::endl;
        std::cout << "R_fwrl1: " << R_frwl1 << std::endl;
        std::cout << "R_fwrl2: " << R_frwl2 << std::endl;
        std::cout << "R_fwrl3: " << R_frwl3 << std::endl;
        //std::cout << "R_ks: " << R_ks << std::endl;

        auto p_fwl1 = eval(grade<1>(e3 * (-r_wheel) + einf * r_wheel * r_wheel * 0.5 + e0));
        auto p_wfl1 = eval(grade<1>(D_wfl * p_fwl1 * (~(D_wfl)))); //Radaufstandspunkt

        //Bestimmung Kraftaufteilung

        auto phi_fll = (p_fll ^ p_fll2 ^ p_flol ^ einf) * Ic; //Ebene Querlenker unten
        auto phi_ful = (p_ful ^ p_ful2 ^ p_fuol ^ einf) * Ic; //Ebene Querlenker oben
        auto ma_fl = eval((phi_fll ^ phi_ful) * Ic); //Momentanpolachse
        auto prc_wfl = (ma_fl ^ p_wfl1) * Ic; //Kraftebene
        auto nprc_wfl = ((prc_wfl ^ E) * E) * (!magnitude((prc_wfl ^ E) * E)); //Normalenvektor
        auto nrc_wfl_ks = nprc_wfl * (1.0 / sqrt(eval(nprc_wfl & nprc_wfl))); //normierter Normalenvektor

        //Drehung des Koordinatensytems um 180° um die z-Achse
        //nrc_wfl = R_ks*nrc_wfl_ks;
        nrc_wfl = nrc_wfl_ks;
    }

    void Radaufhaengung_wfr(double u_wfr, double steerAngle, D_type &D_wfr, Vector &nrc_wfr) const
    {
        u_wfr = std::max(-0.02, std::min(0.02, u_wfr));
        //Feder-Daempfer System
        auto r_fsr = e1 * 0.004 + e2 * 0.05159 + e3 * 0.58555;
        auto p_fsr = eval(grade<1>(r_fsr + (r_fsr & r_fsr) * einf * 0.5 + e0));
        auto r_fbr = e1 * 0.004 + e2 * 0.266 + e3 * 0.456;
        auto p_fbr = eval(grade<1>(r_fbr + (r_fbr & r_fbr) * einf * 0.5 + e0));
        //Querlenkerpunkt front lower frame left
        auto r_flr = e1 * 0.004 + e2 * 0.195 + e3 * 0.097;
        auto p_flr = eval(grade<1>(r_flr + (r_flr & r_flr) * einf * 0.5 + e0));
        auto r_flr2 = e1 * 0.280 + e2 * 0.195 + e3 * 0.097;
        auto p_flr2 = eval(grade<1>(r_flr2 + (r_flr2 & r_flr2) * einf * 0.5 + e0));

        double r_fsb = 0.04633;
        double r_fsd = 0.25772 - u_wfr;

        auto s_fsr = eval(grade<1>(p_fsr - einf * r_fsd * r_fsd * 0.5));
        auto s_fbsr = eval(grade<1>(p_fbr - einf * r_fsb * r_fsb * 0.5));
        auto c_fsbr = (s_fsr ^ s_fbsr);
        auto phi_fsd = (p_flr ^ p_fsr ^ p_fbr ^ einf) * Ic;
        auto Pp_fsbr = (phi_fsd ^ c_fsbr) * Ic;
        auto p_fsbr = eval(grade<1>((Pp_fsbr + one * sqrt((Pp_fsbr & Pp_fsbr).element<0x00>())) * (!(Pp_fsbr & einf))));

        double r_fpb = 0.0764;
        double r_fpsb = 0.05116;

        auto s_fsbr = grade<1>(p_fsbr - einf * r_fpsb * r_fpsb * 0.5);
        auto s_fbr = grade<1>(p_fbr - einf * r_fpb * r_fpb * 0.5);
        auto c_fpbr = (s_fsbr ^ s_fbr);
        auto Pp_fpbr = (phi_fsd ^ c_fpbr) * Ic;
        auto p_fpbr = eval(grade<1>((Pp_fpbr + one * sqrt((Pp_fpbr & Pp_fpbr).element<0x00>())) * (!(Pp_fpbr & einf))));

        //Querlenker

        double r_fp = 0.38418;
        double r_fl = 0.35726;

        auto s_flr = grade<1>(p_flr - einf * r_fl * r_fl * 0.5);
        auto s_fpbr = grade<1>(p_fpbr - einf * r_fp * r_fp * 0.5);
        auto c_flor = (s_fpbr ^ s_flr);
        auto Pp_flor = (phi_fsd ^ c_flor) * Ic;
        auto p_flor = eval(grade<1>((Pp_flor + one * sqrt((Pp_flor & Pp_flor).element<0x00>())) * (!(Pp_flor & einf))));

        auto r_fur = e1 * 0.037 + e2 * 0.288 + e3 * 0.261;
        auto p_fur = eval(grade<1>(r_fur + (r_fur & r_fur) * einf * 0.5 + e0));
        auto r_fur2 = e1 * 0.210 + e2 * 0.288 + e3 * 0.261;
        auto p_fur2 = eval(grade<1>(r_fur2 + (r_fur2 & r_fur2) * einf * 0.5 + e0));

        double r_fo = 0.21921;
        double r_fu = 0.26086;

        //Punkte fuer Ebene oberer Querlenker
        auto r_phi1 = e1 * 0.037 + e2 * 0.0 + e3 * 0.0;
        auto p_phi1 = eval(grade<1>(r_phi1 + (r_phi1 & r_phi1) * einf * 0.5 + e0));
        auto r_phi2 = e1 * 0.037 + e2 * 1.0 + e3 * 0.0;
        auto p_phi2 = eval(grade<1>(r_phi2 + (r_phi2 & r_phi2) * einf * 0.5 + e0));

        auto s_fur = grade<1>(p_fur - einf * r_fu * r_fu * 0.5);
        auto s_flor = grade<1>(p_flor - einf * r_fo * r_fo * 0.5);
        auto c_flur = (s_flor ^ s_fur);
        auto phi_fuo = (p_fur ^ p_phi1 ^ p_phi2 ^ einf) * Ic;
        auto Pp_fuor = (phi_fuo ^ c_flur) * Ic;
        auto p_fuor = eval(grade<1>((Pp_fuor + one * sqrt((Pp_fuor & Pp_fuor).element<0x00>())) * (!(Pp_fuor & einf))));

        //Spurstange
        double steering = steerAngle * 11.4592; //Umrechnung vom Lenkwinkel auf Lenkgetriebeweg
        auto r_ftr = e1 * (-0.055) + e2 * (0.204 + steering) + e3 * 0.101; //Anbindungspunkt tie rod an Rahmen
        auto p_ftr = eval(grade<1>(r_ftr + (r_ftr & r_ftr) * einf * 0.5 + e0));

        double r_ft = 0.39760; //Länge tie rod
        double r_fto = 0.08377; //Abstand p_flol zu p_ftol
        double r_fuo = 0.23717; //Abstand p_fuol zu p_ftol

        auto s_ftor = grade<1>(p_flor - einf * r_fto * r_fto * 0.5);
        auto s_ftr = grade<1>(p_ftr - einf * r_ft * r_ft * 0.5);
        auto s_fuor = grade<1>(p_fuor - einf * r_fuo * r_fuo * 0.5);
        auto Pp_ftor = (s_ftor ^ s_ftr ^ s_fuor) * Ic;
        auto p_ftor = eval(grade<1>((Pp_ftor + one * sqrt((Pp_ftor & Pp_ftor).element<0>())) * (!(Pp_ftor & einf))));

        //Bestimmung Radaufstandspunkt

        double r_wheel = 0.255; //Reifenradius
        auto phi_fpor = (p_flor ^ p_fuor ^ p_ftor ^ einf) * Ic; //Ebene front points outer left
        auto phi_fporn = eval(phi_fpor * (!(magnitude(phi_fpor))) * (-1.0));
        auto T_fwrr = eval(one + einf * (p_flor - e0) * 0.5); //Definition Translator
        auto phi_fwrr = eval(e0 & (einf ^ (e2 * sqrt(eval(phi_fporn & phi_fporn)) + phi_fporn))); //Ebene front wheel reference left
        auto R_fwrr_raw = part<0, 3, 5, 6>(phi_fwrr * e2); //Definition Rotor
        auto R_fwrr = R_fwrr_raw * !magnitude(R_fwrr_raw);
        //auto R_frwr1 = one*(0.5*(-1.0)) + (e2^e3)*0.00187 + (e1^e2)*0.161;
        auto R_frwr1 = one * (0.986952634679) + ((e2 ^ e3) * 0.00187) + ((e1 ^ e2) * 0.161);
        //auto R_frwr2 = eval(exp((~(sqrt(e1*e1)))*0.5*e1*e1*e2*e3*(2.0*3.141/180.0)*(-1.0)));
        auto R_frwr2 = eval(exp(0.5 * e1 * e1 * e2 * e3 * (2.0 * 3.141 / 180.0) * (-1.0)));
        //auto R_frwr3 = eval(exp((~(sqrt(e3*e3)))*0.5*e3*e1*e2*e3*(0.5*3.141/180.0)));
        auto R_frwr3 = eval(exp(0.5 * e3 * e1 * e2 * e3 * (0.5 * 3.141 / 180.0)));
        auto T_fwp = eval(one + einf * (e1 * (-0.004) + e2 * 0.050 + e3 * 0.1028) * 0.5);
        auto D_wfr_ks = T_fwrr * R_fwrr * R_frwr1 * T_fwp * R_frwr3 * R_frwr2;

        //cm::mv<0,3,5,6>::type R_ks;
        //R_ks[0] = cos(-0.5*M_PI); R_ks[1] = sin(-0.5*M_PI); R_ks[2] = 0.0; R_ks[3] = 0.0;
        //D_wfr = R_ks*D_wfr_ks;
        D_wfr = D_wfr_ks;

        auto p_fwr1 = eval(grade<1>(e3 * (-r_wheel) + einf * r_wheel * r_wheel * 0.5 + e0));
        auto p_wfr1 = eval(grade<1>(D_wfr * p_fwr1 * (~(D_wfr)))); //Radaufstandspunkt

        //Bestimmung Kraftaufteilung

        auto phi_flr = (p_flr ^ p_flr2 ^ p_flor ^ einf) * Ic; //Ebene Querlenker unten
        auto phi_fur = (p_fur ^ p_fur2 ^ p_fuor ^ einf) * Ic; //Ebene Querlenker oben
        auto ma_fr = eval((phi_flr ^ phi_fur) * Ic); //Momentanpolachse
        auto prc_wfr = (ma_fr ^ p_wfr1) * Ic; //Kraftebene
        auto nprc_wfr = ((prc_wfr ^ E) * E) * (!magnitude((prc_wfr ^ E) * E));
        auto nrc_wfr_ks = nprc_wfr * (1.0 / sqrt(eval(nprc_wfr & nprc_wfr)));

        //Drehung des Koordinatensytems um 180° um die z-Achse
        //nrc_wfr = R_ks*nrc_wfr_ks;
        nrc_wfr = nrc_wfr_ks;
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
    static const double m_b = 253.0;
    static const double r_b = 3.0;
    //Sphere
    //static const double I_b = 2.0/5.0*m_b*r_b*r_b;

    //Wheel
    static const double m_w = 10;
    static const double r_w = 0.255;
    static const double I_w = 0.65025;
    static const double u_wn = 0.4;
    static const double v_wn = 1.3;
    static const double w_wn = 0.7;
    //static const double k_wf = 17400.0;
    static const double k_wf = 17400.0;
    //static const double k_wr = 26100.0;
    static const double k_wr = 26100.0;
    static const double d_wf = 2600.0;
    static const double d_wr = 2600.0;
    gaalet::mv<0, 3, 5, 6>::type R_n_wfl;
    gaalet::mv<0, 3, 5, 6>::type R_n_wfr;
    gaalet::mv<0, 3, 5, 6>::type R_n_wrl;
    gaalet::mv<0, 3, 5, 6>::type R_n_wrr;

    //Braking system
    static const double mu_b = 0.135;
    static const double d_b = 0.01;

    //Anti roll bar
    //static const double k_arb = 50000;
    static const double k_arb = 50000;

    //Clutch
    static const double k_cn = 1.5;

    //Engine
    static const double a_e = -0.000862;
    static const double b_e = 0.83;
    static const double c_e = 400;
    static const double I_e = 0.5;
    static const double d_e = 0.5;

    //Transmission
    std::vector<double> i_g;
    static const double i_a = 3.5;

    //Wheel kinematics
    static const double wheel_left = 1.0;
    static const double wheel_right = -1.0;

    //Point p_wbf_fl;
    //Point p_wbf_fr;
    Point p_wbf_rl;
    Point p_wbf_rr;
    //Point p_wbr_fl;
    //Point p_wbr_fr;
    Point p_wbr_rl;
    Point p_wbr_rr;
    //Point p_mps_fl;
    //Point p_mps_fr;
    Point p_mps_rl;
    Point p_mps_rr;
    //Point p_steer0_fl;
    //Point p_steer0_fr;
    Point p_steer0_rl;
    Point p_steer0_rr;

    //Spring-damper attachment to body
    Vector nv_sfl;
    Vector nr_sfl;
    Vector nv_sfr;
    Vector nr_sfr;
};

} //end namespace cardyn

//class PLUGINEXPORT CarDynamicsRtus
class CarDynamicsRtus : public VehicleDynamics, OpenThreads::Thread
{
public:
    CarDynamicsRtus();
    virtual ~CarDynamicsRtus();

    void run();

    virtual void move(VrmlNodeVehicle *vehicle);

    virtual void resetState();

    virtual double getVelocity()
    {
        return 0.0;
    }
    virtual double getAcceleration()
    {
        return 0.0;
    }
    virtual double getEngineSpeed()
    {
        return 0.0;
    }
    virtual double getEngineTorque()
    {
        return 0.0;
    }
    virtual double getTyreSlipFL()
    {
        return 0.0;
    }
    virtual double getTyreSlipFR()
    {
        return 0.0;
    }
    virtual double getTyreSlipRL()
    {
        return 0.0;
    }
    virtual double getTyreSlipRR()
    {
        return 0.0;
    }

    virtual double getSteeringWheelTorque()
    {
        return 0.0;
    }

    virtual const osg::Matrix &getVehicleTransformation()
    {
        chassisTrans;
    };

    virtual void setVehicleTransformation(const osg::Matrix &);

    cardyn_rtus::Plane getRoadTangentPlane(Road *&road, Vector2D v_c);
    void getRoadSystemContactPoint(const cardyn_rtus::Point &p_w, Road *&road, double &u, cardyn_rtus::Plane &s_c);
    void getFirstRoadSystemContactPoint(const cardyn_rtus::Point &p_w, Road *&road, double &u, cardyn_rtus::Plane &s_c);

    std::pair<Road *, Vector2D> getStartPositionOnRoad();

protected:
    cardyn_rtus::InputVector z;
    cardyn_rtus::OutputVector o;
    cardyn_rtus::StateVector y;

    magicformula2004::TyrePropertyPack tyrePropLeft;
    magicformula2004::TyrePropertyPack tyrePropRight;
    magicformula2004::ContactWrench tyreFL;
    magicformula2004::ContactWrench tyreFR;
    magicformula2004::ContactWrench tyreRL;
    magicformula2004::ContactWrench tyreRR;
    cardyn_rtus::StateEquation f;

    Road *road_wheel_fl, *road_wheel_fr, *road_wheel_rl, *road_wheel_rr;
    double u_wheel_fl, u_wheel_fr, u_wheel_rl, u_wheel_rr;

    //EulerIntegrator<cardyn_rtus::StateEquation, cardyn_rtus::StateVector> integrator;
    RungeKuttaClassicIntegrator<cardyn_rtus::StateEquation, cardyn_rtus::StateVector> integrator;

    osg::Matrix chassisTrans;

    bool runTask;
    bool firstMoveCall;

    std::pair<Road *, Vector2D> startPos;
};

#endif
