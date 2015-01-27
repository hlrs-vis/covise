/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "/mnt/raid/home/hpcmdors/src/gealg/head/GeometricAlgebra.h"

//Querlenkerkinematik mir Zeitableitungen
int main()
{

    gealg::mv<1, 0x08, 0x10>::type ep(1.0);
    gealg::mv<1, 0x10, 0x10>::type em(1.0);

    gealg::mv<2, 0x1008, 0x10>::type en((em - ep) * 0.5);
    gealg::mv<2, 0x1008, 0x10>::type ei(em + ep);

    gealg::mv<1, 0x18, 0x10>::type E(ei ^ en);

    gealg::mv<1, 0x01, 0x10>::type e1(1.0);
    gealg::mv<1, 0x02, 0x10>::type e2(1.0);
    gealg::mv<1, 0x04, 0x10>::type e3(1.0);
    gealg::mv<1, 0x1f, 0x10>::type I(e1 * e2 * e3 * ep * em);
    gealg::mv<1, 0x07, 0x10>::type i(e1 * e2 * e3);

    gealg::mv<1, 0x00, 0x10>::type one(1.0);

    //std::cout << "s_fbsl:  " << ddp_fpbl << std::endl; //Ausgabe

    //Feder-Daempfer System
    auto r_fsl = e1 * 0.004 + e2 * 0.05159 + e3 * 0.58555;
    auto p_fsl = (r_fsl + r_fsl % r_fsl * ei * 0.5 + en)();
    auto r_fbl = e1 * 0.004 + e2 * 0.266 + e3 * 0.456;
    auto p_fbl = (r_fbl + r_fbl % r_fbl * ei * 0.5 + en)();
    //Querlenkerpunkt front lower frame left
    auto r_fll = e1 * 0.004 + e2 * 0.195 + e3 * 0.097;
    auto p_fll = (r_fll + r_fll % r_fll * ei * 0.5 + en)();

    double alpha = 0.0;

    double r_fsb = 0.04633;
    double r_fsd = 0.25772 - alpha;
    auto dr_fsd(0.0);
    auto ddr_fsd(0.0);

    auto s_fsl = (p_fsl - ei * r_fsd * r_fsd * 0.5)();
    auto ds_fsl = ei * dr_fsd * r_fsd * (-1.0);
    auto dds_fsl = ei * (ddr_fsd * r_fsd + dr_fsd * dr_fsd) * (-1.0);
    auto s_fbsl = (p_fbl - ei * r_fsb * r_fsb * 0.5)();
    auto c_fsbl = (s_fsl ^ s_fbsl);
    auto dc_fsbl = (ds_fsl ^ s_fbsl);
    auto ddc_fsbl = (dds_fsl ^ s_fbsl);
    auto phi_fsd = (p_fll ^ p_fsl ^ p_fbl ^ ei) * I;
    auto Pp_fsbl = (phi_fsd ^ c_fsbl) * I;
    auto dPp_fsbl = (phi_fsd ^ dc_fsbl) * I;
    auto ddPp_fsbl = (phi_fsd ^ ddc_fsbl) * I;
    auto p_fsbl = (grade<1>((Pp_fsbl + sqrt(element<0x00>(Pp_fsbl % Pp_fsbl))) * (~(Pp_fsbl % ei))))();
    auto dp_fsbl = (grade<1>((dPp_fsbl + dPp_fsbl * Pp_fsbl * (~(sqrt(element<0x00>(Pp_fsbl % Pp_fsbl)))))() * (~(Pp_fsbl % ei)) + (Pp_fsbl + sqrt(element<0x00>(Pp_fsbl % Pp_fsbl)))() * dPp_fsbl * (-1.0) * ei * (~(Pp_fsbl % ei * Pp_fsbl % ei))))();
    auto ddp_fsbl = (grade<1>((ddPp_fsbl + ddPp_fsbl * Pp_fsbl * (~(sqrt(element<0x00>(Pp_fsbl % Pp_fsbl)))) + dPp_fsbl * Pp_fsbl * (dPp_fsbl * (-1.0) * Pp_fsbl) * (~(sqrt(element<0x00>(Pp_fsbl % Pp_fsbl)))))() * (~(Pp_fsbl % ei)) + (dPp_fsbl + dPp_fsbl * Pp_fsbl * sqrt(element<0x00>(Pp_fsbl % Pp_fsbl)))() * (dPp_fsbl * (-1.0)) * ei * (~(Pp_fsbl % ei * Pp_fsbl % ei)) * 2.0 + (Pp_fsbl + sqrt(element<0x00>(Pp_fsbl % Pp_fsbl)))() * ((ddPp_fsbl * (-1.0)) * ei * (~(Pp_fsbl % ei * Pp_fsbl % ei)) + dPp_fsbl % ei * dPp_fsbl % ei * (~(Pp_fsbl % ei * Pp_fsbl % ei * Pp_fsbl % ei)) * 2.0)()))();

    double r_fpb = 0.0764;
    double r_fpsb = 0.05116;

    auto s_fsbl = p_fsbl - ei * r_fpsb * r_fpsb * 0.5;
    auto ds_fsbl = dp_fsbl;
    auto dds_fsbl = ddp_fsbl;
    auto s_fbl = p_fbl - ei * r_fpb * r_fpb * 0.5;
    auto c_fpbl = (s_fsbl ^ s_fbl);
    auto dc_fpbl = (ds_fsbl ^ s_fbl);
    auto ddc_fpbl = (dds_fsbl ^ s_fbl);
    auto Pp_fpbl = (phi_fsd ^ c_fpbl) * I;
    auto dPp_fpbl = (phi_fsd ^ dc_fpbl) * I;
    auto ddPp_fpbl = (phi_fsd ^ ddc_fpbl) * I;
    auto p_fpbl = (grade<1>((Pp_fpbl + sqrt(element<0x00>(Pp_fpbl % Pp_fpbl))) * (~(Pp_fpbl % ei))))();
    auto dp_fpbl = (grade<1>((dPp_fpbl + dPp_fpbl * Pp_fpbl * (~(sqrt(element<0x00>(Pp_fpbl % Pp_fpbl)))))() * (~(Pp_fpbl % ei)) + (Pp_fpbl + sqrt(element<0x00>(Pp_fpbl % Pp_fpbl)))() * dPp_fpbl * (-1.0) * ei * (~(Pp_fpbl % ei * Pp_fpbl % ei))))();
    auto ddp_fpbl = (grade<1>((ddPp_fpbl + ddPp_fpbl * Pp_fpbl * (~(sqrt(element<0x00>(Pp_fpbl % Pp_fpbl))))() + dPp_fpbl * Pp_fpbl * (dPp_fpbl * (-1.0) * Pp_fpbl) * (~(sqrt(element<0x00>(Pp_fpbl % Pp_fpbl)))))() * (~(Pp_fpbl % ei)) + (dPp_fpbl + dPp_fpbl * Pp_fpbl * sqrt(element<0x00>(Pp_fpbl % Pp_fpbl)))() * (dPp_fpbl * (-1.0)) * ei * (~(Pp_fpbl % ei * Pp_fpbl % ei)) * 2.0 + (Pp_fpbl + sqrt(element<0x00>(Pp_fpbl % Pp_fpbl)))() * ((ddPp_fpbl * (-1.0)) * ei * (~(Pp_fpbl % ei * Pp_fpbl % ei)) + dPp_fpbl % ei * dPp_fpbl % ei * (~(Pp_fpbl % ei * Pp_fpbl % ei * Pp_fpbl % ei)) * 2.0)))();

    //Querlenker

    double r_fp = 0.38418;
    double r_fl = 0.35726;

    auto s_fll = p_fll - ei * r_fl * r_fl * 0.5;
    auto s_fpbl = p_fpbl - ei * r_fp * r_fp * 0.5;
    auto ds_fpbl = dp_fpbl;
    auto dds_fpbl = ddp_fpbl;
    auto c_flol = (s_fll ^ s_fpbl);
    auto dc_flol = (s_fll ^ ds_fpbl);
    auto ddc_flol = (s_fll ^ dds_fpbl);
    auto Pp_flol = (phi_fsd ^ c_flol) * I;
    auto dPp_flol = (phi_fsd ^ dc_flol) * I;
    auto ddPp_flol = (phi_fsd ^ ddc_flol) * I;
    auto p_flol = (grade<1>((Pp_flol - sqrt(element<0x00>(Pp_flol % Pp_flol))) * (~(Pp_flol % ei))))();
    auto dp_flol = (grade<1>((dPp_flol + dPp_flol * Pp_flol * (~(sqrt(element<0x00>(Pp_flol % Pp_flol))))) * (~(Pp_flol % ei)) + (Pp_flol + sqrt(element<0x00>(Pp_flol % Pp_flol))) * dPp_flol * (-1.0) * ei * (~(Pp_flol % ei * Pp_flol % ei))))();
    auto ddp_flol = (grade<1>((ddPp_flol + ddPp_flol * Pp_flol * (~(sqrt(element<0x00>(Pp_flol % Pp_flol)))) + dPp_flol * Pp_flol * (dPp_flol * (-1.0) * Pp_flol) * (~(sqrt(element<0x00>(Pp_flol % Pp_flol))))) * (~(Pp_flol % ei)) + (dPp_flol + dPp_flol * Pp_flol * sqrt(element<0x00>(Pp_flol % Pp_flol))) * (dPp_flol * (-1.0)) * ei * (~(Pp_flol % ei * Pp_flol % ei)) * 2.0 + (Pp_flol + sqrt(element<0x00>(Pp_flol % Pp_flol))) * ((ddPp_flol * (-1.0)) * ei * (~(Pp_flol % ei * Pp_flol % ei)) + dPp_flol % ei * dPp_flol % ei * (~(Pp_flol % ei * Pp_flol % ei * Pp_flol % ei)) * 2.0)))();

    auto r_ful = e1 * 0.037 + e2 * 0.288 + e3 * 0.261;
    auto p_ful = r_ful + r_ful % r_ful * ei * 0.5 + en;

    double r_fo = 0.21921;
    double r_fu = 0.26086;

    //Punkte fuer Ebene oberer Querlenker
    auto r_phi1 = e1 * 0.037 + e2 * 0.0 + e3 * 0.0;
    auto p_phi1 = r_phi1 + r_phi1 % r_phi1 * ei * 0.5 + en;
    auto r_phi2 = e1 * 0.037 + e2 * 1.0 + e3 * 0.0;
    auto p_phi2 = r_phi2 + r_phi2 % r_phi2 * ei * 0.5 + en;

    auto s_ful = p_ful - ei * r_fu * r_fu * 0.5;
    auto s_flol = p_flol - ei * r_fo * r_fo * 0.5;
    auto ds_flol = dp_flol;
    auto dds_flol = ddp_flol;
    auto c_flul = (s_ful ^ s_flol);
    auto dc_flul = (s_ful ^ ds_flol);
    auto ddc_flul = (s_ful ^ dds_flol);
    auto phi_fuo = (p_ful ^ p_phi1 ^ p_phi2 ^ ei) * I;
    auto Pp_fuol = (phi_fuo ^ c_flul) * I;
    auto dPp_fuol = (phi_fuo ^ dc_flul) * I;
    auto ddPp_fuol = (phi_fuo ^ ddc_flul) * I;
    auto p_fuol = (grade<1>((Pp_fuol - sqrt(element<0x00>(Pp_fuol % Pp_fuol))) * (~(Pp_fuol % ei))))();
    auto dp_fuol = (grade<1>((dPp_fuol + dPp_fuol * Pp_fuol * (~(sqrt(element<0x00>(Pp_fuol % Pp_fuol))))) * (~(Pp_fuol % ei)) + (Pp_fuol + sqrt(element<0x00>(Pp_fuol % Pp_fuol))) * dPp_fuol * (-1.0) * ei * (~(Pp_fuol % ei * Pp_fuol % ei))))();
    auto ddp_fuol = (grade<1>((ddPp_fuol + ddPp_fuol * Pp_fuol * (~(sqrt(element<0x00>(Pp_fuol % Pp_fuol)))) + dPp_fuol * Pp_fuol * (dPp_fuol * (-1.0) * Pp_fuol) * (~(sqrt(element<0x00>(Pp_fuol % Pp_fuol))))) * (~(Pp_fuol % ei)) + (dPp_fuol + dPp_fuol * Pp_fuol * sqrt(element<0x00>(Pp_fuol % Pp_fuol))) * (dPp_fuol * (-1.0)) * ei * (~(Pp_fuol % ei * Pp_fuol % ei)) * 2.0 + (Pp_fuol + sqrt(element<0x00>(Pp_fuol % Pp_fuol))) * ((ddPp_fuol * (-1.0)) * ei * (~(Pp_fuol % ei * Pp_fuol % ei)) + dPp_fuol % ei * dPp_fuol % ei * (~(Pp_fuol % ei * Pp_fuol % ei * Pp_fuol % ei)) * 2.0)))();

    //Spurstange
    auto r_ftl = e1 * 0.004 + e2 * 0.05159 + e3 * 0.58555; //Anbindungspunkt tie rod an Rahmen
    auto p_ftl = r_ftl + r_ftl % r_ftl * ei * 0.5 + en;

    double r_ftr = 0.04633; //Länge tie rod
    double r_fto = 0.04633; //Abstand p_flol zu p_ftol
    double r_fuo = 0.04633; //Abstand p_fuol zu p_ftol

    auto s_ftol = p_flol - ei * r_fto * r_fto * 0.5;
    auto ds_ftol = dp_flol;
    auto dds_ftol = ddp_flol;
    auto s_ftl = p_ftl - ei * r_ftr * r_ftr * 0.5;
    auto s_fuol = p_fuol - ei * r_fuo * r_fuo * 0.5;
    auto ds_fuol = dp_fuol;
    auto dds_fuol = ddp_fuol;
    auto Pp_ftol = (s_ftol ^ s_ftl ^ s_fuol) * I;
    auto dPp_ftol = ((ds_ftol ^ s_ftl ^ s_fuol) + (s_ftol ^ s_ftl ^ ds_fuol)) * I;
    auto ddPp_ftol = ((dds_ftol ^ s_ftl ^ s_fuol) + (ds_ftol ^ s_ftl ^ ds_fuol) + (ds_ftol ^ s_ftl ^ ds_fuol) + (s_ftol ^ s_ftl ^ dds_fuol)) * I;
    auto p_ftol = (grade<1>((Pp_ftol + sqrt(element<0x00>(Pp_ftol % Pp_ftol))) * (~(Pp_ftol % ei))))();
    auto dp_ftol = (grade<1>((dPp_ftol + dPp_ftol * Pp_ftol * (~(sqrt(element<0x00>(Pp_ftol % Pp_ftol))))) * (~(Pp_ftol % ei)) + (Pp_ftol + sqrt(element<0x00>(Pp_ftol % Pp_ftol))) * dPp_ftol * (-1.0) * ei * (~(Pp_ftol % ei * Pp_ftol % ei))))();
    auto ddp_ftol = (grade<1>((ddPp_ftol + ddPp_ftol * Pp_ftol * (~(sqrt(element<0x00>(Pp_ftol % Pp_ftol)))) + dPp_ftol * Pp_ftol * (dPp_ftol * (-1.0) * Pp_ftol) * (~(sqrt(element<0x00>(Pp_ftol % Pp_ftol))))) * (~(Pp_ftol % ei)) + (dPp_ftol + dPp_ftol * Pp_ftol * sqrt(element<0x00>(Pp_ftol % Pp_ftol))) * (dPp_ftol * (-1.0)) * ei * (~(Pp_ftol % ei * Pp_ftol % ei)) * 2.0 + (Pp_ftol + sqrt(element<0x00>(Pp_ftol % Pp_ftol))) * ((ddPp_ftol * (-1.0)) * ei * (~(Pp_ftol % ei * Pp_ftol % ei)) + dPp_ftol % ei * dPp_ftol % ei * (~(Pp_ftol % ei * Pp_ftol % ei * Pp_ftol % ei)) * 2.0)))();

    //Bestimmung Radaufstandspunkt
    double r_wheel = 0.255; //Reifenradius

    auto phi_fpol = (p_flol ^ p_fuol ^ p_ftol ^ ei) * I; //Ebene front points outer left
    auto dphi_fpol = (((dp_flol ^ p_fuol ^ p_ftol ^ ei) + (p_flol ^ dp_fuol ^ p_ftol ^ ei) + (p_flol ^ p_fuol ^ dp_ftol ^ ei)) * I)();
    auto ddphi_fpol = (((ddp_flol ^ p_fuol ^ p_ftol ^ ei) + (dp_flol ^ dp_fuol ^ p_ftol ^ ei) + (dp_flol ^ p_fuol ^ dp_ftol ^ ei) + (dp_flol ^ dp_fuol ^ p_ftol ^ ei) + (p_flol ^ ddp_fuol ^ p_ftol ^ ei) + (p_flol ^ dp_fuol ^ dp_ftol ^ ei) + (dp_flol ^ p_fuol ^ dp_ftol ^ ei) + (p_flol ^ dp_fuol ^ dp_ftol ^ ei) + (p_flol ^ p_fuol ^ ddp_ftol ^ ei)) * I)();
    auto T_fwrl = (one + ei * (p_flol - en) * 0.5)(); //Definition Translator
    auto dT_fwrl = ei * dp_flol * 0.5;
    auto ddT_fwrl = ei * ddp_flol * 0.5;
    auto phi_fwrl = (en % (ei ^ (e2 * sqrt(phi_fpol * phi_fpol) + phi_fpol)))(); //Ebene front wheel reference left
    auto dphi_fwrl = (en % (ei ^ (e2 * (dphi_fpol * phi_fpol * (~(sqrt(phi_fpol * phi_fpol)))) + dphi_fpol)))();
    auto ddphi_fwrl = (en % (ei ^ (e2 * ((ddphi_fpol * phi_fpol + (dphi_fpol * dphi_fpol)) * (~(sqrt(phi_fpol * phi_fpol))) + (dphi_fpol * phi_fpol) * (dphi_fpol * phi_fpol) * (~(phi_fpol * phi_fpol * sqrt(phi_fpol * phi_fpol))) * (-1.0)) + ddphi_fpol)))();
    auto R_fwrl = phi_fwrl * e2; //Definition Rotor
    auto dR_fwrl = dphi_fwrl * e2;
    auto ddR_fwrl = ddphi_fwrl * e2;
    auto p_frwl = e1 * 0.004603 - e2 * 0.003048 + e3 * 0.086825; //Vektor front rotation wheel left
    auto R_frwl = (exp((~(sqrt(p_frwl * p_frwl))) * 0.5 * p_frwl * e1 * e2 * e3 * (144.548 * 3.141 / 180.0)))();
    auto T_fwp = (one + ei * (e1 * (-0.004) + e2 * 0.059 + e3 * 0.103) * 0.5)();
    auto D_fwp = (T_fwrl * R_fwrl * T_fwp * R_frwl)();
    auto dD_fwp = (dT_fwrl * R_fwrl + T_fwrl * dR_fwrl) * T_fwp * R_frwl;
    auto ddD_fwp = (ddT_fwrl * R_fwrl + dT_fwrl * dR_fwrl * 2.0 + T_fwrl * ddR_fwrl) * T_fwp * R_frwl;
    auto p_fwl = e3 * (-1.0) + ei * r_wheel * r_wheel * 0.5 + en;

    //Bewegung Radaufstandspunkt

    auto p_wfl = D_fwp * p_fwl * (~(D_fwp));
    auto dp_wfl = dD_fwp * p_fwl * (~(D_fwp)) + D_fwp * p_fwl * (~(D_fwp * D_fwp)) * dD_fwp * (-1.0);
    auto ddp_wfl = (ddD_fwp * p_fwl * (~(D_fwp)) + dD_fwp * p_fwl * (~(D_fwp * D_fwp)) * dD_fwp * (-1.0) + dD_fwp * p_fwl * (~(D_fwp * D_fwp)) * dD_fwp * (-1.0) + D_fwp * p_fwl * ((~(D_fwp * D_fwp)) * ddD_fwp * (-1.0) + (~(D_fwp * D_fwp * D_fwp)) * dD_fwp * dD_fwp * 2.0))();
    /*	auto Fp_wfl = (F_wfl%ddp_wfl)*(~(ddp_wfl));
	auto Fsd_wfl = (u_wfl*k + du_wfl*d)*0.5;
	auto ddp_wcfl = (Fp_wfl + Fsd_wfl)*(1.0/m_w);

	//Bewegung Rad

	w_wfl = 
	dw_wfl = (F_wfl*(-r_w) - tanh(w_wfl*d_b)*mu_b*F_b)*(1.0/I_w);
	r_wfl = 
	dr_wfl = dp_b + r_wfl*w_wfl - du_wfl;

	//Beschleunigung Feder-Daempfer
	ddu = 2.0*(sqrt(ddp_wcfl*ddp_wcfl));

	//Bewegung der Karosserie
	
	Fo_wfl = (F_wfl^ddp_wfl)*(~(ddp_wfl));
	Fsd_wfl = u*k + du*d;
	p_b = 
	dp_b = 
	ddp_b = typeof(grade<1>((Fo_wfl + Fsd_wfl)*(1.0/m_b) + g);
	w_b =
	dw_b = typeof(grade<2>(rigidbody::euler<SCHWERPUNKTKOORDINATEN>(w_b, Fsd_wfl + Fo_wfl - (u_wfl-u_wfr)*k_arb);*/
}
