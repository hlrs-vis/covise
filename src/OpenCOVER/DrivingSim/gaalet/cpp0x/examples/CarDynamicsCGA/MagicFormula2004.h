/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///----------------------------------
///Author: Florian Seybold, 2009
///www.hlrs.de
///----------------------------------
///
///Pacejka's magic formula as in "Tyre and Vehicle Dynamics", 2nd edition, 2006

#ifndef __MagicFormula2004_h
#define __MagicFormula2004_h

#include "gaalet.h"
#include <vector>
#include <map>
#include <fstream>
#include <sstream>

namespace magicformula2004
{
//typedef gaalet::algebra< gaalet::signature<3,0> > em;
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

struct TyrePropertyPack
{
    static const int TYRE_LEFT = 1;
    static const int TYRE_RIGHT = -1;

    TyrePropertyPack(const int tyre_side)
        : p_Cx(1)
        , p_Dx(2)
        , p_Ex(4)
        , p_Kx(3)
        , p_Hx(2)
        , p_Vx(2)
        , p_Cy(1)
        , p_Dy(3)
        , p_Ey(4)
        , p_Ky(7)
        , p_Hy(2)
        , p_Vy(4)
        , q_Bz(10)
        , q_Cz(1)
        , q_Dz(11)
        , q_Ez(5)
        , q_Hz(4)
        , q_sx(3)
        , q_sy(2)
        , r_Bx(3)
        , r_Cx(1)
        , r_Ex(2)
        , r_Hx(1)
        , r_By(4)
        , r_Cy(1)
        , r_Ey(2)
        , r_Hy(1)
        , r_Vy(6)
        , s_sz(4)
    {
        init(tyre_side);
    }

    TyrePropertyPack(const std::string &filename, const int tyre_side)
        : p_Cx(1)
        , p_Dx(2)
        , p_Ex(4)
        , p_Kx(3)
        , p_Hx(2)
        , p_Vx(2)
        , p_Cy(1)
        , p_Dy(3)
        , p_Ey(4)
        , p_Ky(7)
        , p_Hy(2)
        , p_Vy(4)
        , q_Bz(10)
        , q_Cz(1)
        , q_Dz(11)
        , q_Ez(5)
        , q_Hz(4)
        , q_sx(3)
        , q_sy(2)
        , r_Bx(3)
        , r_Cx(1)
        , r_Ex(2)
        , r_Hx(1)
        , r_By(4)
        , r_Cy(1)
        , r_Ey(2)
        , r_Hy(1)
        , r_Vy(6)
        , s_sz(4)
    {
        //[LONGITUDINAL_COEFFICIENTS]
        tpf["PCX1"] = &(p_Cx[0]);
        tpf["PDX1"] = &(p_Dx[0]);
        tpf["PDX2"] = &(p_Dx[1]);
        //tpf["PDX3"] = &(p_Dx[2]);
        tpf["PEX1"] = &(p_Ex[0]);
        tpf["PEX2"] = &(p_Ex[1]);
        tpf["PEX3"] = &(p_Ex[2]);
        tpf["PEX4"] = &(p_Ex[3]);
        tpf["PKX1"] = &(p_Kx[0]);
        tpf["PKX2"] = &(p_Kx[1]);
        tpf["PKX3"] = &(p_Kx[2]);
        tpf["PHX1"] = &(p_Hx[0]);
        tpf["PHX2"] = &(p_Hx[1]);
        tpf["PVX1"] = &(p_Vx[0]);
        tpf["PVX2"] = &(p_Vx[1]);
        tpf["RBX1"] = &(r_Bx[0]);
        tpf["RBX2"] = &(r_Bx[1]);
        tpf["RBX3"] = &(r_Bx[2]);
        tpf["RCX1"] = &(r_Cx[0]);
        tpf["REX1"] = &(r_Ex[0]);
        tpf["REX2"] = &(r_Ex[1]);
        tpf["RHX1"] = &(r_Hx[0]);
        //tpf["PTX1"] = &();
        //tpf["PTX2"] = &();
        //tpf["PTX3"] = &();
        tpf["QSX1"] = &(q_sx[0]);
        tpf["QSX2"] = &(q_sx[1]);
        tpf["QSX3"] = &(q_sx[2]);
        //[LATERAL_COEFFICIENTS]
        tpf["PCY1"] = &(p_Cy[0]);
        tpf["PDY1"] = &(p_Dy[0]);
        tpf["PDY2"] = &(p_Dy[1]);
        tpf["PDY3"] = &(p_Dy[2]);
        tpf["PEY1"] = &(p_Ey[0]);
        tpf["PEY2"] = &(p_Ey[1]);
        tpf["PEY3"] = &(p_Ey[2]);
        tpf["PEY4"] = &(p_Ey[3]);
        tpf["PKY1"] = &(p_Ky[0]);
        tpf["PKY2"] = &(p_Ky[1]);
        tpf["PKY3"] = &(p_Ky[2]);
        tpf["PKY4"] = &(p_Ky[3]);
        tpf["PKY5"] = &(p_Ky[4]);
        tpf["PKY6"] = &(p_Ky[5]);
        tpf["PKY7"] = &(p_Ky[6]);
        tpf["PHY1"] = &(p_Hy[0]);
        tpf["PHY2"] = &(p_Hy[1]);
        //tpf["PHY3"] = &(p_Hy[2]);
        tpf["PVY1"] = &(p_Vy[0]);
        tpf["PVY2"] = &(p_Vy[1]);
        tpf["PVY3"] = &(p_Vy[2]);
        tpf["PVY4"] = &(p_Vy[3]);
        tpf["RBY1"] = &(r_By[0]);
        tpf["RBY2"] = &(r_By[1]);
        tpf["RBY3"] = &(r_By[2]);
        tpf["RBY4"] = &(r_By[3]);
        tpf["RCY1"] = &(r_Cy[0]);
        tpf["REY1"] = &(r_Ey[0]);
        tpf["REY2"] = &(r_Ey[1]);
        tpf["RHY1"] = &(r_Hy[0]);
        //tpf["RHY2"] = &(r_Hy[1]);
        tpf["RVY1"] = &(r_Vy[0]);
        tpf["RVY2"] = &(r_Vy[1]);
        tpf["RVY3"] = &(r_Vy[2]);
        tpf["RVY4"] = &(r_Vy[3]);
        tpf["RVY5"] = &(r_Vy[4]);
        tpf["RVY6"] = &(r_Vy[5]);
        //tpf["PTY1"] = &();
        //tpf["PTY2"] = &();
        //[ROLLING_COEFFICIENTS]
        tpf["QSY1"] = &(q_sy[0]);
        tpf["QSY2"] = &(q_sy[1]);
        //tpf["QSY3"] = &(q_sy[2]);
        //tpf["QSY4"] = &(q_sy[3]);
        //[ALIGNING_COEFFICIENTS]
        tpf["QBZ1"] = &(q_Bz[0]);
        tpf["QBZ2"] = &(q_Bz[1]);
        tpf["QBZ3"] = &(q_Bz[2]);
        tpf["QBZ4"] = &(q_Bz[3]);
        tpf["QBZ5"] = &(q_Bz[4]);
        tpf["QBZ9"] = &(q_Bz[8]);
        tpf["QBZ10"] = &(q_Bz[9]);
        tpf["QCZ1"] = &(q_Cz[0]);
        tpf["QDZ1"] = &(q_Dz[0]);
        tpf["QDZ2"] = &(q_Dz[1]);
        tpf["QDZ3"] = &(q_Dz[2]);
        tpf["QDZ4"] = &(q_Dz[3]);
        tpf["QDZ6"] = &(q_Dz[5]);
        tpf["QDZ7"] = &(q_Dz[6]);
        tpf["QDZ8"] = &(q_Dz[7]);
        tpf["QDZ9"] = &(q_Dz[8]);
        tpf["QDZ10"] = &(q_Dz[9]);
        tpf["QDZ11"] = &(q_Dz[10]);
        tpf["QEZ1"] = &(q_Ez[0]);
        tpf["QEZ2"] = &(q_Ez[1]);
        tpf["QEZ3"] = &(q_Ez[2]);
        tpf["QEZ4"] = &(q_Ez[3]);
        tpf["QEZ5"] = &(q_Ez[4]);
        tpf["QHZ1"] = &(q_Hz[0]);
        tpf["QHZ2"] = &(q_Hz[1]);
        tpf["QHZ3"] = &(q_Hz[2]);
        tpf["QHZ4"] = &(q_Hz[3]);
        tpf["SSZ1"] = &(s_sz[0]);
        tpf["SSZ2"] = &(s_sz[1]);
        tpf["SSZ3"] = &(s_sz[2]);
        tpf["SSZ4"] = &(s_sz[3]);
        //tpf["QTZ1"] = &();
        //tpf["MBELT"] = &();

        tpf["UNLOADED_RADIUS"] = &R_0;
        tpf["FNOMIN"] = &F_z0;
        tpf["VERTICAL_STIFFNESS"] = &C_Fz;
        tpf["VERTICAL_DAMPING"] = &D_Fz;

        for (std::map<std::string, double *>::iterator tpfIt = tpf.begin(); tpfIt != tpf.end(); ++tpfIt)
        {
            *(tpfIt->second) = 0.0;
        }

        init(tyre_side);

        std::ifstream tyreFile(filename.c_str(), std::ios::in);
        if (tyreFile.fail())
        {
            std::cerr << "TyrePropertyPack: Couldn't open tyre property file \"" << filename << "\". Setting default properties..." << std::endl;
            //init();
            return;
        }

        while (tyreFile.good())
        {
            std::string line;
            std::getline(tyreFile, line);
            if (line.substr(0, 1) == "$" || line.substr(0, 1) == "[" || line.substr(0, 1) == "{")
                continue;

            std::stringstream lineStream(line);
            std::string spec;
            std::string equal;
            double value;

            lineStream >> spec;
            lineStream >> equal;
            lineStream >> value;

            std::map<std::string, double *>::iterator tpfIt = tpf.find(spec);
            if (tpfIt != tpf.end())
            {
                *(tpfIt->second) = value;
            }
        }

        //Where are these parameters???
        //p_Ky[3] = 1.0;
    }

    void init(const int &tyre_side)
    {
        if (tyre_side == TYRE_LEFT)
        {
            //PTX1     = 2.3657     $Relaxation length SigKap0/Fz at Fznom
            //PTX2     = 1.4112     $Variation of SigKap0/Fz with load
            //PTX3     = 0.56626    $Variation of SigKap0/Fz with exponent of load
            //PHY3     = 0.031415   $Variation of shift Shy with camber
            //PTY1     = 2.1439     $Peak value of relaxation length SigAlp0/R0
            //PTY2     = 1.9829     $Value of Fz/Fznom where SigAlp0 is extreme
            //QTZ1     = 0.2        $Gyration torque constant
            //MBELT    = 5.4        $Belt mass of the wheel

            p_Cx[0] = 1.6411;
            p_Dx[0] = 1.1739;
            p_Dx[1] = -0.16395;
            p_Ex[0] = 0.46403;
            p_Ex[1] = 0.25022;
            p_Ex[2] = 0.067842;
            p_Ex[3] = -3.7604e-005;
            p_Kx[0] = 22.303;
            p_Kx[1] = 0.48896;
            p_Kx[2] = 0.21253;
            p_Hx[0] = 0.0012297;
            p_Hx[1] = 0.0004318;
            p_Vx[0] = -8.8098e-006;
            p_Vx[1] = 1.862e-005;
            p_Cy[0] = 1.3507;
            p_Dy[0] = 1.0489;
            p_Dy[1] = -0.18033;
            p_Dy[2] = -2.8821;
            p_Ey[0] = -0.0074722;
            p_Ey[1] = -0.0063208;
            p_Ey[2] = -9.9935;
            p_Ey[3] = -760.14;
            p_Ky[0] = 21.92;
            p_Ky[1] = 2.0012;
            p_Ky[2] = -0.024778;
            p_Ky[3] = 2.0;
            p_Ky[4] = 0.0;
            p_Ky[5] = 2.5;
            p_Ky[6] = 0.0;
            p_Hy[0] = 0.0026747;
            p_Hy[1] = 8.9094e-005;
            p_Vy[0] = 0.037318;
            p_Vy[1] = -0.010049;
            p_Vy[2] = -0.32931;
            p_Vy[3] = -0.69553;

            q_Bz[0] = 10.904;
            q_Bz[1] = -1.8412;
            q_Bz[2] = -0.52041;
            q_Bz[3] = 0.039211;
            q_Bz[4] = 0.41511;
            q_Bz[8] = 8.9846;
            q_Bz[9] = 0.0;
            q_Cz[0] = 1.2136;
            q_Dz[0] = 0.093509;
            q_Dz[1] = -0.0092183;
            q_Dz[2] = -0.057061;
            q_Dz[3] = 0.73954;
            q_Dz[5] = -0.0067783;
            q_Dz[6] = 0.0052254;
            q_Dz[7] = -0.18175;
            q_Dz[8] = 0.029952;
            q_Dz[9] = 0.0;
            q_Dz[10] = 0.0;
            q_Ez[0] = -1.5697;
            q_Ez[1] = 0.33394;
            q_Ez[2] = 0.0;
            q_Ez[3] = 0.26711;
            q_Ez[4] = -3.594;
            q_Hz[0] = 0.0047326;
            q_Hz[1] = 0.0026687;
            q_Hz[2] = 0.11998;
            q_Hz[3] = 0.059083;
            q_sx[0] = 0.0;
            q_sx[1] = 0.0;
            q_sx[2] = 0.0;
            q_sy[0] = 0.01;
            q_sy[1] = 0.0;

            r_Bx[0] = 13.276;
            r_Bx[1] = -13.778;
            r_Bx[2] = 0.0;
            r_Cx[0] = 1.2568;
            r_Ex[0] = 0.65225;
            r_Ex[1] = -0.24948;
            r_Hx[0] = 0.0050722;
            r_By[0] = 7.1433;
            r_By[1] = 9.1916;
            r_By[2] = -0.027856;
            r_By[3] = 0.0;
            r_Cy[0] = 1.0719;
            r_Ey[0] = -0.27572;
            r_Ey[1] = 0.32802;
            r_Hy[0] = 5.7448e-006;
            r_Vy[0] = -0.027825;
            r_Vy[1] = 0.053604;
            r_Vy[2] = -0.27568;
            r_Vy[3] = 12.12;
            r_Vy[4] = 1.9;
            r_Vy[5] = -10.704;
            s_sz[0] = 0.033372;
            s_sz[1] = 0.0043624;
            s_sz[2] = 0.56742;
            s_sz[3] = -0.24116;
        }
        else if (tyre_side == TYRE_RIGHT)
        {
            p_Cx[0] = 1.6411;
            p_Dx[0] = 1.1739;
            p_Dx[1] = -0.16395;
            p_Ex[0] = 0.46403;
            p_Ex[1] = 0.25022;
            p_Ex[2] = 0.067842;
            p_Ex[3] = -3.7604e-005;
            p_Kx[0] = 22.303;
            p_Kx[1] = 0.48896;
            p_Kx[2] = 0.21253;
            p_Hx[0] = 0.0012297;
            p_Hx[1] = 0.0004318;
            p_Vx[0] = -8.8098e-006;
            p_Vx[1] = 1.862e-005;
            p_Cy[0] = 1.3507;
            p_Dy[0] = 1.0489;
            p_Dy[1] = -0.18033;
            p_Dy[2] = -2.8821;
            p_Ey[0] = -0.0074722;
            p_Ey[1] = -0.0063208;
            p_Ey[2] = 9.9935;
            p_Ey[3] = -760.14;
            p_Ky[0] = 21.92;
            p_Ky[1] = 2.0012;
            p_Ky[2] = -0.024778;
            p_Ky[3] = 2.0;
            p_Ky[4] = 0.0;
            p_Ky[5] = 2.5;
            p_Ky[6] = 0.0;
            p_Hy[0] = -0.0026747;
            p_Hy[1] = -8.9094e-005;
            p_Vy[0] = -0.037318;
            p_Vy[1] = 0.010049;
            p_Vy[2] = -0.32931;
            p_Vy[3] = -0.69553;

            q_Bz[0] = 10.904;
            q_Bz[1] = -1.8412;
            q_Bz[2] = -0.52041;
            q_Bz[3] = -0.039211;
            q_Bz[4] = 0.41511;
            q_Bz[8] = 8.9846;
            q_Bz[9] = 0.0;
            q_Cz[0] = 1.2136;
            q_Dz[0] = 0.093509;
            q_Dz[1] = -0.0092183;
            q_Dz[2] = 0.057061;
            q_Dz[3] = 0.73954;
            q_Dz[5] = 0.0067783;
            q_Dz[6] = -0.0052254;
            q_Dz[7] = -0.18175;
            q_Dz[8] = 0.029952;
            q_Dz[9] = 0.0;
            q_Dz[10] = 0.0;
            q_Ez[0] = -1.5697;
            q_Ez[1] = 0.33394;
            q_Ez[2] = 0.0;
            q_Ez[3] = -0.26711;
            q_Ez[4] = -3.594;
            q_Hz[0] = -0.0047326;
            q_Hz[1] = -0.0026687;
            q_Hz[2] = 0.11998;
            q_Hz[3] = 0.059083;
            q_sx[0] = 0.0;
            q_sx[1] = 0.0;
            q_sx[2] = 0.0;
            q_sy[0] = 0.01;
            q_sy[1] = 0.0;

            r_Bx[0] = 13.276;
            r_Bx[1] = -13.778;
            r_Bx[2] = 0.0;
            r_Cx[0] = 1.2568;
            r_Ex[0] = 0.65225;
            r_Ex[1] = -0.24948;
            r_Hx[0] = -0.0050722;
            r_By[0] = 7.1433;
            r_By[1] = 9.1916;
            r_By[2] = 0.027856;
            r_By[3] = 0.0;
            r_Cy[0] = 1.0719;
            r_Ey[0] = -0.27572;
            r_Ey[1] = 0.32802;
            r_Hy[0] = 5.7448e-006;
            r_Vy[0] = 0.027825;
            r_Vy[1] = -0.053604;
            r_Vy[2] = -0.27568;
            r_Vy[3] = 12.12;
            r_Vy[4] = 1.9;
            r_Vy[5] = -10.704;
            s_sz[0] = -0.033372;
            s_sz[1] = 0.0043624;
            s_sz[2] = 0.56742;
            s_sz[3] = -0.24116;
        }

        R_0 = 0.325;
        F_z0 = 3000.0;
        C_Fz = 200000.0;
        D_Fz = 100.0;
        V_0 = sqrt(9.81 * R_0);
    }

    std::vector<double> p_Cx;
    std::vector<double> p_Dx;
    std::vector<double> p_Ex;
    std::vector<double> p_Kx;
    std::vector<double> p_Hx;
    std::vector<double> p_Vx;
    std::vector<double> p_Cy;
    std::vector<double> p_Dy;
    std::vector<double> p_Ey;
    std::vector<double> p_Ky;
    std::vector<double> p_Hy;
    std::vector<double> p_Vy;

    std::vector<double> q_Bz;
    std::vector<double> q_Cz;
    std::vector<double> q_Dz;
    std::vector<double> q_Ez;
    std::vector<double> q_Hz;
    std::vector<double> q_sx;
    std::vector<double> q_sy;

    std::vector<double> r_Bx;
    std::vector<double> r_Cx;
    std::vector<double> r_Ex;
    std::vector<double> r_Hx;
    std::vector<double> r_By;
    std::vector<double> r_Cy;
    std::vector<double> r_Ey;
    std::vector<double> r_Hy;
    std::vector<double> r_Vy;
    std::vector<double> s_sz;

    std::map<std::string, double *> tpf;

    double R_0;
    double F_z0;
    double C_Fz;
    double D_Fz;
    double V_0;
};

struct ContactWrench
{
    typedef cm::mv<1, 2, 3, 4, 5, 6>::type result_wrench_t;

    ContactWrench(const TyrePropertyPack &setPars)
        : pars(setPars)
    {
    }

    //result_wrench_t operator()(const double& r, const cm::mv<0,3,5,6>::type& R, const cm::mv<1,2,4,5>::type& V) const {
    //result_wrench_t operator()(const Plane& P, const S_type& V) const {
    S_type operator()(const Plane &P, const S_type &V) const
    {
        //r: tyre frame, R: surface frame, V: tyre frame
        //P: tyre frame, V: tyre frame

        Plane Pn = eval(P * (1.0 / sqrt(eval(P & P))));
        double r = eval((-1.0) * (Pn & e0));
        Point p = (e0 ^ Pn) * (~Pn) - 0.5 * r * r * einf;

        //auto Vc = grade<1>(V);
        auto Vc = (V & e0);
        //std::cout << "Vc: " << Vc << std::endl;
        //cm::mv<1,2>::type Vs = part<1,2>(V) + cm::mv<4>::type({-pars.R_0})*part<5>(V);  //assumption R_e = R_0
        auto p_R0 = pars.R_0 * e3 + 0.5 * pars.R_0 * pars.R_0 * einf + e0;
        auto Vs = (V & p_R0);

        //singularity avoidance
        double epsilon_Vx = 0.1;
        double epsilon_x = 0.1;
        double epsilon_y = 0.1;
        double epsilon_V = 0.1;
        double epsilon_K = 0.1;

        //spin values
        double zeta_0 = 1.0;
        double zeta_1 = 1.0;
        double zeta_2 = 1.0;
        double zeta_3 = 1.0;
        double zeta_4 = 1.0;
        double zeta_5 = 1.0;
        double zeta_6 = 1.0;
        double zeta_7 = 1.0;
        double zeta_8 = 1.0;

        //radial tyre deflection, assumption: r_f = R_0
        double rho_z = std::max(pars.R_0 - r, 0.0);

        //Velocity contact point
        //double V_c = magnitude(part<2, 0x0201>(Vc)).element<0x00>();
        double V_c = eval(magnitude(part<1, 2>(Vc)));
        //double V_cx = Vc.element<0x01>();
        double V_cx = Vc.element<1>();
        //double V_cy = Vc.element<0x02>();
        double V_cy = Vc.element<2>();

        //Velocity point S
        double V_sx = Vs.element<1>();
        //double V_sx = eval(magnitude(part<1,4>(Vs)));
        //double V_sx = eval(grade<0>((Ie*Vs)&(Pn^e2)));

        //Speed of rolling
        double V_r = V_cx - V_sx;

        //Camber angle
        //double gamma = asin((R*gealg::mv<1, 0x04>::type(1.0)*(~R)).element<0x02>());

        //nominal load
        double F_z0_prime = pars.F_z0;

        //normal load
        //double F_z = pars.p_z[0]*(F_z0_prime/R_0)*rho_z; //+ 1000.0*V.element<0x04>();
        double F_z = pars.C_Fz * rho_z + pars.D_Fz * V.element<4>();

        //normalised change in vertical load
        double df_z = (F_z - F_z0_prime) / F_z0_prime;

        //lateral slip with small quantity in denominator
        double alpha_star = -V_cy / (fabs(V_cx) + epsilon_x);

        //spin due to camber angle
        //double gamma_star = sin(gamma);
        //double gamma_star = eval(grade<1>(R*cm::mv<1>::type({1.0})*(~R)))[1];
        double gamma_star = ((Pn ^ e3) * (~Pn)).element<2>();

        //longitudinal slip ratio
        double kappa = -V_sx / (fabs(V_cx) + epsilon_x);

        //handling of aligning torque
        double cos_prime_alpha = V_cx / (V_c + epsilon_V);

        //longitudinal force (pure longitudinal slip)
        double S_Vx = F_z * (pars.p_Vx[0] + pars.p_Vx[1] * df_z) * (fabs(V_cx) / (epsilon_Vx + fabs(V_cx))) * zeta_1;
        double S_Hx = (pars.p_Hx[0] + pars.p_Hx[1] * df_z);
        double kappa_x = kappa + S_Hx;
        double K_xk = F_z * (pars.p_Kx[0] + pars.p_Kx[1] * df_z) * exp(pars.p_Kx[2] * df_z);
        double E_x = (pars.p_Ex[0] + pars.p_Ex[1] * df_z + pars.p_Ex[2] * df_z * df_z) * (1.0 - pars.p_Ex[3] * ((kappa_x >= 0) ? 1.0 : -1.0));
        double mu_x = (pars.p_Dx[0] + pars.p_Dx[1] * df_z);
        double D_x = mu_x * F_z * zeta_1;
        double C_x = pars.p_Cx[0];
        double B_x = K_xk / (C_x * D_x + epsilon_x);
        double F_x0 = D_x * sin(C_x * atan(B_x * kappa_x - E_x * (B_x * kappa_x - atan(B_x * kappa_x)))) + S_Vx;

        //lateral force (pure side slip)
        double S_Vyg = F_z * (pars.p_Vy[2] + pars.p_Vy[3] * df_z) * gamma_star * zeta_2;
        double S_Vy = F_z * (pars.p_Vy[0] + pars.p_Vy[1] * df_z) * zeta_2 + S_Vyg;
        double K_yg0 = F_z * (pars.p_Ky[5] + pars.p_Ky[6] * df_z);
        double K_ya = pars.p_Ky[0] * F_z0_prime * sin(pars.p_Ky[3] * atan(F_z / ((pars.p_Ky[1] + pars.p_Ky[4] * gamma_star * gamma_star) * F_z0_prime))) / (1.0 + pars.p_Ky[2] * gamma_star * gamma_star) * zeta_3;
        double S_Hy = (pars.p_Hy[0] + pars.p_Hy[1] * df_z) + (K_yg0 * gamma_star - S_Vyg) * zeta_0 / (K_ya + epsilon_K) + zeta_4 - 1.0;
        double alpha_y = alpha_star + S_Hy;
        double C_y = pars.p_Cy[0];
        double mu_y = ((pars.p_Dy[0] + pars.p_Dy[1] * df_z) / (1.0 + pars.p_Dy[2] * gamma_star * gamma_star));
        double D_y = mu_y * F_z * zeta_2;
        double E_y = (pars.p_Ey[0] + pars.p_Ey[1] * df_z) * (1.0 + pars.p_Ey[4] * gamma_star * gamma_star - (pars.p_Ey[2] + pars.p_Ey[3] * gamma_star) * ((alpha_y >= 0) ? 1.0 : -1.0));
        double B_y = K_ya / (C_y * D_y + epsilon_y);
        double F_y0 = D_y * sin(C_y * atan(B_y * alpha_y - E_y * (B_y * alpha_y - atan(B_y * alpha_y)))) + S_Vy;

        //aligning torque (pure side slip)
        double S_Ht = pars.q_Hz[0] + pars.q_Hz[1] * df_z + (pars.q_Hz[2] + pars.q_Hz[3] * df_z) * gamma_star;
        double alpha_t = alpha_star + S_Ht;
        double B_t = (pars.q_Bz[0] + pars.q_Bz[1] * df_z + pars.q_Bz[2] * df_z * df_z) * (1.0 + pars.q_Bz[4] * fabs(gamma_star) + pars.q_Bz[5] * gamma_star * gamma_star);
        double C_t = pars.q_Cz[0];
        double D_t0 = F_z * (pars.R_0 / F_z0_prime) * (pars.q_Dz[0] + pars.q_Dz[1] * df_z) * ((V_cx >= 0) ? 1.0 : -1.0);
        double D_t = D_t0 * (1.0 + pars.q_Dz[2] * fabs(gamma_star) + pars.q_Dz[3] * gamma_star * gamma_star) * zeta_5;
        double E_t = (pars.q_Ez[0] + pars.q_Ez[1] * df_z + pars.q_Ez[2] * df_z * df_z) * (1.0 + (pars.q_Ez[3] + pars.q_Ez[4] * gamma_star) * 2.0 / M_PI * atan(B_t * C_t * alpha_t));
        double t_0 = D_t * cos(C_t * atan(B_t * alpha_t - E_t * (B_t * alpha_t - atan(B_t * alpha_t)))) * cos_prime_alpha;
        double M_z0_prime = -t_0 * F_y0;
        double B_r = (pars.q_Bz[8] + pars.q_Bz[9] * B_y * C_y) * zeta_6;
        double C_r = zeta_7;
        double D_r = F_z * pars.R_0 * ((pars.q_Dz[5] + pars.q_Dz[6] * df_z) * zeta_2 + (pars.q_Dz[7] + pars.q_Dz[8] * df_z) * gamma_star * zeta_0
                                       + (pars.q_Dz[9] + pars.q_Dz[10] * df_z) * gamma_star * fabs(gamma_star) * zeta_0) * cos_prime_alpha * ((V_cx >= 0) ? 1.0 : -1.0) + zeta_8 - 1.0;
        double K_ya_prime = K_ya + epsilon_K;
        double S_Hf = S_Hy + S_Vy / K_ya_prime;
        double alpha_r = alpha_star + S_Hf;
        double M_zr0 = D_r * cos(C_r * atan(B_r * alpha_r));
        double K_za0 = D_t0 * K_ya; //??? K_za0 = D_t0*K_ya0 ???
        double K_zg0 = F_z * pars.R_0 * (pars.q_Dz[7] + pars.q_Dz[8] * df_z) - D_t0 * K_yg0;
        double M_z0 = M_z0_prime + M_zr0;

        //longitudinal force (combined slip)
        double B_xa = (pars.r_Bx[0] + pars.r_Bx[2] * gamma_star * gamma_star) * cos(atan(pars.r_Bx[1] * kappa));
        double C_xa = pars.r_Cx[0];
        double E_xa = pars.r_Ex[0] + pars.r_Ex[1] * df_z;
        double S_Hxa = pars.r_Hx[0];
        double alpha_S = alpha_star + S_Hxa;
        double G_xa0 = cos(C_xa * atan(B_xa * S_Hxa - E_xa * (B_xa * S_Hxa - atan(B_xa * S_Hxa))));
        double G_xa = cos(C_xa * atan(B_xa * alpha_S - E_xa * (B_xa * alpha_S - atan(B_xa * alpha_S)))) / G_xa0;
        double F_x = G_xa * F_x0;

        //lateral force (combined slip)
        double B_yk = (pars.r_By[0] + pars.r_By[3] * gamma_star * gamma_star) * cos(atan(pars.r_By[1] * (alpha_star - pars.r_By[2])));
        double C_yk = pars.r_Cy[0];
        double D_Vyk = mu_y * F_z * (pars.r_Vy[0] + pars.r_Vy[1] * df_z + pars.r_Vy[2] * gamma_star) * cos(atan(pars.r_Vy[3] * alpha_star)) * zeta_2;
        double E_yk = pars.r_Ey[0] + pars.r_Ey[1] * df_z;
        double S_Hyk = pars.r_Hy[0] + pars.r_Hy[1] * df_z;
        double kappa_S = kappa + S_Hyk;
        double S_Vyk = D_Vyk * sin(pars.r_Vy[4] * atan(pars.r_Vy[5] * kappa));
        double G_yk0 = cos(C_yk * atan(B_yk * S_Hyk - E_yk * (B_yk * S_Hyk - atan(B_yk * S_Hyk))));
        double G_yk = cos(C_yk * atan(B_yk * kappa_S - E_yk * (B_yk * kappa_S - atan(B_yk * kappa_S)))) / G_yk0;
        double F_y = G_yk * F_y0 + S_Vyk;

        //overturning couple
        double M_x = F_z * pars.R_0 * (pars.q_sx[0] - pars.q_sx[1] * gamma_star + pars.q_sx[2] * F_y / F_z0_prime);

        //rolling resistance moment
        double M_y = -F_z * pars.R_0 * (pars.q_sy[0] * atan(V_r / pars.V_0) + pars.q_sy[1] * F_x / F_z0_prime);

        //aligning torque
        double alpha_teq = sqrt(alpha_t * alpha_t + pow(K_xk / K_ya_prime, 2) * kappa * kappa) * ((alpha_t >= 0) ? 1.0 : -1.0);
        double alpha_req = sqrt(alpha_r * alpha_r + pow(K_xk / K_ya_prime, 2) * kappa * kappa) * ((alpha_r >= 0) ? 1.0 : -1.0);
        double t = D_t * cos(C_t * atan(B_t * alpha_teq - E_t * (B_t * alpha_teq - atan(B_t * alpha_teq)))) * cos_prime_alpha;
        double F_y_prime = F_y - S_Vyk;
        double M_z_prime = -t * F_y_prime;
        double M_zr = D_r * cos(C_r * atan(B_r * alpha_req));
        double s = pars.R_0 * (pars.s_sz[0] + pars.s_sz[1] * (F_y / F_z0_prime) + (pars.s_sz[2] + pars.s_sz[3] * df_z) * gamma_star);
        double M_z = M_z_prime + M_zr + s * F_x;

        //result_wrench_t wrench;
        //wrench[0] = F_x;
        //wrench[1] = F_y;
        //wrench[3] = -F_z;

        //wrench[2] = M_z;
        //wrench[4] = -M_y;
        //wrench[5] = M_x;

        S_type F = (F_x * e1 + F_y * e2 + F_z * e3) * e0 - (M_x * e1 + M_y * e2 + M_z * e3) * Ie;

        //return std::move(wrench);
        return std::move(F);
    }

    TyrePropertyPack pars;
};
}
#endif
