/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          solve.h  -  solve heat conduction
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include "solve.h"
#include "grid.h"
#include "solid.h"
#include "simul.h"
#include "laser.h"
#include "material.h"
#include "main.h"

// ***************************************************************************
// class for heat conduction in zeta-direction
// ***************************************************************************

// constructor

HeatConduction::HeatConduction()
{
}

// ***************************************************************************
// calculate heat conduction in zeta-direction of element (i,j)
// ***************************************************************************

// to do: temperature dependent density not implemented correctly yet!
//        better implementation of volume evaporation
//        locally varying material properties?

int HeatConduction::Solve(int i, int j, prec &vn)
{
    int k, ia, ipa, ja, jpa;
    prec hsaisai, hsaizta, hsaieta, hetaeta, hetazta, qfluxt, dvol,
        sumr, term, g11, g12, g22, r1t, vmax, thetamin, thetamin0,
        dthetamin, thetamax, thetamax0, dthetamax, gamma, abscoeff;
    cmplx index;
    rmmatrix MetricTensor(3, 3);
    TPoint3D LaplaceSai;
    bool bAnalyt;

    TempSai.Reallocate(Grid.kVolNodes); // reallocate arrays
    PowerDensity.Reallocate(Grid.kVolNodes);
    m.Reallocate(Grid.kVolNodes + 2);

    if (Solid(i, j, Grid.kVolNodes).ndNode.z < Grid.ndWpThickness)
        m(Grid.kVolNodes).a = 0.0; // temp fixed
    else
        m(Grid.kVolNodes).a = -1.0; // drilled through => gradient fixed
    m(Grid.kVolNodes).b = 1.0;
    m(Grid.kVolNodes).c = 0.0;
    m(Grid.kVolNodes).rhs = 0.0;

    for (k = Grid.kVolNodes - 1; k >= 1; k--) // matrix setup
    {
        Solid.CalcBaseVect(i, j, k); // base vectors
        abszta = Solid.CalcMetricTensor(i, j, k, MetricTensor, LaplaceSai);
        SolidOld.CalcTempDerivative(i, j, k, TempSai(k), hsaisai, hetaeta, hsaieta,
                                    hsaizta, hetazta); // calculate T-derivatives

        if (k > 1) // inner nodes
        {
            g11 = Solid(i, j, k).xsai.Norm();
            g22 = Solid(i, j, k).xeta.Norm();
            g12 = Solid(i, j, k).xsai * Solid(i, j, k).xeta;
            dvol = sqrt(g11 * g22 - g12 * g12) / abszta; // volume of cell
            PowerDensity(k) = Solid(i, j, k).ndHeat / dvol; // convert to power density
            gamma = MetricTensor(0, 0) * hsaisai + MetricTensor(1, 1) * hetaeta + MetricTensor(0, 1) * hsaieta + MetricTensor(0, 2) * hsaizta + MetricTensor(1, 2) * hetazta + LaplaceSai.x * TempSai(k).x + LaplaceSai.y * TempSai(k).y;

            m(k).at = 0.5 * LaplaceSai.z - MetricTensor(2, 2);
            m(k).bt = 2.0 * MetricTensor(2, 2);
            m(k).ct = -0.5 * LaplaceSai.z - MetricTensor(2, 2);
            // explicit expression
            m(k).rhst = gamma + PowerDensity(k) * Laser.ndPower;
            // +Q/Nk
            m(k).ndTemp = Solid(i, j, k).ndTemp; // temperature estimate
        }
        else // surface node
        {
            dvol = 0.5 * SolSurf(i, j).ndArea / abszta; // volume of (half) cell
            // convert to power density
            PowerDensity(k) = Solid(i, j, k).ndHeat / dvol;
            gamma = MetricTensor(0, 0) * hsaisai + MetricTensor(1, 1) * hetaeta + MetricTensor(0, 1) * hsaieta + MetricTensor(0, 2) * hsaizta + MetricTensor(1, 2) * hetazta + LaplaceSai.x * TempSai(k).x + LaplaceSai.y * TempSai(k).y;
            r1t = 2.0 * (MetricTensor(0, 2) * TempSai(k).x + MetricTensor(1, 2) * TempSai(k).y);

            qfluxt = (SolSurf(i, j).ndDirAbs + SolSurf(i, j).ndMultAbs) * Laser.ndPower * abszta; // for maximum ablation velocity

            m(k).at = 0.0;
            m(k).ct = -2.0 * MetricTensor(2, 2) - LaplaceSai.z;
            m(k).bt = -m(k).ct;
            m(k).rhst = gamma + PowerDensity(k) * Laser.ndPower + r1t;
            m(k).ndTemp = Solid(i, j, k).ndTemp;
        }
    }

    sumr = 0.0;
    Solid.LocalKlTangent = 0; // calc inner grid direction by averaging
    for (ia = -Grid.iWarpAverage; ia <= Grid.iWarpAverage; ia++)
    {
        ipa = i + ia;
        if (ipa >= 1 && ipa <= Grid.iVolNodes)
        {
            for (ja = -Grid.iWarpAverage; ja <= Grid.iWarpAverage; ja++)
            {
                if (ia != 0 || ja != 0)
                {
                    jpa = j + ja;
                    if (jpa >= 1 && jpa <= Grid.jVolNodes)
                    {
                        term = 1.0 / pow((SolidOld(i, j, 1).ndNode - SolidOld(ipa, jpa, 1).ndNode).Norm(),
                                         Grid.iWarpPower);
                        sumr += term;
                        Solid.LocalKlTangent += SolSurf(ipa, jpa).Normal * term;
                    }
                }
            }
        }
    }
    Solid.LocalKlTangent.z += sumr * Grid.TangentDir;
    Solid.LocalKlTangent.Normalize();

    bAnalyt = false;
    // maximum recession
    saitmax.z = -qfluxt / Material.ndStefan.Get(m(1).ndTemp);
    // speed
    vmax = max(-saitmax.z / abszta, 1e-30); // vmax in phys. domain
    // estimations for temperature interval
    thetamin = Material.GetSurfTemp(Simulation.ndMinVapVelo, m(1).ndTemp);
    thetamax = min(Material.GetSurfTemp(vmax, m(1).ndTemp),
                   Material.ndCriticalTemp);

    if (!Simulation.bHeatConduct) // no heat conduction
    {
        bAnalyt = true; // analytical solution for heat conduction
        if (qfluxt == 0) // no heat input => final temp = 0
            thetamax = 0;
        if (thetamax > m(1).ndTemp) // heating
            m(1).ndTemp += (thetamax - m(1).ndTemp) * min(Simulation.ndDeltat / Simulation.ndHeatingDelay, 1.0);
        else // cooling
        {
            m(1).ndTemp += (thetamax - m(1).ndTemp) * min(Simulation.ndDeltat / Simulation.ndCoolingDelay, 1.0);
            if (m(1).ndTemp < 0)
                m(1).ndTemp = 0;
        }
        CalcRecession(i, j, m(1).ndTemp, vn); // get recession speed
        Solid.GridMove(i, j, vn); // shift grid
    }
    else if (vmax < Simulation.ndMinVapVelo && // no iteration if material
             m(1).ndTemp < thetamin) // is cool and no significant heating
    {
        vn = max(vmax, (prec)1e-30);
        thetamin = Material.GetSurfTemp(vn, m(1).ndTemp);
        if (CalcTemp(i, j, thetamin, vn) < 0) // only one iteration
            return -1;
    }
    else // solve heat conduction
    {
        if (CalcTemp(i, j, thetamin, vn) < 0) // first estimate of temperatures
            return -1; // with negligable evaporation
        if (m(1).ndTemp <= thetamin) // no iteration if almost cold
        {
            if (CalcTemp(i, j, m(1).ndTemp, vn) < 0) // one more iteration
                return -1;
        }
        else
        {
            thetamin0 = thetamin;
            dthetamin = m(1).ndTemp - thetamin; // change of minumum temperature

            if (vmax < Simulation.ndMinVapVelo) // correct maximum temperature
                thetamax = min(m(1).ndTemp * 1.5, Material.ndCriticalTemp);
            else
                thetamax = max(thetamax, m(1).ndTemp) + 1e-04;
            if (CalcTemp(i, j, thetamax, vn) < 0) // first estimate with maximum
                return -1; // ablation velo
            thetamax0 = thetamax;
            dthetamax = m(1).ndTemp - thetamax;

            if (dthetamax > 0) // larger than maximum temperature
            {
                WarningFunction("  ****** WARNING: multiple roots possible for "
                                "i=%2i, j=%2i; searching for largest root *****\n"
                                "          HLO=%7.5lf, FLO=%12.5lf\n"
                                "          HHI=%7.5lf, FHI=%12.5lf\n"
                                "  HOLD(I,J,1)=%7.5lf",
                                i, j, thetamin, dthetamin,
                                thetamax, dthetamax, SolidOld(i, j, 1).ndTemp);
                for (ia = 0; ia < 1000; ia++)
                {
                    thetamin = thetamax - 1e-3 * (thetamax0 - thetamin0);
                    if (CalcTemp(i, j, thetamin, vn) < 0) // correct lower temperature
                        return -1;
                    dthetamin = m(1).ndTemp - thetamin; // change of minumum temperature
                    if (dthetamin >= 0) // continue with increased lower limit
                        break;
                    thetamax = thetamin;
                    dthetamax = dthetamin;
                }
                if (ia >= 1000)
                {
                    ErrorFunction("*** Maximum iterations to find T_surface exceeded at "
                                  "I=%2i, J=%2i",
                                  i, j);
                    return -1;
                }
            }
            else if (dthetamax < 0) // converging => find root
            {
                if (FindRoot(i, j, thetamin, thetamax, dthetamin, dthetamax, vn, thetamax0) < 0)
                    return -1;
            }
        }
    }

    for (k = 1; k <= Grid.kVolNodes; k++) // temperatures found
    {
        if (m(1).ndTemp < 0) // restrict to room temperature
            Solid(i, j, k).ndTemp = 0;
        else if (k == 1 || !bAnalyt)
            Solid(i, j, k).ndTemp = m(k).ndTemp;
        else
        {
            if (vmax < Simulation.ndMinVapVelo)
                Solid(i, j, k).ndTemp = 0;
            else // analytical expressio for stationary ablation
            {
                abscoeff = Material.GetAbsIndex(m(1).ndTemp, index);
                prec dz = Solid(i, j, k).ndNode.z - Solid(i, j, 1).ndNode.z;
                Solid(i, j, k).ndTemp = (Laser.ndAvePower * Laser.ndPower / (vmax - abscoeff) * (exp(-abscoeff * dz) - exp(-vmax * dz)) + Solid(i, j, 1).ndTemp * exp(-vmax * dz) - Solid(i, j, k).ndTemp) * // dtheta
                                        min(Simulation.ndDeltat / Simulation.ndHeatingDelay, 1.0); // relax
            }
        }
    }

    Solid.ndVapRate += vn * SolSurf(i, j).ndArea; // volumetric removal rate
    if (vn > Solid.ndMaxVelo)
        Solid.ndMaxVelo = vn; // maximum recession speed
    return 0;
}

// ***************************************************************************
// calculate temperature for estimated surface temperature harr
// ***************************************************************************

int HeatConduction::CalcTemp(int i, int j, prec &theta, prec &vn)
{
    int k;
    prec fadti, ac;

    CalcRecession(i, j, theta, vn);
    Solid.GridMove(i, j, vn);
    //  Solid.MoveGridMove(i,j,vn,true);  // for nodal relaxation

    for (k = 1; k < Grid.kVolNodes; k++)
        m(k).fa = Material.ndDiffusivity.Get(m(k).ndTemp);

    for (k = 2; k < Grid.kVolNodes; k++) // set up tridiagonal matrix
    {
        fadti = 1.0 / (Simulation.ndDeltat * m(k).fa);
        ac = 0.5 * Solid(i, j, k).sait.z / m(k).fa;
        m(k).a = m(k).at - ac;
        m(k).b = m(k).bt + fadti;
        m(k).c = m(k).ct + ac;
        if (Simulation.bSurfVapor || saitmax.z == 0) // surface evaporation
            m(k).rhs = m(k).rhst + SolidOld(i, j, k).ndTemp * fadti - (Solid(i, j, k).sait.x * TempSai(k).x + Solid(i, j, k).sait.y * TempSai(k).y) / m(k).fa;
        else // volume evaporation
            m(k).rhs = m(k).rhst + SolidOld(i, j, k).ndTemp * fadti - (Solid(i, j, k).sait.x * TempSai(k).x + Solid(i, j, k).sait.y * TempSai(k).y) / m(k).fa - PowerDensity(k) * Laser.ndPower * Solid(i, j, 1).sait.z / saitmax.z;
    }

    fadti = 1.0 / (Simulation.ndDeltat * m(1).fa);
    ac = Solid(i, j, 1).sait.z / m(1).fa;
    m(1).b = m(1).bt + fadti - ac;
    m(1).c = m(1).ct + ac;
    if (Simulation.bSurfVapor || saitmax.z == 0) // surface evaporation
        m(1).rhs = m(1).rhst + SolidOld(i, j, 1).ndTemp * fadti + 2.0 * Material.ndStefan.Get(m(1).ndTemp) * Solid(i, j, 1).sait.z - (Solid(i, j, 1).sait.x * TempSai(1).x + Solid(i, j, 1).sait.y * TempSai(1).y) / m(1).fa;
    else // volume evaporation
        m(1).rhs = m(1).rhst + SolidOld(i, j, 1).ndTemp * fadti - (Solid(i, j, 1).sait.x * TempSai(1).x + Solid(i, j, 1).sait.y * TempSai(1).y) / m(1).fa - PowerDensity(1) * Laser.ndPower * Solid(i, j, 1).sait.z / saitmax.z;
    m(1).a = 0.0;

    if (m.Solve(Grid.kVolNodes) < 0) // solve matrix
        return -1;

    return 0;
}

// ***************************************************************************
// iteration of surface temperature
// ***************************************************************************

int HeatConduction::FindRoot(int i, int j, prec &hlo, prec &hhi, prec &flo,
                             prec &fhi, prec &vn, prec harrmax)
{
    int knt;
    prec hmid, fmid, s, hnew, fnew, relax, delta;

    for (knt = 1; knt <= 300; knt++)
    {
        hmid = 0.5 * (hhi + hlo); // average of interval
        if (CalcTemp(i, j, hmid, vn) < 0)
            return -1;
        fmid = m(1).ndTemp - hmid; // gradient of average interval
        delta = fmid * fmid - flo * fhi;
        if (delta < 0)
            break;
        s = sqrt(delta);
        if (s == 0)
            return 0;
        hnew = hmid + (hmid - hlo) * fmid / s;
        if (CalcTemp(i, j, hnew, vn) < 0)
            return -1;
        if (fabs(hnew - m(1).ndTemp) <= Simulation.ndTempTol)
            return 0; // root found
        fnew = m(1).ndTemp - hnew;
        if (fmid * fnew < 0)
        {
            if (fmid > 0)
            {
                hlo = hmid;
                flo = fmid;
                hhi = hnew;
                fhi = fnew;
            }
            else
            {
                hlo = hnew;
                flo = fnew;
                hhi = hmid;
                fhi = fmid;
            }
        }
        else
        {
            if (flo * fnew < 0)
            {
                hhi = hnew;
                fhi = fnew;
            }
            else
            {
                hlo = hnew;
                flo = fnew;
            }
        }
    }

    // Newton iteration starting at harrmax if no convergence

    hhi = harrmax;
    CalcTemp(i, j, hhi, vn);
    if (m(1).ndTemp > Material.ndCriticalTemp)
    {
        hhi = Material.ndCriticalTemp;
        CalcTemp(i, j, hhi, vn);
    }
    fhi = m(1).ndTemp - hhi; // DT = f(hhi)
    while (fabs(fhi) > Simulation.ndTempTol / 1e7)
    {
        relax = 1;
        hnew = hhi + Simulation.ndTempTol;
        CalcTemp(i, j, hnew, vn);
        fnew = m(1).ndTemp - hnew; // DT = f(hhi+htol) for f'(hhi)
        do
        {
            if (relax < Simulation.ndTempTol / 1e8)
            {
                ErrorFunction("  **** ITERATION TO FIND HARR DID NOT CONVERGE FOR"
                              " I=%2i, J=%2i\n  **** HARR=%7.5lf, FARR=%12.4lf",
                              i, j, hnew, fnew);
                return 0;
            }
            hnew = hhi - relax * fhi * Simulation.ndTempTol / (fnew - fhi);
            CalcTemp(i, j, hnew, vn);
            flo = m(1).ndTemp - hnew;
            relax /= 2;
        } while (fabs(flo) >= fabs(fhi));
        fhi = flo;
        hhi = hnew;
    }

    return 0;
}

// ***************************************************************************
// calculate evaporation rate vn(harr) and surface displacement
// ***************************************************************************

void HeatConduction::CalcRecession(int i, int j, prec &theta, prec &vn)
{
    theta = max(theta, 0.);
    vn = Material.GetEvapVelocity(theta);
    Solid(i, j, 1).sait.z = -vn * abszta; // velocity in comput. domain
}

// ***************************************************************************
// solve tridiagonal matrix A*u(i-1)+B*u(i)+C*u(i+1)=R for i=0..N-1
// ***************************************************************************

int TTridagMatrix::Solve(int n)
{
    rvector gam(GetSize());
    prec bet;
    int j;

    bet = At(1).b;
    if (bet == 0)
    {
        ErrorFunction("ERROR: bad matrix - b(1)=0");
        return -1;
    }
    At(1).ndTemp = At(1).rhs / bet;
    for (j = 2; j <= n; j++)
    {
        gam(j) = At(j - 1).c / bet;
        bet = At(j).b - At(j).a * gam(j);
        if (bet == 0)
        {
            ErrorFunction("ERROR: division by 0 in tridag");
            return -1;
        }
        At(j).ndTemp = (At(j).rhs - At(j).a * At(j - 1).ndTemp) / bet;
    }
    for (j = n - 1; j >= 1; j--)
        At(j).ndTemp -= gam(j + 1) * At(j + 1).ndTemp;

    return 0;
}
