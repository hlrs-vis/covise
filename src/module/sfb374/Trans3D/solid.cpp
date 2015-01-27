/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          solid.cpp  -  solid routines
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include "solid.h"
#include "simul.h"
#include "grid.h"
#include "laser.h"
#include "material.h"
#include "solve.h"
#include "main.h"

TSolid Solid, SolidOld;
TSurface SolSurf, SolSurfOld;
TSurNodes Surface;

const prec epsgrid = 0.1;

//  control ouput

void WriteSolidData(int imin, int imax, int jmin, int jmax, int kmin, int kmax)
{
    fstream fs("solid.txt", ios::out);

    if (imax < 1)
        imax = Grid.iVolNodes - 1;
    if (jmax < 1)
        jmax = Grid.jVolNodes - 1;
    if (kmax < 1)
        kmax = Grid.kVolNodes - 1;
    fs << "new version" << endl;
    fs << "VapRate: " << Solid.ndVapRate << endl;
    fs << "i\tj\tk\tx\ty\tz\tT\tq" << endl;
    for (int i = imin; i <= imax; i++)
    {
        fs << i << ": " << Solid.jBegin(i) << '-' << Solid.jEnd(i) << endl;
        for (int j = jmin; j <= jmax; j++)
            for (int k = kmin; k <= kmax; k++)
            {
                fs << i << '\t' << j << '\t' << k << '\t';
                fs << Solid(i, j, k).ndNode << '\t';
                fs << Solid(i, j, k).ndTemp << '\t';
                fs << Solid(i, j, k).ndHeat << endl;
            }
    }
}

void WriteSurfaceData(int imin, int imax, int jmin, int jmax)
{
    fstream fs("surface.txt", ios::out);

    if (imax < 1)
        imax = Grid.iVolNodes - 1;
    if (jmax < 1)
        jmax = Grid.jVolNodes - 1;
    fs << "new version" << endl;
    fs << "TotalDir:" << '\t' << SolSurf.ndTotalDirAbs;
    fs << "i\tj\tqdir\tqmult" << endl;
    for (int i = imin; i <= imax; i++)
    {
        for (int j = jmin; j <= jmax; j++)
        {
            fs << i << '\t' << j << '\t';
            fs << SolSurf(i, j).ndDirAbs << '\t';
            fs << SolSurf(i, j).ndMultAbs << endl;
        }
    }
}

// ***************************************************************************
// save variables
//
// input:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

ostream &operator<<(ostream &ps, TSolid &s)
{
    TSolid src = s.GetDimensional();
    int i, j, k;
    point3D pt = point3D(s.GetiMax(), s.GetjMax(), s.GetkMax());

    ps << endl;
    ps << "volume grid:" << endl;
    ps << "array size:\t" << pt << endl;
    ps << "x[m]\ty[m]\tz[m]\tT [K]\tQ [W/m3]" << endl;
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            for (k = 0; k < pt.z; k++)
                ps << src(i, j, k) << endl;
    ps << "begin heat affected zone" << endl;
    ps << src.jBegin;
    ps << "end heat affected zone" << endl;
    ps << src.jEnd;
    ps << "max displacement [m]:\t" << src.ndMaxDSai << endl;
    ps << "max temperature [K]:\t" << src.ndMaxTemp << endl;
    ps << "max velocity [m/s]:\t" << src.ndMaxVelo << endl;
    ps << "vaporization rate [m3/s]:\t" << src.ndVapRate << endl;
    return ps;
}

// ***************************************************************************
// read settings
//
// input:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

istream &operator>>(istream &ps, TSolid &src)
{
    int i, j, k;
    point3D pt;
    bool b;

    ps >> checkstring("volume grid:", &b);
    if (!b)
    {
        ps.setstate(ios::failbit);
        return ps;
    }
    src.bChanged = false;
    ps >> tab >> pt >> endl >> endl;
    src.Reallocate(pt.x, pt.y, pt.z);
    src.ndMinSurZ = 1e20;
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            for (k = 0; k < pt.z; k++)
            {
                ps >> src(i, j, k);
                src.ndMinSurZ = min(src.ndMinSurZ, src(i, j, k).ndNode.z);
            }
    ps >> endl >> endl >> src.jBegin;
    ps >> endl >> src.jEnd;
    ps >> tab >> src.ndMaxDSai;
    ps >> tab >> src.ndMaxTemp;
    ps >> tab >> src.ndMaxVelo;
    ps >> tab >> src.ndVapRate >> endl;
    src.MakeNonDimensional();
    return ps;
}

// ***************************************************************************
// bring all variables into dimensional form
// ***************************************************************************

TSolid TSolid::GetDimensional()
{
    TSolid s = *this;
    point3D pt = point3D(s.GetiMax(), s.GetjMax(), s.GetkMax());
    int i, j, k;

    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            for (k = 0; k < pt.z; k++)
            {
                s(i, j, k).ndNode *= RefLength;
                s(i, j, k).ndTemp = Material.GetKelvin(s(i, j, k).ndTemp);
                s(i, j, k).ndHeat *= Laser.ndPower;
            }
    s.ndMaxDSai *= RefLength;
    s.ndMaxZMove *= RefLength;
    s.ndMaxXYMove *= RefLength;
    s.ndMinSurZ *= RefLength;
    s.ndVapRate *= Laser.ndRate * sqr(RefLength) * RefVelocity;
    s.ndMaxVelo *= RefVelocity;
    s.ndMaxTemp = Material.GetKelvin(s.ndMaxTemp);
    return s;
}

// ***************************************************************************
// bring all variables to non-dimensional form
//
// input:    RefLength, RefTime, RefVelocity
//
// ***************************************************************************

void TSolid::MakeNonDimensional()
{
    point3D pt = point3D(imax, jmax, kmax);
    int i, j, k;

    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            for (k = 0; k < pt.z; k++)
            {
                At(i, j, k).ndNode /= RefLength;
                At(i, j, k).ndTemp = Material.GetKirchhoff(At(i, j, k).ndTemp);
                At(i, j, k).ndHeat /= Laser.ndPower;
            }
    ndMaxDSai /= RefLength;
    ndMaxZMove /= RefLength;
    ndMaxXYMove /= RefLength;
    ndMinSurZ /= RefLength;
    ndVapRate /= Laser.ndRate * sqr(RefLength) * RefVelocity;
    ndMaxVelo /= RefVelocity;
    ndMaxTemp = Material.GetKirchhoff(ndMaxTemp);
}

// ***************************************************************************
// solid class
// ***************************************************************************

// resize jBegin and jEnd together with array

void TSolid::ReSize(unsigned iNewSize, unsigned jNewSize, unsigned kNewSize,
                    int niOffset, int njOffset, int nkOffset)
{
    Tensor<TNodeVariables>::ReSize(iNewSize, jNewSize, kNewSize,
                                   niOffset, njOffset, nkOffset);
    jBegin.ReSize(iNewSize);
    jEnd.ReSize(iNewSize);
}

// update jBegin, jEnd if nodes are inserted

void TSolid::InsertBegin(int i, int j, int k)
{
    int ii;

    Tensor<TNodeVariables>::InsertBegin(i, j, k);
    if (i != 0) // nodes added or dropped
    {
        jBegin.ReSize(imax, i);
        jEnd.ReSize(imax, i);
        ii = 0;
        while (ii < i) // initialize jBegin, jEnd
        {
            jBegin[ii] = jBegin[i];
            jEnd[ii] = jEnd[i];
            ii++;
        }
    }
    if (j != 0)
    {
        for (ii = 0; ii < int(imax); ii++)
        {
            jBegin[ii] = min(max(2, jBegin[ii] + j), int(jmax) - 3);
            jEnd[ii] = min(max(2, jEnd[ii] + j), int(jmax) - 3);
        }
    }
}

void TSolid::InsertEnd(int i, int j, int k)
{
    int ii;
    Tensor<TNodeVariables>::InsertEnd(i, j, k);
    if (i != 0)
    {
        jBegin.ReSize(imax, i);
        jEnd.ReSize(imax, i);
        ii = imax - i;
        while (ii < int(imax)) // nodes added => initialize jBegin, jEnd
        {
            jBegin[ii] = jBegin[imax - i - 1];
            jEnd[ii] = jEnd[imax - i - 1];
            ii++;
        }
    }
    if (j != 0)
    {
        for (ii = 0; ii < int(imax); ii++)
        {
            jBegin[ii] = min(max(2, jBegin[ii] + j), int(jmax) - 3);
            jEnd[ii] = min(max(2, jEnd[ii] + j), int(jmax) - 3);
        }
    }
}

// initialize Temperature

void TSolid::ResetTemperature()
{
    if (p == 0)
        return;
    for (unsigned i = 0; i < imax; i++)
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned k = 0; k < kmax; k++)
                //        {
                At(i, j, k).ndTemp = 0;
    //        At(i,j,k).heat = 0;
    //        }
}

// ***************************************************************************
// calculate co- and contravariant base vectors at node (i,j,k)
// ***************************************************************************

void TSolid::CalcBaseVect(int i, int j, int k)
{
    int ip, im, jp, jm, kp, km;
    prec dz;

    ip = i + 1;
    im = i - 1;
    jp = j + 1;
    jm = j - 1;
    kp = k + 1;
    km = k - 1;
    At(i, j, k).xsai = 0.5 * (At(ip, j, k).ndNode - At(im, j, k).ndNode);
    At(i, j, k).xeta = 0.5 * (At(i, jp, k).ndNode - At(i, jm, k).ndNode);
    if (k > 1)
        At(i, j, k).xzta = 0.5 * (At(i, j, kp).ndNode - At(i, j, km).ndNode);
    else
        At(i, j, k).xzta = At(i, j, kp).ndNode - At(i, j, k).ndNode;
    At(i, j, k).saix = CrossProduct(At(i, j, k).xeta, At(i, j, k).xzta);
    At(i, j, k).etax = CrossProduct(At(i, j, k).xzta, At(i, j, k).xsai);
    At(i, j, k).ztax = CrossProduct(At(i, j, k).xsai, At(i, j, k).xeta);
    dz = 1.0 / (At(i, j, k).xsai.x * At(i, j, k).saix.x + At(i, j, k).xeta.x * At(i, j, k).etax.x + At(i, j, k).xzta.x * At(i, j, k).ztax.x);
    At(i, j, k).saix *= dz;
    At(i, j, k).etax *= dz;
    At(i, j, k).ztax *= dz;
}

// ***************************************************************************
// calculate metric tensor and laplace sai at element (i,j,k)
// ***************************************************************************

prec TSolid::CalcMetricTensor(int i, int j, int k, rmmatrix &A, TPoint3D &lapsai)
{
    int ip, im, jp, jm, kp, km;
    TPoint3D D;

    ip = i + 1;
    im = i - 1;
    jp = j + 1;
    jm = j - 1;
    kp = k + 1;
    km = k - 1;
    A(0, 0) = At(i, j, k).saix * At(i, j, k).saix;
    A(0, 1) = At(i, j, k).saix * At(i, j, k).etax;
    A(0, 2) = At(i, j, k).saix * At(i, j, k).ztax;
    A(1, 0) = A(0, 1);
    A(1, 1) = At(i, j, k).etax * At(i, j, k).etax;
    A(1, 2) = At(i, j, k).etax * At(i, j, k).ztax;
    A(2, 0) = A(0, 2);
    A(2, 1) = A(1, 2);
    A(2, 2) = At(i, j, k).ztax * At(i, j, k).ztax;
    if (k > 1)
        D = A(0, 0) * (At(ip, j, k).ndNode - 2. * At(i, j, k).ndNode + At(im, j, k).ndNode)
            + A(1, 1) * (At(i, jp, k).ndNode - 2. * At(i, j, k).ndNode + At(i, jm, k).ndNode)
            + A(2, 2) * (At(i, j, kp).ndNode - 2. * At(i, j, k).ndNode + At(i, j, km).ndNode)
            + (A(0, 1) * (At(ip, jp, k).ndNode - At(im, jp, k).ndNode + At(im, jm, k).ndNode - At(ip, jm, k).ndNode)
               + A(0, 2) * (At(ip, j, kp).ndNode - At(im, j, kp).ndNode + At(im, j, km).ndNode - At(ip, j, km).ndNode)
               + A(1, 2) * (At(i, jp, kp).ndNode - At(i, jm, kp).ndNode + At(i, jm, km).ndNode - At(i, jp, km).ndNode)) * 0.5;
    else
        D = A(0, 0) * (At(ip, j, k).ndNode - 2. * At(i, j, k).ndNode + At(im, j, k).ndNode)
            + A(1, 1) * (At(i, jp, k).ndNode - 2. * At(i, j, k).ndNode + At(i, jm, k).ndNode)
            + A(0, 1) * (At(ip, jp, k).ndNode - At(im, jp, k).ndNode + At(im, jm, k).ndNode - At(ip, jm, k).ndNode) * 0.5
            + A(0, 2) * (At(ip, j, kp).ndNode - At(im, j, kp).ndNode + At(im, j, k).ndNode - At(ip, j, k).ndNode)
            + A(1, 2) * (At(i, jp, kp).ndNode - At(i, jm, kp).ndNode + At(i, jm, k).ndNode - At(i, jp, k).ndNode);

    lapsai.x = -D * At(i, j, k).saix;
    lapsai.y = -D * At(i, j, k).etax;
    lapsai.z = -D * At(i, j, k).ztax;
    return sqrt(A(2, 2));
}

// ***************************************************************************
// calculate temperature derivative
// ***************************************************************************

void TSolid::CalcTempDerivative(int i, int j, int k, TPoint3D &hs,
                                prec &hss, prec &hee,
                                prec &hse, prec &hsz, prec &hez)
{
    int ip, im, jp, jm, kp, km;

    ip = i + 1;
    im = i - 1;
    jp = j + 1;
    jm = j - 1;
    kp = k + 1;
    km = k - 1;
    if (i == 1)
    {
        hs.x = At(2, j, k).ndTemp - At(1, j, k).ndTemp;
        hss = 2. * hs.x;
        if (k == 1)
            hsz = At(2, j, 2).ndTemp - At(2, j, 1).ndTemp;
        else
            hsz = 0.5 * (At(2, j, kp).ndTemp - At(2, j, km).ndTemp);
        if (j == 1)
            hse = At(2, 2, k).ndTemp - At(1, 1, k).ndTemp;
        else if (j == Grid.jVolNodes)
            hse = At(1, j, k).ndTemp - At(2, jm, k).ndTemp;
        else
            hse = 0.5 * (At(2, jp, k).ndTemp - At(2, jm, k).ndTemp);
    }
    else if (i == Grid.iVolNodes)
    {
        hs.x = At(i, j, k).ndTemp - At(im, j, k).ndTemp;
        hss = -2. * hs.x;
        if (k == 1)
            hsz = At(im, j, 1).ndTemp - At(im, j, 2).ndTemp;
        else
            hsz = 0.5 * (At(im, j, km).ndTemp - At(im, j, kp).ndTemp);
        if (j == 1)
            hse = At(i, 1, k).ndTemp - At(im, 2, k).ndTemp;
        else if (j == Grid.jVolNodes)
            hse = At(im, jm, k).ndTemp - At(i, j, k).ndTemp;
        else
            hse = 0.5 * (At(im, jm, k).ndTemp - At(im, jp, k).ndTemp);
    }
    else
    {
        hs.x = 0.5 * (At(ip, j, k).ndTemp - At(im, j, k).ndTemp);
        hss = At(ip, j, k).ndTemp - 2. * At(i, j, k).ndTemp + At(im, j, k).ndTemp;
        if (k == 1)
            hsz = 0.5 * (At(ip, j, 2).ndTemp - At(im, j, 2).ndTemp - At(ip, j, 1).ndTemp + At(im, j, 1).ndTemp);
        else
            hsz = 0.25 * (At(ip, j, kp).ndTemp - At(im, j, kp).ndTemp - At(ip, j, km).ndTemp + At(im, j, km).ndTemp);
        if (j == 1)
            hse = 0.5 * (At(ip, 2, k).ndTemp - At(im, 2, k).ndTemp);
        else if (j == Grid.jVolNodes)
            hse = 0.5 * (At(im, jm, k).ndTemp - At(ip, jm, k).ndTemp);
        else
            hse = 0.25 * (At(ip, jp, k).ndTemp - At(im, jp, k).ndTemp - At(ip, jm, k).ndTemp + At(im, jm, k).ndTemp);
    }
    if (j == 1)
    {
        hs.y = At(i, 2, k).ndTemp - At(i, 1, k).ndTemp;
        hee = 2. * hs.y;
        if (k == 1)
            hez = At(i, 2, 2).ndTemp - At(i, 2, 1).ndTemp;
        else
            hez = 0.5 * (At(i, 2, kp).ndTemp - At(i, 2, km).ndTemp);
    }
    else if (j == Grid.jVolNodes)
    {
        hs.y = At(i, j, k).ndTemp - At(i, jm, k).ndTemp;
        hee = -2. * hs.y;
        if (k == 1)
            hez = At(i, jm, 1).ndTemp - At(i, jm, 2).ndTemp;
        else
            hez = 0.5 * (At(i, jm, km).ndTemp - At(i, jm, kp).ndTemp);
    }
    else
    {
        hs.y = 0.5 * (At(i, jp, k).ndTemp - At(i, jm, k).ndTemp);
        hee = At(i, jp, k).ndTemp - 2. * At(i, j, k).ndTemp + At(i, jm, k).ndTemp;
        if (k == 1)
            hez = 0.5 * (At(i, jp, 2).ndTemp - At(i, jm, 2).ndTemp - At(i, jp, 1).ndTemp + At(i, jm, 1).ndTemp);
        else
            hez = 0.25 * (At(i, jp, kp).ndTemp - At(i, jm, kp).ndTemp - At(i, jp, km).ndTemp + At(i, jm, km).ndTemp);
    }
}

// ***************************************************************************
// set inner nodes starting at surface with normal vector SolSurf.Normal
//
// ***************************************************************************

void TSolid::SetInnerNodes(int i, int j, prec dmin, bool bPotential)
{
    int k, kl1, km1;
    prec depth_kl, depth_kl3, nspace, spacek, spacekm1, ncos,
        nuk, cnu, pnu, tmp, adepth, deltas;
    TPoint dbound, bbend;
    TPoint3D ZtaTangent1, ZtaTangentK;

    kl1 = Grid.kVolNodes - 1;
    depth_kl = Grid.ndDepth / kl1;
    depth_kl3 = depth_kl / sqr(kl1);
    if (dmin == 0) // nodal spacing at surface; constant A in eqn(27)
        nspace = (depth_kl - Grid.ndSpacing) / (Grid.ndSpacing - depth_kl3);
    else
        nspace = (depth_kl - dmin) / (dmin - depth_kl3);
    adepth = Grid.ndDepth / (1 + nspace); // D/(1+A) in eqn(26)

    dbound.x = min(At(i, j, 1).ndNode.x, Grid.ndWpLength - At(i, j, 1).ndNode.x);
    dbound.y = min(At(i, j, 1).ndNode.y, Grid.ndWpWidth - At(i, j, 1).ndNode.y);
    bbend.x = 1 - exp(-Grid.TangentBound * dbound.x); // bending at boundary
    bbend.y = 1 - exp(-Grid.TangentBound * dbound.y);

    spacek = 0;
    ZtaTangent1 = SolSurf(i, j).Normal; // inward pointing normal at surface
    ncos = ZtaTangent1 * LocalKlTangent; // cosine between top and bottom normal
    for (k = 2; k < Grid.kVolNodes; k++)
    {
        km1 = k - 1;
        nuk = prec(km1) / kl1; // nu(k) in eqn(24)
        //c(nu) in eqn(24)
        cnu = max(0., 1. - sqr(nuk) - km1 * Grid.TangentWarp * (1. - nuk));
        tmp = cnu * ncos;
        pnu = sqrt(sqr(tmp) - sqr(cnu) + 1.) - tmp; // p(nu) in eqn(24)
        spacekm1 = spacek;
        spacek = nuk * (1. + nspace * sqr(nuk)); // nodal position s(k)
        deltas = (spacek - spacekm1) * adepth; // spacing between node k and k-1
        if (bPotential && k > 2) // relaxation of inner nodes
        {
            prec nukp1 = prec(k) / kl1;
            prec spacekp1 = nukp1 * (1. + nspace * sqr(nukp1));
            prec deltas1 = (spacekp1 - spacek) * adepth; // spacing node k+1 and k
            Grid.ShiftVolNode(i, j, k, deltas, deltas1);
        }
        else
        {
            if (bPotential) // first inner node in normal direction
            {
                ZtaTangentK.x = ZtaTangent1.x * bbend.x;
                ZtaTangentK.y = ZtaTangent1.y * bbend.y;
            }
            else
            {
                ZtaTangentK.x = (cnu * ZtaTangent1.x + pnu * LocalKlTangent.x) * bbend.x;
                ZtaTangentK.y = (cnu * ZtaTangent1.y + pnu * LocalKlTangent.y) * bbend.y;
            }
            ZtaTangentK.z = sqrt(1. - sqr(ZtaTangentK.x) - sqr(ZtaTangentK.y));
            At(i, j, k).ndNode = At(i, j, km1).ndNode + deltas * ZtaTangentK;
        }
        // fix z position if drilled through
        if (At(i, j, k).ndNode.z > Grid.ndWpThickness + km1 * Grid.ndSpacing / 100)
            At(i, j, k).ndNode.z = Grid.ndWpThickness + km1 * Grid.ndSpacing / 100;
        if (dmin != 0) // set velocities if called during simulation
        {
            At(i, j, k).xtau = (Solid(i, j, k).ndNode - SolidOld(i, j, k).ndNode) / Simulation.ndTime;
            At(i, j, k).sait.x = -At(i, j, k).saix * At(i, j, k).xtau;
            At(i, j, k).sait.y = -At(i, j, k).etax * At(i, j, k).xtau;
            At(i, j, k).sait.z = -At(i, j, k).ztax * At(i, j, k).xtau;
        }
    }
    // artificial node for second order derivative
    At(i, j, Grid.kVolNodes).ndNode = 3.0 * (At(i, j, kl1).ndNode - At(i, j, kl1 - 1).ndNode) + At(i, j, kl1 - 2).ndNode;
    return;
}

// ***************************************************************************
// temporary grid movement with ablation velocity ztat i.e. sait.z
// ***************************************************************************

void TSolid::GridMove(int i, int j, prec ztat, bool bPotential)
{
    int iztain, kl1;
    prec vn, delta_zta, ztain, dist12;
    TPoint3D xin;

    if (ztat < 0.99 * Simulation.ndMinVapVelo && Grid.iGridMove == TGrid::no_update)
    { // no ablation
        At(i, j, 1).sait.x = 0.0;
        At(i, j, 1).sait.y = 0.0;
        for (int k = 2; k < Grid.kVolNodes; k++)
            At(i, j, k).sait = At(i, j, 1).sait; // move all nodes with same velocity
        return;
    }

    if (Grid.bGridMove) // move x,y-nodes on surface
    {
        // velo/abs(ztax) in phys.
        vn = -At(i, j, 1).sait.z / At(i, j, 1).ztax.Norm();
        At(i, j, 1).xtau = SolSurf(i, j).xtau + // domain multiply by normal
                           vn * At(i, j, 1).ztax; // combine with other xtaus
    }
    else // only in z-direction => projection of normal velocities
    {
        vn = -At(i, j, 1).sait.z / At(i, j, 1).ztax.z; // projected velo
        At(i, j, 1).xtau = SolSurf(i, j).xtau; // xtau from grid update
        At(i, j, 1).xtau.z += vn; // velocity times normal z
    }
    At(i, j, 1).ndNode = SolidOld(i, j, 1).ndNode + At(i, j, 1).xtau * Simulation.ndDeltat;
    if (At(i, j, 1).ndNode.z > Grid.ndWpThickness) // if drilled through
        At(i, j, 1).ndNode.z = Grid.ndWpThickness;
    // sait component
    At(i, j, 1).sait.x = -At(i, j, 1).saix * At(i, j, 1).xtau;
    // etat component
    At(i, j, 1).sait.y = -At(i, j, 1).etax * At(i, j, 1).xtau;

    kl1 = Grid.kVolNodes - 1;
    if (Simulation.bHeatConduct) // if heat condcution is calculated
    { // calc position of first inner node xin
        // estimate by constant temp. gradient
        ztain = 1.0 + (epsgrid + SolidOld(i, j, 1).ndTemp) // zeta of node 2
                      / (epsgrid + SolidOld(i, j, 1).ndTemp - SolidOld(i, j, 2).ndTemp * kl1);
        ztain = min(max(ztain, 2.0 - Simulation.ndMaxSurfMove),
                    2.0 + Simulation.ndMaxSurfMove); // limit movement
        iztain = int(ztain);
        delta_zta = ztain - iztain; // delta zeta to next node iztain
        // interpolate node
        xin = SolidOld(i, j, iztain).ndNode * (1 - delta_zta) + SolidOld(i, j, iztain + 1).ndNode * delta_zta;
    }
    else
        xin = SolidOld(i, j, 2).ndNode; // keep position
    dist12 = min(max((SolidOld(i, j, 1).ndNode - xin).Abs(), Grid.ndSpacing),
                 Grid.ndDepth / kl1); // limit movement to minimum spacing and
    // average spacing
    SetInnerNodes(i, j, dist12, bPotential); // set inner nodes
}

// ***************************************************************************
// set artificial nodes for boundary conditions
// ***************************************************************************

void TSolid::SetBoundaryNodes(void)
{
    int i, j, k, k1;

    for (j = 0; j <= Grid.jVolNodes + 1; j++) // i=0,i=iVolNodes+1 boundaries
    {
        if (Grid.iVolMin > 1) // volume smaller than surface array
        {
            k1 = 2; // further update starting at k=2
            At(0, j, 1).ndNode = Surface(Grid.iVolMin - 1, Grid.jVolMin - 1 + j);
        }
        else
            k1 = 1; // also first node has to be generated
        for (k = k1; k <= Grid.kVolNodes; k++)
        {
            At(0, j, k).ndNode.x = 2 * At(1, j, k).ndNode.x - At(2, j, k).ndNode.x;
            if (Grid.iFrontBound == TGrid::symmetric) // symmetric boundary
            {
                // value constant
                At(0, j, k).ndNode.y = At(2, j, k).ndNode.y;
                At(0, j, k).ndNode.z = At(2, j, k).ndNode.z;
            }
            else // finite volume => point symmetry (gradient constant)
            {
                At(0, j, k).ndNode.y = 2 * At(1, j, k).ndNode.y - At(2, j, k).ndNode.y;
                At(0, j, k).ndNode.z = 2 * At(1, j, k).ndNode.z - At(2, j, k).ndNode.z;
            }
        }
        i = Grid.iVolNodes + 1;
        if (Grid.iVolMax < Grid.iSurNodes) // volume smaller than surface array
        {
            k1 = 2; // further update starting at k=2
            At(i, j, 1).ndNode = Surface(Grid.iVolMax + 1, Grid.jVolMin - 1 + j);
        }
        else
            k1 = 1; // also first node has to be generated
        for (k = k1; k <= Grid.kVolNodes; k++)
        {
            At(i, j, k).ndNode.x = 2 * At(Grid.iVolNodes, j, k).ndNode.x - At(i - 2, j, k).ndNode.x;
            if (Grid.iBackBound == TGrid::symmetric) // symmetric boundary
            {
                // value constant
                At(i, j, k).ndNode.y = At(i - 2, j, k).ndNode.y;
                At(i, j, k).ndNode.z = At(i - 2, j, k).ndNode.z;
            }
            else // finite volume => point symmetry (gradient constant)
            {
                At(i, j, k).ndNode.y = 2 * At(Grid.iVolNodes, j, k).ndNode.y - At(i - 2, j, k).ndNode.y;
                At(i, j, k).ndNode.z = 2 * At(Grid.iVolNodes, j, k).ndNode.z - At(i - 2, j, k).ndNode.z;
            }
        }
    }

    for (i = 0; i <= Grid.iVolNodes + 1; i++) // j=0,j=jVolNodes+1 boundaries
    {
        if (Grid.jVolMin > 1) // volume smaller than surface array
        {
            k1 = 2; // further update starting at k=2
            At(i, 0, 1).ndNode = Surface(Grid.iVolMin - 1 + i, Grid.jVolMin - 1);
        }
        else
            k1 = 1; // also first node has to be generated
        for (k = k1; k <= Grid.kVolNodes; k++)
        {
            At(i, 0, k).ndNode.y = 2 * At(i, 1, k).ndNode.y - At(i, 2, k).ndNode.y;
            if (Grid.iRightBound == TGrid::symmetric) // symmetric boundary
            {
                // value constant
                At(i, 0, k).ndNode.x = At(i, 2, k).ndNode.x;
                At(i, 0, k).ndNode.z = At(i, 2, k).ndNode.z;
            }
            else // finite volume => point symmetry (gradient constant)
            {
                At(i, 0, k).ndNode.x = 2 * At(i, 1, k).ndNode.x - At(i, 2, k).ndNode.x;
                At(i, 0, k).ndNode.z = 2 * At(i, 1, k).ndNode.z - At(i, 2, k).ndNode.z;
            }
        }
        j = Grid.jVolNodes + 1;
        if (Grid.jVolMax < Grid.jSurNodes) // volume smaller than surface array
        {
            k1 = 2; // further update starting at k=2
            At(i, j, 1).ndNode = Surface(Grid.iVolMin - 1 + i, Grid.jVolMax + 1);
        }
        else
            k1 = 1; // also first node has to be generated
        for (k = k1; k <= Grid.kVolNodes; k++)
        {
            At(i, j, k).ndNode.y = 2 * At(i, Grid.jVolNodes, k).ndNode.y - At(i, j - 2, k).ndNode.y;
            if (Grid.iLeftBound == TGrid::symmetric) // symmetric boundary
            {
                // value constant
                At(i, j, k).ndNode.x = At(i, j - 2, k).ndNode.x;
                At(i, j, k).ndNode.z = At(i, j - 2, k).ndNode.z;
            }
            else // finite volume => point symmetry (gradient constant)
            {
                At(i, j, k).ndNode.x = 2 * At(i, Grid.jVolNodes, k).ndNode.x - At(i, j - 2, k).ndNode.x;
                At(i, j, k).ndNode.z = 2 * At(i, Grid.jVolNodes, k).ndNode.z - At(i, j - 2, k).ndNode.z;
            }
        }
    }
}

// ***************************************************************************
// calculate isotherms
// ***************************************************************************

/*void GetIsoPos(prec temp, prec i, prec j, prec& ptx, prec& pty, prec& ptz)
  {
  int    k, km1;
  prec  htemp;
  prec  s12sq, s12,
        adepth, gridx, dxb, dyb, facx, facy, sk, dotp, anuk,
        cn, cndot, pn, skm1, szta;
  TPoint3D  ds;

  if(temp<=Material.Traum)
    {
ptx = Solid(i,j,Grid.Kl).ndNode.x*Laser.Radius;
pty = Solid(i,j,Grid.Kl).ndNode.y*Laser.Radius;
ptz = Solid(i,j,Grid.Kl).ndNode.z*Laser.Radius;
return;
}
htemp = Material.GetTheta(temp);
if(htemp>=Solid(i,j,1).ndTemp)
{
ptx = Solid(i,j,1).ndNode.x*Laser.Radius;
pty = Solid(i,j,1).ndNode.y*Laser.Radius;
ptz = Solid(i,j,1).ndNode.z*Laser.Radius;
return;
}
s12sq = (Solid(i,j,1).ndNode-Solid(i,j,2).ndNode).Norm();
//  s12sq=sqr(Solid(i,j,1).ndNode.x-Solid(i,j,2).ndNode.x)+
//        sqr(Solid(i,j,1).ndNode.y-Solid(i,j,2).ndNode.y)+
//        sqr(Solid(i,j,1).ndNode.z-Solid(i,j,2).ndNode.z);
s12=sqrt(s12sq);
if(s12<Grid.smin)
s12=Grid.smin;
if(s12>Grid.depth1)
s12=Grid.depth1;

gridx=(Grid.depth1-Grid.smin)/(s12-Grid.depth3);
adepth=Grid.depth/(1.0+gridx);
dxb=min(Solid(i,j,1).ndNode.x,Grid.wplength-Solid(i,j,1).ndNode.x);
dyb=min(Solid(i,j,1).ndNode.y,Grid.wpwidth-Solid(i,j,1).ndNode.y);
facx=1.0-exp(-Grid.Ck3*dxb);
facy=1.0-exp(-Grid.Ck3*dyb);
sk=0.0;
for(k=2;k<=Grid.Kl1;k++)
{
km1=k-1;
anuk=(k-1.0)/Grid.Kl1;
cn=max(0.0,1.0-anuk*anuk-km1*Grid.Ck1*(1.0-anuk));
dotp=ZtaTangent(1)*LocalKlTangent;
cndot=cn*dotp;
pn=sqrt(cn*cn*(dotp*dotp-1.0)+1.0)-cndot;
ds.x=(cn*ZtaTangent(1).x+pn*LocalKlTangent.x)*facx;
ds.y=(cn*ZtaTangent(1).y+pn*LocalKlTangent.y)*facy;
ds.z=sqrt(1.0-ds.x*ds.x-ds.y*ds.y);
skm1=sk;
sk=anuk*(1.0+gridx*anuk*anuk);
if(htemp>=Solid(i,j,k).ndTemp)
{
szta=(sk-skm1)*adepth*(Solid(i,j,k-1).ndTemp-htemp)/
(Solid(i,j,k-1).ndTemp-Solid(i,j,k).ndTemp);
ptx=Solid(i,j,km1).ndNode.x+szta*ds.x;
pty=Solid(i,j,km1).ndNode.y+szta*ds.y;
ptz=Solid(i,j,km1).ndNode.z+szta*ds.z;
if(ptz>Grid.wpthickn+km1*Grid.smin/100)
ptz = Grid.wpthickn+km1*Grid.smin/100;
ptx *= Laser.Radius;
pty *= Laser.Radius;
ptz *= Laser.Radius;
return;
}
}
ptx = Solid(i,j,Grid.Kl).ndNode.x*Laser.Radius;
pty = Solid(i,j,Grid.Kl).ndNode.y*Laser.Radius;
ptz = Solid(i,j,Grid.Kl).ndNode.z*Laser.Radius;
return;
}*/

// ***************************************************************************
// calculate heat conduction
// ***************************************************************************

int TSolid::Solve(prec &ress, prec &resd, prec &dTmax)
{
    int i, j, k;
    prec resd2, dT2max, vn, absztat, dT2;
    HeatConduction hc;

    Laser.CalcAbsorbedHeat(); // calc local laser absorption

    ress = 0.0; // sum vn^2
    resd2 = 0.0; // sum dtheta^2
    dT2max = 0.0; // max dtheta
    Solid.ndMaxDSai = 0; // max dsai
    Solid.ndMaxTemp = 0; // max theta

    Solid.ndMinSurZ = 0; // min z
    Solid.ndVapRate = 0; // ndVdot

    //  WriteSolidData(20,30,2,2,1,3);
    //  WriteSurfaceData(20,30,2,2);
    for (i = 1; i <= Grid.iVolNodes; i++) // surface loop
    {
        for (j = jBegin(i); j <= jEnd(i); j++) // heat affected zone
        {
            if (hc.Solve(i, j, vn) < 0) // solve implicit direction
                return -1;
            absztat = fabs(At(i, j, 1).sait.z); // magnitude of velocity sai
            if (absztat > ndMaxDSai.z) // max velocity
                ndMaxDSai.z = absztat;
            ress += sqr(vn);
            if (At(i, j, 1).ndTemp > Solid.ndMaxTemp) // max temperature
                ndMaxTemp = At(i, j, 1).ndTemp;
            for (k = 1; k < Grid.kVolNodes; k++)
            {
                dT2 = sqr(At(i, j, k).ndTemp - SolidOld(i, j, k).ndTemp);
                resd2 += dT2; // temperature change
                if (dT2 > dT2max)
                    dT2max = dT2; // max temperature change
            }
        }

        // add node in haz
        while (At(i, jEnd(i), 1).ndTemp > Simulation.ndMinTemp)
        { // at left

            if (jEnd(i) < Grid.jVolNodes - 1) // node within volume grid
            {
                jEnd(i)++;
                if (hc.Solve(i, jEnd(i), vn) < 0)
                    return -1;
                if (At(i, jEnd(i), 1).ndTemp > ndMaxTemp)
                    ndMaxTemp = At(i, jEnd(i), 1).ndTemp;
            }
            else // node from surface grid
            {
                if (Grid.jVolMax != Grid.jSurNodes)
                    WarningFunction("warning: ymax too small on left");
                else if (hc.Solve(i, Grid.jVolNodes, vn) < 0)
                    return -1;
                if (At(i, Grid.jVolNodes, 1).ndTemp > ndMaxTemp)
                    ndMaxTemp = Solid(i, Grid.jVolNodes, 1).ndTemp;
                break;
            }
        }
        if (At(i, jEnd(i), 1).ndTemp <= Simulation.ndMinTemp)
        {
            for (j = jEnd(i) + 1; j <= Grid.jVolNodes; j++) // do grid update
            { // for not calculated nodes
                At(i, j, 1).ndNode = SolidOld(i, j, 1).ndNode + SolSurf(i, j).xtau * Simulation.ndDeltat;
                if (At(i, j, 1).ndNode.z > Grid.ndWpThickness)
                    At(i, j, 1).ndNode.z = Grid.ndWpThickness;
                SetInnerNodes(i, j);
                for (k = 1; k <= Grid.kVolNodes; k++)
                    At(i, j, k).ndTemp = 0.0;
            }
        }
        // drop node of
        while (At(i, jEnd(i) - 1, 1).ndTemp < Simulation.ndMinTemp &&
               // haz at left
               At(i, jEnd(i) - 1, 1).ndNode.y > Laser.ndPosition.y && jEnd(i) > jBegin(i) + 1)
        {
            for (k = 1; k <= Grid.kVolNodes; k++)
                At(i, jEnd(i), k).ndTemp = 0.0;
            jEnd(i)--;
        }
        // add note to haz
        while (At(i, jBegin(i), 1).ndTemp > Simulation.ndMinTemp)
        { // at right
            if (jBegin(i) > 2) // take from volume grid
            {
                jBegin(i)--;
                if (hc.Solve(i, jBegin(i), vn) < 0)
                    return -1;
                if (At(i, jBegin(i), 1).ndTemp > ndMaxTemp)
                    ndMaxTemp = At(i, jBegin(i), 1).ndTemp;
            }
            else // take from surface grid
            {
                if (Grid.jVolMin != 1)
                    WarningFunction("warning: ymax too small on right");
                else if (hc.Solve(i, 1, vn) < 0)
                    return -1;
                if (At(i, 1, 1).ndTemp > ndMaxTemp)
                    ndMaxTemp = At(i, 1, 1).ndTemp;
                break;
            }
        }
        // do grid update
        if (At(i, jBegin(i), 1).ndTemp <= Simulation.ndMinTemp)
        { // for not calculated nodes
            for (j = 1; j <= jBegin(i) - 1; j++)
            {
                At(i, j, 1).ndNode = SolidOld(i, j, 1).ndNode + SolSurf(i, j).xtau * Simulation.ndDeltat;
                if (At(i, j, 1).ndNode.z > Grid.ndWpThickness)
                    At(i, j, 1).ndNode.z = Grid.ndWpThickness;
                SetInnerNodes(i, j);
                for (k = 1; k <= Grid.kVolNodes; k++)
                    At(i, j, k).ndTemp = 0.0;
            }
        }
        // drop nodes of
        while (At(i, jBegin(i) + 1, 1).ndTemp < Simulation.ndMinTemp &&
               // haz at right
               At(i, jBegin(i) + 1, 1).ndNode.y < Laser.ndPosition.y && jBegin(i) < jEnd(i) - 1)
        {
            for (k = 1; k <= Grid.kVolNodes; k++)
                At(i, jBegin(i), k).ndTemp = 0.0;
            jBegin(i)++;
        }
    }

    //  WriteSolidData(20,30,2,2,1,3);
    SetBoundaryNodes(); // update artificial boundary nodes
    Solid.ndVapRate /= Laser.ndRate; // relative to ideal gaussian ablation
    dTmax = sqrt(dT2max);
    // vn^2
    ress = sqrt(ress) / (Grid.iVolNodes - 1) / (Grid.jVolNodes - 1);
    resd2 = resd2 / prec((Grid.iVolNodes - 1) * (Grid.jVolNodes - 1) * (Grid.kVolNodes - 1)); // dtheta^2
    resd = sqrt(resd2) / Simulation.ndDeltat;
    return 0;
}

// ***************************************************************************
// calculate local surface normal (out of solid surface)
// ***************************************************************************

void TSurface::CalcOutwardNormal(int i, int j, TSolid &ts)
{
    int ip, im, jp, jm;

    ip = i + 1;
    im = i - 1;
    jp = j + 1;
    jm = j - 1;
    ts(i, j, 1).xsai = 0.5 * (ts(ip, j, 1).ndNode - ts(im, j, 1).ndNode);
    ts(i, j, 1).xeta = 0.5 * (ts(i, jp, 1).ndNode - ts(i, jm, 1).ndNode);
    ts(i, j, 1).ztax = CrossProduct(ts(i, j, 1).xsai, ts(i, j, 1).xeta);
    At(i, j).Normal = ts(i, j, 1).ztax;
    At(i, j).Normal.Normalize();
}

// ***************************************************************************
// calculate local surface normal (into solid surface)
// ***************************************************************************

void TSurface::CalcInwardNormal(int i, int j, TSolid &ts)
{
    int ip, im, jp, jm;

    ip = i + 1;
    im = i - 1;
    jp = j + 1;
    jm = j - 1;
    ts(i, j, 1).xsai = 0.5 * (ts(ip, j, 1).ndNode - ts(im, j, 1).ndNode);
    ts(i, j, 1).xeta = 0.5 * (ts(i, jp, 1).ndNode - ts(i, jm, 1).ndNode);
    ts(i, j, 1).ztax = CrossProduct(ts(i, j, 1).xsai, ts(i, j, 1).xeta);
    At(i, j).Normal = -ts(i, j, 1).ztax;
    At(i, j).Normal.Normalize();
}

// ***************************************************************************
// calculate local surface area
// ***************************************************************************

void TSurface::CalcSurfaceArea(int i, int j, TSolid &ts)
{
    prec g11, g12, g22;

    g11 = ts(i, j, 1).xsai.Norm();
    g12 = ts(i, j, 1).xsai * ts(i, j, 1).xeta;
    g22 = ts(i, j, 1).xeta.Norm();
    At(i, j).ndCoarseArea = sqrt(g11 * g22 - g12 * g12);
}

// ***************************************************************************
// save variables
//
// input:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

ostream &operator<<(ostream &ps, TSurface &s)
{
    TSurface src = s.GetDimensional();
    int i, j;
    point pt = point(s.GetiMax(), s.GetjMax());

    ps << endl;
    ps << "surface data:" << endl;
    ps << "array size:\t" << pt << endl;
    ps << "Qin[W/m2]\tQdir[W/m2]\tQmult[W/m2]\t"
          "Qpar[W/m2]\tQper[W/m2]\tvx[m/s]\tvy[m/s]\tvz[m/s]" << endl;
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            ps << src(i, j) << endl;
    ps << "total incident power [W]:\t" << src.ndTotalIn << endl;
    ps << "total direct absorbed power [W]:\t" << src.ndTotalDirAbs << endl;
    ps << "total indirect absorbed power [W]:\t" << src.ndTotalMultAbs << endl;
    ps << "total transmitted power [W]:\t" << src.ndTotalTrans << endl;
    return ps;
}

// ***************************************************************************
// read settings
//
// input:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

istream &operator>>(istream &ps, TSurface &src)
{
    int i, j;
    point pt;
    bool b;

    ps >> checkstring("surface data:", &b);
    if (!b)
    {
        ps.setstate(ios::failbit);
        return ps;
    }
    src.bChanged = false;
    ps >> tab >> pt >> endl >> endl;
    src.Reallocate(pt.x, pt.y);
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            ps >> src(i, j);
    ps >> tab >> src.ndTotalIn;
    ps >> tab >> src.ndTotalDirAbs;
    ps >> tab >> src.ndTotalMultAbs;
    ps >> tab >> src.ndTotalTrans >> endl;
    src.MakeNonDimensional();
    return ps;
}

// ***************************************************************************
// bring all variables into dimensional form
// ***************************************************************************

// to do: correct RefPower (dimensional) and ndPower...
TSurface TSurface::GetDimensional()
{
    TSurface s;
    point pt = point(GetiMax(), GetjMax());
    int i, j;
    prec nd;

    s.Reallocate(pt.x, pt.y);
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
        {
            s(i, j) = At(i, j);
            s(i, j).ndIncident *= Laser.RefIntensity;
            s(i, j).ndDirAbs *= Laser.RefIntensity;
            s(i, j).ndMultAbs *= Laser.RefIntensity;
            s(i, j).ndParAbs *= Laser.RefIntensity;
            s(i, j).ndPerAbs *= Laser.RefIntensity;
            s(i, j).xtau *= RefVelocity;
        }
    nd = Laser.RefIntensity * sqr(RefLength);
    s.ndTotalTrans = ndTotalTrans * nd;
    s.ndTotalIn = ndTotalIn * nd;
    s.ndTotalDirAbs = ndTotalDirAbs * nd;
    s.ndTotalMultAbs = ndTotalMultAbs * nd;
    return s;
}

// ***************************************************************************
// bring all variables to non-dimensional form
//
// input:    RefLength, RefTime, RefVelocity
//
// ***************************************************************************

void TSurface::MakeNonDimensional()
{
    point pt = point(imax, jmax);
    int i, j;

    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
        {
            At(i, j).ndIncident /= Laser.RefIntensity;
            At(i, j).ndDirAbs /= Laser.RefIntensity;
            At(i, j).ndMultAbs /= Laser.RefIntensity;
            At(i, j).ndParAbs /= Laser.RefIntensity;
            At(i, j).ndPerAbs /= Laser.RefIntensity;
            At(i, j).xtau /= RefVelocity;
        }
    ndTotalTrans /= Laser.RefIntensity * sqr(RefLength);
    ndTotalIn /= Laser.RefIntensity * sqr(RefLength);
    ndTotalDirAbs /= Laser.RefIntensity * sqr(RefLength);
    ndTotalMultAbs /= Laser.RefIntensity * sqr(RefLength);
}

// ***************************************************************************
// save variables
//
// input:   RefLength
//
// ***************************************************************************

ostream &operator<<(ostream &ps, TSurNodes &s)
{
    TSurNodes src = s.GetDimensional();
    int i, j;
    point pt = point(s.GetiMax(), s.GetjMax());

    ps << endl;
    ps << "surface grid:" << endl;
    ps << "array size:\t" << pt << endl;
    ps << "i\tj\tx[m]\ty[m]\tz[m]" << endl;
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            ps << i << tab << j << tab << src(i, j) << endl;
    return ps;
}

// ***************************************************************************
// read settings
//
// input:   RefLength
//
// ***************************************************************************

istream &operator>>(istream &ps, TSurNodes &src)
{
    int i, j;
    point pt;
    bool b;

    ps >> checkstring("surface grid:", &b);
    if (!b)
    {
        ps.setstate(ios::failbit);
        return ps;
    }
    src.bChanged = false;
    ps >> tab >> pt >> endl >> endl;
    src.Reallocate(pt.x, pt.y);
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            ps >> tab >> tab >> src(i, j) >> endl;
    src.MakeNonDimensional();
    return ps;
}

// ***************************************************************************
// bring all variables into dimensional form
// ***************************************************************************

TSurNodes TSurNodes::GetDimensional()
{
    TSurNodes s;
    point pt = point(GetiMax(), GetjMax());
    int i, j;

    s.Reallocate(pt.x, pt.y);
    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            s(i, j) = At(i, j) * RefLength;
    return s;
}

// ***************************************************************************
// bring all variables to non-dimensional form
//
// input:    RefLength, RefTime, RefVelocity
//
// ***************************************************************************

void TSurNodes::MakeNonDimensional()
{
    point pt = point(imax, jmax);
    int i, j;

    for (i = 0; i < pt.x; i++)
        for (j = 0; j < pt.y; j++)
            At(i, j) /= RefLength;
}
