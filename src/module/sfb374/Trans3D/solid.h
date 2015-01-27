/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          solid.h  -  volume grid classes
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __SOLID_H_

#define __SOLID_H_

#include "classext.h"

// ***************************************************************************
// nodal variables
// ***************************************************************************

class TNodeVariables
{
public:
    TNodeVariables()
    {
        ndTemp = ndHeat = 0;
        ndNode = 0;
        xsai = xeta = xzta = xtau = 0;
        saix = etax = ztax = sait = 0;
    }

    prec ndTemp; // local temperature
    TPoint3D ndNode; // nodal coordinates
    prec ndHeat; // absorbed flux in node
    // metric
    TPoint3D xsai, xeta, xzta, xtau, saix, etax, ztax, sait;
};

inline ostream &operator<<(ostream &ps, TNodeVariables &src)
{
    ps << src.ndNode << '\t' << src.ndTemp << '\t' << src.ndHeat;
    return ps;
}

inline istream &operator>>(istream &ps, TNodeVariables &src)
{
    ps >> src.ndNode >> src.ndTemp >> src.ndHeat;
    return ps;
}

// ***************************************************************************
// solid class
// ***************************************************************************

class TSolid : public Tensor<TNodeVariables>
{
public:
    TSolid()
        : Tensor<TNodeVariables>()
    {
    }
    TSolid(unsigned i, unsigned j, unsigned k)
        : Tensor<TNodeVariables>(i, j, k)
    {
    }
    TSolid(const TSolid &tsrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new TNodeVariables[tsrc.imax * tsrc.jmax * tsrc.kmax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = tsrc.imax;
        jmax = tsrc.jmax;
        kmax = tsrc.kmax;
        for (unsigned k = 0; k < kmax; k++)
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * (j + jmax * k)] = tsrc(i, j, k);
        bChanged = tsrc.bChanged;
        LocalKlTangent = tsrc.LocalKlTangent;
        ndMaxDSai = tsrc.ndMaxDSai;
        ndMaxZMove = tsrc.ndMaxZMove;
        ndMaxXYMove = tsrc.ndMaxXYMove;
        ndMaxTemp = tsrc.ndMaxTemp;
        ndMinSurZ = tsrc.ndMinSurZ;
        ndVapRate = tsrc.ndVapRate;
        ndMaxVelo = tsrc.ndMaxVelo;
        jBegin = tsrc.jBegin;
        jEnd = tsrc.jEnd;
    }

    TSolid &operator=(const TSolid &tsrc)
    {
        if (&tsrc != this)
        {
            if (p)
                delete[] p;
            try
            {
                p = new TNodeVariables[tsrc.imax * tsrc.jmax * tsrc.kmax];
            }
            catch (bad_alloc)
            {
                throw TException("ERROR: not enough memory");
            }
            imax = tsrc.imax;
            jmax = tsrc.jmax;
            kmax = tsrc.kmax;
            for (unsigned k = 0; k < kmax; k++)
                for (unsigned j = 0; j < jmax; j++)
                    for (unsigned i = 0; i < imax; i++)
                        p[i + imax * (j + jmax * k)] = tsrc(i, j, k);
            bChanged = tsrc.bChanged;
            LocalKlTangent = tsrc.LocalKlTangent;
            ndMaxDSai = tsrc.ndMaxDSai;
            ndMaxZMove = tsrc.ndMaxZMove;
            ndMaxXYMove = tsrc.ndMaxXYMove;
            ndMaxTemp = tsrc.ndMaxTemp;
            ndMinSurZ = tsrc.ndMinSurZ;
            ndVapRate = tsrc.ndVapRate;
            ndMaxVelo = tsrc.ndMaxVelo;
            jBegin = tsrc.jBegin;
            jEnd = tsrc.jEnd;
        }
        return *this;
    }
    TSolid GetDimensional(); // all values to dimensional form
    void MakeNonDimensional(); // all values to non-dimensional form
    void ReSize(unsigned iNewSize, unsigned jNewSize, unsigned kNewSize,
                int niOffset = 0, int njOffset = 0, int nkOffset = 0);
    void InsertBegin(int i, int j, int k);
    void InsertEnd(int i, int j, int k);

    int Solve(prec &, prec &, prec &); // solve heat conduction

    void CalcBaseVect(int, int, int);
    prec CalcMetricTensor(int, int, int, rmmatrix &, TPoint3D &);
    void CalcTempDerivative(int, int, int, TPoint3D &,
                            prec &, prec &, prec &, prec &, prec &);

    void ResetTemperature(); // set all temperatures to 0
    void SetBoundaryNodes(void); // set artificial boundary nodes
    void SetInnerNodes(int, int, prec dmin = 0,
                       bool bPotential = false); // set inner nodes
    void GridMove(int, int, prec, // update surface nodes
                  bool bPotential = false);

    bool bChanged; // change flag;
    TPoint3D LocalKlTangent; // local zta-tangent in volume at kmax
    TPoint3D ndMaxDSai; // maximum nodal displacement
    prec ndMaxZMove; // maximum zta displacement
    prec ndMaxXYMove; // maximum sai,eta displacement
    prec ndMaxTemp; // maximum temperature in solid
    prec ndMinSurZ; // deepest surface point
    prec ndVapRate; // vaporization rate
    prec ndMaxVelo; // maximum ablation velocity

    ivector jBegin; // begin of heat affected zone
    ivector jEnd; // end of heat affected zone
};

// ***************************************************************************
// surface variables
// ***************************************************************************

class TSurfaceVariables
{
public:
    TSurfaceVariables()
    {
        ndIncident = ndDirAbs = ndMultAbs = ndParAbs = ndPerAbs = 0;
        ndArea = ndCoarseArea = 0;
        Normal.Set(0, 0, 1);
        xtau = 0;
    }

    prec ndIncident; // incident power
    prec ndDirAbs; // direct absorbed power
    prec ndMultAbs; // power due to multiple absorption
    prec ndParAbs; // parallel polarized absorption
    prec ndPerAbs; // perpendicular polarized absorption
    prec ndArea; // area surface patch
    prec ndCoarseArea; // area calculated w/o sub-patches
    int iNumRays; // number i sub-patches
    int jNumRays; // number j sub-patches
    TPoint3D Normal; // surface inward normal
    TPoint3D xtau; // nodal velocity
};

inline ostream &operator<<(ostream &ps, TSurfaceVariables &src)
{
    ps << src.ndIncident << '\t' << src.ndDirAbs << '\t' << src.ndMultAbs
       << '\t' << src.ndParAbs << '\t' << src.ndPerAbs << '\t' << src.xtau;
    return ps;
}

inline istream &operator>>(istream &ps, TSurfaceVariables &src)
{
    ps >> src.ndIncident >> src.ndDirAbs >> src.ndMultAbs
        >> src.ndParAbs >> src.ndPerAbs >> src.xtau;
    return ps;
}

// ***************************************************************************
// surface class
// ***************************************************************************

class TSurface : public Matrix<TSurfaceVariables>
{
public:
    TSurface()
        : Matrix<TSurfaceVariables>()
    {
    }
    TSurface(unsigned i, unsigned j)
        : Matrix<TSurfaceVariables>(i, j)
    {
    }
    TSurface(const TSurface &msrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new TSurfaceVariables[msrc.imax * msrc.jmax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = msrc.imax;
        jmax = msrc.jmax;
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] = msrc(i, j);
        bChanged = msrc.bChanged;
        ndTotalIn = msrc.ndTotalIn;
        ndTotalDirAbs = msrc.ndTotalDirAbs;
        ndTotalMultAbs = msrc.ndTotalMultAbs;
        ndTotalTrans = msrc.ndTotalTrans;
    }

    TSurface &operator=(const TSurface &msrc)
    {
        if (&msrc != this)
        {
            if (p)
                delete[] p;
            try
            {
                p = new TSurfaceVariables[msrc.imax * msrc.jmax];
            }
            catch (bad_alloc)
            {
                throw TException("ERROR: not enough memory");
            }
            imax = msrc.imax;
            jmax = msrc.jmax;
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * j] = msrc(i, j);
            bChanged = msrc.bChanged;
            ndTotalIn = msrc.ndTotalIn;
            ndTotalDirAbs = msrc.ndTotalDirAbs;
            ndTotalMultAbs = msrc.ndTotalMultAbs;
            ndTotalTrans = msrc.ndTotalTrans;
        }
        return *this;
    }
    TSurface GetDimensional(); // all values to dimensional form
    void MakeNonDimensional(); // all values to non-dimensional form
    void CalcInwardNormal(int, int, TSolid &);
    void CalcOutwardNormal(int, int, TSolid &);
    void CalcSurfaceArea(int, int, TSolid &);

    bool bChanged; // change flag;
    prec ndTotalTrans; // total transmitted power
    prec ndTotalIn; // total incident power
    prec ndTotalDirAbs; // total direct absorbed power
    prec ndTotalMultAbs; // total indirect absorbed power
};

// ***************************************************************************
// surface node class
// ***************************************************************************

class TSurNodes : public Matrix<TPoint3D>
{
public:
    TSurNodes()
        : Matrix<TPoint3D>(){};
    TSurNodes(unsigned i, unsigned j)
        : Matrix<TPoint3D>(i, j){};
    TSurNodes(const TSurNodes &msrc)
    {
        if (p)
            delete[] p;
        try
        {
            p = new TPoint3D[msrc.imax * msrc.jmax];
        }
        catch (bad_alloc)
        {
            throw TException("ERROR: not enough memory");
        }
        imax = msrc.imax;
        jmax = msrc.jmax;
        for (unsigned j = 0; j < jmax; j++)
            for (unsigned i = 0; i < imax; i++)
                p[i + imax * j] = msrc(i, j);
        bChanged = msrc.bChanged;
    }

    TSurNodes &operator=(const TSurNodes &msrc)
    {
        if (&msrc != this)
        {
            if (p)
                delete[] p;
            try
            {
                p = new TPoint3D[msrc.imax * msrc.jmax];
            }
            catch (bad_alloc)
            {
                throw TException("ERROR: not enough memory");
            }
            imax = msrc.imax;
            jmax = msrc.jmax;
            for (unsigned j = 0; j < jmax; j++)
                for (unsigned i = 0; i < imax; i++)
                    p[i + imax * j] = msrc(i, j);
            bChanged = msrc.bChanged;
        }
        return *this;
    }
    TSurNodes GetDimensional(); // all values to dimensional form
    void MakeNonDimensional(); // all values to non-dimensional form

    bool bChanged; // change flag;
};

ostream &operator<<(ostream &, TSolid &);
istream &operator>>(istream &, TSolid &);
ostream &operator<<(ostream &, TSurface &);
istream &operator>>(istream &, TSurface &);
ostream &operator<<(ostream &, TSurNodes &);
istream &operator>>(istream &, TSurNodes &);

extern TSolid Solid, SolidOld;
extern TSurface SolSurf, SolSurfOld;
extern TSurNodes Surface;

extern void WriteSolidData(int imin = 1, int imax = -1, int jmin = 1, int jmax = -1,
                           int kmin = 1, int kmax = -1);
extern void WriteSurfaceData(int imin = 1, int imax = -1, int jmin = 1, int jmax = -1);
#endif
