/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          grid.cpp  -  grid settings
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#include "grid.h"
#include "laser.h"
#include "material.h"
#include "solid.h"
#include "simul.h"
#include "fortran.h"
#include "main.h"
#include "matvect.h"

TGrid Grid, GridOld;

// ***************************************************************************
// random number between 0 and imax
// ***************************************************************************

int random(int imax)
{
    int i;

    i = int(imax * prec(rand()) / RAND_MAX + 0.5);
    return i;
}

// ***************************************************************************
// grid settings class
// ***************************************************************************

// copy constructor

TGrid::TGrid(const TGrid &src)
{
    ProfileName = src.ProfileName;
    bChanged = src.bChanged;
    ndWpLength = src.ndWpLength;
    ndWpWidth = src.ndWpWidth;
    ndWpThickness = src.ndWpThickness;
    ndXFront = src.ndXFront;
    ndXBack = src.ndXBack;
    ndYSide = src.ndYSide;
    iVolNodes = src.iVolNodes;
    jVolNodes = src.jVolNodes;
    kVolNodes = src.kVolNodes;
    iIniNodes = src.iIniNodes;
    jIniNodes = src.jIniNodes;
    kIniNodes = src.kIniNodes;
    iSurNodes = src.iSurNodes;
    jSurNodes = src.jSurNodes;
    iVolMin = src.iVolMin;
    iVolMax = src.iVolMax;
    jVolMin = src.jVolMin;
    jVolMax = src.jVolMax;
    ndSpacing = src.ndSpacing;
    ndDepth = src.ndDepth;
    iLeftBound = src.iLeftBound;
    iRightBound = src.iRightBound;
    iFrontBound = src.iFrontBound;
    iBackBound = src.iBackBound;
    bGridMove = src.bGridMove;
    iGridMove = src.iGridMove;
    TangentWarp = src.TangentWarp;
    TangentDir = src.TangentDir;
    TangentBound = src.TangentBound;
    iWarpPower = src.iWarpPower;
    iWarpAverage = src.iWarpAverage;
    iRelax = src.iRelax;
    Dconst = src.Dconst;
    MinRadius = src.MinRadius;
    iSymmetry = src.iSymmetry;
    jSymmetry = src.jSymmetry;
    xDistance = src.xDistance;
    yDistance = src.yDistance;
}

// reset to default values

void TGrid::Reset()
{
    ProfileName.erase();
    bChanged = false;
    ndWpLength = 3;
    ndWpWidth = 3;
    ndWpThickness = 25;
    ndXFront = 1.5;
    ndXBack = 1.5;
    ndYSide = 1.5;
    iVolNodes = 20;
    jVolNodes = 20;
    kVolNodes = 20;
    ndSpacing = 1e-3;
    ndDepth = 0.5;
    iLeftBound = symmetric;
    iRightBound = symmetric;
    iFrontBound = symmetric;
    iBackBound = symmetric;
    bGridMove = false;
    iGridMove = updated;
    TangentWarp = 0.2;
    TangentDir = 1e20;
    TangentBound = 5.0;
    iWarpPower = 2;
    iWarpAverage = 3;
    iRelax = 1;
    Dconst = 25;
    MinRadius = 1;
    iSymmetry = -1;
    jSymmetry = -1;
}

// ***************************************************************************
// save settings
//
// input:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

ostream &operator<<(ostream &ps, TGrid &g)
{
    TGrid src = g.GetDimensional();

    ps << "grid settings:" << endl;
    ps << endl;
    ps << "profile name:\t" << src.ProfileName << endl;
    ps << "workpiece length [m]:\t" << src.ndWpLength << endl;
    ps << "workpiece width [m]:\t" << src.ndWpWidth << endl;
    ps << "workpiece thickness [m]:\t" << src.ndWpThickness << endl;
    ps << "volume grid in front of laser [m]:\t" << src.ndXFront << endl;
    ps << "volume grid behind laser [m]:\t" << src.ndXBack << endl;
    ps << "volume grid beneath laser [m]:\t" << src.ndYSide << endl;
    ps << "x number volume nodes:\t" << src.iVolNodes << endl;
    ps << "y number volume nodes:\t" << src.jVolNodes << endl;
    ps << "z number volume nodes:\t" << src.kVolNodes << endl;
    ps << "x number initial volume nodes:\t" << src.iIniNodes << endl;
    ps << "y number initial volume nodes:\t" << src.jIniNodes << endl;
    ps << "z number initial volume nodes:\t" << src.kIniNodes << endl;
    ps << "x number surface nodes:\t" << src.iSurNodes << endl;
    ps << "y number surface nodes:\t" << src.jSurNodes << endl;
    ps << "x begin volume grid:\t" << src.iVolMin << endl;
    ps << "x end volume grid:\t" << src.iVolMax << endl;
    ps << "y begin volume grid:\t" << src.jVolMin << endl;
    ps << "y end volume grid:\t" << src.jVolMax << endl;
    ps << "first z nodal spacing [m]:\t" << src.ndSpacing << endl;
    ps << "depth of volume grid [m]:\t" << src.ndDepth << endl;
    ps << "left boundary condition:\t" << src.iLeftBound << endl;
    ps << "right boundary condition:\t" << src.iRightBound << endl;
    ps << "front boundary condition:\t" << src.iFrontBound << endl;
    ps << "rear boundary condition:\t" << src.iBackBound << endl;
    ps << "surface remeshing:\t" << src.bGridMove << endl;
    ps << "grid status:\t" << src.iGridMove << endl;
    ps << "tangent warp constant:\t" << src.TangentWarp << endl;
    ps << "tangent direction constant:\t" << src.TangentDir << endl;
    ps << "tangent boundary constant:\t" << src.TangentBound << endl;
    ps << "tangent warp power:\t" << src.iWarpPower << endl;
    ps << "tangent warp average:\t" << src.iWarpAverage << endl;
    ps << "relaxation steps:\t" << src.iRelax << endl;
    ps << "grid potential constant:\t" << src.Dconst << endl;
    ps << "grid potential minimum:\t" << src.MinRadius << endl;
    ps << "x symmetry plane:\t" << src.iSymmetry << endl;
    ps << "y symmetry plane:\t" << src.jSymmetry << endl;
    return ps << endl;
}

// ***************************************************************************
// read settings
//
// output:   RefLength, RefTime, RefVelocity, RefIntensity
//
// ***************************************************************************

istream &operator>>(istream &ps, TGrid &src)
{
    src.bChanged = true;
    if (!CheckHeader(ps, "grid settings:"))
        return ps;
    src.bChanged = false;
    ps >> tab;
    getline(ps, src.ProfileName);
    ps >> tab >> src.ndWpLength;
    ps >> tab >> src.ndWpWidth;
    ps >> tab >> src.ndWpThickness;
    ps >> tab >> src.ndXFront;
    ps >> tab >> src.ndXBack;
    ps >> tab >> src.ndYSide;
    ps >> tab >> src.iVolNodes;
    ps >> tab >> src.jVolNodes;
    ps >> tab >> src.kVolNodes;
    ps >> tab >> src.iIniNodes;
    ps >> tab >> src.jIniNodes;
    ps >> tab >> src.kIniNodes;
    ps >> tab >> src.iSurNodes;
    ps >> tab >> src.jSurNodes;
    ps >> tab >> src.iVolMin;
    ps >> tab >> src.iVolMax;
    ps >> tab >> src.jVolMin;
    ps >> tab >> src.jVolMax;
    ps >> tab >> src.ndSpacing;
    ps >> tab >> src.ndDepth;
    ps >> tab >> src.iLeftBound;
    ps >> tab >> src.iRightBound;
    ps >> tab >> src.iFrontBound;
    ps >> tab >> src.iBackBound;
    ps >> tab >> src.bGridMove;
    ps >> tab >> src.iGridMove;
    ps >> tab >> src.TangentWarp;
    ps >> tab >> src.TangentDir;
    ps >> tab >> src.TangentBound;
    ps >> tab >> src.iWarpPower;
    ps >> tab >> src.iWarpAverage;
    ps >> tab >> src.iRelax;
    ps >> tab >> src.Dconst;
    ps >> tab >> src.MinRadius;
    ps >> tab >> src.iSymmetry;
    ps >> tab >> src.jSymmetry >> endl >> endl;
    src.MakeNonDimensional();
    src.Update();

    return ps;
}

// ***************************************************************************
// bring all variables into dimensional form
//
// input:   RefLength
//
// ***************************************************************************

TGrid TGrid::GetDimensional()
{
    TGrid g = *this;

    g.ndWpLength *= RefLength;
    g.ndWpWidth *= RefLength;
    g.ndWpThickness *= RefLength;
    g.ndXFront *= RefLength;
    g.ndXBack *= RefLength;
    g.ndYSide *= RefLength;
    g.ndSpacing *= RefLength;
    g.ndDepth *= RefLength;
    return g;
}

// ***************************************************************************
// bring all variables to non-dimensional form
//
// input:    RefLength
//
// ***************************************************************************

void TGrid::MakeNonDimensional()
{
    ndWpLength /= RefLength;
    ndWpWidth /= RefLength;
    ndWpThickness /= RefLength;
    ndXFront /= RefLength;
    ndXBack /= RefLength;
    ndYSide /= RefLength;
    ndSpacing /= RefLength;
    ndDepth /= RefLength;
}

// ***************************************************************************
// update dependent variables
//
// input:   ndPosition, ndWpLength, ndWpWidth, ndDepth, ndSpacing, VolNodes...
// output:  iVolNodes, iSurNodes, Surface.x, Solid.ndNode, Solid.jMin,
//          Solid.jMax, SolSur.Normal...
//
// call after laser update!
//
// ***************************************************************************

void TGrid::Update()
{
    SetupGrid();
    xDistance = Solid(1, 1, 1).ndNode.x - Solid(2, 1, 1).ndNode.x;
    yDistance = Solid(1, 2, 1).ndNode.y - Solid(1, 1, 1).ndNode.y;
}

// ***************************************************************************
// save settings (temporary)
// ***************************************************************************

void TGrid::Save(int unit)
{
    unitwriteln(unit, "\nGeometrie:");
    unitwrite(unit, "Profilname =\t");
    unitwriteln(unit, ProfileName.c_str());
    unitwrite(unit, ndWpLength * RefLength, "Werkstücklänge =\t%le\n");
    unitwrite(unit, ndWpWidth * RefLength, "Werkstückbreite =\t%le\n");
    unitwrite(unit, ndWpThickness * RefLength, "Werkstückdicke =\t%le\n");
    unitwrite(unit, Laser.ndInitialPos.x * RefLength, "Laserposition x =\t%le\n");
    unitwrite(unit, Laser.ndInitialPos.y * RefLength, "Laserposition y =\t%le\n");
    unitwrite(unit, ndXFront * RefLength, "WEZ vor Laser =\t%le\n");
    unitwrite(unit, ndXBack * RefLength, "WEZ hinter Laser =\t%le\n");
    unitwrite(unit, ndYSide * RefLength, "Breite WEZ =\t%le\n");
    unitwrite(unit, iVolNodes, "x-Gitterpunkte =\t%i\n");
    unitwrite(unit, jVolNodes, "y-Gitterpunkte =\t%i\n");
    unitwrite(unit, kVolNodes, "z-Gitterpunkte =\t%i\n");
    unitwrite(unit, iIniNodes, "x-Anfangspunkte =\t%i\n");
    unitwrite(unit, jIniNodes, "y-Anfangspunkte =\t%i\n");
    unitwrite(unit, kIniNodes, "z-Anfangspunkte =\t%i\n");
    unitwrite(unit, ndSpacing * RefLength, "min. Abstand =\t%le\n");
    unitwrite(unit, ndDepth * RefLength, "Rechentiefe =\t%le\n");
    unitwrite(unit, iLeftBound, "Rand links =\t%i\n");
    unitwrite(unit, iRightBound, "Rand rechts =\t%i\n");
    unitwrite(unit, iFrontBound, "Rand vorne =\t%i\n");
    unitwrite(unit, iBackBound, "Rand hinten =\t%i\n");
    unitwrite(unit, 0.0, "cs1 =\t%le\n");
    unitwrite(unit, TangentWarp, "ck1 =\t%le\n");
    unitwrite(unit, TangentDir, "ck2 =\t%le\n");
    unitwrite(unit, TangentBound, "ck3 =\t%le\n");
    unitwrite(unit, iWarpPower, "ick2 =\t%i\n");
    unitwrite(unit, iWarpAverage, "ic2ij =\t%i\n");
    unitwrite(unit, bGridMove, "bGridMove =\t%i\n");
    unitwrite(unit, iRelax, "nRelax =\t%le\n");
    unitwrite(unit, Dconst, "Dconst =\t%le\n");
    unitwrite(unit, MinRadius, "r0 =\t%i\n");
}

// ***************************************************************************
// read settings (temporary)
// ***************************************************************************

void TGrid::Read(int unit, float vers)
{
    char buffer[100];

    unitreadln(unit, buffer);
    unitreadln(unit, buffer);
    ProfileName = unitreadln(unit, buffer);
    ndWpLength = readreal(unit) / RefLength;
    ndWpWidth = readreal(unit) / RefLength;
    ndWpThickness = readreal(unit) / RefLength;
    Laser.ndInitialPos.x = readreal(unit) / RefLength;
    Laser.ndInitialPos.y = readreal(unit) / RefLength;
    ndXFront = readreal(unit) / RefLength;
    ndXBack = readreal(unit) / RefLength;
    ndYSide = readreal(unit) / RefLength;
    iVolNodes = readint(unit);
    jVolNodes = readint(unit);
    kVolNodes = readint(unit);
    iIniNodes = readint(unit);
    jIniNodes = readint(unit);
    kIniNodes = readint(unit);
    ndSpacing = readreal(unit) / RefLength;
    ndDepth = readreal(unit) / RefLength;
    iLeftBound = readint(unit);
    iRightBound = readint(unit);
    iFrontBound = readint(unit);
    iBackBound = readint(unit);
    readreal(unit); // cs1
    TangentWarp = readreal(unit);
    TangentDir = readreal(unit);
    TangentBound = readreal(unit);
    iWarpPower = readint(unit);
    iWarpAverage = readint(unit);
    bGridMove = readbool(unit);
    iRelax = readint(unit);
    Dconst = readreal(unit);
    MinRadius = readreal(unit);
    Update();
}

// ***************************************************************************
// generate volume grid from surface grid
//
// input:   Surface, SurNodes, VolNodes, ...
// output:  SolSurf.Normal, Solid
//
// ***************************************************************************

int TGrid::SetVolumeGrid()
{
    int i, j, k, is, js, jmin, jmax;
    prec xfront, xrear, yright, yleft;

    xfront = Laser.ndPosition.x + ndXFront;
    xrear = Laser.ndPosition.x - ndXBack;
    yright = Laser.ndPosition.y - ndYSide;
    yleft = Laser.ndPosition.y + ndYSide;
    if (xfront < Surface(iSurNodes - 3, 1).x)
    {
        xfront = Surface(iSurNodes - 3, 1).x;
        WarningFunction("*** LASER DOES NOT HIT WORKPIECE AT TAU=0;"
                        " ADVANCING IN TIME");
    }
    if (xrear > ndWpLength - 0.5)
    {
        ErrorFunction("*** LASER HAS ALREADY PASSED WORKPIECE AT TAU=0;"
                      " PROGRAM TERMINATED");
        return -1;
    }
    if (yleft < Surface(1, 4).y || yright > Surface(1, jSurNodes - 3).y)
    {
        ErrorFunction("*** LASER PASSES ON EITHER SIDE OF WORKPIECE W/O CONTACT;"
                      " PROGRAM TERMINATED");
        return -1;
    }

    // determine volume grid position within surface grid

    if (xfront >= ndWpLength)
        iVolMin = 1; // start at first node
    else
    {
        for (is = 2; is <= iSurNodes; is++)
        {
            iVolMin = is - 1;
            if (Surface(is, 1).x < xfront)
                break;
        }
    }
    if (xrear <= 0.0)
        iVolMax = iSurNodes; // end at last node
    else
    {
        for (is = iVolMin + 1; is <= iSurNodes; is++)
        {
            iVolMax = is;
            if (Surface(is, 1).x < xrear)
                break;
        }
    }
    iVolNodes = 1 + iVolMax - iVolMin; // number of volume nodes

    jVolMin = jSurNodes;
    jVolMax = 1;
    for (is = 1; is <= iVolMax; is++) // check for all nodes
    {
        if (yright <= 0.0)
            jmin = 1; // start at first node
        else
        {
            for (js = 2; js <= jSurNodes; js++)
            {
                jmin = js - 1;
                if (Surface(is, js).y > yright)
                    break;
            }
        }
        if (yleft >= ndWpWidth)
            jmax = jSurNodes; // end at last node
        else
        {
            for (js = jmin + 1; js <= jSurNodes; js++)
            {
                jmax = js;
                if (Surface(is, js).y >= yleft)
                    break;
            }
        }
        if (jmin < jVolMin) // look for minimum
            jVolMin = jmin;
        if (jmax > jVolMax) // look for maximum
            jVolMax = jmax;
    }
    jVolNodes = 1 + jVolMax - jVolMin; // number of j volume nodes
    ResizeVolumeArrays();

    Solid.ndMinSurZ = 1.0e10;
    for (is = iVolMin - 1; is <= iVolMax + 1; is++) // surface volume nodes
    {
        i = is - iVolMin + 1;
        for (js = jVolMin - 1; js <= jVolMax + 1; js++)
        {
            j = js - jVolMin + 1;
            Solid(i, j, 1).ndNode = Surface(is, js);
            if (Solid(i, j, 1).ndNode.z < Solid.ndMinSurZ)
                // deepest surface point
                Solid.ndMinSurZ = Solid(i, j, 1).ndNode.z;
        }
    }

    Solid.LocalKlTangent = TPoint3D(0, 0, 1); // z-direction at ndDepth
    for (i = 1; i <= iVolNodes; i++)
    {
        Solid.jBegin(i) = 2;
        Solid.jEnd(i) = jVolNodes - 1;
        for (j = 1; j <= jVolNodes; j++)
        {
            SolSurf(i, j).xtau = 0;
            SolSurf.CalcInwardNormal(i, j, Solid);
            Solid.SetInnerNodes(i, j); // inner nodes
            for (k = 1; k <= kVolNodes; k++)
                Solid(i, j, k).ndTemp = 0.0;
        }
    }
    Solid.SetBoundaryNodes(); // artificial boundary nodes
    return 0;
}

// ***************************************************************************
// grid setup
//
// input:   ndWpLength, ndWpWidth, ndDepth, ndSpacing, iVolNodes...
// output:  iVolNodes, iSurNodes, Surface.x, Solid.ndNode, Solid.jMin,
//          Solid.jMax, SolSurf.Normal...
//
// ***************************************************************************

int TGrid::SetupGrid()
{
    int is, js;
    prec dx, dy;

    if (Simulation.iStatus != TSimulation::new_grid)
        return 0; // no new grid

    // surface grid

    dx = (ndXFront + ndXBack) / (iIniNodes - 1.0); // nodal spcaing
    dy = (ndYSide + ndYSide) / (jIniNodes - 1.0);
    iSurNodes = int(ndWpLength / dx + 1.5); // number surface nodes
    jSurNodes = int(ndWpWidth / dy + 1.5);
    dx = ndWpLength / (iSurNodes - 1);
    dy = ndWpWidth / (jSurNodes - 1);
    ResizeSurfaceArrays();

    for (is = 0; is <= iSurNodes + 1; is++) // set surface nodes
        for (js = 0; js <= jSurNodes + 1; js++)
        {
            Surface(is, js).x = (iSurNodes - is) * dx; // reverse i direction
            Surface(is, js).y = (js - 1) * dy;
            Surface(is, js).z = 0.0;
        }
    return SetVolumeGrid();
}

// ***************************************************************************
// resize arrays for volume grid
// ***************************************************************************

void TGrid::ResizeVolumeArrays()
{
    Solid.ReSize(iVolNodes + 2, jVolNodes + 2, kVolNodes + 2);
    SolSurf.ReSize(iVolNodes + 2, jVolNodes + 2);
}

// ***************************************************************************
// resize surface arrays
// ***************************************************************************

void TGrid::ResizeSurfaceArrays()
{
    Surface.ReSize(iSurNodes + 2, jSurNodes + 2);
}

// ***************************************************************************
// resize arrays
// ***************************************************************************

void TGrid::ResizeArrays()
{
    ResizeSurfaceArrays();
    ResizeVolumeArrays();
}

// ***************************************************************************
// insert/drop nodes at begin of volume arrays
// ***************************************************************************

void TGrid::InsertBegin(int i, int j)
{
    Solid.InsertBegin(i, j, 0);
    SolSurf.InsertBegin(i, j);
}

// ***************************************************************************
// insert/drop nodes at end of volume arrays
// ***************************************************************************

void TGrid::InsertEnd(int i, int j)
{
    Solid.InsertEnd(i, j, 0);
    SolSurf.InsertEnd(i, j);
}

// ***************************************************************************
// move grid with laser and heat (add or drop nodes)
// ***************************************************************************

int TGrid::UpdateVolumeGrid()
{
    int i, j, is, js, iOldMin, iOldMax, jOldMin, jOldMax,
        ifront, irear, ileft, iright;
    prec xmin, xmax, ymin, ymax;

    Solid.LocalKlTangent = TPoint3D(0, 0, 1); // far inside: z direction
    if (Simulation.bFollowLaser) // move grid with laser
    {
        xmin = xmax = Laser.ndPosition.x;
        ymin = ymax = Laser.ndPosition.y;
    }
    else
    {
        xmin = xmax = Laser.ndInitialPos.x;
        ymin = ymax = Laser.ndInitialPos.y;
    }
    xmin -= ndXBack; // desired volume grid size
    xmax += ndXFront;
    ymin -= ndYSide;
    ymax += ndYSide;
    ymin = max(ymin, 0.); // restrict to >1 (for symmetric calc)
    iOldMin = iVolMin;
    iOldMax = iVolMax;
    jOldMin = jVolMin;
    jOldMax = jVolMax;
    xmin -= 1e-8; // temporary

    if (xmin >= ndWpLength || xmax < 0 || ymin >= ndWpWidth || ymax < 0)
    {
        ErrorFunction("*** LASER HAS MOVED BEYOND WORKPIECE ***");
        return -1;
    }

    is = iVolMin;
    for (js = 1; js <= jSurNodes; js++) // check number nodes to be added at front
    {
        //    if(Surface(is,js).x<xmax)  // add node at front
        if (Surface(is - 1, js).x <= xmax) // add node at front
        {
            is--;
            if (is <= 1)
            {
                ErrorFunction("Too many nodes added at front");
                return -1;
            }
            js = 1; // start again
        }
        //    else if(Surface(is+1,js).x>=xmax)  // drop node at front
        else if (Surface(is, js).x > xmax) // drop node at front
        {
            is++;
            if (is >= iSurNodes - 1)
            {
                ErrorFunction("Too many nodes dropped at front");
                return -1;
            }
            js = 1; // start again
        }
    }
    ifront = iVolMin - is; // number of nodes to be added or dropped at front
    is = iVolMax;
    for (js = 1; js <= jSurNodes; js++) // check nodes to be added or dropped at rear
    {
        //    if(Surface(is,js).x>xmin)  // add node at rear
        if (Surface(is + 1, js).x >= xmin) // add node at rear
        {
            is++;
            if (is >= iSurNodes)
            {
                ErrorFunction("Too many nodes added at rear");
                return -1;
            }
            js = 1; // start again
        }
        //    else if(Surface(is-1,js).x<=xmin)  // drop node at rear
        else if (Surface(is, js).x <= xmin) // drop node at rear
        {
            is--;
            if (is <= 2)
            {
                ErrorFunction("Too many nodes dropped at rear");
                return -1;
            }
            js = 1; // start again
        }
    }
    irear = is - iVolMax; // number of nodes to be added or dropped at front

    if (ifront > 0) // add ifront nodes at front
    {
        iVolMin -= ifront;
        iVolNodes += ifront;
        InsertBegin(ifront, 0);
        for (j = 0; j <= jVolNodes + 1; j++) // copy coordinates from surface array
        {
            js = jVolMin - 1 + j;
            for (i = 0; i <= ifront; i++)
                Solid(i, j, 1).ndNode = Surface(iVolMin - 1 + i, js);
        }
        for (j = 1; j <= jVolNodes; j++) // inner nodes
            for (i = 1; i <= ifront; i++)
            {
                SolSurf.CalcInwardNormal(i, j, Solid);
                Solid.SetInnerNodes(i, j);
            }
    }
    if (irear > 0) // add irear nodes at rear
    {
        iVolMax += irear;
        iVolNodes += irear;
        InsertEnd(irear, 0);
        for (j = 0; j <= jVolNodes + 1; j++) // copy coordinates from surface array
        {
            js = jVolMin - 1 + j;
            for (i = iVolNodes + 1 - irear; i <= iVolNodes + 1; i++)
                Solid(i, j, 1).ndNode = Surface(iVolMin - 1 + i, js);
        }
        for (j = 1; j <= jVolNodes; j++) // inner nodes
            for (i = iVolNodes + 1 - irear; i <= iVolNodes; i++)
            {
                SolSurf.CalcInwardNormal(i, j, Solid);
                Solid.SetInnerNodes(i, j);
            }
    }
    if (ifront < 0) // drop -ifront nodes at front
    {
        iVolMin -= ifront;
        iVolNodes += ifront;
        InsertBegin(ifront, 0);
    }
    if (irear < 0) // drop -irear nodes at rear
    {
        iVolMax += irear;
        iVolNodes += irear;
        InsertEnd(irear, 0);
    }

    js = jVolMin;
    for (is = 1; is <= iSurNodes; is++) // check number nodes to be added at right
    {
        if (Surface(is, js).y > ymin) // add node at right
        {
            js--;
            if (js <= 1)
            {
                ErrorFunction("Too many nodes added at right");
                return -1;
            }
            is = 1; // start again
        }
        else if (Surface(is, js + 1).y <= ymin) // drop node at right
        {
            js++;
            if (js >= jSurNodes - 1)
            {
                ErrorFunction("Too many nodes dropped at right");
                return -1;
            }
            is = 1; // start again
        }
    }
    iright = jVolMin - js; // number of nodes to be added or dropped at right
    js = jVolMax;
    for (is = 1; is <= iSurNodes; is++) // check nodes to be added or dropped at left
    {
        if (Surface(is, js).y < ymax) // add node at left
        {
            js++;
            if (js >= jSurNodes)
            {
                ErrorFunction("Too many nodes added at left");
                return -1;
            }
            is = 1; // start again
        }
        else if (Surface(is, js - 1).y >= ymax) // drop node at left
        {
            js--;
            if (js <= 2)
            {
                ErrorFunction("Too many nodes dropped at left");
                return -1;
            }
            is = 1; // start again
        }
    }
    ileft = js - jVolMax; // number of nodes to be added or dropped at left

    if (iright > 0) // add iright nodes at right
    {
        jVolMin -= iright;
        jVolNodes += iright;
        InsertBegin(0, iright);
        for (i = 0; i <= iVolNodes + 1; i++) // copy coordinates from surface array
        {
            is = iVolMin - 1 + i;
            for (j = 0; j <= iright; j++)
                Solid(i, j, 1).ndNode = Surface(is, jVolMin - 1 + j);
        }
        for (i = 1; i <= iVolNodes; i++) // inner nodes
            for (j = 1; j <= iright; j++)
            {
                SolSurf.CalcInwardNormal(i, j, Solid);
                Solid.SetInnerNodes(i, j);
            }
    }
    if (ileft > 0) // add ileft nodes at left
    {
        jVolMax += ileft;
        jVolNodes += ileft;
        InsertEnd(0, ileft);
        for (i = 0; i <= iVolNodes + 1; i++) // copy coordinates from surface array
        {
            is = iVolMin - 1 + i;
            for (j = jVolNodes + 1 - ileft; j <= jVolNodes + 1; j++)
                Solid(i, j, 1).ndNode = Surface(is, jVolMin - 1 + j);
        }
        for (i = 1; i <= iVolNodes; i++) // inner nodes
            for (j = jVolNodes + 1 - ileft; j <= jVolNodes; j++)
            {
                SolSurf.CalcInwardNormal(i, j, Solid);
                Solid.SetInnerNodes(i, j);
            }
    }
    if (iright < 0) // drop -iright nodes at right
    {
        jVolMin -= iright;
        jVolNodes += iright;
        InsertBegin(0, iright);
    }
    if (ileft < 0) // drop -ileft nodes at left
    {
        jVolMax += ileft;
        jVolNodes += ileft;
        InsertEnd(0, ileft);
    }
    if (iOldMin != iVolMin || iOldMax != iVolMax || jOldMin != jVolMin || jOldMax != jVolMax)
        Solid.SetBoundaryNodes(); // update boundary nodes if necessary
    return 0;
}

// ***************************************************************************
// adjust grid after iteration
// ***************************************************************************

void TGrid::UpdateGrid(void)
{
    int i, j, is, js, count, n, m, nRow,
        iinit, iend, istep, jinit, jend, jstep;
    rmatrix MatA(5, 6), MatB(5, 1);
    Matrix<TPoint3D> SurfOld;
    rvector SurfVect(7);
    MatrixOp matop;
    TPoint3D pts[5];
    prec dx, dxmax, z0, p;

    dxmax = 0.0;
    for (i = 1; i <= iVolNodes; i++) // displacement
        for (j = 1; j <= jVolNodes; j++)
        {
            SolSurf(i, j).xtau = 0;
            dx = (SolidOld(i, j, 1).ndNode - Solid(i, j, 1).ndNode).Norm();
            if (dx > dxmax)
                dxmax = dx;
        }
    Solid.ndMaxZMove *= Simulation.ndDeltat;
    iGridMove = TGrid::reduced_step;
    if (Solid.ndMaxZMove > Simulation.ndMaxZMove) // reduce time step for too high
        return; // ablation rate

    for (i = 1; i <= iVolNodes; i++) // updating surface array
    {
        is = iVolMin - 1 + i;
        for (j = 1; j <= jVolNodes; j++)
        {
            js = jVolMin - 1 + j;
            Surface(is, js) = Solid(i, j, 1).ndNode;
        }
    }
    Solid.ndMaxXYMove = sqrt(Solid.ndMaxXYMove);
    if (dxmax <= Simulation.ndMaxGridMove || !bGridMove || iRelax == 0) // small displacement => no update
    {
        iGridMove = TGrid::no_update;
        if (Solid.ndMaxXYMove < Simulation.ndMaxGridMove && (Laser.ndPosition - LaserOld.ndPosition).Norm() < sqr(Simulation.ndMaxGridMove))
            iGridMove = TGrid::small_change;
        return;
    }

    SurfOld.Reallocate(iVolNodes, jVolNodes);
    for (i = 2; i < iVolNodes; i++) // store initial values
        for (j = 2; j < jVolNodes; j++)
            SurfOld(i, j) = Solid(i, j, 1).ndNode;

    for (count = 0; count < iRelax; count++) // loop over grid relaxation
    {
        i = random(4); // starting randomly at different corners
        if (i & 1)
        {
            iinit = 2;
            iend = iVolNodes;
            istep = 1;
        }
        else
        {
            iinit = iVolNodes - 1;
            iend = 1;
            istep = -1;
        }
        if (i & 2)
        {
            jinit = 2;
            jend = jVolNodes;
            jstep = 1;
        }
        else
        {
            jinit = jVolNodes - 1;
            jend = 1;
            jstep = -1;
        }

        for (i = iinit; i != iend; i += istep) // loop over surface nodes
        {
            for (j = jinit; j != jend; j += jstep)
            {
                for (n = 0; n < 5; n++)
                {
                    SurfVect(n) = 0; // constants of local surface equation
                    MatA(n, 0) = 1;
                }
                SurfVect(6) = 1;

                if (Solid(i, j, 1).ndNode.z == 0) // normalizing constant
                    z0 = 1;
                else
                    z0 = SurfOld(i, j).z;
                // neighbor nodes
                pts[0] = Solid(i - 1, j, 1).ndNode - SurfOld(i, j);
                pts[0].z /= z0; // relative to node
                pts[1] = Solid(i + 1, j, 1).ndNode - SurfOld(i, j);
                pts[1].z /= z0;
                pts[2] = Solid(i, j - 1, 1).ndNode - SurfOld(i, j);
                pts[2].z /= z0;
                pts[3] = Solid(i, j + 1, 1).ndNode - SurfOld(i, j);
                pts[3].z /= z0;
                pts[4].Set(0, 0, 0); // node to adjust in origin

                for (n = 5; n > 0; n--) // set up matrices for local surface
                { // function calculation
                    SurfVect(n) = 1;
                    for (m = 0; m < 5; m++)
                        MatA(m, n) = SurfaceFunction(pts[m].x, pts[m].y, SurfVect);
                    SurfVect(n) = 0;
                }

                for (m = 0; m < 5; m++)
                    MatB(m, 0) = pts[m].z;

                if (matop.GaussElim(MatA, MatB, 5, 6) != // calculate local surface fct
                    MatrixOp::matErr_None)
                {
                    WarningFunction("local surface function undefined at (%i,%i)", i, j);
                    continue; // unsolvable => skip update;
                }

                for (n = 0; n < 6; n++)
                    SurfVect(n) = MATVECT_BAD_RESULT;

                for (nRow = 4; nRow >= 0; nRow--) // back substitution to determine
                { // surface function constants
                    for (n = 0; n < 6; n++) // first non-zero element
                        if (MatA(nRow, n) != 0)
                            break;

                    if (n == 6 && MatB(nRow, 0) != 0)
                    {
                        ErrorFunction("local surface function undefined at (%i,%i)", i, j);
                        return; // unsolvable => skip update;
                    }
                    if (n < 6)
                    {
                        p = MatB(nRow, 0);
                        for (m = n + 1; m < 6; m++)
                        {
                            if (SurfVect(m) == MATVECT_BAD_RESULT)
                                SurfVect(m) = 0;
                            else
                                p -= SurfVect(m) * MatA(nRow, m);
                        }
                        SurfVect(n) = p / MatA(nRow, n);
                    }
                }

                for (m = 0; m < 5; m++)
                    pts[m].z *= z0;
                SurfVect(6) = z0; // surface constant vector found

                ShiftSurNode(pts, SurfVect, i != iSymmetry,
                             j != jSymmetry); // shift node

                SurfOld(i, j) += pts[4]; // new position
                // to do: use deltat of next time step!
                SolSurf(i, j).xtau = (SurfOld(i, j) - Solid(i, j, 1).ndNode) / Simulation.ndDeltat; // nodal velocity
            }
        }
    }
    iGridMove = TGrid::updated;
}

// ***************************************************************************
// "interaction potential" for node pt with neighbor ptn in volume
// pot = 1/r+D*(r-r0)^2 mit r = abs(pt-ptn)/distn
// ***************************************************************************

prec TGrid::GetVolPotential(TPoint3D pt, TPoint3D ptn, prec distn)
{
    prec r = (pt - ptn).Abs() / distn;
    return 1 / r + Dconst * sqr(r - MinRadius);
    //  p = sqr(1-exp(-0.8*(r-1.0)));
    //  p = 1/r+ 0.5*fabs(r-1.0);
}

// ***************************************************************************
// total "interaction potential" for node pt with 6 neighbor nodes in volume
// ***************************************************************************

prec TGrid::GetTotalVolPotential(TPoint3D pt, TPoint3D *pts, prec *dist)
{
    prec pot = 0;
    int i;

    for (i = 0; i < 6; i++)
        pot += GetVolPotential(pt, pts[i], dist[i]);
    return pot;
}

// ***************************************************************************
// "interaction potential" for node pt with neighbor ptn on surface
// pot = 1/r+D*(r-r0)^2 mit r = abs(pt-ptn)/distn
// ***************************************************************************

prec TGrid::GetSurPotential(TPoint3D pt, TPoint3D ptn, rvector &SurfVect)
{
    prec r = Distance(ptn, pt, SurfVect) / xDistance;
    return 1 / r + Dconst * sqr(r - MinRadius);
    //  p = sqr(1-exp(-0.8*(r-1.0)));
    //  p = 1/r+ 0.5*fabs(r-1.0);
}

// ***************************************************************************
// total "interaction potential" for node pt with 4 neighbor nodes on surface
// ***************************************************************************

prec TGrid::GetTotalSurPotential(TPoint3D pt, TPoint3D *pts,
                                 rvector &SurfVect)
{
    prec pot = 0;
    int i;

    for (i = 0; i < 4; i++)
        pot += GetSurPotential(pt, pts[i], SurfVect);
    return pot;
}

// ***************************************************************************
// surface polynomial
// (sv0+sv1*x+sv2*y+sv3*x^2+sv4*y^2+sv5*x*y)*sv6
// ***************************************************************************

prec TGrid::SurfaceFunction(prec x, prec y, rvector &SurfVect)
{
    return (SurfVect(0) + SurfVect(1) * x + SurfVect(2) * y + SurfVect(3) * x * x + SurfVect(4) * y * y + SurfVect(5) * x * y) * SurfVect(6);
}

// ***************************************************************************
// x gradient of surface polynomial
// (sv1+2*sv3*x+sv5*y)*sv6
// ***************************************************************************

prec TGrid::SurfaceGradientX(prec x, prec y, rvector &SurfVect)
{
    return (SurfVect(1) + 2.0 * SurfVect(3) * x + SurfVect(5) * y) * SurfVect(6);
}

// ***************************************************************************
// y gradient of surface polynomial
// (sv2+2*sv4*y+sv5*x)*sv6
// ***************************************************************************

prec TGrid::SurfaceGradientY(prec x, prec y, rvector &SurfVect)
{
    return (SurfVect(2) + 2.0 * SurfVect(4) * y + SurfVect(5) * x) * SurfVect(6);
}

// ***************************************************************************
// xx gradient of surface polynomial
// 2*sv3*sv6
// ***************************************************************************

prec TGrid::SurfaceGradientXX(prec x, prec y, rvector &SurfVect)
{
    return 2.0 * SurfVect(3) * SurfVect(6);
}

// ***************************************************************************
// yy gradient of surface polynomial
// 2*sv4*sv6
// ***************************************************************************

prec TGrid::SurfaceGradientYY(prec x, prec y, rvector &SurfVect)
{
    return 2.0 * SurfVect(4) * SurfVect(6);
}

// ***************************************************************************
// distance on surface between points ptx0 and ptx1
// ***************************************************************************

prec TGrid::Distance(TPoint3D ptx0, TPoint3D ptx1, rvector &SurfVect)
{
    prec gam1, gam2, gam3, ax, ay, bx, by, dx, dy, p, a, b, r, ra, q;

    dx = ptx1.x - ptx0.x;
    dy = ptx1.y - ptx0.y;
    ax = SurfaceGradientX(ptx0.x, ptx0.y, SurfVect);
    bx = SurfaceGradientX(ptx1.x, ptx1.y, SurfVect) - ax;
    ay = SurfaceGradientY(ptx0.x, ptx0.y, SurfVect);
    by = SurfaceGradientY(ptx1.x, ptx1.y, SurfVect) - ay;
    gam1 = sqr(bx * dx + by * dy);
    gam2 = (bx * dx + by * dy) * (ax * dx + ay * dy);
    gam3 = sqr(dx) + sqr(dy) + sqr(ax * dx + ay * dy);
    a = gam1 / gam3;
    b = gam2 / gam3;
    ra = sqrt(a);

    if (a < 1e-10)
    {
        if (fabs(b) < 1e-10)
            p = 1.0;
        else
            p = (sqrt(2 * b + 1) * (2 * b + 1) - 1) / 3.0 / b;
    }
    else
    {
        r = b / a;
        q = ra * r + 1.0;
        p = (1 + r) * sqrt(a + 2 * b + 1) - r;
        if (q != 0)
            p += log((ra * (1 + r) + sqrt(a + 2 * b + 1)) / q) * (1 - r * r * a) / ra;
        p /= 2.0;
    }

    return p * sqrt(gam3);
    ;
}

// ***************************************************************************
// newton iteration for surface movement
// ***************************************************************************

void TGrid::ShiftSurNode(TPoint3D *pts, rvector &SurfVect, bool bx, bool by)
{
    TPoint3D ptold, ptnew;
    prec pot[3], dpx, ddpx, dpy, ddpy, pnew, relax, xmin, xmax, ymin, ymax,
        dx, dy, dxy, deltax, deltay, dxx, dyy;
    int i;

    ptold = pts[4];
    ptnew = ptold;
    pot[1] = GetTotalSurPotential(ptold, pts, SurfVect);
    xmin = pts[3].x;
    xmax = xmin;
    ymin = pts[3].y;
    ymax = ymin;
    deltax = xDistance / 1000.0;
    deltay = yDistance / 1000.0;
    for (i = 0; i < 3; i++)
    {
        xmin = min(xmin, pts[i].x);
        xmax = max(xmax, pts[i].x);
        ymin = min(ymin, pts[i].y);
        ymax = max(ymax, pts[i].y);
    }
    xmin += deltax;
    xmax -= deltax;
    ymin += deltay;
    ymax -= deltay;
    do
    {
        relax = 1;
        if (bx == true)
        {
            ptold.x -= deltax;
            pot[0] = GetTotalSurPotential(ptold, pts, SurfVect);
            ptold.x += deltax + deltax;
            pot[2] = GetTotalSurPotential(ptold, pts, SurfVect);
            ptold.x -= deltax;
            dpx = (pot[2] - pot[0]) / (deltax + deltax);
            ddpx = (pot[2] + pot[0] - 2.0 * pot[1]) / sqr(deltax);
            if (ddpx == 0)
            {
                if (dpx > 0)
                    dx = -10.0 * deltax;
                else if (dpx < 0)
                    dx = 10.0 * deltax;
                else
                    dx = 0;
            }
            else
                dx = -dpx / ddpx;
        }
        else
            dx = 0;
        if (by == true)
        {
            ptold.y -= deltay;
            pot[0] = GetTotalSurPotential(ptold, pts, SurfVect);
            ptold.y += deltay + deltay;
            pot[2] = GetTotalSurPotential(ptold, pts, SurfVect);
            ptold.y -= deltay;
            dpy = (pot[2] - pot[0]) / (deltay + deltay);
            ddpy = (pot[2] + pot[0] - 2.0 * pot[1]) / sqr(deltay);
            if (ddpy == 0)
            {
                if (dpy > 0)
                    dy = -10.0 * deltay;
                else if (dpy < 0)
                    dy = 10.0 * deltay;
                else
                    dy = 0;
            }
            else
                dy = -dpy / ddpy;
        }
        else
            dy = 0;
        do
        {
            if (relax < 1e-3)
            {
                dx = 0;
                dy = 0;
                break;
            }
            dxx = dx * relax;
            dyy = dy * relax;
            dxy = sqr(dxx) + sqr(dyy);
            if (dxy > sqr(Simulation.ndMaxXYMove))
                relax = sqr(Simulation.ndMaxXYMove) / dxy;
            ptnew.x = max(min(ptold.x + dxx, xmax), xmin);
            dxx = ptnew.x - ptold.x;
            ptnew.y = max(min(ptold.y + dyy, ymax), ymin);
            dyy = ptnew.y - ptold.y;
            pnew = GetTotalSurPotential(ptnew, pts, SurfVect);
            relax /= 10.0;
        } while (pnew > pot[1]);

        if (dxx != 0)
        {
            ptold.x = ptnew.x;
            pot[1] = pnew;
        }
        if (dyy != 0)
        {
            ptold.y = ptnew.y;
            pot[1] = pnew;
        }
    } while (fabs(dxx) > deltax || fabs(dyy) > deltay);

    ptold.z = SurfaceFunction(ptold.x, ptold.y, SurfVect);
    pts[4] = ptold;
}

// ***************************************************************************
// newton iteration for nodal movement in volume
// shift node (i,j,k) with ideal distances k1 and k2 to upper and lower
// k nodes respectively
// ***************************************************************************

void TGrid::ShiftVolNode(int i, int j, int k, prec k1, prec k2)
{
    TPoint3D ptold, ptnew, ptmin, ptmax, ptdelta, pts[7], dp, ddp, dx;
    prec pot[6], pnew, relax, dmax, dist[6];
    int l;

    pts[0] = SolidOld(i - 1, j, k).ndNode; // position of neighbors (old step)
    pts[1] = SolidOld(i + 1, j, k).ndNode;
    pts[2] = SolidOld(i, j - 1, k).ndNode;
    pts[3] = SolidOld(i, j + 1, k).ndNode;
    pts[4] = SolidOld(i, j, k - 1).ndNode;
    if (k < kVolNodes - 1)
    {
        pts[5] = SolidOld(i, j, k + 1).ndNode;
        dist[5] = k2;
    }
    else
    {
        pts[5] = pts[4];
        dist[5] = k1;
    }
    pts[6] = SolidOld(i, j, k).ndNode;
    dist[0] = dist[1] = xDistance; // ideal spacing in x-y-direction
    dist[2] = dist[3] = yDistance;
    dist[4] = k1;

    ptold = pts[6];
    pot[1] = GetTotalVolPotential(ptold, pts, dist);
    ptmin = ptmax = pts[0];
    for (l = 1; l < 6; l++)
    {
        ptmin = Min(ptmin, pts[l]); // maximum displacement
        ptmax = Max(ptmax, pts[l]);
    }
    ptdelta.x = xDistance / 1000.0;
    ptdelta.y = yDistance / 1000.0;
    ptdelta.z = ndSpacing / 1000.0;
    ptmin += ptdelta;
    ptmax -= ptdelta;
    do
    {
        relax = 1;
        for (l = 0; l < 3; l++)
        {
            ptold[l] -= ptdelta[l];
            pot[0] = GetTotalVolPotential(ptold, pts, dist);
            ptold[l] += ptdelta[l] + ptdelta[l];
            pot[2] = GetTotalVolPotential(ptold, pts, dist);
            ptold[l] -= ptdelta[l];
            dp[l] = (pot[2] - pot[0]) / (ptdelta[l] + ptdelta[l]);
            ddp[l] = (pot[2] + pot[0] - 2.0 * pot[1]) / sqr(ptdelta[l]);
        }
        do
        {
            if (relax < 1e-3) // underrelaxation
            {
                dx = 0;
                break;
            }
            for (l = 0; l < 3; l++)
            {
                if (ddp[l] == 0)
                {
                    if (dp[l] > 0)
                        dx[l] = -10.0 * ptdelta[l];
                    else if (dp[l] < 0)
                        dx[l] = 10.0 * ptdelta[l];
                    else
                        dx[l] = 0;
                }
                else
                    dx[l] = -dp[l] / ddp[l];
            }

            dx *= relax;
            dmax = sqr(dx.z);
            if (dmax > sqr(Simulation.ndMaxZMove))
            {
                dx /= relax;
                relax = sqr(Simulation.ndMaxZMove) / dmax;
                dx *= relax;
            }
            else
            {
                dmax = sqr(dx.x) + sqr(dx.y);
                if (dmax > sqr(Simulation.ndMaxXYMove))
                {
                    dx /= relax;
                    relax = sqr(Simulation.ndMaxXYMove) / dmax;
                    dx *= relax;
                }
            }
            if (i == 1 || i == iVolNodes || i == iSymmetry)
                dx.x = 0;
            if (j == 1 || j == jVolNodes || j == jSymmetry)
                dx.y = 0;
            ptnew = Max(Min(ptold + dx, ptmax), ptmin);
            pnew = GetTotalVolPotential(ptnew, pts, dist);
            relax /= 10.0;
        } while (pnew > pot[1]);

        for (l = 0; l < 3; l++)
        {
            if (dx[l] != 0)
            {
                ptold[l] = ptnew[l];
                pot[1] = pnew;
            }
        }
    } while ((fabs(dx.x) > ptdelta.x || fabs(dx.y) > ptdelta.y || fabs(dx.z) > ptdelta.z)
             && (ptold.x < ptmax.x && ptold.x > ptmin.x)
             && (ptold.y < ptmax.y && ptold.y > ptmin.y)
             && (ptold.z < ptmax.z && ptold.z > ptmin.z));

    Solid(i, j, k).ndNode = ptold; // set new position
}
