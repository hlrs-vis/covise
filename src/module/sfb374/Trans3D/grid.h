/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***************************************************************************
                          grid.h  -  grid settings
                             -------------------
    begin                : Sat Apr 1 2000
    copyright            : (C) 2000 by Andreas Ruf
    email                : ruf@ifsw.uni-stuttgart.de
 ***************************************************************************/

#ifndef __GRID_H_

#define __GRID_H_

#include <string>

#include "arrays.h"
#include "classext.h"

// ***************************************************************************
// grid settings
// ***************************************************************************

class TGrid
{
public:
    TGrid()
    {
        Reset();
    }
    TGrid(const TGrid &src);

    void Save(int); // save settings
    void Read(int, float vers = 1.0); // read settings
    void Reset(); // reset variables
    void Update(); // update variables
    TGrid GetDimensional(); // all values to dimensional form
    void MakeNonDimensional(); // all values to non-dimensional form

    int SetupGrid(); // set nodes for new grid
    int SetVolumeGrid(); // volume grid from surface nodes
    int UpdateVolumeGrid(); // move grid with laser
    void UpdateGrid(); // adjust grid after iteration
    void ResizeSurfaceArrays(); // resize surface arrays
    void ResizeVolumeArrays(); // resize volume arrays
    void ResizeArrays(); // resize solid arrays
    void InsertBegin(int, int); // insert/drop nodes at begin
    void InsertEnd(int, int); // insert/drop nodes at end

    prec GetVolPotential(TPoint3D, // get interaction potential
                         TPoint3D, prec);
    prec GetTotalVolPotential(TPoint3D, // total potential
                              TPoint3D *, prec *);
    prec GetSurPotential(TPoint3D, // interaction pot on surface
                         TPoint3D, rvector &);
    prec GetTotalSurPotential(TPoint3D, // total pot on surface
                              TPoint3D *, rvector &);
    // surface ploynomial
    prec SurfaceFunction(prec, prec, rvector &);
    // x derivative
    prec SurfaceGradientX(prec, prec, rvector &);
    // y derivative
    prec SurfaceGradientY(prec, prec, rvector &);
    // xx derivative
    prec SurfaceGradientXX(prec, prec, rvector &);
    // yy derivative
    prec SurfaceGradientYY(prec, prec, rvector &);
    // dist on surface
    prec Distance(TPoint3D, TPoint3D, rvector &);
    void ShiftSurNode(TPoint3D *, // shift surface node in pot
                      rvector &, bool, bool);
    void ShiftVolNode(int, int, int, // shift volume node in pot
                      prec, prec);

    string ProfileName; // name of used profile
    bool bChanged; // change flag

    prec ndWpLength; // length of workpiece (scale w)
    prec ndWpWidth; // width of workpiece (scale w)
    prec ndWpThickness; // thickness of workpiece (scale w)

    prec ndXFront; // volume grid in fromt of laser
    prec ndXBack; // volume grid behind laser
    prec ndYSide; // volume grid on side of laser

    int iVolNodes; // number volume nodes in x dir.
    int jVolNodes; // number volume nodes in y dir.
    int kVolNodes; // number volume nodes in z dir.
    int iIniNodes; // initial number of x-nodes
    int jIniNodes; // initial number of y-nodes
    int kIniNodes; // initial number of z-nodes
    int iSurNodes; // number surface nodes in x dir.
    int jSurNodes; // number surface nodes in y dir.
    int iVolMin; // start of volume grid (x)
    int iVolMax; // end of volume grid (x)
    int jVolMin; // start of volume grid (y)
    int jVolMax; // end of volume grid (y)

    prec ndSpacing; // spacing between first z-nodes
    prec ndDepth; // non-dim depth

    int iLeftBound; // left boundary condition
    int iRightBound; // right boundary condition
    int iFrontBound; // front boundary condition
    int iBackBound; // back boundary condition
    enum Boundary
    {
        symmetric = 0,
        finite = 1
    };

    bool bGridMove; // flag grid movement
    prec TangentWarp; // const <1 for warping of tangent
    prec TangentDir; // tangent direction (z for >>0)
    prec TangentBound; // constant for tangent at boundary
    int iWarpPower; // power in warping function
    int iWarpAverage; // number of neighbor nodes for warping

    int iRelax; // number of relaxations
    prec Dconst; // harmonic constant in grid potential
    prec MinRadius; // minimum radius in potential
    int iSymmetry; // x-value of symmetry plane
    int jSymmetry; // y-value of symmetry plane
    prec xDistance; // ideal x nodal spacing
    prec yDistance; // ideal y nodal spacing

    int iGridMove; // flag for grid movement
    enum Gridmove
    {
        updated = 0,
        reduced_step = 1,
        no_update = 2,
        small_change = 5
    };
};

extern TGrid Grid, GridOld;

ostream &operator<<(ostream &, TGrid &);
istream &operator>>(istream &, TGrid &);
#endif
