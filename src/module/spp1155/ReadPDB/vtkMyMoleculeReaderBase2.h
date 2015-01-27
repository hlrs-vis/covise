/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMoleculeReaderBase.h,v $
  Language:  C++
  Date:      $Date: 2003/12/23 14:08:27 $
  Version:   $Revision: 1.4 $

Copyright (c) 1993-2001 Ken Martin, Will Schroeder, Bill Lorensen
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither name of Ken Martin, Will Schroeder, or Bill Lorensen nor the names
of any contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

* Modified source versions must be plainly marked as such, and must not be
misrepresented as being the original software.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=========================================================================*/
// .NAME vtkMoleculeReaderBase - read Molecular Data files
// .SECTION Description
// vtkMoleculeReaderBase is a source object that reads Molecule files
// The FileName must be specified
//
// .SECTION Thanks
// Dr. Jean M. Favre who developed and contributed this class

#ifndef __vtkMoleculeReaderBase_h
#define __vtkMoleculeReaderBase_h

#include "vtkPolyDataSource.h"

class vtkCellArray;
class vtkFloatArray;
class vtkDataArray;
class vtkIdTypeArray;
class vtkUnsignedCharArray;
class vtkPoints;
#include <vector>

static double vtkMoleculeReaderBaseAtomColors[][3] = {
    { 255, 255, 255 },
    { 127, 0, 127 },
    { 255, 0, 255 },
    { 127, 127, 127 },
    { 127, 0, 127 },
    { 0, 255, 0 },
    { 0, 0, 255 },
    { 255, 0, 0 },
    { 0, 255, 255 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 178, 153, 102 },
    { 127, 127, 127 },
    { 51, 127, 229 },
    { 0, 255, 255 },
    { 255, 255, 0 },
    { 255, 127, 127 },
    { 255, 255, 127 },
    { 127, 127, 127 },
    { 51, 204, 204 },
    { 127, 127, 127 },
    { 0, 178, 178 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 204, 0, 255 },
    { 255, 0, 255 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 229, 102, 51 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 255, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 102, 51, 204 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 51, 127, 51 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 },
    { 127, 127, 127 }
};

static double vtkMoleculeReaderBaseRadius[] = {
    1.2, 1.22, 1.75, /* "H " "He" "Li" */
    1.50, 1.90, 1.80, /* "Be" "B " "C " */
    1.70, 1.60, 1.35, /* "N " "O " "F " */
    1.60, 2.31, 1.70,
    2.05, 2.00, 2.70,
    1.85, 1.81, 1.91,
    2.31, 1.74, 1.80,
    1.60, 1.50, 1.40, /* Ti-Cu and Ge are guestimates. */
    1.40, 1.40, 1.40,
    1.60, 1.40, 1.40,
    1.90, 1.80, 2.00,
    2.00, 1.95, 1.98,
    2.44, 2.40, 2.10, /* Sr-Rh and Ba and La are guestimates. */
    2.00, 1.80, 1.80,
    1.80, 1.80, 1.80,
    1.60, 1.70, 1.60,
    1.90, 2.20, 2.20,
    2.20, 2.15, 2.20,
    2.62, 2.30, 2.30,
    2.30, 2.30, 2.30, /* All of these are guestimates. */
    2.30, 2.30, 2.40,
    2.30, 2.30, 2.30,
    2.30, 2.30, 2.30,
    2.40, 2.50, 2.30,
    2.30, 2.30, 2.30, /* All but Pt and Bi are guestimates. */
    2.30, 2.30, 2.40,
    2.30, 2.40, 2.50,
    2.50, 2.40, 2.40,
    2.40, 2.40, 2.90,
    2.60, 2.30, 2.30, /* These are all guestimates. */
    2.30, 2.30, 2.30,
    2.30, 2.30, 2.30,
    2.30, 2.30, 2.30,
    2.30, 2.30, 2.30,
    2.30, 1.50
};

static double vtkMoleculeReaderBaseCovRadius[103] = {
    0.32, 1.6, 0.68, 0.352, 0.832, 0.72,
    0.68, 0.68, 0.64, 1.12, 0.972, 1.1, 1.352, 1.2, 1.036,
    1.02, 1, 1.568, 1.328, 0.992, 1.44, 1.472, 1.328, 1.352,
    1.352, 1.34, 1.328, 1.62, 1.52, 1.448, 1.22, 1.168, 1.208,
    1.22, 1.208, 1.6, 1.472, 1.12, 1.78, 1.56, 1.48, 1.472,
    1.352, 1.4, 1.448, 1.5, 1.592, 1.688, 1.632, 1.46, 1.46,
    1.472, 1.4, 1.7, 1.672, 1.34, 1.872, 1.832, 1.82, 1.808,
    1.8, 1.8, 1.992, 1.792, 1.76, 1.752, 1.74, 1.728, 1.72,
    1.94, 1.72, 1.568, 1.432, 1.368, 1.352, 1.368, 1.32, 1.5,
    1.5, 1.7, 1.552, 1.54, 1.54, 1.68, 1.208, 1.9, 1.8,
    1.432, 1.18, 1.02, 0.888, 0.968, 0.952, 0.928, 0.92, 0.912,
    0.9, 0.888, 0.88, 0.872, 0.86, 0.848, 0.84
};

class vtkMyMoleculeReaderBase : public vtkPolyDataSource
{
    //		std::vector<double> vtkMoleculeReaderBaseCovRadius;

public:
    vtkTypeRevisionMacro(vtkMyMoleculeReaderBase, vtkPolyDataSource);
    void PrintSelf(ostream &os, vtkIndent indent);

    vtkSetStringMacro(FileName);
    vtkGetStringMacro(FileName);

    vtkSetMacro(BScale, double);
    vtkGetMacro(BScale, double);

    vtkSetMacro(HBScale, double);
    vtkGetMacro(HBScale, double);

    vtkGetMacro(NumberOfAtoms, int);

protected:
    vtkMyMoleculeReaderBase();
    ~vtkMyMoleculeReaderBase();

    char *FileName;
    double BScale;
    // a scaling factor to compute bonds between non-hydrogen atoms
    double HBScale;
    // a scaling factor to compute bonds with hydrogen atoms
    int NumberOfAtoms;

    virtual void Execute();
    int ReadMolecule(FILE *fp);
    int MakeAtomType(const char *atype);
    int MakeBonds(vtkPoints *, vtkIdTypeArray *, vtkCellArray *);

    vtkPoints *Points;
    vtkUnsignedCharArray *RGB;
    vtkFloatArray *Radii;
    vtkIdTypeArray *AtomType;

    virtual void ReadSpecificMolecule(FILE *fp) = 0;

private:
    // Not implemented.
    vtkMyMoleculeReaderBase(const vtkMyMoleculeReaderBase &);
    // Not implemented.
    void operator=(const vtkMyMoleculeReaderBase &);
};
#endif
