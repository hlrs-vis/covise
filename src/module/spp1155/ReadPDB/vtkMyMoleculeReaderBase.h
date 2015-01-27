/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkMyMoleculeReaderBase.h,v $
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
// .NAME vtkMyMoleculeReaderBase - read Molecular Data files
// .SECTION Description
// vtkMyMoleculeReaderBase is a source object that reads Molecule files
// The FileName must be specified
//
// .SECTION Thanks
// Dr. Jean M. Favre who developed and contributed this class

#ifndef __vtkMyMoleculeReaderBase_h
#define __vtkMyMoleculeReaderBase_h

#include "vtkPolyDataSource.h"
#include <vector>

class vtkCellArray;
class vtkFloatArray;
class vtkDataArray;
class vtkIdTypeArray;
class vtkUnsignedCharArray;
class vtkPoints;

class vtkMyMoleculeReaderBase : public vtkPolyDataSource
{
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

    virtual vtkFloatArray *GetRadii();
    //vtkGetMacro(Radii, virtual vtkFloatArray *);

    virtual vtkIdTypeArray *GetAtomType();
    virtual vtkIdType GetAtomType(int iPos);
    //vtkGetMacro(AtomType, virtual vtkIdTypeArray *);

    virtual vtkUnsignedCharArray *GetRGB();
    void SetRadiiVectorMap(std::vector<float> vRadii);

    vtkCellArray *GetGroupsLines();
    vtkIdType GetNumberOfGroupsLines();

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
    int MakeGroups(vtkPoints *, vtkIdTypeArray *, vtkCellArray *);

    vtkPoints *Points;
    vtkUnsignedCharArray *RGB;
    vtkFloatArray *Radii;
    vtkIdTypeArray *AtomType;
    vtkIdTypeArray *GroupType;
    vtkCellArray *GroupLines;

    virtual void ReadSpecificMolecule(FILE *fp) = 0;

private:
    // Not implemented.
    vtkMyMoleculeReaderBase(const vtkMyMoleculeReaderBase &);
    // Not implemented.
    void operator=(const vtkMyMoleculeReaderBase &);
    std::vector<float> m_vRadii;
};
#endif
