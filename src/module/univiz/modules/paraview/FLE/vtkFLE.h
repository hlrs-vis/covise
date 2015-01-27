/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Finite Lyapunov Exponents
  Module:    $RCSfile: vtkFLE.h,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
// .NAME vtkFLE - computes finite Lyapunov exponent variants
// .SECTION Description
// vtkFLE is a filter that computes variants of finite Lyapunov exponents from
// velocity data defined on the nodes of an unstructured grid.

#ifndef __vtkFLE_h
#define __vtkFLE_h

//#include "vtkUnstructuredGridAlgorithm.h"
#include "vtkMultiBlockDataSetAlgorithm.h"

#ifdef CSCS_PARAVIEW_INTERNAL
#define VTK_FLE_EXPORT VTK_EXPORT
#else
#include "vtkFLEConfigure.h"
#endif

//class vtkPVFileEntry;
class vtkDataArray;
class vtkMultiBlockDataSet;

//class VTK_FLE_EXPORT vtkFLE : public vtkUnstructuredGridAlgorithm
class VTK_FLE_EXPORT vtkFLE : public vtkMultiBlockDataSetAlgorithm
{
public:
    static vtkFLE *New();
    //vtkTypeRevisionMacro(vtkFLE,vtkUnstructuredGridAlgorithm);
    vtkTypeRevisionMacro(vtkFLE, vtkMultiBlockDataSetAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

    vtkSetVector3Macro(Origin, float);
    vtkGetVectorMacro(Origin, float, 3);

    vtkSetVector3Macro(Cells, int);
    vtkGetVectorMacro(Cells, int, 3);

    vtkSetClampMacro(CellSize, float, 0.0, VTK_FLOAT_MAX);
    vtkGetMacro(CellSize, float);

    vtkSetMacro(Unsteady, int);
    vtkGetMacro(Unsteady, int);

    // ### TODO: use a file browser widget
    vtkSetStringMacro(VelocityFile);
    vtkGetStringMacro(VelocityFile);

    vtkSetClampMacro(StartTime, float, -VTK_FLOAT_MAX, VTK_FLOAT_MAX);
    vtkGetMacro(StartTime, float);

    vtkSetMacro(Mode, int);
    vtkGetMacro(Mode, int);

    vtkSetMacro(Ln, int);
    vtkGetMacro(Ln, int);

    vtkSetMacro(DivT, int);
    vtkGetMacro(DivT, int);

    vtkSetClampMacro(IntegrationTime, float, 0.0, VTK_FLOAT_MAX);
    vtkGetMacro(IntegrationTime, float);

    vtkSetClampMacro(IntegrationLength, float, 0.0, VTK_FLOAT_MAX);
    vtkGetMacro(IntegrationLength, float);

    vtkSetClampMacro(TimeIntervals, int, 1, VTK_INT_MAX);
    vtkGetMacro(TimeIntervals, int);

    vtkSetClampMacro(SepFactorMin, float, 1.0, VTK_FLOAT_MAX);
    vtkGetMacro(SepFactorMin, float);

    vtkSetClampMacro(IntegStepsMax, int, 1, VTK_INT_MAX);
    vtkGetMacro(IntegStepsMax, int);

    vtkSetMacro(Forward, int);
    vtkGetMacro(Forward, int);

    vtkSetClampMacro(SmoothingRange, int, 1, VTK_INT_MAX);
    vtkGetMacro(SmoothingRange, int);

    vtkSetMacro(OmitBoundaryCells, int);
    vtkGetMacro(OmitBoundaryCells, int);

    vtkSetMacro(GradNeighDisabled, int);
    vtkGetMacro(GradNeighDisabled, int);

    vtkSetMacro(Execute, int);
    vtkGetMacro(Execute, int);

    vtkSetMacro(ResampleOnTrajectories, int);
    vtkGetMacro(ResampleOnTrajectories, int);

    // Description:
    // Get the file entry.
    //vtkGetObjectMacro(FileEntry, vtkPVFileEntry);

protected:
    vtkFLE();
    ~vtkFLE();

    // Usual data generation method
    virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    virtual int FillInputPortInformation(int port, vtkInformation *info);

    float Origin[3];
    int Cells[3];
    float CellSize;
    int Unsteady;
    char *VelocityFile;
    float StartTime;
    int Mode;
    int Ln;
    int DivT;
    float IntegrationTime;
    float IntegrationLength;
    int TimeIntervals;
    float SepFactorMin;
    int IntegStepsMax;
    int Forward;
    int SmoothingRange;
    int OmitBoundaryCells;
    int GradNeighDisabled;
    int Execute;
    int ResampleOnTrajectories;

    //vtkPVFileEntry* FileEntry;

private:
    vtkFLE(const vtkFLE &); // Not implemented.
    void operator=(const vtkFLE &); // Not implemented.
};

#endif
