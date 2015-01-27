/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Statistics
  Module:    $RCSfile: vtkStatistics.h,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.

=========================================================================*/
// .NAME vtkStatistics - computes statistics
// .SECTION Description
// vtkStatistics is a filter that computes statistics from data
// defined on the nodes of an unstructured grid.

#ifndef __vtkStatistics_h
#define __vtkStatistics_h

#include "vtkUnstructuredGridAlgorithm.h"

#ifdef CSCS_PARAVIEW_INTERNAL
#define VTK_Statistics_EXPORT VTK_EXPORT
#else
#include "vtkStatisticsConfigure.h"
#endif

class vtkDataArray;

class VTK_Statistics_EXPORT vtkStatistics : public vtkUnstructuredGridAlgorithm
{
public:
    static vtkStatistics *New();
    vtkTypeRevisionMacro(vtkStatistics, vtkUnstructuredGridAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

protected:
    vtkStatistics();
    ~vtkStatistics();

    // Usual data generation method
    virtual int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    virtual int FillInputPortInformation(int port, vtkInformation *info);

private:
    vtkStatistics(const vtkStatistics &); // Not implemented.
    void operator=(const vtkStatistics &); // Not implemented.
};

#endif
