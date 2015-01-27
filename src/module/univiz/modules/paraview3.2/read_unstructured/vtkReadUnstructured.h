/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Read unstructured grid in Unstructured format
  Module:    $RCSfile: vtkReadUnstructured.h,v $

  Copyright (c) Filip Sadlo, CGL - ETH Zurich
  All rights reserved.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkReadUnstructured - reads a dataset in "Unstructured" format
// .SECTION Description
// vtkReadUnstructured creates an unstructured grid dataset. The class can
// not automatically detect the endian-ness of the binary files! (data is in
// native format)

#ifndef __vtkReadUnstructured_h
#define __vtkReadUnstructured_h

#include "vtkUnstructuredGridAlgorithm.h"

#include "unstructured.h"

class vtkIntArray;
class vtkFloatArray;
class vtkIdTypeArray;
class vtkDataArraySelection;
class vtkCallbackCommand;

class VTK_IO_EXPORT vtkReadUnstructured : public vtkUnstructuredGridAlgorithm
{
public:
    static vtkReadUnstructured *New();
    vtkTypeRevisionMacro(vtkReadUnstructured, vtkUnstructuredGridAlgorithm);
    void PrintSelf(ostream &os, vtkIndent indent);

    // Description:
    // Specify file name of Unstructured datafile to read
    vtkSetStringMacro(FileName);
    vtkGetStringMacro(FileName);

    // Description:
    // Get the total number of cells.
    vtkGetMacro(NumberOfCells, int);

    // Description:
    // Get the total number of nodes.
    vtkGetMacro(NumberOfNodes, int);

    // Description:
    // Get the number of data fields at the nodes.
    vtkGetMacro(NumberOfNodeFields, int);

    // Description:
    // Get the number of data fields for the model. Unused because VTK
    // has no methods for it.
    vtkGetMacro(NumberOfFields, int);

    // Description:
    // Get the number of data components at the nodes cells.
    vtkGetMacro(NumberOfNodeComponents, int);

    // Description:
    // The following methods allow selective reading of solutions fields.  by
    // default, ALL data fields at the nodes are read, but this can
    // be modified.
    int GetNumberOfPointArrays();
    int GetNumberOfCellArrays();
    const char *GetPointArrayName(int index);
    int GetPointArrayStatus(const char *name);
    void SetPointArrayStatus(const char *name, int status);

    void DisableAllPointArrays();
    void EnableAllPointArrays();

    // get min and max value for the index-th value of a node component
    // index varies from 0 to (veclen - 1)
    void GetNodeDataRange(int nodeComp, int index, float *min, float *max);

protected:
    vtkReadUnstructured();
    ~vtkReadUnstructured();
    int RequestInformation(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
    int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);

    // Callback registered with the SelectionObserver.
    static void SelectionModifiedCallback(vtkObject *caller, unsigned long eid,
                                          void *clientdata, void *calldata);
    void SelectionModified();

    char *FileName;

    int NumberOfNodes;
    int NumberOfCells;
    int NumberOfNodeFields;
    int NumberOfNodeComponents;
    int NumberOfFields;
    int NlistNodes;

    vtkDataArraySelection *PointDataArraySelection;

    // The observer to modify this object when the array selections are
    // modified.
    vtkCallbackCommand *SelectionObserver;

    // Whether the SelectionModified callback should invoke Modified.
    // This is used when we are copying to/from the internal reader.
    int SelectionModifiedDoNotCallModified;

    //BTX

    struct DataInfo
    {
        long foffset; // offset in binary file
        int veclen; // number of components in the node variable
        float min[3]; // pre-calculated data minima (max size 3 for vectors)
        float max[3]; // pre-calculated data maxima (max size 3 for vectors)
    };
    //ETX

    DataInfo *NodeDataInfo;

private:
    void Convert(Unstructured *unst_all, vtkUnstructuredGrid *output);

    vtkReadUnstructured(const vtkReadUnstructured &); // Not implemented.
    void operator=(const vtkReadUnstructured &); // Not implemented.
};

#endif
