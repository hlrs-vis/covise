#ifndef _VTKSETREADER_H
#define _VTKSETREADER_H
/**********************************************************************************\
 **                                                                    (C)2009   **
 **                                                                              **
 ** Description: VTK data reading Module                                         **
 **              reads data in vtk format                                        **
 **              either a single file or a set of files (different timesteps)    **
 **                                                                              **
 ** Name:        ReadVTU                                                    **
 ** Category:    IO_Module                                                       **
 **                                                                              **
 ** Author:      Julia Portl                                                     **
 **              Visualization and Numerical Geometry Group, IWR                 **
 **              Heidelberg University                                           **
 **                                                                              **
 ** History:     April 2009                                                      **
 **                                                                              **
 **                                                                              **
\**********************************************************************************/

#include <api/coModule.h>
#include <api/coFileBrowserParam.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

using namespace covise;

class ReadVTU: public coModule
{
	private:

		//  member functions
		virtual int compute(const char *);
		virtual void param(const char *name);

		coDoUnstructuredGrid *readUnstructuredGrid(FILE * fp, int step, char * objectName);
//	Not implemented yet are:
//		readStructuredPoints
//		readStructuredGrid
//		readRectilinearGrid
//		readPolygonalData

		coDoFloat *readScalars(FILE * fp, int step, char * objectName, int num_point);
		coDoVec3 *readVectors(FILE * fp, int step, char * objectName, int num_point);
//	Not implemented yet are:
//		readLookupTable
//		readNormals
//		readTextureCoordinates
//		readTensors
//		readFieldData

		coFileBrowserParam *fileBrowser;
		coIntScalarParam *numberOfFiles;


		//  Ports
		coOutputPort *GridOutPort;
		coOutputPort *VectorOutPort;
		coOutputPort *ScalarOutPort;

  public:

    ReadVTU(int argc, char *argv[]);
    virtual ~ReadVTU();
};
#endif
