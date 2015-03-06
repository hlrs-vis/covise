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

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <api/coFeedback.h>
#include <covise/covise.h>
#include "ReadVTU.h"


int main(int argc, char *argv[])
{
   ReadVTU *application = new ReadVTU(argc, argv);
   application->start(argc,argv);
}


ReadVTU::ReadVTU(int argc, char *argv[])            // vvvv --- this info appears in the module setup window
:coModule(argc, argv, "Reads (a set) of VTK-files")
{
	GridOutPort = addOutputPort("Grid_Set","UnstructuredGrid","Set of unstructured grids");
	//VectorOutPort = addOutputPort("VectorData_Set", "Vec3", "Set of Vector Data at vertex");
	ScalarOutPort = addOutputPort("ScalarData_Set","Float","Set of Scalar Data at vertex");
	fileBrowser = addFileBrowserParam("first_VTK_file", "select file with first timestep");
	fileBrowser->setValue("~/", "*.vtu/*");

	numberOfFiles = addInt32Param("number_of_vtu_files", "index of last file");
	numberOfFiles->setValue(1);

}

int ReadVTU::compute(const char *) {

  const char* fileName;
  fileName = fileBrowser->getValue();

  char str[511];
 
  float* pointData = NULL;




  FILE *fp;

  if((fp = fopen(fileName, "r")) == NULL) {
    sendError("Problems while loading file %s", fileName);
    return STOP_PIPELINE;
  }
  char temp[255];
  int littleEndian = 0;
  int numberOfPoints = 0;
  int numberOfCells = 0;

  int counter = 0;

  while(!feof(fp) && counter < 5) {

    fscanf(fp, "%s", str);

    //sendInfo(str);

    if(strstr(str, "byte_order=\"LittleEndian\"")) {
      littleEndian = 1;
      sendInfo("Little Endian");
    }


    if(strstr(str, "byte_order=\"BigEndian\"")) {
      littleEndian = 0;
      sendInfo("Big Endian");
    }

    if(strstr(str, "NumberOfPoints")) {
      
      numberOfPoints = atoi(strncpy(temp, &str[16], strlen(str) - 17));

      sendInfo("NumberOfPoints = %i", numberOfPoints); 
    }

    if(strstr(str, "NumberOfCells")) {
      
      numberOfCells = atoi(strncpy(temp, &str[15], strlen(str) - 16));

      sendInfo("NumberOfCells = %i", numberOfCells); 
    }

    if(strstr(str, "<Points>")) {

      fscanf(fp, "%s", str);
      fscanf(fp, "%s", str);
      fscanf(fp, "%s", str);
      fscanf(fp, "%s", str);


      sendInfo("%s", str);
      
      fseek(fp, 2, SEEK_CUR);
      pointData = new float[3 * numberOfPoints];

      fread(pointData, sizeof(float), 3 * numberOfPoints, fp);

      for(int i = 0; i < 3 /*numberOfPoints*/; i++) {
        sendInfo("%f %f %f", pointData[3*i], pointData[3*i+1], pointData[3*i+2]);
      }

      fscanf(fp, "%s", str);

      sendInfo("%s", str);
      

    }





  }



  fclose(fp);

  return CONTINUE_PIPELINE;
}

void ReadVTU::param(const char *)
{


}



ReadVTU::~ReadVTU()
{
}

