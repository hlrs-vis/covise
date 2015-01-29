/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <util/coviseCompat.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoData.h>

#include <appl/ApplInterface.h>
#include "ReadFconfig.h"

/*************************************************************
 *************************************************************
 **                                                         **
 **                  K o n s t r u k t o r                  **
 **                                                         **
 *************************************************************
 *************************************************************/

ReadFconfig::ReadFconfig(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Simulation coupling")
{
////////// set up default parameters
    set_module_description("ReadFconfig Simulation");

    fileName = addFileBrowserParam("configFile", "configFile");
    fileName->setValue("C:/src/test/testit/Release","Fconfig.txt");
	fileName->setFilter("*.txt");

    dimX = addInt32Param("X", "DimensionInX");
    dimX->setValue(100);

    dimY = addInt32Param("Y", "DimensionInY");
    dimY->setValue(100);

    dimZ = addInt32Param("Z", "DimensionInZ");
    dimZ->setValue(100);

    // Output ports:

    mesh = addOutputPort("mesh", "UniformGrid", "Mesh Output");
    data = addOutputPort("data", "Float", "ScalarData");
}

int ReadFconfig::compute(const char *port)
{
    (void)port;

    // create mesh
    createMesh();
	
	int x_dim = dimX->getValue();
	int y_dim = dimY->getValue();
	int z_dim = dimZ->getValue();
	float *fData;
	coDoFloat *floatData = new coDoFloat(data->getNewObjectInfo(), x_dim * y_dim * z_dim);
	floatData->getAddress(&fData);
	memset(fData,0,x_dim * y_dim * z_dim*sizeof(float));
	FILE *fp = fopen(fileName->getValue(),"r");
	char buf[200];
	if(fp !=NULL)
	{
		while(!feof(fp))
		{
			fgets(buf,200,fp);
			int i,j,k;
			sscanf(buf,"%d %d %d",&i,&j,&k);
			fData[i*x_dim*y_dim + j*y_dim + k] = 1.0;
		}
		fclose(fp);
	}

    return SUCCESS;
}

/*************************************************************

 *************************************************************/

// create a Grid
void ReadFconfig::createMesh()
{
    int xDim, yDim, zDim;

    xDim = dimX->getValue();
    yDim = dimY->getValue();
    zDim = dimZ->getValue();

    coDoUniformGrid *grid = new coDoUniformGrid(mesh->getObjName(), xDim, yDim, zDim, 0, xDim - 1, 0, yDim - 1, 0, zDim - 1);

    mesh->setCurrentObject(grid);
}

void ReadFconfig::param(const char *paramname, bool inMapLoading)
{
    int connMeth;
    (void)inMapLoading;

}

MODULE_MAIN(IO, ReadFconfig)
