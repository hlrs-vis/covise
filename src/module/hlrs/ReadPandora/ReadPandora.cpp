// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2000 RUS  ++
// ++ Description: "Hello, world!" in COVISE API                          ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Andreas Werner                           ++
// ++               Computer Center University of Stuttgart               ++
// ++                            Allmandring 30                           ++
// ++                           70550 Stuttgart                           ++
// ++                                                                     ++
// ++ Date:  10.01.2000  V2.0                                             ++
// ++**********************************************************************/

// this includes our own class's headers
#include "ReadPandora.h"
#include <do/coDoPoints.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoPolygons.h>

using namespace covise;

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ReadPandora::ReadPandora(int argc, char *argv[])
    : coModule(argc, argv, "simple reader for HDF5 data")
{
    filename = addFileBrowserParam("filename", "HDF5 file name");
    meshOut = addOutputPort("mesh", "Mesh", "Polygon mesh");
    dataOut = addOutputPort("ressources", "Float", "Ressources field");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once every time the module is executed
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadPandora::compute(const char *port)
{
    (void)port;
    width=0;
    height=0;
    numSteps=0;
    numTasks=0;
    herr_t status;

    /* open file from ex_lite1.c */
    hid_t file_id = H5Fopen(filename->getValue(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file_id < 0)
    {
        sendError("unable to open %s", filename->getValue());
        sendInfo("file_id: %d", (int)file_id);
        return STOP_PIPELINE;
    }

    /* get the dimensions of the dataset */
    status = H5LTget_attribute_int(file_id, "/global","height", &height);
    status = H5LTget_attribute_int(file_id, "/global","width", &width);
    status = H5LTget_attribute_int(file_id, "/global","numSteps", &numSteps);
    status = H5LTget_attribute_int(file_id, "/global","numTasks", &numTasks);
    if (width==0 || height ==0)
        return STOP_PIPELINE;
    coDoPolygons **meshes = new coDoPolygons*[numSteps+1];
    coDoFloat **dataObjects = new coDoFloat*[numSteps+1];
    char *meshName = new char[strlen(meshOut->getObjName())+100];
    sprintf(meshName,"%s_mesh",meshOut->getObjName());
    int numPoints = width*height;
    int numQuats = (width-1)*(height-1);
    int numVertices = numQuats*4;
    meshes[0] = new coDoPolygons(meshName,numPoints,numVertices,numQuats);
    float *xc,*yc,*zc;
    int *vl, *ll;
    meshes[0]->getAddresses(&xc,&yc,&zc,&vl,&ll);
    for(int w=0;w<width;w++)
    {
        for(int h=0;h<height;h++)
        {
            int index = h*width+w;
            xc[index] = (float)w;
            yc[index] = (float)h;
            zc[index] = (float)0.0;
        }
    }
    int vert=0;
    int polygon = 0;
    for(int w=1;w<width;w++)
    {
        for(int h=1;h<height;h++)
        {
            vl[vert] =h*width+w;
            vl[vert+1] =h*width+w-1;
            vl[vert+2] =(h-1)*width+w-1;
            vl[vert+3] =(h-1)*width+w;
            ll[polygon]=vert;
            polygon++;
            vert+=4;
        }
    }
    delete[] meshName;
    for(int i=0;i<numSteps;i++)
    {
        char *dataName = new char[strlen(dataOut->getObjName())+100];
        sprintf(dataName,"%s_%d",dataOut->getObjName(),i);
        coDoFloat *fdata = new coDoFloat(dataName,width*height);
        char datasetName[200];
        sprintf(datasetName,"/resources/step%d",i);
        int *data = new int[width*height];
        H5LTread_dataset_int (file_id, datasetName, data);
        float *fd = fdata->getAddress();
        for(int n=0;n<32*32;n++)
        {
            fd[n]=(float)data[n];
        }
        dataObjects[i]=fdata;
        dataObjects[i+1]=NULL;
        if(i>0)
        {
            meshes[i] = meshes[i-1];
            meshes[i]->incRefCount();
        }
        meshes[i+1]=NULL;
        delete[] dataName;
    }
    coDoSet *meshSet = new coDoSet(meshOut->getObjName(),(coDistributedObject **)meshes);
    coDoSet *dataSet = new coDoSet(dataOut->getObjName(),(coDistributedObject **)dataObjects);
    delete[] meshes;
    delete[] dataObjects;
    meshSet->addAttribute("TIMESTEP","0 0");
    dataSet->addAttribute("TIMESTEP","0 0");

    meshOut->setCurrentObject(meshSet);
    dataOut->setCurrentObject(dataSet);

    /* close file */
    status = H5Fclose(file_id);

    return CONTINUE_PIPELINE;
}

// instantiate an object of class ReadHDF5 and register with COVISE
MODULE_MAIN(Module, ReadPandora)
