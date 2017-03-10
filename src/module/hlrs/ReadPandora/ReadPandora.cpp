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
#include <do/coDoUniformGrid.h>

using namespace covise;

// define tokens for ports
enum
{
	FILE_BROWSER,
	GEOPORT3D,
	DPORT1_3D,
	DPORT2_3D,
	DPORT3_3D,
	DPORT4_3D,
	DPORT5_3D,
	GEOPORT2D,
	DPORT1_2D,
	DPORT2_2D,
	DPORT3_2D,
	DPORT4_2D,
	DPORT5_2D,
};
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ReadPandora::ReadPandora(int argc, char *argv[])
	: coReader(argc, argv, string("simple reader for Pandora HDF5 data"))
{
	

}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once every time the module is executed
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
herr_t
file_info(hid_t loc_id, const char *name, const H5L_info_t *linfo, void *opdata)
{
	hid_t group;
	auto group_names = reinterpret_cast< std::vector<std::string>* >(opdata);
	group_names->push_back(name);
	group = H5Gopen2(loc_id, name, H5P_DEFAULT);
	//step0
	//	H5LTread_dataset_int(file_id, datasetName, data);
	H5Gclose(group);
	cout << "Name : " << name << endl;
	return 0;
}

int ReadPandora::compute(const char *port)
{
    (void)port;
    width=0;
    height=0;
    numSteps=0;
    numTasks=0;
    herr_t status;

    /* open file from ex_lite1.c */
    hid_t file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if(file_id < 0)
    {
        sendError("unable to open %s", fileName.c_str());
        sendInfo("file_id: %d", (int)file_id);
        return STOP_PIPELINE;
    }
	std::vector<std::string> groupNames;
	herr_t idx = H5Literate(file_id, H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, &groupNames);

    /* get the dimensions of the dataset */
    status = H5LTget_attribute_int(file_id, "/global","height", &height);
    status = H5LTget_attribute_int(file_id, "/global","width", &width);
    status = H5LTget_attribute_int(file_id, "/global","numSteps", &numSteps);
    status = H5LTget_attribute_int(file_id, "/global","numTasks", &numTasks);
    if (width==0 || height ==0)
        return STOP_PIPELINE;
    std::vector<const coDistributedObject *> meshes(numSteps+1);
    coDoFloat **dataObjects = new coDoFloat*[numSteps+1];
	string objNameBase2d = READER_CONTROL->getAssocObjName(GEOPORT2D);
    int numPoints = width*height;
    int numQuats = (width-1)*(height-1);
    int numVertices = numQuats*4;
    meshes[0] = new coDoUniformGrid(objNameBase2d+"_0", width, height, 1, 0., width-1, 0, height-1, 0., 0.);
	for (int i = 0; i < numSteps; i++)
	{
		if (i > 0)
		{
			meshes[i] = meshes[i - 1];
			meshes[i]->incRefCount();
		}
		meshes[i + 1] = NULL;
	}

	coDoSet *meshSet = new coDoSet(objNameBase2d.c_str(), &meshes[0]);
	meshSet->addAttribute("TIMESTEP", "0 0");
	READER_CONTROL->setAssocPortObj(GEOPORT2D, meshSet);

	for (int dNum = DPORT1_2D; dNum <= DPORT3_2D; dNum++)
	{
		int dataChoice = READER_CONTROL->getPortChoice(dNum);
		if (dataChoice > 0)
		{
			std::string objNameBase = READER_CONTROL->getAssocObjName(dNum);

			dataObjects[0] = NULL;
			for (int i = 0; i < numSteps; i++)
			{
				char ch[64];
				sprintf(ch, "%d", i);
				string num(ch);
				string objName(objNameBase);
				objName = objNameBase + "_" + num;
				coDoFloat *fdata = new coDoFloat(objName.c_str(), width*height);
				char datasetName[200];
				sprintf(datasetName, "/%s/step%d", scalChoices[dataChoice].c_str(), i);
				if (true)
				{
					float *fd = fdata->getAddress();
					H5LTread_dataset_float(file_id, datasetName, fd);
				}
				else
				{
                    std::vector<int> data(width*height);
					H5LTread_dataset_int(file_id, datasetName, &data[0]);
					float *fd = fdata->getAddress();
					for (int n = 0; n < width * height; n++)
					{
						fd[n] = (float)data[n];
					}
				}

				dataObjects[i] = fdata;
				dataObjects[i + 1] = NULL;
			}
			coDoSet *dataSet = new coDoSet(objNameBase.c_str(), (coDistributedObject **)dataObjects);
			dataSet->addAttribute("TIMESTEP", "0 0");
			READER_CONTROL->setAssocPortObj(dNum, dataSet);
		}
	}

	delete[] dataObjects;

    /* close file */
    status = H5Fclose(file_id);

    return CONTINUE_PIPELINE;
}

void
ReadPandora::param(const char *paramName, bool inMapLoading)
{
	
	FileItem *fii = READER_CONTROL->getFileItem(FILE_BROWSER);

	string fileBrowserName;
	if (fii)
	{
		fileBrowserName = fii->getName();
	}
	// cerr << "ReadEnsight::param(..)  case browser name <" << caseBrowserName << ">" << endl;

	/////////////////  CALLED BY FILE BROWSER  //////////////////
	if (fileBrowserName == string(paramName))
	{
		FileItem *fi(READER_CONTROL->getFileItem(string(paramName)));
		if (fi)
		{
			coFileBrowserParam *bP = fi->getBrowserPtr();

			if (bP)
			{
				fileName = bP->getValue();
				if (fileName.empty())
				{
					cerr << "ReadPandora::param(..) no hdf5 file found " << endl;
				}
				else
				{
					// cerr << "ReadPandora::param(..) filename " << fileNm << endl;

					hid_t file_id = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
					if (file_id < 0)
					{
						sendError("unable to open %s", fileName.c_str());
						sendInfo("file_id: %d", (int)file_id);
						return;
					}
					std::vector<std::string> groupNames;
					herr_t idx = H5Literate(file_id, H5_INDEX_NAME, H5_ITER_INC, NULL, file_info, &groupNames);


					// fill in NONE to READ no data
					string noneStr("NONE");
					scalChoices.push_back(noneStr);
					vectChoices.push_back(noneStr);

					// fill in all species for the appropriate Ports/Choices
					for (std::vector<std::string>::iterator it = groupNames.begin(); it != groupNames.end(); ++it)
					{
						if((*it) != "global")
						{ 
							scalChoices.push_back((*it));
						}
					}
					if (inMapLoading)
						return;
					READER_CONTROL->updatePortChoice(DPORT1_3D, scalChoices);
					READER_CONTROL->updatePortChoice(DPORT2_3D, scalChoices);
					READER_CONTROL->updatePortChoice(DPORT3_3D, scalChoices);
					READER_CONTROL->updatePortChoice(DPORT4_3D, vectChoices);
					READER_CONTROL->updatePortChoice(DPORT5_3D, vectChoices);
					READER_CONTROL->updatePortChoice(DPORT1_2D, scalChoices);
					READER_CONTROL->updatePortChoice(DPORT2_2D, scalChoices);
					READER_CONTROL->updatePortChoice(DPORT3_2D, scalChoices);
					READER_CONTROL->updatePortChoice(DPORT4_2D, vectChoices);
					READER_CONTROL->updatePortChoice(DPORT5_2D, vectChoices);

					/* close file */
					H5Fclose(file_id);
				}
				return;
			}

			else
			{
				cerr << "ReadPandora::param(..) BrowserPointer NULL " << endl;
			}
		}
	}
}


// instantiate an object of class ReadHDF5 and register with COVISE

int main(int argc, char *argv[])
{
	// define outline of reader
    READER_CONTROL->addFile(FILE_BROWSER, "filename", "HDF5 file name", ".", "*.h5;*.H5/*");

	READER_CONTROL->addOutputPort(GEOPORT3D, "geoOut_3D", "UnstructuredGrid", "Geometry", false);

	READER_CONTROL->addOutputPort(DPORT1_3D, "sdata1_3D", "Float", "data1-3d");
	READER_CONTROL->addOutputPort(DPORT2_3D, "sdata2_3D", "Float", "data2-3d");
	READER_CONTROL->addOutputPort(DPORT3_3D, "sdata3_3D", "Float", "data3-3d");
	READER_CONTROL->addOutputPort(DPORT4_3D, "vdata1_3D", "Vec3", "data2-3d");
	READER_CONTROL->addOutputPort(DPORT5_3D, "vdata2_3D", "Vec3", "data2-3d");

    READER_CONTROL->addOutputPort(GEOPORT2D, "geoOut_2D", "UniformGrid", "Geometry", false);

	READER_CONTROL->addOutputPort(DPORT1_2D, "sdata1_2D", "Float", "data1-2d");
	READER_CONTROL->addOutputPort(DPORT2_2D, "sdata2_2D", "Float", "data2-2d");
	READER_CONTROL->addOutputPort(DPORT3_2D, "sdata3_2D", "Float", "data3-2d");
	READER_CONTROL->addOutputPort(DPORT4_2D, "vdata1_2D", "Vec3", "data1-2d");
	READER_CONTROL->addOutputPort(DPORT5_2D, "vdata2_2D", "Vec3", "data2-2d");

	// create the module
	coReader *application = new ReadPandora(argc, argv);

	// this call leaves with exit(), so we ...
	application->start(argc, argv);

	// ... never reach this point
	return 0;
}
