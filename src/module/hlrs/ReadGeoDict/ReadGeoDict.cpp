/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

 // +++++++++++++++++++++++++++++++++++++++++
 // MODULE ReadGeoDict
 //
 // This module Reads GeoDict files
 //

#include "ReadGeoDict.h"

#ifndef _WIN32
#include <inttypes.h>
#endif

#include <sys/types.h>
#include <sys/stat.h>

#include <errno.h>

#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoPoints.h>
#include <do/coDoUniformGrid.h>

// remove  path from filename
inline const char *coBasename(const char *str)
{
	const char *lastslash = strrchr(str, '/');
	const char *lastslash2 = strrchr(str, '\\');
	if (lastslash2 > lastslash)
		lastslash = lastslash2;
	if (lastslash)
		return lastslash + 1;
	else
	{
		return str;
	}
}

VarInfo::VarInfo()
{
	vector = false;
	imageNumber = -1;
	dataObjs = NULL;
	x_d = 0; y_d = 0; z_d = 0;
	read = false;
}

VarInfo::~VarInfo()
{
}

ArrayInfo::ArrayInfo()
{
	arrayNumber = -1;
	numColumns = 0;
	numRows = 0;
	columnSize = 0;
	dataBuf = NULL;
}

ArrayInfo::~ArrayInfo()
{
	for (std::vector<ArrayColumn *>::iterator it = columnInfos.begin(); it != columnInfos.end(); it++)
	{
		delete *it;
	}
	delete[] dataBuf;
}

int ArrayColumn::varSize[NUM_TYPES] = { sizeof(unsigned char),sizeof(int),sizeof(int64_t),sizeof(float),sizeof(double) };
ArrayColumn::ArrayColumn()
{
	vector = false;
	ColumnNumber = -1;
	ColumnOffset = -1;
	dataObjs = NULL;
	x_d = 0; y_d = 0; z_d = 0;
	d_d = 0; i_d = 0; c_d = 0;
	ll_d = 0;;
	read = false;

}

ArrayColumn::~ArrayColumn()
{
}

// Module set-up in Constructor
ReadGeoDict::ReadGeoDict(int argc, char *argv[])
	: coReader(argc, argv, "Read GeoDict data")
{
	d_dataFile = NULL;
}

ReadGeoDict::~ReadGeoDict()
{
}

// param callback read header again after all changes
void
ReadGeoDict::param(const char *paramName, bool inMapLoading)
{

	FileItem *fii = READER_CONTROL->getFileItem(TXT_BROWSER);

	string txtBrowserName;
	if (fii)
	{
		txtBrowserName = fii->getName();
	}

	/////////////////  CALLED BY FILE BROWSER  //////////////////
	if (txtBrowserName == string(paramName))
	{
		FileItem *fi(READER_CONTROL->getFileItem(string(paramName)));
		if (fi)
		{
			coFileBrowserParam *bP = fi->getBrowserPtr();

			if (bP)
			{
				string dataNm(bP->getValue());
				if (dataNm.empty())
				{
					cerr << "ReadGeoDict::param(..) no data file found " << endl;
				}
				else
				{
					if (d_dataFile != NULL)
						fclose(d_dataFile);
					fileName = dataNm;
					d_dataFile = fopen(dataNm.c_str(), "rb");
					fileName = dataNm;
					int result = STOP_PIPELINE;

					if (d_dataFile != NULL)
					{
						result = readHeader();
						fseek(d_dataFile, 0, SEEK_SET);
					}

					if (result == STOP_PIPELINE)
					{
						cerr << "ReadGeoDict::param(..) could not read file: " << dataNm << endl;
					}
					else
					{

						if (varInfos.size() > 0 && Nx > 0 && Ny > 0 && Nz > 0)
						{
							// lists for Choice Labels
							vector<string> dataChoices;

							// fill in NONE to READ no data
							string noneStr("NONE");
							dataChoices.push_back(noneStr);

							// fill in all species for the appropriate Ports/Choices
							for (int i = 0; i < varInfos.size(); i++)
							{
								dataChoices.push_back(varInfos[i]->name);
							}
							if (inMapLoading)
								return;
							READER_CONTROL->updatePortChoice(DPORT1_3D, dataChoices);
							READER_CONTROL->updatePortChoice(DPORT2_3D, dataChoices);
							READER_CONTROL->updatePortChoice(DPORT3_3D, dataChoices);
							READER_CONTROL->updatePortChoice(DPORT4_3D, dataChoices);
							READER_CONTROL->updatePortChoice(DPORT5_3D, dataChoices);
						}
						else
						{
							// lists for Choice Labels
							vector<string> dataChoices;

							// fill in NONE to READ no data
							string noneStr("NONE");
							dataChoices.push_back(noneStr);
							if(arrayInfos.size()>1)
							{
								ArrayInfo *array = arrayInfos[1];
								// fill in all species for the appropriate Ports/Choices
								for (int i = 2; i < array->columnInfos.size(); i++)
								{
									dataChoices.push_back(array->columnInfos[i]->name);
								}
								if (inMapLoading)
									return;
								READER_CONTROL->updatePortChoice(DPORT1_3D, dataChoices);
								READER_CONTROL->updatePortChoice(DPORT2_3D, dataChoices);
								READER_CONTROL->updatePortChoice(DPORT3_3D, dataChoices);
								READER_CONTROL->updatePortChoice(DPORT4_3D, dataChoices);
								READER_CONTROL->updatePortChoice(DPORT5_3D, dataChoices);
							}
						}
					}
				}
				return;
			}

			else
			{
				cerr << "ReadGeoDict::param(..) BrowserPointer NULL " << endl;
			}
		}
	}
}

int ReadGeoDict::nextLine()
{
	currentLine = nextLineBuf;
	char *c = currentLine;
	while (*c != '\0' && (*c != '\r' &&  *c != '\n'))
	{
		c++;
	}
	if (*c != '\0')
	{
		*c = '\0';
		c++;
		nextLineBuf = c;
	}
	else
	{
		currentLine = c;
		return false;
	}
	return 1;
}

int ReadGeoDict::readHeader()
{
	if (d_dataFile == NULL)
	{
		d_dataFile = fopen(fileName.c_str(), "rb");
	}
	rewind(d_dataFile);
	size_t headerSize = fread(buf, 1, 1024, d_dataFile);
	if (headerSize != 1024)
	{
		coModule::sendWarning("Error reading header %zu", headerSize);
		return STOP_PIPELINE;
	}
	else
	{
		nextLineBuf = buf;
	}
	nextLine();

	buf[1024] = '\0';
	for (std::vector<VarInfo *>::iterator it = varInfos.begin(); it != varInfos.end(); it++)
	{
		delete *it;
	}
	for (std::vector<ArrayInfo *>::iterator it = arrayInfos.begin(); it != arrayInfos.end(); it++)
	{
		delete *it;
	}
	varInfos.clear();
	arrayInfos.clear();

	int lastImageNumber = 0;
	int lastArrayNumber = 0;
	while (strlen(currentLine) > 0)
	{
		if (strncmp(currentLine, "HeaderLength", 12) == 0)
		{
			int hl;
			sscanf(currentLine + 12, "%d", &hl);
			int toRead = hl - 1024;
			if (hl > 5000)
			{
				coModule::sendError("Header too large %d", hl);
				return STOP_PIPELINE;
			}
			if (toRead > 0)
			{

				size_t headerSize = fread(buf + 1024, 1, toRead, d_dataFile);// read the rest of the header
				buf[hl] = '\0';
			}

		}
		if (strncmp(currentLine, "Image", 5) == 0)
		{
			int imageNumber = 0;
			sscanf(currentLine + 5, "%d", &imageNumber);
			if (imageNumber != lastImageNumber)
			{
				VarInfo *vi = new VarInfo;
				vi->imageNumber = imageNumber;
				varInfos.push_back(vi);
				lastImageNumber = imageNumber;
			}
			char *c = currentLine + 5;
			while (*c != '\0' && (*c != ':'))
			{
				c++;
			}
			if (*c != '\0')
			{
				c++;
				if (strncmp(c, "Names", 5) == 0)
				{
					varInfos[imageNumber - 1]->name = c + 6;
				}
				if (strncmp(c, "Meaning vector", 14) == 0)
				{
					varInfos[imageNumber - 1]->vector = true;
				}
			}
		}
		if (strncmp(currentLine, "Array", 5) == 0)
		{
			int arrayNumber = 0;
			sscanf(currentLine + 5, "%d", &arrayNumber);
			if (arrayNumber != lastArrayNumber)
			{
				ArrayInfo *ai = new ArrayInfo();
				ai->arrayNumber = arrayNumber;
				arrayInfos.push_back(ai);
				lastArrayNumber = arrayNumber;
			}
			char *c = currentLine + 5;
			while (*c != '\0' && (*c != ':'))
			{
				c++;
			}
			if (*c != '\0')
			{
				c++;
				if (strncmp(c, "NumberOfColumns", 15) == 0)
				{
					sscanf(c + 16, "%d", &arrayInfos[arrayNumber - 1]->numColumns);
				}
				if (strncmp(c, "NumberOfRows", 12) == 0)
				{
					sscanf(c + 13, "%d", &arrayInfos[arrayNumber - 1]->numRows);
				}
				if (strncmp(c, "ColumnNames", 11) == 0)
				{
					c += 12;
					for (int i = 0; i < arrayInfos[arrayNumber - 1]->numColumns; i++)
					{
						char *varName = c;
						while ((*c != ',') && (*c != '\n') && (*c != '\r') && (*c != '\0'))
							c++;
						*c = '\0';
						ArrayColumn *ac = new ArrayColumn();
						ac->name = varName;
						ac->ColumnNumber = i;
						if ((ac->name == "Position X") || (ac->name == "Velocity X"))
						{
							ac->vector = true;
							c++;
							while ((*c != ',') && (*c != '\n') && (*c != '\r') && (*c != '\0'))
								c++;
							*c = '\0';
							c++;
							while ((*c != ',') && (*c != '\n') && (*c != '\r') && (*c != '\0'))
								c++;
							*c = '\0';
							i += 2;
						}
						arrayInfos[arrayNumber - 1]->columnInfos.push_back(ac);
						c++;
					}
				}

				if (strncmp(c, "Types", 5) == 0)
				{
					c += 6;
					int column = 0;
					int columnOffset = 0;
					for (int i = 0; i < arrayInfos[arrayNumber - 1]->numColumns; i++)
					{
						char *varName = c;
						while ((*c != ',') && (*c != '\n') && (*c != '\r') && (*c != '\0'))
							c++;
						*c = '\0';
						ArrayColumn *ac = arrayInfos[arrayNumber - 1]->columnInfos[column];
						if (ac->vector) 
						{
							while ((*c != ',') && (*c != '\n') && (*c != '\r') && (*c != '\0'))
								c++;
							*c = '\0';
							c++;
							while ((*c != ',') && (*c != '\n') && (*c != '\r') && (*c != '\0'))
								c++;
							*c = '\0';
							c++;
							i += 2;
						}
						ac->ColumnOffset = columnOffset;
						ac->ColumnNumber = i;
						ac->variableType = ArrayColumn::T_FLOAT;
						if ((strcmp(varName, "int_8") == 0) || (strcmp(varName, "int8") == 0))
						{
							ac->variableType = ArrayColumn::T_INT_8;
						}
						else if ((strcmp(varName, "int_32") == 0) || (strcmp(varName, "int32") == 0))
						{
							ac->variableType = ArrayColumn::T_INT_32;
						}
						else if ((strcmp(varName, "int_64") == 0) || (strcmp(varName, "int64") == 0))
						{
							ac->variableType = ArrayColumn::T_INT_64;
						}
						else if (strcmp(varName, "float") == 0)
						{
							ac->variableType = ArrayColumn::T_FLOAT;
						}
						else if (strcmp(varName, "double") == 0)
						{
							ac->variableType = ArrayColumn::T_DOUBLE;
						}
						columnOffset += ArrayColumn::varSize[ac->variableType];
						if (ac->vector)
						{
							columnOffset += ArrayColumn::varSize[ac->variableType];
							columnOffset += ArrayColumn::varSize[ac->variableType];
						}
						c++;
						column++;
					}
					arrayInfos[arrayNumber - 1]->columnSize = columnOffset;
				}
			}
		}
		if (strncmp(currentLine, "Nx", 2) == 0)
		{
			sscanf(currentLine + 3, "%d", &Nx);
		}
		if (strncmp(currentLine, "Ny", 2) == 0)
		{
			sscanf(currentLine + 3, "%d", &Ny);
		}
		if (strncmp(currentLine, "Nz", 2) == 0)
		{
			sscanf(currentLine + 3, "%d", &Nz);
		}
		if (strncmp(currentLine, "VoxelLength", 11) == 0)
		{
			sscanf(currentLine + 12, "%f,%f,%f", &sx, &sy, &sz);
		}

		nextLine();
	}
	return CONTINUE_PIPELINE;
}

// taken from old ReadGeoDict module: 2-Pass reading
int ReadGeoDict::readData()
{
	if (d_dataFile != NULL)
		fclose(d_dataFile);
	d_dataFile = fopen(fileName.c_str(), "rb");
	int res = readHeader();
	int varNumber = 0;
	if (res != STOP_PIPELINE)
	{
		std::string objNameBase = READER_CONTROL->getAssocObjName(MESHPORT3D);
		if (varInfos.size() > 0 && Nx > 0 && Ny > 0 && Nz > 0)
		{
			coDoUniformGrid *grid = new coDoUniformGrid(objNameBase.c_str(), Nx, Ny, Nz, 0, Nx*sx * 1000000, 0, Ny*sy * 1000000, 0, Nz*sz * 1000000);
			int numValues = Nx*Ny*Nz;
			while (varNumber < varInfos.size())
			{

				int portID = 0;
				for (int n = 0; n < 5; n++)
				{
					int pos = READER_CONTROL->getPortChoice(DPORT1_3D + n);
					//printf("%d %d\n",pos,n);
					if (pos > 0)
					{
						if (varInfos[pos - 1]->imageNumber == varNumber + 1 && varInfos[pos - 1]->read == false) // this port needs want to have this data
						{
							portID = DPORT1_3D + n;
							if (varInfos[pos - 1]->vector)
							{
								coDoVec3 *dataObj = new coDoVec3(READER_CONTROL->getAssocObjName(DPORT1_3D + n).c_str(), numValues);
								dataObj->getAddresses(&varInfos[pos - 1]->x_d, &varInfos[pos - 1]->y_d, &varInfos[pos - 1]->z_d);
								float *dataBuf = new float[numValues * 3];
								fread(dataBuf, sizeof(float), numValues * 3, d_dataFile);
								for (int u = 0; u < Nx; u++)
									for (int v = 0; v < Ny; v++)
										for (int w = 0; w < Nz; w++)
										{
											varInfos[pos - 1]->x_d[u*Ny*Nz + v*Nz + w] = dataBuf[w*Ny*Nx * 3 + v*Nx * 3 + u * 3];
											varInfos[pos - 1]->y_d[u*Ny*Nz + v*Nz + w] = dataBuf[w*Ny*Nx * 3 + v*Nx * 3 + u * 3 + 1];
											varInfos[pos - 1]->z_d[u*Ny*Nz + v*Nz + w] = dataBuf[w*Ny*Nx * 3 + v*Nx * 3 + u * 3 + 2];
										}
								delete[] dataBuf;
							}
							else
							{
								coDoFloat *dataObj = new coDoFloat(READER_CONTROL->getAssocObjName(DPORT1_3D + n).c_str(), numValues);
								dataObj->getAddress(&varInfos[pos - 1]->x_d);

								float *dataBuf = new float[numValues];
								fread(dataBuf, sizeof(float), numValues, d_dataFile);
								for (int u = 0; u < Nx; u++)
									for (int v = 0; v < Ny; v++)
										for (int w = 0; w < Nz; w++)
										{
											varInfos[pos - 1]->x_d[u*Ny*Nz + v*Nz + w] = dataBuf[w*Ny*Nx + v*Nx + u];
										}
								delete[] dataBuf;
							}
							varInfos[pos - 1]->read = true;
						}
					}
				}
				varNumber++;
			}
		}
		else if(arrayInfos.size() == 2)
		{
			//read array data
			for (std::vector<ArrayInfo *>::iterator it = arrayInfos.begin(); it != arrayInfos.end(); it++)
			{
				ArrayInfo *array = *it;
				array->dataBuf = new unsigned char[array->numRows * array->columnSize];
				size_t read = fread(array->dataBuf, array->columnSize, array->numRows, d_dataFile);
				if (read != array->columnSize*array->numRows)
				{
					fprintf(stderr, "read %zu bytes\n", read);
				}
			}
			ArrayInfo *IDArray = arrayInfos[0];
			ArrayInfo *ParticleArray = arrayInfos[1];
			ParticleInfo **particles = new ParticleInfo*[ParticleArray->numRows];
			ParticleInfo ***timestepsPerParticle = new ParticleInfo**[IDArray->numRows];
			int64_t oldPn=0;
			int numSteps = 0;
			int maxSteps = 0;
			int *numstepsPerParticle = new int[IDArray->numRows];
			int n = 0;
			for (int i = 0; i < ParticleArray->numRows; i++)
			{
				particles[i] = (ParticleInfo *)(ParticleArray->dataBuf + i*ParticleArray->columnSize);
				if ((i < 10) || (i > ParticleArray->numRows - 10))
				{
					printf("%zd: %f\n", (ssize_t)particles[i]->ID, particles[i]->time);
				}
				if (i == 0)
					oldPn = particles[i]->ID;
				if (oldPn != particles[i]->ID)
				{
					if (numSteps > maxSteps)
						maxSteps = numSteps;
					oldPn = particles[i]->ID;
					if (n < IDArray->numRows)
					{
						numstepsPerParticle[n] = numSteps;
						timestepsPerParticle[n] = new ParticleInfo*[numSteps];
					}
					else
					{
						fprintf(stderr, "too many particle IDs %d\n", n);
					}
					numSteps = 0;
					n++;
				}
				else
				{
					numSteps++;
				}
			}
			timestepsPerParticle[n] = new ParticleInfo*[numSteps];
			numstepsPerParticle[n] = numSteps;
			n++;
			n = 0;
			oldPn = 0;
			numSteps = 0;
			for (int i = 0; i < ParticleArray->numRows; i++)
			{
				if (i == 0)
					oldPn = particles[i]->ID;
				if (oldPn != particles[i]->ID)
				{
					oldPn = particles[i]->ID;
					numSteps = 0;
					n++;
				}
				else
				{
					timestepsPerParticle[n][numSteps] = particles[i];
					numSteps++;
				}
			}
			n++;
			if (n < IDArray->numRows)
			{
				fprintf(stderr, "too manot enough particle IDs %d\n", n);
			}
			coDistributedObject **pointsObjects = new coDistributedObject*[numSteps + 1];

			for(int t = 0;t < numSteps;t++)
			{
				pointsObjects[t + 1] = NULL;
				std::string objName = objNameBase + std::to_string(t);
				int numParticles = 0;
				for (int i = 0; i < n; i++)
				{
					if (t < numstepsPerParticle[i])
						numParticles++;
				}
				coDoPoints * p = new coDoPoints(objName.c_str(), numParticles);
				pointsObjects[t] = p;
				float *xc, *yc, *zc;
				p->getAddresses(&xc, &yc, &zc);
				numParticles = 0;
				for (int i = 0; i < n; i++)
				{
					if (t < numstepsPerParticle[i])
					{
						xc[numParticles] = timestepsPerParticle[i][t]->x;
						yc[numParticles] = timestepsPerParticle[i][t]->y;
						zc[numParticles] = timestepsPerParticle[i][t]->z;
						numParticles++;
					}
				}
			}
			coDoSet *timeStepSet = new coDoSet(objNameBase.c_str(),pointsObjects);
			timeStepSet->addAttribute("TIMESTEP", "1");
			delete[] numstepsPerParticle;
		}

	}
	return CONTINUE_PIPELINE;
}
int ReadGeoDict::compute(const char *)
{
	if (d_dataFile == NULL)
	{
		if (fileName.empty())
		{
			cerr << "ReadGeoDict::param(..) no data file found " << endl;
		}
		else
		{

			d_dataFile = fopen(fileName.c_str(), "rb");
		}
	}
	int result = STOP_PIPELINE;

	if (d_dataFile != NULL)
	{
		result = readData();
		fclose(d_dataFile);
		d_dataFile = NULL;
	}
	return result;
}

int main(int argc, char *argv[])
{

	// define outline of reader
	READER_CONTROL->addFile(TXT_BROWSER, "data_file_path", "Data file path", ".", "*.vap;*.GeoDict");

	READER_CONTROL->addOutputPort(MESHPORT3D, "geoOut_3D", "structured Grid", "Mesh", false);

	READER_CONTROL->addOutputPort(DPORT1_3D, "data1_3D", "Float|Vec3", "data1-3d");
	READER_CONTROL->addOutputPort(DPORT2_3D, "data2_3D", "Float|Vec3", "data2-3d");
	READER_CONTROL->addOutputPort(DPORT3_3D, "data3_3D", "Float|Vec3", "data3-3d");
	READER_CONTROL->addOutputPort(DPORT4_3D, "data4_3D", "Float|Vec3", "data4-3d");
	READER_CONTROL->addOutputPort(DPORT5_3D, "data5_3D", "Float|Vec3", "data5-3d");

	// create the module
	coReader *application = new ReadGeoDict(argc, argv);

	// this call leaves with exit(), so we ...
	application->start(argc, argv);

	// ... never reach this point
	return 0;
}
