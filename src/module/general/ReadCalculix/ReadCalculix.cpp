/**************************************************************************\
**                                                  (C)2015 Stellba Hydro **
**                                                                        **
** Description: READ Calculix FEM                                         **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Martin Becker                                                  **
**                                                                        **
\**************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <do/coDoPolygons.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include "ReadCalculix.h"
#include <limits.h>

#define sqr(x) ((x)*(x))

ReadCalculix::ReadCalculix(int argc, char *argv[])
	:coModule(argc, argv, "Calculix Reader")
{

	// the output ports
	p_mesh = addOutputPort("mesh", "UnstructuredGrid", "the grid");
	p_displacement = addOutputPort("displacement", "Vec3", "displacement, vector");
	p_vonMises = addOutputPort("vonMises", "Float", "von Mises stress, scalar");
	p_strain = addOutputPort("strain", "Float", "strain, scalar");
	p_normalStress = addOutputPort("stress", "Vec3", "normal stress SXX SYY SZZ (without shear stress), vector");


	// parameters

	// input file
	char filePath[200];
	sprintf(filePath, "%s", getenv("HOME"));
	p_inpFile = addFileBrowserParam("inp_file", "Calculix input file");
	p_inpFile->setValue(filePath, "*.inp");

	//p_readFrd = addBooleanParam("read_frd", "do you want to read the results file");
	//p_readFrd->setValue(true);

	p_frdFile = addFileBrowserParam("resultsFile", "Calculix results file");
	p_frdFile->setValue(filePath, "*.frd");

	p_data_step_to_read = addInt32Param("data_step_to_read", "frd-file can contain multiple steps. Which do you want to read [1..n]?");
	p_data_step_to_read->setValue(1);

}


ReadCalculix::~ReadCalculix()
{

}


int ReadCalculix::compute(const char *)
{
	coDoUnstructuredGrid *grid = NULL;

	float *xCoord, *yCoord, *zCoord;
	std::vector<float> xCoordRead;
	std::vector<float> yCoordRead;
	std::vector<float> zCoordRead;
	int *elem, *conn, *type;  // element, connectivity and type list
	std::vector<int> elemRead;
	std::vector<int> connRead;
	std::vector<int> typeRead;
	xCoordRead.reserve(10000);
	yCoordRead.reserve(10000);
	zCoordRead.reserve(10000);
	elemRead.reserve(10000);
	connRead.reserve(10000);
	typeRead.reserve(10000);

	// read input file (contains mesh and boundaries)
	ifstream inpfile;
	inpfile.open(p_inpFile->getValue());

	// read results file (contains mesh and boundaries)
	ifstream frdfile;
	frdfile.open(p_frdFile->getValue());

	if ((inpfile.is_open()) && (!frdfile.is_open()))
	{
		// by now, input file is only read if frd file is not given

		fprintf(stderr, "\n\nreading calculix input file %s\n", p_inpFile->getValue());

		float  x, y, z;
		int idummy;
		string t;

		// read the leading comment lines	
		int numCommLines = 0;
		getline(inpfile, t, '\n');
		//fprintf(stderr,"comm %d: %s\n",numCommLines, t.c_str());
		while ((t[0] == '*') && (t[1] == '*'))
		{
			numCommLines++;
			getline(inpfile, t, '\n');
		}

		// *NODE line
		fprintf(stderr, "\t*NODE line: %s\n", t.c_str());

		// read all the nodes ...
		getline(inpfile, t, '\n');
		while (t[0] != '*')
		{
			sscanf(t.c_str(), "%d,%f,%f,%f\n", &idummy, &x, &y, &z);
			xCoordRead.push_back(x);
			yCoordRead.push_back(y);
			zCoordRead.push_back(z);
			getline(inpfile, t, '\n');
		}

		// *ELEMENT line
		fprintf(stderr, "\t*ELEMENT line: %s\n", t.c_str());

		// read all the elements ...
		int nodes[10];

		getline(inpfile, t, '\n');
		while (t[0] != '*')
		{
			sscanf(t.c_str(), "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d, %d\n", &idummy, &nodes[0],
				&nodes[1],
				&nodes[2],
				&nodes[3],
				&nodes[4],
				&nodes[5],
				&nodes[6],
				&nodes[7],
				&nodes[8],
				&nodes[9]);

			// 10 nodes tetrahedron is converted to linear (4-nodes) tetrahedron
			if (connRead.size() > INT_MAX)
			{
				sendError("file too large");
				return STOP_PIPELINE;
			}
			elemRead.push_back((int)connRead.size());
			connRead.push_back(nodes[0] - 1);	// we start to count with 0!
			connRead.push_back(nodes[1] - 1);
			connRead.push_back(nodes[2] - 1);
			connRead.push_back(nodes[3] - 1);
			typeRead.push_back(TYPE_TETRAHEDER);	// = tetrahedron
			getline(inpfile, t, '\n');
		}


		fprintf(stderr, "\tnElem = %zd\n", elemRead.size());
		fprintf(stderr, "\tnConn = %zd\n", connRead.size());
		fprintf(stderr, "\tnCoord = %zd\n", xCoordRead.size());

		if (elemRead.size() > INT_MAX)
		{
			sendError("file too large");
			return STOP_PIPELINE;
		}
		if (connRead.size() > INT_MAX)
		{
			sendError("file too large");
			return STOP_PIPELINE;
		}
		if (xCoordRead.size() > INT_MAX)
		{
			sendError("file too large");
			return STOP_PIPELINE;
		}

		grid = new coDoUnstructuredGrid(p_mesh->getObjName(), (int)elemRead.size(), (int)connRead.size(), (int)xCoordRead.size(), &elemRead[0], &connRead[0], &xCoordRead[0], &yCoordRead[0], &zCoordRead[0], &typeRead[0]);

		elemRead.clear();
		connRead.clear();
		typeRead.clear();
		xCoordRead.clear();
		yCoordRead.clear();
		zCoordRead.clear();

		// TODO: read boundaries (node sets, element sets) ...

		inpfile.close();

		p_mesh->setCurrentObject(grid);
	}
	else
	{
		if (!inpfile.is_open())
		{
			sendError("could not open File %s", p_inpFile->getValue());
		}
		else
		{
			inpfile.close();
		}

	}




	// read results file
	// if results file is given, read also mesh from results file!
	// the file has already been opened above

	if (frdfile.is_open())
	{
		fprintf(stderr, "\n\nreading calculix results file %s\n", p_frdFile->getValue());

		float fdummy;
		int idummy;
		char buf[255];
		string t;

		// read the leading comment lines
		getline(frdfile, t, '\n');
		std::size_t found = t.find("    2C");
		//fprintf(stderr,"\tline1: %s\n",t.c_str());

		while (found == std::string::npos)
		{
			getline(frdfile, t, '\n');
			//fprintf(stderr,"\tline: %s\n",t.c_str());
			found = t.find("    2C");
		}

		// parse number of nodes
		int nCoord;
		sscanf(t.c_str(), "%s %d\n", buf, &nCoord);
		fprintf(stderr, "\tnCoord = %d\n", nCoord);


		float *xCoordRead = new float[nCoord];
		float *yCoordRead = new float[nCoord];
		float *zCoordRead = new float[nCoord];
		int *nodenrs = new int[nCoord];
		int maxNodeNr = 0;
		for (int i = 0; i < nCoord; i++)
		{
			getline(frdfile, t, '\n');

			// it might be the case that there are some nodes missing in the list, so we need to read the node nr
			strncpy(buf, t.c_str() + 3, 10);
			buf[10] = '\0';
			sscanf(buf, "%d", &nodenrs[i]);
			if (nodenrs[i] > maxNodeNr)
				maxNodeNr = nodenrs[i];

			buf[12] = '\0';
			strncpy(buf, t.c_str() + 13, 12);
			sscanf(buf, "%f", &xCoordRead[i]);

			strncpy(buf, t.c_str() + 25, 12);
			sscanf(buf, "%f", &yCoordRead[i]);

			strncpy(buf, t.c_str() + 37, 12);
			sscanf(buf, "%f", &zCoordRead[i]);
		}

		getline(frdfile, t, '\n');
		getline(frdfile, t, '\n');
		int nElem;
		// parse number of elements
		sscanf(t.c_str(), "%s %d\n", buf, &nElem);
		fprintf(stderr, "\tnElem = %d\n", nElem);

		int *connRead = new int[nElem * 4];
		int *elemRead = new int[nElem];
		int *typeRead = new int[nElem];

		for (int i = 0; i < nElem; i++)
		{
			getline(frdfile, t, '\n');
			getline(frdfile, t, '\n');

			buf[10] = '\0';
			strncpy(buf, t.c_str() + 3, 10);
			sscanf(buf, "%d\n", &connRead[4 * i + 0]);

			strncpy(buf, t.c_str() + 13, 10);
			sscanf(buf, "%d\n", &connRead[4 * i + 1]);

			strncpy(buf, t.c_str() + 23, 10);
			sscanf(buf, "%d\n", &connRead[4 * i + 2]);

			strncpy(buf, t.c_str() + 33, 10);
			sscanf(buf, "%d\n", &connRead[4 * i + 3]);

			/*
			strncpy(buf, t.c_str()+43, 10);
			sscanf(buf,"%d\n", &connRead[10*i+4]);

			strncpy(buf, t.c_str()+53, 10);
			sscanf(buf,"%d\n", &connRead[10*i+5]);

			strncpy(buf, t.c_str()+63, 10);
			sscanf(buf,"%d\n", &connRead[10*i+6]);

			strncpy(buf, t.c_str()+73, 10);
			sscanf(buf,"%d\n", &connRead[10*i+7]);

			strncpy(buf, t.c_str()+83, 10);
			sscanf(buf,"%d\n", &connRead[10*i+8]);

			strncpy(buf, t.c_str()+93, 10);
			sscanf(buf,"%d\n", &connRead[10*i+9]);
			*/

			// transform to C-notation
			for (int j = 0; j < 4; j++)
			{
				connRead[4 * i + j]--;
			}

			elemRead[i] = 4 * i;
			typeRead[i] = TYPE_TETRAHEDER;
		}

		int nConn = 4 * nElem;
		fprintf(stderr, "\tnConn = %d\n", nConn);

		grid = new coDoUnstructuredGrid(p_mesh->getObjName(), nElem, nConn, maxNodeNr, 1);
		grid->getAddresses(&elem, &conn, &xCoord, &yCoord, &zCoord);
		grid->getTypeList(&type);

		memcpy(elem, &elemRead[0], nElem * sizeof(int));
		memcpy(conn, &connRead[0], nConn * sizeof(int));
		memcpy(type, &typeRead[0], nElem * sizeof(int));
		for (int i = 0; i < nCoord; i++)
		{
			xCoord[nodenrs[i] - 1] = xCoordRead[i];
			yCoord[nodenrs[i] - 1] = yCoordRead[i];
			zCoord[nodenrs[i] - 1] = zCoordRead[i];
		}

		delete[] xCoordRead;
		delete[] yCoordRead;
		delete[] zCoordRead;
		delete[] nodenrs;

		delete[] elemRead;
		delete[] connRead;
		delete[] typeRead;


		//
		// read data
		//

		int data_step_to_read = p_data_step_to_read->getValue() - 1;	// we start to count at 0 internally
		if (data_step_to_read > 0)
		{
			fprintf(stderr, "jumping over %d dataset(s)!\n", data_step_to_read);
		}

		// read displacement (vector)
		float *dispX, *dispY, *dispZ;
		coDoVec3 *displacement = NULL;
		displacement = new coDoVec3(p_displacement->getObjName(), maxNodeNr);	// nCoord can be different to nData (I don't know why)!
		displacement->getAddresses(&dispX, &dispY, &dispZ);
		memset(dispX, 0, maxNodeNr * sizeof(float));
		memset(dispY, 0, maxNodeNr * sizeof(float));
		memset(dispZ, 0, maxNodeNr * sizeof(float));

		// sometimes there is a second coordinate array
		// I don't know why, this might be dependent of the calculix version
		// however, we only read the first coordinate array
		// we want to read the first line PSTEP
		int lines_read = 0;
		do
		{
			getline(frdfile, t, '\n');
			lines_read++;
			found = t.find("PSTEP");
			if (frdfile.eof())
			{
				fprintf(stderr, "the file contains no data!\n");
				return STOP_PIPELINE;
			}
		} while (found == std::string::npos);

		if (lines_read >= nCoord)
		{
			fprintf(stderr, "skipping second coordinate and element list!\n");
		}

		// now we are at first occurence of PSTEP
		// jump to dataset nr. data_step_to_read 
		for (int i = 0; i < 3 * data_step_to_read; i++)	// factor 3 as we have displacement, stress and strain
		{
			if (i % 3 == 0)
				fprintf(stderr, "\tjumping over step %d\n", i / 3 + 1);

			// accelerate jumping: look for number of data, jump over lines, search for next PSTEP .... and so on
			do
			{
				getline(frdfile, t, '\n');
				if (frdfile.eof())
				{
					fprintf(stderr, "did not find '100CL'!\n");
					return STOP_PIPELINE;
				}
				found = t.find("100CL");
			} while (found == std::string::npos);

			int nLines;
			sscanf(t.c_str(), "%s %d %f %d\n", buf, &idummy, &fdummy, &nLines);

			// read until line contains no letters
			// check whether line contains letters
			do
			{
				getline(frdfile, t, '\n');

				// detect whether this is a comment
				//	a comment contains letters (A-Z)
				//	'E' from scientific notation should not be treated as letter --> replace by zero
				std::replace(t.begin(), t.end(), 'E', '0');

			} while (containsLetters(t));

			// we now have the first line with data ...
			// get position in file, read another line, then we know how far we have to go forward (seekg)
			std::streampos pos0 = frdfile.tellg();
			getline(frdfile, t, '\n');
			std::streampos pos1 = frdfile.tellg();
			frdfile.seekg(pos0 + (nLines - 1)*(pos1 - pos0), frdfile.beg);

			// look for next PSTEP
			do
			{
				getline(frdfile, t, '\n');
				if (frdfile.eof())
				{
					fprintf(stderr, "the file contains no additional dataset - consider to set data_step_to_read to 1!\n");
					return STOP_PIPELINE;
				}
				found = t.find("PSTEP");
			} while (found == std::string::npos);

			/*
					do
					{
						getline(frdfile, t, '\n');
						if (frdfile.eof())
						{
							fprintf(stderr,"the file contains no additional dataset - consider to set data_step_to_read to 1!\n");
							return STOP_PIPELINE;
						}
						found = t.find("PSTEP");
					} while (found == std::string::npos);
			*/
		}

		// read number of data values
		// read until line contains "100CL"
		do
		{
			getline(frdfile, t, '\n');
			if (frdfile.eof())
			{
				fprintf(stderr, "did not find '100CL'!\n");
				return STOP_PIPELINE;
			}

			found = t.find("100CL");
		} while (found == std::string::npos);


		int nData;
		sscanf(t.c_str(), "%s %d %f %d\n", buf, &idummy, &fdummy, &nData);
		fprintf(stderr, "\tnData = %d\n", nData);

		// jump to first data line
		for (int i = 0; i < 5; i++)
		{
			getline(frdfile, t, '\n');
			//fprintf(stderr,"line %d: '%s'\n", i, t.c_str());
		}


		for (int i = 0; i < nData; i++)
		{
			getline(frdfile, t, '\n');

			// it might be the case that there are some nodes missing in the list, so we need to read the node nr
			int nodenr;
			strncpy(buf, t.c_str() + 3, 10);
			buf[10] = '\0';
			sscanf(buf, "%d\n", &nodenr);

			buf[12] = '\0';
			if (nodenr - 1 >= maxNodeNr)
			{
				fprintf(stderr, "index out of range %d\n", nodenr - 1);
			}
			else
			{
				strncpy(buf, t.c_str() + 13, 12);
				sscanf(buf, "%f", &dispX[nodenr - 1]);

				strncpy(buf, t.c_str() + 25, 12);
				sscanf(buf, "%f", &dispY[nodenr - 1]);

				strncpy(buf, t.c_str() + 37, 12);
				sscanf(buf, "%f", &dispZ[nodenr - 1]);
			}
		}

		p_displacement->setCurrentObject(displacement);


		// read stress (and calculate scalar vonMises value)

		float *sxx, *syy, *szz;
		float sxy, syz, szx;
		coDoVec3 *normalStress = NULL;
		normalStress = new coDoVec3(p_normalStress->getObjName(), maxNodeNr);	// nCoord can be different to nData (I don't know why)!
		normalStress->getAddresses(&sxx, &syy, &szz);
		memset(sxx, 0, maxNodeNr * sizeof(float));
		memset(syy, 0, maxNodeNr * sizeof(float));
		memset(szz, 0, maxNodeNr * sizeof(float));

		float *vonMiseVal;
		coDoFloat *vonMises = NULL;
		vonMises = new coDoFloat(p_vonMises->getObjName(), maxNodeNr);	// nCoord can be different to nData (I don't know why)!
		vonMises->getAddress(&vonMiseVal);
		memset(vonMiseVal, 0, maxNodeNr * sizeof(float));

		// jump to first data line
		// read until line contains "100CL"
		do
		{
			getline(frdfile, t, '\n');
			if (frdfile.eof())
			{
				fprintf(stderr, "did not find '100CL'!\n");
				return STOP_PIPELINE;
			}

			found = t.find("100CL");
		} while (found == std::string::npos);

		// read until line contains "SZX"
		do
		{
			getline(frdfile, t, '\n');
			if (frdfile.eof())
			{
				fprintf(stderr, "did not find 'SZX'!\n");
				return STOP_PIPELINE;
			}

			found = t.find("SZX");
		} while (found == std::string::npos);

		// read data values
		for (int i = 0; i < nData; i++)
		{
			getline(frdfile, t, '\n');

			// it might be the case that there are some nodes missing in the list, so we need to read the node nr
			int nodenr;
			strncpy(buf, t.c_str() + 3, 10);
			buf[10] = '\0';
			sscanf(buf, "%d\n", &nodenr);

			buf[12] = '\0';
			if (nodenr - 1 >= maxNodeNr)
			{
				fprintf(stderr, "index out of range %d\n", nodenr - 1);
			}
			else
			{
				strncpy(buf, t.c_str() + 13, 12);
				sscanf(buf, "%f", &sxx[i]);

				strncpy(buf, t.c_str() + 25, 12);
				sscanf(buf, "%f", &syy[i]);

				strncpy(buf, t.c_str() + 37, 12);
				sscanf(buf, "%f", &szz[i]);

				strncpy(buf, t.c_str() + 49, 12);
				sscanf(buf, "%f", &sxy);

				strncpy(buf, t.c_str() + 61, 12);
				sscanf(buf, "%f", &syz);

				strncpy(buf, t.c_str() + 73, 12);
				sscanf(buf, "%f", &szx);

				vonMiseVal[nodenr - 1] = sqrt(sqr(sxx[i]) + sqr(syy[i]) + sqr(szz[i])
					- sxx[i] * syy[i] - sxx[i] * szz[i] - syy[i] * szz[i]
					+ 3 * (sqr(sxy) + sqr(syz) + sqr(szx)));
			}
		}

		// read strain

		// TODO
		// for now, we only read until next PSTEP as reading strain is not implemented yet
		do
		{
			getline(frdfile, t, '\n');
			if (frdfile.eof()) break;
			found = t.find("PSTEP");
		} while (found == std::string::npos);



		// check whether there is more data ...
		lines_read = 0;
		do
		{
			getline(frdfile, t, '\n');
			if (frdfile.eof()) break;
			lines_read++;
			found = t.find("PSTEP");

			if (found != std::string::npos)
			{
				fprintf(stderr, "found an additional data step. Set parameter data_step_to_read = %d to read it!\n", data_step_to_read + 2);
			}

		} while (found == std::string::npos);



		vonMises->addAttribute("SPECIES", "vonMises [N/mm^2]");
		p_vonMises->setCurrentObject(vonMises);

		displacement->addAttribute("SPECIES", "displacement [mm]");
		p_displacement->setCurrentObject(displacement);

		p_normalStress->setCurrentObject(normalStress);







		frdfile.close();

		p_mesh->setCurrentObject(grid);
	}
	else
	{
		sendError("could not open calculix results file %s", p_inpFile->getValue());
		return STOP_PIPELINE;
	}





	return SUCCESS;
}


bool ReadCalculix::containsLetters(string t)
{
	for (int i = 0; i < t.length(); i++)
	{
		if (isalpha(t.at(i)))
		{
			return(true);
		}
	}

	return(false);
}




MODULE_MAIN(IO_Module, ReadCalculix)
