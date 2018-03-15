/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <map>
#include <osg/Matrix>
#include <osg/Vec3>
#include <util/unixcompat.h>

#include <stdint.h>
#ifdef HAVE_E57
#include <e57/E57Foundation.h>
#include <e57/E57Simple.h>
#endif

using namespace std;

float min_x, min_y, min_z;
float max_x, max_y, max_z;
bool intensityOnly;

enum formatTypes
{
	FORMAT_IRGB,
	FORMAT_RGBI,
	FORMAT_If,
	FORMAT_RGB,
	FORMAT_UVRGBI
};

struct Point
{
	float x;
	float y;
	float z;
	uint32_t rgba;
};

void ReadData(char *filename, std::vector<Point> &vec, formatTypes format)
{

	FILE *inputFile;
	cout << "Input Data: " << filename << endl;

	inputFile = fopen(filename, "r");

	if (inputFile == NULL)
	{
		cout << "Error opening file:" << filename;
		return;
	}

	int in, r, g, b, u, v;
	float rf, gf, bf;
	Point point;

	char buf[1000];
	int toRead = 1000;
	if (format == FORMAT_UVRGBI)
		toRead = 8;
	else if (format == FORMAT_IRGB)
		toRead = 7;
	else if (format == FORMAT_RGBI)
		toRead = 6;
	else if (format == FORMAT_If)
		toRead = 4;
	else if (format == FORMAT_RGB)
		toRead = 6;

	while (fgets(buf, 1000, inputFile) != NULL)
	{
		if (buf[0] == '/')
		{
			if (strcmp(buf, "//X,Y,Z,Scalar field,R,G,B") == 0)
			{
				fgets(buf, 1000, inputFile); // num points
				fgets(buf, 1000, inputFile);
			}
		}
		int numValues;
		if (format == FORMAT_UVRGBI)
			numValues = sscanf(buf, "%d %d %f %f %f %d %d %d", &u, &v, &(point.x), &(point.y), &(point.z), &r, &g, &b);
		else if (format == FORMAT_IRGB)
		{
			char *c = buf;
			char *nc = c;
			numValues = 0;
#ifdef WIN32
			point.x = strtod(c, &nc);
#else
			point.x = strtof(c, &nc);
#endif
			if (nc != c)
				numValues++;
			c = nc;
			if (*c == ',')
				c++;
			if (*c == ';')
				c++;
#ifdef WIN32
			point.y = strtod(c, &nc);
#else
			point.y = strtof(c, &nc);
#endif
			if (nc != c)
				numValues++;
			c = nc;
			if (*c == ',')
				c++;
			if (*c == ';')
				c++;
#ifdef WIN32
			point.z = strtod(c, &nc);
#else
			point.z = strtof(c, &nc);
#endif
			if (nc != c)
				numValues++;
			c = nc;
			if (*c == ',')
				c++;
			if (*c == ';')
				c++;
#ifdef WIN32
			in = strtod(c, &nc);
#else
			in = strtof(c, &nc);
#endif
			if (nc != c)
				numValues++;
			c = nc;
			if (*c == ',')
				c++;
			if (*c == ';')
				c++;
			r = strtol(c, &nc, 10);
			if (nc != c)
				numValues++;
			c = nc;
			if (*c == ',')
				c++;
			if (*c == ';')
				c++;
			g = strtol(c, &nc, 10);
			if (nc != c)
				numValues++;
			c = nc;
			if (*c == ',')
				c++;
			if (*c == ';')
				c++;
			b = strtol(c, &nc, 10);
			if (nc != c)
				numValues++;
			c = nc;
			if (*c == ',')
				c++;
			if (*c == ';')
				c++;
			//numValues =sscanf (buf, "%f %f %f %f %d %d %d", &(point.x),&(point.y),&(point.z), &in, &r, &g, &b);
		}
		else if (format == FORMAT_RGBI)
			numValues = sscanf(buf, "%f %f %f %d %d %d", &(point.x), &(point.y), &(point.z), &r, &g, &b);
		else if (format == FORMAT_If)
		{
			numValues = sscanf(buf, "%f %f %f %f", &(point.x), &(point.y), &(point.z), &rf);
			g = (int)((rf / 2000.0) * 255);
			b = (int)((rf / 2000.0) * 255);
			r = (int)((rf / 2000.0) * 255);
		}
		else if (format == FORMAT_RGB)
		{

			numValues = sscanf(buf, "%f %f %f %f %f %f", &(point.x), &(point.y), &(point.z), &rf, &gf, &bf);
			r = (int)(rf * 255);
			g = (int)(gf * 255);
			b = (int)(bf * 255);
		}
		if (numValues == toRead && !(point.x == 0.0 && point.y == 0.0 && point.z == 0.0))
		{
			point.rgba = r | g << 8 | b << 16;
			vec.push_back(point);
		}
	}

	fclose(inputFile);
}

void ReadPTX(char *filename, std::vector<Point> &vec)
{

	FILE *inputFile;
	cout << "Input Data PTX: " << filename << endl;

	inputFile = fopen(filename, "r");

	if (inputFile == NULL)
	{
		cout << "Error opening file:" << filename;
		return;
	}

	int r, g, b;
	float fa, fb, fc;
	Point point;

	char buf[1000];
	int toRead = 7;
	bool readHeader = true;
	int numRows;
	int numCols;
	unsigned int numLines;
	unsigned int linesRead;
	osg::Matrix m;
	m.makeIdentity();
	while (fgets(buf, 1000, inputFile) != NULL)
	{
		if (readHeader)
		{
			sscanf(buf, "%d", &numRows);
			fgets(buf, 1000, inputFile);
			sscanf(buf, "%d", &numCols);
			fgets(buf, 1000, inputFile);
			fgets(buf, 1000, inputFile);
			fgets(buf, 1000, inputFile);
			fgets(buf, 1000, inputFile); // scanner pos.
			fgets(buf, 1000, inputFile); // 4x4 Matrix
			sscanf(buf, "%f %f %f", &fa, &fb, &fc);
			m(0, 0) = fa;
			m(1, 0) = fb;
			m(2, 0) = fc;
			fgets(buf, 1000, inputFile);
			sscanf(buf, "%f %f %f", &fa, &fb, &fc);
			m(0, 1) = fa;
			m(1, 1) = fb;
			m(2, 1) = fc;
			fgets(buf, 1000, inputFile);
			sscanf(buf, "%f %f %f", &fa, &fb, &fc);
			m(0, 2) = fa;
			m(1, 2) = fb;
			m(2, 2) = fc;
			fgets(buf, 1000, inputFile); // scanner pos.
			sscanf(buf, "%f %f %f", &fa, &fb, &fc);
			m(0, 3) = fa;
			m(1, 3) = fb;
			m(2, 3) = fc;
			readHeader = false;
			linesRead = 0;
			numLines = numRows * numCols;

			printf("Reading in %d lines\n", numLines);
		}
		else
		{
			linesRead++;
			if (linesRead == numLines)
			{
				fprintf(stderr, "Restart %d %d %d\n", linesRead, numRows, numCols);
				readHeader = true;
			}

			int numValues;

			char *c = buf;
			char *nc = c;
			numValues = 0;
#ifdef WIN32
			point.x = strtod(c, &nc);
#else
			point.x = strtof(c, &nc);
#endif
			if (nc != c)
				numValues++;
			c = nc;
#ifdef WIN32
			point.y = strtod(c, &nc);
#else
			point.y = strtof(c, &nc);
#endif
			if (nc != c)
				numValues++;
			c = nc;
#ifdef WIN32
			point.z = strtod(c, &nc);
#else
			point.z = strtof(c, &nc);
#endif
			if (nc != c)
				numValues++;
			c = nc;
#ifdef WIN32
			double in = strtod(c, &nc);
#else
			float in = strtof(c, &nc);
#endif
			if (nc != c)
				numValues++;
			c = nc;
			r = strtol(c, &nc, 10);
			if (nc != c)
				numValues++;
			c = nc;
			g = strtol(c, &nc, 10);
			if (nc != c)
				numValues++;
			c = nc;
			b = strtol(c, &nc, 10);
			if (nc != c)
				numValues++;
			c = nc;
			if (numValues == toRead && !(point.x == 0.0 && point.y == 0.0 && point.z == 0.0))
			{
				osg::Vec3 p(point.x, point.y, point.z);
				p = m * p;
				point.x = p[0];
				point.y = p[1];
				point.z = p[2];
				if (intensityOnly)
				{
					unsigned char intensity = (unsigned char)(in*255.99);
					point.rgba = intensity | intensity << 8 | intensity << 16;
				}
				else
				{
					point.rgba = r | g << 8 | b << 16;
				}
				vec.push_back(point);
			}
		}
	}

	fclose(inputFile);
}
void ReadE57(char *filename, std::vector<Point> &vec)
{

#ifdef HAVE_E57

	osg::Matrix m;
	m.makeIdentity();
	try
	{
		e57::Reader	eReader(filename);
		e57::E57Root	rootHeader;
		eReader.GetE57Root(rootHeader);


		//Get the number of scan images available
		int data3DCount = eReader.GetData3DCount();
		int scanIndex = 0;
		e57::Data3D		scanHeader;
		eReader.ReadData3D(scanIndex, scanHeader);
		fprintf(stderr, "reading Name: %s\n", scanHeader.name.c_str());
		osg::Matrix trans;
		trans.makeTranslate(scanHeader.pose.translation.x, scanHeader.pose.translation.y, scanHeader.pose.translation.z);
		osg::Matrix rot;
		rot.makeRotate(osg::Quat(scanHeader.pose.rotation.x, scanHeader.pose.rotation.y, scanHeader.pose.rotation.z, scanHeader.pose.rotation.w));
		m = rot*trans;

		int64_t nColumn = 0;
		int64_t nRow = 0;


		int64_t nPointsSize = 0;	//Number of points


		int64_t nGroupsSize = 0;	//Number of groups
		int64_t nCountSize = 0;		//Number of points per group
		bool	bColumnIndex = false; //indicates that idElementName is "columnIndex"


		eReader.GetData3DSizes(scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, nCountSize, bColumnIndex);


		int64_t nSize = nRow;
		if (nSize == 0) nSize = 1024;	// choose a chunk size

		int8_t * isInvalidData = NULL;
		if (scanHeader.pointFields.cartesianInvalidStateField)
			isInvalidData = new int8_t[nSize];


		double * xData = NULL;
		if (scanHeader.pointFields.cartesianXField)
			xData = new double[nSize];
		double * yData = NULL;
		if (scanHeader.pointFields.cartesianYField)
			yData = new double[nSize];
		double * zData = NULL;
		if (scanHeader.pointFields.cartesianZField)
			zData = new double[nSize];

		double *	intData = NULL;
		bool		bIntensity = false;
		double		intRange = 0;
		double		intOffset = 0;


		if (scanHeader.pointFields.intensityField)
		{
			bIntensity = true;
			intData = new double[nSize];
			intRange = scanHeader.intensityLimits.intensityMaximum - scanHeader.intensityLimits.intensityMinimum;
			intOffset = scanHeader.intensityLimits.intensityMinimum;
		}


		uint16_t *	redData = NULL;
		uint16_t *	greenData = NULL;
		uint16_t *	blueData = NULL;
		bool		bColor = false;
		int32_t		colorRedRange = 1;
		int32_t		colorRedOffset = 0;
		int32_t		colorGreenRange = 1;
		int32_t		colorGreenOffset = 0;
		int32_t		colorBlueRange = 1;
		int32_t		colorBlueOffset = 0;


		if (scanHeader.pointFields.colorRedField)
		{
			bColor = true;
			redData = new uint16_t[nSize];
			greenData = new uint16_t[nSize];
			blueData = new uint16_t[nSize];
			colorRedRange = scanHeader.colorLimits.colorRedMaximum - scanHeader.colorLimits.colorRedMinimum;
			colorRedOffset = scanHeader.colorLimits.colorRedMinimum;
			colorGreenRange = scanHeader.colorLimits.colorGreenMaximum - scanHeader.colorLimits.colorGreenMinimum;
			colorGreenOffset = scanHeader.colorLimits.colorGreenMinimum;
			colorBlueRange = scanHeader.colorLimits.colorBlueMaximum - scanHeader.colorLimits.colorBlueMinimum;
			colorBlueOffset = scanHeader.colorLimits.colorBlueMinimum;
		}



		int64_t * idElementValue = NULL;
		int64_t * startPointIndex = NULL;
		int64_t * pointCount = NULL;
		if (nGroupsSize > 0)
		{
			idElementValue = new int64_t[nGroupsSize];
			startPointIndex = new int64_t[nGroupsSize];
			pointCount = new int64_t[nGroupsSize];

			if (!eReader.ReadData3DGroupsData(scanIndex, nGroupsSize, idElementValue,
				startPointIndex, pointCount))
				nGroupsSize = 0;
		}

		int8_t * rowIndex = NULL;
		int32_t * columnIndex = NULL;
		if (scanHeader.pointFields.rowIndexField)
			rowIndex = new int8_t[nSize];
		if (scanHeader.pointFields.columnIndexField)
			columnIndex = new int32_t[nRow];




		e57::CompressedVectorReader dataReader = eReader.SetUpData3DPointsData(
			scanIndex,			//!< data block index given by the NewData3D
			nRow,				//!< size of each of the buffers given
			xData,				//!< pointer to a buffer with the x data
			yData,				//!< pointer to a buffer with the y data
			zData,				//!< pointer to a buffer with the z data
			isInvalidData,		//!< pointer to a buffer with the valid indication
			intData,			//!< pointer to a buffer with the lidar return intesity
			NULL,
			redData,			//!< pointer to a buffer with the color red data
			greenData,			//!< pointer to a buffer with the color green data
			blueData/*,*/			//!< pointer to a buffer with the color blue data
			/*NULL,
			NULL,
			NULL,
			NULL,
			rowIndex,			//!< pointer to a buffer with the rowIndex
			columnIndex			//!< pointer to a buffer with the columnIndex*/
		);

		int64_t		count = 0;
		unsigned	size = 0;
		int			col = 0;
		int			row = 0;

		Point point;
		while (size = dataReader.read())
		{
			for (unsigned int i = 0; i < size; i++)
			{

				if (isInvalidData[i] == 0)
				{
					osg::Vec3 p(xData[i], yData[i], zData[i]);
					p = p*m;
					point.x = p[0];
					point.y = p[1];
					point.z = p[2];


					if (bIntensity) {		//Normalize intensity to 0 - 1.
						int intensity = ((intData[i] - intOffset) / intRange) * 255;
						point.rgba = intensity | intensity << 8 | intensity << 16;
					}


					if (bColor) {			//Normalize color to 0 - 255
						int red = ((redData[i] - colorRedOffset) * 255) / colorRedRange;
						int green = ((greenData[i] - colorGreenOffset) * 255) / colorBlueRange;
						int blue = ((blueData[i] - colorBlueOffset) * 255) / colorBlueRange;

						point.rgba = red | green << 8 | blue << 16;

					}
					vec.push_back(point);

				}
			}

		}
		dataReader.close();

		if (isInvalidData) delete isInvalidData;
		if (xData) delete xData;
		if (yData) delete yData;
		if (zData) delete zData;
		if (intData) delete intData;
		if (redData) delete redData;
		if (greenData) delete greenData;
		if (blueData) delete blueData;
		eReader.Close();
	}
	catch (e57::E57Exception& ex) {
		ex.report(__FILE__, __LINE__, __FUNCTION__);
	}
	catch (std::exception& ex) {
		cerr << "Got an std::exception, what=" << ex.what() << endl;
	}
	catch (...) {
		cerr << "Got an unknown exception" << endl;
	}
#else
	cout << "Missing e75 library " << filename << endl;
#endif

}

void WriteData(char *filename, std::vector<Point> &vec)
{
	cout << "Output Data: " << filename << endl;

	ofstream file(filename, ios::out | ios::binary | ios::ate);

	if (file.is_open())
	{
		int number_of_sets = 1;
		int index = 0;

		//printf("Vector size is %d\n", vec->size());
		//printf("Number of sets is %d\n", number_of_sets);

		// write the number of sets
		file.write((char *)&(number_of_sets), sizeof(int));

		uint32_t numPoints = vec.size();
		file.write((char *)&(numPoints), sizeof(uint32_t));
		for (int i = 0; i < vec.size(); i++)
		{
			file.write((char *)&(vec.at(i).x), sizeof(float) * 3);
		}
		for (int i = 0; i < vec.size(); i++)
		{
			file.write((char *)&(vec.at(i).rgba), sizeof(uint32_t));
		}
	}
	file.close();

	cout << "Data Written!" << endl;
}

int main(int argc, char **argv)
{

	// TODO these values should be command line arguments
	formatTypes format = FORMAT_IRGB;
	//format = FORMAT_RGB;
	std::vector<Point> vec;
	vec.reserve(1000000);

	min_x = min_y = min_z = FLT_MAX;
	max_x = max_y = max_z = FLT_MIN;

    int nread = 0;
	if (argc < 3) /* argc should be > 3 for correct execution */
	{
		printf("Minimal two params required. read README.txt\n");
	}
	else
	{
		for (int i = 1; i < argc - 1; i++)
		{
			int len = strlen(argv[i]);
			if ((len > 1) && argv[i][0] == '-')
			{
				if (argv[i][1] == 'i')
				{
					intensityOnly = true;
				}
				if (argv[i][1] == 'r')
                {
                    format = FORMAT_RGB;
                }

			}
			else
			{
                ++nread;
                printf("Reading in %s\n", argv[i]);
				if ((len > 4) && strcasecmp((argv[i] + len - 4), ".ptx") == 0)
				{
					ReadPTX(argv[i], vec);
				}
				else if ((len > 4) && strcasecmp((argv[i] + len - 4), ".e57") == 0)
				{
					ReadE57(argv[i], vec);
				}
				else if ((len > 4) && strcmp((argv[i] + len - 4), ".xyz") == 0)
				{
					format = FORMAT_RGBI;
					ReadData(argv[i], vec, format);
				}
				else if ((len > 4) && strcmp((argv[i] + len - 4), ".pts") == 0)
				{   if (format != FORMAT_RGB)
					    format = FORMAT_If;
					ReadData(argv[i], vec, format);
				}
				else
				{
					ReadData(argv[i], vec, format);
				}
			}
		}
        if (nread > 0)
            WriteData(argv[argc - 1], vec);
        else
            printf("Did not read any data\n");
	}

    //printf("maxPointsPerCube=%d, divisionSize=%d\n", maxPointsPerCube, divisionSize);
	return 0;
}
