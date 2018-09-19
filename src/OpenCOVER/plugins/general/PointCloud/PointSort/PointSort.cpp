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
#include <stdint.h>
#include <osg/Matrix>
#include <osg/Vec3>
#include <util/unixcompat.h>
#ifdef HAVE_E57
#include <e57/E57Foundation.h>
#include <e57/E57Simple.h>
#endif
#include <util/unixcompat.h>

#if defined(__GNUC__) && !defined(__clang__)
#include <parallel/algorithm>
namespace alg = __gnu_parallel;
#else
namespace alg = std;
#endif

using namespace std;
bool intensityOnly;
bool readScannerPositions = false;
uint32_t fileVersion=1;

float min_x, min_y, min_z;
float max_x, max_y, max_z;

enum formatTypes
{
    FORMAT_IRGB,
    FORMAT_RGBI,
    FORMAT_UVRGBI
};

struct Point
{
    float x;
    float y;
    float z;
    uint32_t rgba;
    int l;
    uint32_t ID = 0;
};

struct ScannerPosition
{
    uint32_t ID;
    osg::Vec3 point;
};

bool sortfunction(Point i, Point j) { return (i.l < j.l); };

void ReadData(char *filename, std::vector<Point> &vec, formatTypes format, std::vector<ScannerPosition> &posVec)
{

	cout << "Input Data: " << filename << endl;

	int len = strlen(filename);
	if ((len > 4) && strcasecmp((filename + len - 4), ".e57") == 0)
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
			e57::Data3D		scanHeader;
			cerr << "Total num of sets is " << data3DCount << endl;
			for (int scanIndex = 0; scanIndex < data3DCount; scanIndex++)
			{
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
                isInvalidData = new int8_t[nSize];
                if (!scanHeader.pointFields.cartesianInvalidStateField)
                {
                    for (int i = 0; i < nSize; i++)
                        isInvalidData[i] = 0;
                }


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
					nSize,				//!< size of each of the buffers given
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
							p = p * m;
							point.x = p[0];
							point.y = p[1];
							point.z = p[2];
							if (point.x < min_x)
								min_x = point.x;
							if (point.y < min_y)
								min_y = point.y;
							if (point.z < min_z)
								min_z = point.z;

							if (point.x > max_x)
								max_x = point.x;
							if (point.y > max_y)
								max_y = point.y;
							if (point.z > max_z)
								max_z = point.z;


							if (bIntensity) {		//Normalize intensity to 0 - 1.
								int intensity = ((intData[i] - intOffset) / intRange) * 255;
								point.l = intensity;
								point.rgba = intensity | intensity << 8 | intensity << 16;
							}


							if (bColor) {			//Normalize color to 0 - 255
								int r = ((redData[i] - colorRedOffset)*255.0) / colorRedRange;
								int g = ((greenData[i] - colorGreenOffset)*255.0) / colorBlueRange;
								int b = ((blueData[i] - colorBlueOffset)*255.0) / colorBlueRange;
								point.rgba = r | g << 8 | b << 16;

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

			}

			eReader.Close();
			return;
		}
		catch (e57::E57Exception& ex) {
			ex.report(__FILE__, __LINE__, __FUNCTION__);
			return;
		}
		catch (std::exception& ex) {
			cerr << "Got an std::exception, what=" << ex.what() << endl;
			return;
		}
		catch (...) {
			cerr << "Got an unknown exception" << endl;
			return;
		}
#else
		cout << "Missing e75 library " << filename << endl;
#endif
	}
	else
	{

		ifstream file(filename, ios::in | ios::binary);

		int count = 0;
		if (file.is_open())
		{
            size_t vecBegin = vec.size();
			uint32_t size;
			file.read((char *)&size, sizeof(uint32_t));
			cerr << "Total num of sets is " << (size) << endl;
			cerr << "max_size: " << vec.max_size() << "\n";
			cerr << "size: " << vec.size() << "\n";
			for (uint32_t i = 0; i < size; i++)
			{
				unsigned int psize;
				file.read((char *)&psize, sizeof(psize));
				printf("Size of set %d is %d\n", i, psize);
				// read point data
				size_t numP = psize;
				size_t numF = 3 * numP;
				printf("numFloats %zu is %d\n", (size_t)i, (int)numF);
				float *coord = new float[numF];
				uint32_t *icolor = new uint32_t[psize];
				file.read((char *)(coord), (sizeof(float) * 3 * psize));
				//read color data
				file.read((char *)(icolor), (sizeof(uint32_t) * psize));
				Point point;
				for (size_t j = 0; j < psize; j++)
				{
					point.x = coord[j * 3];
					point.y = coord[j * 3 + 1];
					point.z = coord[j * 3 + 2];
					point.rgba = icolor[j];
					point.l = 0;
					vec.push_back(point);

					if (point.x < min_x)
						min_x = point.x;
					if (point.y < min_y)
						min_y = point.y;
					if (point.z < min_z)
						min_z = point.z;

					if (point.x > max_x)
						max_x = point.x;
					if (point.y > max_y)
						max_y = point.y;
					if (point.z > max_z)
						max_z = point.z;
				}
				delete[] coord;
				delete[] icolor;
			}
            if (readScannerPositions)
            {
                //read Scanner positions
                uint32_t version;
                file.read((char *)&version,sizeof(uint32_t));
                cerr << "Version " << (version) << endl;
                uint32_t numPositions;
                file.read((char *)&numPositions, sizeof(uint32_t));
                for (int i=0; i!=numPositions; i++)
                {
                    ScannerPosition pos;
                    file.read((char *)&pos.ID, sizeof(uint32_t));
                    file.read((char *)&pos.point._v, sizeof(float) * 3);
                    posVec.push_back(pos);
                    cerr << "Scannerposition " << pos.ID << " x: " << pos.point.x() << " y: " << pos.point.y() << " z: " << pos.point.z() << endl;
                }

                //uint32_t size;
                file.read((char *)&size, sizeof(uint32_t));
                cerr << "Total num of sets with scanner position is " << (size) << endl;
                for (uint32_t i = 0; i < size; i++)
                {
                    unsigned int psize;
                    file.read((char *)&psize, sizeof(psize));
                    printf("Size of set %d is %d\n", i, psize);
                    // read position ID data
                    size_t numP = psize;

                    uint32_t *IDs = new uint32_t[numP];

                    file.read((char *)(IDs), (sizeof(uint32_t) * psize));

                    for (size_t j = 0; j < psize; j++)
                    {
                        vec.at(j+vecBegin).ID = IDs[j];
                    }
                    delete[] IDs;
                }
            }
            file.close();
		}
	}
}

void LabelData(int grid, std::vector<Point> &vec, std::map<int, int> &lookUp)
{

    int xl, yl, zl, xs, ys, zs;

    float xsize, ysize, zsize;
    xs = ys = zs = grid;

    xsize = (max_x - min_x) / grid;
    ysize = (max_y - min_y) / grid;
    zsize = (max_z - min_z) / grid;

    // compute preportional grid sizes
    if (xsize <= ysize && xsize <= zsize)
    {
        xs = grid;
        ys = (int)((max_y - min_y) / xsize);
        ysize = (max_y - min_y) / ys;
        zs = (int)((max_z - min_z) / xsize);
        zsize = (max_z - min_z) / zs;
    }
    else if (ysize <= xsize && ysize <= zsize)
    {
        ys = grid;
        xs = (int)((max_x - min_x) / ysize);
        xsize = (max_x - min_x) / xs;
        zs = (int)((max_z - min_z) / ysize);
        zsize = (max_z - min_z) / zs;
    }
    else
    {
        zs = grid;
        ys = (int)((max_y - min_y) / zsize);
        ysize = (max_y - min_y) / ys;
        xs = (int)((max_x - min_x) / zsize);
        xsize = (max_x - min_x) / zs;
    }

    std::map<int, int>::iterator it;

	printf("Number of points is %d\n", (int)vec.size());
	printf("min %f %f %f\n", min_x, min_y, min_z);
	printf("max %f %f %f\n", max_x, max_y, max_z);
	printf("size %f %f %f\n", xsize, ysize, zsize);
	printf("grid %d\n", grid);

    for (int i = 0; i < vec.size(); i++)
    {

        xl = (int)((vec.at(i).x - min_x) / xsize);
        yl = (int)((vec.at(i).y - min_y) / ysize);
        zl = (int)((vec.at(i).z - min_z) / zsize);

        if (xl == xs)
            xl--;
        if (yl == ys)
            yl--;
        if (zl == zs)
            zl--;

        vec.at(i).l = xl + (yl * xs) + (zl * xs * ys);

        it = lookUp.find(vec.at(i).l);
        if (it != lookUp.end())
        {
            (*it).second++;
        }
        else
        {
            lookUp.insert(std::pair<int, int>(vec.at(i).l, 1));
        }
    }

    cout << "Total Number of sets is " << lookUp.size() << endl;

    // randomize all the data
    //std::random_shuffle(vec.begin(), vec.end());
    alg::random_shuffle(vec.begin(), vec.end());

    // sort data
    //std::sort(vec.begin(), vec.end(), sortfunction);
    alg::sort(vec.begin(), vec.end(), sortfunction);
}

void WriteData(char *filename, std::vector<Point> &vec, std::map<int, int> &lookUp, int maxPointsPerCube, std::vector<ScannerPosition> scanPositions)
{
    cout << "Output Data: " << filename << endl;

    ofstream file(filename, ios::out | ios::binary | ios::ate);
    int numPointsToWrite;

    if (file.is_open())
    {
        int number_of_sets = (int)lookUp.size();
        int index = 0;

        // write the number of sets
        file.write((char *)&(number_of_sets), sizeof(int));

        for (int i = 0; i < number_of_sets; i++)
        {
            // get first in set to find set number
            Point first = vec.at(index);

            std::map<int, int>::iterator it;
            it = lookUp.find(first.l);
            if (it != lookUp.end())
            {
                int numPoints = (*it).second;
                //printf("Number of points in set is %d\n", numPoints);

                // restrict number of points written
                if (maxPointsPerCube != -1 && maxPointsPerCube < numPoints)
                {
                    numPointsToWrite = maxPointsPerCube;
                }
                else
                {
                    numPointsToWrite = numPoints;
                }

                // write size
                file.write((char *)&(numPointsToWrite), sizeof(int));

                // write points
                for (int j = index; j < numPointsToWrite + index; j++)
                {
                    file.write((char *)&(vec.at(j).x), sizeof(float) * 3);
                }

                // write colors
                for (int j = index; j < numPointsToWrite + index; j++)
                {
                    file.write((char *)&(vec.at(j).rgba), sizeof(uint32_t));
                }

                // increment offset
                index = index + numPoints;
            }
        }
        if (readScannerPositions)
        {
            //  Do the same thing for scanner positions
            file.write((char *)&(fileVersion), sizeof(uint32_t));
            uint32_t numPositions= scanPositions.size();
            file.write((char *)&(numPositions), sizeof(uint32_t));
            for (std::vector<ScannerPosition>::const_iterator posIter = scanPositions.begin(); posIter!=scanPositions.end(); posIter++)
            {
                file.write((char *)&(posIter->ID), sizeof(uint32_t));
                file.write((char *)&(posIter->point._v), sizeof(float) * 3);
            }

            index = 0;

            // write the number of sets
            file.write((char *)&(number_of_sets), sizeof(int));

            for (int i = 0; i < number_of_sets; i++)
            {
                // get first in set to find set number
                Point first = vec.at(index);

                std::map<int, int>::iterator it;
                it = lookUp.find(first.l);
                if (it != lookUp.end())
                {
                    int numPoints = (*it).second;
                    //printf("Number of points in set is %d\n", numPoints);

                    // restrict number of points written
                    if (maxPointsPerCube != -1 && maxPointsPerCube < numPoints)
                    {
                        numPointsToWrite = maxPointsPerCube;
                    }
                    else
                    {
                        numPointsToWrite = numPoints;
                    }

                    // write size
                    file.write((char *)&(numPointsToWrite), sizeof(int));

                    // write position ID
                    for (int j = index; j < numPointsToWrite + index; j++)
                    {
                        file.write((char *)&(vec.at(j).ID), sizeof(uint32_t));
                    }
                    // increment offset
                    index = index + numPoints;
                }
            }
        }
    }
    file.close();

    cout << "Data Written!" << endl;
}

int main(int argc, char **argv)
{

    // TODO these values should be command line arguments
    int maxPointsPerCube = 100000000; // note set to -1 if no max points per cube is specified
    int divisionSize = 25;
    formatTypes format = FORMAT_IRGB;
    std::vector<Point> vec;
    std::map<int, int> lookUp;
    std::vector<ScannerPosition> positions;

    min_x = min_y = min_z = FLT_MAX;
    max_x = max_y = max_z = FLT_MIN;

    int nread = 0;
    intensityOnly=false;
    if (argc < 3) /* argc should be > 3 for correct execution */
    {
        printf("Minimal two params required. read README.txt\n");
    }
    else
    {
        for (int i = 1; i < argc - 1; i++)
        {
            if(argv[i][0] == '-')
            {
               if(argv[i][1] == 'i')
               {
                   intensityOnly=true;
               }

                if (argv[i][1] == 'd')
                {
                    ++i;
                    if (i >= argc)
                    {
                        printf("-d requires divisionSize argument\n");
                        continue;
                    }
                    divisionSize = atoi(argv[i]);
                    if (divisionSize == 0)
                    {
                        printf("-d divisionSize cannot be 0\n");
                        continue;
                        divisionSize = 25;
                    }
                }

                if (argv[i][1] == 'p')
                {
                    ++i;
                    if (i >= argc)
                    {
                        printf("-p requires maxPointsPerCube argument\n");
                        continue;
                    }
                    maxPointsPerCube = atoi(argv[i]);
                    if (maxPointsPerCube == 0)
                    {
                        printf("-p maxPointsPerCube cannot be 0\n");
                        continue;
                        maxPointsPerCube = 100000000;
                    }
                }
                if(argv[i][1] == 's')
                {
                    readScannerPositions=true;
                }

            }
            else
            {
                ++nread;
                printf("Reading %s\n", argv[i]);
                ReadData(argv[i], vec, format, positions);

				printf("min %f %f %f\n", min_x, min_y, min_z);
				printf("max %f %f %f\n", max_x, max_y, max_z);
            }
        }
        if (nread > 0)
        {
            printf("Sorting data\n");
            LabelData(divisionSize, vec, lookUp);
            printf("Persisting data\n");
            WriteData(argv[argc - 1], vec, lookUp, maxPointsPerCube, positions);
        }
        else
        {
            printf("Nothing read\n");
        }
    }
    return 0;
}
