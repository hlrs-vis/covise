/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <vector>
#include <algorithm>
#include <random>
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
bool bTestMode = false;
bool bShowHeatMap = false;
bool readScannerPositions = false;
uint32_t fileVersion=1;

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

struct PointCloudStatistics
{
    float min_x;
    float min_y;
    float min_z;

    float max_x;
    float max_y;
    float max_z;
    
    uint32_t psetMaxSize;
    uint32_t psetMinSize;

    uint64_t pcSize;
    
    PointCloudStatistics()
        : min_x(FLT_MAX), min_y(FLT_MAX), min_z(FLT_MAX),
          max_x(FLT_MIN), max_y(FLT_MIN), max_z(FLT_MIN),
          psetMaxSize(0), psetMinSize(UINT_MAX), pcSize(0)
        {
        }
};

// ----------------------------------------------------------------------------
// readE57()
// ----------------------------------------------------------------------------
void readE57(char *filename, std::vector<Point> &vec, formatTypes format,
             std::vector<ScannerPosition> &posVec,
             PointCloudStatistics &pcStats)
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
				double * rangeData = NULL;
				if (scanHeader.pointFields.sphericalRangeField)
					rangeData = new double[nSize];
				double * azData = NULL;
				if (scanHeader.pointFields.sphericalAzimuthField)
					azData = new double[nSize];
				double * elData = NULL;
				if (scanHeader.pointFields.sphericalElevationField)
					elData = new double[nSize];

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
					blueData,			//!< pointer to a buffer with the color blue data
											NULL, //sColorInvalid
											rangeData,
											azData,
											elData
											/*rowIndex,			//!< pointer to a buffer with the rowIndex
											columnIndex			//!< pointer to a buffer with the columnIndex*/
				);

				int64_t		count = 0;
				unsigned	size = 0;
				int			col = 0;
				int			row = 0;
				Point point;
				while ((size = dataReader.read()))
				{
					for (unsigned int i = 0; i < size; i++)
					{

						if (isInvalidData[i] == 0)
						{
						       if(xData)
						       {
							osg::Vec3 p(xData[i], yData[i], zData[i]);
							p = p * m;
							point.x = p[0];
							point.y = p[1];
							point.z = p[2];
							if (point.x < pcStats.min_x)
								pcStats.min_x = point.x;
							if (point.y < pcStats.min_y)
								pcStats.min_y = point.y;
							if (point.z < pcStats.min_z)
								pcStats.min_z = point.z;

							if (point.x > pcStats.max_x)
								pcStats.max_x = point.x;
							if (point.y > pcStats.max_y)
								pcStats.max_y = point.y;
							if (point.z > pcStats.max_z)
								pcStats.max_z = point.z;


							if (bIntensity) {		//Normalize intensity to 0 - 1.
								int intensity = ((intData[i] - intOffset) / intRange) * 255;
								point.l = intensity;
								point.rgba = intensity | intensity << 8 | intensity << 16;
							}


							if (bColor) {			//Normalize color to 0 - 255
							     if(!intensityOnly)
							     {
								int r = ((redData[i] - colorRedOffset)*255.0) / colorRedRange;
								int g = ((greenData[i] - colorGreenOffset)*255.0) / colorGreenRange;
								int b = ((blueData[i] - colorBlueOffset)*255.0) / colorBlueRange;
								point.rgba = r | g << 8 | b << 16;
							     }

							}
							vec.push_back(point);
							}
							else if(rangeData)
							{
							osg::Vec3 p(rangeData[i]*cos( elData[i])*cos(azData[i]), rangeData[i]*cos(elData[i])*sin(azData[i]),rangeData[i]*sin( elData[i]));
							p = p * m;
							point.x = p[0];
							point.y = p[1];
							point.z = p[2];
							if (point.x < pcStats.min_x)
								pcStats.min_x = point.x;
							if (point.y < pcStats.min_y)
								pcStats.min_y = point.y;
							if (point.z < pcStats.min_z)
								pcStats.min_z = point.z;

							if (point.x > pcStats.max_x)
								pcStats.max_x = point.x;
							if (point.y > pcStats.max_y)
								pcStats.max_y = point.y;
							if (point.z > pcStats.max_z)
								pcStats.max_z = point.z;


							if (bIntensity) {		//Normalize intensity to 0 - 1.
								int intensity = ((intData[i] - intOffset) / intRange) * 255;
								point.l = intensity;
								point.rgba = intensity | intensity << 8 | intensity << 16;
							}


							if (bColor) {			//Normalize color to 0 - 255
							     if(!intensityOnly)
							     {
								int r = ((redData[i] - colorRedOffset)*255.0) / colorRedRange;
								int g = ((greenData[i] - colorGreenOffset)*255.0) / colorGreenRange;
								int b = ((blueData[i] - colorBlueOffset)*255.0) / colorBlueRange;
								point.rgba = r | g << 8 | b << 16;
							     }

							}
							vec.push_back(point);
							}

						}
					}

				}

				dataReader.close();

				delete[] isInvalidData;
				delete[] xData;
				delete[] yData;
				delete[] zData;
				delete[] rangeData;
				delete[] azData;
				delete[] elData;
				delete[] intData;
				delete[] redData;
				delete[] greenData;
				delete[] blueData;
                                delete[] idElementValue;
                                delete[] startPointIndex;
                                delete[] pointCount;
                                delete[] rowIndex;
                                delete[] columnIndex;
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

// ----------------------------------------------------------------------------
// readPTSB()
// ----------------------------------------------------------------------------
int readPTSB(char *filename, std::vector<Point> &vec, formatTypes format,
             std::vector<ScannerPosition> &posVec,
             PointCloudStatistics &pcStats)
{
    cout << "reading ptsb point cloud from " << filename << endl;

    uint8_t result = 0;
    
    ifstream file(filename, ios::in | ios::binary);

    if (file.is_open())
    {
        size_t vecBegin = vec.size();
        uint32_t size;
        
        file.read((char *)&size, sizeof(uint32_t));

        cout << "    total num of sets    : " << size << endl;

        uint64_t count_avg = 0;
        vector<uint32_t> vecFindMedian;
        
        for (uint32_t i = 0; i < size; ++i)
        {
            unsigned int psize;

            // read size of point data in this set
            file.read((char *)&psize, sizeof(psize));
            size_t numP = psize;
            size_t numF = 3 * numP;

            count_avg += psize;
            vecFindMedian.push_back(psize);
            
            if (numP < pcStats.psetMinSize) pcStats.psetMinSize = numP;
            if (numP > pcStats.psetMaxSize) pcStats.psetMaxSize = numP;
            
            // read point coordinates
            float *coord = new float[numF];
            file.read((char *)(coord), (sizeof(float) * 3 * psize));

            // read point color data
            uint32_t *icolor = new uint32_t[psize];
            file.read((char *)(icolor), (sizeof(uint32_t) * psize));

            Point point;

            for (size_t j = 0; j < psize; ++j)
            {
                point.x = coord[j * 3];
                point.y = coord[j * 3 + 1];
                point.z = coord[j * 3 + 2];
                point.rgba = icolor[j];
                point.l = 0;

                vec.push_back(point);

                if (point.x < pcStats.min_x) pcStats.min_x = point.x;
                if (point.y < pcStats.min_y) pcStats.min_y = point.y;
                if (point.z < pcStats.min_z) pcStats.min_z = point.z;
                
                if (point.x > pcStats.max_x) pcStats.max_x = point.x;
                if (point.y > pcStats.max_y) pcStats.max_y = point.y;
                if (point.z > pcStats.max_z) pcStats.max_z = point.z;
            }

            delete[] coord;
            delete[] icolor;
        }

        std::sort(vecFindMedian.begin(), vecFindMedian.end(), [](const int a, const int b){ return a < b; });
        
        cout << "    set size avg         : " << count_avg / size << endl;
        cout << "    set size median      : " << vecFindMedian.at(vecFindMedian.size()/2) << endl;
        cout << "    smallest set size    : " << pcStats.psetMinSize << endl;
        cout << "    largest set size     : " << pcStats.psetMaxSize << endl;
        cout << "    min coord values       " << pcStats.min_x << " / " << pcStats.min_y << " / " << pcStats.min_z << endl;
        cout << "    max coord values       " << pcStats.max_x << " / " << pcStats.max_y << " / " << pcStats.max_z << endl;

        // read scanner positions and ids from file
        // and map the ids to the point set
        if (readScannerPositions)
        {
            uint32_t version;
            file.read((char *)&version,sizeof(uint32_t));
            
            cout << "    version: " << version << endl;

            // read size of scanner positions data set
            uint32_t numPositions;
            file.read((char *)&numPositions, sizeof(uint32_t));

            // read scanner positions
            for (int i = 0; i != numPositions; ++i)
            {
                ScannerPosition pos;

                file.read((char *)&pos.ID, sizeof(uint32_t));
                file.read((char *)&pos.point._v, sizeof(float) * 3);

                posVec.push_back(pos);

                cout << "    scanner position " << pos.ID << " x: " << pos.point.x() << " y: " << pos.point.y() << " z: " << pos.point.z() << endl;
            }
            
            // read num of sets
            file.read((char *)&size, sizeof(uint32_t));
            cout << "    total num of sets with scanner position: " << size << endl;

            for (uint32_t i = 0; i < size; ++i)
            {
                // read size of set
                unsigned int psize;
                file.read((char *)&psize, sizeof(psize));

                // read position ID data
                size_t numP = psize;
                uint32_t *IDs = new uint32_t[numP];
                file.read((char *)(IDs), (sizeof(uint32_t) * psize));

                // map IDs to points
                for (size_t j = 0; j < psize; j++)
                {
                    vec.at(j+vecBegin).ID = IDs[j];
                }
                
                delete[] IDs;
            }
        }
        
        file.close();
    }
    else
    {
        cout << "error: could not open file" << endl;
        result = -1;
    }
    
    return result;
}

// ----------------------------------------------------------------------------
// labelData()
//! computes a grid structure and maps the points to the structure, then sorts
//! the points in vector<Point> according to their positions within the grid
// ----------------------------------------------------------------------------
void labelData(int grid, std::vector<Point> &vec, std::map<int, int> &lookUp,
               PointCloudStatistics &pcStats)
{
    cout << "sorting point cloud" << endl;
    
    int xl;
    int yl;
    int zl;

    // init: number of grid elements per axis
    int xs = grid;
    int ys = grid;
    int zs = grid;

    // init: length of grid element in x,y,z direction with non isometric grid
    float xsize = (pcStats.max_x - pcStats.min_x) / grid;
    float ysize = (pcStats.max_y - pcStats.min_y) / grid;
    float zsize = (pcStats.max_z - pcStats.min_z) / grid;

    // compute proportional grid sizes, dependent on smallest grid element length 
    if (xsize <= ysize && xsize <= zsize)
    {
        xs = grid;
        ys = (int)((pcStats.max_y - pcStats.min_y) / xsize);
        ysize = (pcStats.max_y - pcStats.min_y) / ys;
        zs = (int)((pcStats.max_z - pcStats.min_z) / xsize);
        zsize = (pcStats.max_z - pcStats.min_z) / zs;
    }
    else if (ysize <= xsize && ysize <= zsize)
    {
        ys = grid;
        xs = (int)((pcStats.max_x - pcStats.min_x) / ysize);
        xsize = (pcStats.max_x - pcStats.min_x) / xs;
        zs = (int)((pcStats.max_z - pcStats.min_z) / ysize);
        zsize = (pcStats.max_z - pcStats.min_z) / zs;
    }
    else
    {
        zs = grid;
        ys = (int)((pcStats.max_y - pcStats.min_y) / zsize);
        ysize = (pcStats.max_y - pcStats.min_y) / ys;
        xs = (int)((pcStats.max_x - pcStats.min_x) / zsize);
        xsize = (pcStats.max_x - pcStats.min_x) / xs;
    }

    pcStats.pcSize = vec.size();
    
    cout << "    number of points     : " << pcStats.pcSize << endl;
    cout << "    size of grid element   " << xsize << " " << ysize << " " << zsize << endl;
    cout << "    number of grid elements " << xs << " x " << ys << " x " << zs << " = " << xs * ys * zs << endl;

    // find corresponding grid element for each point
    std::map<int, int>::iterator it;

    for (int i = 0; i < vec.size(); ++i)
    {
        // compute grid number
        xl = (int)((vec.at(i).x - pcStats.min_x) / xsize);
        yl = (int)((vec.at(i).y - pcStats.min_y) / ysize);
        zl = (int)((vec.at(i).z - pcStats.min_z) / zsize);
        
        if (xl == xs)
            xl--;
        if (yl == ys)
            yl--;
        if (zl == zs)
            zl--;

        // store grid element serialized label with point
        vec.at(i).l = xl + (yl * xs) + (zl * xs * ys);

        // count points per grid element for sorting
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

    cout << "    number of sets with points: " << lookUp.size() << "   ( "<< int((((float)lookUp.size() / (xs * ys * zs)) * 100) + 0.5) << "% )" << endl;

    uint64_t count_avg = 0;
    vector<uint32_t> vecFindMedian;
    uint32_t setMaxSize = 0;
    uint32_t setMinSize = UINT_MAX;

    for (auto const& it : lookUp)
    {
        if (it.second > setMaxSize) setMaxSize = it.second;
        if (it.second < setMinSize) setMinSize = it.second;
        
        count_avg += it.second;
        vecFindMedian.push_back(it.second);
    }

    std::sort(vecFindMedian.begin(), vecFindMedian.end(), [](const int a, const int b){ return a < b; });
        
    cout << "    element size avg     : " << count_avg / vecFindMedian.size() << endl;
    cout << "    element size median  : " << vecFindMedian.at(vecFindMedian.size()/2) << endl;
    cout << "    smallest element size: " << setMinSize << endl;
    cout << "    largest element size : " << setMaxSize << endl;
     
    // randomize all the data
    // needed to distribute points in grid cell equally in vector<> for filtering
    //ald::random_shuffle(vec.begin(), vec.end());
    std::shuffle(vec.begin(), vec.end(), std::default_random_engine(1));

    // sort data
    //std::sort(vec.begin(), vec.end(), sortfunction);
    //bool sortfunction(Point i, Point j) { return (i.l < j.l); };
    alg::sort(vec.begin(), vec.end(), [](const Point i, const Point j){ return (i.l < j.l); });
}

// ----------------------------------------------------------------------------
// writeData()
// ----------------------------------------------------------------------------
void writeData(char *filename, std::vector<Point> &vec, std::map<int, int> &lookUp,
               int maxPointsPerCube, std::vector<ScannerPosition> scanPositions,
               PointCloudStatistics &pcStats)
{
    cout << "writing point cloud data to " << filename << endl;

    ofstream file(filename, ios::out | ios::binary | ios::ate);
    uint32_t numPointsToWrite;

    if (file.is_open())
    {
        int number_of_sets = (int)lookUp.size();
        int index = 0;

        uint64_t count_avg = 0;
        uint64_t count_removed = 0;
        vector<uint32_t> vecFindMedian;
        uint32_t setMaxSize = 0;
        uint32_t setMinSize = UINT_MAX;

        int r = 55;
        int g = 55;
        int b = 55;
        std::srand(std::time(nullptr));
        int maxHeatmapValue = 0;

        // we need to know the maxPointsPerCube in advance
        if ((bTestMode == true) && (bShowHeatMap == true))
        {
            for (int i = 0; i < number_of_sets; ++i)
            {
                // get first in set to find set number
                Point first = vec.at(index);
                
                std::map<int, int>::iterator it;
                it = lookUp.find(first.l);
                if (it != lookUp.end())
                {
                    int numPoints = (*it).second;
                    
                    // restrict number of points written
                    if (maxPointsPerCube != -1 && maxPointsPerCube < numPoints)
                    {
                        numPointsToWrite = maxPointsPerCube;
                        count_removed += numPoints - maxPointsPerCube;
                    }
                    else
                    {
                        numPointsToWrite = numPoints;
                    }

                    if (numPointsToWrite > maxHeatmapValue) maxHeatmapValue = numPointsToWrite;

                    index = index + numPoints;
                }
            }
        }

        index = 0;
        
        // write the number of sets
        file.write((char *)&(number_of_sets), sizeof(int));

        for (int i = 0; i < number_of_sets; ++i)
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
                    count_removed += numPoints - maxPointsPerCube;
                }
                else
                {
                    numPointsToWrite = numPoints;
                }

                count_avg += numPointsToWrite;
                vecFindMedian.push_back(numPointsToWrite);
                if (numPointsToWrite > setMaxSize) setMaxSize = numPointsToWrite;
                if (numPointsToWrite < setMinSize) setMinSize = numPointsToWrite;
                        
                // write size
                file.write((char *)&(numPointsToWrite), sizeof(int));

                // write points
                for (int j = index; j < numPointsToWrite + index; ++j)
                {
                    file.write((char *)&(vec.at(j).x), sizeof(float) * 3);
                }

                if (bTestMode == true)
                {
                    if (bShowHeatMap == true)
                    {
                        if (maxHeatmapValue >= 0)
                        {
                            r =  (numPointsToWrite * 255 / maxHeatmapValue) ;
                        }
                        else
                        {
                            r = 0;
                        }
                    }
                    else
                    {
                        g = rand() % 200 + 55;
                    }
                }
               
                // write colors
                for (int j = index; j < numPointsToWrite + index; ++j)
                {
                    if (bTestMode == true)
                    {
                        vec.at(j).rgba = r | g << 8 | b << 16;
                    }

                    file.write((char *)&(vec.at(j).rgba), sizeof(uint32_t));
                }

                // increment offset
                index = index + numPoints;
            }
            else
            {
                cout << "error: sorting went wrong" << endl;
            }
        }

        std::sort(vecFindMedian.begin(), vecFindMedian.end(), [](const int a, const int b){ return a < b; });
        
        cout << "    element size avg     : " << count_avg / vecFindMedian.size() << endl;
        cout << "    element size median  : " << vecFindMedian.at(vecFindMedian.size()/2) << endl;
        cout << "    smallest element size: " << setMinSize << endl;
        cout << "    largest element size : " << setMaxSize << endl;
        cout << "    points removed       : " << count_removed << "   ( "<< int(((float)count_removed / pcStats.pcSize * 100) + 0.5) << "% )" << endl;    
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
        file.close();
        cout << "    data written" << endl;
    }
    else
    {
        cout << "error: could not open output file" << endl;
    }
}

// ----------------------------------------------------------------------------
// printHelpPage()
// ----------------------------------------------------------------------------
void printHelpPage()
{
    cout << endl;
    cout << "PointSort - rearrange and filter point cloud data" << endl;
    cout << endl;
    cout << "Usage: PointSort [options ...] inputfile outputfile" << endl;
    cout << endl;
    cout << "options" << endl;
    cout << "  -d              divisionsize (default: 10)" << endl;
    cout << "  -p              max points per cube (default: 100000000)" << endl;
    cout << "                  note: set to -1 if no max points per cube is specified" << endl;
    cout << "  -i              use intensity only" << endl;
    cout << "  -s              read scanner position" << endl;
    cout << "  -t              test mode to examine point cloud structure" << endl;
    cout << "  -h              heat map representation of point density (only in test mode)" << endl; 
    cout << endl;
    cout << "examples" << endl;
    cout << "  PointSort -t -h -d 10 -p 20000 input.ptsb output_sorted.ptsb" << endl;
    cout << endl;
}

// ----------------------------------------------------------------------------
// main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int maxPointsPerCube = 100000000; 
    int divisionSize = 10;
    formatTypes format = FORMAT_IRGB;
    
    std::vector<Point> vec;
    std::map<int, int> lookUp;
    std::vector<ScannerPosition> positions;

    int nread = 0;
    intensityOnly=false;
    bTestMode = false;
    bShowHeatMap = false;
    
    if (argc < 3) /* argc should be >= 3 for correct execution */
    {
        printf("error: minimal two params required\n");
        printHelpPage();
        
        return -1;
    }
    else
    {
        PointCloudStatistics pcStats;
        
        for (int i = 1; i < argc - 1; i++)
        {
            if(argv[i][0] == '-')
            {
                if(argv[i][1] == 'i')
                {
                    intensityOnly=true;
                }

                if(argv[i][1] == 't')
                {
                    bTestMode = true;
                    cout << "WARNING: point cloud output will be modified!" << endl << endl;
                }

                if(argv[i][1] == 'h')
                {
                    bShowHeatMap = true;
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
                        divisionSize = 10;
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
                
                int len = strlen(argv[i]);
                if ((len > 4) && strcasecmp((argv[i] + len - 4), ".e57") == 0)
                {
                    readE57(argv[i], vec, format, positions, pcStats);
                }
                else
                {
                    readPTSB(argv[i], vec, format, positions, pcStats);
                }
                
            }            
        }
        
        if (nread > 0)
        {
            labelData(divisionSize, vec, lookUp, pcStats);
            writeData(argv[argc - 1], vec, lookUp, maxPointsPerCube, positions, pcStats);
        }
        else
        {
            printf("Nothing read\n");
        }
        
        return 0;
    }
    return 0;
}

// ----------------------------------------------------------------------------
