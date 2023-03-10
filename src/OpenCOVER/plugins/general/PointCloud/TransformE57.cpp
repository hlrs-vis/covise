
#include <osg/Matrix>

#ifdef HAVE_E57
#include <e57/E57Foundation.h>
#include <e57/E57Simple.h>
#endif

int main(int argc, char** argv)
{
	if (argc == 3)
	{
		osg::Matrix offsetMat;
		offsetMat.makeIdentity();
		std::string inFileName = argv[1];
		std::string outFileName = argv[2];
		try
		{
			e57::Reader eReader(inFileName);
			e57::E57Root rootHeader;
			eReader.GetE57Root(rootHeader);
			e57::Data3D	scanHeader;
			e57::Writer eWriter(outFileName, rootHeader.coordinateMetadata);
			int data3DCount = 0;
			data3DCount = eReader.GetData3DCount();
			for (int scanIndex = 0; scanIndex < data3DCount; scanIndex++)
			{
				eReader.ReadData3D(scanIndex, scanHeader);
				scanHeader.originalGuids.clear();
				osg::Matrix cloudMat;
				osg::Matrix trans;
				trans.makeTranslate(scanHeader.pose.translation.x, scanHeader.pose.translation.y, scanHeader.pose.translation.z);
				osg::Matrix rot;
				rot.makeRotate(osg::Quat(scanHeader.pose.rotation.x, scanHeader.pose.rotation.y, scanHeader.pose.rotation.z, scanHeader.pose.rotation.w));
				cloudMat = rot * trans;
				cloudMat = cloudMat * offsetMat;
				scanHeader.pose.translation.x = 0;
				scanHeader.pose.translation.y = 0;
				scanHeader.pose.translation.z = 0;
				osg::Quat q;
				scanHeader.pose.rotation.x = q.x();
				scanHeader.pose.rotation.y = q.y();
				scanHeader.pose.rotation.z = q.z();
				scanHeader.pose.rotation.w = q.w();
				int64_t nColumn = 0;
				int64_t nRow = 0;
				int64_t nPointsSize = 0;	//Number of points
				int64_t nGroupsSize = 0;	//Number of groups
				int64_t nCountSize = 0;		//Number of points per group
				int64_t nSize = 0;
				int8_t* isInvalidData = NULL;
				bool	bColumnIndex = false; //indicates that idElementName is "columnIndex"
				eReader.GetData3DSizes(scanIndex, nRow, nColumn, nPointsSize, nGroupsSize, nCountSize, bColumnIndex);
				nSize = nRow;
				if (nSize == 0)
				{
					nSize = 1024;	// choose a chunk size
				}
				isInvalidData = new int8_t[nSize];
				if (!scanHeader.pointFields.cartesianInvalidStateField)
				{
					for (int i = 0; i < nSize; i++)
					{
						isInvalidData[i] = 0;
					}
				}
				double* xData = NULL;
				if (scanHeader.pointFields.cartesianXField)
				{
					xData = new double[nSize];
				}
				double* yData = NULL;
				if (scanHeader.pointFields.cartesianYField)
				{
					yData = new double[nSize];
				}
				double* zData = NULL;
				if (scanHeader.pointFields.cartesianZField)
				{
					zData = new double[nSize];
				}
				double* intData = NULL;
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
				uint16_t* redData = NULL;
				uint16_t* greenData = NULL;
				uint16_t* blueData = NULL;
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
				int64_t* idElementValue = NULL;
				int64_t* startPointIndex = NULL;
				int64_t* pointCount = NULL;
				int			row = 0;
				if (nGroupsSize > 0)
				{
					idElementValue = new int64_t[nGroupsSize];
					startPointIndex = new int64_t[nGroupsSize];
					pointCount = new int64_t[nGroupsSize];
					if (!eReader.ReadData3DGroupsData(scanIndex, nGroupsSize, idElementValue,
						startPointIndex, pointCount))
					{
						nGroupsSize = 0;
					}
				}
				int32_t* rowIndex = NULL;
				int32_t* columnIndex = NULL;
				int64_t		count = 0;
				unsigned	size = 0;
				if (scanHeader.pointFields.rowIndexField)
				{
					rowIndex = new int32_t[nSize];
				}
				if (scanHeader.pointFields.columnIndexField)
				{
					columnIndex = new int32_t[nRow];
				}
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
				scanHeader.pointFields.rowIndexField = false;
				scanHeader.pointFields.columnIndexField = false;
				//scanHeader.pointFields.cartesianInvalidStateField = false;
				//scanHeader.pointFields.sphericalInvalidStateField = false;
				//scanHeader.pointFields.cartesianXField = false;
				//scanHeader.pointFields.cartesianYField = false;
				//scanHeader.pointFields.cartesianZField = false;
				//scanHeader.cartesianBounds.xMinimum = cloudMat.preMult(Vec3(scanHeader.cartesianBounds.xMinimum, 0, 0)).x();
				//scanHeader.cartesianBounds.xMaximum = cloudMat.preMult(Vec3(scanHeader.cartesianBounds.xMaximum, 0, 0)).x();
				//scanHeader.cartesianBounds.yMinimum = cloudMat.preMult(Vec3(0, scanHeader.cartesianBounds.yMinimum, 0)).y();
				//scanHeader.cartesianBounds.yMaximum = cloudMat.preMult(Vec3(0, scanHeader.cartesianBounds.yMaximum, 0)).y();
				//scanHeader.cartesianBounds.zMinimum = cloudMat.preMult(Vec3(0, 0, scanHeader.cartesianBounds.zMinimum)).z();
				//scanHeader.cartesianBounds.zMaximum = cloudMat.preMult(Vec3(0, 0, scanHeader.cartesianBounds.zMaximum)).z();


				int scanIndex2 = eWriter.NewData3D(scanHeader);

				double* xData2 = new double[nSize];
				double* yData2 = new double[nSize];
				double* zData2 = new double[nSize];
				double* intData2 = intData;
				uint16_t* redData2 = new uint16_t[nSize];
				uint16_t* greenData2 = new uint16_t[nSize];
				uint16_t* blueData2 = new uint16_t[nSize];
				for (size_t i = 0; i < nGroupsSize; i++)
				{
					if (startPointIndex[i] >= nPointsSize)
					{
						startPointIndex[i] = nPointsSize - 1;
					}
					if (pointCount[i] >= nPointsSize)
					{
						pointCount[i] = nPointsSize - 1;
					}
				}
				eWriter.WriteData3DGroupsData(scanIndex2, nGroupsSize, idElementValue, startPointIndex, pointCount);
				e57::CompressedVectorWriter dataWriter = eWriter.SetUpData3DPointsData(
					scanIndex2,
					nSize,
					xData2,
					yData2,
					zData2,
					isInvalidData,
					intData2,
					NULL,
					redData2,			//!< pointer to a buffer with the color red data
					greenData2,			//!< pointer to a buffer with the color green data
					blueData2,
					NULL,
					NULL,
					NULL,
					NULL,
					NULL,
					NULL,
					NULL,
					NULL,
					NULL,
					NULL,
					NULL
				);
				//::Point point;
				//Color color;
				while ((size = dataReader.read()) != 0u)
				{
					for (unsigned int i = 0; i < size; i++)
					{
						if (isInvalidData[i] == 0 && (xData[i] != 0.0 && yData[i] != 0.0 && zData[i] != 0.0))
						{
							osg::Vec3d p(xData[i], yData[i], zData[i]);
							p = p * cloudMat;
							
							xData2[i] = p.x();
							yData2[i] = p.y();
							zData2[i] = p.z();

							if (bIntensity)
							{	//Normalize intensity to 0 - 1.
								intData2[i] = intData[i];
							}
							if (bColor)
							{	//Normalize color to 0 - 1
								redData2[i] = redData[i];
								greenData2[i] = greenData[i];
								blueData2[i] = blueData[i];
							}
						}
					}
					dataWriter.write(nSize);
				}

				dataWriter.close();
				dataReader.close();
				delete isInvalidData;
				delete xData;
				delete yData;
				delete zData;
				delete intData;
				delete redData;
				delete greenData;
				delete blueData;
				delete[] xData2;
				delete[] yData2;
				delete[] zData2;
			}
			eReader.Close();
			eReader.Close();
			return 0;
		}
		catch (e57::E57Exception& ex) {
			ex.report(__FILE__, __LINE__, __FUNCTION__);
			return -1;
		}
	}
	else
	{
	    fprintf(stderr,"usage: TransformE57 input.e57 output.e57\n");
    }
	return 0;
}
