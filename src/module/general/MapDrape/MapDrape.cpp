/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <util/coviseCompat.h>
#include <do/coDoAbstractStructuredGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoSet.h>
#include <api/coFeedback.h>
#include <algorithm>
#include "MapDrape.h"
#include <vector>
#include <string>
#include <functional>

MapDrape::MapDrape(int argc, char *argv[])
	: coSimpleModule(argc, argv, "map coordinates and drape to height field")
{


	p_mapping_from_ = addStringParam("from",
		"proj4 mapping from");
	p_mapping_from_->setValue("+proj=latlong +datum=WGS84");

	p_mapping_to_ = addStringParam("to",
		"proj4 mapping to");

	p_offset_ = addFloatVectorParam("offset",
		"offset");
	p_heightfield_ = addFileBrowserParam("heightfield", "height field as geotif");
	p_heightfield_->setValue("/tmp/foo.tif", "*.tif");

	p_geo_in_ = addInputPort("geo_in", "Polygons|TriangleStrips|Points|Lines|UnstructuredGrid|UniformGrid|RectilinearGrid|StructuredGrid", "polygon/grid input");
	p_geo_out_ = addOutputPort("geo_out", "Polygons|TriangleStrips|Points|Lines|UnstructuredGrid|UniformGrid|RectilinearGrid|StructuredGrid", "polygon/grid output");
#ifdef WIN32
	char *pValue;
	size_t len;
	errno_t err = _dupenv_s(&pValue, &len, "ODDLOTDIR");
	if (err || pValue == NULL || strlen(pValue) == 0)
		err = _dupenv_s(&pValue, &len, "COVISEDIR");
	if (err)
		pValue = "";
	std::string covisedir = pValue;
#else

	char *pValue; = getenv("ODDLOTDIR");
	if (pValue[0] == '\0')
		pValue = getenv("COVISEDIR");
	std::string covisedir = pValue;
#endif
	dir = covisedir + "/share/covise/";

	std::string proj = "+proj=tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=3500000 +y_0=0 +ellps=bessel +datum=potsdam +nadgrids=" + dir + std::string("BETA2007.gsb");
	p_mapping_to_->setValue(proj.c_str());
	// setup GDAL
	GDALAllRegister();
}


int
MapDrape::compute(const char *)
{

	const char *imageName = p_heightfield_->getValue();
	if (imageName)
	{
		std::string name = imageName;
		openImage(name);
	}
	p_offset_->getValue(Offset[0], Offset[1], Offset[2]);

	if (!(pj_from = pj_init_plus(p_mapping_from_->getValue())))
	{
		closeImage();
		std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection source: " << p_mapping_from_->getValue() << std::endl;
		return STOP_PIPELINE;
	}
	if (!(pj_to = pj_init_plus(p_mapping_to_->getValue())))
	{
		closeImage();
		std::cerr << "RoadSystem::parseIntermapRoad(): couldn't initalize projection target: " << p_mapping_to_->getValue() << std::endl;
		return STOP_PIPELINE;
	}

    // output geometry
    std::string outGeoName(p_geo_out_->getObjName());

	
	const coDistributedObject *out_geo = NULL;
	if (p_geo_in_->getCurrentObject()->isType("POLYGN"))
	{
		coDoPolygons *polygons = (coDoPolygons *)(p_geo_in_->getCurrentObject());

		// get dimensions
		int nPoints = polygons->getNumPoints();
		int nCorners = polygons->getNumVertices();
		int nPolygons = polygons->getNumPolygons();

		// create new arrays
		int *cl = NULL, *pl = NULL;
		float *coords[3];
		for (int i = 0; i < 3; ++i)
			coords[i] = NULL;

		polygons->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

		// create new DO
		coDoPolygons *outObj = new coDoPolygons(outGeoName.c_str(), nPoints, coords[0], coords[1], coords[2],
			nCorners, cl, nPolygons, pl);
		if (!outObj->objectOk())
		{
			closeImage();
			return STOP_PIPELINE;
		}
		out_geo = outObj;

		float *outCoords[3];
		int *outCl, *outPl;
		outObj->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2], &outCl, &outPl);

		transformCoordinates(nPoints, coords[0], coords[1], coords[2], outCoords[0], outCoords[1], outCoords[2]);
		memcpy(outCl, cl, nCorners * sizeof(int));
		memcpy(outPl, pl, nPolygons * sizeof(int));
	}
	else if (p_geo_in_->getCurrentObject()->isType("LINES"))
	{
		//      cerr << "GetSetElem::createNewSimpleObj(..) POLYGN" << endl;
		coDoLines *lines = (coDoLines *)(p_geo_in_->getCurrentObject());

		// get dimensions
		int nPoints = lines->getNumPoints();
		int nCorners = lines->getNumVertices();
		int nLines = lines->getNumLines();

		// create new arrays
		int *cl = NULL, *pl = NULL;
		float *coords[3];
		for (int i = 0; i < 3; ++i)
			coords[i] = NULL;

		lines->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

		// create new DO
		coDoLines *outObj =  new coDoLines(outGeoName.c_str(), nPoints, coords[0], coords[1], coords[2],
			nCorners, cl, nLines, pl);
		out_geo = outObj;
		if (!outObj->objectOk())
		{
			closeImage();
			return STOP_PIPELINE;
		}
		// now use the matrix to apply the transformation
		float *outCoords[3];
		int *outCl, *outPl;
		outObj->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2], &outCl, &outPl);

		transformCoordinates(nPoints, coords[0], coords[1], coords[2], outCoords[0], outCoords[1], outCoords[2]);
		memcpy(outCl, cl, nCorners * sizeof(int));
		memcpy(outPl, pl, nLines * sizeof(int));
	}
	else if (p_geo_in_->getCurrentObject()->isType("POINTS"))
	{
		coDoPoints *lines = (coDoPoints *)(p_geo_in_->getCurrentObject());

		// get dimensions
		int nPoints = lines->getNumPoints();

		// create new arrays
		float *coords[3];
		for (int i = 0; i < 3; ++i)
			coords[i] = NULL;

		lines->getAddresses(&coords[0], &coords[1], &coords[2]);

		// create new DO
		coDoPoints *outObj = new coDoPoints(outGeoName.c_str(), nPoints, coords[0], coords[1], coords[2]);
		out_geo = outObj;
		if (!outObj->objectOk())
		{
			closeImage();
			return STOP_PIPELINE;
		}
		// now use the matrix to apply the transformation
		float *outCoords[3];
		outObj->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2]);

		transformCoordinates(nPoints, coords[0], coords[1], coords[2], outCoords[0], outCoords[1], outCoords[2]);
	}
	else if (p_geo_in_->getCurrentObject()->isType("TRIANG"))
	{
		coDoTriangleStrips *polygons = (coDoTriangleStrips *)(p_geo_in_->getCurrentObject());

		// get dimensions
		int nPoints = polygons->getNumPoints();
		int nCorners = polygons->getNumVertices();
		int nPolygons = polygons->getNumStrips();

		// create new arrays
		int *cl = NULL, *pl = NULL;
		float *coords[3];
		for (int i = 0; i < 3; ++i)
			coords[i] = NULL;

		polygons->getAddresses(&coords[0], &coords[1], &coords[2], &cl, &pl);

		// create new DO
		coDoTriangleStrips * outObj = new coDoTriangleStrips(outGeoName.c_str(), nPoints,
			coords[0], coords[1], coords[2],
			nCorners, cl, nPolygons, pl);
		out_geo = outObj;
		if (!outObj->objectOk())
		{
			closeImage();
			return STOP_PIPELINE;
		}
		// now use the matrix to apply the transformation
		float *outCoords[3];
		int *outCl, *outPl;
		outObj->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2], &outCl, &outPl);

		transformCoordinates(nPoints, coords[0], coords[1], coords[2], outCoords[0], outCoords[1], outCoords[2]);
		memcpy(outCl, cl, nCorners * sizeof(int));
		memcpy(outPl, pl, nPolygons * sizeof(int));

	}
	else if (p_geo_in_->getCurrentObject()->isType("UNSGRD"))
	{
		coDoUnstructuredGrid *unsgrid = (coDoUnstructuredGrid *)(p_geo_in_->getCurrentObject());

		// get dimensions
		int nPoints;
		int nCorners;
		int nElems;
		unsgrid->getGridSize(&nElems, &nCorners, &nPoints);

		// create new arrays
		int *cl, *pl, *tl;
		pl = NULL;
		cl = NULL;
		float *coords[3];
		int i;
		for (i = 0; i < 3; ++i)
			coords[i] = NULL;

		unsgrid->getAddresses(&pl, &cl, &coords[0], &coords[1], &coords[2]);
		unsgrid->getTypeList(&tl);

		// create new DO
		coDoUnstructuredGrid * outObj = new coDoUnstructuredGrid(outGeoName.c_str(), nElems, nCorners, nPoints,
			pl, cl, coords[0], coords[1], coords[2], tl);
		out_geo = outObj;
		if (!outObj->objectOk())
		{
			closeImage();
			return STOP_PIPELINE;
		}
		// now use the matrix to apply the transformation
		float *outCoords[3];
		int *outCl, *outPl;
		outObj->getAddresses(&outPl, &outCl, &outCoords[0], &outCoords[1], &outCoords[2]);

		transformCoordinates(nPoints, coords[0], coords[1], coords[2], outCoords[0], outCoords[1], outCoords[2]);
		memcpy(outCl, cl, nCorners * sizeof(int));
		memcpy(outPl, pl, nElems * sizeof(int));
	}
	else if (p_geo_in_->getCurrentObject()->isType("STRGRD"))
	{
		coDoStructuredGrid *strgrid = (coDoStructuredGrid *)(p_geo_in_->getCurrentObject());
		int sx, sy, sz;
		strgrid->getGridSize(&sx, &sy, &sz);
		float *xc, *yc, *zc;
		strgrid->getAddresses(&xc, &yc, &zc);
		coDoStructuredGrid *outObj = new coDoStructuredGrid(outGeoName.c_str(), sx, sy, sz, xc, yc, zc);
		if (!outObj->objectOk())
		{
			closeImage();
			return STOP_PIPELINE;
		}
		out_geo = outObj;
		float *outCoords[3];
		((coDoStructuredGrid *)(outObj))->getAddresses(&outCoords[0], &outCoords[1], &outCoords[2]);
		transformCoordinates(sx*sy*sz, xc, yc, zc, outCoords[0], outCoords[1], outCoords[2]);
	}
	closeImage();
	if (out_geo == NULL)
	{
		sendError("Could not create output geometry");
		return STOP_PIPELINE;
	}
	p_geo_out_->setCurrentObject(const_cast<coDistributedObject *>(out_geo));

	return SUCCESS;
}

void MapDrape::transformCoordinates(int numCoords, float *xIn, float *yIn, float *zIn, float *xOut, float *yOut, float *zOut)
{
	for (int i = 0; i < numCoords; i++)
	{

		double x = xIn[i] * DEG_TO_RAD;
		double y = yIn[i] * DEG_TO_RAD;
		double z = zIn[i];
		if (heightDataset)
			z = getAlt(xIn[i], yIn[i]);
		pj_transform(pj_from, pj_to, 1, 1, &x, &y, &z);

		xOut[i] = x + Offset[0];
		yOut[i] = y + Offset[1];
		zOut[i] = z + Offset[2];
	}
}

void MapDrape::closeImage()
{
	if (heightDataset)
		GDALClose(heightDataset);
}
void MapDrape::openImage(std::string &name)
{

	heightDataset = (GDALDataset *)GDALOpen(name.c_str(), GA_ReadOnly);
	if (heightDataset != NULL)
	{
		int             nBlockXSize, nBlockYSize;
		int             bGotMin, bGotMax;
		double          adfMinMax[2];
		double        adfGeoTransform[6];

		printf( "Size is %dx%dx%d\n", heightDataset->GetRasterXSize(), heightDataset->GetRasterYSize(), heightDataset->GetRasterCount() );
		if (heightDataset->GetGeoTransform(adfGeoTransform) == CE_None)
		{
			printf("Origin = (%.6f,%.6f)\n",
				adfGeoTransform[0], adfGeoTransform[3]);
			printf("Pixel Size = (%.6f,%.6f)\n",
				adfGeoTransform[1], adfGeoTransform[5]);
		}

		heightBand = heightDataset->GetRasterBand(1);
		heightBand->GetBlockSize(&nBlockXSize, &nBlockYSize);

		adfMinMax[0] = heightBand->GetMinimum(&bGotMin);
		adfMinMax[1] = heightBand->GetMaximum(&bGotMax);
		if (!(bGotMin && bGotMax))
			GDALComputeRasterMinMax((GDALRasterBandH)heightBand, TRUE, adfMinMax);

		printf("Min=%.3fd, Max=%.3f\n", adfMinMax[0], adfMinMax[1]);

	}
}

float MapDrape::getAlt(double x, double y)
{
	float *pafScanline;
	int   nXSize = heightBand->GetXSize();

	pafScanline = new float[nXSize];
	heightBand->RasterIO(GF_Read, x, y, 1, 1,
		pafScanline, nXSize, 1, GDT_Float32,
		0, 0);
	float height = pafScanline[0];
	delete[] pafScanline;

	return height;
}
MODULE_MAIN(Tools, MapDrape)
