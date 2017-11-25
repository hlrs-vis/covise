/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- OpenSceneGraph - Copyright (C) 1998-2009 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

#include <vpb/SourceData>
#include <vpb/Destination>
#include <vpb/DataSet>
#include <vpb/System>

#include <osg/Notify>
#include <osg/io_utils>
#include <osgDB/ReadFile>
#include <osgDB/FileNameUtils>

#include <gdal_priv.h>
#include <gdalwarper.h>
#include <stdio.h>
#include <VehicleUtil/RoadSystem/RoadSystem.h>

using namespace vpb;

struct ValidValueOperator
{
    ValidValueOperator(GDALRasterBand *band)
        : defaultValue(0.0f)
        , noDataValue(-32767.0f)
        , minValue(-32000.0f)
        , maxValue(FLT_MAX)
    {
        if (band)
        {
            int success = 0;
            float value = band->GetNoDataValue(&success);
            if (success)
            {
                noDataValue = value;
            }
        }
    }

    inline bool isNoDataValue(float value)
    {
        if (noDataValue == value)
            return true;
        if (value < minValue)
            return true;
        return (value > maxValue);
    }

    inline float getValidValue(float value)
    {
        if (isNoDataValue(value))
            return value;
        else
            return defaultValue;
    }

    float defaultValue;
    float noDataValue;
    float minValue;
    float maxValue;
};

SourceData::~SourceData()
{
}

float SourceData::getInterpolatedValue(osg::HeightField *hf, double x, double y)
{
    double r, c;
    c = (x - hf->getOrigin().x()) / hf->getXInterval();
    r = (y - hf->getOrigin().y()) / hf->getYInterval();

    int rowMin = osg::maximum((int)floor(r), 0);
    int rowMax = osg::maximum(osg::minimum((int)ceil(r), (int)(hf->getNumRows() - 1)), 0);
    int colMin = osg::maximum((int)floor(c), 0);
    int colMax = osg::maximum(osg::minimum((int)ceil(c), (int)(hf->getNumColumns() - 1)), 0);

    if (rowMin > rowMax)
        rowMin = rowMax;
    if (colMin > colMax)
        colMin = colMax;

    float urHeight = hf->getHeight(colMax, rowMax);
    float llHeight = hf->getHeight(colMin, rowMin);
    float ulHeight = hf->getHeight(colMin, rowMax);
    float lrHeight = hf->getHeight(colMax, rowMin);

    double x_rem = c - (int)c;
    double y_rem = r - (int)r;

    double w00 = (1.0 - y_rem) * (1.0 - x_rem) * (double)llHeight;
    double w01 = (1.0 - y_rem) * x_rem * (double)lrHeight;
    double w10 = y_rem * (1.0 - x_rem) * (double)ulHeight;
    double w11 = y_rem * x_rem * (double)urHeight;

    float result = (float)(w00 + w01 + w10 + w11);

    return result;
}

float SourceData::getInterpolatedValue(GDALRasterBand *band, double x, double y, float originalHeight)
{
    if (RoadSystem::Instance())
    {
        
    double xr = x - 3493000;
    double yr = y - 5412700;
const RoadSystemHeader& header = RoadSystem::Instance()->getHeader();
        xr = x - header.xoffset;
	yr = y - header.yoffset;
        Vector2D v_c(std::numeric_limits<float>::signaling_NaN(), std::numeric_limits<float>::signaling_NaN());
        static Road *currentRoad = NULL;
        static double currentLongPos = -1;
        Vector3D v_w(xr, yr, 0);
        if (currentRoad)
        {
            v_c = RoadSystem::Instance()->searchPositionFollowingRoad(v_w, currentRoad, currentLongPos);
            if (!v_c.isNaV())
            {
                if (currentRoad->isOnRoad(v_c))
                {
                    RoadPoint point = currentRoad->getRoadPoint(v_c.u(), v_c.v());
		    fprintf(stderr,"height:  %f \n",point.z() + header.zoffset);
                    return point.z() + header.zoffset;
                }
                else
                {
                    currentRoad = NULL;
                }
            }
            else if (currentLongPos < -0.1 || currentLongPos > currentRoad->getLength() + 0.1)
            {
                // left road searching for the next road over all roads in the system
                v_c = RoadSystem::Instance()->searchPosition(v_w, currentRoad, currentLongPos);
                if (!v_c.isNaV())
                {
                    if (currentRoad->isOnRoad(v_c))
                    {
                        RoadPoint point = currentRoad->getRoadPoint(v_c.u(), v_c.v());
                        return point.z() + header.zoffset;
                    }
                    else
                    {
                        currentRoad = NULL;
                    }
                }
                else
                {
                    currentRoad = NULL;
                }
            }
        }
        if (currentRoad==NULL)
        {
            // left road searching for the next road over all roads in the system
            v_c = RoadSystem::Instance()->searchPosition(v_w, currentRoad, currentLongPos);
            if (!v_c.isNaV())
            {
                if (currentRoad->isOnRoad(v_c))
                {
                    RoadPoint point = currentRoad->getRoadPoint(v_c.u(), v_c.v());
                    return point.z() + header.zoffset;
                }
                else
                {
                    currentRoad = NULL;
                }
            }
            else
            {
                currentRoad = NULL;
            }
        }
    }
    double geoTransform[6];
    geoTransform[0] = _geoTransform(3, 0);
    geoTransform[1] = _geoTransform(0, 0);
    geoTransform[2] = _geoTransform(1, 0);
    geoTransform[3] = _geoTransform(3, 1);
    geoTransform[4] = _geoTransform(0, 1);
    geoTransform[5] = _geoTransform(1, 1);

// shift the transform to the middle of the cell if a raster format is used
#ifdef SHIFT_RASTER_BY_HALF_CELL
    if (_dataType == RASTER)
    {
        geoTransform[0] += 0.5 * geoTransform[1];
        geoTransform[3] += 0.5 * geoTransform[5];
    }
#endif

    double invTransform[6];
    if (!GDALInvGeoTransform(geoTransform, invTransform))
    {
        std::cerr << "VirtualPlanetBuilder:" << __FILE__ << ": matrix inversion failed" << std::endl;
    }
    double r, c;
    GDALApplyGeoTransform(invTransform, x, y, &c, &r);

    int rowMin = osg::maximum((int)floor(r), 0);
    int rowMax = osg::maximum(osg::minimum((int)ceil(r), (int)(_numValuesY - 1)), 0);
    int colMin = osg::maximum((int)floor(c), 0);
    int colMax = osg::maximum(osg::minimum((int)ceil(c), (int)(_numValuesX - 1)), 0);

    if (rowMin > rowMax)
        rowMin = rowMax;
    if (colMin > colMax)
        colMin = colMax;

    float urHeight, llHeight, ulHeight, lrHeight;

    band->RasterIO(GF_Read, colMin, rowMin, 1, 1, &llHeight, 1, 1, GDT_Float32, 0, 0);
    band->RasterIO(GF_Read, colMin, rowMax, 1, 1, &ulHeight, 1, 1, GDT_Float32, 0, 0);
    band->RasterIO(GF_Read, colMax, rowMin, 1, 1, &lrHeight, 1, 1, GDT_Float32, 0, 0);
    band->RasterIO(GF_Read, colMax, rowMax, 1, 1, &urHeight, 1, 1, GDT_Float32, 0, 0);

    ValidValueOperator validValueOperator(band);

    if (validValueOperator.isNoDataValue(llHeight))
        llHeight = originalHeight;
    if (validValueOperator.isNoDataValue(ulHeight))
        ulHeight = originalHeight;
    if (validValueOperator.isNoDataValue(lrHeight))
        lrHeight = originalHeight;
    if (validValueOperator.isNoDataValue(urHeight))
        urHeight = originalHeight;

    double x_rem = c - (int)c;
    double y_rem = r - (int)r;

    double w00 = (1.0 - y_rem) * (1.0 - x_rem) * (double)llHeight;
    double w01 = (1.0 - y_rem) * x_rem * (double)lrHeight;
    double w10 = y_rem * (1.0 - x_rem) * (double)ulHeight;
    double w11 = y_rem * x_rem * (double)urHeight;

    float result = (float)(w00 + w01 + w10 + w11);

    return result;
}

SourceData *SourceData::readData(Source *source)
{
    if (!source)
        return 0;

    switch (source->getType())
    {
    case (Source::IMAGE):
    case (Source::HEIGHT_FIELD):
    {
        // try osgDB for source if height data is a vector data set
        if ((source->getType() == Source::HEIGHT_FIELD) && (source->_dataType == Source::VECTOR))
        {
            osg::HeightField *hf = (osg::HeightField *)source->getHFDataset();
            if (!hf)
                hf = osgDB::readHeightFieldFile(source->getFileName().c_str());

            if (hf)
            {
                SourceData *data = new SourceData(source);

                // need to set vector or raster
                data->_dataType = source->_dataType;

                data->_hfDataset = hf;

                data->_numValuesX = hf->getNumColumns();
                data->_numValuesY = hf->getNumRows();
                data->_numValuesZ = 1;
                data->_hasGCPs = false;
                data->_cs = new osg::CoordinateSystemNode("WKT", "");

                double geoTransform[6];
                for (int i = 0; i < 6; i++)
                    geoTransform[i] = 0.0;
                // top left vertex - represent as top down
                osg::Vec3 vertex = hf->getVertex(0, hf->getNumRows() - 1);
                geoTransform[0] = vertex.x();
                geoTransform[3] = vertex.y();
                geoTransform[1] = hf->getXInterval();
                geoTransform[5] = -hf->getYInterval();
                data->_geoTransform.set(geoTransform[1], geoTransform[4], 0.0, 0.0,
                                        geoTransform[2], geoTransform[5], 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0,
                                        geoTransform[0], geoTransform[3], 0.0, 1.0);

                data->computeExtents();
                return data;
            }
        }

        osg::ref_ptr<GeospatialDataset> gdalDataSet = source->getGeospatialDataset(READ_ONLY);

        if (gdalDataSet.valid())
        {
            SourceData *data = new SourceData(source);

            // need to set vector or raster
            data->_dataType = source->_dataType;

            data->_numValuesX = gdalDataSet->GetRasterXSize();
            data->_numValuesY = gdalDataSet->GetRasterYSize();
            data->_numValuesZ = gdalDataSet->GetRasterCount();
            data->_hasGCPs = gdalDataSet->GetGCPCount() != 0;

            const char *pszSourceSRS = gdalDataSet->GetProjectionRef();
            if (!pszSourceSRS || strlen(pszSourceSRS) == 0)
                pszSourceSRS = gdalDataSet->GetGCPProjection();

            data->_cs = new osg::CoordinateSystemNode("WKT", pszSourceSRS);

            double geoTransform[6];
            if (gdalDataSet->GetGeoTransform(geoTransform) == CE_None)
            {
#ifdef SHIFT_RASTER_BY_HALF_CELL
                // shift the transform to the middle of the cell if a raster interpreted as vector
                if (data->_dataType == VECTOR)
                {
                    geoTransform[0] += 0.5 * geoTransform[1];
                    geoTransform[3] += 0.5 * geoTransform[5];
                }
#endif
                data->_geoTransform.set(geoTransform[1], geoTransform[4], 0.0, 0.0,
                                        geoTransform[2], geoTransform[5], 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0,
                                        geoTransform[0], geoTransform[3], 0.0, 1.0);

                data->computeExtents();
            }
            else if (gdalDataSet->GetGCPCount() > 0 && gdalDataSet->GetGCPProjection())
            {
                log(osg::INFO, "    Using GCP's");

                /* -------------------------------------------------------------------- */
                /*      Create a transformation object from the source to               */
                /*      destination coordinate system.                                  */
                /* -------------------------------------------------------------------- */
                void *hTransformArg = GDALCreateGenImgProjTransformer(gdalDataSet->getGDALDataset(), pszSourceSRS,
                                                                      NULL, pszSourceSRS,
                                                                      TRUE, 0.0, 1);

                if (hTransformArg == NULL)
                {
                    log(osg::INFO, " failed to create transformer");
                    return NULL;
                }

                /* -------------------------------------------------------------------- */
                /*      Get approximate output definition.                              */
                /* -------------------------------------------------------------------- */
                double adfDstGeoTransform[6];
                int nPixels = 0, nLines = 0;
                if (GDALSuggestedWarpOutput(gdalDataSet->getGDALDataset(),
                                            GDALGenImgProjTransform, hTransformArg,
                                            adfDstGeoTransform, &nPixels, &nLines)
                    != CE_None)
                {
                    log(osg::INFO, " failed to create warp");
                    return NULL;
                }

                GDALDestroyGenImgProjTransformer(hTransformArg);

                data->_geoTransform.set(adfDstGeoTransform[1], adfDstGeoTransform[4], 0.0, 0.0,
                                        adfDstGeoTransform[2], adfDstGeoTransform[5], 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0,
                                        adfDstGeoTransform[0], adfDstGeoTransform[3], 0.0, 1.0);

                data->computeExtents();
            }
            else
            {
                log(osg::INFO, "    No GeoTransform or GCP's - unable to compute position in space");

                data->_geoTransform.set(1.0, 0.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0,
                                        0.0, 0.0, 0.0, 1.0);

                data->computeExtents();
            }
            return data;
        }
    }
    case (Source::MODEL):
    {
        osg::Node *model = osgDB::readNodeFile(source->getFileName().c_str());
        if (model)
        {
            SourceData *data = new SourceData(source);
            data->_model = model;
            data->_extents.expandBy(model->getBound());
        }
    }
    break;
    case (Source::SHAPEFILE):
        break;
    }

    return 0;
}

GeospatialExtents SourceData::getExtents(const osg::CoordinateSystemNode *cs) const
{
    return computeSpatialProperties(cs)._extents;
}

const SpatialProperties &SourceData::computeSpatialProperties(const osg::CoordinateSystemNode *cs) const
{
    // check to see it exists in the _spatialPropertiesMap first.
    SpatialPropertiesMap::const_iterator itr = _spatialPropertiesMap.find(cs);
    if (itr != _spatialPropertiesMap.end())
    {
        return itr->second;
    }

    if (areCoordinateSystemEquivalent(_cs.get(), cs))
    {
        return *this;
    }

    if (_cs.valid() && cs)
    {

        osg::ref_ptr<GeospatialDataset> _gdalDataset = _source->getGeospatialDataset(READ_ONLY);
        if (_gdalDataset.valid())
        {

            //log(osg::INFO,"Projecting bounding volume for "<<_source->getFileName());

            // insert into the _spatialPropertiesMap for future reuse.
            _spatialPropertiesMap[cs] = *this;
            SpatialProperties &sp = _spatialPropertiesMap[cs];

            /* -------------------------------------------------------------------- */
            /*      Create a transformation object from the source to               */
            /*      destination coordinate system.                                  */
            /* -------------------------------------------------------------------- */
            void *hTransformArg = GDALCreateGenImgProjTransformer(_gdalDataset->getGDALDataset(), _cs->getCoordinateSystem().c_str(),
                                                                  NULL, cs->getCoordinateSystem().c_str(),
                                                                  TRUE, 0.0, 1);

            if (!hTransformArg)
            {
                log(osg::INFO, " failed to create transformer");
                return sp;
            }

            double adfDstGeoTransform[6];
            int nPixels = 0, nLines = 0;
            if (GDALSuggestedWarpOutput(_gdalDataset->getGDALDataset(),
                                        GDALGenImgProjTransform, hTransformArg,
                                        adfDstGeoTransform, &nPixels, &nLines)
                != CE_None)
            {
                log(osg::INFO, " failed to create warp");
                return sp;
            }

            sp._numValuesX = nPixels;
            sp._numValuesY = nLines;
            sp._cs = const_cast<osg::CoordinateSystemNode *>(cs);
            sp._geoTransform.set(adfDstGeoTransform[1], adfDstGeoTransform[4], 0.0, 0.0,
                                 adfDstGeoTransform[2], adfDstGeoTransform[5], 0.0, 0.0,
                                 0.0, 0.0, 1.0, 0.0,
                                 adfDstGeoTransform[0], adfDstGeoTransform[3], 0.0, 1.0);

            GDALDestroyGenImgProjTransformer(hTransformArg);

            sp.computeExtents();

            return sp;
        }
    }
    log(osg::INFO, "DataSet::DataSource::assuming compatible coordinates.");
    return *this;
}

bool SourceData::intersects(const SpatialProperties &sp) const
{
    return sp._extents.intersects(getExtents(sp._cs.get()));
}

void SourceData::read(DestinationData &destination)
{
    log(osg::INFO, "A");

    if (!_source)
        return;

    log(osg::INFO, "B");

    switch (_source->getType())
    {
    case (Source::IMAGE):
        log(osg::INFO, "B.1");
        readImage(destination);
        break;
    case (Source::HEIGHT_FIELD):
        log(osg::INFO, "B.2");
        readHeightField(destination);
        break;
    case (Source::SHAPEFILE):
        log(osg::INFO, "B.3");
        readShapeFile(destination);
        break;
    case (Source::MODEL):
        log(osg::INFO, "B.5");
        readModels(destination);
        break;
    }
    log(osg::INFO, "C");
}

void SourceData::readImage(DestinationData &destination)
{
    log(osg::INFO, "readImage ");

    if (destination._image.valid())
    {
        osg::ref_ptr<GeospatialDataset> _gdalDataset = _source->getOptimumGeospatialDataset(destination, READ_ONLY);
        if (!_gdalDataset)
            return;

        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_gdalDataset->getMutex());

        GeospatialExtents s_bb = getExtents(destination._cs.get());
        GeospatialExtents d_bb = destination._extents;

        // note, we have to handle the possibility of goegraphic datasets wrapping over on themselves when they pass over the dateline
        // to do this we have to test geographic datasets via two passes, each with a 360 degree shift of the source cata.
        double xoffset = ((d_bb.xMin() < s_bb.xMin()) && (d_bb._isGeographic)) ? -360.0 : 0.0;
        unsigned int numXChecks = d_bb._isGeographic ? 2 : 1;
        for (unsigned int ic = 0; ic < numXChecks; ++ic, xoffset += 360.0)
        {

            log(osg::INFO, "Testing %f", xoffset);
            log(osg::INFO, "  s_bb %f %f", s_bb.xMin() + xoffset, s_bb.xMax() + xoffset);
            log(osg::INFO, "  d_bb %f %f", d_bb.xMin(), d_bb.xMax());

            GeospatialExtents intersect_bb(d_bb.intersection(s_bb, xoffset));
            if (!intersect_bb.nonZeroExtents())
            {
                log(osg::INFO, "Reading image but it does not intesection destination - ignoring");
                continue;
            }

            log(osg::INFO, "readImage s_bb is geographic %d", s_bb._isGeographic);
            log(osg::INFO, "readImage d_bb is geographic %d", d_bb._isGeographic);
            log(osg::INFO, "readImage intersect_bb is geographic %d", intersect_bb._isGeographic);

            int windowX = osg::maximum((int)floorf((float)_numValuesX * (intersect_bb.xMin() - xoffset - s_bb.xMin()) / (s_bb.xMax() - s_bb.xMin())), 0);
            int windowY = osg::maximum((int)floorf((float)_numValuesY * (intersect_bb.yMin() - s_bb.yMin()) / (s_bb.yMax() - s_bb.yMin())), 0);
            int windowWidth = osg::minimum((int)ceilf((float)_numValuesX * (intersect_bb.xMax() - xoffset - s_bb.xMin()) / (s_bb.xMax() - s_bb.xMin())), (int)_numValuesX) - windowX;
            int windowHeight = osg::minimum((int)ceilf((float)_numValuesY * (intersect_bb.yMax() - s_bb.yMin()) / (s_bb.yMax() - s_bb.yMin())), (int)_numValuesY) - windowY;

            int destX = osg::maximum((int)floorf((float)destination._image->s() * (intersect_bb.xMin() - d_bb.xMin()) / (d_bb.xMax() - d_bb.xMin())), 0);
            int destY = osg::maximum((int)floorf((float)destination._image->t() * (intersect_bb.yMin() - d_bb.yMin()) / (d_bb.yMax() - d_bb.yMin())), 0);
            int destWidth = osg::minimum((int)ceilf((float)destination._image->s() * (intersect_bb.xMax() - d_bb.xMin()) / (d_bb.xMax() - d_bb.xMin())), (int)destination._image->s()) - destX;
            int destHeight = osg::minimum((int)ceilf((float)destination._image->t() * (intersect_bb.yMax() - d_bb.yMin()) / (d_bb.yMax() - d_bb.yMin())), (int)destination._image->t()) - destY;

            log(osg::INFO, "   copying from %d\t%d\t%d\t%d", windowX, windowY, windowWidth, windowHeight);
            log(osg::INFO, "             to %d\t%d\t%d\t%d", destX, destY, destWidth, destHeight);

            int readWidth = destWidth;
            int readHeight = destHeight;
            bool doResample = false;

            float destWindowWidthRatio = (float)destWidth / (float)windowWidth;
            float destWindowHeightRatio = (float)destHeight / (float)windowHeight;
            const float resizeTolerance = 1.1;

            bool interpolateSourceImagery = destination._dataSet->getUseInterpolatedImagerySampling();

            if (interpolateSourceImagery && (destWindowWidthRatio > resizeTolerance || destWindowHeightRatio > resizeTolerance) && windowWidth >= 2 && windowHeight >= 2)
            {
                readWidth = windowWidth;
                readHeight = windowHeight;
                doResample = true;
            }

            bool hasRGB = _gdalDataset->GetRasterCount() >= 3;
            bool hasAlpha = _gdalDataset->GetRasterCount() >= 4;
            bool hasColorTable = _gdalDataset->GetRasterCount() >= 1 && _gdalDataset->GetRasterBand(1)->GetColorTable();
            bool hasGreyScale = _gdalDataset->GetRasterCount() == 1;
            unsigned int numSourceComponents = hasAlpha ? 4 : 3;

            if (hasRGB || hasColorTable || hasGreyScale)
            {
                // RGB

                unsigned int numBytesPerPixel = destination._image->getDataType() == GL_FLOAT ? 4 : 1;
                GDALDataType targetGDALType = destination._image->getDataType() == GL_FLOAT ? GDT_Float32 : GDT_Byte;

                int pixelSpace = numSourceComponents * numBytesPerPixel;

                log(osg::INFO, "reading RGB");

                unsigned char *tempImage = new unsigned char[readWidth * readHeight * pixelSpace];

                /* New code courtesy of Frank Warmerdam of the GDAL group */

                // RGB images ... or at least we assume 3+ band images can be treated
                // as RGB.
                if (hasRGB)
                {
                    GDALRasterBand *bandRed = _gdalDataset->GetRasterBand(1);
                    GDALRasterBand *bandGreen = _gdalDataset->GetRasterBand(2);
                    GDALRasterBand *bandBlue = _gdalDataset->GetRasterBand(3);
                    GDALRasterBand *bandAlpha = hasAlpha ? _gdalDataset->GetRasterBand(4) : 0;

                    bandRed->RasterIO(GF_Read,
                                      windowX, _numValuesY - (windowY + windowHeight),
                                      windowWidth, windowHeight,
                                      (void *)(tempImage + 0), readWidth, readHeight,
                                      targetGDALType, pixelSpace, pixelSpace * readWidth);
                    bandGreen->RasterIO(GF_Read,
                                        windowX, _numValuesY - (windowY + windowHeight),
                                        windowWidth, windowHeight,
                                        (void *)(tempImage + 1 * numBytesPerPixel), readWidth, readHeight,
                                        targetGDALType, pixelSpace, pixelSpace * readWidth);
                    bandBlue->RasterIO(GF_Read,
                                       windowX, _numValuesY - (windowY + windowHeight),
                                       windowWidth, windowHeight,
                                       (void *)(tempImage + 2 * numBytesPerPixel), readWidth, readHeight,
                                       targetGDALType, pixelSpace, pixelSpace * readWidth);

                    if (bandAlpha)
                    {
                        bandAlpha->RasterIO(GF_Read,
                                            windowX, _numValuesY - (windowY + windowHeight),
                                            windowWidth, windowHeight,
                                            (void *)(tempImage + 3 * numBytesPerPixel), readWidth, readHeight,
                                            targetGDALType, pixelSpace, pixelSpace * readWidth);
                    }
                }

                else if (hasColorTable)
                {
                    // Pseudocolored image.  Convert 1 band + color table to 24bit RGB.

                    GDALRasterBand *band;
                    GDALColorTable *ct;
                    int i;

                    band = _gdalDataset->GetRasterBand(1);

                    band->RasterIO(GF_Read,
                                   windowX, _numValuesY - (windowY + windowHeight),
                                   windowWidth, windowHeight,
                                   (void *)(tempImage + 0), readWidth, readHeight,
                                   targetGDALType, pixelSpace, pixelSpace * readWidth);

                    ct = band->GetColorTable();

                    for (i = 0; i < readWidth * readHeight; i++)
                    {
                        GDALColorEntry sEntry;

                        // default to greyscale equilvelent.
                        sEntry.c1 = tempImage[i * 3];
                        sEntry.c2 = tempImage[i * 3];
                        sEntry.c3 = tempImage[i * 3];

                        ct->GetColorEntryAsRGB(tempImage[i * 3], &sEntry);

                        // Apply RGB back over destination image.
                        tempImage[i * 3 + 0] = sEntry.c1;
                        tempImage[i * 3 + 1] = sEntry.c2;
                        tempImage[i * 3 + 2] = sEntry.c3;
                    }
                }

                else if (hasGreyScale)
                {
                    // Greyscale image.  Convert 1 band to 24bit RGB.
                    GDALRasterBand *band;

                    band = _gdalDataset->GetRasterBand(1);

                    band->RasterIO(GF_Read,
                                   windowX, _numValuesY - (windowY + windowHeight),
                                   windowWidth, windowHeight,
                                   (void *)(tempImage + 0), readWidth, readHeight,
                                   targetGDALType, pixelSpace, pixelSpace * readWidth);
                    band->RasterIO(GF_Read,
                                   windowX, _numValuesY - (windowY + windowHeight),
                                   windowWidth, windowHeight,
                                   (void *)(tempImage + 1 * numBytesPerPixel), readWidth, readHeight,
                                   targetGDALType, pixelSpace, pixelSpace * readWidth);
                    band->RasterIO(GF_Read,
                                   windowX, _numValuesY - (windowY + windowHeight),
                                   windowWidth, windowHeight,
                                   (void *)(tempImage + 2 * numBytesPerPixel), readWidth, readHeight,
                                   targetGDALType, pixelSpace, pixelSpace * readWidth);
                }

                if (doResample || readWidth != destWidth || readHeight != destHeight)
                {
                    unsigned char *destImage = new unsigned char[destWidth * destHeight * pixelSpace];

                    // rescale image by hand as glu seem buggy....
                    for (int j = 0; j < destHeight; ++j)
                    {
                        float t_d = (destHeight > 1) ? ((float)j / ((float)destHeight - 1)) : 0;
                        for (int i = 0; i < destWidth; ++i)
                        {
                            float s_d = (destWidth > 1) ? ((float)i / ((float)destWidth - 1)) : 0;

                            float flt_read_i = s_d * ((float)readWidth - 1);
                            float flt_read_j = t_d * ((float)readHeight - 1);

                            int read_i = (int)flt_read_i;
                            if (read_i >= readWidth)
                                read_i = readWidth - 1;

                            float flt_read_ir = flt_read_i - read_i;
                            if (read_i == readWidth - 1)
                                flt_read_ir = 0.0f;

                            int read_j = (int)flt_read_j;
                            if (read_j >= readHeight)
                                read_j = readHeight - 1;

                            float flt_read_jr = flt_read_j - read_j;
                            if (read_j == readHeight - 1)
                                flt_read_jr = 0.0f;

                            unsigned char *dest = destImage + (j * destWidth + i) * pixelSpace;
                            float *dest_float = (float *)dest;
                            if (flt_read_ir == 0.0f) // no need to interpolate i axis.
                            {
                                if (flt_read_jr == 0.0f) // no need to interpolate j axis.
                                {
                                    // copy pixels
                                    unsigned char *src = tempImage + (read_j * readWidth + read_i) * pixelSpace;
                                    if (targetGDALType == GDT_Float32)
                                    {
                                        float *src_float = (float *)src;
                                        dest_float[0] = src_float[0];
                                        dest_float[1] = src_float[1];
                                        dest_float[2] = src_float[2];
                                        if (numSourceComponents == 4)
                                            dest_float[3] = src_float[3];
                                    }
                                    else
                                    {
                                        dest[0] = src[0];
                                        dest[1] = src[1];
                                        dest[2] = src[2];
                                        if (numSourceComponents == 4)
                                            dest[3] = src[3];
                                    }
                                    //std::cout<<"copy");
                                }
                                else // need to interpolate j axis.
                                {
                                    // copy pixels
                                    unsigned char *src_0 = tempImage + (read_j * readWidth + read_i) * pixelSpace;
                                    unsigned char *src_1 = src_0 + readWidth * pixelSpace;
                                    float r_0 = 1.0f - flt_read_jr;
                                    float r_1 = flt_read_jr;
                                    if (targetGDALType == GDT_Float32)
                                    {
                                        float *src_0_float = (float *)src_0;
                                        float *src_1_float = (float *)src_1;
                                        dest[0] = src_0_float[0] * r_0 + src_1_float[0] * r_1;
                                        dest[1] = src_0_float[1] * r_0 + src_1_float[1] * r_1;
                                        dest[2] = src_0_float[2] * r_0 + src_1_float[2] * r_1;
                                        if (numSourceComponents == 4)
                                            dest[3] = src_0_float[3] * r_0 + src_1_float[3] * r_1;
                                    }
                                    else
                                    {
                                        dest[0] = (unsigned char)((float)src_0[0] * r_0 + (float)src_1[0] * r_1);
                                        dest[1] = (unsigned char)((float)src_0[1] * r_0 + (float)src_1[1] * r_1);
                                        dest[2] = (unsigned char)((float)src_0[2] * r_0 + (float)src_1[2] * r_1);
                                        if (numSourceComponents == 4)
                                            dest[3] = (unsigned char)((float)src_0[3] * r_0 + (float)src_1[3] * r_1);
                                    }
                                    //std::cout<<"interpolate j axis");
                                }
                            }
                            else // need to interpolate i axis.
                            {
                                if (flt_read_jr == 0.0f) // no need to interpolate j axis.
                                {
                                    // copy pixels
                                    unsigned char *src_0 = tempImage + (read_j * readWidth + read_i) * pixelSpace;
                                    unsigned char *src_1 = src_0 + pixelSpace;
                                    float r_0 = 1.0f - flt_read_ir;
                                    float r_1 = flt_read_ir;
                                    if (targetGDALType == GDT_Float32)
                                    {
                                        float *src_0_float = (float *)src_0;
                                        float *src_1_float = (float *)src_1;
                                        dest[0] = src_0_float[0] * r_0 + src_1_float[0] * r_1;
                                        dest[1] = src_0_float[1] * r_0 + src_1_float[1] * r_1;
                                        dest[2] = src_0_float[2] * r_0 + src_1_float[2] * r_1;
                                        if (numSourceComponents == 4)
                                            dest[3] = src_0_float[3] * r_0 + src_1_float[3] * r_1;
                                    }
                                    else
                                    {
                                        dest[0] = (unsigned char)((float)src_0[0] * r_0 + (float)src_1[0] * r_1);
                                        dest[1] = (unsigned char)((float)src_0[1] * r_0 + (float)src_1[1] * r_1);
                                        dest[2] = (unsigned char)((float)src_0[2] * r_0 + (float)src_1[2] * r_1);
                                        if (numSourceComponents == 4)
                                            dest[3] = (unsigned char)((float)src_0[3] * r_0 + (float)src_1[3] * r_1);
                                    }
                                    //std::cout<<"interpolate i axis");
                                }
                                else // need to interpolate i and j axis.
                                {
                                    unsigned char *src_0 = tempImage + (read_j * readWidth + read_i) * pixelSpace;
                                    unsigned char *src_1 = src_0 + readWidth * pixelSpace;
                                    unsigned char *src_2 = src_0 + pixelSpace;
                                    unsigned char *src_3 = src_1 + pixelSpace;
                                    float r_0 = (1.0f - flt_read_ir) * (1.0f - flt_read_jr);
                                    float r_1 = (1.0f - flt_read_ir) * flt_read_jr;
                                    float r_2 = (flt_read_ir) * (1.0f - flt_read_jr);
                                    float r_3 = (flt_read_ir)*flt_read_jr;
                                    if (targetGDALType == GDT_Float32)
                                    {
                                        float *src_0_float = (float *)src_0;
                                        float *src_1_float = (float *)src_1;
                                        float *src_2_float = (float *)src_2;
                                        float *src_3_float = (float *)src_3;
                                        dest[0] = src_0_float[0] * r_0 + src_1_float[0] * r_1 + src_2_float[0] * r_2 + src_3_float[0] * r_3;
                                        dest[1] = src_0_float[1] * r_0 + src_1_float[1] * r_1 + src_2_float[1] * r_2 + src_3_float[1] * r_3;
                                        dest[2] = src_0_float[2] * r_0 + src_1_float[2] * r_1 + src_2_float[2] * r_2 + src_3_float[2] * r_3;
                                        if (numSourceComponents == 4)
                                            dest[3] = src_0_float[3] * r_0 + src_1_float[3] * r_1 + src_2_float[3] * r_2 + src_3_float[3] * r_3;
                                    }
                                    else
                                    {
                                        dest[0] = (unsigned char)(((float)src_0[0]) * r_0 + ((float)src_1[0]) * r_1 + ((float)src_2[0]) * r_2 + ((float)src_3[0]) * r_3);
                                        dest[1] = (unsigned char)(((float)src_0[1]) * r_0 + ((float)src_1[1]) * r_1 + ((float)src_2[1]) * r_2 + ((float)src_3[1]) * r_3);
                                        dest[2] = (unsigned char)(((float)src_0[2]) * r_0 + ((float)src_1[2]) * r_1 + ((float)src_2[2]) * r_2 + ((float)src_3[2]) * r_3);
                                        if (numSourceComponents == 4)
                                            dest[3] = (unsigned char)(((float)src_0[3]) * r_0 + ((float)src_1[3]) * r_1 + ((float)src_2[3]) * r_2 + ((float)src_3[3]) * r_3);
                                    }
                                    //std::cout<<"interpolate i & j axis");
                                }
                            }
                        }
                    }

                    delete[] tempImage;
                    tempImage = destImage;
                }

                // now copy into destination image
                unsigned char *sourceRowPtr = tempImage;
                int sourceRowDelta = pixelSpace * destWidth;
                unsigned char *destinationRowPtr = destination._image->data(destX, destY + destHeight - 1);
                int destinationRowDelta = -(int)(destination._image->getRowSizeInBytes());
                int destination_pixelSpace = destination._image->getPixelSizeInBits() / 8;
                bool destination_hasAlpha = osg::Image::computeNumComponents(destination._image->getPixelFormat()) == 4;

                // copy image to destination image
                for (int row = 0;
                     row < destHeight;
                     ++row, sourceRowPtr += sourceRowDelta, destinationRowPtr += destinationRowDelta)
                {
                    unsigned char *sourceColumnPtr = sourceRowPtr;
                    unsigned char *destinationColumnPtr = destinationRowPtr;

                    for (int col = 0;
                         col < destWidth;
                         ++col, sourceColumnPtr += pixelSpace, destinationColumnPtr += destination_pixelSpace)
                    {
                        if (targetGDALType == GDT_Float32)
                        {
                            float *sourceColumnPtr_float = (float *)sourceColumnPtr;
                            float *destinationColumnPtr_float = (float *)destinationColumnPtr;
                            if (hasAlpha)
                            {
                                // only copy over source pixel if its alpha value is not 0
                                if (sourceColumnPtr_float[3] > 0.0f)
                                {
                                    if (sourceColumnPtr_float[3] >= 1.0f)
                                    {
                                        // source alpha is full on so directly copy over.
                                        destinationColumnPtr_float[0] = sourceColumnPtr_float[0];
                                        destinationColumnPtr_float[1] = sourceColumnPtr_float[1];
                                        destinationColumnPtr_float[2] = sourceColumnPtr_float[2];

                                        if (destination_hasAlpha)
                                            destinationColumnPtr_float[3] = sourceColumnPtr_float[3];
                                    }
                                    else
                                    {
                                        // source value isn't full on so blend it with destination
                                        float rs = sourceColumnPtr_float[3];
                                        float rd = 1.0f - rs;

                                        destinationColumnPtr_float[0] = rd * destinationColumnPtr_float[0] + rs * sourceColumnPtr_float[0];
                                        destinationColumnPtr_float[1] = rd * destinationColumnPtr_float[1] + rs * sourceColumnPtr_float[1];
                                        destinationColumnPtr_float[2] = rd * destinationColumnPtr_float[2] + rs * sourceColumnPtr_float[2];

                                        if (destination_hasAlpha)
                                            destinationColumnPtr_float[3] = osg::maximum(destinationColumnPtr_float[3], sourceColumnPtr_float[3]);
                                    }
                                }
                            }
                            else if (sourceColumnPtr_float[0] > 0.0f || sourceColumnPtr_float[1] > 0.0f || sourceColumnPtr_float[2] > 0.0f)
                            {
                                destinationColumnPtr_float[0] = sourceColumnPtr_float[0];
                                destinationColumnPtr_float[1] = sourceColumnPtr_float[1];
                                destinationColumnPtr_float[2] = sourceColumnPtr_float[2];
                                if (destination_hasAlpha)
                                    destinationColumnPtr_float[3] = 1.0f;
                            }
                        }
                        else
                        {
                            if (hasAlpha)
                            {
                                // only copy over source pixel if its alpha value is not 0
                                if (sourceColumnPtr[3] != 0)
                                {
                                    if (sourceColumnPtr[3] == 255)
                                    {
                                        // source alpha is full on so directly copy over.
                                        destinationColumnPtr[0] = sourceColumnPtr[0];
                                        destinationColumnPtr[1] = sourceColumnPtr[1];
                                        destinationColumnPtr[2] = sourceColumnPtr[2];

                                        if (destination_hasAlpha)
                                            destinationColumnPtr[3] = sourceColumnPtr[3];
                                    }
                                    else
                                    {
                                        // source value isn't full on so blend it with destination
                                        float rs = (float)sourceColumnPtr[3] / 255.0f;
                                        float rd = 1.0f - rs;

                                        destinationColumnPtr[0] = (int)(rd * (float)destinationColumnPtr[0] + rs * (float)sourceColumnPtr[0]);
                                        destinationColumnPtr[1] = (int)(rd * (float)destinationColumnPtr[1] + rs * (float)sourceColumnPtr[1]);
                                        destinationColumnPtr[2] = (int)(rd * (float)destinationColumnPtr[2] + rs * (float)sourceColumnPtr[2]);

                                        if (destination_hasAlpha)
                                            destinationColumnPtr[3] = osg::maximum(destinationColumnPtr[3], sourceColumnPtr[3]);
                                    }
                                }
                            }
                            else if (sourceColumnPtr[0] != 0 || sourceColumnPtr[1] != 0 || sourceColumnPtr[2] != 0)
                            {
                                destinationColumnPtr[0] = sourceColumnPtr[0];
                                destinationColumnPtr[1] = sourceColumnPtr[1];
                                destinationColumnPtr[2] = sourceColumnPtr[2];
                                if (destination_hasAlpha)
                                    destinationColumnPtr[3] = 255;
                            }
                        }
                    }
                }

                delete[] tempImage;
            }
            else
            {
                log(osg::INFO, "Warning image not read as Red, Blue and Green bands not present");
            }
        }
    }
}

void SourceData::readHeightField(DestinationData &destination)
{
    log(osg::INFO, "In SourceData::readHeightField");

    if (destination._heightField.valid())
    {
        log(osg::INFO, "Reading height field");

        osg::ref_ptr<GeospatialDataset> _gdalDataset = _source->getOptimumGeospatialDataset(destination, READ_ONLY);
        if (!_gdalDataset.valid())
            return;

        OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_gdalDataset->getMutex());

        GeospatialExtents s_bb = getExtents(destination._cs.get());
        GeospatialExtents d_bb = destination._extents;

        // note, we have to handle the possibility of goegraphic datasets wrapping over on themselves when they pass over the dateline
        // to do this we have to test geographic datasets via two passes, each with a 360 degree shift of the source cata.
        double xoffset = ((d_bb.xMin() < s_bb.xMin()) && (d_bb._isGeographic)) ? -360.0 : 0.0;
        unsigned int numXChecks = d_bb._isGeographic ? 2 : 1;
        for (unsigned int ic = 0; ic < numXChecks; ++ic, xoffset += 360.0)
        {

            log(osg::INFO, "Testing %f", xoffset);
            log(osg::INFO, "  s_bb %f %f", s_bb.xMin() + xoffset, s_bb.xMax() + xoffset);
            log(osg::INFO, "  d_bb %f %f", d_bb.xMin(), d_bb.xMax());

            GeospatialExtents intersect_bb(d_bb.intersection(s_bb, xoffset));
            if (!intersect_bb.nonZeroExtents())
            {
                log(osg::INFO, "Reading height field but it does not intesection destination - ignoring");
                continue;
            }

            int destX = osg::maximum((int)floorf((float)destination._heightField->getNumColumns() * (intersect_bb.xMin() - d_bb.xMin()) / (d_bb.xMax() - d_bb.xMin())), 0);
            int destY = osg::maximum((int)floorf((float)destination._heightField->getNumRows() * (intersect_bb.yMin() - d_bb.yMin()) / (d_bb.yMax() - d_bb.yMin())), 0);
            int destWidth = osg::minimum((int)ceilf((float)destination._heightField->getNumColumns() * (intersect_bb.xMax() - d_bb.xMin()) / (d_bb.xMax() - d_bb.xMin())), (int)destination._heightField->getNumColumns()) - destX;
            int destHeight = osg::minimum((int)ceilf((float)destination._heightField->getNumRows() * (intersect_bb.yMax() - d_bb.yMin()) / (d_bb.yMax() - d_bb.yMin())), (int)destination._heightField->getNumRows()) - destY;

            // use heightfield if it exists
            if (_hfDataset.valid())
            {
                // read the data.
                osg::HeightField *hf = destination._heightField.get();

                //float noDataValueFill = 0.0f;
                //bool ignoreNoDataValue = true;

                //Sample terrain at each vert to increase accuracy of the terrain.
                int endX = destX + destWidth;
                int endY = destY + destHeight;

                double orig_X = hf->getOrigin().x();
                double orig_Y = hf->getOrigin().y();
                double delta_X = hf->getXInterval();
                double delta_Y = hf->getYInterval();

                for (int c = destX; c < endX; ++c)
                {
                    double geoX = orig_X + (delta_X * (double)c);
                    for (int r = destY; r < endY; ++r)
                    {
                        double geoY = orig_Y + (delta_Y * (double)r);
                        float h = getInterpolatedValue(_hfDataset.get(), geoX - xoffset, geoY);
                        hf->setHeight(c, r, h);
                    }
                }
                return;
            }

            // which band do we want to read from...
            int numBands = _gdalDataset->GetRasterCount();
            GDALRasterBand *bandGray = 0;
            GDALRasterBand *bandRed = 0;
            GDALRasterBand *bandGreen = 0;
            GDALRasterBand *bandBlue = 0;
            GDALRasterBand *bandAlpha = 0;

            for (int b = 1; b <= numBands; ++b)
            {
                GDALRasterBand *band = _gdalDataset->GetRasterBand(b);
                if (band->GetColorInterpretation() == GCI_GrayIndex)
                    bandGray = band;
                else if (band->GetColorInterpretation() == GCI_RedBand)
                    bandRed = band;
                else if (band->GetColorInterpretation() == GCI_GreenBand)
                    bandGreen = band;
                else if (band->GetColorInterpretation() == GCI_BlueBand)
                    bandBlue = band;
                else if (band->GetColorInterpretation() == GCI_AlphaBand)
                    bandAlpha = band;
                else if (bandGray == 0)
                    bandGray = band;
            }

            GDALRasterBand *bandSelected = 0;
            if (!bandSelected && bandGray)
                bandSelected = bandGray;
            else if (!bandSelected && bandAlpha)
                bandSelected = bandAlpha;
            else if (!bandSelected && bandRed)
                bandSelected = bandRed;
            else if (!bandSelected && bandGreen)
                bandSelected = bandGreen;
            else if (!bandSelected && bandBlue)
                bandSelected = bandBlue;

            if (bandSelected)
            {

                if (bandSelected->GetUnitType())
                    log(osg::INFO, "bandSelected->GetUnitType()= %d", bandSelected->GetUnitType());
                else
                    log(osg::INFO, "bandSelected->GetUnitType()= null");

                ValidValueOperator validValueOperator(bandSelected);

                int success = 0;
                float offset = bandSelected->GetOffset(&success);
                if (success)
                {
                    log(osg::INFO, "We have Offset = %f", offset);
                }
                else
                {
                    log(osg::INFO, "We have no Offset");
                    offset = 0.0f;
                }

                float scale = bandSelected->GetScale(&success);
                if (success)
                {
                    log(osg::INFO, "We have Scale from GDALRasterBand = %f", scale);
                }

                float datasetScale = destination._dataSet->getVerticalScale();
                log(osg::INFO, "We have Scale from DataSet = %f", datasetScale);

                scale *= datasetScale;
                log(osg::INFO, "We have total scale = %f", scale);

                log(osg::INFO, "********* getLinearUnits = %f", getLinearUnits(_cs.get()));

                // raad the data.
                osg::HeightField *hf = destination._heightField.get();

                float noDataValueFill = 0.0f;
                bool ignoreNoDataValue = true;

                bool interpolateTerrain = destination._dataSet->getUseInterpolatedTerrainSampling();

                if (interpolateTerrain)
                {
                    //Sample terrain at each vert to increase accuracy of the terrain.
                    int endX = destX + destWidth;
                    int endY = destY + destHeight;

                    double orig_X = hf->getOrigin().x();
                    double orig_Y = hf->getOrigin().y();
                    double delta_X = hf->getXInterval();
                    double delta_Y = hf->getYInterval();

                    for (int c = destX; c < endX; ++c)
                    {
                        double geoX = orig_X + (delta_X * (double)c);
                        for (int r = destY; r < endY; ++r)
                        {
                            double geoY = orig_Y + (delta_Y * (double)r);
                            float h = getInterpolatedValue(bandSelected, geoX - xoffset, geoY, hf->getHeight(c, r) / scale);
                            if (!validValueOperator.isNoDataValue(h))
                                hf->setHeight(c, r, offset + h * scale);
                            else if (!ignoreNoDataValue)
                                hf->setHeight(c, r, noDataValueFill);
                        }
                    }
                }
                else
                {
                    // compute dimensions to read from.
                    int windowX = osg::maximum((int)floorf((float)_numValuesX * (intersect_bb.xMin() - xoffset - s_bb.xMin()) / (s_bb.xMax() - s_bb.xMin())), 0);
                    int windowY = osg::maximum((int)floorf((float)_numValuesY * (intersect_bb.yMin() - s_bb.yMin()) / (s_bb.yMax() - s_bb.yMin())), 0);
                    int windowWidth = osg::minimum((int)ceilf((float)_numValuesX * (intersect_bb.xMax() - xoffset - s_bb.xMin()) / (s_bb.xMax() - s_bb.xMin())), (int)_numValuesX) - windowX;
                    int windowHeight = osg::minimum((int)ceilf((float)_numValuesY * (intersect_bb.yMax() - s_bb.yMin()) / (s_bb.yMax() - s_bb.yMin())), (int)_numValuesY) - windowY;

                    log(osg::INFO, "   copying from %d\t%s\t%d\t%d", windowX, windowY, windowWidth, windowHeight);
                    log(osg::INFO, "             to %d\t%s\t%d\t%d", destX, destY, destWidth, destHeight);

                    // read data into temporary array
                    float *heightData = new float[destWidth * destHeight];

                    //bandSelected->RasterIO(GF_Read,windowX,_numValuesY-(windowY+windowHeight),windowWidth,windowHeight,floatdata,destWidth,destHeight,GDT_Float32,numBytesPerZvalue,lineSpace);
                    bandSelected->RasterIO(GF_Read, windowX, _numValuesY - (windowY + windowHeight), windowWidth, windowHeight, heightData, destWidth, destHeight, GDT_Float32, 0, 0);

                    float *heightPtr = heightData;

                    for (int r = destY + destHeight - 1; r >= destY; --r)
                    {
                        for (int c = destX; c < destX + destWidth; ++c)
                        {
                            float h = *heightPtr++;
                            if (!validValueOperator.isNoDataValue(h))
                                hf->setHeight(c, r, offset + h * scale);
                            else if (!ignoreNoDataValue)
                                hf->setHeight(c, r, noDataValueFill);
                            h = hf->getHeight(c, r);
                        }
                    }

                    delete[] heightData;
                }
            }
        }
    }
}

void SourceData::readShapeFile(DestinationData &destination)
{
    if (_model.valid())
    {
        log(osg::INFO, "SourceData::readShapeFile");
        destination._shapeFiles.push_back(_model);
    }
}

void SourceData::readModels(DestinationData &destination)
{
    if (_model.valid())
    {
        log(osg::INFO, "SourceData::readShapeFile");
        destination._models.push_back(_model);
    }
}
