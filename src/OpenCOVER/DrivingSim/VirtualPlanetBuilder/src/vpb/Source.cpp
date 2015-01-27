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

#include <vpb/Source>
#include <vpb/Destination>
#include <vpb/DataSet>
#include <vpb/System>
#include <vpb/BuildOptions>

#include <osg/Geometry>
#include <osg/Notify>
#include <osg/io_utils>

#include <osgDB/ReadFile>
#include <osgDB/FileNameUtils>

#include <cpl_string.h>
#include <gdal_priv.h>
#include <gdalwarper.h>
#include <ogr_spatialref.h>

using namespace vpb;

Source::Source(Type type, osg::Node *model)
    : _type(type)
    , _patchStatus(UNASSIGNED)
    , _revisionNumber(0)
    , _sortValue(0.0)
    , _filename(model->getName())
    , _temporaryFile(false)
    , _coordinateSystemPolicy(PREFER_FILE_SETTINGS)
    , _geoTransformPolicy(PREFER_FILE_SETTINGS)
    , _minLevel(0)
    , _maxLevel(MAXIMUM_NUMBER_OF_LEVELS)
    , _layer(0)
    , _gdalDataset(0)
    , _hfDataset(0)
{
    _sourceData = new SourceData(this);
    _sourceData->_model = model;
}

void Source::setSetName(const std::string &setname, BuildOptions *bo)
{
    _setname = setname;
    if (bo)
    {
        if (bo->isOptionalLayerSet(setname))
            _switchSetName = setname;
        else
            _switchSetName.clear();
    }
}

GeospatialDataset *Source::getOptimumGeospatialDataset(const SpatialProperties &sp, AccessMode accessMode) const
{
    if (_gdalDataset)
        return new GeospatialDataset(_gdalDataset);
    else
        return System::instance()->openOptimumGeospatialDataset(_filename, sp, accessMode);
}

GeospatialDataset *Source::getGeospatialDataset(AccessMode accessMode) const
{
    if (_gdalDataset)
        return new GeospatialDataset(_gdalDataset);
    else
        return System::instance()->openGeospatialDataset(_filename, accessMode);
}

void Source::setGdalDataset(GDALDataset *gdalDataSet)
{
    _gdalDataset = gdalDataSet;
}

GDALDataset *Source::getGdalDataset()
{
    return _gdalDataset;
}

const GDALDataset *Source::getGdalDataset() const
{
    return _gdalDataset;
}

void Source::setHFDataset(osg::HeightField *hfDataSet)
{
    _hfDataset = hfDataSet;
}

osg::HeightField *Source::getHFDataset()
{
    return _hfDataset.get();
}

const osg::HeightField *Source::getHFDataset() const
{
    return _hfDataset.get();
}

void Source::setSortValueFromSourceDataResolution()
{
    if (_sourceData.valid())
    {
        double dx = (_sourceData->_extents.xMax() - _sourceData->_extents.xMin()) / (double)(_sourceData->_numValuesX - 1);
        double dy = (_sourceData->_extents.yMax() - _sourceData->_extents.yMin()) / (double)(_sourceData->_numValuesY - 1);

        setSortValue(sqrt(dx * dx + dy * dy));
    }
}

void Source::setSortValueFromSourceDataResolution(const osg::CoordinateSystemNode *cs)
{
    if (_sourceData.valid())
    {

        const SpatialProperties &sp = _sourceData->computeSpatialProperties(cs);
        setSortValue(sp.computeResolution());
    }
}

void Source::loadSourceData()
{
    log(osg::INFO, "Source::loadSourceData() %s", _filename.c_str());

    if (!_sourceData)
    {

        if (System::instance()->getFileCache())
        {
            osg::ref_ptr<SourceData> sourceData = new SourceData;
            if (System::instance()->getFileCache()->getSpatialProperties(getFileName(), *sourceData))
            {
                log(osg::INFO, "Source::loadSourceData() %s assigned from FileCache", _filename.c_str());

                sourceData->_source = this;
                _sourceData = sourceData;

                assignCoordinateSystemAndGeoTransformAccordingToParameterPolicy();

                return;
            }
        }

        _sourceData = SourceData::readData(this);

        assignCoordinateSystemAndGeoTransformAccordingToParameterPolicy();
    }
}

void Source::assignCoordinateSystemAndGeoTransformAccordingToParameterPolicy()
{
    if (getCoordinateSystemPolicy() == PREFER_CONFIG_SETTINGS)
    {
        _sourceData->_cs = _cs;

        log(osg::INFO, "assigning CS from Source to Data.");
    }
    else
    {
        _cs = _sourceData->_cs;
        log(osg::INFO, "assigning CS from Data to Source.");
    }

    if (getGeoTransformPolicy() == PREFER_CONFIG_SETTINGS)
    {
        _sourceData->_geoTransform = _geoTransform;

        log(osg::INFO, "assigning GeoTransform from Source to Data:");
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(0, 0), _geoTransform(0, 1), _geoTransform(0, 2), _geoTransform(0, 3));
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(1, 0), _geoTransform(1, 1), _geoTransform(1, 2), _geoTransform(1, 3));
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(2, 0), _geoTransform(2, 1), _geoTransform(2, 2), _geoTransform(2, 3));
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(3, 0), _geoTransform(3, 1), _geoTransform(3, 2), _geoTransform(3, 3));
    }
    else if (getGeoTransformPolicy() == PREFER_CONFIG_SETTINGS_BUT_SCALE_BY_FILE_RESOLUTION)
    {

        // scale the x and y axis.
        double div_x;
        double div_y;

        // set up properly for vector and raster (previously always vector)
        if (_dataType == SpatialProperties::VECTOR)
        {
            div_x = 1.0 / (double)(_sourceData->_numValuesX - 1);
            div_y = 1.0 / (double)(_sourceData->_numValuesY - 1);
        }
        else // if (_dataType == SpatialProperties::RASTER)
        {
            div_x = 1.0 / (double)(_sourceData->_numValuesX);
            div_y = 1.0 / (double)(_sourceData->_numValuesY);
        }

#if 1
        _sourceData->_geoTransform = _geoTransform;

        _sourceData->_geoTransform(0, 0) *= div_x;
        _sourceData->_geoTransform(1, 0) *= div_x;
        _sourceData->_geoTransform(2, 0) *= div_x;

        _sourceData->_geoTransform(0, 1) *= div_y;
        _sourceData->_geoTransform(1, 1) *= div_y;
        _sourceData->_geoTransform(2, 1) *= div_y;
#else
        _geoTransform(0, 0) *= div_x;
        _geoTransform(1, 0) *= div_x;
        _geoTransform(2, 0) *= div_x;

        _geoTransform(0, 1) *= div_y;
        _geoTransform(1, 1) *= div_y;
        _geoTransform(2, 1) *= div_y;

        _sourceData->_geoTransform = _geoTransform;
#endif

        log(osg::INFO, "assigning GeoTransform from Source to Data based on file resolution:");
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(0, 0), _geoTransform(0, 1), _geoTransform(0, 2), _geoTransform(0, 3));
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(1, 0), _geoTransform(1, 1), _geoTransform(1, 2), _geoTransform(1, 3));
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(2, 0), _geoTransform(2, 1), _geoTransform(2, 2), _geoTransform(2, 3));
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(3, 0), _geoTransform(3, 1), _geoTransform(3, 2), _geoTransform(3, 3));
    }
    else
    {
        _geoTransform = _sourceData->_geoTransform;
        log(osg::INFO, "assigning GeoTransform from Data to Source:");
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(0, 0), _geoTransform(0, 1), _geoTransform(0, 2), _geoTransform(0, 3));
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(1, 0), _geoTransform(1, 1), _geoTransform(1, 2), _geoTransform(1, 3));
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(2, 0), _geoTransform(2, 1), _geoTransform(2, 2), _geoTransform(2, 3));
        log(osg::INFO, "           %f\t%f\t%f\t%f", _geoTransform(3, 0), _geoTransform(3, 1), _geoTransform(3, 2), _geoTransform(3, 3));
    }

    _sourceData->computeExtents();

    _extents = _sourceData->_extents;
}

bool Source::needReproject(const osg::CoordinateSystemNode *cs) const
{
    return needReproject(cs, 0.0, 0.0);
}

bool Source::needReproject(const osg::CoordinateSystemNode *cs, double minResolution, double maxResolution) const
{
    if (!_sourceData)
        return false;

    // handle modles by using a matrix transform only.
    if (_type == MODEL)
        return false;

    // always need to reproject imagery with GCP's.
    if (_sourceData->_hasGCPs)
    {
        log(osg::INFO, "Need to to reproject due to presence of GCP's");
        return true;
    }

    if (!areCoordinateSystemEquivalent(_cs.get(), cs))
    {
        log(osg::INFO, "Need to do reproject !areCoordinateSystemEquivalent(_cs.get(),cs)");

        return true;
    }

    if (minResolution == 0.0 && maxResolution == 0.0)
        return false;

    // now check resolutions.
    const osg::Matrixd &m = _sourceData->_geoTransform;
    double currentResolution = sqrt(osg::square(m(0, 0)) + osg::square(m(1, 0)) + osg::square(m(0, 1)) + osg::square(m(1, 1)));

    if (currentResolution < minResolution)
        return true;
    if (currentResolution > maxResolution)
        return true;

    return false;
}

Source *Source::doRasterReprojection(const std::string &filename, osg::CoordinateSystemNode *cs, double targetResolution) const
{
    // return nothing when repoject is inappropriate.
    if (!_sourceData)
        return 0;

    if (!isRaster())
    {
        log(osg::NOTICE, "Source::doReprojection() reprojection of a model/shapefile not appropriate.");
        return 0;
    }

    log(osg::NOTICE, "reprojecting to file %s", filename.c_str());

    GDALDriverH hDriver = GDALGetDriverByName("GTiff");

    if (hDriver == NULL)
    {
        log(osg::INFO, "Unable to load driver for GTiff");
        return 0;
    }

    if (GDALGetMetadataItem(hDriver, GDAL_DCAP_CREATE, NULL) == NULL)
    {
        log(osg::INFO, "GDAL driver does not support create for %s", osgDB::getFileExtension(filename).c_str());
        return 0;
    }

    /* -------------------------------------------------------------------- */
    /*      Create a transformation object from the source to               */
    /*      destination coordinate system.                                  */
    /* -------------------------------------------------------------------- */

    osg::ref_ptr<GeospatialDataset> dataset = getGeospatialDataset(READ_ONLY);

    void *hTransformArg = GDALCreateGenImgProjTransformer(dataset->getGDALDataset(), _sourceData->_cs->getCoordinateSystem().c_str(),
                                                          NULL, cs->getCoordinateSystem().c_str(),
                                                          TRUE, 0.0, 1);

    if (!hTransformArg)
    {
        log(osg::INFO, " failed to create transformer");
        return 0;
    }

    double adfDstGeoTransform[6];
    int nPixels = 0, nLines = 0;
    if (GDALSuggestedWarpOutput(dataset->getGDALDataset(),
                                GDALGenImgProjTransform, hTransformArg,
                                adfDstGeoTransform, &nPixels, &nLines)
        != CE_None)
    {
        log(osg::INFO, " failed to create warp");
        return 0;
    }

    if (targetResolution > 0.0f)
    {
        log(osg::INFO, "recomputing the target transform size");

        double currentResolution = sqrt(osg::square(adfDstGeoTransform[1]) + osg::square(adfDstGeoTransform[2]) + osg::square(adfDstGeoTransform[4]) + osg::square(adfDstGeoTransform[5]));

        log(osg::INFO, "        default computed resolution %f nPixels=%d nLines=%d", currentResolution, nPixels, nLines);

        double extentsPixels = sqrt(osg::square(adfDstGeoTransform[1]) + osg::square(adfDstGeoTransform[2])) * (double)(nPixels - 1);
        double extentsLines = sqrt(osg::square(adfDstGeoTransform[4]) + osg::square(adfDstGeoTransform[5])) * (double)(nLines - 1);

        double ratio = targetResolution / currentResolution;
        adfDstGeoTransform[1] *= ratio;
        adfDstGeoTransform[2] *= ratio;
        adfDstGeoTransform[4] *= ratio;
        adfDstGeoTransform[5] *= ratio;

        log(osg::INFO, "    extentsPixels=%d", extentsPixels);
        log(osg::INFO, "    extentsLines=%d", extentsLines);
        log(osg::INFO, "    targetResolution=%d", targetResolution);

        nPixels = (int)ceil(extentsPixels / sqrt(osg::square(adfDstGeoTransform[1]) + osg::square(adfDstGeoTransform[2]))) + 1;
        nLines = (int)ceil(extentsLines / sqrt(osg::square(adfDstGeoTransform[4]) + osg::square(adfDstGeoTransform[5]))) + 1;

        log(osg::INFO, "        target computed resolution %f nPixels=%d nLines=%d", targetResolution, nPixels, nLines);
    }

    GDALDestroyGenImgProjTransformer(hTransformArg);

    GDALDataType eDT = GDALGetRasterDataType(dataset->GetRasterBand(1));

    /* --------------------------------------------------------------------- */
    /*    Create the file                                                    */
    /* --------------------------------------------------------------------- */

    int numSourceBands = dataset->GetRasterCount();
    int numDestinationBands = (numSourceBands >= 3) ? 4 : numSourceBands; // expand RGB to RGBA, but leave other formats unchanged

    char **papszOptions = NULL;

    papszOptions = CSLSetNameValue(papszOptions, "TILED", "YES");
    papszOptions = CSLSetNameValue(papszOptions, "COMPRESS", "PACKBITS");

    GDALDatasetH hDstDS = GDALCreate(hDriver, filename.c_str(), nPixels, nLines,
                                     numDestinationBands, eDT,
                                     papszOptions);

    if (hDstDS == NULL)
        return NULL;

    /* -------------------------------------------------------------------- */
    /*      Write out the projection definition.                            */
    /* -------------------------------------------------------------------- */
    GDALSetProjection(hDstDS, cs->getCoordinateSystem().c_str());
    GDALSetGeoTransform(hDstDS, adfDstGeoTransform);

    // Set up the transformer along with the new datasets.

    hTransformArg = GDALCreateGenImgProjTransformer(dataset->getGDALDataset(), _sourceData->_cs->getCoordinateSystem().c_str(),
                                                    hDstDS, cs->getCoordinateSystem().c_str(),
                                                    TRUE, 0.0, 1);

    GDALTransformerFunc pfnTransformer = GDALGenImgProjTransform;

    log(osg::INFO, "Setting projection %s", cs->getCoordinateSystem().c_str());

    /* -------------------------------------------------------------------- */
    /*      Copy the color table, if required.                              */
    /* -------------------------------------------------------------------- */
    GDALColorTableH hCT;

    hCT = GDALGetRasterColorTable(dataset->GetRasterBand(1));
    if (hCT != NULL)
        GDALSetRasterColorTable(GDALGetRasterBand(hDstDS, 1), hCT);

    /* -------------------------------------------------------------------- */
    /*      Setup warp options.                                             */
    /* -------------------------------------------------------------------- */
    GDALWarpOptions *psWO = GDALCreateWarpOptions();

    psWO->hSrcDS = dataset->getGDALDataset();
    psWO->hDstDS = hDstDS;

    psWO->pfnTransformer = pfnTransformer;
    psWO->pTransformerArg = hTransformArg;

    psWO->pfnProgress = GDALTermProgress;

    /* -------------------------------------------------------------------- */
    /*      Setup band mapping.                                             */
    /* -------------------------------------------------------------------- */
    psWO->nBandCount = numSourceBands; //numDestinationBands;
    psWO->panSrcBands = (int *)CPLMalloc(numDestinationBands * sizeof(int));
    psWO->panDstBands = (int *)CPLMalloc(numDestinationBands * sizeof(int));

    int i;
    for (i = 0; i < psWO->nBandCount; i++)
    {
        psWO->panSrcBands[i] = i + 1;
        psWO->panDstBands[i] = i + 1;
    }

    /* -------------------------------------------------------------------- */
    /*      Setup no datavalue                                              */
    /* -----------------------------------------------------`--------------- */

    psWO->padfSrcNoDataReal = (double *)CPLMalloc(psWO->nBandCount * sizeof(double));
    psWO->padfSrcNoDataImag = (double *)CPLMalloc(psWO->nBandCount * sizeof(double));

    psWO->padfDstNoDataReal = (double *)CPLMalloc(psWO->nBandCount * sizeof(double));
    psWO->padfDstNoDataImag = (double *)CPLMalloc(psWO->nBandCount * sizeof(double));

    for (i = 0; i < psWO->nBandCount; i++)
    {
        int success = 0;
        GDALRasterBand *band = (i < numSourceBands) ? dataset->GetRasterBand(i + 1) : 0;
        double noDataValue = band ? band->GetNoDataValue(&success) : 0.0;
        double new_noDataValue = 0;
        if (success)
        {
            log(osg::INFO, "\tassinging no data value %f to band %d", noDataValue, i + 1);

            psWO->padfSrcNoDataReal[i] = noDataValue;
            psWO->padfSrcNoDataImag[i] = 0.0;
            psWO->padfDstNoDataReal[i] = new_noDataValue;
            psWO->padfDstNoDataImag[i] = 0.0;

            GDALRasterBandH dest_band = GDALGetRasterBand(hDstDS, i + 1);
            GDALSetRasterNoDataValue(dest_band, new_noDataValue);
        }
        else
        {
            psWO->padfSrcNoDataReal[i] = 0.0;
            psWO->padfSrcNoDataImag[i] = 0.0;
            psWO->padfDstNoDataReal[i] = new_noDataValue;
            psWO->padfDstNoDataImag[i] = 0.0;

            GDALRasterBandH dest_band = GDALGetRasterBand(hDstDS, i + 1);
            GDALSetRasterNoDataValue(dest_band, new_noDataValue);
        }
    }

    psWO->papszWarpOptions = (char **)CPLMalloc(2 * sizeof(char *));
    psWO->papszWarpOptions[0] = strdup("INIT_DEST=NO_DATA");
    psWO->papszWarpOptions[1] = 0;

    if (numDestinationBands == 4)
    {
        /*
        GDALSetRasterColorInterpretation( 
            GDALGetRasterBand( hDstDS, numDestinationBands ), 
            GCI_AlphaBand );
*/
        psWO->nDstAlphaBand = numDestinationBands;
    }

    /* -------------------------------------------------------------------- */
    /*      Initialize and execute the warp.                                */
    /* -------------------------------------------------------------------- */
    GDALWarpOperation oWO;

    if (oWO.Initialize(psWO) == CE_None)
    {
        bool multithreaded = false;

        if (multithreaded)
        {
            oWO.ChunkAndWarpMulti(0, 0,
                                  GDALGetRasterXSize(hDstDS),
                                  GDALGetRasterYSize(hDstDS));
        }
        else
        {
            oWO.ChunkAndWarpImage(0, 0,
                                  GDALGetRasterXSize(hDstDS),
                                  GDALGetRasterYSize(hDstDS));
        }
    }

    log(osg::INFO, "new projection is %s", GDALGetProjectionRef(hDstDS));

    /* -------------------------------------------------------------------- */
    /*      Cleanup.                                                        */
    /* -------------------------------------------------------------------- */
    GDALDestroyGenImgProjTransformer(hTransformArg);

#if 0
    int anOverviewList[4] = { 2, 4, 8, 16 };
    GDALBuildOverviews( hDstDS, "AVERAGE", 4, anOverviewList, 0, NULL, 
                            GDALTermProgress/*GDALDummyProgress*/, NULL );
#endif

    GDALClose(hDstDS);

    Source *newSource = new Source;
    newSource->_type = _type;
    newSource->_filename = filename;
    newSource->_temporaryFile = true;
    newSource->_cs = cs;

    newSource->_coordinateSystemPolicy = _coordinateSystemPolicy;
    newSource->_geoTransformPolicy = _geoTransformPolicy;

    newSource->_minLevel = _minLevel;
    newSource->_maxLevel = _maxLevel;
    newSource->_layer = _layer;

    newSource->_requiredResolutions = _requiredResolutions;

    newSource->_numValuesX = nPixels;
    newSource->_numValuesY = nLines;
    newSource->_geoTransform.set(adfDstGeoTransform[1], adfDstGeoTransform[4], 0.0, 0.0,
                                 adfDstGeoTransform[2], adfDstGeoTransform[5], 0.0, 0.0,
                                 0.0, 0.0, 1.0, 0.0,
                                 adfDstGeoTransform[0], adfDstGeoTransform[3], 0.0, 1.0);

    newSource->computeExtents();

    // reload the newly created file.
    newSource->loadSourceData();

    return newSource;
}

Source *Source::doRasterReprojectionUsingFileCache(osg::CoordinateSystemNode *cs)
{
    FileCache *fileCache = System::instance()->getFileCache();
    if (!fileCache)
        return 0;

    // see if we can use the FileCache to remap the source file.
    std::string optimumFile = fileCache->getOptimimumFile(getFileName(), cs);
    if (!optimumFile.empty())
    {
        Source *newSource = new Source;

        newSource->_type = _type;
        newSource->_filename = optimumFile;
        newSource->_temporaryFile = false;
        newSource->_cs = cs;

        newSource->_coordinateSystemPolicy = _coordinateSystemPolicy;
        newSource->_geoTransformPolicy = _geoTransformPolicy;

        newSource->_minLevel = _minLevel;
        newSource->_maxLevel = _maxLevel;
        newSource->_layer = _layer;

        newSource->_requiredResolutions = _requiredResolutions;

        // reaload the new file
        newSource->loadSourceData();

        return newSource;
    }
    return 0;
}

class ReprojectionVisitor : public osg::NodeVisitor
{
public:
    ReprojectionVisitor()
        : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
    {
    }

    void reset()
    {
        _geometries.clear();
    }

    void apply(osg::Geode &geode)
    {
        for (unsigned int i = 0; i < geode.getNumDrawables(); ++i)
        {
            osg::Geometry *geometry = dynamic_cast<osg::Geometry *>(geode.getDrawable(i));
            if (geometry)
                _geometries.insert(geometry);
            else
                log(osg::NOTICE, "Warning: ReprojectionVisitor unable to reproject non standard Drawables.");
        }
    }

    bool transform(osg::CoordinateSystemNode *sourceCS, osg::CoordinateSystemNode *destinationCS)
    {
        if (!sourceCS)
        {
            log(osg::NOTICE, "Warning: no source coordinate system to reproject from.");
            return false;
        }

        if (!destinationCS)
        {
            log(osg::NOTICE, "Warning: no target coordinate system to reproject from.");
            return false;
        }

        char *source_projection_string = strdup(sourceCS->getCoordinateSystem().c_str());
        char *importString = source_projection_string;
        OGRSpatialReference *sourceProjection = new OGRSpatialReference;
        sourceProjection->importFromWkt(&importString);

        char *destination_projection_string = strdup(destinationCS->getCoordinateSystem().c_str());
        importString = destination_projection_string;
        OGRSpatialReference *destinationProjection = new OGRSpatialReference;
        destinationProjection->importFromWkt(&importString);

        OGRCoordinateTransformation *ct = OGRCreateCoordinateTransformation(sourceProjection, destinationProjection);

        bool success = false;

        if (ct)
        {
            success = transform(ct);

            delete ct;
        }

        delete destinationProjection;
        delete sourceProjection;

        free(destination_projection_string);
        free(source_projection_string);

        return success;
    }

    bool transform(OGRCoordinateTransformation *ct)
    {
        for (Geometries::iterator itr = _geometries.begin();
             itr != _geometries.end();
             ++itr)
        {
            transform(ct, const_cast<osg::Geometry *>(*itr));
        }

        return true;
    }

    bool transform(OGRCoordinateTransformation *ct, osg::Geometry *geometry)
    {
        osg::Vec3dArray *vec3darray = dynamic_cast<osg::Vec3dArray *>(geometry->getVertexArray());
        osg::Vec3Array *vec3farray = (vec3darray != 0) ? 0 : dynamic_cast<osg::Vec3Array *>(geometry->getVertexArray());

        unsigned int nCount = geometry->getVertexArray()->getNumElements();

        // if no data to work with return;
        if ((!vec3darray && !vec3farray) || nCount == 0)
            return true;

        double *xArray = new double[nCount];
        double *yArray = new double[nCount];
        double *zArray = new double[nCount];

        // copy the soure array to the temporary arrays
        if (vec3darray)
        {
            for (unsigned int i = 0; i < nCount; ++i)
            {
                osg::Vec3d &v = (*vec3darray)[i];
                xArray[i] = v.x();
                yArray[i] = v.y();
                zArray[i] = v.z();
            }
        }
        else if (vec3farray)
        {
            for (unsigned int i = 0; i < nCount; ++i)
            {
                osg::Vec3 &v = (*vec3farray)[i];
                xArray[i] = v.x();
                yArray[i] = v.y();
                zArray[i] = v.z();
            }
        }

        // log(osg::NOTICE,"   reprojecting %i vertices",nCount);

        ct->Transform(nCount, xArray, yArray, zArray);

        // copy the data back to the original arrays.
        if (vec3darray)
        {
            for (unsigned int i = 0; i < nCount; ++i)
            {
                osg::Vec3d &v = (*vec3darray)[i];
                //osg::notify(osg::NOTICE)<<"   Before "<<v;
                v.x() = xArray[i];
                v.y() = yArray[i];
                v.z() = zArray[i];
                // osg::notify(osg::NOTICE)<<"  after "<<v<<std::endl;
            }
        }
        else if (vec3farray)
        {
            for (unsigned int i = 0; i < nCount; ++i)
            {
                osg::Vec3 &v = (*vec3farray)[i];
                v.x() = xArray[i];
                v.y() = yArray[i];
                v.z() = zArray[i];
            }
        }

        // clean up the temporary arrays
        delete[] xArray;
        delete[] yArray;
        delete[] zArray;

        return true;
    }

    typedef std::set<osg::Geometry *> Geometries;
    Geometries _geometries;
};

bool Source::do3DObjectReprojection(osg::CoordinateSystemNode *cs)
{
    // return if we aren't a 3D object
    if (!is3DObject())
        return false;

    // return if there is nothing to work on.
    if (!_sourceData || !_sourceData->_model)
        return false;

    log(osg::NOTICE, "Source::do3DObjectReprojectionUsingFileCache(), should be setting projection to %s", cs->getCoordinateSystem().c_str());

    ReprojectionVisitor rpv;

    // collect all the geoemtries of interest
    _sourceData->_model->accept(rpv);

    if (rpv.transform(_cs.get(), cs))
    {
        _cs = cs;
        // need to recompute extents.

        return true;
    }
    else
    {
        return false;
    }
}

void Source::consolodateRequiredResolutions()
{
    if (_requiredResolutions.size() <= 1)
        return;

    ResolutionList consolodated;

    ResolutionList::iterator itr = _requiredResolutions.begin();

    double minResX = itr->_resX;
    double minResY = itr->_resY;
    double maxResX = itr->_resX;
    double maxResY = itr->_resY;
    ++itr;
    for (; itr != _requiredResolutions.end(); ++itr)
    {
        minResX = osg::minimum(minResX, itr->_resX);
        minResY = osg::minimum(minResY, itr->_resY);
        maxResX = osg::maximum(maxResX, itr->_resX);
        maxResY = osg::maximum(maxResY, itr->_resY);
    }

    double currResX = minResX;
    double currResY = minResY;
    while (currResX <= maxResX && currResY <= maxResY)
    {
        consolodated.push_back(ResolutionPair(currResX, currResY));
        currResX *= 2.0f;
        currResY *= 2.0f;
    }

    _requiredResolutions.swap(consolodated);
}

void Source::buildOverviews()
{
    osg::ref_ptr<GeospatialDataset> dataset = getGeospatialDataset(READ_AND_WRITE);
    if (dataset.valid())
    {
        int anOverviewList[5] = { 2, 4, 8, 16, 32 };
        dataset->BuildOverviews("AVERAGE", 4, anOverviewList, 0, NULL,
                                GDALTermProgress /*GDALDummyProgress*/, NULL);
    }
}

template <class T>
struct DerefLessFunctor
{
    bool operator()(const T &lhs, const T &rhs)
    {
        if (!lhs || !rhs)
            return lhs < rhs;
        if (lhs->getLayer() < rhs->getLayer())
            return true;
        if (rhs->getLayer() < lhs->getLayer())
            return false;
        return (lhs->getSortValue() > rhs->getSortValue());
    }
};

void CompositeSource::setSortValueFromSourceDataResolution()
{
    for (SourceList::iterator sitr = _sourceList.begin(); sitr != _sourceList.end(); ++sitr)
    {
        (*sitr)->setSortValueFromSourceDataResolution();
    }

    for (ChildList::iterator citr = _children.begin(); citr != _children.end(); ++citr)
    {
        (*citr)->setSortValueFromSourceDataResolution();
    }
}

void CompositeSource::setSortValueFromSourceDataResolution(const osg::CoordinateSystemNode *cs)
{
    for (SourceList::iterator sitr = _sourceList.begin(); sitr != _sourceList.end(); ++sitr)
    {
        (*sitr)->setSortValueFromSourceDataResolution(cs);
    }

    for (ChildList::iterator citr = _children.begin(); citr != _children.end(); ++citr)
    {
        (*citr)->setSortValueFromSourceDataResolution(cs);
    }
}

void CompositeSource::sortBySourceSortValue()
{
    // sort the sources.
    std::sort(_sourceList.begin(), _sourceList.end(), DerefLessFunctor<osg::ref_ptr<Source> >());

    // sort the composite sources internal data
    for (ChildList::iterator itr = _children.begin(); itr != _children.end(); ++itr)
    {
        if (itr->valid())
            (*itr)->sortBySourceSortValue();
    }
}

template <class T>
struct DerefLessSourceDetailsFunctor
{
    bool operator()(const T &lhs, const T &rhs)
    {
        if (!lhs || !rhs)
            return lhs < rhs;

        if (lhs->getType() < rhs->getType())
            return true;
        if (rhs->getType() < lhs->getType())
            return false;

        if (lhs->getFileName() < rhs->getFileName())
            return true;
        if (rhs->getFileName() < lhs->getFileName())
            return false;

        if (lhs->getRevisionNumber() < rhs->getRevisionNumber())
            return true;
        if (rhs->getRevisionNumber() < lhs->getRevisionNumber())
            return false;

        if (lhs->getLayer() < rhs->getLayer())
            return true;
        if (rhs->getLayer() < lhs->getLayer())
            return false;

        return (lhs->getSortValue() > rhs->getSortValue());
    }
};

void CompositeSource::sortBySourceDetails()
{
    // sort the sources.
    std::sort(_sourceList.begin(), _sourceList.end(), DerefLessSourceDetailsFunctor<osg::ref_ptr<Source> >());

    // sort the composite sources internal data
    for (ChildList::iterator itr = _children.begin(); itr != _children.end(); ++itr)
    {
        if (itr->valid())
            (*itr)->sortBySourceDetails();
    }
}

void CompositeSource::assignSourcePatchStatus()
{
    if (_sourceList.empty())
        return;

    sortBySourceDetails();

    SourceList::iterator itr = _sourceList.begin();
    unsigned int lowerRevisionNumber = (*itr)->getRevisionNumber();
    unsigned int upperRevisionNumber = (*itr)->getRevisionNumber();
    ++itr;
    for (;
         itr != _sourceList.end();
         ++itr)
    {
        Source *current = itr->get();
        if (current->getRevisionNumber() > upperRevisionNumber)
            upperRevisionNumber = current->getRevisionNumber();
        if (current->getRevisionNumber() < lowerRevisionNumber)
            lowerRevisionNumber = current->getRevisionNumber();
    }

    if (lowerRevisionNumber == upperRevisionNumber)
    {
        osg::notify(osg::NOTICE) << "Equal revision numbers, " << lowerRevisionNumber << " and " << upperRevisionNumber << std::endl;
        return;
    }

    itr = _sourceList.begin();
    Source *previous = itr->get();
    ++itr;
    for (;
         itr != _sourceList.end();
         ++itr)
    {
        Source *current = itr->get();
        if (previous->getType() == current->getType() && previous->getFileName() == current->getFileName())
        {
            previous->setPatchStatus(Source::UNCHANGED);
            current->setPatchStatus(Source::UNCHANGED);
        }

        previous = current;
    }

    itr = _sourceList.begin();
    for (;
         itr != _sourceList.end();
         ++itr)
    {
        Source *current = itr->get();
        if (current && current->getPatchStatus() == Source::UNASSIGNED)
        {
            if (current->getRevisionNumber() == lowerRevisionNumber)
                current->setPatchStatus(Source::REMOVED);
            else if (current->getRevisionNumber() == upperRevisionNumber)
                current->setPatchStatus(Source::ADDED);
        }
    }

    // sort the composite sources internal data
    for (ChildList::iterator itr = _children.begin(); itr != _children.end(); ++itr)
    {
        if (itr->valid())
            (*itr)->assignSourcePatchStatus();
    }
}

unsigned int CompositeSource::getNumberAlteredSources()
{
    unsigned int number = 0;

    for (SourceList::iterator itr = _sourceList.begin();
         itr != _sourceList.end();
         ++itr)
    {
        Source *current = itr->get();
        if (current->getPatchStatus() != Source::UNCHANGED)
            ++number;
    }

    // sort the composite sources internal data
    for (ChildList::iterator itr = _children.begin(); itr != _children.end(); ++itr)
    {
        number += (*itr)->getNumberAlteredSources();
    }

    return number;
}
