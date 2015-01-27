/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- VirtualPlanetBuilder - Copyright (C) 1998-2009 Robert Osfield
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

#include <vpb/BuildOptions>

#include <osgDB/FileNameUtils>

using namespace vpb;

///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// ImageOptions
//
ImageOptions::ImageOptions()
    : osg::Object(true)
{
    _imageryQuantization = 0;
    _imageryErrorDiffusion = false;
    _maxAnisotropy = 1.0;
    _defaultColor.set(0.5f, 0.5f, 1.0f, 1.0f);
    _useInterpolatedImagerySampling = true;
    _imageExtension = ".dds";
    _powerOfTwoImages = true;
    _textureType = COMPRESSED_TEXTURE;
    _maximumTileImageSize = 256;
    _mipMappingMode = MIP_MAPPING_IMAGERY;
}

ImageOptions::ImageOptions(const ImageOptions &rhs, const osg::CopyOp &copyop)
    : osg::Object(rhs, copyop)
{
    setImageOptions(rhs);
}

ImageOptions &ImageOptions::operator=(const ImageOptions &rhs)
{
    if (this == &rhs)
        return *this;

    setImageOptions(rhs);

    return *this;
}

void ImageOptions::setImageOptions(const ImageOptions &rhs)
{
    _defaultColor = rhs._defaultColor;
    _useInterpolatedImagerySampling = rhs._useInterpolatedImagerySampling;
    _imageExtension = rhs._imageExtension;
    _powerOfTwoImages = rhs._powerOfTwoImages;
    _imageryQuantization = rhs._imageryQuantization;
    _imageryErrorDiffusion = rhs._imageryErrorDiffusion;
    _maxAnisotropy = rhs._maxAnisotropy;
    _maximumTileImageSize = rhs._maximumTileImageSize;
    _mipMappingMode = rhs._mipMappingMode;
    _textureType = rhs._textureType;
}

bool ImageOptions::compatible(ImageOptions &rhs) const
{
    if (_defaultColor != rhs._defaultColor)
        return false;
    if (_useInterpolatedImagerySampling != rhs._useInterpolatedImagerySampling)
        return false;
    if (_imageExtension != rhs._imageExtension)
        return false;
    if (_powerOfTwoImages != rhs._powerOfTwoImages)
        return false;
    if (_imageryQuantization != rhs._imageryQuantization)
        return false;
    if (_imageryErrorDiffusion != rhs._imageryErrorDiffusion)
        return false;
    if (_maxAnisotropy != rhs._maxAnisotropy)
        return false;
    if (_maximumTileImageSize != rhs._maximumTileImageSize)
        return false;
    if (_mipMappingMode != rhs._mipMappingMode)
        return false;
    if (_textureType != rhs._textureType)
        return false;
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// BuildOptions
//
BuildOptions::BuildOptions()
//    :osg::Object(true)
{
    _archiveName = "";
    _buildOverlays = false;
    _reprojectSources = true;
    _generateTiles = true;
    _comment = "";
    _convertFromGeographicToGeocentric = false;
    _databaseType = PagedLOD_DATABASE;
    _decorateWithCoordinateSystemNode = true;
    _decorateWithMultiTextureControl = true;
    _useInterpolatedTerrainSampling = true;
    _destinationCoordinateSystemString = "";
    _destinationCoordinateSystem = new osg::CoordinateSystemNode;
    _destinationCoordinateSystem->setEllipsoidModel(new osg::EllipsoidModel);
    _directory = "";
    _outputTaskDirectories = true;
    _geometryType = TERRAIN;
    _intermediateBuildName = "";
    _logFileName = "";
    _taskFileName = "";
    _maximumNumOfLevels = 30;
    _maximumTileTerrainSize = 64;
    _maximumVisiableDistanceOfTopLevel = 1e10;
    _radiusToMaxVisibleDistanceRatio = 7.0f;
    _simplifyTerrain = true;
    _skirtRatio = 0.02f;
    _tileBasename = "output";
    _tileExtension = ".osgb";
    _useLocalTileTransform = true;
    _verticalScale = 1.0f;
    _writeNodeBeforeSimplification = false;

    _distributedBuildSplitLevel = 0;
    _distributedBuildSecondarySplitLevel = 0;
    _recordSubtileFileNamesOnLeafTile = false;
    _generateSubtile = false;
    _subtileLevel = 0;
    _subtileX = 0;
    _subtileY = 0;

    _notifyLevel = NOTICE;
    _disableWrites = false;

    _numReadThreadsToCoresRatio = 0.0f;
    _numWriteThreadsToCoresRatio = 0.0f;

    _layerInheritance = INHERIT_NEAREST_AVAILABLE;

    _abortTaskOnError = true;
    _abortRunOnError = false;

    _defaultImageLayerOutputPolicy = INLINE;
    _defaultElevationLayerOutputPolicy = INLINE;
    _optionalImageLayerOutputPolicy = EXTERNAL_SET_DIRECTORY;
    _optionalElevationLayerOutputPolicy = EXTERNAL_SET_DIRECTORY;

    _revisionNumber = 0;

    _blendingPolicy = INHERIT;

    _compressionMethod = GL_DRIVER;
    _compressionQuality = FASTEST;
}

BuildOptions::BuildOptions(const BuildOptions &rhs, const osg::CopyOp &copyop)
//:osg::Object(rhs,copyop)
{
    setBuildOptions(rhs);
}

BuildOptions &BuildOptions::operator=(const BuildOptions &rhs)
{
    if (this == &rhs)
        return *this;

    setBuildOptions(rhs);

    return *this;
}

void BuildOptions::setBuildOptions(const BuildOptions &rhs)
{
    setImageOptions(rhs);

    _archiveName = rhs._archiveName;
    _buildOverlays = rhs._buildOverlays;
    _reprojectSources = rhs._reprojectSources;
    _generateTiles = rhs._generateTiles;
    _comment = rhs._comment;
    _convertFromGeographicToGeocentric = rhs._convertFromGeographicToGeocentric;
    _databaseType = rhs._databaseType;
    _decorateWithCoordinateSystemNode = rhs._decorateWithCoordinateSystemNode;
    _decorateWithMultiTextureControl = rhs._decorateWithMultiTextureControl;
    _useInterpolatedTerrainSampling = rhs._useInterpolatedTerrainSampling;
    _destinationCoordinateSystemString = rhs._destinationCoordinateSystemString;
    _destinationCoordinateSystem = rhs._destinationCoordinateSystem;
    _directory = rhs._directory;
    _outputTaskDirectories = rhs._outputTaskDirectories;
    _extents = rhs._extents;
    _geometryType = rhs._geometryType;
    _intermediateBuildName = rhs._intermediateBuildName;
    _logFileName = rhs._logFileName;
    _taskFileName = rhs._taskFileName;
    _maximumNumOfLevels = rhs._maximumNumOfLevels;
    _maximumTileTerrainSize = rhs._maximumTileTerrainSize;
    _maximumVisiableDistanceOfTopLevel = rhs._maximumVisiableDistanceOfTopLevel;
    _radiusToMaxVisibleDistanceRatio = rhs._radiusToMaxVisibleDistanceRatio;
    _simplifyTerrain = rhs._simplifyTerrain;
    _skirtRatio = rhs._skirtRatio;
    _tileBasename = rhs._tileBasename;
    _tileExtension = rhs._tileExtension;
    _useLocalTileTransform = rhs._useLocalTileTransform;
    _verticalScale = rhs._verticalScale;
    _writeNodeBeforeSimplification = rhs._writeNodeBeforeSimplification;
    _distributedBuildSplitLevel = rhs._distributedBuildSplitLevel;
    _distributedBuildSecondarySplitLevel = rhs._distributedBuildSecondarySplitLevel;
    _recordSubtileFileNamesOnLeafTile = rhs._recordSubtileFileNamesOnLeafTile;
    _generateSubtile = rhs._generateSubtile;
    _subtileLevel = rhs._subtileLevel;
    _subtileX = rhs._subtileX;
    _subtileY = rhs._subtileY;

    _notifyLevel = rhs._notifyLevel;
    _disableWrites = rhs._disableWrites;

    _numReadThreadsToCoresRatio = rhs._numReadThreadsToCoresRatio;
    _numWriteThreadsToCoresRatio = rhs._numWriteThreadsToCoresRatio;

    _buildOptionsString = rhs._buildOptionsString;
    _writeOptionsString = rhs._writeOptionsString;

    _layerInheritance = rhs._layerInheritance;

    _abortTaskOnError = rhs._abortTaskOnError;
    _abortRunOnError = rhs._abortRunOnError;

    _defaultImageLayerOutputPolicy = rhs._defaultImageLayerOutputPolicy;
    _defaultElevationLayerOutputPolicy = rhs._defaultElevationLayerOutputPolicy;
    _optionalImageLayerOutputPolicy = rhs._optionalImageLayerOutputPolicy;
    _optionalElevationLayerOutputPolicy = rhs._optionalElevationLayerOutputPolicy;

    _optionalLayerSet = rhs._optionalLayerSet;

    _revisionNumber = rhs._revisionNumber;

    _blendingPolicy = rhs._blendingPolicy;

    _compressionMethod = rhs._compressionMethod;

    _imageOptions.clear();
    for (unsigned int i = 0; i < rhs.getNumLayerImageOptions(); ++i)
    {
        _imageOptions.push_back(rhs.getLayerImageOptions(i) ? osg::clone(rhs.getLayerImageOptions(i)) : 0);
    }
}

void BuildOptions::setDestinationName(const std::string &filename)
{
    std::string path = osgDB::getFilePath(filename);
    std::string base = osgDB::getStrippedName(filename);
    std::string extension = '.' + osgDB::getLowerCaseFileExtension(filename);

    osg::notify(osg::INFO) << "setDestinationName(" << filename << ")" << std::endl;
    osg::notify(osg::INFO) << "   path " << path << std::endl;
    osg::notify(osg::INFO) << "   base " << base << std::endl;
    osg::notify(osg::INFO) << "   extension " << extension << std::endl;

    setDirectory(path);
    setDestinationTileBaseName(base);
    setDestinationTileExtension(extension);
}

void BuildOptions::setDirectory(const std::string &directory)
{
    _directory = directory;

    if (_directory.empty())
        return;

#ifdef WIN32
    // convert trailing forward slash if any to back slash.
    if (_directory[_directory.size() - 1] == '/')
        _directory[_directory.size() - 1] = '\\';

    // if no trailing back slash exists add one.
    if (_directory[_directory.size() - 1] != '\\')
        _directory.push_back('\\');
#else
    // convert trailing back slash if any to forward slash.
    if (_directory[_directory.size() - 1] == '\\')
        _directory[_directory.size() - 1] = '/';

    // if no trailing forward slash exists add one.
    if (_directory[_directory.size() - 1] != '/')
        _directory.push_back('/');
#endif

    osg::notify(osg::INFO) << "directory name set " << _directory << std::endl;
}

void BuildOptions::setDestinationCoordinateSystem(const std::string &wellKnownText)
{
    _destinationCoordinateSystemString = wellKnownText;
    setDestinationCoordinateSystemNode(new osg::CoordinateSystemNode("WKT", wellKnownText));
}

void BuildOptions::setDestinationCoordinateSystemNode(osg::CoordinateSystemNode *cs)
{
    // Keep any settings from the already configured ellipsoid model
    osg::ref_ptr<osg::EllipsoidModel> configuredEllipsoid = _destinationCoordinateSystem->getEllipsoidModel();

    _destinationCoordinateSystem = cs;

    if (_destinationCoordinateSystem.valid())
    {
        if (_destinationCoordinateSystem->getEllipsoidModel() == 0)
        {
            _destinationCoordinateSystem->setEllipsoidModel(configuredEllipsoid.get() ? configuredEllipsoid.get() : new osg::EllipsoidModel);
        }

        _destinationCoordinateSystemString = _destinationCoordinateSystem->getCoordinateSystem();
    }
    else
    {
        _destinationCoordinateSystemString.clear();
    }
}

void BuildOptions::setNotifyLevel(NotifyLevel level)
{
    _notifyLevel = level;
}

void BuildOptions::setNotifyLevel(const std::string &notifyLevel)
{
    if (notifyLevel == "ALWAYS")
        setNotifyLevel(ALWAYS);
    else if (notifyLevel == "DISABLE")
        setNotifyLevel(ALWAYS);
    else if (notifyLevel == "OFF")
        setNotifyLevel(ALWAYS);
    else if (notifyLevel == "FATAL")
        setNotifyLevel(FATAL);
    else if (notifyLevel == "WARN")
        setNotifyLevel(WARN);
    else if (notifyLevel == "NOTICE")
        setNotifyLevel(NOTICE);
    else if (notifyLevel == "INFO")
        setNotifyLevel(INFO);
    else if (notifyLevel == "DEBUG")
        setNotifyLevel(DEBUG_INFO);
    else if (notifyLevel == "DEBUG_INFO")
        setNotifyLevel(DEBUG_INFO);
    else if (notifyLevel == "DEBUG_FP")
        setNotifyLevel(DEBUG_FP);
}

bool BuildOptions::compatible(BuildOptions &rhs) const
{
    // if (!ImageOptions::compatible(rhs)) return false;

    if (_archiveName != rhs._archiveName)
        return false;
    if (_buildOverlays != rhs._buildOverlays)
        return false;
    if (_reprojectSources != rhs._reprojectSources)
        return false;
    if (_generateTiles != rhs._generateTiles)
        return false;
    if (_convertFromGeographicToGeocentric != rhs._convertFromGeographicToGeocentric)
        return false;
    if (_databaseType != rhs._databaseType)
        return false;
    if (_decorateWithCoordinateSystemNode != rhs._decorateWithCoordinateSystemNode)
        return false;
    if (_decorateWithMultiTextureControl != rhs._decorateWithMultiTextureControl)
        return false;
    if (_useInterpolatedTerrainSampling != rhs._useInterpolatedTerrainSampling)
        return false;
    if (_destinationCoordinateSystemString != rhs._destinationCoordinateSystemString)
        return false;
    if (_directory != rhs._directory)
        return false;
    if (_outputTaskDirectories != rhs._outputTaskDirectories)
        return false;
    if (_extents != rhs._extents)
        return false;
    if (_geometryType != rhs._geometryType)
        return false;
    if (_intermediateBuildName != rhs._intermediateBuildName)
        return false;
    if (_logFileName != rhs._logFileName)
        return false;
    if (_taskFileName != rhs._taskFileName)
        return false;
    if (_maximumNumOfLevels != rhs._maximumNumOfLevels)
        return false;
    if (_maximumTileTerrainSize != rhs._maximumTileTerrainSize)
        return false;
    if (_maximumVisiableDistanceOfTopLevel != rhs._maximumVisiableDistanceOfTopLevel)
        return false;
    if (_radiusToMaxVisibleDistanceRatio != rhs._radiusToMaxVisibleDistanceRatio)
        return false;
    if (_simplifyTerrain != rhs._simplifyTerrain)
        return false;
    if (_skirtRatio != rhs._skirtRatio)
        return false;
    if (_tileBasename != rhs._tileBasename)
        return false;
    if (_tileExtension != rhs._tileExtension)
        return false;
    if (_useLocalTileTransform != rhs._useLocalTileTransform)
        return false;
    if (_verticalScale != rhs._verticalScale)
        return false;
    if (_writeNodeBeforeSimplification != rhs._writeNodeBeforeSimplification)
        return false;
    if (_distributedBuildSplitLevel != rhs._distributedBuildSplitLevel)
        return false;
    if (_distributedBuildSecondarySplitLevel != rhs._distributedBuildSecondarySplitLevel)
        return false;
    if (_recordSubtileFileNamesOnLeafTile != rhs._recordSubtileFileNamesOnLeafTile)
        return false;
    if (_generateSubtile != rhs._generateSubtile)
        return false;
    if (_subtileLevel != rhs._subtileLevel)
        return false;
    if (_subtileX != rhs._subtileX)
        return false;
    if (_subtileY != rhs._subtileY)
        return false;

    if (_disableWrites != rhs._disableWrites)
        return false;

    if (_numReadThreadsToCoresRatio != rhs._numReadThreadsToCoresRatio)
        return false;
    if (_numWriteThreadsToCoresRatio != rhs._numWriteThreadsToCoresRatio)
        return false;

    if (_buildOptionsString != rhs._buildOptionsString)
        return false;
    if (_writeOptionsString != rhs._writeOptionsString)
        return false;

    if (_layerInheritance != rhs._layerInheritance)
        return false;

    if (_abortTaskOnError != rhs._abortTaskOnError)
        return false;
    if (_abortRunOnError != rhs._abortRunOnError)
        return false;

    if (_defaultImageLayerOutputPolicy != rhs._defaultImageLayerOutputPolicy)
        return false;
    if (_defaultElevationLayerOutputPolicy != rhs._defaultElevationLayerOutputPolicy)
        return false;
    if (_optionalImageLayerOutputPolicy != rhs._optionalImageLayerOutputPolicy)
        return false;
    if (_optionalElevationLayerOutputPolicy != rhs._optionalElevationLayerOutputPolicy)
        return false;

    if (getRadiusEquator() != rhs.getRadiusEquator())
        return false;
    if (getRadiusPolar() != rhs.getRadiusPolar())
        return false;

    unsigned minNumImageOptions = std::min(_imageOptions.size(), rhs._imageOptions.size());
    for (unsigned int i = 0; i < minNumImageOptions; ++i)
    {
        if (_imageOptions[i].valid() && rhs._imageOptions[i])
        {
            if (!(_imageOptions[i]->compatible(*(rhs._imageOptions[i]))))
                return false;
        }
    }

    // following properties don't require checking as they don't effect compatibility
    // if (_comment != rhs._comment) return false;
    // if (_notifyLevel != rhs._notifyLevel) return false;
    // if (_revisionNumber != rhs._revisionNumber) return false;
    // if (_destinationCoordinateSystem != rhs._destinationCoordinateSystem) return false;
    return true;
}

void BuildOptions::setLayerImageOptions(unsigned int layerNum, vpb::ImageOptions *imageOptions)
{
    if (layerNum >= _imageOptions.size())
    {
        for (unsigned int i = _imageOptions.size(); i < layerNum; ++i)
        {
            _imageOptions.push_back(new vpb::ImageOptions(*this));
        }
        _imageOptions.push_back(imageOptions);
    }
    else
    {
        _imageOptions[layerNum] = imageOptions;
    }
}

vpb::ImageOptions *BuildOptions::getLayerImageOptions(unsigned int layerNum)
{
    return layerNum < _imageOptions.size() ? _imageOptions[layerNum].get() : 0;
}

const vpb::ImageOptions *BuildOptions::getLayerImageOptions(unsigned int layerNum) const
{
    return layerNum < _imageOptions.size() ? _imageOptions[layerNum].get() : 0;
}
