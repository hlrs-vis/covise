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

#include <vpb/Source>
#include <vpb/Destination>
#include <vpb/DataSet>
#include <vpb/TextureUtils>

#include <osg/Texture2D>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/Notify>
#include <osg/ImageUtils>
#include <osg/PagedLOD>
#include <osg/io_utils>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileNameUtils>

#include <osgUtil/SmoothingVisitor>
#include <osgUtil/Simplifier>

using namespace vpb;

#define SHIFT_RASTER_BY_HALF_CELL

/////////////////////////////////////////////////////////////////////////////////////////
//
//
//  DestinationVisitor
//
void DestinationVisitor::traverse(CompositeDestination &cd)
{
    for (CompositeDestination::ChildList::iterator citr = cd._children.begin();
         citr != cd._children.end();
         ++citr)
    {
        (*citr)->accept(*this);
    }

    for (CompositeDestination::TileList::iterator titr = cd._tiles.begin();
         titr != cd._tiles.end();
         ++titr)
    {
        (*titr)->accept(*this);
    }
}

void DestinationVisitor::apply(CompositeDestination &cd)
{
    traverse(cd);
}

void DestinationVisitor::apply(DestinationTile &dt)
{
}

/////////////////////////////////////////////////////////////////////////////////////////
//
//
//  DestinationTile
//

DestinationTile::DestinationTile()
    : _dataSet(0)
    , _level(0)
    , _tileX(0)
    , _tileY(0)
    , _maxSourceLevel(0)
    , _terrain_maxNumColumns(1024)
    , _terrain_maxNumRows(1024)
    , _terrain_maxSourceResolutionX(0.0f)
    , _terrain_maxSourceResolutionY(0.0f)
    , _complete(false)
{
    for (int i = 0; i < NUMBER_OF_POSITIONS; ++i)
    {
        _neighbour[i] = 0;
        _equalized[i] = false;
    }
}

const vpb::ImageOptions *DestinationTile::getImageOptions(unsigned int layerNum) const
{
    return _dataSet->getValidLayerImageOptions(layerNum);
}

GLenum DestinationTile::getPixelFormat(unsigned int layerNum) const
{
    ImageOptions::TextureType textureType = getImageOptions(layerNum)->getTextureType();
    return (textureType == ImageOptions::COMPRESSED_RGBA_TEXTURE || textureType == ImageOptions::RGBA || textureType == ImageOptions::RGBA_16 || textureType == ImageOptions::RGBA_S3TC_DXT1 || textureType == ImageOptions::RGBA_S3TC_DXT3 || textureType == ImageOptions::RGBA_S3TC_DXT5 || textureType == ImageOptions::RGBA32F) ? GL_RGBA : GL_RGB;
}

GLenum DestinationTile::getPixelType(unsigned int layerNum) const
{
    ImageOptions::TextureType textureType = getImageOptions(layerNum)->getTextureType();
    return (textureType == ImageOptions::RGBA32F || textureType == ImageOptions::RGB32F ? GL_FLOAT : GL_UNSIGNED_BYTE);
}

void DestinationTile::requiresDivision(float resolutionSensitivityScale, bool &needToDivideX, bool &needToDivideY)
{
    for (unsigned int layerNum = 0;
         layerNum < getNumLayers();
         ++layerNum)
    {
        ImageSet &imageSet = getImageSet(layerNum);
        for (ImageSet::LayerSetImageDataMap::iterator itr = imageSet._layerSetImageDataMap.begin();
             itr != imageSet._layerSetImageDataMap.end();
             ++itr)
        {
            ImageData &imageData = itr->second;
            unsigned int texture_numColumns;
            unsigned int texture_numRows;
            double texture_dx;
            double texture_dy;
            if (computeImageResolution(layerNum, itr->first, texture_numColumns, texture_numRows, texture_dx, texture_dy))
            {
                if (texture_dx * resolutionSensitivityScale > imageData._image_maxSourceResolutionX)
                    needToDivideX = true;
                if (texture_dy * resolutionSensitivityScale > imageData._image_maxSourceResolutionY)
                    needToDivideY = true;
            }
        }
    }

    unsigned int dem_numColumns;
    unsigned int dem_numRows;
    double dem_dx;
    double dem_dy;
    if (computeTerrainResolution(dem_numColumns, dem_numRows, dem_dx, dem_dy))
    {
        if (dem_dx * resolutionSensitivityScale > _terrain_maxSourceResolutionX)
            needToDivideX = true;
        if (dem_dy * resolutionSensitivityScale > _terrain_maxSourceResolutionY)
            needToDivideY = true;
    }
}

void DestinationTile::computeMaximumSourceResolution(Source *source)
{
    if (!source || source->getMaxLevel() < _level)
    {
        // skip the contribution of this source since this destination tile exceeds its contribution level.
        return;
    }

    SourceData *data = source->getSourceData();
    if (data && (source->getType() == Source::IMAGE || source->getType() == Source::HEIGHT_FIELD))
    {

        SpatialProperties sp = data->computeSpatialProperties(_cs.get());

        if (!sp._extents.intersects(_extents))
        {
            // skip this source since it doesn't overlap this tile.
            return;
        }

        if (sp._numValuesX != 0 && sp._numValuesY != 0)
        {
            _maxSourceLevel = osg::maximum(source->getMaxLevel(), _maxSourceLevel);

            float sourceResolutionX;
            float sourceResolutionY;
            // set up properly for vector and raster (previously always raster)
            if (sp._dataType == SpatialProperties::VECTOR)
            {
                sourceResolutionX = (sp._extents.xMax() - sp._extents.xMin()) / (float)(sp._numValuesX - 1);
                sourceResolutionY = (sp._extents.yMax() - sp._extents.yMin()) / (float)(sp._numValuesY - 1);
            }
            else // if (sp._dataType == SpatialProperties::RASTER)
            {
                sourceResolutionX = (sp._extents.xMax() - sp._extents.xMin()) / (float)sp._numValuesX;
                sourceResolutionY = (sp._extents.yMax() - sp._extents.yMin()) / (float)sp._numValuesY;
            }

            std::string sourceSetName;
            if (_dataSet->isOptionalLayerSet(source->getSetName()))
            {
                sourceSetName = source->getSetName();
            }

            switch (source->getType())
            {
            case (Source::IMAGE):
            {
                ImageData &imageData = getImageData(source->getLayer(), sourceSetName);
                if (imageData._image_maxSourceResolutionX == 0.0f)
                    imageData._image_maxSourceResolutionX = sourceResolutionX;
                else
                    imageData._image_maxSourceResolutionX = osg::minimum(imageData._image_maxSourceResolutionX, sourceResolutionX);
                if (imageData._image_maxSourceResolutionY == 0.0f)
                    imageData._image_maxSourceResolutionY = sourceResolutionY;
                else
                    imageData._image_maxSourceResolutionY = osg::minimum(imageData._image_maxSourceResolutionY, sourceResolutionY);
                break;
            }
            case (Source::HEIGHT_FIELD):
                if (_terrain_maxSourceResolutionX == 0.0f)
                    _terrain_maxSourceResolutionX = sourceResolutionX;
                else
                    _terrain_maxSourceResolutionX = osg::minimum(_terrain_maxSourceResolutionX, sourceResolutionX);
                if (_terrain_maxSourceResolutionY == 0.0f)
                    _terrain_maxSourceResolutionY = sourceResolutionY;
                else
                    _terrain_maxSourceResolutionY = osg::minimum(_terrain_maxSourceResolutionY, sourceResolutionY);
                break;
            default:
                break;
            }
        }
    }
}

void DestinationTile::computeMaximumSourceResolution(CompositeSource *sourceGraph)
{
    for (CompositeSource::source_iterator itr(sourceGraph); itr.valid(); ++itr)
    {
        Source *source = itr->get();
        computeMaximumSourceResolution(source);
    }
}

void DestinationTile::computeMaximumSourceResolution()
{
    for (Sources::iterator itr = _sources.begin();
         itr != _sources.end();
         ++itr)
    {
        computeMaximumSourceResolution(itr->get());
    }
}

bool DestinationTile::computeImageResolution(unsigned int layer, const std::string &setname, unsigned int &numColumns, unsigned int &numRows, double &resX, double &resY)
{
    bool result = false;
    ImageData &imageData = getImageData(layer, setname);

    if (imageData._image_maxSourceResolutionX != 0.0f && imageData._image_maxSourceResolutionY != 0.0f && _dataSet->getLayerMaximumTileImageSize(layer) != 0)
    {
        // set up properly for vector and raster (previously always vector)
        // assume raster if _dataType not set (default for Destination Tile)
        unsigned int numColumnsAtFullRes;
        unsigned int numRowsAtFullRes;
        if (_dataType == SpatialProperties::VECTOR)
        {
            numColumnsAtFullRes = 1 + (unsigned int)ceilf((_extents.xMax() - _extents.xMin()) / imageData._image_maxSourceResolutionX);
            numRowsAtFullRes = 1 + (unsigned int)ceilf((_extents.yMax() - _extents.yMin()) / imageData._image_maxSourceResolutionY);
        }
        else // if (_dataType == SpatialProperties::RASTER)
        {
            numColumnsAtFullRes = (unsigned int)ceilf((_extents.xMax() - _extents.xMin()) / imageData._image_maxSourceResolutionX);
            numRowsAtFullRes = (unsigned int)ceilf((_extents.yMax() - _extents.yMin()) / imageData._image_maxSourceResolutionY);
        }

        unsigned int numColumnsRequired = osg::minimum(_dataSet->getLayerMaximumTileImageSize(layer), numColumnsAtFullRes);
        unsigned int numRowsRequired = osg::minimum(_dataSet->getLayerMaximumTileImageSize(layer), numRowsAtFullRes);

        if (getImageOptions(layer)->getPowerOfTwoImages())
        {
            // use a minimum image size of 4x4 to avoid mipmap generation problems in OpenGL at sizes at 2x2.
            numColumns = 4;
            numRows = 4;

            // round to nearest power of two above or equal to the required resolution
            while (numColumns < numColumnsRequired)
                numColumns *= 2;
            while (numRows < numRowsRequired)
                numRows *= 2;
        }
        else
        {
            numColumns = osg::maximum(4u, numColumnsRequired);
            numRows = osg::maximum(4u, numRowsRequired);

            if (getImageOptions(layer)->getDestinationImageExtension() == ".dds")
            {
                // when doing compressed textures make sure that it's an multiple of four
                if ((numColumns % 4) != 0)
                    numColumns = ((numColumns >> 2) << 2) + 4;
                if ((numRows % 4) != 0)
                    numRows = ((numRows >> 2) << 2) + 4;
            }

            log(osg::NOTICE, "numColumnsAtFullRes = %i\tnumColumns = %i", numColumnsRequired, numColumns);
            log(osg::NOTICE, "numRowsAtFullRes = %i\tnumRows = %i", numRowsRequired, numRows);
        }

        // set up properly for vector and raster (previously always vector)
        // assume raster if _dataType not set (default for Destination Tile)
        if (_dataType == SpatialProperties::VECTOR)
        {
            resX = (_extents.xMax() - _extents.xMin()) / (double)(numColumns - 1);
            resY = (_extents.yMax() - _extents.yMin()) / (double)(numRows - 1);
        }
        else // if (_dataType == SpatialProperties::RASTER)
        {
            resX = (_extents.xMax() - _extents.xMin()) / (double)numColumns;
            resY = (_extents.yMax() - _extents.yMin()) / (double)numRows;
        }
        result = true;
    }
    return result;
}

bool DestinationTile::computeTerrainResolution(unsigned int &numColumns, unsigned int &numRows, double &resX, double &resY)
{
    if (_terrain_maxSourceResolutionX != 0.0f && _terrain_maxSourceResolutionY != 0.0f && _terrain_maxNumColumns != 0 && _terrain_maxNumRows != 0)
    {
        // set up properly for vector and raster (previously always vector)
        // assume vector if _dataType not set (default for Destination Tile)
        unsigned int numColumnsAtFullRes;
        unsigned int numRowsAtFullRes;
        if (_dataType == SpatialProperties::RASTER)
        {
            numColumnsAtFullRes = (unsigned int)ceilf((_extents.xMax() - _extents.xMin()) / _terrain_maxSourceResolutionX);
            numRowsAtFullRes = (unsigned int)ceilf((_extents.yMax() - _extents.yMin()) / _terrain_maxSourceResolutionY);
        }
        else // if (_dataType == SpatialProperties::VECTOR)
        {
            numColumnsAtFullRes = 1 + (unsigned int)ceilf((_extents.xMax() - _extents.xMin()) / _terrain_maxSourceResolutionX);
            numRowsAtFullRes = 1 + (unsigned int)ceilf((_extents.yMax() - _extents.yMin()) / _terrain_maxSourceResolutionY);
        }

        numColumns = osg::minimum(_terrain_maxNumColumns, numColumnsAtFullRes);
        numRows = osg::minimum(_terrain_maxNumRows, numRowsAtFullRes);

        // set up properly for vector and raster (previously always vector)
        // assume vector if _dataType not set (default for Destination Tile)
        if (_dataType == SpatialProperties::RASTER)
        {
            resX = (_extents.xMax() - _extents.xMin()) / (double)(numColumns);
            resY = (_extents.yMax() - _extents.yMin()) / (double)(numRows);
        }
        else // if (_dataType == SpatialProperties::VECTOR)
        {
            resX = (_extents.xMax() - _extents.xMin()) / (double)(numColumns - 1);
            resY = (_extents.yMax() - _extents.yMin()) / (double)(numRows - 1);
        }

        return true;
    }
    return false;
}

void DestinationTile::allocate()
{
    unsigned int texture_numColumns, texture_numRows;
    double texture_dx, texture_dy;
    for (unsigned int layerNum = 0;
         layerNum < getNumLayers();
         ++layerNum)
    {
        ImageSet &imageSet = getImageSet(layerNum);
        for (ImageSet::LayerSetImageDataMap::iterator itr = imageSet._layerSetImageDataMap.begin();
             itr != imageSet._layerSetImageDataMap.end();
             ++itr)
        {
            const std::string &setName = itr->first;
            if (computeImageResolution(layerNum, setName,
                                       texture_numColumns, texture_numRows, texture_dx, texture_dy))
            {

                ImageData &imageData = itr->second;
                imageData._imageDestination = new DestinationData(_dataSet);
                imageData._imageDestination->_cs = _cs;
                imageData._imageDestination->_extents = _extents;
                imageData._imageDestination->_geoTransform.set(texture_dx, 0.0, 0.0, 0.0,
                                                               0.0, -texture_dy, 0.0, 0.0,
                                                               0.0, 0.0, 1.0, 1.0,
                                                               _extents.xMin(), _extents.yMax(), 0.0, 1.0);

                imageData._imageDestination->_image = new osg::Image;

                osg::Image::WriteHint writeHint = osg::Image::NO_PREFERENCE;
                std::string imageName = _name;

                if (layerNum > 0)
                {
                    imageName += "_l";
                    imageName += char('0' + layerNum);
                }

                if (!setName.empty())
                {
                    switch (_dataSet->getOptionalImageLayerOutputPolicy())
                    {
                    case (BuildOptions::INLINE):
                        imageName += "_";
                        imageName += setName;
                        writeHint = osg::Image::EXTERNAL_FILE;
                        break;
                    case (BuildOptions::EXTERNAL_LOCAL_DIRECTORY):
                        imageName += "_";
                        imageName += setName;
                        writeHint = osg::Image::EXTERNAL_FILE;
                        break;
                    case (BuildOptions::EXTERNAL_SET_DIRECTORY):
                        imageName = _parent->getRelativePathForExternalSet(setName) + imageName;
                        writeHint = osg::Image::EXTERNAL_FILE;
                        break;
                    }
                }

                imageName += getImageOptions(layerNum)->getDestinationImageExtension();

                _dataSet->log(osg::NOTICE, "imageName = %s", imageName.c_str());

                imageData._imageDestination->_image->setFileName(imageName.c_str());
                imageData._imageDestination->_image->setWriteHint(writeHint);

                imageData._imageDestination->_image->allocateImage(texture_numColumns, texture_numRows, 1, getPixelFormat(layerNum), getPixelType(layerNum));
                unsigned char *data = imageData._imageDestination->_image->data();
                unsigned int totalSize = imageData._imageDestination->_image->getTotalSizeInBytesIncludingMipmaps();
                for (unsigned int i = 0; i < totalSize; ++i)
                {
                    *(data++) = 0;
                }
            }
        }
    }

    unsigned int dem_numColumns, dem_numRows;
    double dem_dx, dem_dy;
    if (computeTerrainResolution(dem_numColumns, dem_numRows, dem_dx, dem_dy))
    {
        _terrain = new DestinationData(_dataSet);
        _terrain->_cs = _cs;
        _terrain->_extents = _extents;
        _terrain->_geoTransform.set(dem_dx, 0.0, 0.0, 0.0,
                                    0.0, -dem_dy, 0.0, 0.0,
                                    0.0, 0.0, 1.0, 1.0,
                                    _extents.xMin(), _extents.yMax(), 0.0, 1.0);
        _terrain->_heightField = new osg::HeightField;
        _terrain->_heightField->allocate(dem_numColumns, dem_numRows);
        _terrain->_heightField->setOrigin(osg::Vec3(_extents.xMin(), _extents.yMin(), 0.0f));
        _terrain->_heightField->setXInterval(dem_dx);
        _terrain->_heightField->setYInterval(dem_dy);

        //float xMax = _terrain->_heightField->getOrigin().x()+_terrain->_heightField->getXInterval()*(float)(dem_numColumns-1);
        //log(osg::INFO, "ErrorX = %f",xMax-_extents.xMax());

        //float yMax = _terrain->_heightField->getOrigin().y()+_terrain->_heightField->getYInterval()*(float)(dem_numRows-1);
        //log(osg::INFO, "ErrorY = "<<yMax-_extents.yMax());
    }
}

void DestinationTile::computeNeighboursFromQuadMap()
{
    if (_dataSet)
    {
        setNeighbours(_dataSet->getTile(_level, _tileX - 1, _tileY), _dataSet->getTile(_level, _tileX - 1, _tileY - 1),
                      _dataSet->getTile(_level, _tileX, _tileY - 1), _dataSet->getTile(_level, _tileX + 1, _tileY - 1),
                      _dataSet->getTile(_level, _tileX + 1, _tileY), _dataSet->getTile(_level, _tileX + 1, _tileY + 1),
                      _dataSet->getTile(_level, _tileX, _tileY + 1), _dataSet->getTile(_level, _tileX - 1, _tileY + 1));
    }
}

void DestinationTile::setNeighbours(DestinationTile *left, DestinationTile *left_below,
                                    DestinationTile *below, DestinationTile *below_right,
                                    DestinationTile *right, DestinationTile *right_above,
                                    DestinationTile *above, DestinationTile *above_left)
{
    _neighbour[LEFT] = left;
    _neighbour[LEFT_BELOW] = left_below;
    _neighbour[BELOW] = below;
    _neighbour[BELOW_RIGHT] = below_right;
    _neighbour[RIGHT] = right;
    _neighbour[RIGHT_ABOVE] = right_above;
    _neighbour[ABOVE] = above;
    _neighbour[ABOVE_LEFT] = above_left;

    //     log(osg::INFO,"LEFT="<<_neighbour[LEFT]);
    //     log(osg::INFO,"LEFT_BELOW="<<_neighbour[LEFT_BELOW]);
    //     log(osg::INFO,"BELOW="<<_neighbour[BELOW]);
    //     log(osg::INFO,"BELOW_RIGHT="<<_neighbour[BELOW_RIGHT]);
    //     log(osg::INFO,"RIGHT="<<_neighbour[RIGHT]);
    //     log(osg::INFO,"RIGHT_ABOVE="<<_neighbour[RIGHT_ABOVE]);
    //     log(osg::INFO,"ABOVE="<<_neighbour[ABOVE]);
    //     log(osg::INFO,"ABOVE_LEFT="<<_neighbour[ABOVE_LEFT]);

    for (int i = 0; i < NUMBER_OF_POSITIONS; ++i)
    {
        _equalized[i] = false;
    }
}

void DestinationTile::checkNeighbouringTiles()
{
    for (int i = 0; i < NUMBER_OF_POSITIONS; ++i)
    {
        if (_neighbour[i] && _neighbour[i]->_neighbour[(i + 4) % NUMBER_OF_POSITIONS] != this)
        {
            log(osg::INFO, "Error:: Tile %d's _neighbour[%d] does not point back to it.", this, i);
        }
    }
}

void DestinationTile::allocateEdgeNormals()
{
    osg::HeightField *hf = _terrain->_heightField.get();
    if (!hf)
        return;
}

void DestinationTile::equalizeCorner(Position position)
{
    // don't need to equalize if already done.
    if (_equalized[position])
        return;

    typedef std::pair<DestinationTile *, Position> TileCornerPair;
    typedef std::vector<TileCornerPair> TileCornerList;

    TileCornerList cornersToProcess;
    DestinationTile *tile = 0;

    cornersToProcess.push_back(TileCornerPair(this, position));

    tile = _neighbour[(position - 1) % NUMBER_OF_POSITIONS];
    if (tile)
        cornersToProcess.push_back(TileCornerPair(tile, (Position)((position + 2) % NUMBER_OF_POSITIONS)));

    tile = _neighbour[(position) % NUMBER_OF_POSITIONS];
    if (tile)
        cornersToProcess.push_back(TileCornerPair(tile, (Position)((position + 4) % NUMBER_OF_POSITIONS)));

    tile = _neighbour[(position + 1) % NUMBER_OF_POSITIONS];
    if (tile)
        cornersToProcess.push_back(TileCornerPair(tile, (Position)((position + 6) % NUMBER_OF_POSITIONS)));

    // make all these tiles as equalised upfront before we return.
    TileCornerList::iterator itr;
    for (itr = cornersToProcess.begin();
         itr != cornersToProcess.end();
         ++itr)
    {
        TileCornerPair &tcp = *itr;
        tcp.first->_equalized[tcp.second] = true;
    }

    // if there is only one valid corner to process then there is nothing to equalize against so return.
    if (cornersToProcess.size() == 1)
        return;

    for (unsigned int layerNum = 0;
         layerNum < getNumLayers();
         ++layerNum)
    {

        typedef std::pair<osg::Image *, Position> ImageCornerPair;
        typedef std::vector<ImageCornerPair> ImageCornerList;

        ImageCornerList imagesToProcess;

        for (itr = cornersToProcess.begin();
             itr != cornersToProcess.end();
             ++itr)
        {
            TileCornerPair &tcp = *itr;
            if (layerNum < tcp.first->getNumLayers())
            {
                ImageSet &imageSet = tcp.first->getImageSet(layerNum);
                for (ImageSet::LayerSetImageDataMap::iterator itr = imageSet._layerSetImageDataMap.begin();
                     itr != imageSet._layerSetImageDataMap.end();
                     ++itr)
                {
                    ImageData &imageData = itr->second;
                    if (imageData._imageDestination.valid() && imageData._imageDestination->_image.valid())
                    {
                        imagesToProcess.push_back(ImageCornerPair(imageData._imageDestination->_image.get(), tcp.second));
                    }
                }
            }
        }

        if (imagesToProcess.size() > 1)
        {
            int red = 0;
            int green = 0;
            int blue = 0;
            // accumulate colours.
            ImageCornerList::iterator iitr;
            for (iitr = imagesToProcess.begin();
                 iitr != imagesToProcess.end();
                 ++iitr)
            {
                ImageCornerPair &icp = *iitr;
                unsigned char *data = 0;
                switch (icp.second)
                {
                case LEFT_BELOW:
                    data = icp.first->data(0, 0);
                    break;
                case BELOW_RIGHT:
                    data = icp.first->data(icp.first->s() - 1, 0);
                    break;
                case RIGHT_ABOVE:
                    data = icp.first->data(icp.first->s() - 1, icp.first->t() - 1);
                    break;
                case ABOVE_LEFT:
                    data = icp.first->data(0, icp.first->t() - 1);
                    break;
                default:
                    break;
                }
                red += *(data++);
                green += *(data++);
                blue += *(data);
            }

            // divide them.
            red /= imagesToProcess.size();
            green /= imagesToProcess.size();
            blue /= imagesToProcess.size();

            // apply colour to corners.
            for (iitr = imagesToProcess.begin();
                 iitr != imagesToProcess.end();
                 ++iitr)
            {
                ImageCornerPair &icp = *iitr;
                unsigned char *data = 0;
                switch (icp.second)
                {
                case LEFT_BELOW:
                    data = icp.first->data(0, 0);
                    break;
                case BELOW_RIGHT:
                    data = icp.first->data(icp.first->s() - 1, 0);
                    break;
                case RIGHT_ABOVE:
                    data = icp.first->data(icp.first->s() - 1, icp.first->t() - 1);
                    break;
                case ABOVE_LEFT:
                    data = icp.first->data(0, icp.first->t() - 1);
                    break;
                default:
                    break;
                }
                *(data++) = red;
                *(data++) = green;
                *(data) = blue;
            }
        }
    }

    typedef std::pair<osg::HeightField *, TileCornerPair> HeightFieldCornerPair;
    typedef std::vector<HeightFieldCornerPair> HeightFieldCornerList;
    HeightFieldCornerList heightFieldsToProcess;

    for (itr = cornersToProcess.begin();
         itr != cornersToProcess.end();
         ++itr)
    {
        TileCornerPair &tcp = *itr;
        if (tcp.first->_terrain.valid() && tcp.first->_terrain->_heightField.valid())
        {
            heightFieldsToProcess.push_back(HeightFieldCornerPair(tcp.first->_terrain->_heightField.get(), tcp));
        }
    }

    if (heightFieldsToProcess.size() > 1)
    {
        float height = 0;
        osg::Vec2 heightDelta;
        // accumulate heights & normals
        HeightFieldCornerList::iterator hitr;
        for (hitr = heightFieldsToProcess.begin();
             hitr != heightFieldsToProcess.end();
             ++hitr)
        {
            HeightFieldCornerPair &hfcp = *hitr;
            switch (hfcp.second.second)
            {
            case LEFT_BELOW:
                height += hfcp.first->getHeight(0, 0);
                heightDelta += hfcp.first->getHeightDelta(0, 0);
                break;
            case BELOW_RIGHT:
                height += hfcp.first->getHeight(hfcp.first->getNumColumns() - 1, 0);
                heightDelta += hfcp.first->getHeightDelta(hfcp.first->getNumColumns() - 1, 0);
                break;
            case RIGHT_ABOVE:
                height += hfcp.first->getHeight(hfcp.first->getNumColumns() - 1, hfcp.first->getNumRows() - 1);
                heightDelta += hfcp.first->getHeightDelta(hfcp.first->getNumColumns() - 1, hfcp.first->getNumRows() - 1);
                break;
            case ABOVE_LEFT:
                height += hfcp.first->getHeight(0, hfcp.first->getNumRows() - 1);
                heightDelta += hfcp.first->getHeightDelta(0, hfcp.first->getNumRows() - 1);
                break;
            default:
                break;
            }
        }

        // divide them.
        height /= heightFieldsToProcess.size();
        heightDelta /= heightFieldsToProcess.size();

        // apply height and normals to corners.
        for (hitr = heightFieldsToProcess.begin();
             hitr != heightFieldsToProcess.end();
             ++hitr)
        {
            HeightFieldCornerPair &hfcp = *hitr;
            TileCornerPair &tcp = hfcp.second;
            switch (tcp.second)
            {
            case LEFT_BELOW:
                hfcp.first->setHeight(0, 0, height);
                break;
            case BELOW_RIGHT:
                hfcp.first->setHeight(hfcp.first->getNumColumns() - 1, 0, height);
                break;
            case RIGHT_ABOVE:
                hfcp.first->setHeight(hfcp.first->getNumColumns() - 1, hfcp.first->getNumRows() - 1, height);
                break;
            case ABOVE_LEFT:
                hfcp.first->setHeight(0, hfcp.first->getNumRows() - 1, height);
                break;
            default:
                break;
            }
            tcp.first->_heightDeltas[tcp.second].clear();
            tcp.first->_heightDeltas[tcp.second].push_back(heightDelta);
        }
    }
}

const char *edgeString(DestinationTile::Position position)
{
    switch (position)
    {
    case DestinationTile::LEFT:
        return "left";
    case DestinationTile::BELOW:
        return "below";
    case DestinationTile::RIGHT:
        return "right";
    case DestinationTile::ABOVE:
        return "above";
    default:
        return "<not an edge>";
    }
}

void DestinationTile::setTileComplete(bool complete)
{
    _complete = complete;
    log(osg::INFO, "setTileComplete(%d) for %d\t%d%d", complete, _level, _tileX, _tileY);
}

void DestinationTile::equalizeEdge(Position position)
{
    // don't need to equalize if already done.
    if (_equalized[position])
        return;

    DestinationTile *tile2 = _neighbour[position];
    Position position2 = (Position)((position + 4) % NUMBER_OF_POSITIONS);

    _equalized[position] = true;

    // no neighbour of this edge so nothing to equalize.
    if (!tile2)
        return;

    tile2->_equalized[position2] = true;

    for (unsigned int layerNum = 0;
         layerNum < getNumLayers();
         ++layerNum)
    {
        ImageSet &imageSet = getImageSet(layerNum);
        for (ImageSet::LayerSetImageDataMap::iterator itr = imageSet._layerSetImageDataMap.begin();
             itr != imageSet._layerSetImageDataMap.end();
             ++itr)
        {
            ImageData &imageData1 = itr->second;
            ImageData &imageData2 = tile2->getImageData(layerNum, itr->first);

            // do we have a image to equalize?
            if (!imageData1._imageDestination.valid())
                continue;

            // does the neighbouring tile have an image to equalize?
            if (layerNum >= tile2->getNumLayers())
                continue;
            if (!(imageData2._imageDestination.valid()))
                continue;

            osg::Image *image1 = imageData1._imageDestination->_image.get();
            osg::Image *image2 = imageData2._imageDestination->_image.get();

            //log(osg::INFO,"Equalizing edge "<<edgeString(position)<<" of \t"<<_level<<"\t"<<_tileX<<"\t"<<_tileY
            //         <<"  neighbour "<<tile2->_level<<"\t"<<tile2->_tileX<<"\t"<<tile2->_tileY);

            //   if (_tileY==0) return;

            if (image1 && image2 && image1->getPixelFormat() == image2->getPixelFormat() && image1->getDataType() == image2->getDataType() && image1->getPixelFormat() == GL_RGB && image1->getDataType() == GL_UNSIGNED_BYTE)
            {

                //log(osg::INFO,"   Equalizing image1= "<<image1<<                         " with image2 = "<<image2);
                //log(osg::INFO,"              data1 = 0x"<<std::hex<<(int)image1->data()<<" with data2  = 0x"<<(int)image2->data());

                unsigned char *data1 = 0;
                unsigned char *data2 = 0;
                unsigned int delta1 = 0;
                unsigned int delta2 = 0;
                int num = 0;

                switch (position)
                {
                case LEFT:
                    data1 = image1->data(0, 1); // LEFT hand side
                    delta1 = image1->getRowSizeInBytes();
                    data2 = image2->data(image2->s() - 1, 1); // RIGHT hand side
                    delta2 = image2->getRowSizeInBytes();
                    num = (image1->t() == image2->t()) ? image2->t() - 2 : 0; // note miss out corners.
                    //log(osg::INFO,"       left "<<num);
                    break;
                case BELOW:
                    data1 = image1->data(1, 0); // BELOW hand side
                    delta1 = 3;
                    data2 = image2->data(1, image2->t() - 1); // ABOVE hand side
                    delta2 = 3;
                    num = (image1->s() == image2->s()) ? image2->s() - 2 : 0; // note miss out corners.
                    //log(osg::INFO,"       below "<<num);
                    break;
                case RIGHT:
                    data1 = image1->data(image1->s() - 1, 1); // LEFT hand side
                    delta1 = image1->getRowSizeInBytes();
                    data2 = image2->data(0, 1); // RIGHT hand side
                    delta2 = image2->getRowSizeInBytes();
                    num = (image1->t() == image2->t()) ? image2->t() - 2 : 0; // note miss out corners.
                    //log(osg::INFO,"       right "<<num);
                    break;
                case ABOVE:
                    data1 = image1->data(1, image1->t() - 1); // ABOVE hand side
                    delta1 = 3;
                    data2 = image2->data(1, 0); // BELOW hand side
                    delta2 = 3;
                    num = (image1->s() == image2->s()) ? image2->s() - 2 : 0; // note miss out corners.
                    //log(osg::INFO,"       above "<<num);
                    break;
                default:
                    //log(osg::INFO,"       default "<<num);
                    break;
                }

                for (int i = 0; i < num; ++i)
                {
                    unsigned char red = (unsigned char)((((int)*data1 + (int)*data2) / 2));
                    unsigned char green = (unsigned char)((((int)*(data1 + 1)) + (int)(*(data2 + 1))) / 2);
                    unsigned char blue = (unsigned char)((((int)*(data1 + 2)) + (int)(*(data2 + 2))) / 2);
#if 1
                    *data1 = red;
                    *(data1 + 1) = green;
                    *(data1 + 2) = blue;

                    *data2 = red;
                    *(data2 + 1) = green;
                    *(data2 + 2) = blue;
#endif

#if 0
                    *data1 = 255;
                    *(data1+1) = 0;
                    *(data1+2) = 0;

                    *data2 = 0;
                    *(data2+1) = 0;
                    *(data2+2) = 0;
#endif
                    data1 += delta1;
                    data2 += delta2;

                    //log(osg::INFO,"    equalizing colour "<<(int)data1<<"  "<<(int)data2);
                }
            }
        }
    }

    osg::HeightField *heightField1 = _terrain.valid() ? _terrain->_heightField.get() : 0;
    osg::HeightField *heightField2 = tile2->_terrain.valid() ? tile2->_terrain->_heightField.get() : 0;

    if (heightField1 && heightField2)
    {
        //log(osg::INFO,"   Equalizing heightfield");

        float *data1 = 0;
        float *data2 = 0;
        unsigned int delta1 = 0;
        unsigned int delta2 = 0;
        int num = 0;

        unsigned int i1 = 0;
        unsigned int j1 = 0;
        unsigned int i2 = 0;
        unsigned int j2 = 0;
        unsigned int deltai = 0;
        unsigned int deltaj = 0;

        switch (position)
        {
        case LEFT:
            i1 = 0;
            j1 = 1;
            i2 = heightField2->getNumColumns() - 1;
            j2 = 1;
            deltai = 0;
            deltaj = 1;

            data1 = &(heightField1->getHeight(0, 1)); // LEFT hand side
            delta1 = heightField1->getNumColumns();
            data2 = &(heightField2->getHeight(heightField2->getNumColumns() - 1, 1)); // RIGHT hand side
            delta2 = heightField2->getNumColumns();
            num = (heightField1->getNumRows() == heightField2->getNumRows()) ? heightField1->getNumRows() - 2 : 0; // note miss out corners.
            break;

        case BELOW:
            i1 = 1;
            j1 = 0;
            i2 = 1;
            j2 = heightField2->getNumRows() - 1;
            deltai = 1;
            deltaj = 0;

            data1 = &(heightField1->getHeight(1, 0)); // BELOW hand side
            delta1 = 1;
            data2 = &(heightField2->getHeight(1, heightField2->getNumRows() - 1)); // ABOVE hand side
            delta2 = 1;
            num = (heightField1->getNumColumns() == heightField2->getNumColumns()) ? heightField1->getNumColumns() - 2 : 0; // note miss out corners.
            break;

        case RIGHT:
            i1 = heightField1->getNumColumns() - 1;
            j1 = 1;
            i2 = 0;
            j2 = 1;
            deltai = 0;
            deltaj = 1;

            data1 = &(heightField1->getHeight(heightField1->getNumColumns() - 1, 1)); // LEFT hand side
            delta1 = heightField1->getNumColumns();
            data2 = &(heightField2->getHeight(0, 1)); // LEFT hand side
            delta2 = heightField2->getNumColumns();
            num = (heightField1->getNumRows() == heightField2->getNumRows()) ? heightField1->getNumRows() - 2 : 0; // note miss out corners.
            break;

        case ABOVE:
            i1 = 1;
            j1 = heightField1->getNumRows() - 1;
            i2 = 1;
            j2 = 0;
            deltai = 1;
            deltaj = 0;

            data1 = &(heightField1->getHeight(1, heightField1->getNumRows() - 1)); // ABOVE hand side
            delta1 = 1;
            data2 = &(heightField2->getHeight(1, 0)); // BELOW hand side
            delta2 = 1;
            num = (heightField1->getNumColumns() == heightField2->getNumColumns()) ? heightField1->getNumColumns() - 2 : 0; // note miss out corners.
            break;
        default:
            break;
        }

        _heightDeltas[position].clear();
        _heightDeltas[position].reserve(num);
        tile2->_heightDeltas[(position + 4) % NUMBER_OF_POSITIONS].clear();
        tile2->_heightDeltas[(position + 4) % NUMBER_OF_POSITIONS].reserve(num);

        for (int i = 0; i < num; ++i)
        {
            // equalize height
            float z = (*data1 + *data2) / 2.0f;

            *data1 = z;
            *data2 = z;

            data1 += delta1;
            data2 += delta2;

            // equailize normals
            osg::Vec2 heightDelta = (heightField1->getHeightDelta(i1, j1) + heightField2->getHeightDelta(i2, j2)) * 0.5f;

            // pass the normals on to the tiles.
            _heightDeltas[position].push_back(heightDelta);
            tile2->_heightDeltas[(position + 4) % NUMBER_OF_POSITIONS].push_back(heightDelta);

            i1 += deltai;
            i2 += deltai;
            j1 += deltaj;
            j2 += deltaj;
        }
    }
}

void DestinationTile::equalizeBoundaries()
{
    log(osg::INFO, "DestinationTile::equalizeBoundaries()");

    equalizeCorner(LEFT_BELOW);
    equalizeCorner(BELOW_RIGHT);
    equalizeCorner(RIGHT_ABOVE);
    equalizeCorner(ABOVE_LEFT);

    equalizeEdge(LEFT);
    equalizeEdge(BELOW);
    equalizeEdge(RIGHT);
    equalizeEdge(ABOVE);
}

void DestinationTile::optimizeResolution()
{
    if (_terrain.valid() && _terrain->_heightField.valid())
    {
        osg::HeightField *hf = _terrain->_heightField.get();

        // compute min max of height field
        float minHeight = hf->getHeight(0, 0);
        float maxHeight = minHeight;
        for (unsigned int r = 0; r < hf->getNumRows(); ++r)
        {
            for (unsigned int c = 0; c < hf->getNumColumns(); ++c)
            {
                float h = hf->getHeight(c, r);
                if (h < minHeight)
                    minHeight = h;
                if (h > maxHeight)
                    maxHeight = h;
            }
        }

        if (minHeight == maxHeight)
        {
            log(osg::INFO, "******* We have a flat tile ******* ");

            unsigned int minimumSize = 8;

            unsigned int numColumns = minimumSize;
            unsigned int numRows = minimumSize;

            float ratio_y_over_x = (_extents.yMax() - _extents.yMin()) / (_extents.xMax() - _extents.xMin());
            if (ratio_y_over_x > 1.2)
                numRows = (unsigned int)ceilf((float)numRows * ratio_y_over_x);
            else if (ratio_y_over_x < 0.8)
                numColumns = (unsigned int)ceilf((float)numColumns / ratio_y_over_x);

            hf->allocate(numColumns, numRows);
            hf->setOrigin(osg::Vec3(_extents.xMin(), _extents.yMin(), 0.0f));
            hf->setXInterval((_extents.xMax() - _extents.xMin()) / (float)(numColumns - 1));
            hf->setYInterval((_extents.yMax() - _extents.yMin()) / (float)(numRows - 1));

            for (unsigned int r = 0; r < numRows; ++r)
            {
                for (unsigned int c = 0; c < numColumns; ++c)
                {
                    hf->setHeight(c, r, minHeight);
                }
            }
        }
    }
}

void DestinationTile::addNodeToScene(osg::Node *node, bool transformIfRequired)
{
    if (!_createdScene)
        _createdScene = new osg::Group;

    osg::MatrixTransform *transform = dynamic_cast<osg::MatrixTransform *>(_createdScene.get());
    if (transformIfRequired && transform)
    {
        transform->addChild(node);
    }
    else
    {
        osg::Group *group = dynamic_cast<osg::Group *>(_createdScene.get());
        if (!group)
        {
            group = new osg::Group;
            group->addChild(_createdScene.get());
            _createdScene = group;
        }

        group->addChild(node);
    }
}

osg::Node *DestinationTile::createScene()
{
    if (_createdScene.valid())
        return _createdScene.get();

    if (_dataSet->getGeometryType() == DataSet::HEIGHT_FIELD)
    {
        _createdScene = createHeightField();
    }
    else if (_dataSet->getGeometryType() == DataSet::TERRAIN)
    {
        _createdScene = createTerrainTile();
    }
    else
    {
        _createdScene = createPolygonal();
    }

    if (_models.valid())
    {
        if (_dataSet->getModelPlacer())
        {
            for (ModelList::iterator itr = _models->_models.begin();
                 itr != _models->_models.end();
                 ++itr)
            {
                _dataSet->getModelPlacer()->place(*this, itr->get());
            }
        }
        else
        {
            for (ModelList::iterator itr = _models->_models.begin();
                 itr != _models->_models.end();
                 ++itr)
            {
                addNodeToScene(itr->get());
            }
        }

        if (_dataSet->getShapeFilePlacer())
        {
            for (ModelList::iterator itr = _models->_shapeFiles.begin();
                 itr != _models->_shapeFiles.end();
                 ++itr)
            {
                _dataSet->getShapeFilePlacer()->place(*this, itr->get());
            }
        }
        else
        {
            for (ModelList::iterator itr = _models->_shapeFiles.begin();
                 itr != _models->_shapeFiles.end();
                 ++itr)
            {
                addNodeToScene(itr->get());
            }
        }
    }

    return _createdScene.get();
}

struct QuantizeOperator
{
    QuantizeOperator(unsigned int rowSize, int bits, bool errorDiffusion)
        : _rowSize(rowSize)
        , _bits(bits)
        , _errorDiffusion(errorDiffusion)
        , _i(0)
    {
        _max = float((2 << (_bits - 1)) - 1);
        _currentRow.resize(rowSize);
        _nextRow.resize(rowSize);
    }

    typedef std::vector<osg::Vec4> Errors;

    void quantize(float &v, int colour_index) const
    {
        if (_max < 255.0)
        {
            //osg::notify(osg::NOTICE)<<"quantizing "<<max<<std::endl;

            if (_errorDiffusion)
            {
                float new_v = v - _currentRow[_i][colour_index];
                new_v = floorf(new_v * _max + 0.499999f) / _max;
                if (new_v < 0.0f)
                    new_v = 0.0f;
                if (new_v > 1.0f)
                    new_v = 1.0f;

                float new_error = (new_v - v);
                v = new_v;

                if (_i + 1 < _rowSize)
                {
                    _currentRow[_i + 1][colour_index] += new_error * 0.33;
                    _nextRow[_i + 1][colour_index] += new_error * 0.16;
                }
                if (_i > 0)
                {
                    _nextRow[_i - 1][colour_index] += new_error * 0.16;
                }
                _nextRow[_i][colour_index] += new_error * 0.33;
            }
            else
            {
                v = floorf(v * _max + 0.499999f) / _max;
            }
        }
    }

    inline void luminance(float &l) const { quantize(l, 0); }
    inline void alpha(float &a) const { quantize(a, 3); }
    inline void luminance_alpha(float &l, float &a) const
    {
        quantize(l, 0);
        quantize(a, 3);
    }
    inline void rgb(float &r, float &g, float &b) const
    {
        quantize(r, 0);
        quantize(g, 1);
        quantize(b, 2);
    }
    inline void rgba(float &r, float &g, float &b, float &a) const
    {
        quantize(r, 0);
        quantize(g, 1);
        quantize(b, 2);
        quantize(a, 3);
    }

    unsigned int _rowSize;
    int _bits;
    bool _errorDiffusion;
    float _max;
    mutable unsigned int _i;
    mutable Errors _currentRow;
    mutable Errors _nextRow;
};

osg::StateSet *DestinationTile::createStateSet()
{
    if (_stateset.valid())
        return _stateset.get();

    if (_imageLayerSet.empty())
        return 0;

    unsigned int numValidImagerLayers = 0;
    unsigned int layerNum;
    for (layerNum = 0;
         layerNum < getNumLayers();
         ++layerNum)
    {
        if (getImageSet(layerNum).valid())
        {
            ++numValidImagerLayers;
        }
    }

    if (numValidImagerLayers == 0)
        return 0;

    _stateset = new osg::StateSet;

    for (layerNum = 0;
         layerNum < getNumLayers();
         ++layerNum)
    {
        ImageSet &imageSet = getImageSet(layerNum);
        if (imageSet._layerSetImageDataMap.empty())
            continue;

        ImageData &imageData = imageSet._layerSetImageDataMap.begin()->second;
        if (!imageData._imageDestination.valid() || !imageData._imageDestination->_image.valid())
            continue;

        osg::Image *image = imageData._imageDestination->_image.get();
        std::string imageExtension = osgDB::getFileExtension(image->getFileName());

        osg::Texture2D *texture = new osg::Texture2D;

        texture->setImage(image);
        texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
        texture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
        switch (getImageOptions(layerNum)->getMipMappingMode())
        {
        case (DataSet::NO_MIP_MAPPING):
        {
            log(osg::NOTICE, "DestinationTile::createStateSet() - DataSet::NO_MIP_MAPPING");

            texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
            texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
        }
        break;
        case (DataSet::MIP_MAPPING_HARDWARE):
        {
            log(osg::NOTICE, "DestinationTile::createStateSet() - DataSet::MIP_MAPPING_HARDWARE");
            texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
            texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
        }
        break;
        case (DataSet::MIP_MAPPING_IMAGERY):
        {
            log(osg::NOTICE, "DestinationTile::createStateSet() - DataSet::MIP_MAPPING_IMAGERY");
            texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
            texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
        }
        break;
        }

        texture->setMaxAnisotropy(getImageOptions(layerNum)->getMaxAnisotropy());
        _stateset->setTextureAttributeAndModes(layerNum, texture, osg::StateAttribute::ON);

        bool inlineImageFile = _dataSet->getDestinationTileExtension() == ".ive" || _dataSet->getDestinationTileExtension() == ".osgb";
        bool compressedImageSupported = inlineImageFile || imageExtension == "dds";
        bool mipmapImageSupported = compressedImageSupported; // inlineImageFile;

        if (getImageOptions(layerNum)->getImageryQuantization() != 0 && getImageOptions(layerNum)->getImageryQuantization() < 8)
        {

            log(osg::NOTICE, "Quantize image to %i bits", getImageOptions(layerNum)->getImageryQuantization());
            osg::modifyImage(image, QuantizeOperator(image->s(), getImageOptions(layerNum)->getImageryQuantization(), getImageOptions(layerNum)->getImageryErrorDiffusion()));
        }

        // int minumCompressedTextureSize = 64;
        // int minumDXT3CompressedTextureSize = 256;

        osg::Texture::InternalFormatMode internalFormatMode = osg::Texture::USE_IMAGE_DATA_FORMAT;
        switch (getImageOptions(layerNum)->getTextureType())
        {
        case (BuildOptions::RGB_S3TC_DXT1):
            internalFormatMode = osg::Texture::USE_S3TC_DXT1_COMPRESSION;
            break;
        case (BuildOptions::RGBA_S3TC_DXT1):
            internalFormatMode = osg::Texture::USE_S3TC_DXT1_COMPRESSION;
            break;
        case (BuildOptions::RGBA_S3TC_DXT3):
            internalFormatMode = osg::Texture::USE_S3TC_DXT3_COMPRESSION;
            break;
        case (BuildOptions::RGBA_S3TC_DXT5):
            internalFormatMode = osg::Texture::USE_S3TC_DXT5_COMPRESSION;
            break;
        case (BuildOptions::ARB_COMPRESSED):
            internalFormatMode = osg::Texture::USE_ARB_COMPRESSION;
            break;
        case (BuildOptions::COMPRESSED_TEXTURE):
            internalFormatMode = osg::Texture::USE_S3TC_DXT1_COMPRESSION;
            break;
        case (BuildOptions::COMPRESSED_RGBA_TEXTURE):
            internalFormatMode = osg::Texture::USE_S3TC_DXT3_COMPRESSION;
            break;
        default:
            break;
        }

        bool compressedImageRequired = (internalFormatMode != osg::Texture::USE_IMAGE_DATA_FORMAT);
        //  image->s()>=minumCompressedTextureSize && image->t()>=minumCompressedTextureSize &&

        if (compressedImageSupported && compressedImageRequired && (image->getPixelFormat() == GL_RGB || image->getPixelFormat() == GL_RGBA))
        {
            log(osg::NOTICE, "Compressed image");

            bool generateMiMap = getImageOptions(layerNum)->getMipMappingMode() == DataSet::MIP_MAPPING_IMAGERY;
            bool resizePowerOfTwo = getImageOptions(layerNum)->getPowerOfTwoImages();
            vpb::compress(*_dataSet->getState(), *texture, internalFormatMode, generateMiMap, resizePowerOfTwo, _dataSet->getCompressionMethod(), _dataSet->getCompressionQuality());

            log(osg::INFO, ">>>>>>>>>>>>>>>compressed image.<<<<<<<<<<<<<<");
        }
        else
        {
            log(osg::NOTICE, "Non compressed image mipmapImageSupported=%i imageExtension=%s", mipmapImageSupported, imageExtension.c_str());

            if (_dataSet->getTextureType() == DataSet::RGB_16 && image->getPixelFormat() == GL_RGB)
            {
                image->scaleImage(image->s(), image->t(), image->r(), GL_UNSIGNED_SHORT_5_6_5);
            }
            else if (_dataSet->getTextureType() == DataSet::RGBA_16 && image->getPixelFormat() == GL_RGBA)
            {
                image->scaleImage(image->s(), image->t(), image->r(), GL_UNSIGNED_SHORT_5_5_5_1);
            }

            if (mipmapImageSupported && getImageOptions(layerNum)->getMipMappingMode() == DataSet::MIP_MAPPING_IMAGERY)
            {
                log(osg::NOTICE, "Doing mipmapping");

                bool resizePowerOfTwo = getImageOptions(layerNum)->getPowerOfTwoImages();
                vpb::generateMipMap(*_dataSet->getState(), *texture, resizePowerOfTwo, _dataSet->getCompressionMethod());

                log(osg::INFO, ">>>>>>>>>>>>>>>mip mapped image.<<<<<<<<<<<<<<");
            }
        }
    }

    switch (_dataSet->getLayerInheritance())
    {
    case (BuildOptions::INHERIT_LOWEST_AVAILABLE):
    {
        osg::StateAttribute *texture = 0;
        // first look for an available texture
        for (layerNum = 0;
             layerNum < _dataSet->getNumOfTextureLevels() && texture == 0;
             ++layerNum)
        {
            texture = _stateset->getTextureAttribute(layerNum, osg::StateAttribute::TEXTURE);
        }

        if (texture)
        {
            // now fill in any blanks
            for (layerNum = 0;
                 layerNum < _dataSet->getNumOfTextureLevels();
                 ++layerNum)
            {
                if (!_stateset->getTextureAttribute(layerNum, osg::StateAttribute::TEXTURE))
                {
                    _stateset->setTextureAttributeAndModes(layerNum, texture, osg::StateAttribute::ON);
                }
            }
        }
        break;
    }
    case (BuildOptions::INHERIT_NEAREST_AVAILABLE):
    {
        osg::StateAttribute *texture = 0;
        unsigned int noBlanks = 0;
        for (unsigned int layerNum = 0;
             layerNum < _dataSet->getNumOfTextureLevels();
             ++layerNum)
        {
            osg::StateAttribute *localTexture = _stateset->getTextureAttribute(layerNum, osg::StateAttribute::TEXTURE);
            if (localTexture)
                texture = localTexture;
            else if (texture)
                _stateset->setTextureAttributeAndModes(layerNum, texture, osg::StateAttribute::ON);
            else
                ++noBlanks;
        }

        if (noBlanks > 0 && noBlanks != _dataSet->getNumOfTextureLevels())
        {
            // inherit downards to fill in any blanks
            for (int layerNum = _dataSet->getNumOfTextureLevels() - 1;
                 layerNum >= 0;
                 --layerNum)
            {
                osg::StateAttribute *localTexture = _stateset->getTextureAttribute(layerNum, osg::StateAttribute::TEXTURE);
                if (localTexture)
                    texture = localTexture;
                else if (texture)
                    _stateset->setTextureAttributeAndModes(layerNum, texture, osg::StateAttribute::ON);
            }
        }
        break;
    }
    case (BuildOptions::NO_INHERITANCE):
    {
        // do nothing..
        break;
    }
    }

    if (_dataSet->getCompressionMethod() == vpb::BuildOptions::GL_DRIVER)
    {
        _dataSet->getState()->checkGLErrors("DestinationTile::createStateSet()");
    }

    return _stateset.get();
}

osg::Node *DestinationTile::createHeightField()
{
    osg::ShapeDrawable *shapeDrawable = 0;

    if (!_terrain)
        _terrain = new DestinationData(_dataSet);

    if (!_terrain->_heightField)
    {
        log(osg::INFO, "**** No terrain to build tile from use flat terrain fallback ****");
        // create a dummy height field to file in the gap
        _terrain->_heightField = new osg::HeightField;
        _terrain->_heightField->allocate(8, 8);
        _terrain->_heightField->setOrigin(osg::Vec3(_extents.xMin(), _extents.yMin(), 0.0f));
        _terrain->_heightField->setXInterval(_extents.xMax() - _extents.xMin() / 7.0);
        _terrain->_heightField->setYInterval(_extents.yMax() - _extents.yMin() / 7.0);
    }

    osg::HeightField *hf = _terrain->_heightField.get();
    shapeDrawable = new osg::ShapeDrawable(hf);
    if (!shapeDrawable)
        return 0;

    hf->setSkirtHeight(shapeDrawable->getBound().radius() * _dataSet->getSkirtRatio());

    osg::StateSet *stateset = createStateSet();
    if (stateset)
    {
        shapeDrawable->setStateSet(stateset);
    }
    else
    {
        shapeDrawable->setColor(_dataSet->getDefaultColor());
    }

    osg::Geode *geode = new osg::Geode;
    geode->addDrawable(shapeDrawable);

    return geode;
}

osg::ClusterCullingCallback *DestinationTile::createClusterCullingCallback()
{
    // make sure we are dealing with a geocentric database
    if (!_dataSet->mapLatLongsToXYZ())
        return 0;

    osg::HeightField *grid = _terrain.valid() ? _terrain->_heightField.get() : 0;
    if (!grid)
        return 0;

    const osg::EllipsoidModel *et = _dataSet->getEllipsoidModel();
    double globe_radius = et ? et->getRadiusPolar() : 1.0;
    unsigned int numColumns = grid->getNumColumns();
    unsigned int numRows = grid->getNumRows();

    double midLong = grid->getOrigin().x() + grid->getXInterval() * ((double)(numColumns - 1)) * 0.5;
    double midLat = grid->getOrigin().y() + grid->getYInterval() * ((double)(numRows - 1)) * 0.5;
    double midZ = grid->getOrigin().z();

    double midX, midY;
    et->convertLatLongHeightToXYZ(osg::DegreesToRadians(midLat), osg::DegreesToRadians(midLong), midZ, midX, midY, midZ);

    osg::Vec3 center_position(midX, midY, midZ);

    osg::Vec3 center_normal(midX, midY, midZ);
    center_normal.normalize();

    osg::Vec3 transformed_center_normal = center_normal;

    unsigned int r, c;

    // populate the vertex/normal/texcoord arrays from the grid.
    double orig_X = grid->getOrigin().x();
    double delta_X = grid->getXInterval();
    double orig_Y = grid->getOrigin().y();
    double delta_Y = grid->getYInterval();
    double orig_Z = grid->getOrigin().z();

    float min_dot_product = 1.0f;
    float max_cluster_culling_height = 0.0f;
    float max_cluster_culling_radius = 0.0f;

    for (r = 0; r < numRows; ++r)
    {
        for (c = 0; c < numColumns; ++c)
        {
            double X = orig_X + delta_X * (double)c;
            double Y = orig_Y + delta_Y * (double)r;
            double Z = orig_Z + grid->getHeight(c, r);
            double height = Z;

            et->convertLatLongHeightToXYZ(osg::DegreesToRadians(Y), osg::DegreesToRadians(X), Z,
                                          X, Y, Z);

            osg::Vec3d v(X, Y, Z);
            osg::Vec3 dv = v - center_position;
            double d = sqrt(dv.x() * dv.x() + dv.y() * dv.y() + dv.z() * dv.z());
            double theta = acos(globe_radius / (globe_radius + fabs(height)));
            double phi = 2.0 * asin(d * 0.5 / globe_radius); // d/globe_radius;
            double beta = theta + phi;
            double cutoff = osg::PI_2 - 0.1;

            //log(osg::INFO,"theta="<<theta<<"\tphi="<<phi<<" beta "<<beta);
            if (phi < cutoff && beta < cutoff)
            {

                float local_dot_product = -sin(theta + phi);
                float local_m = globe_radius * (1.0 / cos(theta + phi) - 1.0);
                float local_radius = static_cast<float>(globe_radius * tan(beta)); // beta*globe_radius;
                min_dot_product = osg::minimum(min_dot_product, local_dot_product);
                max_cluster_culling_height = osg::maximum(max_cluster_culling_height, local_m);
                max_cluster_culling_radius = osg::maximum(max_cluster_culling_radius, local_radius);
            }
            else
            {
                //log(osg::INFO,"Turning off cluster culling for wrap around tile.");
                return 0;
            }
        }
    }

    // set up cluster cullling
    osg::ClusterCullingCallback *ccc = new osg::ClusterCullingCallback;

    ccc->set(center_position + transformed_center_normal * max_cluster_culling_height,
             transformed_center_normal,
             min_dot_product,
             max_cluster_culling_radius);

    return ccc;
}

osg::Node *DestinationTile::createTerrainTile()
{
    if (!_terrain)
        _terrain = new DestinationData(_dataSet);

    // call createStateSet() here even when we don't use it directly as
    // we still want any OpenGL compression and mipmap generation that it'll provide, will
    // replace with a more tuned implementation later.
    createStateSet();

    // create a default height field when non has already been created.
    if (!_terrain->_heightField)
    {
        log(osg::INFO, "**** No terrain to build tile from use flat terrain fallback ****");
        // create a dummy height field to file in the gap
        _terrain->_heightField = new osg::HeightField;
        _terrain->_heightField->allocate(8, 8);
        _terrain->_heightField->setOrigin(osg::Vec3(_extents.xMin(), _extents.yMin(), 0.0f));
        _terrain->_heightField->setXInterval((_extents.xMax() - _extents.xMin()) / 7.0);
        _terrain->_heightField->setYInterval((_extents.yMax() - _extents.yMin()) / 7.0);
    }

    osg::HeightField *hf = _terrain->_heightField.get();
    osg::EllipsoidModel *em = _dataSet->getEllipsoidModel();

    // set up the locator place the data all in the correction position
    osgTerrain::Locator *locator = new osgTerrain::Locator;
    locator->setEllipsoidModel(em);

    if (_dataSet->getDestinationCoordinateSystemNode())
    {
        locator->setFormat(_dataSet->getDestinationCoordinateSystemNode()->getFormat());
        locator->setCoordinateSystem(_dataSet->getDestinationCoordinateSystemNode()->getCoordinateSystem());
    }

    double radius = (_extents._max - _extents._min).length() * 0.5;

    if (_dataSet->getConvertFromGeographicToGeocentric())
    {
        locator->setCoordinateSystemType(osgTerrain::Locator::GEOCENTRIC);
        locator->setTransformAsExtents(osg::DegreesToRadians(_extents.xMin()),
                                       osg::DegreesToRadians(_extents.yMin()),
                                       osg::DegreesToRadians(_extents.xMax()),
                                       osg::DegreesToRadians(_extents.yMax()));

        if (em)
        {
            double midLong = hf->getOrigin().x() + hf->getXInterval() * ((double)(hf->getNumColumns() - 1)) * 0.5;
            double midLat = hf->getOrigin().y() + hf->getYInterval() * ((double)(hf->getNumRows() - 1)) * 0.5;

            double X, Y, Z = hf->getOrigin().z();
            em->convertLatLongHeightToXYZ(osg::DegreesToRadians(midLat), osg::DegreesToRadians(midLong), Z, X, Y, Z);
            osg::Vec3d center_position(X, Y, Z);

            Z = hf->getOrigin().z();
            em->convertLatLongHeightToXYZ(osg::DegreesToRadians(hf->getOrigin().y()), osg::DegreesToRadians(hf->getOrigin().x()), Z, X, Y, Z);
            osg::Vec3d origin(X, Y, Z);

            radius = (origin - center_position).length();
        }
    }
    else
    {
        locator->setTransformAsExtents(_extents.xMin(),
                                       _extents.yMin(),
                                       _extents.xMax(),
                                       _extents.yMax());

        locator->setCoordinateSystemType(osgTerrain::Locator::PROJECTED);
        // locator->setCoordinateSystemType(osgTerrain::Locator::GEOGRAPHIC);
    }

    // need to work out what the skirt should be...
    hf->setSkirtHeight(radius * _dataSet->getSkirtRatio());

    // create the terrain node that we'll hang the height field off
    osgTerrain::TerrainTile *terrainTile = new osgTerrain::TerrainTile;

    // set the location of the tile
    terrainTile->setLocator(locator);

    // set the tile ID
    terrainTile->setTileID(osgTerrain::TileID(_level, _tileX, _tileY));

    // set the blending policy (defaults to INEHRIT).
    terrainTile->setBlendingPolicy(static_cast<osgTerrain::TerrainTile::BlendingPolicy>(_dataSet->getBlendingPolicy()));

    // assign height field
    {
        osgTerrain::HeightFieldLayer *hfLayer = new osgTerrain::HeightFieldLayer;
        hfLayer->setHeightField(hf);
        hfLayer->setLocator(locator);

        terrainTile->setElevationLayer(hfLayer);
    }

    // assign the imagery
    for (unsigned int layerNum = 0;
         layerNum < getNumLayers();
         ++layerNum)
    {
        osg::Texture::FilterMode minFilter = getImageOptions(layerNum)->getMipMappingMode() == BuildOptions::NO_MIP_MAPPING ? osg::Texture::LINEAR : osg::Texture::LINEAR_MIPMAP_LINEAR;

        osg::Texture::FilterMode magFilter = osg::Texture::LINEAR;

        ImageSet &imageSet = getImageSet(layerNum);
        if (imageSet._layerSetImageDataMap.empty())
            continue;

        if (imageSet._layerSetImageDataMap.size() > 1 || !(imageSet._layerSetImageDataMap.begin()->first.empty()))
        {
            log(osg::NOTICE, "Have optional layers = %i", imageSet._layerSetImageDataMap.size());
            osgTerrain::SwitchLayer *switchLayer = new osgTerrain::SwitchLayer;
            switchLayer->setLocator(locator);
            switchLayer->setMinFilter(minFilter);
            switchLayer->setMagFilter(magFilter);
            for (ImageSet::LayerSetImageDataMap::iterator litr = imageSet._layerSetImageDataMap.begin();
                 litr != imageSet._layerSetImageDataMap.end();
                 ++litr)
            {
                ImageData &imageData = litr->second;
                if (imageData._imageDestination.valid() && imageData._imageDestination->_image.valid())
                {
                    osg::Image *image = imageData._imageDestination->_image.get();

                    osgTerrain::ImageLayer *imageLayer = new osgTerrain::ImageLayer;
                    imageLayer->setMinFilter(minFilter);
                    imageLayer->setMagFilter(magFilter);
                    imageLayer->setImage(image);
                    imageLayer->setSetName(litr->first);
                    imageLayer->setLocator(locator);

                    switchLayer->addLayer(imageLayer);
                }
            }
            terrainTile->setColorLayer(layerNum, switchLayer);
        }
        else
        {
            ImageData &imageData = imageSet._layerSetImageDataMap.begin()->second;
            if (imageData._imageDestination.valid() && imageData._imageDestination->_image.valid())
            {
                osg::Image *image = imageData._imageDestination->_image.get();

                osgTerrain::ImageLayer *imageLayer = new osgTerrain::ImageLayer;
                imageLayer->setImage(image);
                imageLayer->setLocator(locator);
                imageLayer->setMinFilter(minFilter);
                imageLayer->setMagFilter(magFilter);

                terrainTile->setColorLayer(layerNum, imageLayer);
            }
        }
    }

    switch (_dataSet->getLayerInheritance())
    {
    case (BuildOptions::INHERIT_LOWEST_AVAILABLE):
    {
        osgTerrain::Layer *layer = 0;
        // first look for an available Layer
        for (unsigned int layerNum = 0;
             layerNum < _dataSet->getNumOfTextureLevels() && layer == 0;
             ++layerNum)
        {
            layer = terrainTile->getColorLayer(layerNum);
        }

        if (layer)
        {
            // now fill in any blanks
            for (unsigned int layerNum = 0;
                 layerNum < _dataSet->getNumOfTextureLevels();
                 ++layerNum)
            {
                if (!terrainTile->getColorLayer(layerNum))
                {
                    terrainTile->setColorLayer(layerNum, layer);
                }
            }
        }
        break;
    }
    case (BuildOptions::INHERIT_NEAREST_AVAILABLE):
    {
        osgTerrain::Layer *layer = 0;
        unsigned int noBlanks = 0;
        for (unsigned int layerNum = 0;
             layerNum < _dataSet->getNumOfTextureLevels();
             ++layerNum)
        {
            osgTerrain::Layer *localLayer = terrainTile->getColorLayer(layerNum);
            if (localLayer)
                layer = localLayer;
            else if (layer)
                terrainTile->setColorLayer(layerNum, layer);
            else
                ++noBlanks;
        }

        if (noBlanks > 0 && noBlanks != _dataSet->getNumOfTextureLevels())
        {
            // inherit downards to fill in any blanks
            for (int layerNum = _dataSet->getNumOfTextureLevels() - 1;
                 layerNum >= 0;
                 --layerNum)
            {
                osgTerrain::Layer *localLayer = terrainTile->getColorLayer(layerNum);
                if (localLayer)
                    layer = localLayer;
                else if (layer)
                    terrainTile->setColorLayer(layerNum, layer);
            }
        }
        break;
    }
    case (BuildOptions::NO_INHERITANCE):
    {
        // do nothing..
        break;
    }
    }

    // assign cluster culling callback to terrain
    terrainTile->setCullCallback(createClusterCullingCallback());

    return terrainTile;
}

static osg::Vec3 computeLocalPosition(const osg::Matrixd &worldToLocal, double X, double Y, double Z)
{
    return osg::Vec3(X * worldToLocal(0, 0) + Y * worldToLocal(1, 0) + Z * worldToLocal(2, 0) + worldToLocal(3, 0),
                     X * worldToLocal(0, 1) + Y * worldToLocal(1, 1) + Z * worldToLocal(2, 1) + worldToLocal(3, 1),
                     X * worldToLocal(0, 2) + Y * worldToLocal(1, 2) + Z * worldToLocal(2, 2) + worldToLocal(3, 2));
}

static inline osg::Vec3 computeLocalSkirtVector(const osg::EllipsoidModel *et, const osg::HeightField *grid, unsigned int i, unsigned int j, float length, bool useLocalToTileTransform, const osg::Matrixd &localToWorld)
{
    // no local to tile transform + mapping from lat+longs to XYZ so we need to use
    // a rotatated skirt vector - use the gravity vector.
    double longitude = grid->getOrigin().x() + grid->getXInterval() * ((double)(i));
    double latitude = grid->getOrigin().y() + grid->getYInterval() * ((double)(j));
    double midZ = grid->getOrigin().z();
    double X, Y, Z;
    et->convertLatLongHeightToXYZ(osg::DegreesToRadians(latitude), osg::DegreesToRadians(longitude), midZ, X, Y, Z);
    osg::Vec3 gravitationVector = et->computeLocalUpVector(X, Y, Z);
    gravitationVector.normalize();

    if (useLocalToTileTransform)
        gravitationVector = osg::Matrixd::transform3x3(localToWorld, gravitationVector);

    return gravitationVector * -length;
}

osg::Node *DestinationTile::createPolygonal()
{
    log(osg::INFO, "--------- DestinationTile::createDrawableGeometry() ------------- ");

    const osg::EllipsoidModel *et = _dataSet->getEllipsoidModel();
    bool mapLatLongsToXYZ = _dataSet->mapLatLongsToXYZ();
    bool useLocalToTileTransform = _dataSet->getUseLocalTileTransform();

    osg::ref_ptr<osg::HeightField> grid = 0;

    if (_terrain.valid() && _terrain->_heightField.valid())
    {
        log(osg::INFO, "--- Have terrain build tile ----");
        grid = _terrain->_heightField.get();
    }
    else
    {
        unsigned int minimumSize = 4;
        unsigned int numColumns = minimumSize;
        unsigned int numRows = minimumSize;

        if (mapLatLongsToXYZ)
        {
            float longitude_range = (_extents.xMax() - _extents.xMin());
            float latitude_range = (_extents.yMax() - _extents.yMin());

            if (longitude_range > 45.0)
                numColumns = (unsigned int)ceilf((float)numColumns * sqrtf(longitude_range / 45.0));
            if (latitude_range > 45.0)
                numRows = (unsigned int)ceilf((float)numRows * sqrtf(latitude_range / 45.0));

            log(osg::INFO, "numColumns = %d numRows=%d", numColumns, numRows);
        }
        else
        {
            float ratio_y_over_x = (_extents.yMax() - _extents.yMin()) / (_extents.xMax() - _extents.xMin());
            if (ratio_y_over_x > 1.2)
                numRows = (unsigned int)ceilf((float)numRows * ratio_y_over_x);
            else if (ratio_y_over_x < 0.8)
                numColumns = (unsigned int)ceilf((float)numColumns / ratio_y_over_x);
        }

        grid = new osg::HeightField;
        grid->allocate(numColumns, numRows);
        grid->setOrigin(osg::Vec3(_extents.xMin(), _extents.yMin(), 0.0f));
        grid->setXInterval(double(_extents.xMax() - _extents.xMin()) / double(numColumns - 1));
        grid->setYInterval(double(_extents.yMax() - _extents.yMin()) / double(numRows - 1));

        if (!_terrain)
            _terrain = new DestinationData(_dataSet);

        _terrain->_heightField = grid;
    }

    if (!grid)
    {
        log(osg::INFO, "**** No terrain to build tile from use flat terrain fallback ****");

        return 0;
    }

    bool createSkirt = true;
    float skirtRatio = _dataSet->getSkirtRatio();
    if (skirtRatio == 0.0f)
        createSkirt = false;

    // compute sizes.
    unsigned int numColumns = grid->getNumColumns();
    unsigned int numRows = grid->getNumRows();
    unsigned int numVerticesInBody = numColumns * numRows;
    unsigned int numVerticesInSkirt = createSkirt ? numColumns * 2 + numRows * 2 - 4 : 0;
    unsigned int numVertices = numVerticesInBody + numVerticesInSkirt;

    // create the geometry.
    osg::Geometry *geometry = new osg::Geometry;

    osg::Vec3Array &v = *(new osg::Vec3Array(numVertices));
    osg::Vec2Array &t = *(new osg::Vec2Array(numVertices));
    osg::Vec4ubArray &color = *(new osg::Vec4ubArray(1));

    color[0].set(255, 255, 255, 255);

    osg::ref_ptr<osg::Vec3Array> n = new osg::Vec3Array(numVertices); // must use ref_ptr so the array isn't removed when smooothvisitor is used

    _localToWorld.makeIdentity();
    _worldToLocal.makeIdentity();
    osg::Vec3 skirtVector(0.0f, 0.0f, 0.0f);

    osg::Vec3 center_position(0.0f, 0.0f, 0.0f);
    osg::Vec3 center_normal(0.0f, 0.0f, 1.0f);
    osg::Vec3 transformed_center_normal(0.0f, 0.0f, 1.0f);
    double globe_radius = et ? et->getRadiusPolar() : 1.0;
    float skirtLength = _extents.radius() * skirtRatio;

    bool useClusterCullingCallback = mapLatLongsToXYZ;

    if (useLocalToTileTransform)
    {
        if (mapLatLongsToXYZ)
        {
            double midLong = grid->getOrigin().x() + grid->getXInterval() * ((double)(numColumns - 1)) * 0.5;
            double midLat = grid->getOrigin().y() + grid->getYInterval() * ((double)(numRows - 1)) * 0.5;
            double midZ = grid->getOrigin().z();
            et->computeLocalToWorldTransformFromLatLongHeight(osg::DegreesToRadians(midLat), osg::DegreesToRadians(midLong), midZ, _localToWorld);

            double minLong = grid->getOrigin().x();
            double minLat = grid->getOrigin().y();

            double minX, minY, minZ;
            et->convertLatLongHeightToXYZ(osg::DegreesToRadians(minLat), osg::DegreesToRadians(minLong), midZ, minX, minY, minZ);

            double midX, midY;
            et->convertLatLongHeightToXYZ(osg::DegreesToRadians(midLat), osg::DegreesToRadians(midLong), midZ, midX, midY, midZ);

            double length = sqrt((midX - minX) * (midX - minX) + (midY - minY) * (midY - minY));

            skirtLength = length * skirtRatio;
            skirtVector.set(0.0f, 0.0f, -skirtLength);

            center_normal.set(midX, midY, midZ);
            center_normal.normalize();

            _worldToLocal.invert(_localToWorld);

            center_position = computeLocalPosition(_worldToLocal, midX, midY, midZ);
            transformed_center_normal = osg::Matrixd::transform3x3(_localToWorld, center_normal);
        }
        else
        {
            double midX = grid->getOrigin().x() + grid->getXInterval() * ((double)(numColumns - 1)) * 0.5;
            double midY = grid->getOrigin().y() + grid->getYInterval() * ((double)(numRows - 1)) * 0.5;
            double midZ = grid->getOrigin().z();
            _localToWorld.makeTranslate(midX, midY, midZ);
            _worldToLocal.invert(_localToWorld);

            skirtVector.set(0.0f, 0.0f, -skirtLength);
        }
    }
    else if (mapLatLongsToXYZ)
    {
        // no local to tile transform + mapping from lat+longs to XYZ so we need to use
        // a rotatated skirt vector - use the gravity vector.
        double midLong = grid->getOrigin().x() + grid->getXInterval() * ((double)(numColumns - 1)) * 0.5;
        double midLat = grid->getOrigin().y() + grid->getYInterval() * ((double)(numRows - 1)) * 0.5;
        double midZ = grid->getOrigin().z();
        double X, Y, Z;
        et->convertLatLongHeightToXYZ(osg::DegreesToRadians(midLat), osg::DegreesToRadians(midLong), midZ, X, Y, Z);
        osg::Vec3 gravitationVector = et->computeLocalUpVector(X, Y, Z);
        gravitationVector.normalize();
        skirtVector = gravitationVector * skirtLength;
    }
    else
    {
        skirtVector.set(0.0f, 0.0f, -skirtLength);
    }

    unsigned int vi = 0;
    unsigned int r, c;

    // populate the vertex/normal/texcoord arrays from the grid.
    double orig_X = _extents.xMin();
    double orig_Y = _extents.yMin();
    double orig_Z = 0.0;

    double delta_X = double(_extents.xMax() - _extents.xMin()) / double(numColumns - 1);
    double delta_Y = double(_extents.yMax() - _extents.yMin()) / double(numRows - 1);

    float min_dot_product = 1.0f;
    float max_cluster_culling_height = 0.0f;
    float max_cluster_culling_radius = 0.0f;

    for (r = 0; r < numRows; ++r)
    {
        for (c = 0; c < numColumns; ++c)
        {
            double X = orig_X + delta_X * (double)c;
            double Y = orig_Y + delta_Y * (double)r;
            double Z = orig_Z + grid->getHeight(c, r);
            double height = Z;

            if (mapLatLongsToXYZ)
            {
                et->convertLatLongHeightToXYZ(osg::DegreesToRadians(Y), osg::DegreesToRadians(X), Z,
                                              X, Y, Z);
            }

            if (useLocalToTileTransform)
            {
                v[vi] = computeLocalPosition(_worldToLocal, X, Y, Z);
            }
            else
            {
                v[vi].set(X, Y, Z);
            }

            if (useClusterCullingCallback)
            {
                osg::Vec3 dv = v[vi] - center_position;
                double d = sqrt(dv.x() * dv.x() + dv.y() * dv.y() + dv.z() * dv.z());
                double theta = acos(globe_radius / (globe_radius + fabs(height)));
                double phi = 2.0 * asin(d * 0.5 / globe_radius); // d/globe_radius;
                double beta = theta + phi;
                double cutoff = osg::PI_2 - 0.1;
                //log(osg::INFO,"theta="<<theta<<"\tphi="<<phi<<" beta "<<beta);
                if (phi < cutoff && beta < cutoff)
                {

                    float local_dot_product = -sin(theta + phi);
                    float local_m = globe_radius * (1.0 / cos(theta + phi) - 1.0);
                    float local_radius = static_cast<float>(globe_radius * tan(beta)); // beta*globe_radius;
                    min_dot_product = osg::minimum(min_dot_product, local_dot_product);
                    max_cluster_culling_height = osg::maximum(max_cluster_culling_height, local_m);
                    max_cluster_culling_radius = osg::maximum(max_cluster_culling_radius, local_radius);
                }
                else
                {
                    //log(osg::INFO,"Turning off cluster culling for wrap around tile.");
                    useClusterCullingCallback = false;
                }
            }

            // note normal will need rotating.
            if (n.valid())
            {
                (*n)[vi] = grid->getNormal(c, r);
            }

            t[vi].x() = (c == numColumns - 1) ? 1.0f : (float)(c) / (float)(numColumns - 1);
            t[vi].y() = (r == numRows - 1) ? 1.0f : (float)(r) / (float)(numRows - 1);

            ++vi;
        }
    }

    //geometry->setUseDisplayList(false);
    geometry->setVertexArray(&v);

    if (n.valid())
    {
        geometry->setNormalArray(n.get());
        geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    geometry->setColorArray(&color);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    bool fillInAllTextureUnits = true;
    if (fillInAllTextureUnits)
    {
        for (unsigned int layerNum = 0;
             layerNum < _dataSet->getNumOfTextureLevels();
             ++layerNum)
        {
            geometry->setTexCoordArray(layerNum, &t);
        }
    }
    else
    {
        for (unsigned int layerNum = 0;
             layerNum < getNumLayers();
             ++layerNum)
        {
            ImageSet &imageSet = getImageSet(layerNum);
            if (imageSet._layerSetImageDataMap.empty())
                continue;

            ImageData &imageData = imageSet._layerSetImageDataMap.begin()->second;
            if (imageData._imageDestination.valid() && imageData._imageDestination->_image.valid())
            {
                geometry->setTexCoordArray(layerNum, &t);
            }
        }
    }

    osg::DrawElementsUInt &drawElements = *(new osg::DrawElementsUInt(GL_TRIANGLES, 2 * 3 * (numColumns - 1) * (numRows - 1)));
    geometry->addPrimitiveSet(&drawElements);
    int ei = 0;
    for (r = 0; r < numRows - 1; ++r)
    {
        for (c = 0; c < numColumns - 1; ++c)
        {
            unsigned int i00 = (r)*numColumns + c;
            unsigned int i10 = (r)*numColumns + c + 1;
            unsigned int i01 = (r + 1) * numColumns + c;
            unsigned int i11 = (r + 1) * numColumns + c + 1;

            float diff_00_11 = fabsf(v[i00].z() - v[i11].z());
            float diff_01_10 = fabsf(v[i01].z() - v[i10].z());
            if (diff_00_11 < diff_01_10)
            {
                // diagonal between 00 and 11
                drawElements[ei++] = i00;
                drawElements[ei++] = i10;
                drawElements[ei++] = i11;

                drawElements[ei++] = i00;
                drawElements[ei++] = i11;
                drawElements[ei++] = i01;
            }
            else
            {
                // diagonal between 01 and 10
                drawElements[ei++] = i01;
                drawElements[ei++] = i00;
                drawElements[ei++] = i10;

                drawElements[ei++] = i01;
                drawElements[ei++] = i10;
                drawElements[ei++] = i11;
            }
        }
    }

#if 1
    osgUtil::SmoothingVisitor sv;
    sv.smooth(*geometry); // this will replace the normal vector with a new one

    // now we have to reassign the normals back to the orignal pointer.
    n = dynamic_cast<osg::Vec3Array *>(geometry->getNormalArray());
    if (n.valid() && n->size() != numVertices)
        n->resize(numVertices);
#endif
    // now apply the normals computed through equalization
    for (unsigned int position = 0; position < NUMBER_OF_POSITIONS; ++position)
    {
        if (!_heightDeltas[position].empty())
        {
            // we have normal to apply
            unsigned int i = 0;
            unsigned int j = 0;
            unsigned int deltai = 0;
            unsigned int deltaj = 0;
            switch (position)
            {
            case LEFT:
                i = 0;
                j = 1;
                deltai = 0;
                deltaj = 1;
                break;
            case LEFT_BELOW:
                i = 0;
                j = 0;
                deltai = 0;
                deltaj = 0;
                break;
            case BELOW:
                i = 1;
                j = 0;
                deltai = 1;
                deltaj = 0;
                break;
            case BELOW_RIGHT:
                i = numColumns - 1;
                j = 0;
                deltai = 0;
                deltaj = 0;
                break;
            case RIGHT:
                i = numColumns - 1;
                j = 1;
                deltai = 0;
                deltaj = 1;
                break;
            case RIGHT_ABOVE:
                i = numColumns - 1;
                j = numRows - 1;
                deltai = 0;
                deltaj = 0;
                break;
            case ABOVE:
                i = 1;
                j = numRows - 1;
                deltai = 1;
                deltaj = 0;
                break;
            case ABOVE_LEFT:
                i = 0;
                j = numRows - 1;
                deltai = 0;
                deltaj = 0;
                break;
            }

            // need to reproject normals.
            for (HeightDeltaList::iterator itr = _heightDeltas[position].begin();
                 itr != _heightDeltas[position].end();
                 ++itr, i += deltai, j += deltaj)
            {
                osg::Vec3 &normal = (*n)[i + j * numColumns];
                osg::Vec2 heightDelta = *itr;

                if (mapLatLongsToXYZ)
                {

                    double X = orig_X + delta_X * (double)i;
                    double Y = orig_Y + delta_Y * (double)j;
                    double Z = orig_Z + grid->getHeight(i, j);
                    osg::Matrixd normalLocalToWorld;
                    et->computeLocalToWorldTransformFromLatLongHeight(osg::DegreesToRadians(Y), osg::DegreesToRadians(X), Z, normalLocalToWorld);
                    osg::Matrixd normalToLocalReferenceFrame(normalLocalToWorld * _worldToLocal);

                    // need to compute the x and y delta for this point in space.
                    double X0, Y0, Z0;
                    double X1, Y1, Z1;
                    double X2, Y2, Z2;

                    et->convertLatLongHeightToXYZ(osg::DegreesToRadians(Y), osg::DegreesToRadians(X), Z,
                                                  X0, Y0, Z0);

                    et->convertLatLongHeightToXYZ(osg::DegreesToRadians(Y), osg::DegreesToRadians(X + delta_X), Z,
                                                  X1, Y1, Z1);

                    et->convertLatLongHeightToXYZ(osg::DegreesToRadians(Y + delta_Y), osg::DegreesToRadians(X), Z,
                                                  X2, Y2, Z2);

                    X1 -= X0;
                    Y1 -= Y0;
                    Z1 -= Z0;

                    X2 -= X0;
                    Y2 -= Y0;
                    Z2 -= Z0;

                    float xInterval = sqrt(X1 * X1 + Y1 * Y1 + Z1 * Z1);
                    float yInterval = sqrt(X2 * X2 + Y2 * Y2 + Z2 * Z2);

                    // need to set up the normal from the scaled heightDelta.
                    normal.x() = -heightDelta.x() / xInterval;
                    normal.y() = -heightDelta.y() / yInterval;
                    normal.z() = 1.0f;

                    normal = osg::Matrixd::transform3x3(normal, normalToLocalReferenceFrame);
                    normal.normalize();
                }
                else
                {
                    normal.x() = -heightDelta.x() / grid->getXInterval();
                    normal.y() = -heightDelta.y() / grid->getYInterval();
                    normal.z() = 1.0f;
                    normal.normalize();
                }
            }
        }
    }

#if 0
    std::cout<<"Normals");
    for(osg::Vec3Array::iterator nitr = n->begin();
        nitr != n->end();
        ++nitr)
    {
        osg::Vec3& normal = *nitr;
        std::cout<<"   Local normal = "<<normal<< " vs "<<transformed_center_normal);
    }
#endif

    if (useClusterCullingCallback)
    {
        // set up cluster cullling
        osg::ClusterCullingCallback *ccc = new osg::ClusterCullingCallback;

        ccc->set(center_position + transformed_center_normal * max_cluster_culling_height,
                 transformed_center_normal,
                 min_dot_product,
                 max_cluster_culling_radius);
        geometry->setCullCallback(ccc);
    }

    osgUtil::Simplifier::IndexList pointsToProtectDuringSimplification;

    if (numVerticesInSkirt > 0)
    {
        osg::DrawElementsUInt &skirtDrawElements = *(new osg::DrawElementsUInt(GL_QUAD_STRIP, 2 * numVerticesInSkirt + 2));
        geometry->addPrimitiveSet(&skirtDrawElements);
        int ei = 0;
        int firstSkirtVertexIndex = vi;
        // create bottom skirt vertices
        r = 0;
        for (c = 0; c < numColumns - 1; ++c)
        {
            // assign indices to primitive set
            skirtDrawElements[ei++] = (r)*numColumns + c;
            skirtDrawElements[ei++] = vi;

            // mark these points as protected to prevent them from being removed during simplification
            pointsToProtectDuringSimplification.push_back((r)*numColumns + c);
            pointsToProtectDuringSimplification.push_back(vi);

            osg::Vec3 localSkirtVector = !mapLatLongsToXYZ ? skirtVector : computeLocalSkirtVector(et, grid.get(), c, r, skirtLength, useLocalToTileTransform, _localToWorld);

            // add in the new point on the bottom of the skirt
            v[vi] = v[(r)*numColumns + c] + localSkirtVector;
            if (n.valid())
                (*n)[vi] = (*n)[r * numColumns + c];
            t[vi++] = t[(r)*numColumns + c];
        }
        // create right skirt vertices
        c = numColumns - 1;
        for (r = 0; r < numRows - 1; ++r)
        {
            // assign indices to primitive set
            skirtDrawElements[ei++] = (r)*numColumns + c;
            skirtDrawElements[ei++] = vi;

            // mark these points as protected to prevent them from being removed during simplification
            pointsToProtectDuringSimplification.push_back((r)*numColumns + c);
            pointsToProtectDuringSimplification.push_back(vi);

            osg::Vec3 localSkirtVector = !mapLatLongsToXYZ ? skirtVector : computeLocalSkirtVector(et, grid.get(), c, r, skirtLength, useLocalToTileTransform, _localToWorld);

            // add in the new point on the bottom of the skirt
            v[vi] = v[(r)*numColumns + c] + localSkirtVector;
            if (n.valid())
                (*n)[vi] = (*n)[(r)*numColumns + c];
            t[vi++] = t[(r)*numColumns + c];
        }
        // create top skirt vertices
        r = numRows - 1;
        for (c = numColumns - 1; c > 0; --c)
        {
            // assign indices to primitive set
            skirtDrawElements[ei++] = (r)*numColumns + c;
            skirtDrawElements[ei++] = vi;

            // mark these points as protected to prevent them from being removed during simplification
            pointsToProtectDuringSimplification.push_back((r)*numColumns + c);
            pointsToProtectDuringSimplification.push_back(vi);

            osg::Vec3 localSkirtVector = !mapLatLongsToXYZ ? skirtVector : computeLocalSkirtVector(et, grid.get(), c, r, skirtLength, useLocalToTileTransform, _localToWorld);

            // add in the new point on the bottom of the skirt
            v[vi] = v[(r)*numColumns + c] + localSkirtVector;
            if (n.valid())
                (*n)[vi] = (*n)[(r)*numColumns + c];
            t[vi++] = t[(r)*numColumns + c];
        }
        // create left skirt vertices
        c = 0;
        for (r = numRows - 1; r > 0; --r)
        {
            // assign indices to primitive set
            skirtDrawElements[ei++] = (r)*numColumns + c;
            skirtDrawElements[ei++] = vi;

            // mark these points as protected to prevent them from being removed during simplification
            pointsToProtectDuringSimplification.push_back((r)*numColumns + c);
            pointsToProtectDuringSimplification.push_back(vi);

            osg::Vec3 localSkirtVector = !mapLatLongsToXYZ ? skirtVector : computeLocalSkirtVector(et, grid.get(), c, r, skirtLength, useLocalToTileTransform, _localToWorld);

            // add in the new point on the bottom of the skirt
            v[vi] = v[(r)*numColumns + c] + localSkirtVector;
            if (n.valid())
                (*n)[vi] = (*n)[(r)*numColumns + c];
            t[vi++] = t[(r)*numColumns + c];
        }
        skirtDrawElements[ei++] = 0;
        skirtDrawElements[ei++] = firstSkirtVertexIndex;
    }

    if (n.valid())
    {
        geometry->setNormalArray(n.get());
        geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    }

    osg::StateSet *stateset = createStateSet();
    if (stateset)
    {
        geometry->setStateSet(stateset);
    }
    else
    {
        osg::Vec4Array *colours = new osg::Vec4Array(1);
        (*colours)[0] = _dataSet->getDefaultColor();

        geometry->setColorArray(colours);
        geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    }

    osg::Geode *geode = new osg::Geode;
    geode->addDrawable(geometry);

    if (_dataSet->getWriteNodeBeforeSimplification())
    {
        osgDB::writeNodeFile(*geode, "NodeBeforeSimplification.osg");
    }

    if (_dataSet->getSimplifyTerrain())
    {
        unsigned int targetMaxNumVertices = 512;
        double sample_ratio = (numVertices <= targetMaxNumVertices) ? 1.0 : (double)targetMaxNumVertices / (double)numVertices;
        double radius = double(geometry->getBound().radius());
        double maximumError = radius / 2000.0;

        osgUtil::Simplifier simplifier(sample_ratio, maximumError);

        simplifier.setDoTriStrip(false);
        simplifier.setSmoothing(false);

        simplifier.simplify(*geometry, pointsToProtectDuringSimplification); // this will replace the normal vector with a new one
    }

    if (useLocalToTileTransform)
    {
        osg::MatrixTransform *mt = new osg::MatrixTransform;
        mt->setMatrix(_localToWorld);
        mt->addChild(geode);

        bool addLocalAxes = false;
        if (addLocalAxes)
        {
            float s = geode->getBound().radius() * 0.5f;
            osg::MatrixTransform *scaleAxis = new osg::MatrixTransform;
            scaleAxis->setMatrix(osg::Matrix::scale(s, s, s));
            scaleAxis->addChild(osgDB::readNodeFile("axes.osg"));
            mt->addChild(scaleAxis);
        }

        return mt;
    }
    else
    {
        return geode;
    }
}

void DestinationTile::readFrom(Source *source)
{
    bool optionalLayerSet = _dataSet->isOptionalLayerSet(source->getSetName());
    log(osg::NOTICE, "DestinationTile::readFrom(SetName=%s, FileName=%s)", source->getSetName().c_str(), source->getFileName().c_str());
    if (optionalLayerSet)
        log(osg::NOTICE, "  is an optional layer set");

    if (source && source->intersects(*this) && _level >= source->getMinLevel() && _level <= source->getMaxLevel() && source->getSourceData())
    {
        log(osg::INFO, "DestinationTile::readFrom -> SourceData::read() ");
        log(osg::INFO, "    destination._level=%d\t%d\t%d", _level, source->getMinLevel(), source->getMaxLevel());

        SourceData *data = source->getSourceData();
        switch (source->getType())
        {
        case (Source::IMAGE):
        {
            unsigned int layerNum = source->getLayer();
            switch (_dataSet->getLayerInheritance())
            {
            case (BuildOptions::INHERIT_LOWEST_AVAILABLE):
            {
                if (layerNum == 0)
                {
                    // copy the base layer 0 into layer 0 and all subsequent layers to provide a backdrop.
                    for (unsigned int i = 0; i < getNumLayers(); ++i)
                    {
                        ImageSet &imageSet = getImageSet(i);
                        if (imageSet._layerSetImageDataMap.empty())
                            continue;

                        ImageData &imageData = imageSet._layerSetImageDataMap.begin()->second;
                        if (imageData._imageDestination.valid())
                        {
                            data->read(*(imageData._imageDestination));
                        }
                    }
                }
                else
                {
                    // copy specific layer.
                    ImageSet &imageSet = getImageSet(layerNum);
                    if (imageSet._layerSetImageDataMap.empty())
                        break;

                    ImageData &imageData = imageSet._layerSetImageDataMap.begin()->second;
                    if (layerNum < getNumLayers() && imageData._imageDestination.valid())
                    {
                        data->read(*(imageData._imageDestination));
                    }
                }
                break;
            }
            case (BuildOptions::INHERIT_NEAREST_AVAILABLE):
            {
                // copy the current layer into this and all subsequent layers to provide a backdrop.
                for (unsigned int i = layerNum; i < getNumLayers(); ++i)
                {
                    ImageSet &imageSet = getImageSet(i);
                    if (imageSet._layerSetImageDataMap.empty())
                        continue;

                    ImageData &imageData = imageSet._layerSetImageDataMap.begin()->second;
                    if (imageData._imageDestination.valid())
                    {
                        data->read(*(imageData._imageDestination));
                    }
                }
                break;
            }
            case (BuildOptions::NO_INHERITANCE):
            {
                if (layerNum < getNumLayers())
                {
                    ImageSet &imageSet = getImageSet(layerNum);
                    if (imageSet._layerSetImageDataMap.empty())
                        break;

                    ImageData &imageData = imageSet._layerSetImageDataMap.begin()->second;
                    if (imageData._imageDestination.valid())
                    {
                        data->read(*(imageData._imageDestination));
                    }
                }
                break;
            }
            }
            break;
        }
        case (Source::HEIGHT_FIELD):
        {
            if (_terrain.valid())
                data->read(*_terrain);
            break;
        }
        case (Source::MODEL):
        {
            if (!_models)
                _models = new DestinationData(_dataSet);
            data->read(*_models);
            break;
        }
        case (Source::SHAPEFILE):
        {
            if (!_models)
                _models = new DestinationData(_dataSet);
            data->read(*_models);
            break;
        }
        default:
        {
            log(osg::NOTICE, "DestinationTile::readFrom() source type of file %s not handled", source->getFileName().c_str());
            break;
        }
        }
    }
}

void DestinationTile::readFrom(CompositeSource *sourceGraph)
{
    if (sourceGraph)
    {

        allocate();

        unsigned int numChecked = 0;
        for (CompositeSource::source_iterator itr(sourceGraph); itr.valid(); ++itr)
        {
            ++numChecked;
            readFrom(itr->get());
        }

        log(osg::INFO, "DestinationTile::readFrom(CompositeSource* ) numChecked %i", numChecked);

        optimizeResolution();
    }
    else
    {
        readFrom();
    }
}

void DestinationTile::readFrom()
{
    allocate();

    log(osg::INFO, "DestinationTile::readFrom() %i", _sources.size());
    for (Sources::iterator itr = _sources.begin();
         itr != _sources.end();
         ++itr)
    {
        readFrom(itr->get());
    }

    optimizeResolution();
}

void DestinationTile::unrefData()
{
    _imageLayerSet.clear();
    _terrain = 0;
    _models = 0;

    _createdScene = 0;
    _stateset = 0;
}

void DestinationTile::addRequiredResolutions(CompositeSource *sourceGraph)
{
    for (CompositeSource::source_iterator itr(sourceGraph); itr.valid(); ++itr)
    {
        Source *source = itr->get();
        if (source && source->intersects(*this))
        {
            if (source->getType() == Source::IMAGE)
            {
                unsigned int numCols, numRows;
                double resX, resY;
                if (computeImageResolution(source->getLayer(), source->getSwitchSetName(), numCols, numRows, resX, resY))
                {
                    source->addRequiredResolution(resX, resY);
                }
            }

            if (source->getType() == Source::HEIGHT_FIELD)
            {
                unsigned int numCols, numRows;
                double resX, resY;
                if (computeTerrainResolution(numCols, numRows, resX, resY))
                {
                    source->addRequiredResolution(resX, resY);
                }
            }
        }
    }
}

void CompositeDestination::computeNeighboursFromQuadMap()
{
    // handle leaves
    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        (*titr)->computeNeighboursFromQuadMap();
    }

    // handle chilren
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        (*citr)->computeNeighboursFromQuadMap();
    }
}

void CompositeDestination::addRequiredResolutions(CompositeSource *sourceGraph)
{
    // handle leaves
    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        (*titr)->addRequiredResolutions(sourceGraph);
    }

    // handle chilren
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        (*citr)->addRequiredResolutions(sourceGraph);
    }
}

void CompositeDestination::readFrom(CompositeSource *sourceGraph)
{
    log(osg::INFO, "CompositeDestination::readFrom() ");

    // handle leaves
    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        (*titr)->readFrom(sourceGraph);
    }

    // handle chilren
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        (*citr)->readFrom(sourceGraph);
    }
}

void CompositeDestination::addSource(Source *source)
{
    // handle leaves
    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        (*titr)->addSource(source);
    }

    // handle chilren
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        (*citr)->addSource(source);
    }
}

void CompositeDestination::computeMaximumSourceResolution()
{
    // handle leaves
    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        (*titr)->computeMaximumSourceResolution();
    }

    // handle chilren
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        (*citr)->computeMaximumSourceResolution();
    }
}

void CompositeDestination::equalizeBoundaries()
{
    // handle leaves
    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        (*titr)->equalizeBoundaries();
    }

    // handle chilren
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        (*citr)->equalizeBoundaries();
    }
}

class CollectClusterCullingCallbacks : public osg::NodeVisitor
{
public:
    struct Triple
    {
        Triple()
            : _object(0)
            , _callback(0)
        {
        }

        Triple(osg::NodePath nodePath, osg::Object *object, osg::ClusterCullingCallback *callback)
            : _nodePath(nodePath)
            , _object(object)
            , _callback(callback)
        {
        }

        Triple(const Triple &t)
            : _nodePath(t._nodePath)
            , _object(t._object)
            , _callback(t._callback)
        {
        }

        Triple &operator=(const Triple &t)
        {
            _nodePath = t._nodePath;
            _object = t._object;
            _callback = t._callback;
            return *this;
        }

        osg::NodePath _nodePath;
        osg::Object *_object;
        osg::ClusterCullingCallback *_callback;
    };

    typedef std::vector<Triple> ClusterCullingCallbackList;

    CollectClusterCullingCallbacks()
        : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
    {
    }

    virtual void apply(osg::Group &group)
    {
        osgTerrain::Terrain *terrain = dynamic_cast<osgTerrain::Terrain *>(&group);
        if (terrain)
        {
            osg::ClusterCullingCallback *callback = dynamic_cast<osg::ClusterCullingCallback *>(terrain->getCullCallback());
            if (callback)
            {
                _callbackList.push_back(Triple(getNodePath(), terrain, callback));
            }
            return;
        }

        osgTerrain::TerrainTile *terrainTile = dynamic_cast<osgTerrain::TerrainTile *>(&group);
        if (terrainTile)
        {
            osg::ClusterCullingCallback *callback = dynamic_cast<osg::ClusterCullingCallback *>(terrainTile->getCullCallback());
            if (callback)
            {
                _callbackList.push_back(Triple(getNodePath(), terrain, callback));
            }
            return;
        }
        else
        {
            osg::NodeVisitor::apply(group);
        }
    }

    virtual void apply(osg::Geode &geode)
    {
        for (unsigned int i = 0; i < geode.getNumDrawables(); ++i)
        {
            osg::ClusterCullingCallback *callback = dynamic_cast<osg::ClusterCullingCallback *>(geode.getDrawable(i)->getCullCallback());
            if (callback)
            {
                _callbackList.push_back(Triple(getNodePath(), geode.getDrawable(i), callback));
            }
        }
    }

    ClusterCullingCallbackList _callbackList;
};

osg::Node *CompositeDestination::createScene()
{
    if (_children.empty() && _tiles.empty())
        return 0;

    if (_children.empty() && _tiles.size() == 1)
        return _tiles.front()->createScene();

    if (_tiles.empty() && _children.size() == 1)
        return _children.front()->createScene();

    if (_type == GROUP)
    {
        osg::Group *group = new osg::Group;
        for (TileList::iterator titr = _tiles.begin();
             titr != _tiles.end();
             ++titr)
        {
            osg::Node *node = (*titr)->createScene();
            if (node)
                group->addChild(node);
        }

        // handle chilren
        for (ChildList::iterator citr = _children.begin();
             citr != _children.end();
             ++citr)
        {
            osg::Node *node = (*citr)->createScene();
            if (node)
                group->addChild(node);
        }
        return group;
    }

#if 1
    typedef std::vector<osg::Node *> NodeList;

    // collect all the local tiles
    NodeList tileNodes;
    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        osg::Node *node = (*titr)->createScene();
        if (node)
            tileNodes.push_back(node);
    }

    NodeList childNodes;
    ChildList::iterator citr;
    for (citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        osg::Node *node = (*citr)->createScene();
        if (node)
            childNodes.push_back(node);
    }

    float cutOffDistance = -FLT_MAX;
    for (citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        cutOffDistance = osg::maximum(cutOffDistance, (*citr)->_maxVisibleDistance);
    }

    osg::LOD *myLOD = new osg::LOD;

    float farDistance = _dataSet->getMaximumVisibleDistanceOfTopLevel();
    if (tileNodes.size() == 1)
    {
        myLOD->addChild(tileNodes.front());
    }
    else if (tileNodes.size() > 1)
    {
        osg::Group *group = new osg::Group;
        for (NodeList::iterator itr = tileNodes.begin();
             itr != tileNodes.end();
             ++itr)
        {
            group->addChild(*itr);
        }
        myLOD->addChild(group);
    }

    if (childNodes.size() == 1)
    {
        myLOD->addChild(childNodes.front());
    }
    else if (childNodes.size() > 1)
    {
        osg::Group *group = new osg::Group;
        for (NodeList::iterator itr = childNodes.begin();
             itr != childNodes.end();
             ++itr)
        {
            group->addChild(*itr);
        }
        myLOD->addChild(group);
    }

    // find cluster culling callbacks on drawables and move them to the myLOD level.
    {
        CollectClusterCullingCallbacks collect;
        myLOD->accept(collect);

        if (!collect._callbackList.empty())
        {
            if (collect._callbackList.size() == 1)
            {
                CollectClusterCullingCallbacks::Triple &triple = collect._callbackList.front();

                osg::Matrixd matrix = osg::computeLocalToWorld(triple._nodePath);

                triple._callback->transform(matrix);

                log(osg::INFO, "cluster culling matrix %f\t%f\t%f\t%f", matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3));
                log(osg::INFO, "                       %f\t%f\t%f\t%f", matrix(1, 0), matrix(1, 1), matrix(1, 2), matrix(1, 3));
                log(osg::INFO, "                       %f\t%f\t%f\t%f", matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3));
                log(osg::INFO, "                       %f\t%f\t%f\t%f", matrix(3, 0), matrix(3, 1), matrix(3, 2), matrix(3, 3));

                // moving cluster culling callback myLOD node.
                myLOD->setCullCallback(triple._callback);

                // remove it from the drawable.

                osg::Drawable *drawable = dynamic_cast<osg::Drawable *>(triple._object);
                if (drawable)
                    drawable->setCullCallback(0);

                osg::Node *node = dynamic_cast<osg::Node *>(triple._object);
                if (node)
                    node->setCullCallback(0);
            }
        }
    }

    cutOffDistance = osg::maximum(cutOffDistance, (float)(myLOD->getBound().radius() * _dataSet->getRadiusToMaxVisibleDistanceRatio()));

    myLOD->setRange(0, cutOffDistance, farDistance);
    myLOD->setRange(1, 0, cutOffDistance);

    if (myLOD->getNumChildren() > 0)
        myLOD->setCenter(myLOD->getBound().center());

    return myLOD;
#else
    // must be either a LOD or a PagedLOD

    typedef std::vector<osg::Node *> NodeList;
    typedef std::map<float, NodeList> RangeNodeListMap;
    RangeNodeListMap rangeNodeListMap;

    // insert local tiles into range map
    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        osg::Node *node = (*titr)->createScene();

        if (node)
        {
            double maxVisibleDistance = osg::maximum(_maxVisibleDistance, node->getBound().radius() * _dataSet->getRadiusToMaxVisibleDistanceRatio());
            rangeNodeListMap[maxVisibleDistance].push_back(node);
        }
    }

    // handle chilren
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        osg::Node *node = (*citr)->createScene();
        if (node)
        {
            double maxVisibleDistance = osg::maximum((*citr)->_maxVisibleDistance, node->getBound().radius() * _dataSet->getRadiusToMaxVisibleDistanceRatio());
            rangeNodeListMap[maxVisibleDistance].push_back(node);
        }
    }

    osg::LOD *lod = new osg::LOD;

    float farDistance = _dataSet->getMaximumVisibleDistanceOfTopLevel();

    unsigned int childNum = 0;
    for (RangeNodeListMap::reverse_iterator rnitr = rangeNodeListMap.rbegin();
         rnitr != rangeNodeListMap.rend();
         ++rnitr)
    {
        float maxVisibleDistance = rnitr->first;
        NodeList &nodeList = rnitr->second;

        if (childNum == 0)
        {
            // by deafult make the first child have a very visible distance so its always seen
            maxVisibleDistance = farDistance;
        }
        else
        {
            // set the previous child's minimum visible distance range
            lod->setRange(childNum - 1, maxVisibleDistance, lod->getMaxRange(childNum - 1));
        }

        osg::Node *child = 0;
        if (nodeList.size() == 1)
        {
            child = nodeList.front();
        }
        else if (nodeList.size() > 1)
        {
            osg::Group *group = new osg::Group;
            for (NodeList::iterator itr = nodeList.begin();
                 itr != nodeList.end();
                 ++itr)
            {
                group->addChild(*itr);
            }
            child = group;
        }

        if (child)
        {

            lod->addChild(child, 0, maxVisibleDistance);

            ++childNum;
        }
    }
    return lod;
#endif
}

bool CompositeDestination::areSubTilesComplete()
{
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        for (TileList::iterator itr = (*citr)->_tiles.begin();
             itr != (*citr)->_tiles.end();
             ++itr)
        {
            if (!(*itr)->getTileComplete())
            {
                return false;
            }
        }
    }
    return true;
}

std::string CompositeDestination::getSubTileName()
{
    std::string filename = _name + "_subtile" + _dataSet->getDestinationTileExtension();
    log(osg::INFO, "CompositeDestination::getSubTileName()=%s", filename.c_str());
    return filename;
}

std::string CompositeDestination::getExternalSubTileName()
{
    std::string filename;

    bool externalFile = _dataSet->getOutputTaskDirectories();
    bool isLeaf = (_level == (_dataSet->getMaximumNumOfLevels() - 1));
    bool isRoot = (_level == 0);

    if (externalFile && isRoot)
    {
        filename = _dataSet->getTaskName(_level, _tileX, _tileY) + std::string("/") + _name + "_subtile" + _dataSet->getDestinationTileExtension();
        log(osg::INFO, "CompositeDestination::getExternalSubTileName()=%s ROOT!!", filename.c_str());
    }
    else if (externalFile && isLeaf)
    {
        filename = std::string("../") + _dataSet->getTaskName(_level, _tileX, _tileY) + std::string("/") + _name + "_subtile" + _dataSet->getDestinationTileExtension();
        log(osg::INFO, "CompositeDestination::getExternalSubTileName()=%s LEAF!!", filename.c_str());
    }
    else
    {
        filename = _name + "_subtile" + _dataSet->getDestinationTileExtension();
        log(osg::INFO, "CompositeDestination::getExternalSubTileName()=%s", filename.c_str());
    }

    return filename;
}

std::string CompositeDestination::getTileFileName()
{
    if (_parent)
    {
        return getTilePath() + _parent->getSubTileName();
    }
    else
    {
        return getTilePath() + _dataSet->getDestinationTileBaseName() + _dataSet->getDestinationTileExtension();
    }
}

std::string CompositeDestination::getTilePath()
{
    if (_parent)
    {
        return _dataSet->getTaskOutputDirectory();
    }
    else
    {
        if (_level == 0)
        {
            return _dataSet->getDirectory();
        }
        else
        {
            return _dataSet->getTaskOutputDirectory();
        }
    }
}

std::string CompositeDestination::getRelativePathForExternalSet(const std::string &setname)
{
    // first compute path to root
    std::string tilePath = getTilePath();
    std::string root = _dataSet->getDirectory();
    std::string::size_type pos = tilePath.find(root);
    if (pos == std::string::npos)
    {
        _dataSet->log(osg::NOTICE, "Error: CompositeDestination::getRelativePathForExternalSet() error in paths, root = %s, getTilePath()=%s", root.c_str(), tilePath.c_str());
        return setname + std::string("/");
    }

    if (pos != 0)
    {
        _dataSet->log(osg::NOTICE, "Error: CompositeDestination::getRelativePathForExternalSet() error in paths, root = %s, getTilePath()=%s", root.c_str(), tilePath.c_str());
        return setname + std::string("/");
    }

    _dataSet->log(osg::NOTICE, "CompositeDestination::getRelativePathForExternalSet() root = %s", root.c_str());
    _dataSet->log(osg::NOTICE, "CompositeDestination::getRelativePathForExternalSet() tilePath = %s", tilePath.c_str());

    std::string relativePath;
    for (unsigned int i = root.size(); i < tilePath.size(); ++i)
    {
        if (tilePath[i] == '/' || tilePath[i] == '\\')
            relativePath += "../";
    }

    relativePath += setname;
    relativePath += std::string("/");
    relativePath += tilePath.substr(root.size(), std::string::npos);

    return relativePath;
}

osg::Node *CompositeDestination::createPagedLODScene()
{
    if (_children.empty() && _tiles.empty())
        return 0;

    if (_children.empty() && _tiles.size() == 1)
        return _tiles.front()->createScene();

    if (_tiles.empty() && _children.size() == 1)
        return _children.front()->createPagedLODScene();

    if (_type == GROUP)
    {
        osg::Group *group = new osg::Group;
        for (TileList::iterator titr = _tiles.begin();
             titr != _tiles.end();
             ++titr)
        {
            osg::Node *node = (*titr)->createScene();
            if (node)
                group->addChild(node);
        }

        // handle chilren
        for (ChildList::iterator citr = _children.begin();
             citr != _children.end();
             ++citr)
        {
            osg::Node *node = (*citr)->createScene();
            if (node)
                group->addChild(node);
        }
        return group;
    }

    // must be either a LOD or a PagedLOD

    typedef std::vector<osg::Node *> NodeList;

    // collect all the local tiles
    NodeList tileNodes;
    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        osg::Node *node = (*titr)->createScene();
        if (node)
            tileNodes.push_back(node);
    }

    float cutOffDistance = -FLT_MAX;
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        cutOffDistance = osg::maximum(cutOffDistance, (*citr)->_maxVisibleDistance);
    }

    osg::PagedLOD *pagedLOD = new osg::PagedLOD;

    float farDistance = _dataSet->getMaximumVisibleDistanceOfTopLevel();
    if (tileNodes.size() == 1)
    {
        pagedLOD->addChild(tileNodes.front());
    }
    else if (tileNodes.size() > 1)
    {
        osg::Group *group = new osg::Group;
        for (NodeList::iterator itr = tileNodes.begin();
             itr != tileNodes.end();
             ++itr)
        {
            group->addChild(*itr);
        }
        pagedLOD->addChild(group);
    }

    // find cluster culling callbacks on drawables and move them to the PagedLOD level.
    {
        CollectClusterCullingCallbacks collect;
        pagedLOD->accept(collect);

        if (!collect._callbackList.empty())
        {
            if (collect._callbackList.size() == 1)
            {
                CollectClusterCullingCallbacks::Triple &triple = collect._callbackList.front();

                osg::Matrixd matrix = osg::computeLocalToWorld(triple._nodePath);

                triple._callback->transform(matrix);

                log(osg::INFO, "cluster culling matrix %f\t%f\t%f\t%f", matrix(0, 0), matrix(0, 1), matrix(0, 2), matrix(0, 3));
                log(osg::INFO, "                       %f\t%f\t%f\t%f", matrix(1, 0), matrix(1, 1), matrix(1, 2), matrix(1, 3));
                log(osg::INFO, "                       %f\t%f\t%f\t%f", matrix(2, 0), matrix(2, 1), matrix(2, 2), matrix(2, 3));
                log(osg::INFO, "                       %f\t%f\t%f\t%f", matrix(3, 0), matrix(3, 1), matrix(3, 2), matrix(3, 3));

                // moving cluster culling callback pagedLOD node.
                pagedLOD->setCullCallback(triple._callback);

                osg::Drawable *drawable = dynamic_cast<osg::Drawable *>(triple._object);
                if (drawable)
                    drawable->setCullCallback(0);

                osg::Node *node = dynamic_cast<osg::Node *>(triple._object);
                if (node)
                    node->setCullCallback(0);
            }
        }
    }

    cutOffDistance = osg::maximum(cutOffDistance, (float)(pagedLOD->getBound().radius() * _dataSet->getRadiusToMaxVisibleDistanceRatio()));

    pagedLOD->setRange(0, cutOffDistance, farDistance);

    pagedLOD->setFileName(1, getExternalSubTileName());
    pagedLOD->setRange(1, 0, cutOffDistance);

    if (pagedLOD->getNumChildren() > 0)
        pagedLOD->setCenter(pagedLOD->getBound().center());

    return pagedLOD;
}

osg::Node *CompositeDestination::createSubTileScene()
{
    if (_type == GROUP || _children.empty() || _tiles.empty())
        return 0;

    // handle chilren
    typedef std::vector<osg::Node *> NodeList;
    NodeList nodeList;
    for (ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        osg::Node *node = (*citr)->createPagedLODScene();
        if (node)
            nodeList.push_back(node);
    }

    if (nodeList.size() == 1)
    {
        return nodeList.front();
    }
    else if (nodeList.size() > 1)
    {
        osg::Group *group = new osg::Group;
        for (NodeList::iterator itr = nodeList.begin();
             itr != nodeList.end();
             ++itr)
        {
            group->addChild(*itr);
        }
        return group;
    }
    else
    {
        return 0;
    }
}

void CompositeDestination::unrefSubTileData()
{
    for (CompositeDestination::ChildList::iterator citr = _children.begin();
         citr != _children.end();
         ++citr)
    {
        (*citr)->unrefLocalData();
    }
}

void CompositeDestination::unrefLocalData()
{
    for (CompositeDestination::TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        DestinationTile *tile = titr->get();
        log(osg::INFO, "   unref tile level=%d X=%d Y=%d", tile->_level, tile->_tileX, tile->_tileY);
        tile->unrefData();
    }
}

DestinationTile::Sources CompositeDestination::getAllContributingSources()
{
    typedef std::set<Source *> SourceSet;
    SourceSet sourceSet;

    for (TileList::iterator titr = _tiles.begin();
         titr != _tiles.end();
         ++titr)
    {
        DestinationTile *dt = titr->get();
        for (DestinationTile::Sources::iterator itr = dt->_sources.begin();
             itr != dt->_sources.end();
             ++itr)
        {
            sourceSet.insert(itr->get());
        }
    }

    return DestinationTile::Sources(sourceSet.begin(), sourceSet.end());
}
