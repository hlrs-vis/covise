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


#include <osg/Texture2D>
#include <osg/ComputeBoundsVisitor>
#include <osg/io_utils>

#include <osg/GLU>

#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileNameUtils>

#include <osgFX/MultiTextureControl>

#include <osgViewer/GraphicsWindow>
#include <osgViewer/Viewer>
#include <osgViewer/Version>

#include <vpb/DataSet>
#include <vpb/DatabaseBuilder>
#include <vpb/TaskManager>
#include <vpb/System>
#include <vpb/FileUtils>
#include <vpb/FilePathManager>

#include <vpb/ShapeFilePlacer>

// GDAL includes
#include <gdal_priv.h>
#include <ogr_spatialref.h>

// standard library includes
#include <sstream>
#include <iostream>
#include <algorithm>


using namespace vpb;


DataSet::DataSet()
{
    init();
}

void DataSet::init()
{
    // make sure GDAL etc. are initialized
    System::instance();

    _C1 = 0;
    _R1 = 0;

    _numTextureLevels = 1;

    _modelPlacer = new ObjectPlacer;
    _shapeFilePlacer = new ShapeFilePlacer;

    _newDestinationGraph = false;

}

void DataSet::addSource(Source* source, unsigned int revisionNumber)
{
    if (!source) return;

    if (!_sourceGraph.valid()) _sourceGraph = new CompositeSource;

    source->setRevisionNumber(revisionNumber);

    _sourceGraph->_sourceList.push_back(source);
}
#if 0
void DataSet::addSource(CompositeSource* composite)
{
    if (!composite) return;

    if (!_sourceGraph.valid()) _sourceGraph = new CompositeSource;

    _sourceGraph->_children.push_back(composite);
}
#endif
void DataSet::loadSources()
{
    assignIntermediateCoordinateSystem();

    FileCache* fileCache = System::instance()->getFileCache();

    for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
    {
        Source* source = itr->get();

        if (source)
        {
            source->loadSourceData();

            if (fileCache && source->needReproject(_intermediateCoordinateSystem.get()))
            {
                if (source->isRaster())
                {
                    Source* newSource = source->doRasterReprojectionUsingFileCache(_intermediateCoordinateSystem.get());

                    if (newSource)
                    {
                        *itr = newSource;
                    }
                }
            }
        }
    }
}

bool DataSet::mapLatLongsToXYZ() const
{
    bool result = getConvertFromGeographicToGeocentric() && getEllipsoidModel();
    return result;
}

bool DataSet::computeCoverage(const GeospatialExtents& extents, int level, int& minX, int& minY, int& maxX, int& maxY)
{
    if (!_destinationExtents.intersects(extents)) return false;

    if (level==0)
    {
        minX = 0;
        maxX = 1;
        minY = 0;
        maxY = 1;
        return true;
    }

    double destination_xRange = _destinationExtents.xMax()-_destinationExtents.xMin();
    double destination_yRange = _destinationExtents.yMax()-_destinationExtents.yMin();

    int Ck = int(pow(2.0, double(level-1))) * _C1;
    int Rk = int(pow(2.0, double(level-1))) * _R1;

    int i_min = int( floor( ((extents.xMin() - _destinationExtents.xMin()) / destination_xRange) * double(Ck) ) );
    int j_min = int( floor( ((extents.yMin() - _destinationExtents.yMin()) / destination_yRange) * double(Rk) ) );

    // note i_max and j_max are one beyond the extents required so that the below for loop can use <
    // and the clamping to the 0..Ck-1 and 0..Rk-1 extents will work fine.
    int i_max = int( ceil( ((extents.xMax() - _destinationExtents.xMin()) / destination_xRange) * double(Ck) ) );
    int j_max = int( ceil( ((extents.yMax() - _destinationExtents.yMin()) / destination_yRange) * double(Rk) ) );

    // clamp j range to 0 to Ck range
    if (i_min<0) i_min = 0;
    if (i_max<0) i_max = 0;
    if (i_min>Ck) i_min = Ck;
    if (i_max>Ck) i_max = Ck;

    // clamp j range to 0 to Rk range
    if (j_min<0) j_min = 0;
    if (j_max<0) j_max = 0;
    if (j_min>Rk) j_min = Rk;
    if (j_max>Rk) j_max = Rk;

    minX = i_min;
    maxX = i_max;
    minY = j_min;
    maxY = j_max;

    return (minX<maxX) && (minY<maxY);
}

bool DataSet::computeOptimumLevel(Source* source, int maxLevel, int& level)
{
    if (source->getType()!=Source::IMAGE && source->getType()!=Source::HEIGHT_FIELD) return false;

    if (maxLevel < static_cast<int>(source->getMinLevel())) return false;

    SourceData* sd = source->getSourceData();
    if (!sd) return false;


    const SpatialProperties& sp = sd->computeSpatialProperties(_intermediateCoordinateSystem.get());

    double destination_xRange = _destinationExtents.xMax()-_destinationExtents.xMin();
    double destination_yRange = _destinationExtents.yMax()-_destinationExtents.yMin();

    double source_xRange = sp._extents.xMax()-sp._extents.xMin();
    double source_yRange = sp._extents.yMax()-sp._extents.yMin();

    float sourceResolutionX = (source_xRange)/(float)sp._numValuesX;
    float sourceResolutionY = (source_yRange)/(float)sp._numValuesY;

    // log(osg::NOTICE,"Source %s resX %f resY %f",source->getFileName().c_str(), sourceResolutionX, sourceResolutionX);

    double tileSize = source->getType()==Source::IMAGE ? getLayerMaximumTileImageSize(source->getLayer())-2 : _maximumTileTerrainSize-1;

    int k_cols = int( ceil( 1.0 + ::log( destination_xRange / (_C1 * sourceResolutionX * tileSize ) ) / ::log(2.0) ) );
    int k_rows = int( ceil( 1.0 + ::log( destination_yRange / (_R1 * sourceResolutionY * tileSize ) ) / ::log(2.0) ) );
    level = std::max(k_cols, k_rows);
    level = std::min(level, int(source->getMaxLevel()));
    level = std::min(level, maxLevel);
    level = std::max(level, 0);
    return true;
}

int DataSet::computeMaximumLevel(int maxNumLevels)
{
    int maxLevel = 0;
    for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
    {
        Source* source = (*itr).get();


        SourceData* sd = (*itr)->getSourceData();
        if (!sd)
        {
            log(osg::NOTICE,"Skipping source %s as no data loaded from it.",source->getFileName().c_str());
            continue;
        }

        if (source->getType()!=Source::IMAGE && source->getType()!=Source::HEIGHT_FIELD)
        {
            // place models and shapefiles into a separate temporary source list and then process these after
            // the main handling of terrainTile/imagery sources.
            continue;

        }

        int k = 0;
        if (!computeOptimumLevel(source, maxNumLevels-1, k)) continue;

        if (k>maxLevel)
        {
            maxLevel = k;
        }
    }

    return maxLevel;
}

bool DataSet::computeOptimumTileSystemDimensions(int& C1, int& R1)
{
    C1 = 1;
    R1 = 1;

    double destination_xRange = _destinationExtents.xMax()-_destinationExtents.xMin();
    double destination_yRange = _destinationExtents.yMax()-_destinationExtents.yMin();
    double AR = destination_xRange / destination_yRange;

    bool swapAxis = AR<1.0;
    if (swapAxis) AR = 1.0/AR;

    double lower_AR = floor(AR);
    double upper_AR = ceil(AR);

    if (AR<sqrt(lower_AR*upper_AR))
    {
        C1 = (int)(lower_AR);
        R1 = 1;
    }
    else
    {
        C1 = (int)(upper_AR);
        R1 = 1;
    }

    if (swapAxis)
    {
        std::swap(C1,R1);
    }

    _C1 = C1;
    _R1 = R1;

    log(osg::NOTICE,"AR=%f C1=%i R1=%i",AR,C1,R1);

    return true;
}

CompositeDestination* DataSet::createDestinationTile(int currentLevel, int currentX, int currentY)
{
    CompositeDestination* parent = 0;
    GeospatialExtents extents;

    if (currentLevel==0)
    {
        extents = _destinationExtents;
    }
    else
    {
        // compute the extents
        double destination_xRange = _destinationExtents.xMax()-_destinationExtents.xMin();
        double destination_yRange = _destinationExtents.yMax()-_destinationExtents.yMin();

        int Ck = int(pow(2.0, double(currentLevel-1))) * _C1;
        int Rk = int(pow(2.0, double(currentLevel-1))) * _R1;

        extents.xMin() = _destinationExtents.xMin() + (double(currentX)/double(Ck)) * destination_xRange;
        extents.xMax() = _destinationExtents.xMin() + (double(currentX+1)/double(Ck)) * destination_xRange;

        extents.yMin() = _destinationExtents.yMin() + (double(currentY)/double(Rk)) * destination_yRange;
        extents.yMax() = _destinationExtents.yMin() + (double(currentY+1)/double(Rk)) * destination_yRange;

        // compute the parent
        if (currentLevel == 1)
        {
            parent = _destinationGraph.get();
        }
        else
        {
            parent = getComposite(currentLevel-1,currentX/2,currentY/2);

            if (!parent)
            {
                log(osg::NOTICE,"Warning: getComposite(%i,%i,%i) return 0",currentLevel-1,currentX/2,currentY/2);
            }
        }

    }

    CompositeDestination* destinationGraph = new CompositeDestination(_intermediateCoordinateSystem.get(),extents);

    if (currentLevel==0) _destinationGraph = destinationGraph;

    if (mapLatLongsToXYZ())
    {
        // we need to project the extents into world coords to get the appropriate size to use for control max visible distance
        float max_range = osg::maximum(extents.xMax()-extents.xMin(),extents.yMax()-extents.yMin());
        float projected_radius =  osg::DegreesToRadians(max_range) * getEllipsoidModel()->getRadiusEquator();
        float center_offset = (max_range/360.0f) * getEllipsoidModel()->getRadiusEquator();
        destinationGraph->_maxVisibleDistance = projected_radius * getRadiusToMaxVisibleDistanceRatio() + center_offset;
    }
    else
    {
        destinationGraph->_maxVisibleDistance = extents.radius()*getRadiusToMaxVisibleDistanceRatio();
    }

    // first create the topmost tile

    // create the name
    std::ostringstream os;
    os << _tileBasename << "_L"<<currentLevel<<"_X"<<currentX<<"_Y"<<currentY;

    destinationGraph->_parent = parent;
    destinationGraph->_name = os.str();
    destinationGraph->_level = currentLevel;
    destinationGraph->_tileX = currentX;
    destinationGraph->_tileY = currentY;
    destinationGraph->_dataSet = this;


    DestinationTile* tile = new DestinationTile;
    tile->_name = destinationGraph->_name;
    tile->_level = currentLevel;
    tile->_tileX = currentX;
    tile->_tileY = currentY;
    tile->_dataSet = this;
    tile->_cs = destinationGraph->_cs;
    tile->_extents = extents;
    tile->_parent = destinationGraph;

    // set to NONE as the tile is a mix of RASTER and VECTOR
    // that way the default of RASTER for image and VECTOR for height is maintained
    tile->_dataType = SpatialProperties::NONE;

    tile->setMaximumTerrainSize(_maximumTileTerrainSize,_maximumTileTerrainSize);

    destinationGraph->_tiles.push_back(tile);

    if (parent)
    {
        parent->_type = LOD;
        parent->addChild(destinationGraph);
    }


    insertTileToQuadMap(destinationGraph);

    return destinationGraph;
}

void DataSet::createNewDestinationGraph(osg::CoordinateSystemNode* cs,
                               const GeospatialExtents& extents,
                               unsigned int maxImageSize,
                               unsigned int maxTerrainSize,
                               unsigned int maxNumLevels)
{
    log(osg::NOTICE,"createNewDestinationGraph");

    _newDestinationGraph = true;

    int highestLevelFound = 0;

    // first populate the destination graph from imagery and DEM sources extents/resolution
    for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
    {
        Source* source = (*itr).get();

        if (source->getMinLevel()>maxNumLevels)
        {
            log(osg::NOTICE,"Skipping source %s as its min level excees destination max level.",source->getFileName().c_str());
            continue;
        }

        if (getGenerateSubtile() && source->getMaxLevel()<getSubtileLevel())
        {
            log(osg::NOTICE,"Skipping source %s as its max level is lower than the subtile level.",source->getFileName().c_str());
            continue;
        }

        SourceData* sd = (*itr)->getSourceData();
        if (!sd)
        {
            log(osg::NOTICE,"Skipping source %s as no data loaded from it.",source->getFileName().c_str());
            continue;
        }

        const SpatialProperties& sp = sd->computeSpatialProperties(cs);

        if (!sp._extents.intersects(extents))
        {
            // skip this source since it doesn't overlap this tile.
            log(osg::NOTICE,"Skipping source %s as its extents don't overlap destination extents.",source->getFileName().c_str());
            continue;
        }


        if (source->getType()!=Source::IMAGE && source->getType()!=Source::HEIGHT_FIELD)
        {
            continue;
        }

        int k = 0;
        if (!computeOptimumLevel(source, maxNumLevels-1, k)) continue;


        if (k>highestLevelFound) highestLevelFound = k;

        int startLevel = 0; // getGenerateSubtile() ? getSubtileLevel() : 0;

        for(int l=startLevel; l<=k; l++)
        {
            int i_min, i_max, j_min, j_max;
            if (computeCoverage(sp._extents, l, i_min, j_min, i_max, j_max))
            {
                // log(osg::NOTICE,"     level=%i i_min=%i i_max=%i j_min=%i j_max=%i",l, i_min, i_max, j_min, j_max);

                if (getGenerateSubtile())
                {
                    int i_lower, i_upper, j_lower, j_upper;

                    if (l<static_cast<int>(getSubtileLevel()))
                    {
                        // divide by 2 to the power of ((getSubtileLevel()-l);
                        int delta = getSubtileLevel()-l;
                        i_lower = getSubtileX() >> delta;
                        j_lower = getSubtileY() >> delta;
                        i_upper = i_lower + 1;
                        j_upper = j_lower + 1;
                    }
                    else
                    {
                        // multiply 2 to the power of ((l-getSubtileLevel());
                        int f = 1 << (l-getSubtileLevel());
                        i_lower = getSubtileX() * f;
                        j_lower = getSubtileY() * f;
                        i_upper = i_lower + f;
                        j_upper = j_lower + f;
                    }

                    if (i_min<i_lower) i_min = i_lower;
                    if (i_max>i_upper) i_max = i_upper;
                    if (j_min<j_lower) j_min = j_lower;
                    if (j_max>j_upper) j_max = j_upper;
                }

                for(int j=j_min; j<j_max;++j)
                {
                    for(int i=i_min; i<i_max;++i)
                    {
                        CompositeDestination* cd = getComposite(l,i,j);
                        if (!cd)
                        {
                            cd = createDestinationTile(l,i,j);
                        }
                    }
                }
            }
        }
    }

    // now extend the sources upwards where required.
    for(QuadMap::iterator qitr = _quadMap.begin();
        qitr != _quadMap.end();
        ++qitr)
    {
        QuadMap::iterator temp_itr = qitr;
        ++temp_itr;
        if (temp_itr==_quadMap.end()) continue;

        int l = qitr->first;

        Level& level = qitr->second;
        for(Level::iterator litr = level.begin();
            litr != level.end();
            ++litr)
        {
            Row& row = litr->second;
            for(Row::iterator ritr = row.begin();
                ritr != row.end();
                ++ritr)
            {
                CompositeDestination* cd = ritr->second;

                int numChildren = cd->_children.size();
                int numChildrenExpected = (l==0) ? (_C1*_R1) : 4;
                if (numChildren!=0 && numChildren!=numChildrenExpected)
                {
#if 0
                    log(osg::NOTICE,"  tile (%i,%i,%i) numTiles=%i numChildren=%i",
                        cd->_level, cd->_tileX, cd->_tileY, cd->_tiles.size(), cd->_children.size());
#endif
                    int i_min = (l==0) ? 0   : (cd->_tileX * 2);
                    int j_min = (l==0) ? 0   : (cd->_tileY * 2);
                    int i_max = (l==0) ? _C1 : i_min + 2;
                    int j_max = (l==0) ? _R1 : j_min + 2;
                    int new_l = l+1;

                    if (getGenerateSubtile())
                    {
                        int i_lower, i_upper, j_lower, j_upper;


                        if (l<static_cast<int>(getSubtileLevel()))
                        {
                            // divide by 2 to the power of ((getSubtileLevel()-new_l);
                            int delta = getSubtileLevel()-new_l;
                            i_lower = getSubtileX() >> delta;
                            j_lower = getSubtileY() >> delta;
                            i_upper = i_lower + 1;
                            j_upper = j_lower + 1;
                        }
                        else
                        {
                            // multiply 2 to the power of ((new_l-getSubtileLevel());
                            int f = 1 << (new_l-getSubtileLevel());
                            i_lower = getSubtileX() * f;
                            j_lower = getSubtileY() * f;
                            i_upper = i_lower + f;
                            j_upper = j_lower + f;
                        }

                        if (i_min<i_lower) i_min = i_lower;
                        if (i_max>i_upper) i_max = i_upper;
                        if (j_min<j_lower) j_min = j_lower;
                        if (j_max>j_upper) j_max = j_upper;
                    }

                    for(int j=j_min; j<j_max;++j)
                    {
                        for(int i=i_min; i<i_max;++i)
                        {
                            CompositeDestination* cd = getComposite(new_l,i,j);
                            if (!cd)
                            {
                                cd = createDestinationTile(new_l,i,j);
                            }
                        }
                    }

                }
            }
        }
    }

    // now insert the sources into the destination graph
    for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
    {
        Source* source = (*itr).get();

        if (source->getMinLevel()>maxNumLevels)
        {
            log(osg::NOTICE,"Skipping source %s as its min level excees destination max level.",source->getFileName().c_str());
            continue;
        }

        if (getGenerateSubtile() && source->getMaxLevel()<getSubtileLevel())
        {
            log(osg::NOTICE,"Skipping source %s as its max level is lower than the subtile level.",source->getFileName().c_str());
            continue;
        }

        SourceData* sd = (*itr)->getSourceData();
        if (!sd)
        {
            log(osg::NOTICE,"Skipping source %s as no data loaded from it.",source->getFileName().c_str());
            continue;
        }

        const SpatialProperties& sp = sd->computeSpatialProperties(cs);

        if (!sp._extents.intersects(extents))
        {
            // skip this source since it doesn't overlap this tile.
            log(osg::NOTICE,"Skipping source %s as its extents don't overlap destination extents.",source->getFileName().c_str());
            continue;
        }

        int k = 0;

        if (source->getType()==Source::IMAGE || source->getType()==Source::HEIGHT_FIELD)
        {
            if (!computeOptimumLevel(source, maxNumLevels-1, k)) continue;
        }
        else
        {
            k = highestLevelFound;
        }


        // log(osg::NOTICE,"     opt level = %i",k);

        int startLevel = 0; // getGenerateSubtile() ? getSubtileLevel() : 0;

        for(int l=startLevel; l<=k; l++)
        {
            int i_min, i_max, j_min, j_max;
            if (computeCoverage(sp._extents, l, i_min, j_min, i_max, j_max))
            {
                // log(osg::NOTICE,"     level=%i i_min=%i i_max=%i j_min=%i j_max=%i",l, i_min, i_max, j_min, j_max);

                if (getGenerateSubtile())
                {
                    int i_lower, i_upper, j_lower, j_upper;

                    if (l<static_cast<int>(getSubtileLevel()))
                    {
                        // divide by 2 to the power of ((getSubtileLevel()-l);
                        int delta = getSubtileLevel()-l;
                        i_lower = getSubtileX() >> delta;
                        j_lower = getSubtileY() >> delta;
                        i_upper = i_lower + 1;
                        j_upper = j_lower + 1;
                    }
                    else
                    {
                        // multiply 2 to the power of ((l-getSubtileLevel());
                        int f = 1 << (l-getSubtileLevel());
                        i_lower = getSubtileX() * f;
                        j_lower = getSubtileY() * f;
                        i_upper = i_lower + f;
                        j_upper = j_lower + f;
                    }

                    if (i_min<i_lower) i_min = i_lower;
                    if (i_max>i_upper) i_max = i_upper;
                    if (j_min<j_lower) j_min = j_lower;
                    if (j_max>j_upper) j_max = j_upper;
                }

                for(int j=j_min; j<j_max;++j)
                {
                    for(int i=i_min; i<i_max;++i)
                    {
                        CompositeDestination* cd = getComposite(l,i,j);
                        if (!cd) continue;


                        if (l==k)
                        {
                            cd->addSource(source);
                        }
                        else
                        {
                            for(CompositeDestination::TileList::iterator titr = cd->_tiles.begin();
                                titr != cd->_tiles.end();
                                ++titr)
                            {
                                DestinationTile* tile = titr->get();
                                tile->_sources.push_back(source);
                            }
                        }
                    }
                }
            }
        }
    }

    osg::Timer_t before_computeMax = osg::Timer::instance()->tick();

    if (_destinationGraph.valid()) _destinationGraph->computeMaximumSourceResolution();

    osg::Timer_t after_computeMax = osg::Timer::instance()->tick();

    log(osg::NOTICE,"Time for _destinationGraph->computeMaximumSourceResolution() = %f", osg::Timer::instance()->delta_s(before_computeMax, after_computeMax));
}

CompositeDestination* DataSet::createDestinationGraph(CompositeDestination* parent,
                                                      osg::CoordinateSystemNode* cs,
                                                      const GeospatialExtents& extents,
                                                      unsigned int maxImageSize,
                                                      unsigned int maxTerrainSize,
                                                      unsigned int currentLevel,
                                                      unsigned int currentX,
                                                      unsigned int currentY,
                                                      unsigned int maxNumLevels)
{

    if (getGenerateSubtile() && (currentLevel == getSubtileLevel()))
    {
        if ((currentX != getSubtileX()) || (currentY != getSubtileY())) return 0;
    }


    CompositeDestination* destinationGraph = new CompositeDestination(cs,extents);

    if (mapLatLongsToXYZ())
    {
        // we need to project the extents into world coords to get the appropriate size to use for control max visible distance
        float max_range = osg::maximum(extents.xMax()-extents.xMin(),extents.yMax()-extents.yMin());
        float projected_radius =  osg::DegreesToRadians(max_range) * getEllipsoidModel()->getRadiusEquator();
        float center_offset = (max_range/360.0f) * getEllipsoidModel()->getRadiusEquator();
        destinationGraph->_maxVisibleDistance = projected_radius * getRadiusToMaxVisibleDistanceRatio() + center_offset;
    }
    else
    {
        destinationGraph->_maxVisibleDistance = extents.radius()*getRadiusToMaxVisibleDistanceRatio();
    }

    // first create the topmost tile

    // create the name
    std::ostringstream os;
    os << _tileBasename << "_L"<<currentLevel<<"_X"<<currentX<<"_Y"<<currentY;

    destinationGraph->_parent = parent;
    destinationGraph->_name = os.str();
    destinationGraph->_level = currentLevel;
    destinationGraph->_tileX = currentX;
    destinationGraph->_tileY = currentY;
    destinationGraph->_dataSet = this;


    DestinationTile* tile = new DestinationTile;
    tile->_name = destinationGraph->_name;
    tile->_level = currentLevel;
    tile->_tileX = currentX;
    tile->_tileY = currentY;
    tile->_dataSet = this;
    tile->_cs = cs;
    tile->_extents = extents;
    tile->_parent = destinationGraph;

    // set to NONE as the tile is a mix of RASTER and VECTOR
    // that way the default of RASTER for image and VECTOR for height is maintained
    tile->_dataType = SpatialProperties::NONE;

    tile->setMaximumTerrainSize(maxTerrainSize,maxTerrainSize);
    tile->computeMaximumSourceResolution(_sourceGraph.get());

    insertTileToQuadMap(destinationGraph);

    if (currentLevel>=maxNumLevels-1 || currentLevel>=tile->_maxSourceLevel)
    {
        // bottom level can't divide any further.
        destinationGraph->_tiles.push_back(tile);
    }
    else
    {
        destinationGraph->_type = LOD;
        destinationGraph->_tiles.push_back(tile);

        bool needToDivideX = false;
        bool needToDivideY = false;

        // note, resolutionSensitivityScale should probably be customizable.. will consider this option for later inclusion.
        double resolutionSensitivityScale = 0.9;

        tile->requiresDivision(resolutionSensitivityScale, needToDivideX, needToDivideY);

        float xCenter = (extents.xMin()+extents.xMax())*0.5f;
        float yCenter = (extents.yMin()+extents.yMax())*0.5f;

        unsigned int newLevel = currentLevel+1;
        unsigned int newX = currentX*2;
        unsigned int newY = currentY*2;

        if (needToDivideX && needToDivideY)
        {
            float aspectRatio = (extents.yMax()- extents.yMin())/(extents.xMax()- extents.xMin());

            if (aspectRatio>1.414) needToDivideX = false;
            else if (aspectRatio<.707) needToDivideY = false;
        }

        if (needToDivideX && needToDivideY)
        {
            log(osg::INFO,"Need to Divide X + Y for level %u",currentLevel);
            // create four tiles.
            GeospatialExtents bottom_left(extents.xMin(),extents.yMin(),xCenter,yCenter, extents._isGeographic);
            GeospatialExtents bottom_right(xCenter,extents.yMin(),extents.xMax(),yCenter, extents._isGeographic);
            GeospatialExtents top_left(extents.xMin(),yCenter,xCenter,extents.yMax(), extents._isGeographic);
            GeospatialExtents top_right(xCenter,yCenter,extents.xMax(),extents.yMax(), extents._isGeographic);

            destinationGraph->addChild(createDestinationGraph(destinationGraph,
                                                                         cs,
                                                                         bottom_left,
                                                                         maxImageSize,
                                                                         maxTerrainSize,
                                                                         newLevel,
                                                                         newX,
                                                                         newY,
                                                                         maxNumLevels));

            destinationGraph->addChild(createDestinationGraph(destinationGraph,
                                                                         cs,
                                                                         bottom_right,
                                                                         maxImageSize,
                                                                         maxTerrainSize,
                                                                         newLevel,
                                                                         newX+1,
                                                                         newY,
                                                                         maxNumLevels));

            destinationGraph->addChild(createDestinationGraph(destinationGraph,
                                                                         cs,
                                                                         top_left,
                                                                         maxImageSize,
                                                                         maxTerrainSize,
                                                                         newLevel,
                                                                         newX,
                                                                         newY+1,
                                                                         maxNumLevels));

            destinationGraph->addChild(createDestinationGraph(destinationGraph,
                                                                         cs,
                                                                         top_right,
                                                                         maxImageSize,
                                                                         maxTerrainSize,
                                                                         newLevel,
                                                                         newX+1,
                                                                         newY+1,
                                                                         maxNumLevels));

            // Set all there max distance to the same value to ensure the same LOD bining.
            float cutOffDistance = destinationGraph->_maxVisibleDistance*0.5f;

            for(CompositeDestination::ChildList::iterator citr=destinationGraph->_children.begin();
                citr!=destinationGraph->_children.end();
                ++citr)
            {
                (*citr)->_maxVisibleDistance = cutOffDistance;
            }

        }
        else if (needToDivideX)
        {
            log(osg::INFO,"Need to Divide X only");

            // create two tiles.
            GeospatialExtents left(extents.xMin(),extents.yMin(),xCenter,extents.yMax(), extents._isGeographic);
            GeospatialExtents right(xCenter,extents.yMin(),extents.xMax(),extents.yMax(), extents._isGeographic);

            destinationGraph->addChild(createDestinationGraph(destinationGraph,
                                                                         cs,
                                                                         left,
                                                                         maxImageSize,
                                                                         maxTerrainSize,
                                                                         newLevel,
                                                                         newX,
                                                                         newY,
                                                                         maxNumLevels));

            destinationGraph->addChild(createDestinationGraph(destinationGraph,
                                                                         cs,
                                                                         right,
                                                                         maxImageSize,
                                                                         maxTerrainSize,
                                                                         newLevel,
                                                                         newX+1,
                                                                         newY,
                                                                         maxNumLevels));


            // Set all there max distance to the same value to ensure the same LOD bining.
            float cutOffDistance = destinationGraph->_maxVisibleDistance*0.5f;

            for(CompositeDestination::ChildList::iterator citr=destinationGraph->_children.begin();
                citr!=destinationGraph->_children.end();
                ++citr)
            {
                (*citr)->_maxVisibleDistance = cutOffDistance;
            }

        }
        else if (needToDivideY)
        {
            log(osg::INFO,"Need to Divide Y only");

            // create two tiles.
            GeospatialExtents top(extents.xMin(),yCenter,extents.xMax(),extents.yMax(), extents._isGeographic);
            GeospatialExtents bottom(extents.xMin(),extents.yMin(),extents.xMax(),yCenter, extents._isGeographic);

            destinationGraph->addChild(createDestinationGraph(destinationGraph,
                                                                         cs,
                                                                         bottom,
                                                                         maxImageSize,
                                                                         maxTerrainSize,
                                                                         newLevel,
                                                                         newX,
                                                                         newY,
                                                                         maxNumLevels));

            destinationGraph->addChild(createDestinationGraph(destinationGraph,
                                                                         cs,
                                                                         top,
                                                                         maxImageSize,
                                                                         maxTerrainSize,
                                                                         newLevel,
                                                                         newX,
                                                                         newY+1,
                                                                         maxNumLevels));

            // Set all there max distance to the same value to ensure the same LOD bining.
            float cutOffDistance = destinationGraph->_maxVisibleDistance*0.5f;

            for(CompositeDestination::ChildList::iterator citr=destinationGraph->_children.begin();
                citr!=destinationGraph->_children.end();
                ++citr)
            {
                (*citr)->_maxVisibleDistance = cutOffDistance;
            }

        }
        else
        {
            log(osg::INFO,"No Need to Divide");
        }
    }

    return destinationGraph;
}

bool DataSet::prepareForDestinationGraphCreation()
{
    if (!_sourceGraph) return false;

    // ensure we have a valid coordinate system
    if (_destinationCoordinateSystemString.empty()&& !getConvertFromGeographicToGeocentric())
    {
        for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
        {
            SourceData* sd = (*itr)->getSourceData();
            if (sd)
            {
                if (sd->_cs.valid())
                {
                    _destinationCoordinateSystem = sd->_cs;
                    log(osg::INFO,"Setting coordinate system to %s",_destinationCoordinateSystem->getCoordinateSystem().c_str());
                    break;
                }
            }
        }
    }


    assignIntermediateCoordinateSystem();

    CoordinateSystemType destinateCoordSytemType = getCoordinateSystemType(_destinationCoordinateSystem.get());
    if (destinateCoordSytemType==GEOGRAPHIC && !getConvertFromGeographicToGeocentric())
    {
        // convert elevation into degrees.
        setVerticalScale(1.0f/111319.0f);
    }

    // get the extents of the sources and
    _destinationExtents = _extents;
    _destinationExtents._isGeographic = destinateCoordSytemType==GEOGRAPHIC;

    // sort the sources so that the lowest res tiles are drawn first.
    {

#if 0
        for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
        {
            Source* source = itr->get();
            if (source)
            {
                source->setSortValueFromSourceDataResolution(_intermediateCoordinateSystem.get());
                log(osg::INFO, "sort %s value %f",source->getFileName().c_str(),source->getSortValue());
            }

        }

        // sort them so highest sortValue is first.
#endif
        _sourceGraph->setSortValueFromSourceDataResolution(_intermediateCoordinateSystem.get());
        _sourceGraph->sortBySourceSortValue();
    }

    if (!_destinationExtents.valid())
    {
        for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
        {
            SourceData* sd = (*itr)->getSourceData();
            if (sd)
            {
                GeospatialExtents local_extents(sd->getExtents(_intermediateCoordinateSystem.get()));
                log(osg::INFO, "local_extents = xMin() %f %f",local_extents.xMin(),local_extents.xMax());
                log(osg::INFO, "                yMin() %f %f",local_extents.yMin(),local_extents.yMax());

                if (destinateCoordSytemType==GEOGRAPHIC)
                {
                    // need to clamp within -180 and 180 range.
                    if (local_extents.xMin()>180.0)
                    {
                        // shift back to -180 to 180 range
                        local_extents.xMin() -= 360.0;
                        local_extents.xMax() -= 360.0;
                    }
                    else if (local_extents.xMin()<-180.0)
                    {
                        // shift back to -180 to 180 range
                        local_extents.xMin() += 360.0;
                        local_extents.xMax() += 360.0;
                    }
                }

                _destinationExtents.expandBy(local_extents);
            }
        }
    }


    if (destinateCoordSytemType==GEOGRAPHIC)
    {
        double xRange = _destinationExtents.xMax() - _destinationExtents.xMin();
        if (xRange>360.0)
        {
            // clamp to proper 360 range.
            _destinationExtents.xMin() = -180.0;
            _destinationExtents.xMax() = 180.0;
        }
    }

    //re-assign extnts so that if we write out a source file it has the correct extents
    _extents = _destinationExtents;

    log(osg::NOTICE, "local_extents = xMin() %f %f",_extents.xMin(),_extents.xMax());
    log(osg::NOTICE, "                yMin() %f %f",_extents.yMin(),_extents.yMax());

    // compute the number of texture layers required.
    unsigned int maxTextureUnit = 0;
    for(CompositeSource::source_iterator sitr(_sourceGraph.get());sitr.valid();++sitr)
    {
        Source* source = sitr->get();
        if (source)
        {
            if (maxTextureUnit<source->getLayer()) maxTextureUnit = source->getLayer();
        }
    }
    _numTextureLevels = maxTextureUnit+1;

    computeOptimumTileSystemDimensions(_C1,_R1);

    log(osg::INFO, "extents = xMin() %f %f",_destinationExtents.xMin(),_destinationExtents.xMax());
    log(osg::INFO, "          yMin() %f %f",_destinationExtents.yMin(),_destinationExtents.yMax());

    return true;
}

void DataSet::computeDestinationGraphFromSources(unsigned int numLevels)
{
    if (!prepareForDestinationGraphCreation()) return;

    osg::Timer_t before = osg::Timer::instance()->tick();

    // then create the destination graph accordingly.
    if (getBuildOptionsString().find("old_dg")!=std::string::npos)
    {
        log(osg::NOTICE,"Old DataSet::createDestinationGraph() selected");

        _destinationGraph = createDestinationGraph(0,
                                                   _intermediateCoordinateSystem.get(),
                                                   _destinationExtents,
                                                   _maximumTileImageSize,
                                                   _maximumTileTerrainSize,
                                                   0,
                                                   0,
                                                   0,
                                                   numLevels);
    }
    else
    {
        // new default scheme.
        createNewDestinationGraph(_intermediateCoordinateSystem.get(),
                                  _destinationExtents,
                                  _maximumTileImageSize,
                                  _maximumTileTerrainSize,
                                  numLevels);
    }

    osg::Timer_t after = osg::Timer::instance()->tick();

    log(osg::NOTICE,"Time for createDestinationGraph %f", osg::Timer::instance()->delta_s(before, after));


    // now traverse the destination graph to build neighbours.
    if (_destinationGraph.valid()) _destinationGraph->computeNeighboursFromQuadMap();

    osg::Timer_t after_computeNeighbours = osg::Timer::instance()->tick();

    log(osg::NOTICE,"Time for after_computeNeighbours %f", osg::Timer::instance()->delta_s(after, after_computeNeighbours));
}

void DataSet::assignDestinationCoordinateSystem()
{
    if (getDestinationCoordinateSystem().empty() && !getConvertFromGeographicToGeocentric())
    {
        for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
        {
            Source* source = itr->get();
            source->loadSourceData();
            _destinationCoordinateSystem = source->_cs;
            _destinationCoordinateSystemString = source->_cs.valid() ? source->_cs->getCoordinateSystem() : std::string();

            log(osg::NOTICE,"DataSet::assignDestinationCoordinateSystem() : assigning first source file as the destination coordinate system");
            break;
        }
    }
}


void DataSet::assignIntermediateCoordinateSystem()
{
    assignDestinationCoordinateSystem();

    if (!_intermediateCoordinateSystem)
    {
        CoordinateSystemType cst = getCoordinateSystemType(_destinationCoordinateSystem.get());

        log(osg::INFO, "new DataSet::createDestination()");
        if (cst!=GEOGRAPHIC && getConvertFromGeographicToGeocentric())
        {
            // need to use the geocentric coordinate system as a base for creating an geographic intermediate
            // coordinate system.
            OGRSpatialReference oSRS;

            if (_destinationCoordinateSystem.valid() && !_destinationCoordinateSystem->getCoordinateSystem().empty())
            {
                setIntermediateCoordinateSystem(_destinationCoordinateSystem->getCoordinateSystem());
            }
            else
            {
                char    *pszWKT = NULL;
                oSRS.SetWellKnownGeogCS( "WGS84" );
                oSRS.exportToWkt( &pszWKT );

                setIntermediateCoordinateSystem(pszWKT);
                setDestinationCoordinateSystem(pszWKT);
            }

        }
        else
        {
            _intermediateCoordinateSystem = _destinationCoordinateSystem;
        }
    }

}

bool DataSet::requiresReprojection()
{
    assignIntermediateCoordinateSystem();

    loadSources();

    for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
    {
        Source* source = itr->get();

        if (source && source->needReproject(_intermediateCoordinateSystem.get()))
        {
            return true;
        }
    }
    return false;
}

void DataSet::reprojectSourcesAndGenerateOverviews()
{
    if (!_sourceGraph) return;

    std::string temporyFilePrefix("temporaryfile_");

    osg::Timer_t before_reproject = osg::Timer::instance()->tick();

    // do standardisation of coordinates systems.
    // do any reprojection if required.
    {
        for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
        {
            Source* source = itr->get();

            log(osg::INFO, "Checking %s",source->getFileName().c_str());

            if (source && source->needReproject(_intermediateCoordinateSystem.get()))
            {

                if (getReprojectSources())
                {
                    if (source->isRaster())
                    {

                        // do the reprojection to a tempory file.
                        std::string newFileName = temporyFilePrefix + osgDB::getStrippedName(source->getFileName()) + ".tif";

                        Source* newSource = source->doRasterReprojection(newFileName,_intermediateCoordinateSystem.get());

                        // replace old source by new one.
                        if (newSource) *itr = newSource;
                        else
                        {
                            log(osg::WARN, "Failed to reproject %s",source->getFileName().c_str());
                            *itr = 0;
                        }
                    }
                    else
                    {
                        source->do3DObjectReprojection(_intermediateCoordinateSystem.get());
                    }
                }
                else
                {
                    log(osg::WARN, "Source file %s requires reprojection, but reprojection switched off.",source->getFileName().c_str());
                }
            }
        }
    }

    osg::Timer_t after_reproject = osg::Timer::instance()->tick();

    log(osg::NOTICE,"Time for after_reproject %f", osg::Timer::instance()->delta_s(before_reproject, after_reproject));

    // do sampling of data to required values.
    if (getBuildOverlays())
    {
        for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
        {
            Source* source = itr->get();
            if (source) source->buildOverviews();
        }
    }

    // osg::Timer_t after_sourceGraphsort = osg::Timer::instance()->tick();
}


void DataSet::updateSourcesForDestinationGraphNeeds()
{
    if (!_destinationGraph || !_sourceGraph) return;


    std::string temporyFilePrefix("temporaryfile_");

    osg::Timer_t before = osg::Timer::instance()->tick();

#if 0
    // compute the resolutions of the source that are required.
    {
        _destinationGraph->addRequiredResolutions(_sourceGraph.get());

        for(CompositeSource::source_iterator sitr(_sourceGraph.get());sitr.valid();++sitr)
        {
            Source* source = sitr->get();
            if (source)
            {
                log(osg::INFO, "Source File %s",source->getFileName().c_str());


                const Source::ResolutionList& resolutions = source->getRequiredResolutions();
                log(osg::INFO, "    resolutions.size() %u",resolutions.size());
                log(osg::INFO, "    { ");
                Source::ResolutionList::const_iterator itr;
                for(itr=resolutions.begin();
                    itr!=resolutions.end();
                    ++itr)
                {
                    log(osg::INFO, "        resX=%f resY=%f",itr->_resX,itr->_resY);
                }
                log(osg::INFO, "    } ");

                source->consolodateRequiredResolutions();

                log(osg::INFO, "    consolodated resolutions.size() %u",resolutions.size());
                log(osg::INFO, "    consolodated { ");
                for(itr=resolutions.begin();
                    itr!=resolutions.end();
                    ++itr)
                {
                    log(osg::INFO, "        resX=%f resY=%f",itr->_resX,itr->_resY);
                }
                log(osg::INFO, "    } ");
            }

        }
    }
#endif

    osg::Timer_t after_consolidate = osg::Timer::instance()->tick();


    log(osg::NOTICE,"Time for consolodateRequiredResolutions %f", osg::Timer::instance()->delta_s(before, after_consolidate));

    reprojectSourcesAndGenerateOverviews();

    log(osg::INFO, "Using source_lod_iterator itr");

    // buggy mips compiler requires this local variable in source_lod_iterator
    // usage below, since using _sourceGraph.get() as it should be was causing
    // a MIPSpro compiler error "The member "vpb::DataSet::_sourceGraph" is inaccessible."
    CompositeSource* my_sourceGraph = _sourceGraph.get();

    for(CompositeSource::source_lod_iterator csitr(my_sourceGraph,CompositeSource::LODSourceAdvancer(0.0));csitr.valid();++csitr)
    {
        Source* source = csitr->get();
        if (source)
        {
            log(osg::INFO, "  LOD %s",(*csitr)->getFileName().c_str());
        }
    }
    log(osg::INFO, "End of Using Source Iterator itr");

}

void DataSet::populateDestinationGraphFromSources()
{
    if (!_destinationGraph || !_sourceGraph) return;

    log(osg::NOTICE, "started DataSet::populateDestinationGraphFromSources)");

    if (_databaseType==LOD_DATABASE)
    {

        // for each DestinationTile populate it.
        _destinationGraph->readFrom(_sourceGraph.get());

        // for each DestinationTile equalize the boundaries so they all fit each other without gaps.
        _destinationGraph->equalizeBoundaries();


    }
    else
    {
        // for each level
        //  compute x and y range
        //  from top row down to bottom row equalize boundairies a write out
    }
    log(osg::NOTICE, "completed DataSet::populateDestinationGraphFromSources)");
}


class ReadFromOperation : public BuildOperation
{
    public:

        ReadFromOperation(ThreadPool* threadPool, BuildLog* buildLog, DestinationTile* tile, CompositeSource* sourceGraph):
            BuildOperation(threadPool, buildLog, "ReadFromOperation", false),
            _tile(tile),
            _sourceGraph(sourceGraph) {}

        virtual void build()
        {
            log(osg::NOTICE, "   ReadFromOperation: reading tile level=%u X=%u Y=%u",_tile->_level,_tile->_tileX,_tile->_tileY);
            _tile->readFrom(_sourceGraph.get());
        }

        osg::ref_ptr<DestinationTile> _tile;
        osg::ref_ptr<CompositeSource> _sourceGraph;
};

void DataSet::_readRow(Row& row)
{
    log(osg::NOTICE, "_readRow %u",row.size());

    CompositeSource* sourceGraph = _newDestinationGraph ? 0 : _sourceGraph.get();

    if (_readThreadPool.valid())
    {
        for(Row::iterator citr=row.begin();
            citr!=row.end();
            ++citr)
        {
            CompositeDestination* cd = citr->second;
            for(CompositeDestination::TileList::iterator titr=cd->_tiles.begin();
                titr!=cd->_tiles.end();
                ++titr)
            {
                _readThreadPool->run(new ReadFromOperation(_readThreadPool.get(), getBuildLog(), titr->get(), sourceGraph));
            }
        }

        // wait for the threads to complete.
        _readThreadPool->waitForCompletion();

    }
    else
    {
        for(Row::iterator citr=row.begin();
            citr!=row.end();
            ++citr)
        {
            CompositeDestination* cd = citr->second;
            for(CompositeDestination::TileList::iterator titr=cd->_tiles.begin();
                titr!=cd->_tiles.end();
                ++titr)
            {
                DestinationTile* tile = titr->get();
                log(osg::NOTICE, "   reading tile level=%u X=%u Y=%u",tile->_level,tile->_tileX,tile->_tileY);
                tile->readFrom(sourceGraph);
            }
        }
    }
}

void DataSet::_equalizeRow(Row& row)
{
    log(osg::NOTICE, "_equalizeRow %d",row.size());
    for(Row::iterator citr=row.begin();
        citr!=row.end();
        ++citr)
    {
        CompositeDestination* cd = citr->second;
        for(CompositeDestination::TileList::iterator titr=cd->_tiles.begin();
            titr!=cd->_tiles.end();
            ++titr)
        {
            DestinationTile* tile = titr->get();
            log(osg::NOTICE, "   equalizing tile level=%u X=%u Y=%u",tile->_level,tile->_tileX,tile->_tileY);
            tile->equalizeBoundaries();
            tile->setTileComplete(true);
        }
    }
}

void DataSet::_writeNodeFile(osg::Node& node,const std::string& filename)
{
    if (getDisableWrites()) return;

    if (_archive.valid()) _archive->writeNode(node,filename);
    else
    {
        osg::NotifySeverity notifylevel = getAbortTaskOnError() ? osg::FATAL : osg::WARN;

        if (vpb::hasWritePermission(filename))
        {
            bool fileExistedBeforeWrite = osgDB::fileExists(filename);

            osgDB::ReaderWriter::WriteResult result =
                osgDB::Registry::instance()->writeNode(node, filename,osgDB::Registry::instance()->getOptions());


            if (result.success())
            {
                if (_databaseRevision.valid())
                {
                    if (fileExistedBeforeWrite)
                    {
                        if (_databaseRevision->getFilesModified()) _databaseRevision->getFilesModified()->addFile(filename);
                    }
                    else
                    {
                        if (_databaseRevision->getFilesAdded()) _databaseRevision->getFilesAdded()->addFile(filename);
                    }
                }
            }
            else
            {
                if (!result.message().empty()) log(notifylevel, result.message());
                else
                {
                    switch(result.status())
                    {
                        case(osgDB::ReaderWriter::WriteResult::NOT_IMPLEMENTED):
                        case(osgDB::ReaderWriter::WriteResult::FILE_NOT_HANDLED):
                            log(notifylevel, "Error, write support for data type not available for node file %s",filename.c_str());
                            break;
                        case(osgDB::ReaderWriter::WriteResult::ERROR_IN_WRITING_FILE):
                            log(notifylevel, "Error, in writing node file %s",filename.c_str());
                            break;
                        default:
                            log(notifylevel, "Error, in writing node file %s",filename.c_str());
                            break;
                    }
                }
            }
        }
        else
        {
           log(notifylevel, "Error: do not have write permission to write out file %s",filename.c_str());
        }
    }
}

void DataSet::_writeImageFile(osg::Image& image,const std::string& filename)
{
    if (getDisableWrites()) return;

    //image.setFileName(filename.c_str());

    // remove any ../ from the filename
    std::string simpliedFileName = vpb::simplifyFileName(filename);

    if (_archive.valid()) _archive->writeImage(image,simpliedFileName);
    else
    {
        osg::NotifySeverity notifylevel = getAbortTaskOnError() ? osg::FATAL : osg::WARN;

        if (FilePathManager::instance()->checkWritePermissionAndEnsurePathAvailability(simpliedFileName))
        {
            bool fileExistedBeforeWrite = osgDB::fileExists(filename);
            bool isDDS = (osgDB::getLowerCaseFileExtension(simpliedFileName)=="dds");
            osg::ref_ptr<osgDB::Options> options = osgDB::Registry::instance()->getOptions();
            if (isDDS)
            {
                const char* ddsNoAtuoFlipWrite = "ddsNoAutoFlipWrite";
                if (options->getOptionString().find(ddsNoAtuoFlipWrite)==std::string::npos)
                {
                    options = osg::clone(options.get());
                    if (options->getOptionString().empty()) options->setOptionString(ddsNoAtuoFlipWrite);
                    else options->setOptionString(options->getOptionString()+" "+ddsNoAtuoFlipWrite);
                }
            }

            osgDB::ReaderWriter::WriteResult result =
                osgDB::Registry::instance()->writeImage(image, simpliedFileName, options.get());

            if (result.success())
            {
                if (_databaseRevision.valid())
                {
                    if (fileExistedBeforeWrite)
                    {
                        if (_databaseRevision->getFilesModified()) _databaseRevision->getFilesModified()->addFile(filename);
                    }
                    else
                    {
                        if (_databaseRevision->getFilesAdded()) _databaseRevision->getFilesAdded()->addFile(filename);
                    }
                }
            }
            else
            {
                if (!result.message().empty()) log(notifylevel, result.message().c_str());
                else
                {
                    switch(result.status())
                    {
                        case(osgDB::ReaderWriter::WriteResult::NOT_IMPLEMENTED):
                        case(osgDB::ReaderWriter::WriteResult::FILE_NOT_HANDLED):
                            log(notifylevel, "Error, write support for data type not available for image file %s",filename.c_str());
                            break;
                        case(osgDB::ReaderWriter::WriteResult::ERROR_IN_WRITING_FILE):
                            log(notifylevel, "Error, in writing image file %s",filename.c_str());
                            break;
                        default:
                            log(notifylevel, "Error, in writing image file %s",filename.c_str());
                            break;
                    }
                }
            }
        }
        else
        {
            log(notifylevel, "Error: do not have write permission to write out file %s",simpliedFileName.c_str());
        }
    }
}

class WriteImageFilesVisitor : public osg::NodeVisitor
{
public:

    WriteImageFilesVisitor(vpb::DataSet* dataSet, const std::string& directory):
        osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN),
        _dataSet(dataSet),
        _directory(directory),
        _writeHint(osg::Image::STORE_INLINE)
    {
        if (!_directory.empty())
        {
            char lastCharacter = _directory[_directory.size()-1];
            if (lastCharacter != '/' && lastCharacter != '\\')
            {
                _directory.push_back('/');
            }
        }

        if (dataSet->getDestinationTileExtension()==".osg" ||
            dataSet->getDestinationTileExtension()==".osgt" ||
            dataSet->getDestinationTileExtension()==".osgx") _writeHint = osg::Image::EXTERNAL_FILE;
    }

    vpb::DataSet*           _dataSet;
    std::string             _directory;
    osg::Image::WriteHint   _writeHint;

    bool needToWriteOutImage(const osg::Image* image) const
    {
        if (image->getWriteHint()!=osg::Image::NO_PREFERENCE)
            return image->getWriteHint()==osg::Image::EXTERNAL_FILE;
        else
            return _writeHint==osg::Image::EXTERNAL_FILE;
    }

    virtual void apply(osg::Node& node)
    {
        if (node.getStateSet()) apply(*(node.getStateSet()));

        traverse(node);
    }

    virtual void apply(osg::Group& group)
    {
        osgTerrain::TerrainTile* terrainTile = dynamic_cast<osgTerrain::TerrainTile*>(&group);
        if (terrainTile)
        {
            applyTerrain(*terrainTile);
        }
        else
        {
            osg::NodeVisitor::apply(group);
        }
    }


    void writeLayer(osgTerrain::Layer* layer)
    {
        if (!layer) return;

        osgTerrain::ImageLayer* imageLayer = dynamic_cast<osgTerrain::ImageLayer*>(layer);
        if (imageLayer)
        {
            osg::Image* image = imageLayer->getImage();
            if (image)
            {
                _dataSet->log(osg::NOTICE,"Writing out image layer %s, _directory=%s ",image->getFileName().c_str(),_directory.c_str());
                if (needToWriteOutImage(image)) _dataSet->_writeImageFile(*image,_directory+image->getFileName());
            }
            return;
        }

        osgTerrain::SwitchLayer* switchLayer = dynamic_cast<osgTerrain::SwitchLayer*>(layer);
        if (switchLayer)
        {
            for(unsigned int i=0; i<switchLayer->getNumLayers(); ++i)
            {
                _dataSet->log(osg::NOTICE,"Writing out switch layer child: %s, %s ",switchLayer->getSetName(i).c_str(),switchLayer->getFileName(i).c_str());
                writeLayer(switchLayer->getLayer(i));
            }
            return;
        }

        osgTerrain::CompositeLayer* compositeLayer = dynamic_cast<osgTerrain::CompositeLayer*>(layer);
        if (compositeLayer)
        {
            for(unsigned int i=0; i<compositeLayer->getNumLayers(); ++i)
            {
                _dataSet->log(osg::NOTICE,"Writing out composite layer child: %s",switchLayer->getFileName(i).c_str());
                writeLayer(compositeLayer->getLayer(i));
            }
            return;
        }
    }

    void applyTerrain(osgTerrain::TerrainTile& terrainTile)
    {
#if 0
        if (terrainTile.getElevationLayer()) writeLayer(terrainTile.getElevationLayer());
#endif

        for(unsigned int i=0; i<terrainTile.getNumColorLayers(); ++i)
        {
            if (terrainTile.getColorLayer(i))
            {
                writeLayer(terrainTile.getColorLayer(i));
            }
        }
    }

    virtual void apply(osg::Geode& geode)
    {
        if (geode.getStateSet()) apply(*(geode.getStateSet()));

        for(unsigned int i=0;i<geode.getNumDrawables();++i)
        {
            if (geode.getDrawable(i)->getStateSet()) apply(*(geode.getDrawable(i)->getStateSet()));
        }

        traverse(geode);
    }

    void apply(osg::StateSet& stateset)
    {
        for(unsigned int i=0;i<stateset.getTextureAttributeList().size();++i)
        {
            osg::Image* image = 0;
            osg::Texture2D* texture2D = dynamic_cast<osg::Texture2D*>(stateset.getTextureAttribute(i,osg::StateAttribute::TEXTURE));
            if (texture2D) image = texture2D->getImage();

            if (image && needToWriteOutImage(image))
            {
                _dataSet->_writeImageFile(*image,_directory+image->getFileName());
            }
        }
    }
};

void DataSet::_writeNodeFileAndImages(osg::Node& node,const std::string& filename)
{
    if (getDisableWrites()) return;

    log(osg::NOTICE,"_writeNodeFile(%s)",filename.c_str());

    // write out any image data that is an external file
    WriteImageFilesVisitor wifv(this, osgDB::getFilePath(filename));
    const_cast<osg::Node&>(node).accept(wifv);

    // write out the nodes
    _writeNodeFile(node,filename);
}



class WriteOperation : public BuildOperation
{
    public:

        WriteOperation(ThreadPool* threadPool, DataSet* dataset,CompositeDestination* cd, const std::string& filename):
            BuildOperation(threadPool, dataset->getBuildLog(), "WriteOperation", false),
            _dataset(dataset),
            _cd(cd),
            _filename(filename) {}

        virtual void build()
        {
            //notify(osg::NOTICE)<<"   WriteOperation"<<std::endl;

            osg::ref_ptr<osg::Node> node = _cd->createSubTileScene();
            if (node.valid())
            {
                if (_buildLog.valid()) _buildLog->log(osg::NOTICE, "   writeSubTile filename= %s",_filename.c_str());

                _dataset->_writeNodeFileAndImages(*node,_filename);

                _cd->setSubTilesGenerated(true);
                _cd->unrefSubTileData();
            }
            else
            {
                log(osg::WARN, "   failed to writeSubTile node for tile, filename=%s",_filename.c_str());
            }
        }

        DataSet*                            _dataset;
        osg::ref_ptr<CompositeDestination>  _cd;
        std::string                         _filename;
};

#define NEW_NAMING

void DataSet::_writeRow(Row& row)
{
    log(osg::NOTICE, "_writeRow %u",row.size());
    for(Row::iterator citr=row.begin();
        citr!=row.end();
        ++citr)
    {
        CompositeDestination* cd = citr->second;
        CompositeDestination* parent = cd->_parent;

        if (parent)
        {
            if (!parent->getSubTilesGenerated() && parent->areSubTilesComplete())
            {
                parent->setSubTilesGenerated(true);

#ifdef NEW_NAMING
                std::string filename = cd->getTileFileName();
#else
                std::string filename = _taskOutputDirectory+parent->getSubTileName();
#endif
                log(osg::NOTICE, "       _taskOutputDirectory= %s",_taskOutputDirectory.c_str());

                if (_writeThreadPool.valid())
                {
                    _writeThreadPool->run(new WriteOperation(_writeThreadPool.get(), this, parent, filename));
                }
                else
                {
                    osg::ref_ptr<osg::Node> node = parent->createSubTileScene();
                    if (node.valid())
                    {
                        log(osg::NOTICE, "   writeSubTile filename= %s",filename.c_str());
                        _writeNodeFileAndImages(*node,filename);


                        parent->setSubTilesGenerated(true);
                        parent->unrefSubTileData();

                    }
                    else
                    {
                        log(osg::WARN, "   failed to writeSubTile node for tile, filename=%s",filename.c_str());
                    }
                }
            }
        }
        else
        {
            osg::ref_ptr<osg::Node> node = cd->createPagedLODScene();

#ifdef NEW_NAMING
            std::string filename = cd->getTileFileName();
#else
            std::string filename;
#endif

            if (cd->_level==0)
            {

#ifndef NEW_NAMING
                filename = getDirectory() + _tileBasename + _tileExtension;
#endif
                if (_decorateWithMultiTextureControl)
                {
                    node = decorateWithMultiTextureControl(node.get());
                }

                if (getGeometryType()==TERRAIN)
                {
                    node = decorateWithTerrain(node.get());
                }
                else if (_decorateWithCoordinateSystemNode)
                {
                    node = decorateWithCoordinateSystemNode(node.get());
                }

                if (!_comment.empty())
                {
                    node->addDescription(_comment);
                }

                log(osg::NOTICE, "       getDirectory()= %s",getDirectory().c_str());
            }
            else
            {
#ifndef NEW_NAMING
                filename = _taskOutputDirectory + _tileBasename + _tileExtension;
#endif

                log(osg::NOTICE, "       _taskOutputDirectory= %s",_taskOutputDirectory.c_str());
            }

            if (node.valid())
            {
                log(osg::NOTICE, "   writeNodeFile = %u X=%u Y=%u filename=%s",cd->_level,cd->_tileX,cd->_tileY,filename.c_str());

                _writeNodeFileAndImages(*node,filename);
            }
            else
            {
                log(osg::WARN, "   faild to write node for tile = %u X=%u Y=%u filename=%s",cd->_level,cd->_tileX,cd->_tileY,filename.c_str());
            }

            // record the top nodes as the rootNode of the database
            _rootNode = node;

        }
    }

#if 0
    if (_writeThreadPool.valid()) _writeThreadPool->waitForCompletion();
#endif

}

void DataSet::createDestination(unsigned int numLevels)
{
    log(osg::NOTICE, "started DataSet::createDestination(%u)",numLevels);

#if 1

    reprojectSourcesAndGenerateOverviews();
    computeDestinationGraphFromSources(numLevels);

#else
    computeDestinationGraphFromSources(numLevels);

    updateSourcesForDestinationGraphNeeds();
#endif

    log(osg::NOTICE, "completed DataSet::createDestination(%u)",numLevels);

}

osg::Node* DataSet::decorateWithTerrain(osg::Node* subgraph)
{
    osgTerrain::Terrain* terrain = new osgTerrain::Terrain;

    if (_destinationCoordinateSystem.valid() && !(_destinationCoordinateSystem->getCoordinateSystem().empty()))
    {
        terrain->setFormat(_destinationCoordinateSystem->getFormat());
        terrain->setCoordinateSystem(_destinationCoordinateSystem->getCoordinateSystem());
    }

    // set the ellipsoid model if geocentric coords are used.
    if (getConvertFromGeographicToGeocentric()) terrain->setEllipsoidModel(getEllipsoidModel());

    // add the a subgraph.
    terrain->addChild(subgraph);

    log(osg::NOTICE, "called DataSet::decorateWithTerrain()");

    return terrain;
}

osg::Node* DataSet::decorateWithCoordinateSystemNode(osg::Node* subgraph)
{
    // don't decorate if no coord system is set.
    if (!_destinationCoordinateSystem || _destinationCoordinateSystem->getCoordinateSystem().empty())
        return subgraph;

    osg::CoordinateSystemNode* csn = new osg::CoordinateSystemNode(
            _destinationCoordinateSystem->getFormat(),
            _destinationCoordinateSystem->getCoordinateSystem());

    // set the ellipsoid model if geocentric coords are used.
    if (getConvertFromGeographicToGeocentric()) csn->setEllipsoidModel(getEllipsoidModel());

    // add the a subgraph.
    csn->addChild(subgraph);

    return csn;
}

osg::Node* DataSet::decorateWithMultiTextureControl(osg::Node* subgraph)
{
    // if only one layer exists don't need to decorate with MultiTextureControl
    if (_numTextureLevels<=1) return subgraph;


    // multiple layers active so use osgFX::MultiTextureControl to manage them
    osgFX::MultiTextureControl* mtc = new osgFX::MultiTextureControl;
    float r = 1.0f/(float)_numTextureLevels;
    for(unsigned int i=0;i<_numTextureLevels;++i)
    {
        mtc->setTextureWeight(i,r);
    }

    // add the a subgraph.
    mtc->addChild(subgraph);

    return mtc;
}


void DataSet::_buildDestination(bool writeToDisk)
{
    //if (!_state) _state = new osg::State;

    osg::ref_ptr<osgDB::ReaderWriter::Options> previous_options = osgDB::Registry::instance()->getOptions();
    if(previous_options.get())
    {
        log(osg::NOTICE, "vpb: adding optionstring %s",previous_options->getOptionString().c_str());
        osgDB::Registry::instance()->setOptions(new osgDB::ReaderWriter::Options(std::string("precision 16") + std::string(" ") + previous_options->getOptionString()) );
    }
    else
    {
        osgDB::Registry::instance()->setOptions(new osgDB::ReaderWriter::Options("precision 16"));
    }

    if (!_archive && !_archiveName.empty())
    {
        unsigned int indexBlockSizeHint=4096;
        _archive = osgDB::openArchive(_archiveName, osgDB::Archive::CREATE, indexBlockSizeHint);
    }

    if (_destinationGraph.valid())
    {
#ifdef NEW_NAMING
        std::string filename = _destinationGraph->getTileFileName();
#else
        std::string filename = _directory+_tileBasename+_tileExtension;
#endif

        if (_archive.valid())
        {
            log(osg::NOTICE, "started DataSet::writeDestination(%s)",_archiveName.c_str());
            log(osg::NOTICE, "        archive file = %s",_archiveName.c_str());
            log(osg::NOTICE, "        archive master file = %s",filename.c_str());
        }
        else
        {
            log(osg::NOTICE, "started DataSet::writeDestination(%s)",filename.c_str());
        }

        if (_databaseType==LOD_DATABASE)
        {
            populateDestinationGraphFromSources();
            _rootNode = _destinationGraph->createScene();

            if (_decorateWithMultiTextureControl)
            {
                _rootNode = decorateWithMultiTextureControl(_rootNode.get());
            }

            if (getGeometryType()==TERRAIN)
            {
                _rootNode = decorateWithTerrain(_rootNode.get());
            }
            else if (_decorateWithCoordinateSystemNode)
            {
                _rootNode = decorateWithCoordinateSystemNode(_rootNode.get());
            }

            if (!_comment.empty())
            {
                _rootNode->addDescription(_comment);
            }

            if (writeToDisk)
            {
                _writeNodeFileAndImages(*_rootNode,filename);
            }
        }
        else  // _databaseType==PagedLOD_DATABASE
        {

            // for each level build read and write the rows.
            for(QuadMap::iterator qitr=_quadMap.begin();
                qitr!=_quadMap.end();
                ++qitr)
            {
                Level& level = qitr->second;

                // skip is level is empty.
                if (level.empty()) continue;

                // skip lower levels if we are generating subtiles
                if (getGenerateSubtile() && qitr->first<=getSubtileLevel()) continue;

                if (getRecordSubtileFileNamesOnLeafTile() && qitr->first>=getMaximumNumOfLevels()) continue;

                log(osg::INFO, "New level");

                Level::iterator prev_itr = level.begin();
                _readRow(prev_itr->second);
                Level::iterator curr_itr = prev_itr;
                ++curr_itr;
                for(;
                    curr_itr!=level.end();
                    ++curr_itr)
                {
                    _readRow(curr_itr->second);

                    _equalizeRow(prev_itr->second);
                    if (writeToDisk) _writeRow(prev_itr->second);

                    prev_itr = curr_itr;
                }

                _equalizeRow(prev_itr->second);

                if (writeToDisk)
                {
                    if (writeToDisk) _writeRow(prev_itr->second);
                }

#if 0
                if (_writeThreadPool.valid()) _writeThreadPool->waitForCompletion();
#endif
            }

        }

        if (_archive.valid())
        {
            log(osg::NOTICE, "completed DataSet::writeDestination(%s)",_archiveName.c_str());
            log(osg::NOTICE, "          archive file = %s",_archiveName.c_str());
            log(osg::NOTICE, "          archive master file = %s",filename.c_str());
        }
        else
        {
            log(osg::NOTICE, "completed DataSet::writeDestination(%s)",filename.c_str());
        }

        if (_writeThreadPool.valid()) _writeThreadPool->waitForCompletion();

    }
    else
    {
        log(osg::WARN, "Error: no scene graph to output, no file written.");
    }

    if (_archive.valid()) _archive->close();

    osgDB::Registry::instance()->setOptions(previous_options.get());

}


bool DataSet::addModel(Source::Type type, osg::Node* model, unsigned int revisionNumber)
{
    vpb::Source* source = new vpb::Source(type, model);

    osgTerrain::Locator* locator = dynamic_cast<osgTerrain::Locator*>(model->getUserData());
    if (locator)
    {
        osg::notify(osg::NOTICE)<<"addModel : Assigned coordinate system ()"<<std::endl;

        source->setGeoTransformPolicy(vpb::Source::PREFER_CONFIG_SETTINGS);
        source->setGeoTransform(locator->getTransform());

        source->setCoordinateSystemPolicy(vpb::Source::PREFER_CONFIG_SETTINGS);
        source->setCoordinateSystem(locator->getCoordinateSystem());

    }

    osg::ComputeBoundsVisitor cbv;


    const osg::Node::DescriptionList& descriptions = model->getDescriptions();
    for(osg::Node::DescriptionList::const_iterator itr = descriptions.begin();
        itr != descriptions.end();
        ++itr)
    {
        int value=0;
        const std::string& desc = *itr;
        if (getAttributeValue(desc, "MinLevel", value))
        {
            source->setMinLevel(value);
        }
        else if (getAttributeValue(desc, "MaxLevel", value))
        {
            source->setMaxLevel(value);
        }
    }


    model->accept(cbv);

    source->_extents.xMin() = cbv.getBoundingBox().xMin();
    source->_extents.xMax() = cbv.getBoundingBox().xMax();

    source->_extents.yMin() = cbv.getBoundingBox().yMin();
    source->_extents.yMax() = cbv.getBoundingBox().yMax();

    source->getSourceData()->_extents._min = source->_extents._min;
    source->getSourceData()->_extents._max = source->_extents._max;

    osg::notify(osg::NOTICE)<<"addModel("<<type<<","<<model->getName()<<")"<<std::endl;
    osg::notify(osg::NOTICE)<<"   extents "<<source->_extents.xMin()<<" "<<source->_extents.xMax()<<std::endl;
    osg::notify(osg::NOTICE)<<"           "<<source->_extents.yMin()<<" "<<source->_extents.yMax()<<std::endl;

    addSource(source, revisionNumber);

    return true;
}

bool DataSet::addLayer(Source::Type type, osgTerrain::Layer* layer, unsigned layerNum, unsigned int revisionNumber)
{

    osgTerrain::HeightFieldLayer* hfl = dynamic_cast<osgTerrain::HeightFieldLayer*>(layer);
    if (hfl)
    {
        // need to read locator.
        vpb::Source* source = new vpb::Source(type, hfl->getFileName());
        source->setSetName(hfl->getSetName(), this);
        source->setLayer(layerNum);
        source->setMinLevel(layer->getMinLevel());
        source->setMaxLevel(layer->getMaxLevel());

        if (layer->getLocator() && !layer->getLocator()->getDefinedInFile())
        {
            vpb::Source::ParameterPolicy geoTransformPolicy = layer->getLocator()->getTransformScaledByResolution() ?
                    vpb::Source::PREFER_CONFIG_SETTINGS :
                    vpb::Source::PREFER_CONFIG_SETTINGS_BUT_SCALE_BY_FILE_RESOLUTION;

            source->setGeoTransformPolicy(geoTransformPolicy);
            source->setGeoTransform(layer->getLocator()->getTransform());

            source->setCoordinateSystemPolicy(vpb::Source::PREFER_CONFIG_SETTINGS);
            source->setCoordinateSystem(layer->getLocator()->getCoordinateSystem());
        }

        addSource(source, revisionNumber);
        return true;
    }

    osgTerrain::ImageLayer* iml = dynamic_cast<osgTerrain::ImageLayer*>(layer);
    if (iml)
    {
        // need to read locator
        vpb::Source* source = new vpb::Source(type, iml->getFileName());
        source->setSetName(iml->getSetName(), this);
        source->setLayer(layerNum);
        source->setMinLevel(layer->getMinLevel());
        source->setMaxLevel(layer->getMaxLevel());

        if (layer->getLocator() && !layer->getLocator()->getDefinedInFile())
        {
            vpb::Source::ParameterPolicy geoTransformPolicy = layer->getLocator()->getTransformScaledByResolution() ?
                    vpb::Source::PREFER_CONFIG_SETTINGS :
                    vpb::Source::PREFER_CONFIG_SETTINGS_BUT_SCALE_BY_FILE_RESOLUTION;

            source->setGeoTransformPolicy(geoTransformPolicy);
            source->setGeoTransform(layer->getLocator()->getTransform());

            source->setCoordinateSystemPolicy(vpb::Source::PREFER_CONFIG_SETTINGS);
            source->setCoordinateSystem(layer->getLocator()->getCoordinateSystem());
        }

        addSource(source, revisionNumber);
        return true;
    }

    osgTerrain::ProxyLayer* pl = dynamic_cast<osgTerrain::ProxyLayer*>(layer);
    if (pl)
    {
        // remove any implementation as we don't need it.
        pl->setImplementation(0);

        vpb::Source* source = new vpb::Source(type, pl->getFileName());
        source->setSetName(pl->getSetName(), this);
        source->setLayer(layerNum);
        source->setMinLevel(layer->getMinLevel());
        source->setMaxLevel(layer->getMaxLevel());

        if (layer->getLocator() && !layer->getLocator()->getDefinedInFile())
        {
            vpb::Source::ParameterPolicy geoTransformPolicy = layer->getLocator()->getTransformScaledByResolution() ?
                    vpb::Source::PREFER_CONFIG_SETTINGS :
                    vpb::Source::PREFER_CONFIG_SETTINGS_BUT_SCALE_BY_FILE_RESOLUTION;

            source->setGeoTransformPolicy(geoTransformPolicy);
            source->setGeoTransform(layer->getLocator()->getTransform());

            source->setCoordinateSystemPolicy(vpb::Source::PREFER_CONFIG_SETTINGS);
            source->setCoordinateSystem(layer->getLocator()->getCoordinateSystem());
        }

        addSource(source, revisionNumber);
        return true;
    }

    osgTerrain::CompositeLayer* compositeLayer = dynamic_cast<osgTerrain::CompositeLayer*>(layer);
    if (compositeLayer)
    {
        for(unsigned int i=0; i<compositeLayer->getNumLayers();++i)
        {
            if (compositeLayer->getLayer(i))
            {
                addLayer(type, compositeLayer->getLayer(i), layerNum, revisionNumber);
            }
            else if (!compositeLayer->getFileName(i).empty())
            {
                vpb::Source* source = new vpb::Source(type, compositeLayer->getFileName(i));
                source->setSetName(compositeLayer->getSetName(i), this);
                source->setMinLevel(layer->getMinLevel());
                source->setMaxLevel(layer->getMaxLevel());
                source->setLayer(layerNum);
                addSource(source, revisionNumber);
            }
        }
        return true;
    }
    return false;
}

bool DataSet::addTerrain(osgTerrain::TerrainTile* terrainTile, unsigned int revisionNumber)
{
    if (terrainTile->getElevationLayer())
    {
        addLayer(vpb::Source::HEIGHT_FIELD, terrainTile->getElevationLayer(), 0, revisionNumber);
    }

    for(unsigned int i=0; i<terrainTile->getNumColorLayers();++i)
    {
        osgTerrain::Layer* layer = terrainTile->getColorLayer(i);
        if (layer)
        {
            addLayer(vpb::Source::IMAGE, layer, i, revisionNumber);
        }
    }

    for(unsigned int ci=0; ci<terrainTile->getNumChildren(); ++ci)
    {

        osg::Node* model = terrainTile->getChild(ci);

        osg::notify(osg::NOTICE)<<"Adding model"<<model->getName()<<std::endl;

        Source::Type type = vpb::Source::MODEL;
        for(unsigned di = 0; di< model->getNumDescriptions(); ++di)
        {
            const std::string& desc = model->getDescription(di);
            if (desc=="SHAPEFILE") type = Source::SHAPEFILE;
        }

        addModel(type, model, revisionNumber);
    }

    return true;
}

bool DataSet::addTerrain(osgTerrain::TerrainTile* terrainTile)
{
    log(osg::NOTICE,"Adding terrainTile %s",terrainTile->getName().c_str());

    vpb::DatabaseBuilder* db = dynamic_cast<vpb::DatabaseBuilder*>(terrainTile->getTerrainTechnique());
    unsigned int revisionNumber = 0;
    if (db && db->getBuildOptions())
    {
        revisionNumber = db->getBuildOptions()->getRevisionNumber();
        setBuildOptions(*(db->getBuildOptions()));
    }

    return addTerrain(terrainTile, revisionNumber);
}


bool DataSet::addPatchedTerrain(osgTerrain::TerrainTile* previous_terrain, osgTerrain::TerrainTile* new_terrain)
{
    log(osg::NOTICE,"Adding previous %s and current terrainTile %s",
        previous_terrain->getName().c_str(),
        new_terrain->getName().c_str());

    vpb::DatabaseBuilder* db = dynamic_cast<vpb::DatabaseBuilder*>(previous_terrain->getTerrainTechnique());
    BuildOptions* previous_bo = db ? db->getBuildOptions() : 0;
    unsigned int previous_revisionNumber = 0;
    if (previous_bo)
    {
        previous_revisionNumber = db->getBuildOptions()->getRevisionNumber();
    }

    db = dynamic_cast<vpb::DatabaseBuilder*>(new_terrain->getTerrainTechnique());
    BuildOptions* new_bo = db ? db->getBuildOptions() : 0;
    unsigned int new_revisionNumber = previous_revisionNumber;
    if (new_bo)
    {
        if (previous_bo)
        {
            // assigning previous exents
            if (new_bo->getDestinationExtents().valid())
            {
                log(osg::NOTICE, "new extents found.");
            }

            if (previous_bo->getDestinationExtents().valid())
            {
                log(osg::NOTICE, "assigning extents found.");
                new_bo->setDestinationExtents(previous_bo->getDestinationExtents());
            }
        }

        new_revisionNumber = new_bo->getRevisionNumber();
        setBuildOptions(*new_bo);
    }

    log(osg::NOTICE,"   previous revision Number %i",previous_revisionNumber);
    log(osg::NOTICE,"   new revision Number %i",new_revisionNumber);

    bool result = addTerrain(previous_terrain, previous_revisionNumber);
    result = addTerrain(new_terrain, new_revisionNumber) | result;

    _sourceGraph->assignSourcePatchStatus();

    for(CompositeSource::source_iterator sitr(_sourceGraph.get());sitr.valid();++sitr)
    {
        Source* source = sitr->get();
        if (source)
        {
            std::string status;
            switch(source->getPatchStatus())
            {
                case(Source::UNASSIGNED): status = "UNASSIGNED"; break;
                case(Source::UNCHANGED): status = "UNCHANGED"; break;
                case(Source::MODIFIED): status = "MODIFIED"; break;
                case(Source::ADDED): status = "ADDED"; break;
                case(Source::REMOVED): status = "REMOVED"; break;
            }

            log(osg::NOTICE, "Source File %s\t%d\t%s",source->getFileName().c_str(), source->getRevisionNumber(), status.c_str());
        }
    }

    unsigned int numberOfAlteredSources = _sourceGraph->getNumberAlteredSources();

    log(osg::NOTICE, "_sourceGraph->getNumberAlteredSources()=%d", numberOfAlteredSources);

    return numberOfAlteredSources!=0;
}


osgTerrain::TerrainTile* DataSet::createTerrainRepresentation()
{
    osg::ref_ptr<osgTerrain::TerrainTile> terrainTile = new osgTerrain::TerrainTile;

    for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
    {
        osg::ref_ptr<Source> source = (*itr);
#if 0
            osg::Locator* locator = new osg::Locator;
            osg::ref_ptr<osg::CoordinateSystemNode>     _cs;
            osg::Matrixd                                _geoTransform;
            GeospatialExtents                           _extents;
            DataType                                    _dataType;
            unsigned int                                _numValuesX;
            unsigned int                                _numValuesY;
            unsigned int                                _numValuesZ;
#endif
        unsigned int layerNum = source->getLayer();


        osg::ref_ptr<osg::Object> loadedObject = osgDB::readObjectFile(source->getFileName()+".gdal");
        osgTerrain::Layer* loadedLayer = dynamic_cast<osgTerrain::Layer*>(loadedObject.get());

        if (loadedLayer)
        {
            if (!loadedLayer->getLocator())
            {
                osgTerrain::Locator* locator = new osgTerrain::Locator;
                locator->setTransform(source->getGeoTransform());

                if (source->_cs.valid())
                {
                    locator->setCoordinateSystem(source->_cs->getCoordinateSystem());
                    locator->setFormat(source->_cs->getFormat());
                    locator->setEllipsoidModel(source->_cs->getEllipsoidModel());

                    switch(getCoordinateSystemType(source->_cs.get()))
                    {
                        case(GEOCENTRIC): locator->setCoordinateSystemType(osgTerrain::Locator::GEOCENTRIC); break;
                        case(GEOGRAPHIC): locator->setCoordinateSystemType(osgTerrain::Locator::GEOGRAPHIC); break;
                        case(PROJECTED): locator->setCoordinateSystemType(osgTerrain::Locator::PROJECTED); break;
                        case(LOCAL): locator->setCoordinateSystemType(osgTerrain::Locator::PROJECTED); break;
                    };
                }

                loadedLayer->setLocator(locator);
            }

            if (source->getType()==Source::IMAGE)
            {
                osgTerrain::Layer* existingLayer = (layerNum < terrainTile->getNumColorLayers()) ? terrainTile->getColorLayer(layerNum) : 0;
                osgTerrain::CompositeLayer* compositeLayer = dynamic_cast<osgTerrain::CompositeLayer*>(existingLayer);

                if (compositeLayer)
                {
                    compositeLayer->addLayer( loadedLayer );
                }
                else if (existingLayer)
                {
                    compositeLayer = new osgTerrain::CompositeLayer;
                    compositeLayer->addLayer( existingLayer );
                    compositeLayer->addLayer( loadedLayer );

                    terrainTile->setColorLayer(layerNum, compositeLayer);
                }
                else
                {
                    terrainTile->setColorLayer(layerNum, loadedLayer);
                }
            }
            else if (source->getType()==Source::HEIGHT_FIELD)
            {
                osgTerrain::Layer* existingLayer = terrainTile->getElevationLayer();
                osgTerrain::CompositeLayer* compositeLayer = dynamic_cast<osgTerrain::CompositeLayer*>(existingLayer);

                if (compositeLayer)
                {
                    compositeLayer->addLayer( loadedLayer );
                }
                else if (existingLayer)
                {
                    compositeLayer = new osgTerrain::CompositeLayer;
                    compositeLayer->addLayer( existingLayer );
                    compositeLayer->addLayer( loadedLayer );

                    terrainTile->setElevationLayer(compositeLayer);
                }
                else
                {
                    terrainTile->setElevationLayer(loadedLayer);
                }
            }
        }
    }

    osg::ref_ptr<DatabaseBuilder> builder = new DatabaseBuilder;
    builder->setBuildOptions(new BuildOptions(*this));
    builder->setBuildLog(getBuildLog());
    terrainTile->setTerrainTechnique(builder.get());

    return terrainTile.release();
}

class MyGraphicsContext : public osg::Referenced {
    public:
        MyGraphicsContext(BuildLog* buildLog)
        {
            osg::ref_ptr<osg::GraphicsContext::Traits> traits = new osg::GraphicsContext::Traits;
            traits->readDISPLAY();
            traits->x = 0;
            traits->y = 0;
            traits->width = 1;
            traits->height = 1;
            traits->windowDecoration = false;
            traits->doubleBuffer = false;
            traits->sharedContext = 0;
            traits->pbuffer = true;

            _graphicsContext = osg::GraphicsContext::createGraphicsContext(traits.get());

            if (!_graphicsContext)
            {
                if (buildLog) buildLog->log(osg::NOTICE,"Failed to create pbuffer, failing back to normal graphics window.");

                traits->pbuffer = false;
                _graphicsContext = osg::GraphicsContext::createGraphicsContext(traits.get());
            }

            if (_graphicsContext.valid())


            {
                _graphicsContext->realize();
                _graphicsContext->makeCurrent();

                if (buildLog) buildLog->log(osg::NOTICE,"Realized window");
            }
        }

        bool valid() const { return _graphicsContext.valid() && _graphicsContext->isRealized(); }

        osg::ref_ptr<osg::GraphicsContext> _graphicsContext;
};


class CollectSubtiles : public DestinationVisitor
{
public:

    CollectSubtiles(unsigned int level):
        _level(level) {}

    virtual void apply(CompositeDestination& cd)
    {
        if (cd._level<_level) traverse(cd);
        else if (cd._level==_level)
        {
            if (!cd._children.empty())
            {
                _subtileList.push_back(&cd);
            }
        }
    }

    typedef std::list< osg::ref_ptr<CompositeDestination> > SubtileList;
    unsigned int    _level;
    SubtileList     _subtileList;

};

bool DataSet::createTileMap(unsigned int level, TilePairMap& tilepairMap)
{
    osg::CoordinateSystemNode* cs = _intermediateCoordinateSystem.get();
    const GeospatialExtents& extents = _destinationExtents;
    unsigned int maxNumLevels = getMaximumNumOfLevels();

    // first populate the destination graph from imagery and DEM sources extents/resolution
    for(CompositeSource::source_iterator itr(_sourceGraph.get());itr.valid();++itr)
    {
        Source* source = (*itr).get();

#if 1
        if (source->getPatchStatus()==Source::UNCHANGED)
        {
            log(osg::NOTICE,"Skipping source %s as it hasn't changed during patching.",source->getFileName().c_str());
            continue;
        }
#endif
        if (source->getMinLevel()>maxNumLevels)
        {
            log(osg::NOTICE,"Skipping source %s as its min level excees destination max level.",source->getFileName().c_str());
            continue;
        }

        SourceData* sd = (*itr)->getSourceData();
        if (!sd)
        {
            log(osg::NOTICE,"Skipping source %s as no data loaded from it.",source->getFileName().c_str());
            continue;
        }

        const SpatialProperties& sp = sd->computeSpatialProperties(cs);

        if (!sp._extents.intersects(extents))
        {
            // skip this source since it doesn't overlap this tile.
            log(osg::NOTICE,"Skipping source %s as its extents don't overlap destination extents.",source->getFileName().c_str());
            continue;
        }

        if (source->getType()!=Source::IMAGE && source->getType()!=Source::HEIGHT_FIELD)
        {
            continue;
        }

        int k = 0;
        if (!computeOptimumLevel(source, maxNumLevels-1, k)) continue;

        // skip if the tiles won't contribute to the tasks with high level number.
        if (k<=static_cast<int>(level)) continue;

        int i_min, i_max, j_min, j_max;
        if (computeCoverage(sp._extents, level, i_min, j_min, i_max, j_max))
        {
            for(int j=j_min; j<j_max;++j)
            {
                for(int i=i_min; i<i_max;++i)
                {
                    TilePair tileID(i,j);
                    TilePairMap::iterator itr = tilepairMap.find(tileID);
                    if (itr != tilepairMap.end())
                    {
                        if (k > static_cast<int>(itr->second)) itr->second = k;
                    }
                    else
                    {
                        tilepairMap[TilePair(i,j)] = k;
                    }
                }
            }
        }
    }

#if 0
    for(TilePairMap::iterator itr = tilepairMap.begin();
        itr != tilepairMap.end();
        ++itr)
    {
        log(osg::NOTICE,"Level %d TilePair (%d, %d) %d",level, itr->first.first, itr->first.second, itr->second);
    }
#endif
    return true;
}

bool DataSet::generateTasks(TaskManager* taskManager)
{
    if (!getLogFileName().empty() && !getBuildLog())
    {
        setBuildLog(new BuildLog(getLogFileName()));
    }

    if (getBuildLog())
    {
        pushOperationLog(getBuildLog());
    }

    bool result = generateTasksImplementation(taskManager);

    if (getBuildLog())
    {
        popOperationLog();
    }

    return result;

}

void DataSet::selectAppropriateSplitLevels()
{
    unsigned int maxLevel = computeMaximumLevel(getMaximumNumOfLevels());
    log(osg::NOTICE,"Computed maximum source level = %i", maxLevel);

    if (getDistributedBuildSplitLevel()==0)
    {
        if (getDistributedBuildSecondarySplitLevel()==0)
        {
            // need to compute both primary and secodary split levels
            if (maxLevel<10)
            {
                // just use primary split level
                setDistributedBuildSplitLevel( maxLevel / 2 );
            }
            else
            {
                setDistributedBuildSplitLevel( maxLevel / 3 );
                setDistributedBuildSecondarySplitLevel( (maxLevel * 2) / 3 );
            }
        }
        else
        {
            // need to compute the primary split level only
            setDistributedBuildSplitLevel( getDistributedBuildSecondarySplitLevel()/2 );
        }
    }
    else
    {
        if (maxLevel>=10 && getDistributedBuildSecondarySplitLevel()==0)
        {
            // need to compute just the seconary split level
            setDistributedBuildSecondarySplitLevel(
                        getDistributedBuildSplitLevel() +
                        (maxLevel-getDistributedBuildSplitLevel())/2 );
        }
        else
        {
            // primary and secondary split levels fully specified.
        }
    }

    if (getDistributedBuildSplitLevel()>=maxLevel)
    {
        log(osg::NOTICE,"Warning: primary split level exceed maximum source level, switching off primary split level.");
        setDistributedBuildSplitLevel(0);
    }

    if (getDistributedBuildSecondarySplitLevel()>=maxLevel)
    {
        log(osg::NOTICE,"Warning: secondary split level exceed maximum source level, switching off secondary split level.");
        setDistributedBuildSecondarySplitLevel(0);
    }

    if (getDistributedBuildSecondarySplitLevel()!=0 &&
        getDistributedBuildSecondarySplitLevel()<=getDistributedBuildSplitLevel())
    {
        log(osg::NOTICE,"Warning: secondary split level is not permited to be lower than or equal to the primary split level, switching off secondary split level.");
        setDistributedBuildSecondarySplitLevel(0);
    }


    if (getDistributedBuildSecondarySplitLevel()==0)
    {
        log(osg::NOTICE,"Selected single split at %i",getDistributedBuildSplitLevel());
    }
    else
    {
        log(osg::NOTICE,"Selected primary split at %i, secondary split at %i",
            getDistributedBuildSplitLevel(),
            getDistributedBuildSecondarySplitLevel());
    }


}

std::string DataSet::getTaskName(unsigned int level, unsigned int X, unsigned int Y) const
{
    if (level==0 && X==0 && Y==0)
    {
        std::ostringstream taskfile;
        taskfile<<_tileBasename<<"_root_L0_X0_Y0";
        return taskfile.str();
    }
    else
    {
        if (getDistributedBuildSecondarySplitLevel()!=0 && level>=getDistributedBuildSecondarySplitLevel()-1)
        {
            unsigned int deltaLevels =  getDistributedBuildSecondarySplitLevel()-getDistributedBuildSplitLevel();
            unsigned int divisor = 1 << deltaLevels;
            unsigned int nestedX = X / divisor;
            unsigned int nestedY = Y / divisor;
            unsigned int nestedLevel = getDistributedBuildSplitLevel()-1;
            log(osg::NOTICE,"getTaskName(%i,%i,%i) requires nesting, divisor = %i (%i,%i,%i)",level,X,Y,divisor,nestedLevel, nestedX,nestedY);

            std::ostringstream taskfile;
            taskfile<<_tileBasename<<"_subtile_L"<<nestedLevel<<"_X"<<nestedX<<"_Y"<<nestedY<<"/";
            taskfile<<_tileBasename<<"_subtile_L"<<level<<"_X"<<X<<"_Y"<<Y;
            return taskfile.str();
        }
        else
        {
            log(osg::NOTICE,"getTaskName(%i,%i,%i) no nest, %i %i",level,X,Y,getDistributedBuildSplitLevel(),getDistributedBuildSecondarySplitLevel());

            std::ostringstream taskfile;
            taskfile<<_tileBasename<<"_subtile_L"<<level<<"_X"<<X<<"_Y"<<Y;
            return taskfile.str();
        }
    }
}

std::string DataSet::getSubtileName(unsigned int level, unsigned int X, unsigned int Y) const
{
    std::ostringstream os;
    os << _tileBasename << "_L"<<level<<"_X"<<X<<"_Y"<<Y<<"_subtile"<<getDestinationTileExtension();
    return os.str();
}

bool DataSet::generateTasksImplementation(TaskManager* taskManager)
{
    log(osg::NOTICE,"DataSet::generateTasks_new");

    loadSources();

    if (!prepareForDestinationGraphCreation()) return false;

    selectAppropriateSplitLevels();

    if (getDistributedBuildSplitLevel()==0) return false;

    int bottomDistributedBuildLevel = getDistributedBuildSecondarySplitLevel()==0 ?
                                        getDistributedBuildSplitLevel() :
                                        getDistributedBuildSecondarySplitLevel();


    if (!prepareForDestinationGraphCreation()) return false;

    // initialize various tasks related settings
    std::string sourceFile = taskManager->getSourceFileName();
    std::string basename = taskManager->getBuildName();

    std::string taskDirectory = getTaskDirectory();
    if (!taskDirectory.empty())
    {
        int result = 0;
        osgDB::FileType type = osgDB::fileType(taskDirectory);
        if (type==osgDB::DIRECTORY)
        {
            log(osg::NOTICE,"   Task directory already created");
        }
        else if (type==osgDB::REGULAR_FILE)
        {
            log(osg::NOTICE,"   Error cannot create Task directory as a conventional file already exists with that name");
            taskDirectory = "";
        }
        else // FILE_NOT_FOUND
        {
            // need to create directory.
            result = vpb::mkpath(taskDirectory.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
            if (result)
            {
                taskDirectory = "";
            }
        }

        if (!taskDirectory.empty()) taskDirectory += "/";
    }

    std::string logDirectory = getLogDirectory();
    if (!logDirectory.empty())
    {
        int result = 0;
        osgDB::FileType type = osgDB::fileType(logDirectory);
        if (type==osgDB::DIRECTORY)
        {
            log(osg::NOTICE,"   Log directory already created");
        }
        else if (type==osgDB::REGULAR_FILE)
        {
            log(osg::NOTICE,"   Error cannot create Log directory as a conventional file already exists with that name");
            logDirectory = "";
        }
        else // FILE_NOT_FOUND
        {
            // need to create directory.
            result = vpb::mkpath(logDirectory.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
            if (result)
            {
                logDirectory = "";
            }
        }

        if (!logDirectory.empty()) logDirectory += "/";
    }

    std::string fileCacheName;
    if (System::instance()->getFileCache()) fileCacheName = System::instance()->getFileCache()->getFileName();

    bool logging = getNotifyLevel() > ALWAYS;


    // create root task
    {
        std::ostringstream taskfile;
        taskfile<<taskDirectory<<basename<<"_root_L0_X0_Y0.task";

        std::ostringstream app;
        app<<"osgdem --run-path "<<taskManager->getRunPath()<<" -s "<<sourceFile<<" --record-subtile-on-leaf-tiles -l "<<getDistributedBuildSplitLevel()<<" --task "<<taskfile.str();

        if (!fileCacheName.empty())
        {
            app<<" --cache "<<fileCacheName;
        }

        if (logging)
        {
            std::ostringstream logfile;
            logfile<<logDirectory<<basename<<"_root_L0_X0_Y0.log";
            app<<" --log "<<logfile.str();
        }

        taskManager->addTask(taskfile.str(), app.str(), sourceFile, getDatabaseRevisionBaseFileName(0,0,0));
    }

    // create the tilemaps for the required split levels
    TilePairMap intermediateTileMap;
    if (getDistributedBuildSecondarySplitLevel()!=0)
    {
        createTileMap(getDistributedBuildSplitLevel()-1, intermediateTileMap);
    }

    TilePairMap bottomTileMap;
    createTileMap(bottomDistributedBuildLevel-1, bottomTileMap);

    unsigned int totalNumOfTasksSansRoot = intermediateTileMap.size() + bottomTileMap.size();
    unsigned int taskCount = 0;
    // unsigned int numTasksPerDirectory = getMaxNumberOfFilesPerDirectory();

    // bool tooManyTasksForOneDirectory = totalNumOfTasksSansRoot > numTasksPerDirectory;

    log(osg::NOTICE,"totalNumOfTasksSansRoot = %d", totalNumOfTasksSansRoot);

    // initialize the variables used for nested secondary tasks within primary tasks
    unsigned int deltaLevels = 0;
    unsigned int divisor = 1;
    unsigned int nestedLevel = getDistributedBuildSplitLevel()-1;

    // need to create an intermediate level if required.
    if (getDistributedBuildSecondarySplitLevel()!=0)
    {
        unsigned int level = getDistributedBuildSplitLevel()-1;

        for(TilePairMap::iterator itr = intermediateTileMap.begin();
            itr != intermediateTileMap.end();
            ++itr)
        {
            unsigned int tileX = itr->first.first;
            unsigned int tileY = itr->first.second;

            // unsigned int taskSet = taskCount / numTasksPerDirectory;

            std::ostringstream taskfile;
            taskfile<<taskDirectory;
            taskfile<<basename<<"_subtile_L"<<level<<"_X"<<tileX<<"_Y"<<tileY<<".task";


            std::ostringstream app;
            app<<"osgdem --run-path "<<taskManager->getRunPath()<<" -s "<<sourceFile<<" --record-subtile-on-leaf-tiles -l "<<getDistributedBuildSecondarySplitLevel()<<" --subtile "<<level<<" "<<tileX<<" "<<tileY<<" --task "<<taskfile.str();


            if (!fileCacheName.empty())
            {
                app<<" --cache "<<fileCacheName;
            }

            if (logging)
            {
                std::ostringstream logfile;

                logfile<<logDirectory;
                logfile<<basename<<"_subtile_L"<<level<<"_X"<<tileX<<"_Y"<<tileY<<".log";
                app<<" --log "<<logfile.str();
            }

            taskManager->addTask(taskfile.str(), app.str(), sourceFile, getDatabaseRevisionBaseFileName(level,tileX,tileY));

            ++taskCount;
        }

        // we have an intermediated level so the bottom level will need to be nested, so initiliaze
        // the deltaLevels and divisor for use in below.
        deltaLevels = getDistributedBuildSecondarySplitLevel()-getDistributedBuildSplitLevel();
        divisor = 1 << deltaLevels;
    }


    // create the bottom level split
    {
        unsigned int level = bottomDistributedBuildLevel-1;

        for(TilePairMap::iterator itr = bottomTileMap.begin();
            itr != bottomTileMap.end();
            ++itr)
        {
            unsigned int tileX = itr->first.first;
            unsigned int tileY = itr->first.second;

            // unsigned int taskSet = taskCount / numTasksPerDirectory;

            std::ostringstream taskfile;
            taskfile<<taskDirectory;
            if (deltaLevels)
            {
                unsigned int nestedX = tileX / divisor;
                unsigned int nestedY = tileY / divisor;
                taskfile<<basename<<"_subtile_L"<<nestedLevel<<"_X"<<nestedX<<"_Y"<<nestedY<<"/";

                std::string path = taskfile.str();
                vpb::mkpath(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
            }
            taskfile<<basename<<"_subtile_L"<<level<<"_X"<<tileX<<"_Y"<<tileY<<".task";


            std::ostringstream app;
            app<<"osgdem --run-path "<<taskManager->getRunPath()<<" -s "<<sourceFile<<" --subtile "<<level<<" "<<tileX<<" "<<tileY<<" --task "<<taskfile.str();

            if (!fileCacheName.empty())
            {
                app<<" --cache "<<fileCacheName;
            }

            if (logging)
            {
                std::ostringstream logfile;

                logfile<<logDirectory;
                if (deltaLevels)
                {
                    unsigned int nestedX = tileX / divisor;
                    unsigned int nestedY = tileY / divisor;
                    logfile<<basename<<"_subtile_L"<<nestedLevel<<"_X"<<nestedX<<"_Y"<<nestedY<<"/";

                    std::string path = logfile.str();
                    vpb::mkpath(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
                }
                logfile<<basename<<"_subtile_L"<<level<<"_X"<<tileX<<"_Y"<<tileY<<".log";
                app<<" --log "<<logfile.str();
            }

            taskManager->addTask(taskfile.str(), app.str(), sourceFile, getDatabaseRevisionBaseFileName(level,tileX,tileY));

            ++taskCount;
        }
    }


    return false;
}

const std::string DataSet::getDatabaseRevisionBaseFileName(unsigned int level, unsigned int x, unsigned y) const
{
    std::string baseName;
    std::stringstream sstr;
    if (level==0)
    {
        sstr << getDirectory() << getDestinationTileBaseName() << getDestinationTileExtension();
    }
    else
    {
        sstr << getDirectory()<<getTaskName(level, x, y)<< "/"
                <<getDestinationTileBaseName()
                << "_L"<<level
                << "_X" <<x
                <<"_Y"<<y<<getDestinationTileExtension();
    }

    sstr << ".task." << getRevisionNumber();

    return sstr.str();
}

int DataSet::run()
{
    if (!getLogFileName().empty() && !getBuildLog())
    {
        setBuildLog(new BuildLog(getLogFileName()));
    }

    if (getBuildLog())
    {
        pushOperationLog(getBuildLog());
    }

    if (!getWriteOptionsString().empty())
    {
        if (osgDB::Registry::instance()->getOptions()==0)
        {
            osgDB::Registry::instance()->setOptions(new osgDB::ReaderWriter::Options);
        }
        osgDB::Registry::instance()->getOptions()->setOptionString(getWriteOptionsString());
    }

    {
        _databaseRevision = new osgDB::DatabaseRevision;


        std::string baseName = getDatabaseRevisionBaseFileName(getSubtileLevel(), getSubtileX(), getSubtileY());

        {
            _databaseRevision->setFilesAdded(new osgDB::FileList);
            _databaseRevision->getFilesAdded()->setName(baseName+".added");
        }

        {
            _databaseRevision->setFilesRemoved(new osgDB::FileList);
            _databaseRevision->getFilesRemoved()->setName(baseName+".removed");
        }

        {
            _databaseRevision->setFilesModified(new osgDB::FileList);
            _databaseRevision->getFilesModified()->setName(baseName+".modified");
        }
    }

    int result = _run();

    if (_databaseRevision.valid())
    {
        log(osg::NOTICE, "Time to write out DatabaseRevision::FileList - FilesAdded %s, %d",_databaseRevision->getFilesAdded()->getName().c_str(), _databaseRevision->getFilesAdded()->getFileNames().size());
        log(osg::NOTICE, "Time to write out DatabaseRevision::FileList - FilesRemoved %s, %d",_databaseRevision->getFilesRemoved()->getName().c_str(), _databaseRevision->getFilesRemoved()->getFileNames().size());
        log(osg::NOTICE, "Time to write out DatabaseRevision::FileList - FilesModified %s, %d",_databaseRevision->getFilesModified()->getName().c_str(), _databaseRevision->getFilesModified()->getFileNames().size());

        if (_databaseRevision->getFilesAdded() && !_databaseRevision->getFilesAdded()->empty())
        {
            osgDB::writeObjectFile(*_databaseRevision->getFilesAdded(), _databaseRevision->getFilesAdded()->getName());
        }

        if (_databaseRevision->getFilesRemoved() && !_databaseRevision->getFilesRemoved()->empty())
        {
            osgDB::writeObjectFile(*_databaseRevision->getFilesRemoved(), _databaseRevision->getFilesRemoved()->getName());
        }

        if (_databaseRevision->getFilesModified() && !_databaseRevision->getFilesModified()->empty())
        {
            osgDB::writeObjectFile(*_databaseRevision->getFilesModified(), _databaseRevision->getFilesModified()->getName());
        }
    }

    if (getBuildLog())
    {
        popOperationLog();
    }


    return result;
}

int DataSet::_run()
{

    log(osg::NOTICE,"DataSet::_run() %i %i",getDistributedBuildSplitLevel(),getDistributedBuildSecondarySplitLevel());

    bool requiresGraphicsContextInMainThread = true;
    bool requiresGraphicsContextInWritingThread = true;

    osgDB::ImageProcessor* imageProcessor = osgDB::Registry::instance()->getImageProcessor();
    if (imageProcessor)
    {
        requiresGraphicsContextInMainThread = (getCompressionMethod() == vpb::BuildOptions::GL_DRIVER);
        requiresGraphicsContextInWritingThread = (getCompressionMethod() == vpb::BuildOptions::GL_DRIVER);
    }

    int numProcessors = OpenThreads::GetNumberOfProcessors();
#if 0
    if (numProcessors>1)
#endif
    {
        int numReadThreads = int(ceilf(getNumReadThreadsToCoresRatio() * float(numProcessors)));
        if (numReadThreads>=1)
        {
            log(osg::NOTICE,"Starting %i read threads.",numReadThreads);
            _readThreadPool = new ThreadPool(numReadThreads, false);
            _readThreadPool->startThreads();
        }

        int numWriteThreads = int(ceilf(getNumWriteThreadsToCoresRatio() * float(numProcessors)));
        if (numWriteThreads>=1)
        {
            log(osg::NOTICE,"Starting %i write threads.",numWriteThreads);
            _writeThreadPool = new ThreadPool(numWriteThreads, requiresGraphicsContextInWritingThread);
            _writeThreadPool->startThreads();

            //requiresGraphicsContextInMainThread = false;
        }
    }


    loadSources();

    int numLevels = getMaximumNumOfLevels();
    if (getRecordSubtileFileNamesOnLeafTile()) ++numLevels;

    createDestination(numLevels);

    if (!_destinationGraph)
    {
        log(osg::WARN, "Error: no destination graph built, cannot proceed with build.");
        return 1;
    }


    bool requiresGenerationOfTiles = getGenerateTiles();

    if (!getIntermediateBuildName().empty())
    {
        osg::ref_ptr<osgTerrain::TerrainTile> terrainTile = createTerrainRepresentation();
        if (terrainTile.valid())
        {
            DatabaseBuilder* db = dynamic_cast<DatabaseBuilder*>(terrainTile->getTerrainTechnique());
            if (db && db->getBuildOptions())
            {
                db->getBuildOptions()->setIntermediateBuildName("");
            }
            _writeNodeFile(*terrainTile,getIntermediateBuildName());
            requiresGenerationOfTiles = false;
        }
    }


    bool printOutContributingSources = true;
    if (printOutContributingSources)
    {
        CompositeDestination* startPoint = _destinationGraph.get();
        if (getGenerateSubtile())
        {
            startPoint = getComposite(getSubtileLevel(), getSubtileX(), getSubtileY());
        }

        if (startPoint)
        {
            DestinationTile::Sources sources = startPoint->getAllContributingSources();
            log(osg::NOTICE,"There are %d contributing source files:",sources.size());

            for(DestinationTile::Sources::iterator itr = sources.begin();
                itr != sources.end();
                ++itr)
            {
                log(osg::NOTICE,"    %s",(*itr)->getFileName().c_str());
            }
        }
        else
        {
            log(osg::NOTICE,"Warning: No destination graph generated.");
        }
    }

    if (requiresGenerationOfTiles)
    {
        // dummy Viewer to get round silly Windows autoregistration problem for GraphicsWindowWin32.cpp
        osgViewer::Viewer viewer;

        osg::ref_ptr<MyGraphicsContext> context;

        if (requiresGraphicsContextInMainThread)
        {
            context  = new MyGraphicsContext(getBuildLog());
            if (!context || !context->valid())
            {
                log(osg::NOTICE,"Error: Unable to create graphis context, problem with running osgViewer-%s, cannot run compression.",osgViewerGetVersion());
                return 1;
            }

            _state = context->_graphicsContext->getState();
        }

        int result = 0;
        osgDB::FileType type = osgDB::fileType(getDirectory());
        if (type==osgDB::DIRECTORY)
        {
            log(osg::NOTICE,"   Base Directory already created");
        }
        else if (type==osgDB::REGULAR_FILE)
        {
            log(osg::NOTICE,"   Error cannot create directory as a conventional file already exists with that name");
            return 1;
        }
        else // FILE_NOT_FOUND
        {
            // need to create directory.
            result = vpb::mkpath(getDirectory().c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
        }

        if (result)
        {
            log(osg::NOTICE,"Error: could not create directory, errorno=%i",errno);
            return 1;
        }


        if (getOutputTaskDirectories())
        {
            _taskOutputDirectory = getDirectory() + getTaskName(getSubtileLevel(), getSubtileX(), getSubtileY());
            log(osg::NOTICE,"Need to create output task directory = %s", _taskOutputDirectory.c_str());
            result = 0;
            type = osgDB::fileType(_taskOutputDirectory);
            if (type==osgDB::DIRECTORY)
            {
                log(osg::NOTICE,"   Directory already created");
            }
            else if (type==osgDB::REGULAR_FILE)
            {
                log(osg::NOTICE,"   Error cannot create directory as a conventional file already exists with that name");
                return 1;
            }
            else // FILE_NOT_FOUND
            {
                // need to create directory.
                result = vpb::mkpath(_taskOutputDirectory.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
            }

            if (result)
            {
                log(osg::NOTICE,"Error: could not create directory, errorno=%i",errno);
                return 1;
            }

#ifdef WIN32
            _taskOutputDirectory.push_back('\\');
#else
            _taskOutputDirectory.push_back('/');
#endif

            if (getGenerateSubtile()) log(osg::NOTICE,"We are a subtile");
            if (getRecordSubtileFileNamesOnLeafTile()) log(osg::NOTICE,"We have to record ../task/name");
        }
        else
        {
            _taskOutputDirectory = getDirectory();
        }

        log(osg::NOTICE,"Task output directory = %s", _taskOutputDirectory.c_str());

        writeDestination();
    }

    return 0;
}

std::string DataSet::checkBuildValidity()
{
    bool isTerrain = getGeometryType()==BuildOptions::TERRAIN;
    bool containsOptionalLayers = !(getOptionalLayerSet().empty());

    if (containsOptionalLayers && !isTerrain) return std::string("Can not mix optional layers with POLYGONAL and HEIGHTFIELD builds, must use --terrain to enable optional layer support.");


    return std::string();
}
