/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "DecoratedGeometryTechnique.h"

#include <osgTerrain/Terrain>

#include <osgUtil/SmoothingVisitor>

#include <osgDB/FileUtils>

#include <osg/io_utils>
#include <osg/Texture2D>
#include <osg/Texture1D>
#include <osg/TexEnvCombine>
#include <osg/Program>
#include <osg/Math>
#include <osg/Timer>

#include <osgSim/ShapeAttribute>

#include "mtrand.h"
#include <VehicleUtil/RoadSystem/RoadSystem.h>

using namespace osgTerrain;

void DecoratedGeometryTechnique::traverse(osg::NodeVisitor &nv)
{
    if (!_terrainTile)
        return;

    // if app traversal update the frame count.
    if (nv.getVisitorType() == osg::NodeVisitor::UPDATE_VISITOR)
    {
        //if (_terrainTile->getDirty()) _terrainTile->init(_terrainTile->getDirtyMask(), false);
        if (_terrainTile->getDirty())
            _terrainTile->init(_terrainTile->getDirtyMask(), true);

        osgUtil::UpdateVisitor *uv = dynamic_cast<osgUtil::UpdateVisitor *>(&nv);
        if (uv)
        {
            update(uv);
            return;
        }
    }
    else if (nv.getVisitorType() == osg::NodeVisitor::CULL_VISITOR)
    {
        osgUtil::CullVisitor *cv = dynamic_cast<osgUtil::CullVisitor *>(&nv);
        if (cv)
        {
            cull(cv);
            return;
        }
    }

    if (_terrainTile->getDirty())
    {
        OSG_INFO << "******* Doing init ***********" << std::endl;
        //_terrainTile->init(_terrainTile->getDirtyMask(), false);
        _terrainTile->init(_terrainTile->getDirtyMask(), true);
    }

    if (_currentBufferData.valid())
    {
        if (_currentBufferData->_transform.valid())
            _currentBufferData->_transform->accept(nv);
    }
}

void DecoratedGeometryTechnique::update(osgUtil::UpdateVisitor *uv)
{
    if (_terrainTile)
        _terrainTile->osg::Group::traverse(*uv);

    /*if (_newBufferData.valid())
    {
        _currentBufferData = _newBufferData;
        _newBufferData = 0;
    }*/

    if (_newBufferData.valid())
    {
        if (!_currentBufferData)
        {
            // no currentBufferData so we must be the first init to be applied
            _currentBufferData = _newBufferData;
            _newBufferData = 0;
        }
        else
        {
            // there is already an active _currentBufferData so we'll request that this gets swapped on next frame.
            if (_terrainTile->getTerrain())
                _terrainTile->getTerrain()->updateTerrainTileOnNextFrame(_terrainTile);
        }
    }
}

#if OPENSCENEGRAPH_SOVERSION < 72
void DecoratedGeometryTechnique::generateGeometry(osgTerrain::Locator *masterLocator, const osg::Vec3d &centerModel)
{
    BufferData &buffer = getWriteBuffer();
#else
void DecoratedGeometryTechnique::generateGeometry(BufferData &buffer, osgTerrain::Locator *masterLocator, const osg::Vec3d &centerModel)
{
#endif

    osgTerrain::Layer *elevationLayer = _terrainTile->getElevationLayer();

    buffer._geode = new osg::Geode;
    buffer._geode->setName("HeightFieldGeode");
    if (buffer._transform.valid())
        buffer._transform->addChild(buffer._geode.get());

    buffer._geometry = new osg::Geometry;
    buffer._geode->addDrawable(buffer._geometry.get());

    osg::Geometry *geometry = buffer._geometry.get();

    unsigned int numRows = 20;
    unsigned int numColumns = 20;

    if (elevationLayer)
    {
        numColumns = elevationLayer->getNumColumns();
        numRows = elevationLayer->getNumRows();
    }

    float sampleRatio = _terrainTile->getTerrain() ? _terrainTile->getTerrain()->getSampleRatio() : 1.0f;

    double i_sampleFactor = 1.0;
    double j_sampleFactor = 1.0;

    // OSG_NOTIFY(osg::NOTICE)<<"Sample ratio="<<sampleRatio<<std::endl;

    if (sampleRatio != 1.0f)
    {

        unsigned int originalNumColumns = numColumns;
        unsigned int originalNumRows = numRows;

        numColumns = std::max((unsigned int)(float(originalNumColumns) * sqrtf(sampleRatio)), 4u);
        numRows = std::max((unsigned int)(float(originalNumRows) * sqrtf(sampleRatio)), 4u);

        i_sampleFactor = double(originalNumColumns - 1) / double(numColumns - 1);
        j_sampleFactor = double(originalNumRows - 1) / double(numRows - 1);
    }

    bool treatBoundariesToValidDataAsDefaultValue = _terrainTile->getTreatBoundariesToValidDataAsDefaultValue();
    //OSG_NOTIFY(osg::INFO)<<"TreatBoundariesToValidDataAsDefaultValue="<<treatBoundariesToValidDataAsDefaultValue<<std::endl;

    float skirtHeight = 0.0f;
    HeightFieldLayer *hfl = dynamic_cast<HeightFieldLayer *>(elevationLayer);
    if (hfl && hfl->getHeightField())
    {
        skirtHeight = hfl->getHeightField()->getSkirtHeight();
    }

    bool createSkirt = skirtHeight != 0.0f;

    unsigned int numVerticesInBody = numColumns * numRows;
    unsigned int numVerticesInSkirt = createSkirt ? numColumns * 2 + numRows * 2 - 4 : 0;
    unsigned int numVertices = numVerticesInBody + numVerticesInSkirt;

    //road
    const TileID &tileId = _terrainTile->getTileID();
    std::map<int, std::map<int, double> > fillBorderMap;

    const RoadSystemHeader &roadSysHeader = RoadSystem::Instance()->getHeader();
    osg::Vec3d roadOffset(roadSysHeader.xoffset, roadSysHeader.yoffset, 0.0);
    osg::Vec3d roadSysToCenterModel = roadOffset - centerModel;

    osg::Vec3d lower_left_tile_road;
    masterLocator->convertLocalToModel(osg::Vec3(0.0, 0.0, 0.0), lower_left_tile_road);
    lower_left_tile_road -= roadOffset;

    osg::Vec3d upper_right_tile_road;
    masterLocator->convertLocalToModel(osg::Vec3(1.0, 1.0, 0.0), upper_right_tile_road);
    upper_right_tile_road -= roadOffset;

#if 0
    if(tileId.level >= 7) {
       osg::Geometry* borderGeometry = new osg::Geometry;
       buffer._geode->addDrawable(borderGeometry);
       osg::Vec3Array* borderVerts = new osg::Vec3Array;
       borderGeometry->setVertexArray(borderVerts);
       int borderVertIt = 0;

       for(int i=0; i<RoadSystem::Instance()->getNumRoads(); ++i) {
          Road* road = RoadSystem::Instance()->getRoad(i);
          double roadLength = road->getLength();
          osg::Geode* roadGeode = road->getRoadGeode();
          if(roadGeode->getNumDrawables()<1) continue;
          osg::Geometry* roadGeometry = dynamic_cast<osg::Geometry*>(roadGeode->getDrawable(0));
          if(!roadGeometry) continue;
          if(roadGeometry->getNumPrimitiveSets()<4) continue;

          osg::BoundingBox roadBB = roadGeode->getBoundingBox();
          if(((roadBB.xMax()>=lower_left_tile_road.x() && roadBB.xMax()<=upper_right_tile_road.x()) ||
                   (roadBB.xMin()>=lower_left_tile_road.x() && roadBB.xMin()<=upper_right_tile_road.x()) ||
                   (roadBB.xMin()<=lower_left_tile_road.x() && roadBB.xMax()>=upper_right_tile_road.x())) &&
                (roadBB.yMax()>=lower_left_tile_road.y() && roadBB.yMin()<=upper_right_tile_road.y()))
          {
             osg::DrawArrays* roadBase = dynamic_cast<osg::DrawArrays*>(roadGeometry->getPrimitiveSet(0));
             if(!roadBase) continue;

             int first = roadBase->getFirst();
             int count = roadBase->getCount();

             if(std::string(roadGeometry->getVertexArray()->className())=="Vec3Array") {
                osg::Vec3Array* vertexArray = static_cast<osg::Vec3Array*>(roadGeometry->getVertexArray());


                for(int vertIt = first; vertIt<first+count-2; vertIt += 2) {
 

                   osg::Vec3 aftRoadPoint = (*vertexArray)[vertIt];

                   double aft_n_x = (aftRoadPoint.x()-lower_left_tile_road.x())/(upper_right_tile_road.x()-lower_left_tile_road.x());
                   double aft_n_y = (aftRoadPoint.y()-lower_left_tile_road.y())/(upper_right_tile_road.y()-lower_left_tile_road.y());

                   osg::Vec3 foreRoadPoint = (*vertexArray)[vertIt+2];
                   double fore_n_x = (foreRoadPoint.x()-lower_left_tile_road.x())/(upper_right_tile_road.x()-lower_left_tile_road.x());
                   double fore_n_y = (foreRoadPoint.y()-lower_left_tile_road.y())/(upper_right_tile_road.y()-lower_left_tile_road.y());

                   if((aft_n_x<0.0 && fore_n_x<0.0) || (aft_n_x>1.0 && fore_n_x>1.0)) continue;
                   if((aft_n_y<0.0 && fore_n_y<0.0) || (aft_n_y>1.0 && fore_n_y>1.0)) continue;

                   //Cut to tile boundary
                   if(aft_n_x<0.0) {
                     double s = aft_n_x/(aft_n_x-fore_n_x);
                     aft_n_x = 0.0;
                     aft_n_y = aft_n_y + s*(fore_n_y-aft_n_y);
                   }
                   if(aft_n_y<0.0) {
                     double s = aft_n_y/(aft_n_y-fore_n_y);
                     aft_n_y = 0.0;
                     aft_n_x = aft_n_x + s*(fore_n_x-aft_n_x);
                   }
                   if(fore_n_x>1.0) {
                     double s = (fore_n_x-1.0)/(fore_n_x-aft_n_x);
                     fore_n_x = 1.0;
                     fore_n_y = fore_n_y - s*(fore_n_y-aft_n_y);
                   }
                   if(fore_n_y>1.0) {
                     double s = (fore_n_y-1.0)/(fore_n_y-aft_n_y);
                     fore_n_y = 1.0;
                     fore_n_x = fore_n_x - s*(fore_n_x-aft_n_x);
                   }

                   int aft_cell_x = (int)floor(aft_n_x*(double)(elevationLayer->getNumRows()-1));
                   int aft_cell_y = (int)floor(aft_n_y*(double)(elevationLayer->getNumColumns()-1));

                   int fore_cell_x = (int)floor(fore_n_x*(double)(elevationLayer->getNumRows()-1));
                   int fore_cell_y = (int)floor(fore_n_y*(double)(elevationLayer->getNumColumns()-1));
                   //std::cout << "aft: n_x: " << aft_n_x << ", n_y: " << aft_n_y << ", cell_x: " << aft_cell_x << ", cell_y: " << aft_cell_y << std::endl;
                   //std::cout << "fore: n_x: " << fore_n_x << ", n_y: " << fore_n_y << ", cell_x: " << fore_cell_x << ", cell_y: " << fore_cell_y << std::endl;

                   std::list<std::pair<int,int> > cellList;
                   std::list<std::pair<int,int> > outerVertList;
                   std::list<std::pair<int,int> > innerVertList;
                   //Bresenham
                   int x0 = aft_cell_x, y0 = aft_cell_y;
                   int x1 = fore_cell_x, y1 = fore_cell_y;
                   int dx = abs(x1 - x0);
                   int dy = abs(y1 - y0); 
                   int sx, cx; if(x0 < x1) {sx = 1; cx=1;} else {sx = -1; cx=0;}
                   int sy, cy; if(y0 < y1) {sy = 1; cy=1;} else {sy = -1; cy=0;}
                   int err = dx-dy;

                   do {
                      cellList.push_back(std::make_pair(x0,y0));

                      int e2 = 2*err;
                      
                      if(e2 > -dy && e2 < dx) {
                        outerVertList.push_back(std::make_pair(x0+cx, y0+cy));
                        innerVertList.push_back(std::make_pair(x0+cx, y0));
                        innerVertList.push_back(std::make_pair(x0+2*cx, y0+cy));
                      }
                      else if(e2 > -dy) {
                        outerVertList.push_back(std::make_pair(x0+cx, y0+cy));
                        innerVertList.push_back(std::make_pair(x0+cx, y0));
                      }
                      else if(e2 < dx) {
                        outerVertList.push_back(std::make_pair(cx, y0+cy));
                        innerVertList.push_back(std::make_pair(x0+cx, y0+cy));
                      }

                      if(e2 > -dy) {
                         err = err - dy;
                         x0 = x0 + sx;
                      }
                      if(e2 < dx) {
                         err = err + dx;
                         y0 = y0 + sy;
                      }
                   }
                   while(!(x0==x1 && y0==y1));
                   cellList.push_back(std::make_pair(x1,y1));

#if 0
                   int lastBorderVertIt = borderVertIt;
                   borderVerts->push_back(foreRoadPoint + roadSysToCenterModel); ++borderVertIt;
                   borderVerts->push_back(aftRoadPoint + roadSysToCenterModel); ++borderVertIt;
                   for(std::list<std::pair<int,int> >::iterator listIt = outerVertList.begin(); listIt != outerVertList.end(); ++listIt) {
                      osg::Vec3d cellVert;
                      osg::Vec3d n_cellVert(
                                 ((double)(listIt->first)/(double)(elevationLayer->getNumRows()-1)),
                                 ((double)(listIt->second)/(double)(elevationLayer->getNumColumns()-1)),
                                 0.0);
                      masterLocator->convertLocalToModel(n_cellVert, cellVert);

                      if(elevationLayer) {
                         float z = 0.0;
                         elevationLayer->getValidValue(listIt->first, listIt->second, z);
                         cellVert.z() = z;
                      }

                      borderVerts->push_back(cellVert - centerModel); ++borderVertIt;
                   }

                   osg::DrawArrays* borderBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_FAN, lastBorderVertIt, borderVertIt-lastBorderVertIt);
                   borderGeometry->addPrimitiveSet(borderBase);
#endif
                   for(std::list<std::pair<int,int> >::iterator listIt = innerVertList.begin(); listIt != innerVertList.end(); ++listIt) {
                     fillBorderMap[listIt->second][listIt->first] = 0.0;
                   }

                  
                   if(outerVertList.size() > 1) {
                      osg::Vec3d u = aftRoadPoint + roadSysToCenterModel;
                      osg::Vec3d v = (foreRoadPoint + roadSysToCenterModel) - u;

                      osg::Vec3d borderVert;
                      osg::Vec3d cellVert;
                      int lastBorderVertIt = borderVertIt;

                      for(std::list<std::pair<int,int> >::iterator listIt = outerVertList.begin(); listIt != outerVertList.end(); ++listIt) {
                         osg::Vec3d n_cellVert(
                               ((double)(listIt->first)/(double)(elevationLayer->getNumRows()-1)),
                               ((double)(listIt->second)/(double)(elevationLayer->getNumColumns()-1)),
                               0.0);
                         masterLocator->convertLocalToModel(n_cellVert, cellVert);

                         if(elevationLayer) {
                            float z = 0.0;
                            elevationLayer->getValidValue(listIt->first, listIt->second, z);
                            cellVert.z() = z;
                         }

                         cellVert -= centerModel;

                         if(listIt == outerVertList.begin()) {
                           borderVert = u;
                         }
                         else if(listIt == --(outerVertList.end())) {
                           borderVert = u+v;
                         }
                         else {
                            double s = ((cellVert-u)*v)/(v*v);
                            borderVert = u + v*s;
                         }

                         borderVerts->push_back(cellVert); ++borderVertIt;
                         borderVerts->push_back(borderVert); ++borderVertIt;
                      }
                   }
                }
                //osg::DrawArrays* borderBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, lastBorderVertIt, borderVertIt-lastBorderVertIt);
                osg::DrawArrays* borderBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, borderVertIt);
                borderGeometry->addPrimitiveSet(borderBase);
             }
          }
       }
    }
#endif

#if 0
      if(tileId.level >= 7) {
       osgTerrain::HeightFieldLayer* hfl = dynamic_cast<HeightFieldLayer*>(elevationLayer);
       osg::HeightField* field = hfl->getHeightField();

       double tileXInterval = field->getXInterval();
       double tileYInterval = field->getYInterval();
       double tileElementArea = tileXInterval*tileYInterval;
       double tileXMin = field->getOrigin().x() - centerModel.x();
       double tileXMax = tileXMin + tileXInterval*(double)field->getNumColumns();
       double tileYMin = field->getOrigin().y() - centerModel.y();
       double tileYMax = tileYMin + tileYInterval*(double)field->getNumRows();

       for(int i=0; i<RoadSystem::Instance()->getNumRoads(); ++i) {
          Road* road = RoadSystem::Instance()->getRoad(i);
          double roadLength = road->getLength();
          osg::BoundingBox roadBB = road->getRoadGeode()->getBoundingBox();
          if(((roadBB.xMax()>=lower_left_tile_road.x() && roadBB.xMax()<=upper_right_tile_road.x()) ||
                   (roadBB.xMin()>=lower_left_tile_road.x() && roadBB.xMin()<=upper_right_tile_road.x()) ||
                   (roadBB.xMin()<=lower_left_tile_road.x() && roadBB.xMax()>=upper_right_tile_road.x())) &&
                (roadBB.yMax()>=lower_left_tile_road.y() && roadBB.yMin()<=upper_right_tile_road.y())) {
             //double h = sqrt(tileXInterval*tileXInterval + tileYInterval*tileYInterval);
             double h = 20.0;
             std::deque<RoadPoint> pointDeque(4);
             double widthRight, widthLeft;
             road->getRoadSideWidths(0.0, widthRight, widthLeft);
             pointDeque.pop_front();
             pointDeque.pop_front();
             pointDeque.push_back(road->getRoadPoint(0.0, widthLeft+10.0));
             pointDeque.push_back(road->getRoadPoint(0.0, widthRight-10.0));

             for(double s_ = h; s_<roadLength+h; s_ = s_+h) {
                double s = s_;
                if(s>roadLength) s=roadLength;

                road->getRoadSideWidths(s, widthRight, widthLeft);
                pointDeque.pop_front();
                pointDeque.pop_front();
                pointDeque.push_back(road->getRoadPoint(s, widthLeft+10.0));
                pointDeque.push_back(road->getRoadPoint(s, widthRight-10.0));

                //Bresenham
                //         y            x      z
                //iteration over four lines
                for(int i=0; i<4; ++i) {
                   int j;
                   switch(i) {
                      case 0: j=1; break;
                      case 1: j=3; break;
                      case 2: j=0; break;
                      case 3: j=2; break;
                   }

                   double tileXInterval = (upper_right_tile_road.x() - lower_left_tile_road.x())/(double)(elevationLayer->getNumRows());
                   double tileYInterval = (upper_right_tile_road.y() - lower_left_tile_road.y())/(double)(elevationLayer->getNumColumns());

                   int x0 = floor((pointDeque[i].x()-lower_left_tile_road.x())/tileXInterval+0.5);
                   //if(x0<0) x0=0; else if(x0>=field->getNumColumns()) x0=field->getNumColumns()-1;
                   int y0 = floor((pointDeque[i].y()-lower_left_tile_road.y())/tileYInterval+0.5);
                   //if(y0<0) y0=0; else if(y0>=field->getNumRows()) y0=field->getNumRows()-1;
                   int x1 = floor((pointDeque[j].x()-lower_left_tile_road.x())/tileXInterval+0.5);
                   //if(x1<0) x1=0; else if(x1>=field->getNumColumns()) x1=field->getNumColumns()-1;
                   int y1 = floor((pointDeque[j].y()-lower_left_tile_road.y())/tileYInterval+0.5);
                   //if(y1<0) y1=0; else if(y1>=field->getNumRows()) y1=field->getNumRows()-1;

                   double z = 0.5*(pointDeque[i].z()+pointDeque[j].z());

                   //wikipedia implementation
                   int dx =  abs(x1-x0);
                   int sx = x0<x1 ? 1 : -1;
                   int dy = -abs(y1-y0);
                   int sy = y0<y1 ? 1 : -1; 
                   int err = dx+dy;
                   int e2; /* error value e_xy */

                   if(!(x0==x1 && y0==y1)) do
                   {  /* loop */
                      //setPixel(x0,y0);
                      fillBorderMap[y0][x0] = z;
                      //std::cout << "Filling y=" << y0 << ", x=" << x0 << ", z=" << z << std::endl;
                      //fillBorderMap[y0][x0] = 0.5;
                      //if (x0==x1 && y0==y1) break;
                      e2 = 2*err;
                      if (e2 >= dy) { err += dy; x0 += sx; } /* e_xy+e_x > 0 */
                      if (e2 <= dx) { err += dx; y0 += sy; } /* e_xy+e_y < 0 */
                   }
                   while(!(x0==x1 && y0==y1));
                }

                for(std::map<int, std::map<int, double> >::iterator yScanlineIt = fillBorderMap.begin();
                      yScanlineIt!=fillBorderMap.end(); ++yScanlineIt)
                {
                   int x0 = (yScanlineIt->second.begin())->first;
                   double z0 = (yScanlineIt->second.begin())->second;
                   int x1 = (--(yScanlineIt->second.end()))->first;
                   double z1 = (--(yScanlineIt->second.end()))->second;
                   unsigned int y = yScanlineIt->first;
                   //if(y==field->getNumRows()-1) continue;

                   for(int x = x0; x < x1; ++x) {
                      if(x>=0 && x<(int)elevationLayer->getNumColumns() && y>=0 && y<elevationLayer->getNumRows()) {
                         //field->setHeight(x,y,0.5*(z0+z1)-0.5);
                         //double z = field->getHeight(x,y);
                         double rl_x = 0.5*(pointDeque[0].x()+pointDeque[1].x());
                         double rl_y = 0.5*(pointDeque[0].y()+pointDeque[1].y());
                         double ru_x = 0.5*(pointDeque[2].x()+pointDeque[3].x());
                         double ru_y = 0.5*(pointDeque[2].y()+pointDeque[3].y());
                         double t_x = ru_x-rl_x;
                         double t_y = ru_y-rl_y;
                         double mag_t = sqrt(t_x*t_x+t_y*t_y);
                         if(mag_t!=mag_t) continue;
                         t_x /= mag_t; t_y /= mag_t;
                         double rp_x = tileXInterval*((double)(x)) + tileXMin - rl_x;
                         double rp_y = tileYInterval*((double)(y)) + tileYMin - rl_y;

                         double d_x = -t_y*t_x*rp_y + t_y*t_y*rp_x;
                         double d_y = t_x*t_x*rp_y - t_x*t_y*rp_x;
                         double d = sqrt(d_x*d_x + d_y*d_y);
                         //double mag_d = sqrt(d_x*d_x + d_y*d_y);
                         //double d = 0.0;
                         //if(mag_d==mag_d) {
                         //   d = (d_x*d_x+d_y*d_y)/mag_d;
                         //}
                         double s_x = rp_x - d_x;
                         double s_y = rp_y - d_y;
                         double s = sqrt(s_x*s_x + s_y*s_y);

                         double wl = sqrt(pow(pointDeque[0].x()-pointDeque[1].x(),2) + pow(pointDeque[0].y()-pointDeque[1].y(),2));
                         double wu = sqrt(pow(pointDeque[2].x()-pointDeque[3].x(),2) + pow(pointDeque[2].y()-pointDeque[3].y(),2));
                         double l = sqrt(pow(0.5*(pointDeque[0].x()-pointDeque[2].x()+pointDeque[1].x()-pointDeque[3].x()),2) + pow(0.5*(pointDeque[0].y()-pointDeque[2].y()+pointDeque[1].y()-pointDeque[3].y()),2));

                         /*double zt;
                           std::map<std::pair<int,int>,double>::iterator coordHeightMapIt = coordHeightMap.find(std::make_pair(x,y));
                           if(coordHeightMapIt == coordHeightMap.end()) {
                           zt = (double)field->getHeight(x,y);
                           coordHeightMap[std::make_pair(x,y)] = zt;
                           }
                           else {
                           zt = coordHeightMapIt->second;
                           }*/
                           float zt;
                           elevationLayer->getValidValue(x,y, zt);

                         double zf0 = 5.0;
                         double zf = 0.5*(z0+z1) - zf0;
                         double kt0 = 50.0;
                         double kt = (d/(0.5*(wl+(wu-wl)*s/l)))*kt0;
                         double kf0 = 10.0;
                         double kf = (1.0 - d/(0.5*(wl+(wu-wl)*s/l)))*kf0;

                         double zl = (zt*kt + zf*kf)/(kt+kf);
                         //field->setHeight(x,y, zl);
                         (yScanlineIt->second)[x] = zl;
                      }
                   }
                }
             }
          }
       }
    }
#endif
    //end road

    // allocate and assign vertices
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
    vertices->reserve(numVertices);
    geometry->setVertexArray(vertices.get());

    // allocate and assign normals
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
    if (normals.valid())
        normals->reserve(numVertices);
    geometry->setNormalArray(normals.get());
    geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    //float minHeight = 0.0;
    float scaleHeight = _terrainTile->getTerrain() ? _terrainTile->getTerrain()->getVerticalScale() : 1.0f;

    // allocate and assign tex coords
    typedef std::pair<osg::ref_ptr<osg::Vec2Array>, Locator *> TexCoordLocatorPair;
    typedef std::map<Layer *, TexCoordLocatorPair> LayerToTexCoordMap;

    LayerToTexCoordMap layerToTexCoordMap;
    for (unsigned int layerNum = 0; layerNum < _terrainTile->getNumColorLayers(); ++layerNum)
    {
        //Layer number one for land usage map
        /*if(layerNum==1) {
            osgTerrain::Layer* colorLayer = _terrainTile->getColorLayer(layerNum);
            if(colorLayer) delete colorLayer;
            continue;
        }*/

        osgTerrain::Layer *colorLayer = _terrainTile->getColorLayer(layerNum);
        if (colorLayer)
        {
            LayerToTexCoordMap::iterator itr = layerToTexCoordMap.find(colorLayer);
            if (itr != layerToTexCoordMap.end())
            {
                geometry->setTexCoordArray(layerNum, itr->second.first.get());
            }
            else
            {

                Locator *locator = colorLayer->getLocator();
                if (!locator)
                {
                    osgTerrain::SwitchLayer *switchLayer = dynamic_cast<osgTerrain::SwitchLayer *>(colorLayer);
                    if (switchLayer)
                    {
                        if (switchLayer->getActiveLayer() >= 0 && static_cast<unsigned int>(switchLayer->getActiveLayer()) < switchLayer->getNumLayers() && switchLayer->getLayer(switchLayer->getActiveLayer()))
                        {
                            locator = switchLayer->getLayer(switchLayer->getActiveLayer())->getLocator();
                        }
                    }
                }

                TexCoordLocatorPair &tclp = layerToTexCoordMap[colorLayer];
                tclp.first = new osg::Vec2Array;
                tclp.first->reserve(numVertices);
                tclp.second = locator ? locator : masterLocator;
                geometry->setTexCoordArray(layerNum, tclp.first.get());
            }
        }
    }

    osg::ref_ptr<osg::FloatArray> elevations = new osg::FloatArray;
    if (elevations.valid())
        elevations->reserve(numVertices);

    // allocate and assign color
    osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array(1);
    (*colors)[0].set(1.0f, 1.0f, 1.0f, 1.0f);

    geometry->setColorArray(colors.get());
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

    typedef std::vector<int> Indices;
    Indices indices(numVertices, -1);

    // populate vertex and tex coord arrays
    unsigned int i, j;
    for (j = 0; j < numRows; ++j)
    {
        for (i = 0; i < numColumns; ++i)
        {
            unsigned int iv = j * numColumns + i;
            osg::Vec3d ndc(((double)i) / (double)(numColumns - 1), ((double)j) / (double)(numRows - 1), 0.0);

            bool validValue = true;

            unsigned int i_equiv = i_sampleFactor == 1.0 ? i : (unsigned int)(double(i) * i_sampleFactor);
            unsigned int j_equiv = i_sampleFactor == 1.0 ? j : (unsigned int)(double(j) * j_sampleFactor);

            if (elevationLayer)
            {
                float value = 0.0f;
                validValue = elevationLayer->getValidValue(i_equiv, j_equiv, value);
                // OSG_NOTIFY(osg::INFO)<<"i="<<i<<" j="<<j<<" z="<<value<<std::endl;
                ndc.z() = value * scaleHeight;

                std::map<int, std::map<int, double> >::iterator fillBorderMapIt = fillBorderMap.find(j_equiv);
                if (fillBorderMapIt != fillBorderMap.end())
                {
                    std::map<int, double>::iterator xzMapIt = fillBorderMapIt->second.find(i_equiv);
                    if (xzMapIt != fillBorderMapIt->second.end())
                    {
                        ndc.z() = xzMapIt->second;
                        //validValue = false;
                        std::cout << "Changing height in tile L_" << tileId.level << "_X" << tileId.x << "_Y" << tileId.y << ": x=" << i_equiv << ", y=" << j_equiv << " to z=" << ndc.z() << std::endl;
                    }
                }
            }

            /*if(i>=0 && i<10) {
               validValue = false;
            }*/
            osg::Vec3d model;
            masterLocator->convertLocalToModel(ndc, model);

            for (int rectIt = 0; rectIt < voidBoundingAreaVector.size(); ++rectIt)
            {
                if (voidBoundingAreaVector[rectIt].contains(osg::Vec2d(model.x(), model.y())))
                {
                    //std::cout << "Checking rect " << rectIt << ": (" << voidBoundingAreaVector[rectIt]._min.x() << ", " << voidBoundingAreaVector[rectIt]._min.y() << ") - (" <<  voidBoundingAreaVector[rectIt]._max.x() << ", " << voidBoundingAreaVector[rectIt]._max.y() << ") for (" << model.x() << ", " << model.y() << ")" << std::endl;
                    validValue = false;
                    break;
                }
            }

            if (validValue)
            {
                indices[iv] = vertices->size();

                //osg::Vec3d model;
                //masterLocator->convertLocalToModel(ndc, model);

                (*vertices).push_back(model - centerModel);

                for (LayerToTexCoordMap::iterator itr = layerToTexCoordMap.begin();
                     itr != layerToTexCoordMap.end();
                     ++itr)
                {
                    osg::Vec2Array *texcoords = itr->second.first.get();
                    Locator *colorLocator = itr->second.second;
                    if (colorLocator != masterLocator)
                    {
                        osg::Vec3d color_ndc;
                        Locator::convertLocalCoordBetween(*masterLocator, ndc, *colorLocator, color_ndc);
                        (*texcoords).push_back(osg::Vec2(color_ndc.x(), color_ndc.y()));
                    }
                    else
                    {
                        (*texcoords).push_back(osg::Vec2(ndc.x(), ndc.y()));
                    }
                }

                if (elevations.valid())
                {
                    (*elevations).push_back(ndc.z());
                }

                // compute the local normal
                osg::Vec3d ndc_one = ndc;
                ndc_one.z() += 1.0;
                osg::Vec3d model_one;
                masterLocator->convertLocalToModel(ndc_one, model_one);
                model_one = model_one - model;
                model_one.normalize();
                (*normals).push_back(model_one);
            }
            else
            {
                indices[iv] = -1;
            }
        }
    }

    // populate primitive sets
    //    bool optimizeOrientations = elevations!=0;
    bool swapOrientation = !(masterLocator->orientationOpenGL());

    bool smallTile = numVertices <= 16384;

    // OSG_NOTIFY(osg::NOTICE)<<"smallTile = "<<smallTile<<std::endl;

    osg::ref_ptr<osg::DrawElements> elements = smallTile ? static_cast<osg::DrawElements *>(new osg::DrawElementsUShort(GL_TRIANGLES)) : static_cast<osg::DrawElements *>(new osg::DrawElementsUInt(GL_TRIANGLES));

    elements->reserveElements((numRows - 1) * (numColumns - 1) * 6);

    geometry->addPrimitiveSet(elements.get());

    for (j = 0; j < numRows - 1; ++j)
    {
        for (i = 0; i < numColumns - 1; ++i)
        {
            int i00;
            int i01;
            if (swapOrientation)
            {
                i01 = j * numColumns + i;
                i00 = i01 + numColumns;
            }
            else
            {
                i00 = j * numColumns + i;
                i01 = i00 + numColumns;
            }

            int i10 = i00 + 1;
            int i11 = i01 + 1;

            // remap indices to final vertex positions
            i00 = indices[i00];
            i01 = indices[i01];
            i10 = indices[i10];
            i11 = indices[i11];

            unsigned int numValid = 0;
            if (i00 >= 0)
                ++numValid;
            if (i01 >= 0)
                ++numValid;
            if (i10 >= 0)
                ++numValid;
            if (i11 >= 0)
                ++numValid;

            if (numValid == 4)
            {
                float e00 = (*elevations)[i00];
                float e10 = (*elevations)[i10];
                float e01 = (*elevations)[i01];
                float e11 = (*elevations)[i11];

                if (fabsf(e00 - e11) < fabsf(e01 - e10))
                {
                    elements->addElement(i01);
                    elements->addElement(i00);
                    elements->addElement(i11);

                    elements->addElement(i00);
                    elements->addElement(i10);
                    elements->addElement(i11);
                }
                else
                {
                    elements->addElement(i01);
                    elements->addElement(i00);
                    elements->addElement(i10);

                    elements->addElement(i01);
                    elements->addElement(i10);
                    elements->addElement(i11);
                }
            }
            else if (numValid == 3)
            {
                if (i00 >= 0)
                    elements->addElement(i00);
                if (i01 >= 0)
                    elements->addElement(i01);
                if (i11 >= 0)
                    elements->addElement(i11);
                if (i10 >= 0)
                    elements->addElement(i10);
            }
        }
    }

    osg::ref_ptr<osg::Vec3Array> skirtVectors = new osg::Vec3Array((*normals));

    if (elevationLayer)
    {
        //       smoothGeometry();

        normals = dynamic_cast<osg::Vec3Array *>(geometry->getNormalArray());

        if (!normals)
            createSkirt = false;
    }

    if (createSkirt)
    {
        osg::ref_ptr<osg::DrawElements> skirtDrawElements = smallTile ? static_cast<osg::DrawElements *>(new osg::DrawElementsUShort(GL_QUAD_STRIP)) : static_cast<osg::DrawElements *>(new osg::DrawElementsUInt(GL_QUAD_STRIP));

        // create bottom skirt vertices
        int r, c;
        r = 0;
        for (c = 0; c < static_cast<int>(numColumns); ++c)
        {
            int orig_i = indices[(r)*numColumns + c]; // index of original vertex of grid
            if (orig_i >= 0)
            {
                unsigned int new_i = vertices->size(); // index of new index of added skirt point
                osg::Vec3 new_v = (*vertices)[orig_i] - ((*skirtVectors)[orig_i]) * skirtHeight;
                (*vertices).push_back(new_v);
                if (normals.valid())
                    (*normals).push_back((*normals)[orig_i]);

                for (LayerToTexCoordMap::iterator itr = layerToTexCoordMap.begin();
                     itr != layerToTexCoordMap.end();
                     ++itr)
                {
                    itr->second.first->push_back((*itr->second.first)[orig_i]);
                }

                skirtDrawElements->addElement(orig_i);
                skirtDrawElements->addElement(new_i);
            }
            else
            {
                if (skirtDrawElements->getNumIndices() != 0)
                {
                    geometry->addPrimitiveSet(skirtDrawElements.get());
                    skirtDrawElements = smallTile ? static_cast<osg::DrawElements *>(new osg::DrawElementsUShort(GL_QUAD_STRIP)) : static_cast<osg::DrawElements *>(new osg::DrawElementsUInt(GL_QUAD_STRIP));
                }
            }
        }

        if (skirtDrawElements->getNumIndices() != 0)
        {
            geometry->addPrimitiveSet(skirtDrawElements.get());
            skirtDrawElements = smallTile ? static_cast<osg::DrawElements *>(new osg::DrawElementsUShort(GL_QUAD_STRIP)) : static_cast<osg::DrawElements *>(new osg::DrawElementsUInt(GL_QUAD_STRIP));
        }

        // create right skirt vertices
        c = numColumns - 1;
        for (r = 0; r < static_cast<int>(numRows); ++r)
        {
            int orig_i = indices[(r)*numColumns + c]; // index of original vertex of grid
            if (orig_i >= 0)
            {
                unsigned int new_i = vertices->size(); // index of new index of added skirt point
                osg::Vec3 new_v = (*vertices)[orig_i] - ((*skirtVectors)[orig_i]) * skirtHeight;
                (*vertices).push_back(new_v);
                if (normals.valid())
                    (*normals).push_back((*normals)[orig_i]);
                for (LayerToTexCoordMap::iterator itr = layerToTexCoordMap.begin();
                     itr != layerToTexCoordMap.end();
                     ++itr)
                {
                    itr->second.first->push_back((*itr->second.first)[orig_i]);
                }

                skirtDrawElements->addElement(orig_i);
                skirtDrawElements->addElement(new_i);
            }
            else
            {
                if (skirtDrawElements->getNumIndices() != 0)
                {
                    geometry->addPrimitiveSet(skirtDrawElements.get());
                    skirtDrawElements = smallTile ? static_cast<osg::DrawElements *>(new osg::DrawElementsUShort(GL_QUAD_STRIP)) : static_cast<osg::DrawElements *>(new osg::DrawElementsUInt(GL_QUAD_STRIP));
                }
            }
        }

        if (skirtDrawElements->getNumIndices() != 0)
        {
            geometry->addPrimitiveSet(skirtDrawElements.get());
            skirtDrawElements = smallTile ? static_cast<osg::DrawElements *>(new osg::DrawElementsUShort(GL_QUAD_STRIP)) : static_cast<osg::DrawElements *>(new osg::DrawElementsUInt(GL_QUAD_STRIP));
        }

        // create top skirt vertices
        r = numRows - 1;
        for (c = numColumns - 1; c >= 0; --c)
        {
            int orig_i = indices[(r)*numColumns + c]; // index of original vertex of grid
            if (orig_i >= 0)
            {
                unsigned int new_i = vertices->size(); // index of new index of added skirt point
                osg::Vec3 new_v = (*vertices)[orig_i] - ((*skirtVectors)[orig_i]) * skirtHeight;
                (*vertices).push_back(new_v);
                if (normals.valid())
                    (*normals).push_back((*normals)[orig_i]);
                for (LayerToTexCoordMap::iterator itr = layerToTexCoordMap.begin();
                     itr != layerToTexCoordMap.end();
                     ++itr)
                {
                    itr->second.first->push_back((*itr->second.first)[orig_i]);
                }

                skirtDrawElements->addElement(orig_i);
                skirtDrawElements->addElement(new_i);
            }
            else
            {
                if (skirtDrawElements->getNumIndices() != 0)
                {
                    geometry->addPrimitiveSet(skirtDrawElements.get());
                    skirtDrawElements = smallTile ? static_cast<osg::DrawElements *>(new osg::DrawElementsUShort(GL_QUAD_STRIP)) : static_cast<osg::DrawElements *>(new osg::DrawElementsUInt(GL_QUAD_STRIP));
                }
            }
        }

        if (skirtDrawElements->getNumIndices() != 0)
        {
            geometry->addPrimitiveSet(skirtDrawElements.get());
            skirtDrawElements = smallTile ? static_cast<osg::DrawElements *>(new osg::DrawElementsUShort(GL_QUAD_STRIP)) : static_cast<osg::DrawElements *>(new osg::DrawElementsUInt(GL_QUAD_STRIP));
        }

        // create left skirt vertices
        c = 0;
        for (r = numRows - 1; r >= 0; --r)
        {
            int orig_i = indices[(r)*numColumns + c]; // index of original vertex of grid
            if (orig_i >= 0)
            {
                unsigned int new_i = vertices->size(); // index of new index of added skirt point
                osg::Vec3 new_v = (*vertices)[orig_i] - ((*skirtVectors)[orig_i]) * skirtHeight;
                (*vertices).push_back(new_v);
                if (normals.valid())
                    (*normals).push_back((*normals)[orig_i]);
                for (LayerToTexCoordMap::iterator itr = layerToTexCoordMap.begin();
                     itr != layerToTexCoordMap.end();
                     ++itr)
                {
                    itr->second.first->push_back((*itr->second.first)[orig_i]);
                }

                skirtDrawElements->addElement(orig_i);
                skirtDrawElements->addElement(new_i);
            }
            else
            {
                if (skirtDrawElements->getNumIndices() != 0)
                {
                    geometry->addPrimitiveSet(skirtDrawElements.get());
                    skirtDrawElements = new osg::DrawElementsUShort(GL_QUAD_STRIP);
                }
            }
        }

        if (skirtDrawElements->getNumIndices() != 0)
        {
            geometry->addPrimitiveSet(skirtDrawElements.get());
            smallTile ? static_cast<osg::DrawElements *>(new osg::DrawElementsUShort(GL_QUAD_STRIP)) : static_cast<osg::DrawElements *>(new osg::DrawElementsUInt(GL_QUAD_STRIP));
        }
    }

    //pseudo-random-number-generator with constant seed
    MTRand mt(10101);

    //Make tree geometry
    if (_terrainTile->getTerrain())
    {
        osg::ref_ptr<osgTerrain::TerrainTile> masterTile = _terrainTile->getTerrain()->getTile(osgTerrain::TileID(0, 0, 0));
        if (!masterTile)
            masterTile = _terrainTile;

        //double treeSpacing = 10.0*ceil(7.0/(double)(tileId.level));
        //std::cout << "Tile level: " << tileId.level << ", x: " << tileId.x << ", y: " << tileId.y << ", treeSpacing: " << treeSpacing << std::endl;

        /*osg::ref_ptr<osg::Geode> treeGeode = new osg::Geode();
       if(buffer._transform.valid())
          buffer._transform->addChild(treeGeode.get());
       treeGeode->setName("TreeGeode");
       if(treeStateSet) {
         treeGeode->setStateSet(treeStateSet);
       }*/
        osg::ref_ptr<osg::Geometry> treeGeometry = new osg::Geometry();
        if (treeStateSet)
        {
            treeGeometry->setStateSet(treeStateSet);
        }
        //treeGeode->addDrawable(treeGeometry);

        osg::ref_ptr<osg::Vec3Array> treeVertices = new osg::Vec3Array();
        treeGeometry->setVertexArray(treeVertices);

        osg::ref_ptr<osg::Vec2Array> treeTexCoords = new osg::Vec2Array();
        treeGeometry->setTexCoordArray(0, treeTexCoords);

        osg::ref_ptr<osg::Geometry> buildingGeometry = new osg::Geometry();
        if (buildingStateSet)
        {
            buildingGeometry->setStateSet(buildingStateSet);
        }

        osg::ref_ptr<osg::Vec3Array> buildingVertices = new osg::Vec3Array();
        buildingGeometry->setVertexArray(buildingVertices);
        unsigned int buildingVertIt = 0;

        osg::ref_ptr<osg::Vec2Array> buildingTexCoords = new osg::Vec2Array();
        buildingGeometry->setTexCoordArray(0, buildingTexCoords);

        osg::ref_ptr<osg::Vec3Array> buildingNormals = new osg::Vec3Array();
        buildingGeometry->setNormalArray(buildingVertices);
        buildingGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        std::cerr << "buildingGeometry->setNormalArray(buildingVertices) - this is probably wrong" << std::endl;
        //std::cout << "Master tile: name: " << masterTile->getName() << ", coordinate system: " << masterTile->getLocator()->getCoordinateSystem() << std::endl;

        /*std::set<std::string> terrainTileForestNames;

       for(int groupIt = 0; groupIt<_terrainTile->getNumChildren(); ++groupIt) {
          osg::Group* terrainTileChildGroup = dynamic_cast<osg::Group*>(_terrainTile->getChild(groupIt));
          if(terrainTileChildGroup) for(int childIt=0; childIt<terrainTileChildGroup->getNumChildren(); ++childIt) {
             osg::Geode* terrainTileChildGeode = dynamic_cast<osg::Geode*>(terrainTileChildGroup->getChild(childIt));

             if((terrainTileChildGeode->getNumDescriptions()>0 && terrainTileChildGeode->getDescription(0)==std::string("SHAPEFILE")))
             {
                for(int drawableIt=0; drawableIt<terrainTileChildGeode->getNumDrawables(); ++drawableIt)
                {
                   osg::Geometry* shapeGeometry = dynamic_cast<osg::Geometry*>(terrainTileChildGeode->getDrawable(drawableIt));
                   if(shapeGeometry) {
                      osgSim::ShapeAttributeList* attributeList = dynamic_cast<osgSim::ShapeAttributeList*>(shapeGeometry->getUserData());
                      if(attributeList) for(int attributeIt=0; attributeIt<attributeList->size(); ++attributeIt) {
                         osgSim::ShapeAttribute attribute = (*attributeList)[attributeIt];
                         if(attribute.getName()==std::string("OBJNAME")) {
                           terrainTileForestNames.insert(std::string(attribute.getString()));
                           break;
                         }
                      }
                   }
                }
             }
          }
       }*/

        osg::Vec3d lower_left_tile;
        masterLocator->convertLocalToModel(osg::Vec3(0.0, 0.0, 0.0), lower_left_tile);
        lower_left_tile -= centerModel;

        osg::Vec3d upper_right_tile;
        masterLocator->convertLocalToModel(osg::Vec3(1.0, 1.0, 0.0), upper_right_tile);
        upper_right_tile -= centerModel;

        for (unsigned int shapeGeodeIt = 0; shapeGeodeIt < shapeFileVector.size(); ++shapeGeodeIt)
        {
            osg::ref_ptr<osg::Geode> shapeGeode = shapeFileVector[shapeGeodeIt];

            /*if(shapeGeode->getName()==std::string("forest")) {
            hasNameForest = true;
          }
          else if(shapeGeode->getName()==std::string("buildings")) {
            hasNameBuilding = true;
          }*/

            for (unsigned int drawableIt = 0; drawableIt < shapeGeode->getNumDrawables(); ++drawableIt)
            {
                osg::ref_ptr<osg::Geometry> shapeGeometry = dynamic_cast<osg::Geometry *>(shapeGeode->getDrawable(drawableIt));
                if (shapeGeometry)
                {
                    bool hasNameForest = false;
                    bool hasNameBuilding = false;

                    //osg::ref_ptr< osg::Referenced > userData = shapeGeometry->getUserData();
                    osg::Referenced *userData = shapeGeometry->getUserData();

                    //if(userData.valid()) {
                    if (userData)
                    {
                        //osg::ref_ptr<osgSim::ShapeAttributeList> attributeList = dynamic_cast<osgSim::ShapeAttributeList*>(userData.get());
                        osgSim::ShapeAttributeList *attributeList = dynamic_cast<osgSim::ShapeAttributeList *>(userData);
                        //bool objNameInList = false;
                        if (attributeList)
                            for (int attributeIt = 0; attributeIt < attributeList->size(); ++attributeIt)
                            {
                                const osgSim::ShapeAttribute &attribute = attributeList->operator[](attributeIt);
                                osgSim::ShapeAttribute::Type attributeType = attribute.getType();
                                const char *attributeString = attribute.getString();

                                if (attributeString && attributeType == osgSim::ShapeAttribute::STRING)
                                {
                                    std::string attributeName = attribute.getName();

                                    //if(attributeName.compare("GM_LAYER")==0 || attributeName.compare("GM_TYPE")==0)
                                    if (attributeName.compare("GM_TYPE") == 0)
                                    {
                                        if (std::string(attribute.getString()).compare(0, 6, std::string("Forest")) == 0)
                                        {
                                            hasNameForest = true;
                                            break;
                                        }
                                        else if (std::string(attribute.getString()).compare(0, 9, std::string("Buildings")) == 0)
                                        {
                                            hasNameBuilding = true;
                                            break;
                                        }
                                    }
                                }
                            }
                    }

                    if (hasNameForest /*&& objNameInList*/)
                    {
                        //std::cout << "Building forest... " << std::endl;
                        if (std::string(shapeGeometry->getVertexArray()->className()) == "Vec3Array")
                        {
                            osg::ref_ptr<osg::Vec3Array> vertexArray = static_cast<osg::Vec3Array *>(shapeGeometry->getVertexArray());

                            osg::Vec3d lower_left = (*vertexArray)[0] - centerModel;
                            osg::Vec3d upper_right = lower_left;

                            for (unsigned int treeIt = 0; treeIt < vertexArray->getNumElements(); treeIt += 2)
                            {
                                osg::Vec3d treePos = (*vertexArray)[treeIt] - centerModel;

                                lower_left.x() = std::min(lower_left.x(), treePos.x());
                                lower_left.y() = std::min(lower_left.y(), treePos.y());
                                upper_right.x() = std::max(upper_right.x(), treePos.x());
                                upper_right.y() = std::max(upper_right.y(), treePos.y());
                            }

                            //Look if tile area (_tile) and tree shape area bounding box overlaps
                            const osg::Vec3d &a_min = lower_left;
                            const osg::Vec3d &a_max = upper_right;
                            const osg::Vec3d &b_min = lower_left_tile;
                            const osg::Vec3d &b_max = upper_right_tile;
                            if (((b_max.x() >= a_min.x() && b_max.x() <= a_max.x())
                                 || (b_min.x() >= a_min.x() && b_min.x() <= a_max.x())
                                 || (b_min.x() <= a_min.x() && b_max.x() >= a_max.x()))
                                && (b_max.y() >= a_min.y() && b_min.y() <= a_max.y()))
                            {
                                std::map<double, std::set<double> > scanLineMap;
                                int scanlineIt = 0;
                                for (double scanline = lower_left.y() + 5.0; scanline <= upper_right.y() - 5.0; scanline += 10.0, ++scanlineIt)
                                {
                                    if ((scanlineIt % (8 - tileId.level)) != 0)
                                        continue;

                                    for (unsigned int treeIt = 0; treeIt < vertexArray->getNumElements() - 2; treeIt += 2)
                                    {
                                        osg::Vec3 pos0 = (*vertexArray)[treeIt] - centerModel;

                                        osg::Vec3 pos1 = (*vertexArray)[treeIt + 2] - centerModel;

                                        double t = (scanline - pos0.y()) / (pos1.y() - pos0.y());
                                        if (t > 0.0 && t < 1.0)
                                        {
                                            //scanLineMap[scanline].insert(std::make_pair(pos0.x() + t*(pos1.x()-pos0.x()), pos0.z() + t*(pos1.z()-pos0.z())));
                                            double x = pos0.x() + t * (pos1.x() - pos0.x());
                                            scanLineMap[scanline].insert(x);
                                        }
                                    }
                                    {
                                        osg::Vec3 pos0 = (*vertexArray)[vertexArray->getNumElements() - 2] - centerModel;
                                        osg::Vec3 pos1 = (*vertexArray)[0] - centerModel;
                                        double t = (scanline - pos0.y()) / (pos1.y() - pos0.y());
                                        if (t >= 0.0 && t <= 1.0)
                                        {
                                            double x = pos0.x() + t * (pos1.x() - pos0.x());
                                            scanLineMap[scanline].insert(x);
                                        }
                                    }
                                }

                                if (scanLineMap.begin() != scanLineMap.end())
                                {
                                    int dropCount = 0;

                                    for (std::map<double, std::set<double> >::iterator scanLineIt = scanLineMap.begin(); scanLineIt != scanLineMap.end(); ++scanLineIt)
                                    {
                                        //if(scanLineIt->second.size() == 0 || (scanLineIt->second.size() % 2) != 0)
                                        if (scanLineIt->second.size() < 2)
                                        {
                                            //std::cout << "continue due to scanLineIt->second.size: " << scanLineIt->second.size() << std::endl;
                                            ++dropCount;
                                            continue;
                                        }
                                        else
                                        {
                                            //std::cout << "found scanLineIt->second.size: " << scanLineIt->second.size() << std::endl;
                                        }

                                        double y = scanLineIt->first;
                                        for (std::set<double>::iterator cutIt = scanLineIt->second.begin(); cutIt != scanLineIt->second.end(); ++cutIt, ++cutIt)
                                        {
                                            std::set<double>::iterator nextCutIt = cutIt;
                                            ++nextCutIt;
                                            if (nextCutIt == scanLineIt->second.end())
                                                break;
                                            double x0 = (*cutIt);
                                            double x1 = (*nextCutIt);

                                            int xIt = 0;
                                            for (double x = x0 + 5.0; x < x1 - 5.0; x += 10.0, ++xIt)
                                            {
                                                if ((xIt % (8 - tileId.level)) != 0)
                                                    continue;

                                                double t = x / (x1 - x0);

                                                if (lower_left_tile.y() <= y && upper_right_tile.y() >= y && lower_left_tile.x() <= x && upper_right_tile.x() >= x)
                                                {
                                                    double ndc_x = (x - lower_left_tile.x()) / (upper_right_tile.x() - lower_left_tile.x());
                                                    double ndc_y = (y - lower_left_tile.y()) / (upper_right_tile.y() - lower_left_tile.y());

                                                    float z = 0.0;
                                                    if (elevationLayer)
                                                    {
                                                        elevationLayer->getInterpolatedValue(ndc_x, ndc_y, z);
                                                    }

                                                    osg::Vec3 treePos(x, y, z);
                                                    //std::cout << "Adding tree at (" << treePos.x() << ", " << treePos.y() << ", " << treePos.z() << std::endl;

                                                    int treeTexture = ((scanlineIt + xIt) % 4);
                                                    double u_min = (double)(treeTexture / 2) * 0.5;
                                                    double v_min = (double)(treeTexture % 2) * 0.5;
                                                    double u_max = u_min + 0.5;
                                                    double v_max = v_min + 0.5;

                                                    treeVertices->push_back(treePos + osg::Vec3(-3.18, 0.0, 0.0));
                                                    treeTexCoords->push_back(osg::Vec2(u_min, v_min));
                                                    treeVertices->push_back(treePos + osg::Vec3(3.18, 0.0, 0.0));
                                                    treeTexCoords->push_back(osg::Vec2(u_max, v_min));
                                                    treeVertices->push_back(treePos + osg::Vec3(3.18, 0.0, 20.0));
                                                    treeTexCoords->push_back(osg::Vec2(u_max, v_max));
                                                    treeVertices->push_back(treePos + osg::Vec3(-3.18, 0.0, 20.0));
                                                    treeTexCoords->push_back(osg::Vec2(u_min, v_max));

                                                    treeVertices->push_back(treePos + osg::Vec3(0.0, -3.18, 0.0));
                                                    treeTexCoords->push_back(osg::Vec2(u_min, v_min));
                                                    treeVertices->push_back(treePos + osg::Vec3(0.0, 3.18, 0.0));
                                                    treeTexCoords->push_back(osg::Vec2(u_max, v_min));
                                                    treeVertices->push_back(treePos + osg::Vec3(0.0, 3.18, 20.0));
                                                    treeTexCoords->push_back(osg::Vec2(u_max, v_max));
                                                    treeVertices->push_back(treePos + osg::Vec3(0.0, -3.18, 20.0));
                                                    treeTexCoords->push_back(osg::Vec2(u_min, v_max));
                                                }
                                            }
                                        }
                                    }
                                    //std::cout << dropCount << " scanlines of " << scanLineMap.size() << " dropped!" << std::endl;
                                }
                            }
                        }
                        else
                        {
                            std::cout << "\tUnknown type: " << shapeGeometry->getVertexArray()->className() << std::endl;
                        }
                    }

                    else if (hasNameBuilding)
                    {
                        if (std::string(shapeGeometry->getVertexArray()->className()) == "Vec3Array")
                        {
                            osg::ref_ptr<osg::Vec3Array> vertexArray = static_cast<osg::Vec3Array *>(shapeGeometry->getVertexArray());

                            bool buildingOutsideTile = false;
                            std::vector<osg::Vec3d> buildingGroundVerts;
                            osg::Vec3d buildingCenter(0.0, 0.0, 0.0);

                            for (unsigned int buildingIt = 0; buildingIt < vertexArray->getNumElements(); buildingIt += 1)
                            {
                                osg::Vec3d buildingPos0 = (*vertexArray)[buildingIt] - centerModel;
                                //osg::Vec3d buildingPos1 = (*vertexArray)[buildingIt+1] - centerModel;

                                if (lower_left_tile.y() <= buildingPos0.y() && upper_right_tile.y() >= buildingPos0.y()
                                    && lower_left_tile.x() <= buildingPos0.x() && upper_right_tile.x() >= buildingPos0.x()
                                                                                                              /*&& lower_left_tile.y() <= buildingPos1.y() && upper_right_tile.y() >= buildingPos1.y() 
                         && lower_left_tile.x() <= buildingPos1.x() && upper_right_tile.x() >= buildingPos1.x()*/)
                                {
                                    double ndc0_x = (buildingPos0.x() - lower_left_tile.x()) / (upper_right_tile.x() - lower_left_tile.x());
                                    double ndc0_y = (buildingPos0.y() - lower_left_tile.y()) / (upper_right_tile.y() - lower_left_tile.y());
                                    //double ndc1_x = (buildingPos1.x()-lower_left_tile.x())/(upper_right_tile.x()-lower_left_tile.x());
                                    //double ndc1_y = (buildingPos1.y()-lower_left_tile.y())/(upper_right_tile.y()-lower_left_tile.y());

                                    float z0 = 0.0;
                                    //float z1 = 0.0;
                                    if (elevationLayer)
                                    {
                                        elevationLayer->getInterpolatedValue(ndc0_x, ndc0_y, z0);
                                        //elevationLayer->getInterpolatedValue(ndc1_x, ndc1_y, z1);
                                    }
                                    buildingPos0.z() = z0;
                                    //buildingPos1.z() = z1;

                                    buildingGroundVerts.push_back(buildingPos0);
                                    //buildingGroundVerts.push_back(buildingPos1);

                                    buildingCenter += buildingPos0;
                                }
                                else
                                {
                                    buildingOutsideTile = true;
                                    break;
                                }
                            }

                            if (!buildingOutsideTile && buildingGroundVerts.size() > 1)
                            {
                                buildingCenter *= 1.0 / (buildingGroundVerts.size());

                                unsigned int wall = 0;

                                unsigned int buildingStartVertIt = buildingVertIt;
                                for (int groundVertIt = 0; groundVertIt < buildingGroundVerts.size(); ++groundVertIt)
                                {
                                    buildingVertices->push_back(buildingGroundVerts[groundVertIt] + osg::Vec3(0.0, 0.0, 0.0));
                                    buildingTexCoords->push_back(osg::Vec2(((double)wall) * 0.5, 0.0));
                                    //buildingVertices->push_back(buildingGroundVerts[groundVertIt+1] + osg::Vec3(0.0, 0.0, 0.0));
                                    //buildingTexCoords->push_back(osg::Vec2(((double)wall)*0.5+0.5, 0.0));
                                    //buildingVertices->push_back(buildingGroundVerts[groundVertIt+1] + osg::Vec3(0.0, 0.0, 10.0));
                                    //buildingTexCoords->push_back(osg::Vec2(((double)wall)*0.5+0.5, 1.0));
                                    buildingVertices->push_back(buildingGroundVerts[groundVertIt] + osg::Vec3(0.0, 0.0, 10.0));
                                    buildingTexCoords->push_back(osg::Vec2(((double)wall) * 0.5, 1.0));

                                    buildingVertIt += 2;

                                    if (++wall > 1)
                                        wall = 0;
                                }
                                osg::ref_ptr<osg::DrawArrays> buildingDrawArrays = new osg::DrawArrays(osg::PrimitiveSet::QUAD_STRIP, buildingStartVertIt, buildingVertIt - buildingStartVertIt);
                                buildingGeometry->addPrimitiveSet(buildingDrawArrays);

                                for (int groundVertIt = 0; groundVertIt < buildingGroundVerts.size() - 1; ++groundVertIt)
                                {
                                    osg::Vec3d wallDir = buildingGroundVerts[groundVertIt + 1] - buildingGroundVerts[groundVertIt];
                                    osg::Vec3d normDir(wallDir.y(), -wallDir.x(), 0.0);
                                    normDir.normalize();
                                    buildingNormals->push_back(normDir);
                                }
                            }
                        }
                    }
                }
            }
        }

        osg::ref_ptr<osg::DrawArrays> treeDrawArrays = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, treeVertices->size());
        treeGeometry->addPrimitiveSet(treeDrawArrays);
        if (treeVertices->size() >= 4)
            buffer._geode->addDrawable(treeGeometry);

        if (buildingVertices->size() >= 4)
            buffer._geode->addDrawable(buildingGeometry);
    }
    //end tree build

    geometry->setUseDisplayList(false);
    geometry->setUseVertexBufferObjects(true);

    if (osgDB::Registry::instance()->getBuildKdTreesHint() == osgDB::ReaderWriter::Options::BUILD_KDTREES && osgDB::Registry::instance()->getKdTreeBuilder())
    {

        //osg::Timer_t before = osg::Timer::instance()->tick();
        //OSG_NOTIFY(osg::NOTICE)<<"osgTerrain::GeometryTechnique::build kd tree"<<std::endl;
        osg::ref_ptr<osg::KdTreeBuilder> builder = osgDB::Registry::instance()->getKdTreeBuilder()->clone();
        buffer._geode->accept(*builder);
        //osg::Timer_t after = osg::Timer::instance()->tick();
        //OSG_NOTIFY(osg::NOTICE)<<"KdTree build time "<<osg::Timer::instance()->delta_m(before, after)<<std::endl;
    }
}
