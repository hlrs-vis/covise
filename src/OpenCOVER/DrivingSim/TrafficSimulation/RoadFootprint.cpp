/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "RoadFootprint.h"

#include "RoadSystem/RoadSystem.h"

#include <iostream>
#include <algorithm>
//#include <osgDB/Options>

void RoadFootprintTileLoadedCallback::loaded(osgTerrain::TerrainTile *tile, const osgDB::ReaderWriter::Options *options) const
{
    osgTerrain::HeightFieldLayer *hflayer = dynamic_cast<osgTerrain::HeightFieldLayer *>(tile->getElevationLayer());
    osg::HeightField *field = NULL;
    if (hflayer)
    {
        field = hflayer->getHeightField();
    }
    else
    {
        return;
    }

    double tileXInterval = field->getXInterval();
    double tileYInterval = field->getYInterval();
    /*double tileXMin = field->getOrigin().x() - tileXInterval*(0.5*(double)(field->getNumRows()));
   double tileXMax = field->getOrigin().x() + tileXInterval*(0.5*(double)(field->getNumRows()));
   double tileYMin = field->getOrigin().y() - tileYInterval*(0.5*(double)(field->getNumColumns()));
   double tileYMax = field->getOrigin().y() + tileYInterval*(0.5*(double)(field->getNumColumns()));*/
    double tileXMin = field->getOrigin().x();
    double tileXMax = field->getOrigin().x() + tileXInterval * std::max((int)field->getNumColumns() - 1, 0);
    double tileYMin = field->getOrigin().y();
    double tileYMax = field->getOrigin().y() + tileYInterval * std::max((int)field->getNumRows() - 1, 0);

    for (int i = 0; i < RoadSystem::Instance()->getNumRoads(); ++i)
    {
        Road *road = RoadSystem::Instance()->getRoad(i);
        double roadLength = road->getLength();

        //Vector3D roadStart = road->getChordLinePlanViewPoint(0.0);
        //Vector3D roadEnd = road->getChordLinePlanViewPoint(roadLength);

        //      if((tileXMin >= roadStart.x() && roadStart.x() <= tileXMax && tileYMin >= roadStart.y() && roadStart.y() <= tileYMax)
        //         || (tileXMin >= roadEnd.x() && roadEnd.x() <= tileXMax && tileYMin >= roadEnd.y() && roadEnd.y() <= tileYMax)) {
        if (true)
        {

            double sOffset = sqrt(pow(tileXInterval, 2) + pow(tileYInterval, 2)) * 0.5;
            for (double s = 0; s < roadLength + sOffset; s += sOffset)
            {
                //std::cerr << "s: " << s << ", sOffset: " << sOffset << ", roadLength: " << roadLength << std::endl;
                if (s > roadLength)
                {
                    s = roadLength;
                }

                //Road side distances and heights
                double tOffset = sOffset;
                double roadLeft, roadRight;
                road->getRoadSideWidths(s, roadRight, roadLeft);
                double roadWidth = fabs(roadLeft - roadRight);
                RoadPoint pointRoadLeft = road->getRoadPoint(s, roadLeft);
                RoadPoint pointRoadRight = road->getRoadPoint(s, roadRight);
                //roadLeft += tOffset;
                //roadRight -= tOffset;
                double roadHeightLeft = pointRoadLeft.z();
                double roadHeightRight = pointRoadRight.z();

                //Bank distances and heights
                double bankRight = roadRight - roadWidth * 0.5;
                double bankLeft = roadLeft + roadWidth * 0.5;
                Vector3D normVectorRoadRightLeft = (pointRoadLeft.pos() - pointRoadRight.pos()).normalized();
                Vector3D pointBankRight = pointRoadRight.pos() + normVectorRoadRightLeft * (bankRight - roadRight);
                Vector3D pointBankLeft = pointRoadLeft.pos() + normVectorRoadRightLeft * (bankLeft - roadLeft);
#ifdef WIN32
#define round(x) ((int)(x + 0.5))
#endif
                int colLeft = (int)round((pointBankLeft.x() - tileXMin) / tileXInterval);
                if (colLeft < 0)
                    colLeft = 0;
                else if (colLeft >= field->getNumColumns())
                    colLeft = field->getNumColumns() - 1;
                int rowLeft = (int)round((pointBankLeft.y() - tileYMin) / tileYInterval);
                if (rowLeft < 0)
                    rowLeft = 0;
                else if (rowLeft >= field->getNumRows())
                    rowLeft = field->getNumRows() - 1;
                double bankHeightLeft = field->getHeight(colLeft, rowLeft);
                int colRight = (int)round((pointBankRight.x() - tileXMin) / tileXInterval);
                if (colRight < 0)
                    colRight = 0;
                else if (colRight >= field->getNumColumns())
                    colRight = field->getNumColumns() - 1;
                int rowRight = (int)round((pointBankRight.y() - tileYMin) / tileYInterval);
                if (rowRight < 0)
                    rowRight = 0;
                else if (rowRight >= field->getNumRows())
                    rowRight = field->getNumRows() - 1;
                double bankHeightRight = field->getHeight(colRight, rowRight);

                //Banking parameters (tanh)
                double steepness = 0.5; //Steepness of bank (tanh parameter)

                double a_r = (roadHeightRight - bankHeightRight) * 0.5;
                double b_r = 2.0 * steepness / (bankRight - roadRight) - steepness;
                double c_r = (bankHeightRight + roadHeightRight) * 0.5;
                double t0_r = (roadRight + bankRight) * 0.5;
                //std::cout << "a_r: " << a_r << ", b_r: " << b_r << ", c_r: " << c_r << ", t0_r: " << t0_r << std::endl;

                double a_l = (roadHeightLeft - bankHeightLeft) * 0.5;
                double b_l = 2.0 * steepness / (bankLeft - roadLeft) - steepness;
                double c_l = (bankHeightLeft + roadHeightLeft) * 0.5;
                double t0_l = (roadLeft + bankLeft) * 0.5;
                //std::cout << "a_l: " << a_l << ", b_l: " << b_l << ", c_l: " << c_l << ", t0_l: " << t0_l << std::endl;

                for (double t = bankRight; t < bankLeft + tOffset; t += tOffset)
                {
                    if (t > bankLeft)
                    {
                        t = bankLeft;
                    }

                    RoadPoint point = road->getRoadPoint(s, t);

                    int c = (int)(round((point.x() - tileXMin) / tileXInterval));
                    int r = (int)(round((point.y() - tileYMin) / tileYInterval));
                    if ((c >= 0 && c < field->getNumColumns()) && (r >= 0 && r < field->getNumRows()))
                    {
                        //std::cerr << "roadLeft, roadRight: " << roadLeft << ", " << roadRight << "; s, t: " << s << ", " << t << "; c, r: " << c << ", " << r << ": new height!" << std::endl;
                        //double height = 1000;
                        //double height = road->getChordLineElevation(s);
                        if (t < roadRight)
                        {
                            field->setHeight(c, r, a_r * tanh((t - t0_r) * b_r) + c_r);
                        }
                        else if (roadRight <= t && t <= roadLeft)
                        {
                            field->setHeight(c, r, point.z());
                        }
                        else if (t > roadLeft)
                        {
                            field->setHeight(c, r, a_l * tanh((t - t0_l) * b_l) + c_l);
                        }
                    }
                }

                //sOffset = std::min(tileXInterval/fabs(cos(point.z())), tileYInterval/fabs(sin(point.z())));
            }
        }
    }

    //std::cout << "tile: x min: " << tileXMin << ", x max: " << tileXMax << ", y min: " << tileYMin << ", y max: " << tileYMax << std::endl;
}
