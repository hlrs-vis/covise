/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- VirtualPlanetBuilder - Copyright (C) 1998-2009 Robert Osfield
 *
 * This application is open source and may be redistributed and/or modified
 * freely and without restriction, both in commericial and non commericial applications,
 * as long as this copyright notice is maintained.
 * 
 * This application is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <vpb/BuildOperation>
#include <osg/CoordinateSystemNode>

#include <osgDB/ReadFile>
#include <osgTerrain/Terrain>

#include <iostream>

unsigned int computeNumTiles(unsigned int numTilesLevel1, int level)
{
    if (level == 0)
        return 1;
    else
        return static_cast<unsigned int>(numTilesLevel1 * pow(2.0f, level - 1));
}

unsigned int computeTotalSize(unsigned int numTilesLevel1, unsigned int s, bool wrap, int level)
{
    return computeNumTiles(numTilesLevel1, level) * (s - 1) + (wrap ? 0 : 1);
}

unsigned int numLevelsRequired(unsigned int numTilesLevel1, unsigned int s, bool wrap, unsigned int targetSize)
{
    unsigned int level = 0;
    while (computeTotalSize(numTilesLevel1, s, wrap, level) < targetSize)
    {
        ++level;
    }
    return level;
}

unsigned int numLevelsRequired(unsigned int numXTilesLevel1, unsigned int numYTilesLevel1, unsigned int s, bool wrap, unsigned int width, unsigned int height)
{
    return std::max(
        numLevelsRequired(numXTilesLevel1, s, wrap, width),
        numLevelsRequired(numYTilesLevel1, s, wrap, height));
}

double computeEfficiency(unsigned int numXTilesLevel1, unsigned int numYTilesLevel1, unsigned int s, bool wrap, unsigned int width, unsigned int height, unsigned int numLevels)
{
    unsigned int finalWidth = computeTotalSize(numXTilesLevel1, s, wrap, numLevels);
    unsigned int finalHeight = computeTotalSize(numYTilesLevel1, s, wrap, numLevels);

    double finalMemorySize = double(finalWidth) * double(finalHeight);

    double originalMemorySize = double(width) * double(height);

    double efficiency = finalMemorySize / originalMemorySize;

    return efficiency;
}

int main(int argc, char **argv)
{
    // use an ArgumentParser object to manage the program arguments.
    osg::ArgumentParser arguments(&argc, argv);

    osg::ref_ptr<osg::EllipsoidModel> ellipsoid = new osg::EllipsoidModel;
    double circumferance = ellipsoid->getRadiusEquator() * 2.0 * osg::PI;

    unsigned int width = 43200;
    unsigned int height = 21600;
    while (arguments.read("--size", width, height))
    {
    }

    bool wholeearth = true;
    while (arguments.read("--wholeearth"))
    {
        wholeearth = false;
    }

    unsigned int numXTilesLevel1 = 2;
    unsigned int numYTilesLevel1 = 1;
    while (arguments.read("--level1", numXTilesLevel1, numYTilesLevel1))
    {
    }

    double r = 1000;
    while (arguments.read("-r", r))
    {
        width = static_cast<int>(circumferance / r);
        height = width / 2;
    }

    if (wholeearth)
        std::cout << "source data resolution = " << circumferance / double(width) << "m" << std::endl;

    std::cout << "Source width = " << width << " height = " << height << std::endl;

    unsigned int s = 256;
    while (arguments.read("-s", s))
    {
    }

    bool wrap = wholeearth;

    for (unsigned int i = 1; i < 20; ++i)
    {
        unsigned int numLevels = numLevelsRequired(numXTilesLevel1 * i, numYTilesLevel1 * i, s, wrap, width, height);
        {
            double efficiency = computeEfficiency(numXTilesLevel1 * i, numYTilesLevel1 * i, s, wrap, width, height, numLevels);
            unsigned int finalWidth = computeTotalSize(numXTilesLevel1 * i, s, wrap, numLevels);
            unsigned int finalHeight = computeTotalSize(numYTilesLevel1 * i, s, wrap, numLevels);
            std::cout << " level 1 dimension = (" << numXTilesLevel1 *i << ", " << numYTilesLevel1 *i << ")"
                      << " numLevels = " << numLevels
                      << " finalSize = (" << finalWidth << " " << finalHeight << ")"
                      << " efficiency = " << efficiency;
        }

        {
            --numLevels;
            double efficiency = computeEfficiency(numXTilesLevel1 * i, numYTilesLevel1 * i, s, wrap, width, height, numLevels);
            unsigned int finalWidth = computeTotalSize(numXTilesLevel1 * i, s, wrap, numLevels);
            unsigned int finalHeight = computeTotalSize(numYTilesLevel1 * i, s, wrap, numLevels);
            std::cout << "\tnumLevels = " << numLevels
                      << " finalSize = (" << finalWidth << " " << finalHeight << ")"
                      << " undersampling = " << efficiency << std::endl;
        }
    }

    return 0;
}
