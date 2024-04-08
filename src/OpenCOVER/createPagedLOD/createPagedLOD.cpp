/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>
#include <iostream>
#include <boost/filesystem.hpp>
#include <osg/Group>
#include <osg/PagedLOD>
#include <osgTerrain/Terrain>
#include <osg/MatrixTransform>
#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
using namespace std;
using namespace boost::filesystem;
int main(int argc, char **argv)
{
    // check arguments
    if (argc != 3)
    {
        std::cerr << "Usage: createPagedLOD geometry_source_directory outputFile" << std::endl;
        return 1;
    }
    path p(argv[1]);  // avoid repeated path construction below
    std::vector<osg::Vec3> positions;
    double dx = 0;
    double dy = 0;
    double minx = 100000000000000;
    double maxx = -100000000000000;
    double miny = 100000000000000;
    double maxy = -100000000000000;
    double distance = 5000;


    if (exists(p))    // does path p actually exist?
    {
        if (is_directory(p))      // is path p a directory?
        {
            for (directory_entry& x : directory_iterator(p))
            {
                cout << "    " << x.path() << '\n';
                osg::Node *node = osgDB::readNodeFile(x.path().string());
                osg::MatrixTransform* mt = dynamic_cast<osg::MatrixTransform*>(node);
                osg::Vec3 pos;
                if (mt)
                {
                    pos = mt->getMatrix().getTrans();
                }
                else
                {
                    pos = node->getBound().center();
                }
                positions.push_back(pos);
            }
            std::cerr << "numGeometries: " << positions.size() << std::endl;
            if (positions.size() > 0)
            {
                for (const auto& p : positions)
                {
                    if (p.x() < positions[0].x() - 10 || p.x() > positions[0].x() + 10) // dx
                    {
                        if (dx == 0 || abs(positions[0].x() - p.x()) < dx)
                            dx = abs(positions[0].x() - p.x());
                    }
                    if (p.y() < positions[0].y() - 10 || p.y() > positions[0].y() + 10) // dy
                    {
                        if (dy == 0 || abs(positions[0].y() - p.y()) < dy)
                            dy = abs(positions[0].y() - p.y());
                    }
                    if (p.x() > maxx)
                        maxx = p.x();
                    if (p.y() > maxy)
                        maxy = p.y();
                    if (p.x() < minx)
                        minx = p.x();
                    if (p.y() < miny)
                        miny = p.y();
                }
                std::cerr << "dx: " << dx << std::endl;
                std::cerr << "dy: " << dy << std::endl;
                int numx = ((int)(((maxx - minx) / dx) + 0.5)) + 1;
                int numy = ((int)(((maxy - miny) / dy) + 0.5)) + 1;
                std::cerr << "numx: " << numx << std::endl;
                std::cerr << "numy: " << numy << std::endl;
            }
            osgTerrain::Terrain *terrain = new osgTerrain::Terrain();
            osg::Group* group=new osg::Group();
            terrain->addChild(group);
            terrain->setName(p.filename().string() + "Terrain");
            terrain->setFormat("WKT");
            terrain->setBlendingPolicy(osgTerrain::TerrainTile::INHERIT);
            terrain->setCoordinateSystem("PROJCS[\"ETRS89 / UTM zone 32N\",GEOGCS[\"ETRS89\",DATUM[\"European_Terrestrial_Reference_System_1989\",SPHEROID[\"GRS 1980\",6378137,298.257222101,AUTHORITY[\"EPSG\",\"7019\"]],AUTHORITY[\"EPSG\",\"6258\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4258\"]],PROJECTION[\"Transverse_Mercator\"],PARAMETER[\"latitude_of_origin\",0],PARAMETER[\"central_meridian\",9],PARAMETER[\"scale_factor\",0.9996],PARAMETER[\"false_easting\",500000],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",EAST],AXIS[\"Northing\",NORTH],AUTHORITY[\"EPSG\",\"25832\"]]");
            group->setName(p.filename().string() + "GeometryGroup");
            int i = 0;
            for (directory_entry& x : directory_iterator(p))
            {
                cout << "    " << x.path() << '\n';
                osg::PagedLOD* plod = new osg::PagedLOD();
                plod->setName(x.path().string() + "PLOD");
                plod->setFileName(0, "");
                plod->setFileName(1, x.path().generic_path().string());
                plod->setCenter(positions[i]);
                plod->setRadius(2000);
                plod->setRangeMode(osg::LOD::DISTANCE_FROM_EYE_POINT);
                plod->setRange(0, distance, 10000000000);
                plod->setRange(1, 0, distance);
                group->addChild(plod);
                osg::Group* dummy = new osg::Group();
                dummy->setName("dummyNode");
                plod->addChild(dummy);
                i++;
            }
            osgDB::writeNodeFile(*terrain, std::string(argv[2]));

        }
        else
            cout << p << " exists, but is not a directory\n";
    }
    else
        cout << p << " does not exist\n";
}
