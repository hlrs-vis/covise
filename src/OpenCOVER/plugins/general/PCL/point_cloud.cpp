/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * point_cloud.cpp
 *
 *  Created on: Jun 23, 2012
 *      Author: asher
 */

#include <impl/point_cloud.hpp>
#include <osg/Point>
#include <pcl/point_types.h>

#include <iostream>
#include <osg/Geode>

namespace osgpcl
{

template class PointCloudColoredFactory<pcl::PointXYZ>;
template class PointCloudRGBFactory<pcl::PointXYZ, pcl::RGB>;
template class PointCloudCRangeFactory<pcl::PointXYZ, pcl::PointXYZ>;
template class PointCloudLabelFactory<pcl::PointXYZ, pcl::Label>;
template class PointCloudIFactory<pcl::PointXYZ, pcl::Intensity>;
template class PointCloudNormalFactory<pcl::PointXYZ, pcl::Normal>;
}

osgpcl::PointCloudFactory::PointCloudFactory()
{
    stateset_ = new osg::StateSet;
    osg::Point *p = new osg::Point;
    p->setSize(4);
    stateset_->setAttribute(p);
    stateset_->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
}

void osgpcl::PointCloudFactory::setPointSize(int size)
{
    osg::Point *p = new osg::Point;
    p->setSize(size);
    stateset_->setAttribute(p);
}

osg::Node *osgpcl::PointCloudFactory::buildNode()
{
    osg::Geode *geode = new osg::Geode;
    geode->getDescriptions().push_back("PointCloud");
    osg::Geometry *geom = buildGeometry();
    if (geom == NULL)
        std::cout << "Could not build point cloud\n";
    geode->addDrawable(geom);
    return geode;
}
/* namespace osgPCL */
