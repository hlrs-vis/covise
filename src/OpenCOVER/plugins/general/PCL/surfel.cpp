/*
 * surfel.cpp
 *
 *  Created on: Sep 4, 2012
 *      Author: asher
 */

#include <impl/surfel.hpp>
#include <utility_point_types.h>

namespace osgpcl
{
template class SurfelFactory<pcl::PointXYZ, pcl::Normal>;
template class SurfelFactoryI<pcl::PointXYZ, pcl::Normal, pcl::Intensity>;
template class SurfelFactoryFFI<pcl::PointXYZ, pcl::Normal, pcl::Intensity>;
template class SurfelFactory<pcl::PointNormal, pcl::PointNormal>;

template class SurfelFactoryFF<pcl::PointXYZ, pcl::Normal, osgpcl::RadiusPointT>;
}
