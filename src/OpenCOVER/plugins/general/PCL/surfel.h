/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * surfel.h
 *
 *  Created on: Sep 4, 2012
 *      Author: asher
 */

#ifndef SURFEL_H_
#define SURFEL_H_

#include <point_cloud.h>

/*
 * EXPERIMENTAL SURFEL IMPLEMENTATIONS
 *
 * Outside use of these factories are not recommended.
 * These factories were implemented to test various ways of rendering
 * surfels.
 */

namespace osgpcl
{
template <typename PointT, typename NormalT, typename RadiusT>
class SurfelFactoryFF : public PointCloudFactory
{
public:
    SurfelFactoryFF();
    ~SurfelFactoryFF()
    {
    }
    using PointCloudFactory::setInputCloud;

    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);

    virtual PointCloudGeometry *buildGeometry(bool unique_state = false);

private:
    Eigen::MatrixXf circle_cache;
};

template <typename PointT, typename NormalT>
class SurfelFactory : public PointCloudFactory
{
public:
    SurfelFactory(float radius = 0.1);
    ~SurfelFactory()
    {
    }
    using PointCloudFactory::setInputCloud;

    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);

    virtual PointCloudGeometry *buildGeometry(bool unique_state = false);
    void setRadius(float radius);
};

template <typename PointT, typename NormalT, typename IntensityT>
class SurfelFactoryI : public PointCloudFactory
{
public:
    SurfelFactoryI(float radius = 0.01);
    ~SurfelFactoryI()
    {
    }
    using PointCloudFactory::setInputCloud;

    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);

    virtual PointCloudGeometry *buildGeometry(bool unique_state = false);
    void setRadius(float radius);
};

template <typename PointT, typename NormalT, typename IntensityT>
class SurfelFactoryFFI : public PointCloudFactory
{
public:
    SurfelFactoryFFI(float radius = 0.01);
    ~SurfelFactoryFFI()
    {
    }
    using PointCloudFactory::setInputCloud;

    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);

    virtual PointCloudGeometry *buildGeometry(bool unique_state = false);
    void setRadius(float radius);

private:
    float radius_;
    Eigen::MatrixXf circle_cache;
};
}

#endif /* SURFEL_H_ */
