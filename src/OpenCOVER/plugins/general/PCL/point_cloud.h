/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * point_cloud.h
 *
 *  Created on: Jun 23, 2012
 *      Author: Adam Stambler
 *
 *      This file contains factories for generating 3D point cloud
 *      representations from pcl point clouds.
 */

#ifndef _OSGPCL_POINT_CLOUD_H_
#define _OSGPCL_POINT_CLOUD_H_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <osg/Geometry>
#include <osg/Uniform>

#include <boost/any.hpp>
#include <string>
#include <map>
#include <pcl/common/io.h>

#include <pcl/PCLPointCloud2.h>

namespace osgpcl
{

//Convience typedef for point cloud Geometrey.  In the future, it
//this may become a specialized class with convience functions or data
typedef osg::Geometry PointCloudGeometry;

/*
   * PointCloudFactory
   * Abstract base class for Point Cloud 3D model generation
   */
class PointCloudFactory : public osg::Referenced
{
public:
    PointCloudFactory();
    virtual ~PointCloudFactory()
    {
    }

    /*
       * buildGeometry
       * Build the 3D geometry for the point cloud
       *
       * unique_stateset -  controls wheather the state stet ( see OSG StateSets for more information)
       * 					defining things like Point Size are shared between generated models
       */
    virtual PointCloudGeometry *buildGeometry(bool unique_stateset = false) = 0;

    /*
       * buildNode
       *  builds a PointCloud Geometry and packages it in a osg::Node for addition to a scene graph
       */
    virtual osg::Node *buildNode();

private:
    std::map<std::string, boost::any> input_clouds_;

public:
    /*
       * setInputCloud
       * Add a templated input point cloud to the factory for constructing 3d model
       */
    template <typename PointT>
    void setInputCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud)
    {
        input_clouds_[pcl::getFieldsList(*cloud)] = cloud;
    }

    /*
       * setInputCloud
       * Add a  point cloud blob to the factory as an input for constructing 3d model
       */
    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud) = 0;

    /*
       * clearInput
       * remove all factory inputs
       */
    void clearInput()
    {
        input_clouds_.clear();
    }

    /*
       * setIndices
       * select indices within the point clouds that will be used for generating
       * the 3d model
       */
    void setIndices(const pcl::IndicesConstPtr &indices)
    {
        indices_ = indices;
    }

protected:
    osg::ref_ptr<osg::StateSet> stateset_;
    pcl::IndicesConstPtr indices_;

    template <typename PointT>
    typename pcl::PointCloud<PointT>::ConstPtr getInputCloud() const;

    template <typename PointT>
    void addXYZToVertexBuffer(osg::Geometry &, const pcl::PointCloud<pcl::PointXYZ> &cloud) const;

public:
    /*
       * setPointSize
       * set the point size in pixels of the 3D model points
       */
    void setPointSize(int size);
};

/*
   * PointCloudColoredFactory
   * Factory for generated single colored osg models of point clouds.
   */
template <typename PointT = pcl::PointXYZ>
class PointCloudColoredFactory : public PointCloudFactory
{

public:
    PointCloudColoredFactory();

    virtual PointCloudGeometry *buildGeometry(bool unique_stateset = false);

    using PointCloudFactory::setInputCloud;
    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);

    void setColor(float r, float g, float b, float alpha = 1);

private:
};

/*
   * PointCloudCRangeFactory
   * Factory for generating 3d visualizations of point clouds where the colors
   * are mapped to the points via a color table.  The user can select a point field
   * and use the point field value within the max/min range to select a color
   * from the table.  The point field must be defined in the PointTF type.
   *
   * Example:
   * A  vertical cylinder's point cloud extends from z=0 to z=10.
   * By default, the color table maps from red (1,0,0) to white( 1,1,1).
   * If the PointTF is set to field "z", then the points at the bottom of the cloud
   * will be red while the top of the cloud will be white.
   * If the field was set to "x", then the left side of the cylinder would be red,
   * while the right side would be white.
   *
   * The color table is user definable.
   */
template <typename PointTXYZ = pcl::PointXYZ, typename PointTF = pcl::PointXYZ>
class PointCloudCRangeFactory : public PointCloudFactory
{

public:
    PointCloudCRangeFactory(std::string field = "");

    typedef boost::shared_ptr<PointCloudCRangeFactory<PointTXYZ, PointTF> > Ptr;
    typedef boost::shared_ptr<typename pcl::PointCloud<PointTXYZ>::ConstPtr> CloudConstPtr;

    void setField(std::string field);
    void setRange(double min, double max);
    void setColorTable(const std::vector<osg::Vec4> &table);

    virtual PointCloudGeometry *buildGeometry(bool unique_stateset = false);
    void setPointSize(int size);
    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);
    using PointCloudFactory::setInputCloud;

protected:
    std::string field_name_;
    double min_range_, max_range_;
    std::vector<osg::Vec4> color_table_;
};

/*
   * PointCloudRGBFactory
   * generates a osg 3D model of a point cloud with each point colored
   * by the RGB field.
   */
template <typename PointTXYZ = pcl::PointXYZ, typename RGBT = pcl::RGB>
class PointCloudRGBFactory : public PointCloudFactory
{
public:
    virtual PointCloudGeometry *buildGeometry(bool unique_stateset = false);
    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);
    using PointCloudFactory::setInputCloud;
};

/*
   * PointCloudIFactory
   * generates a osg 3D model of a point cloud with each point colored
   * by the Intensity field of the point cloud.
   *
   * The intensity field value from 0-1.0 is mapped to a range of 30-255 grayscale value.
   */
template <typename PointTXYZ, typename IntensityT>
class PointCloudIFactory : public PointCloudFactory
{
public:
    virtual PointCloudGeometry *buildGeometry(bool unique_stateset = false);
    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);
    using PointCloudFactory::setInputCloud;
};

/*
   * PointCloudLabelFactory
   * Factory for generating osg 3D models of point clouds with indexed coloring.
   * Each points color is given by its label number.  The label maps to a user
   * definable color map.  If the label is not in the color map, then either
   * a random color is set for that label or that point is not displayed.
   */
template <typename PointTXYZ, typename LabelT>
class PointCloudLabelFactory : public PointCloudFactory
{

public:
    PointCloudLabelFactory();

    virtual PointCloudGeometry *buildGeometry(bool unique_stateset = false);

    using PointCloudFactory::setInputCloud;
    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);

    /*
     * setColorMap
     * set the color map used for coloring each point label.
     */
    typedef std::map<uint32_t, osg::Vec4f> ColorMap;
    void setColorMap(const ColorMap &color_map);

    /*
     * If the color map does not have a color assigned to the label,
     * the factory generates a random set of colors
     */
    void enableRandomColoring(bool enable);

private:
    std::map<uint32_t, osg::Vec4f> color_map_;
    bool random_coloring_;
};

template <typename PointTXYZ, typename NormalT>
class PointCloudNormalFactory : public PointCloudFactory
{

public:
    PointCloudNormalFactory();

    virtual PointCloudGeometry *buildGeometry(bool unique_stateset = false);

    using PointCloudFactory::setInputCloud;
    virtual void setInputCloud(const pcl::PCLPointCloud2::ConstPtr &cloud);

    void setColor(osg::Vec4f color);

    void setNormalLength(float length)
    {
        nlength = length;
    };

private:
    osg::Vec4f color_;
    float nlength;
};
}

/* namespace osgPCL */
#endif /* POINT_CLOUD_H_ */
