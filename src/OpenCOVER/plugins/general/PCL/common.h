/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * common.h
 *
 *  Created on: Jul 27, 2012
 *      Author: Adam Stambler
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <point_cloud.h>

#include <osgDB/Options>
#include <osg/Object>

namespace osgpcl
{

/*
   * Inspects the point cloud field list to choose the best default representation
   * Point Clouds with RGB- >  PointCloudRGBFactory
   * Intensity ->  PointCloudIntensityFactory
   * Label ->  PointCloudLabelFactory
   * XYZ   ->  PointCloudCRange over field Z
   */
PointCloudFactory *chooseDefaultRepresentation(const std::vector<pcl::PCLPointField> &flist);

class CloudReaderOptions : public osgDB::Options
{
public:
    META_Object(osgpcl::CloudReaderOptions, CloudReaderOptions);

    CloudReaderOptions(float sampling_rate = 1);
    CloudReaderOptions(PointCloudFactory *factory, float sampling_rate = 1);
    CloudReaderOptions(const CloudReaderOptions &options,
                       const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);

protected:
    //the reader will randomly subsample the cloud at this
    //rate
    float sampling_rate_;

    //Only load these indices of the point cloud
    pcl::IndicesConstPtr indices;

    //generate the point cloud representation using this node factory
    osg::ref_ptr<osgpcl::PointCloudFactory> factory_;

public:
    /*
       * Set the random sampling rate of the point cloud
       * Cannot be used in conjunction with indices
       */
    void setSamplingRate(float sampling_rate)
    {
        sampling_rate_ = sampling_rate;
    };

    /*
       * Get the current random sampling rate
       */
    float getSamplingRate()
    {
        return sampling_rate_;
    }

    /*
       * Get the current point cloud node factory
       */

    const osg::ref_ptr<osgpcl::PointCloudFactory> &getFactory() const
    {
        return factory_;
    }

    /*
       * Sets the point cloud node factory for the reader to use
       */
    void setFactory(const osg::ref_ptr<osgpcl::PointCloudFactory> &factory)
    {
        this->factory_ = factory;
    }

    //Get & Set the indices of the point cloud to load
    const pcl::IndicesConstPtr &getIndices() const
    {
        return indices;
    }
    void setIndices(const pcl::IndicesConstPtr &indices)
    {
        this->indices = indices;
    }
};
}

#endif /* COMMON_H_ */
