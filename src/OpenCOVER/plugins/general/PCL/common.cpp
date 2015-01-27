/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/*
 * common.cpp
 *
 *  Created on: Aug 6, 2012
 *      Author: Adam Stambler
 */

#include <common.h>

osgpcl::CloudReaderOptions::CloudReaderOptions(float sampling_rate)
    : sampling_rate_(sampling_rate)
{
}

osgpcl::CloudReaderOptions::CloudReaderOptions(PointCloudFactory *factory,
                                               float sampling_rate)
    : factory_(factory)
    , sampling_rate_(sampling_rate)
{
}

osgpcl::CloudReaderOptions::CloudReaderOptions(
    const CloudReaderOptions &options, const osg::CopyOp &copyop)
{
    //only supports shallow copies
    //TODO add support for deep copies
    factory_ = options.factory_;
    indices = options.indices;
    sampling_rate_ = options.sampling_rate_;
}

bool field_present(const std::string &name, const std::vector<pcl::PCLPointField> &flist)
{
    for (int i = 0; i < flist.size(); i++)
    {
        if (flist[i].name == name)
            return true;
    }
    return false;
}

osgpcl::PointCloudFactory *osgpcl::chooseDefaultRepresentation(
    const std::vector<pcl::PCLPointField> &flist)
{
    if (field_present("rgb", flist))
    {
        return new PointCloudRGBFactory<pcl::PointXYZ, pcl::RGB>();
    }
    if (field_present("rgba", flist))
    {
        return new PointCloudRGBFactory<pcl::PointXYZ, pcl::RGB>();
    }
    //NOT yet implemented
    if (field_present("intensity", flist))
    {
        //    return NULL; //PointCloudIFactory<pcl::PointXYZ, pcl::Intensity>;
    }
    if (field_present("label", flist))
    {
        return new PointCloudLabelFactory<pcl::PointXYZ, pcl::Label>();
    }
    return new PointCloudCRangeFactory<>("z");
}
