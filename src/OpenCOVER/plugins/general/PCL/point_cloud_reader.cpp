/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * point_cloud_reader.cpp
 *
 *  Created on: Jul 27, 2012
 *      Author: Adam Stambler
 */

#include "osgpcl/point_cloud_reader.h"
#include <boost/filesystem.hpp>

#include <pcl/io/pcd_io.h>
#include <osg/Geode>

#include <osgpcl/common.h>

namespace osgpcl
{
REGISTER_OSGPLUGIN(pcd, PointCloudReader);

PointCloudReader::PointCloudReader()
{
    supportsExtension("pcd", "PCL Point Cloud Format");
}

PointCloudReader::PointCloudReader(const ReaderWriter &rw,
                                   const osg::CopyOp &copyop)
{
    supportsExtension("pcd", "PCL Point Cloud Format");
}

PointCloudReader::~PointCloudReader()
{
    // TODO Auto-generated destructor stub
}

osgDB::ReaderWriter::ReadResult PointCloudReader::readNode(const std::string &filename,
                                                           const osgDB::ReaderWriter::Options *options) const
{
    boost::filesystem::path fpath(filename);

    if (fpath.extension().string() != ".pcd")
    {
        return ReadResult();
    }

    osg::ref_ptr<CloudReaderOptions> coptions = dynamic_cast<CloudReaderOptions *>(const_cast<osgDB::Options *>(options));

    if (coptions == NULL)
    {
        coptions = new CloudReaderOptions(new PointCloudCRangeFactory<>);
    }

    if (!boost::filesystem::exists(fpath))
    {
        return ReadResult(ReaderWriter::ReadResult::FILE_NOT_FOUND);
    }

    pcl::PCLPointCloud2Ptr cloud(new pcl::PCLPointCloud2);

    pcl::PCDReader reader;
    if (reader.read(filename, *cloud) < 0)
    {
        return ReadResult("Failed to read point cloud\n");
    }

    if (coptions->getFactory() == NULL)
    {
        coptions->setFactory(chooseDefaultRepresentation(cloud->fields));
    }

    coptions->getFactory()->setInputCloud(cloud);

    osg::Node *node = coptions->getFactory()->buildNode();
    if (node == NULL)
    {
        return ReadResult("Failed to build point cloud geometry\n");
    }
    node->setName(filename.c_str());

    return node;
}

osgDB::ReaderWriter::Features PointCloudReader::supportedFeatures() const
{
    return FEATURE_READ_NODE;
}

} /* namespace osgPCL */
