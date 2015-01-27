/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * point_cloud_reader.h
 *
 *  Created on: Jul 27, 2012
 *      Author: Adam Stambler
 */

#ifndef _OSGPCL_POINT_CLOUD_READER_H_
#define _OSGPCL_POINT_CLOUD_READER_H_

#include <osgpcl/point_cloud.h>

#include <osgDB/ReaderWriter>
#include <osgDB/Options>

#include <osgDB/Registry>

namespace osgpcl
{

/*
 * PointCloudReader
 * osgDB::ReaderWriter implementation for the point cloud library pcd files.
 * Pass a CloudReaderOptions to readNode to choose which type of PointCloudFactory
 * is used to generate the point cloud model.
 */
class PointCloudReader : public osgDB::ReaderWriter
{
public:
    PointCloudReader();
    PointCloudReader(const ReaderWriter &rw, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);
    virtual ~PointCloudReader();

    META_Object(osgpcl, PointCloudReader);

    /** Return available features*/
    virtual Features supportedFeatures() const;

    virtual ReadResult readNode(const std::string &fileName, const osgDB::ReaderWriter::Options *options) const;
};

} /* namespace osgPCL */

//USE_OSGPLUGIN(pcd)

#endif /* POINT_CLOUD_READER_H_ */
