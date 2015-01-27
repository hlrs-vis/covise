/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * OutofCoreOctreeReader.h
 *
 *  Created on: Aug 4, 2012
 *      Author: Adam Stambler
 */

#ifndef _OSGPCL_OUTOFCORE_OCTREE_READER_
#define _OSGPCL_OUTOFCORE_OCTREE_READER_

#include <osgpcl/point_cloud.h>
#include <osgpcl/common.h>

#include <osgDB/ReaderWriter>
#include <osgDB/Options>
#include <osgDB/Registry>

#include <osgpcl/outofcore.h>

#include <boost/random.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/bernoulli_distribution.hpp>
#include <pcl/common/io.h>

namespace osgpcl
{

/*
 * OutoFCoreOctree
 * Untemplated Abstract base class wrapper used for querying the templated
 * outofcore Octree structure.
 */
class OutOfCoreOctree
{
public:
    virtual ~OutOfCoreOctree();

    typedef boost::shared_ptr<OutOfCoreOctree> Ptr;

    virtual boost::uint64_t getTreeDepth() const = 0;
    virtual void getBoundingBox(double *min, double *max) = 0;

    virtual void queryBBIncludes(const double min[3], const double max[3], size_t query_depth,
                                 const pcl::PCLPointCloud2::Ptr &dst_blob) const = 0;

    virtual void queryBBIncludes_subsample(const double min[3], const double max[3], size_t query_depth,
                                           float subsample, const pcl::PCLPointCloud2::Ptr &dst_blob) const = 0;
};

/*
 * OutofCoreOctreeT
 * Templated wrapper around the standard PCL outofcore octree datastructure
 */
template <typename PointT>
class OutofCoreOctreeT : public OutOfCoreOctree
{
public:
    typedef pcl::outofcore::OutofcoreOctreeBase<pcl::outofcore::OutofcoreOctreeDiskContainer<PointT>, PointT> Octree;
    typedef boost::shared_ptr<Octree> OctreePtr;
    typedef boost::shared_ptr<OutofCoreOctreeT<PointT> > Ptr;

    OutofCoreOctreeT(const OctreePtr &octree);
    virtual void queryBBIncludes(const double min[3], const double max[3], size_t query_depth,
                                 const pcl::PCLPointCloud2::Ptr &dst_blob) const;
    virtual void queryBBIncludes_subsample(const double min[3], const double max[3], size_t query_depth,
                                           float subsample, const pcl::PCLPointCloud2::Ptr &dst_blob) const;

protected:
    OctreePtr octree_;
    typedef Eigen::Map<const Eigen::Vector3d> ConstVec3dMap;
    typedef Eigen::Map<Eigen::Vector3d> Vec3dMap;

public:
    virtual boost::uint64_t getTreeDepth() const
    {
        return octree_->getDepth();
    };
    virtual void getBoundingBox(double *min, double *max)
    {
        Eigen::Vector3d bmin, bmax;
        octree_->getBoundingBox(bmin, bmax);
        for (int i = 0; i < 3; i++)
        {
            min[i] = bmin[i];
            max[i] = bmax[i];
        }
    }
};

/*
 * OutOfCoreOctreeReader
 * osgDB::Readwriter implementation for PCL OutOfCore trees.  Reads  ".oct_idx" extension
 */
class OutofCoreOctreeReader : public osgDB::ReaderWriter
{
public:
    OutofCoreOctreeReader();
    OutofCoreOctreeReader(const OutofCoreOctreeReader &rw, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);
    virtual ~OutofCoreOctreeReader();

    META_Object(osgPCL, OutofCoreOctreeReader);

    /** Return available features*/
    virtual Features supportedFeatures() const;

    /*
     * readNode
     * Recursively loads each octant outofcore octree data structure into a osg::Node
     * for visualization
     */
    virtual ReadResult readNode(const std::string &fileName, const osgDB::ReaderWriter::Options *options) const;

    /*
     * OutOfCoreOptions
     * Spectialization of CloudReaderOptions to provide additional options specific
     * to OutOfCore point clouds.  This can control which level of detail the
     * system should display or which portion of the point cloud should be displayed.
     */
    class OutOfCoreOptions : public CloudReaderOptions
    {
    public:
        OutOfCoreOptions(float sample = 1.0);
        OutOfCoreOptions(osgpcl::PointCloudFactory *_factory, float sample = 1.0f);
        OutOfCoreOptions(const OutOfCoreOctree::Ptr &_octree,
                         osgpcl::PointCloudFactory *_factory);
        OutOfCoreOptions(const OutOfCoreOptions &options, const osg::CopyOp &copyop = osg::CopyOp::SHALLOW_COPY);

        META_Object(osgpcl::OutofCoreOctreeReader::OutOfCoreOptions, OutOfCoreOptions);

        bool init(const OutOfCoreOctree::Ptr &octree);

        void setDepth(boost::uint64_t depth, boost::uint64_t max_depth);
        bool depthIsSet();

        boost::uint64_t getDepth();
        boost::uint64_t getMaxDepth();

        bool isRoot();
        void setRoot(bool enable);

        void setBoundingBox(const osg::Vec3d &bbmin, const osg::Vec3d &bbmax);
        void getBoundingBox(osg::Vec3d &bbmin, osg::Vec3d &bbmax);

    private:
        bool depth_set_;
        OutOfCoreOctree::Ptr octree_;
        boost::uint64_t depth_;
        boost::uint64_t max_depth_;
        osg::Vec3d bbmin_, bbmax_;
        bool isRoot_;
        bool isLeaf_;

    public:
        OutOfCoreOctree::Ptr getOctree()
        {
            return octree_;
        }
        osgpcl::PointCloudFactory *getFactory()
        {
            return factory_.get();
        }
        const osg::Vec3d &getBBmax()
        {
            return bbmax_;
        }
        const osg::Vec3d &getBBmin()
        {
            return bbmin_;
        }
        bool isLeaf()
        {
            return isLeaf_;
        }
        void setLeaf(bool enable)
        {
            isLeaf_ = enable;
        }
    };
};
}

template <typename PointT>
inline osgpcl::OutofCoreOctreeT<PointT>::OutofCoreOctreeT(
    const OctreePtr &octree)
{
    octree_ = octree;
}

template <typename PointT>
inline void osgpcl::OutofCoreOctreeT<PointT>::queryBBIncludes(
    const double min[3], const double max[3], size_t query_depth,
    const pcl::PCLPointCloud2::Ptr &dst_blob) const
{
    octree_->queryBBIncludes(ConstVec3dMap(min), ConstVec3dMap(max), query_depth, dst_blob);
}

template <typename PointT>
inline void osgpcl::OutofCoreOctreeT<PointT>::queryBBIncludes_subsample(
    const double min[3], const double max[3], size_t query_depth,
    float subsample, const pcl::PCLPointCloud2::Ptr &dst_blob) const
{
    pcl::PCLPointCloud2::Ptr rblob(new pcl::PCLPointCloud2);
    if (subsample > 0.999)
    {
        std::cout << "Querying " << query_depth << " \n";
        octree_->queryBBIncludes(ConstVec3dMap(min), ConstVec3dMap(max), query_depth, dst_blob);
        return;
    }
    else
        octree_->queryBBIncludes(ConstVec3dMap(min), ConstVec3dMap(max), query_depth, rblob);

    std::vector<int> sub_indices;
    sub_indices.resize(rblob->width * rblob->height);

    boost::mt19937 rand_gen(std::time(NULL));
    boost::uniform_int<uint64_t> filedist(0, sub_indices.size() - 1);
    boost::variate_generator<boost::mt19937 &, boost::uniform_int<uint64_t> > filedie(rand_gen, filedist);

    for (int i = 0; i < sub_indices.size(); i++)
    {
        sub_indices[i] = filedie();
    }
    pcl::copyPointCloud(*rblob, sub_indices, *dst_blob);
}

//USE_OSGPLUGIN(oct_idx)

/* namespace osgPCL */
#endif /* OUTOFCOREOCTREEREADER_H_ */
