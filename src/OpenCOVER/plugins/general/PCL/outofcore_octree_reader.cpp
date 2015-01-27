/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * OutofCoreOctreeReader.cpp
 *
 *  Created on: Aug 4, 2012
 *      Author: Adam Stambler
 */

#include <osgpcl/outofcore_octree_reader.h>

#include <pcl/outofcore/impl/octree_base.hpp>
#include <pcl/outofcore/impl/octree_disk_container.hpp>
#include <pcl/outofcore/impl/octree_base_node.hpp>
#include <pcl/outofcore/impl/octree_ram_container.hpp>

#include <boost/assign.hpp>
#include <boost/assign/std/vector.hpp>
#include <osg/PagedLOD>

#include <osg/Geode>
#include <osg/ShapeDrawable>

using namespace boost::assign;

namespace osgpcl
{
REGISTER_OSGPLUGIN(oct_idx, OutofCoreOctreeReader);

OutOfCoreOctree::~OutOfCoreOctree()
{
}

OutofCoreOctreeReader::OutofCoreOctreeReader()
{
    supportsExtension("oct_idx", "PCL Point Cloud OutofCore Octree Format");
}

OutofCoreOctreeReader::OutofCoreOctreeReader(const OutofCoreOctreeReader &rw,
                                             const osg::CopyOp &copyop)
{
}

OutofCoreOctreeReader::~OutofCoreOctreeReader()
{
}

osgDB::ReaderWriter::Features OutofCoreOctreeReader::supportedFeatures() const
{
    return osgDB::ReaderWriter::FEATURE_READ_NODE;
}

void printBB(std::ostream &cout, OutofCoreOctreeReader::OutOfCoreOptions &opts)
{
    cout << " " << opts.getBBmin()[0] << " " << opts.getBBmin()[1] << " " << opts.getBBmin()[2] << " to  ";
    std::cout << opts.getBBmax()[0] << " " << opts.getBBmax()[1] << " " << opts.getBBmax()[2] << " \n";
}

osgDB::ReaderWriter::ReadResult OutofCoreOctreeReader::readNode(const std::string &fileName,
                                                                const osgDB::ReaderWriter::Options *options) const
{

    boost::filesystem::path fpath(fileName);

    if (fpath.extension().string() != ".oct_idx")
    {
        return ReadResult();
    }

    //Check to see if an OutOfCore Options was given
    //Make sure all the required values are already created ( PointCloudFactory / OutOfCore datastructure)
    //Load/set up anything not already loaded
    osg::ref_ptr<OutOfCoreOptions> coptions = dynamic_cast<OutOfCoreOptions *>(const_cast<osgDB::Options *>(options));
    if (coptions != NULL)
    {
        coptions = new OutOfCoreOptions(*coptions, osg::CopyOp::DEEP_COPY_ALL);
    }
    else if (dynamic_cast<CloudReaderOptions *>(const_cast<osgDB::Options *>(options)) != NULL)
    {
        CloudReaderOptions *cro = dynamic_cast<CloudReaderOptions *>(const_cast<osgDB::Options *>(options));
        coptions = new OutOfCoreOptions(cro->getFactory(), cro->getSamplingRate());
    }
    else
    {
        coptions = new OutOfCoreOptions();
    }

    if (coptions->getOctree() == NULL)
    {
        if (!boost::filesystem::exists(fileName))
            return osgDB::ReaderWriter::ReadResult::FILE_NOT_FOUND;
        OutofCoreOctreeT<pcl::PointXYZ>::OctreePtr ot(new OutofCoreOctreeT<pcl::PointXYZ>::Octree(fileName, true));
        OutofCoreOctreeT<pcl::PointXYZ>::Ptr tree(new OutofCoreOctreeT<pcl::PointXYZ>(ot));
        coptions->init(tree);
    }

    if (coptions->getFactory() == NULL)
    {
        osgpcl::PointCloudCRangeFactory<> *fact = new osgpcl::PointCloudCRangeFactory<>("z");
        fact->setRange(coptions->getBBmin()[2], coptions->getBBmax()[2]);
        coptions->setFactory(fact);
    }

    const osg::Vec3d &bbmin = coptions->getBBmin();
    const osg::Vec3d &bbmax = coptions->getBBmax();
    osg::Vec3d size = coptions->getBBmax() - coptions->getBBmin();
    osg::Vec3d child_size = size / 2; //size of this octants children
    double radius = child_size.length();

    /*
     * If this voxel is supposed to be a leaf, we load the 3D visualization
     * and exit
     */
    if (coptions->isLeaf())
    {
        //   std::cout << "Loaded leaf at depth " << coptions->getDepth() << " \n";
        pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
        if (coptions->getSamplingRate() > 0.999)
        {
            coptions->getOctree()->queryBBIncludes(coptions->getBBmin()._v, coptions->getBBmax()._v, coptions->getDepth(), cloud);
        }
        else
        {
            coptions->getOctree()->queryBBIncludes_subsample(coptions->getBBmin()._v, coptions->getBBmax()._v, coptions->getDepth(), coptions->getSamplingRate(), cloud);
        }
        if (cloud->width * cloud->height == 0)
            return new osg::Node;
        coptions->getFactory()->setInputCloud(cloud);
        return coptions->getFactory()->buildNode();
    }

    //Calculate the bounding boxes/spheres for all of the octants children

    osg::ref_ptr<osg::PagedLOD> lod = new osg::PagedLOD;
    lod->setCenterMode(osg::LOD::USER_DEFINED_CENTER);
    osg::Vec3d center = (bbmax + bbmin) / 2.0f;
    lod->setCenter(center);
    lod->setRadius(radius);

    std::vector<osg::Vec3d> minbbs;
    minbbs += bbmin, bbmin + osg::Vec3d(child_size[0], 0, 0), bbmin + osg::Vec3d(child_size[0], child_size[1], 0),
        bbmin + osg::Vec3d(0, child_size[1], 0), bbmin + osg::Vec3d(0, child_size[1], child_size[2]),
        bbmin + osg::Vec3d(child_size[0], child_size[1], child_size[2]), bbmin + osg::Vec3d(child_size[0], 0, child_size[2]),
        bbmin + osg::Vec3d(0, 0, child_size[2]);

    float child_rad = child_size.length() / 2;
    int cdepth = coptions->getDepth() + 1;

    bool build_children = true;
    if (cdepth >= coptions->getMaxDepth())
        build_children = false;

    if (build_children)
    {
        osg::Group *group = new osg::Group;

        //Load the children as LOD actors so that they will be automatically loaded
        //from the disk when the camera is looking at them
        for (int i = 0; i < 8; i++)
        {
            //todo add some way to check the number of points within a bounding box without actually retrieving them
            osg::PagedLOD *clod = new osg::PagedLOD;

            OutOfCoreOptions *child_opts = new OutOfCoreOptions(*coptions, osg::CopyOp::DEEP_COPY_ALL);

            osg::Vec3d vmax = minbbs[i] + child_size;
            osg::Vec3d ccenter = (vmax + minbbs[i]) / 2.0f;
            child_opts->setBoundingBox(minbbs[i], minbbs[i] + child_size);
            child_opts->setDepth(cdepth, coptions->getMaxDepth());

            clod->setFileName(0, fileName);
            clod->setDatabaseOptions(child_opts);
            clod->setRange(0, 0, child_rad * 3.0f);
            clod->setCenterMode(osg::LOD::USER_DEFINED_CENTER);
            clod->setCenter(ccenter);
            clod->setRadius(radius / 2.0);
            group->addChild(clod);
        }
        //Add the child group to the lod actor
        if (!lod->addChild(group, 0, child_rad * 2))
        {
            std::cout << "Failed to add group \n";
        }
    }

    int rep_id = (build_children) ? 1 : 0; //place the current nodes visualization in the correct LOD slot
    {
        OutOfCoreOptions *child_opts = new OutOfCoreOptions(*coptions, osg::CopyOp::DEEP_COPY_ALL);
        child_opts->setLeaf(true);
        lod->setDatabaseOptions(child_opts);
        lod->setFileName(rep_id, fileName);
    }

    if (coptions->isRoot())
    { //if it is the root node, it should always be visible
        lod->setRange(rep_id, 0, FLT_MAX);
        coptions->setRoot(false);
    }
    else
    {
        lod->setRange(rep_id, 0, radius * 3);
    }
    return lod.get();
}

OutofCoreOctreeReader::OutOfCoreOptions::OutOfCoreOptions(float sample)
    : CloudReaderOptions(sample)
    , isRoot_(true)
    , depth_(0)
    , max_depth_(0)
    , depth_set_(false)
    , bbmin_(0, 0, 0)
    , bbmax_(0, 0, 0)
    , isLeaf_(false)
{
}

OutofCoreOctreeReader::OutOfCoreOptions::OutOfCoreOptions(
    osgpcl::PointCloudFactory *factory, float sample)
    : CloudReaderOptions(factory, sample)
    , isRoot_(true)
    , depth_(0)
    , max_depth_(0)
    , depth_set_(false)
    , bbmin_(0, 0, 0)
    , bbmax_(0, 0, 0)
    , isLeaf_(false)
{
}

OutofCoreOctreeReader::OutOfCoreOptions::OutOfCoreOptions(
    const OutOfCoreOctree::Ptr &_octree, osgpcl::PointCloudFactory *factory)
    : CloudReaderOptions(factory, 1)
    , isRoot_(true)
    , depth_(0)
    , max_depth_(0)
    , depth_set_(false)
    , bbmin_(0, 0, 0)
    , bbmax_(0, 0, 0)
    , isLeaf_(false)
{
    this->init(octree_);
}

bool OutofCoreOctreeReader::OutOfCoreOptions::init(const OutOfCoreOctree::Ptr &_octree)
{
    if (!depth_set_)
    {
        depth_ = 0;
        max_depth_ = _octree->getTreeDepth();
        depth_set_ = true;
    }
    this->octree_ = _octree;
    if (bbmax_ == bbmin_)
    {
        this->octree_->getBoundingBox(bbmin_._v, bbmax_._v);
    }
    return true;
}

void OutofCoreOctreeReader::OutOfCoreOptions::setDepth(boost::uint64_t depth,
                                                       boost::uint64_t max_depth)
{
    depth_set_ = true;
    depth_ = depth;
    max_depth_ = max_depth;
}

bool OutofCoreOctreeReader::OutOfCoreOptions::depthIsSet()
{
    return depth_set_;
}

boost::uint64_t OutofCoreOctreeReader::OutOfCoreOptions::getDepth()
{
    return depth_;
}

boost::uint64_t OutofCoreOctreeReader::OutOfCoreOptions::getMaxDepth()
{
    return max_depth_;
}

bool OutofCoreOctreeReader::OutOfCoreOptions::isRoot()
{
    return isRoot_;
}

void OutofCoreOctreeReader::OutOfCoreOptions::setRoot(bool enable)
{
    isRoot_ = enable;
}

void OutofCoreOctreeReader::OutOfCoreOptions::setBoundingBox(
    const osg::Vec3d &bbmin, const osg::Vec3d &bbmax)
{
    bbmin_ = bbmin;
    bbmax_ = bbmax;
}

OutofCoreOctreeReader::OutOfCoreOptions::OutOfCoreOptions(
    const OutOfCoreOptions &options, const osg::CopyOp &copyop)
    : CloudReaderOptions(options, copyop)
{
    this->bbmax_ = options.bbmax_;
    this->bbmin_ = options.bbmin_;
    this->isRoot_ = options.isRoot_;
    this->depth_ = options.depth_;
    this->max_depth_ = options.max_depth_;
    this->octree_ = options.octree_;
    this->factory_ = options.factory_;
    this->sampling_rate_ = options.sampling_rate_;
    this->isLeaf_ = options.isLeaf_;
    this->depth_set_ = options.depth_set_;
}

void OutofCoreOctreeReader::OutOfCoreOptions::getBoundingBox(
    osg::Vec3d &bbmin, osg::Vec3d &bbmax)
{
    bbmin = bbmin_;
    bbmax = bbmax_;
}
}

/* namespace osgPCL */
