/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/****************************************************************************\
**                                                            (C)2005 HLRS  **
**                                                                          **
** Description: PCLParticle Viewer                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                 **
**                                                                          **
** History:  								                                 **
** April-05  v1	    				       		                         **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "PCLPlugin.h"
#define USE_MATH_DEFINES
#include <math.h>
#include <QDir>
#include <config/coConfig.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Array>
#include <osg/Material>
#include <osg/PrimitiveSet>
#include <osg/LineWidth>

#include <osg/LineSegment>
#include <osg/Matrix>
#include <osg/Vec3>
#include <cover/coVRAnimationManager.h>

#include <impl/surfel.hpp>
#include <point_cloud.h>

#include <boost/filesystem.hpp>

#include <osgDB/ReaderWriter>
#include <osgDB/Options>

#include <osgDB/Registry>
#include <pcl/io/pcd_io.h>
#include "common.h"
using namespace osg;
using namespace osgUtil;

PCLPlugin *PCLPlugin::thePlugin = NULL;

PCLPlugin *PCLPlugin::instance()
{
    if (!thePlugin)
        thePlugin = new PCLPlugin();
    return thePlugin;
}

PCLPlugin::PCLPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    //positions=NULL;
    thePlugin = this;
}

static FileHandler handlers[] = {
    { NULL,
      PCLPlugin::sloadPCD,
      PCLPlugin::unloadPCD,
      "pcd" },
    { NULL,
      PCLPlugin::sloadOCT,
      PCLPlugin::unloadOCT,
      "oct_idx" }
};

int PCLPlugin::sloadPCD(const char *filename, osg::Group *loadParent, const char *)
{

    instance()->loadPCD(filename, loadParent);
    return 0;
}

int PCLPlugin::loadPCD(const char *filename, osg::Group *loadParent)
{
    boost::filesystem::path fpath(filename);

    if (fpath.extension().string() != ".pcd")
    {
        return -1;
    }
    osgDB::Options *options = new osgDB::Options();

    osg::ref_ptr<osgpcl::CloudReaderOptions> coptions = dynamic_cast<osgpcl::CloudReaderOptions *>(const_cast<osgDB::Options *>(options));

    if (coptions == NULL)
    {
        coptions = new osgpcl::CloudReaderOptions(new osgpcl::PointCloudCRangeFactory<>);
    }

    if (!boost::filesystem::exists(fpath))
    {
        return -1;
    }

    pcl::PCLPointCloud2Ptr cloud(new pcl::PCLPointCloud2);

    pcl::PCDReader reader;
    if (reader.read(filename, *cloud) < 0)
    {
        return -1;
    }

    if (coptions->getFactory() == NULL)
    {
        coptions->setFactory(osgpcl::chooseDefaultRepresentation(cloud->fields));
    }

    osgpcl::PointCloudFactory *pclf = osgpcl::chooseDefaultRepresentation(cloud->fields);
    pclf->setInputCloud(cloud);
    osg::Node *node = pclf->buildNode();
    if (node == NULL)
    {
        //return ReadResult("Failed to build point cloud geometry\n");
        return -1;
    }
    node->setName(filename);
    osg::Group *parentNode = loadParent;
    if (parentNode == NULL)
        parentNode = cover->getObjectsRoot();
    parentNode->addChild(node);
    ;
    return 0;
}
int PCLPlugin::sloadOCT(const char *filename, osg::Group *loadParent, const char *)
{

    instance()->loadOCT(filename, loadParent);
    return 0;
}

int PCLPlugin::loadOCT(const char *filename, osg::Group *loadParent)
{
    boost::filesystem::path fpath(filename);

    if (fpath.extension().string() != ".oct_idx")
    {
        return -1;
    }

    /*
    //Check to see if an OutOfCore Options was given
    //Make sure all the required values are already created ( PointCloudFactory / OutOfCore datastructure)
    //Load/set up anything not already loaded
    osg::ref_ptr< OutOfCoreOptions>   coptions = dynamic_cast< OutOfCoreOptions*>( const_cast<osgDB::Options*>(options) );
    if (coptions  != NULL){
      coptions = new OutOfCoreOptions( *coptions, osg::CopyOp::DEEP_COPY_ALL);
    }
    else if ( dynamic_cast< CloudReaderOptions*>( const_cast<osgDB::Options*>(options) ) !=NULL){
      CloudReaderOptions* cro = dynamic_cast< CloudReaderOptions*>( const_cast<osgDB::Options*>(options) );
      coptions = new OutOfCoreOptions(cro->getFactory(), cro->getSamplingRate());
    }
    else{
      coptions = new OutOfCoreOptions();
      }

    if (coptions->getOctree() == NULL){
      if ( ! boost::filesystem::exists(fileName))  return osgDB::ReaderWriter::ReadResult::FILE_NOT_FOUND;
      OutofCoreOctreeT<pcl::PointXYZ>::OctreePtr ot (new OutofCoreOctreeT<pcl::PointXYZ>::Octree(fileName, true));
      OutofCoreOctreeT<pcl::PointXYZ>::Ptr tree (new OutofCoreOctreeT<pcl::PointXYZ>(ot));
      coptions->init( tree);
    }

    if (coptions->getFactory() == NULL){
      osgpcl::PointCloudCRangeFactory<>* fact = new osgpcl::PointCloudCRangeFactory<>("z");
      fact->setRange(coptions->getBBmin()[2], coptions->getBBmax()[2]);
      coptions->setFactory(fact);
    }



    const osg::Vec3d & bbmin =coptions->getBBmin();
    const osg::Vec3d& bbmax =coptions->getBBmax();
    osg::Vec3d size = coptions->getBBmax() -coptions->getBBmin();
    osg::Vec3d child_size = size/2; //size of this octants children
    double radius = child_size.length();

    //
    // If this voxel is supposed to be a leaf, we load the 3D visualization
    // and exit
    //
    if (coptions->isLeaf()){
   //   std::cout << "Loaded leaf at depth " << coptions->getDepth() << " \n";
      pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2);
      if (coptions->getSamplingRate() > 0.999){
        coptions->getOctree()->queryBBIncludes(coptions->getBBmin()._v, coptions->getBBmax()._v,coptions->getDepth(), cloud);
      }
      else{
        coptions->getOctree()->queryBBIncludes_subsample(coptions->getBBmin()._v, coptions->getBBmax()._v,coptions->getDepth(), coptions->getSamplingRate(), cloud);
      }
      if (cloud->width*cloud->height == 0 ) return new osg::Node;
      coptions->getFactory()->setInputCloud(cloud);
      //return coptions->getFactory()->buildNode();
    }


    //Calculate the bounding boxes/spheres for all of the octants children

    osg::ref_ptr<osg::PagedLOD> lod = new osg::PagedLOD;
    lod->setCenterMode( osg::LOD::USER_DEFINED_CENTER );
    osg::Vec3d center = (bbmax + bbmin)/2.0f ;
    lod->setCenter( center );
    lod->setRadius( radius );

    std::vector<osg::Vec3d > minbbs;
    minbbs += bbmin, bbmin+ osg::Vec3d(child_size[0],0,0), bbmin+ osg::Vec3d(child_size[0],child_size[1],0),
        bbmin+ osg::Vec3d(0,child_size[1],0), bbmin+ osg::Vec3d(0, child_size[1], child_size[2]),
        bbmin+ osg::Vec3d(child_size[0], child_size[1], child_size[2]),  bbmin+osg::Vec3d(child_size[0], 0 , child_size[2]),
        bbmin+osg::Vec3d(0, 0, child_size[2]);

    float child_rad = child_size.length()/2;
     int cdepth = coptions->getDepth()+1;

     bool build_children =true;
    if (cdepth >= coptions->getMaxDepth()) build_children = false;



    if (build_children) {
      osg::Group* group = new osg::Group;


      //Load the children as LOD actors so that they will be automatically loaded
      //from the disk when the camera is looking at them
      for(int i=0; i<8; i++){
        //todo add some way to check the number of points within a bounding box without actually retrieving them
        osg::PagedLOD* clod = new osg::PagedLOD;

        OutOfCoreOptions* child_opts = new OutOfCoreOptions(*coptions, osg::CopyOp::DEEP_COPY_ALL);

        osg::Vec3d vmax = minbbs[i]+child_size;
        osg::Vec3d ccenter = (vmax+ minbbs[i])/2.0f;
        child_opts->setBoundingBox( minbbs[i],  minbbs[i]+child_size);
        child_opts->setDepth(cdepth, coptions->getMaxDepth());

        clod->setFileName(0, filename);
        clod->setDatabaseOptions(child_opts);
        clod->setRange(0,0,child_rad*3.0f);
        clod->setCenterMode( osg::LOD::USER_DEFINED_CENTER );
        clod->setCenter( ccenter );
        clod->setRadius( radius/2.0 );
        group->addChild(clod);
      }
      //Add the child group to the lod actor
      if (! lod->addChild(group,0, child_rad*2)){
        std::cout << "Failed to add group \n";
      }
    }


    int rep_id = (build_children) ?  1 :0; //place the current nodes visualization in the correct LOD slot
    {
      OutOfCoreOptions* child_opts = new OutOfCoreOptions(*coptions, osg::CopyOp::DEEP_COPY_ALL);
      child_opts->setLeaf(true);
      lod->setDatabaseOptions( child_opts);
      lod->setFileName(rep_id, fileName);
    }


    if(coptions->isRoot()){ //if it is the root node, it should always be visible
      lod->setRange(rep_id,  0, FLT_MAX);
        coptions->setRoot(false);
    }
    else{
      lod->setRange(rep_id, 0, radius*3);
    }
    lod->setName(filename);
	osg::Group *parentNode = loadParent;
	if(parentNode==NULL)
		parentNode = cover->getObjectsRoot();
	parentNode->addChild(lod.get());;*/

    return -1;
}

int PCLPlugin::unloadPCD(const char *filename, const char *)
{
    (void)filename;

    return 0;
}

int PCLPlugin::unloadOCT(const char *filename, const char *)
{
    (void)filename;

    return 0;
}

bool PCLPlugin::init()
{
    fprintf(stderr, "PCLPlugin::PCLPlugin\n");

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);

    return true;
}

// this is called if the plugin is removed at runtime
PCLPlugin::~PCLPlugin()
{
    fprintf(stderr, "PCLPlugin::~PCLPlugin\n");

    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);

    /*if(geode->getNumParents() > 0)
   {
	   parentNode = geode->getParent(0);
	   if(parentNode)
		   parentNode->removeChild(geode.get());
   }*/
}

void
PCLPlugin::preFrame()
{
}

COVERPLUGIN(PCLPlugin)
