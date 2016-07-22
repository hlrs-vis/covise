 
/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include <fstream>
#include <iostream>
#include <ostream>
#include <boost/filesystem.hpp>
#include <GL/glew.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include "PointRayTracerPlugin.h"

using namespace osg;
using namespace visionaray;
PointRayTracerPlugin *PointRayTracerPlugin::plugin = NULL;

//-----------------------------------------------------------------------------

PointRayTracerPlugin::PointRayTracerPlugin()
{
    //create drawable
    m_drawable = new PointRayTracerDrawable;
}

bool PointRayTracerPlugin::init()
{
    if (cover->debugLevel(1)) fprintf(stderr, "\n    new PointRayTracerPlugin\n");

    static const std::string cacheDir = "/var/tmp/";

    //read config
    std::string filename = covise::coCoviseConfig::getEntry("COVER.Plugin.PointRayTracer.Filename");
    if(filename.empty()) filename = "/data/KleinAltendorf/ausschnitte/test_UTM_klein.pts";

    //TODO: we might want to check if the cached BVH was created
    //with the same point size that is required from the config
    float pointSize = covise::coCoviseConfig::getFloat("COVER.Plugin.PointRayTracer.PointSize",0.01f);


    //check if loading from cache is enabled
    bool ignore;
    bool useCache = covise::coCoviseConfig::isOn("value", "COVER.Plugin.PointRayTracer.CacheBinaryFile", false, &ignore);

    //path to binary cache file
    boost::filesystem::path p(filename);
    std::string basename = p.stem().string();
    std::string binaryPath = cacheDir + basename;

    //optionally load data from cache
    bool binaryLoaded = false;
    if (useCache && boost::filesystem::exists(binaryPath))
    {
        std::cout << "Load binary data from " << binaryPath << '\n';
        binaryLoaded = loadBvh(binaryPath);
        std::cout << "Ready\n";
    }

    if (!binaryLoaded)
    {
        //create reader and read data into arrays
        m_reader = new PointReader();
        if(!m_reader->readFile(filename, pointSize, m_points, m_colors, m_bbox, true)) return false;

        //build bvh
        std::cout << "Creating BVH...\n";
        m_host_bvh = visionaray::build<host_bvh_type>(m_points.data(), m_points.size());
        std::cout << "Ready\n";

        if (useCache && !boost::filesystem::exists(binaryPath)) //don't overwrite..
        {
            std::cout << "Storing binary file to " << binaryPath << "...\n";
            storeBvh(binaryPath);
            std::cout << "Ready\n";
        }
    }

    //init geode and add it to the scenegraph
    m_geode = new osg::Geode;
    m_geode->setName("PointRayTracer");
    m_geode->addDrawable(m_drawable);
    opencover::cover->getScene()->addChild(m_geode);

    return true;
}

PointRayTracerPlugin::~PointRayTracerPlugin()
{
    if (cover->debugLevel(1))
        fprintf(stderr, "\n    delete PointRayTracerPlugin\n");

    delete m_reader;

}

void PointRayTracerPlugin::preDraw(osg::RenderInfo &info)
{
    static bool initialized = false;
    if (!initialized)
    {
        m_drawable->initData(m_host_bvh, m_points, m_colors);
        initialized = true;
    }
//    if (cover->debugLevel(1)) fprintf(stderr, "\n    preFrame PointRayTracerPlugin\n");
}

void PointRayTracerPlugin::expandBoundingSphere(osg::BoundingSphere &bs)
{
    m_drawable->expandBoundingSphere(bs);
}

bool PointRayTracerPlugin::loadBvh(std::string filename)
{
    std::ifstream stream;
    stream.open(filename, std::ios::in | std::ios::binary);

    if(!stream.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    uint64_t num_primitives = 0;
    uint64_t num_indices = 0;
    uint64_t num_nodes = 0;
    uint64_t num_colors = 0;

    stream.read((char*)&num_primitives, sizeof(num_primitives));
    stream.read((char*)&num_indices, sizeof(num_indices));
    stream.read((char*)&num_nodes, sizeof(num_nodes));
    stream.read((char*)&num_colors, sizeof(num_colors));

    m_host_bvh.primitives().resize(num_primitives);
    m_host_bvh.indices().resize(num_indices);
    m_host_bvh.nodes().resize(num_nodes);
    m_colors.resize(num_colors);

    stream.read((char*)m_host_bvh.primitives().data(), num_primitives * sizeof(host_bvh_type::primitive_type));
    stream.read((char*)m_host_bvh.indices().data(), num_indices * sizeof(int));
    stream.read((char*)m_host_bvh.nodes().data(), num_nodes * sizeof(bvh_node));
    stream.read((char*)m_colors.data(), num_colors * sizeof(visionaray::vector<3, unorm<8>>));

    return true;
}

bool PointRayTracerPlugin::storeBvh(std::string filename)
{
    std::ofstream stream;
    stream.open(filename, std::ios::out | std::ios::binary);

    if(!stream.is_open())
    {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    uint64_t num_primitives = m_host_bvh.primitives().size();
    uint64_t num_indices = m_host_bvh.indices().size();
    uint64_t num_nodes = m_host_bvh.nodes().size();
    uint64_t num_colors = m_colors.size();

    stream.write((char const*)&num_primitives, sizeof(num_primitives));
    stream.write((char const*)&num_indices, sizeof(num_indices));
    stream.write((char const*)&num_nodes, sizeof(num_nodes));
    stream.write((char const*)&num_colors, sizeof(num_colors));

    stream.write((char const*)m_host_bvh.primitives().data(), num_primitives * sizeof(host_bvh_type::primitive_type));
    stream.write((char const*)m_host_bvh.indices().data(), num_indices * sizeof(int));
    stream.write((char const*)m_host_bvh.nodes().data(), num_nodes * sizeof(bvh_node));
    stream.write((char const*)m_colors.data(), num_colors * sizeof(visionaray::vector<3, unorm<8>>));

    return true;
}

COVERPLUGIN(PointRayTracerPlugin)
