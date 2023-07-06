#include "ImageDescription.h"
#include <iostream>
#include <array>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>

#include <osg/ComputeBoundsVisitor>
using namespace osg;

void ImageDescriptor::open(const std::string& filename)
{
    if(m_file.is_open())
    {
        m_file << "\n]";
        m_file.close();
    }
    m_frame_id = 0;
    if(!filename.empty())
    {
        m_file.open(filename, std::ios::out);
        m_file << "[\n";
    }
}


osg::Matrix objToScreen()
{
    auto &conf = *opencover::coVRConfig::instance();

    osg::Matrix headMat = opencover::cover->getViewerMat();
    osg::Matrix view, proj;

    opencover::VRViewer::Eye eye = opencover::VRViewer::EyeMiddle;
    osg::Vec3 off = opencover::VRViewer::instance()->eyeOffset(eye);
    constexpr int channel = 0;
    const opencover::channelStruct &chan = conf.channels[channel];
    auto &xyz = conf.screens[channel].xyz;
    auto &hpr = conf.screens[channel].hpr;
    float dx = conf.screens[channel].hsize;
    float dz = conf.screens[channel].vsize;
    auto viewProj = opencover::computeViewProjFixedScreen(headMat, off, xyz, hpr, osg::Vec2(dx, dz),
                                                            conf.nearClip(), conf.farClip(), conf.orthographic());
    view = viewProj.first;
    proj = viewProj.second;

    auto cam = conf.channels[0].camera;
    const osg::Matrix &transform = opencover::cover->getXformMat();
    const osg::Matrix &scale =  opencover::cover->getObjectsScale()->getMatrix();
    const osg::Matrix model = scale * transform;
    return model * view * proj;
}

void ImageDescriptor::update()
{
    if(m_file.is_open())
    {
        writeTimestepHeader();
        bool addSpace = false;
        for(const auto &roadUser : m_roadUsers)
        {
            if(addSpace)
                m_file << ",\n";
            addSpace = writeObject(roadUser, objToScreen());
        }
        writeTimestepFooter();
        ++m_frame_id;
    }
}

void ImageDescriptor::writeTimestepHeader()
{
    if(m_frame_id > 0)
        m_file << ",\n";
    m_file << "{\n" <<
                    "\"frame_id\":" << m_frame_id << ", \"frame_time\":" << opencover::cover->frameTime() << ",\n" <<
                    "\"objects\": [\n";
}

void ImageDescriptor::writeTimestepFooter()
{
    m_file << "\n]\n}\n";
}

bool ImageDescriptor::writeObject(const RoadUser &roadUser, const osg::Matrix& objToScreen)
{
    osg::ComputeBoundsVisitor cbv;
    roadUser.transform->accept(cbv);
    osg::BoundingBox bb = cbv.getBoundingBox(); // in local coords.
    auto centerMinusOneToOne = bb.center() * objToScreen;
    osg::Vec3f centerZeroToOne;
    for (size_t i = 0; i < 3; i++)
    {
        centerZeroToOne[i] = (centerMinusOneToOne[i] + 1) / 2;
    }
    osg::Vec3f min, max;
    for (size_t i = 0; i < 3; i++)
    {
        min[i] = max[i] = centerZeroToOne[i] ;
    }

    for (size_t i = 0; i < 8; i++)
    {
        auto corner = bb.corner(i) * objToScreen;
        for (size_t i = 0; i < 3; i++)
        {
            auto c = (corner[i] + 1) / 2;
            min[i] = std::min(min[i], c);
            max[i] = std::max(max[i], c);
        }
    }
    float minSize = 0.03;
    std::array<float, 2> sizes {max[0] - min[0], max[1] - min[1] };
    if ((min[0] < 1 && min[0] > 0 || 
        max[0] < 1 && max[0] > 0 || 
        min[0] < 0 && max[0] > 0) &&
        (min[1] < 1 && min[1] > 0 || 
        max[1] < 1 && max[1] > 0 || 
        min[1] < 0 && max[1] > 0) && 
        sizes[0] > minSize &&
        sizes[1] > minSize)
    {
        for (size_t i = 0; i < 2; i++)
        {
            min[i] = std::max(0.0f, min[0]);
            max[i] = std::min(1.0f, max[0]);
            centerZeroToOne[i] = min[i] + (max[i] - min[i])/2;
        }
        
        m_file << "{\"class_id\":" << roadUser.id << ", \"name\":" << "\"" <<  roadUser.name << "\"" <<
        ", \"relative_coordinates\":{\"center_x\": " << centerZeroToOne.x() << ", \"center_y\":" << centerZeroToOne.y() << ", \"width\":" << sizes[0] << ", \"height\":" << sizes[1] << "}, \"confidence\":1}";
        return true;
    }
    return false;
}



void ImageDescriptor::registerRoadUser(int id, const std::string& name, osg::MatrixTransform* transform)
{
    std::cerr << "registered road user " << name << ":" << id << std::endl;
    auto it = std::find_if(m_roadUsers.begin(), m_roadUsers.end(), [transform](const RoadUser &roadUser){
        return roadUser.transform == transform;
    });
    if (it != m_roadUsers.end())
    {
        std::cerr << "ImageDescriptor warning: duplicate road user registered!" << std::endl;
    }
    else 
    {
        m_roadUsers.emplace_back(RoadUser{id, name, transform});
    }
    
}

void ImageDescriptor::unregisterRoadUser(osg::MatrixTransform* transform)
{
    m_roadUsers.erase(std::remove_if(m_roadUsers.begin(), m_roadUsers.end(), [transform](const RoadUser &roadUser){
            return roadUser.transform == transform;
        }), m_roadUsers.end());
}