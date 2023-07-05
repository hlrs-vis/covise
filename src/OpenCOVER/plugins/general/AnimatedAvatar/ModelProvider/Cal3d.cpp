#include "Cal3d.h"

#include <osgCal/CoreModel>


#include <osg/MatrixTransform>

#include <map>
#include <memory>
#include <array>

std::map<std::string, osg::ref_ptr<osgCal::CoreModel>> coreModels;

osg::Node *Cal3dProvider::loadModel(const std::string &filename)
{
    
    auto coreModel = coreModels.find(filename);
    if(coreModel == coreModels.end())
    {
        osg::ref_ptr<osgCal::CoreModel>  cm = new osgCal::CoreModel();
        try
        {
            cm->load(filename);
            coreModel = coreModels.emplace(filename, std::move(cm)).first;
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
            
        }
    }
    if(coreModel == coreModels.end())
        return nullptr;
    m_model = new osgCal::Model;
    m_model->load(coreModel->second.get());
    auto transform = new osg::MatrixTransform;
    transform->addChild(m_model);
    auto scale = osg::Matrix::scale(osg::Vec3f(10, 10, 10));
    auto rot = osg::Matrix::rotate(osg::PI, 0,0,1);
    auto trans = osg::Matrix::translate(osg::Vec3f{0,0,-100});
    transform->setMatrix(trans * scale * rot);

    auto animNames = coreModel->second->getAnimationNames();
    for(const auto &animName : animNames)
        std::cerr << "animeName: " << animName << std::endl;
    m_model->blendCycle(0, 1, 0);
    return m_model;
}

// const auto animations = {"skeleton_idle", "skeleton_strut", "skeleton_walk", "skeleton_jog", "skeleton_wave", "skeleton_shoot_arrow", "skeleton_hiphop"};


void Cal3dProvider::m_playAnimation(Animation animation, float weight, float delay)
{
    constexpr std::array<int, 4> animationConversion = {0, 2, 3, 4}; //convert from enum to the model specific animations of the used test avatar
    
    for (size_t i = 0; i < animationConversion.size(); i++)
    {
        if(i != (int)animation)
        {
            m_model->clearCycle(i, delay);
        }
    }
    int animID = animationConversion[(int)animation];
    m_model->blendCycle(animID, weight, delay);
}