#include "Changer.h"
#include "Utility.h"
#include <cassert>

Changer::Changer(osg::Node* model, osg::Group* parent)
{
    m_toolModels[End::front] = findTransform(model, "TauscharmAufsatz_1");
    m_toolModels[End::back] = findTransform(model, "TauscharmAufsatz_2");
    assert(m_toolModels[End::front] && m_toolModels[End::back]);
    model->accept(m_animation);
    m_animation.m_am->getAnimationList().front()->setPlayMode(osgAnimation::Animation::ONCE);
    parent->addChild(model);

}

void Changer::play()
{
    m_animation.m_am->playAnimation(m_animation.m_am->getAnimationList().front(), 1, 1); // Play once
}

float Changer::getAnimationDuration() const
{
    return m_animation.m_am->getAnimationList().front()->getDuration();
}

ToolModel *Changer::tool(End end)
{
    return m_tools[end].get();
}

void Changer::giveTool(ToolModel::ptr &&tool, End end)
{
    auto &t = m_tools[end];
    assert(t == nullptr);
    t = std::move(tool);
    if(!t)
        return;
    t->setParent(m_toolModels[end]);

    auto m = t->model();
    m_toolMatrix = m->getMatrix();
    auto mat = m_toolMatrix;
    mat.setTrans(0,0,0);
    m->setMatrix(mat);

}

ToolModel::ptr Changer::takeTool(End end)
{
    auto &t = m_tools[end];
    auto tool = std::move(m_tools[end]);
    m_tools[end] = nullptr;
    if(tool)
        tool->model()->setMatrix(m_toolMatrix);
    return tool;
}
