#ifndef COVER_PLUGIN_TOOL_MASCHIE_CHANGER_H
#define COVER_PLUGIN_TOOL_MASCHIE_CHANGER_H

#include "Tool.h"
#include "Utility.h"

#include <osg/Node>
#include <osg/MatrixTransform>
#include <array>

class Changer {

    public:
        Changer(osg::Node* model, osg::Group* parent);
        enum End { front, back, LAST};
        void play();
        void update();
        ToolModel *tool(End end);
        void giveTool(ToolModel::ptr &&tool, End end);
        ToolModel::ptr takeTool(End end);
        float getAnimationDuration() const;
    private:
        osg::ref_ptr<osg::Node> m_model;
        std::array<osg::ref_ptr<osg::MatrixTransform>, static_cast<int>(End::LAST)> m_toolModels;
        std::array<ToolModel::ptr, static_cast<int>(End::LAST)> m_tools;
        AnimationManagerFinder m_animation;
        float m_time;
        osg::Matrix m_toolMatrix;
};





#endif // COVER_PLUGIN_TOOL_MASCHIE_CHANGER_H
