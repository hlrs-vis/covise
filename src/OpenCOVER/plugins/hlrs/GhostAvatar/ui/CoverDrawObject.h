#ifndef COVER_PLUGIN_GHOSTAVATAR_UI_CoverDrawObject_H
#define COVER_PLUGIN_GHOSTAVATAR_UI_CoverDrawObject_H

#include <osg/ref_ptr>
#include <osg/MatrixTransform>
#include <osg/Vec3>
#include <osg/Vec4>

class CoverDrawObject
{
public:
    virtual void draw() = 0;
    void clean();

    void setNodeName(const std::string &name) { m_nodeName = name; }

protected:
    osg::ref_ptr<osg::MatrixTransform> m_objectPtr = nullptr;
    std::string m_nodeName = "CoverDrawObject";

    void addToCover();
};

class CoverLine : public CoverDrawObject
{
public:
    CoverLine();
    CoverLine(const osg::Vec4 &color, float lineWidth, const std::string &nodeName);

    void draw() override;
    void draw(const osg::Vec3 &origin, const osg::Vec3 &end);

    void setOrigin(const osg::Vec3 &origin) { m_origin = origin; }
    void setEnd(const osg::Vec3 &end) { m_end = end; }

    void setColor(const osg::Vec4 &color) { m_color = color; }
    void setLineWidth(float lineWidth) { m_lineWidth = lineWidth; }

private:
    osg::Vec3 m_origin = { 0.0, 0.0, 0.0 };
    osg::Vec3 m_end = { 0.0, 0.0, 0.0 };

    osg::Vec4 m_color = { 0.0, 0.0, 0.0, 1.0 }; // black

    float m_lineWidth = 1.0;
};

class CoverFrame : public CoverDrawObject
{
public:
    CoverFrame();
    CoverFrame(float axisLength, float lineWidth, const std::string& nodeName);

    void draw() override;
    void draw(const osg::Vec3 &origin, const osg::Matrix &frame);

    void setOrigin(const osg::Vec3 &origin) { m_origin = origin; }
    void setFrame(const osg::Matrix &frame) { m_frame = frame; }
    void setAxisLength(float length) { m_axisLength = length; }

    void setXColor(const osg::Vec4 &color) { m_xColor = color; }
    void setYColor(const osg::Vec4 &color) { m_yColor = color; }
    void setZColor(const osg::Vec4 &color) { m_zColor = color; }

    void setLineWidth(float lineWidth) { m_lineWidth = lineWidth; }

private:
    osg::Vec3 m_origin = { 0.0, 0.0, 0.0 };
    osg::Matrix m_frame = osg::Matrix::identity();
    float m_axisLength = 1.0;

    osg::Vec4 m_xColor = { 1.0, 0.0, 0.0, 1.0 }; // red
    osg::Vec4 m_yColor = { 0.0, 1.0, 0.0, 1.0 }; // green
    osg::Vec4 m_zColor = { 0.0, 0.0, 1.0, 1.0 }; // blue

    float m_lineWidth = 1.0;
};

#endif // COVER_PLUGIN_GHOSTAVATAR_UI_CoverDrawObject_H