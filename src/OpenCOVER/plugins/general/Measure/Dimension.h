#ifndef MEASURE_DIMENSION_H
#define MEASURE_DIMENSION_H

#include "Pin.h"

#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Switch>
#include <osgText/Text>

#include <array>
#include <memory>

#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/Slider.h>
#include <cover/coVRPlugin.h>

class Measure;

class Scaling
{
public:
    Scaling(opencover::coVRPlugin *plugin, opencover::ui::Group *parent);
    Scaling(const Scaling&) = delete;
    Scaling(Scaling&&) = default;
    Scaling &operator=(const Scaling &other);
    Scaling &operator=(Scaling&&) = default;
    ~Scaling();

    float coneSize() const;
    float fontFactor() const;
    float lineWidth() const;
    void setCallback(const std::function<void()> &onScalingChanged);
private:
    opencover::coVRPlugin *m_plugin;
    opencover::ui::Slider *m_coneSize;
    opencover::ui::Slider *m_fontFactor;
    opencover::ui::Slider *m_lineWidth;

    std::function<void()> m_callback;
    
};





/** Text field that points to the user*/
class TextBoard
{
public:
    TextBoard();
    ~TextBoard();
    void setText(const std::string &text);
    void setPosition(const osg::Matrix &pos);
    void setPosition(const osg::Vec3f &pos);
private:
    osg::ref_ptr<osgText::Text> m_text;
    osg::ref_ptr<osg::MatrixTransform> m_transform;
};

class Dimension
{
public:

    Dimension(int id, const std::string &name, opencover::coVRPlugin *plugin, opencover::ui::Group *parent, const Scaling &scale);
    Dimension(const Dimension &other) = delete;
    Dimension(Dimension &&other) = default;
    Dimension &operator=(const Dimension &) = delete;
    Dimension &operator=(Dimension &&) = default;

    virtual ~Dimension() = default;
    virtual bool isplaced();
    virtual void update();
    int getID()
    {
        return m_id;
    };
    bool isSelected();
    void setScaling(const Scaling &scaling);
    std::chrono::system_clock::time_point getLastChange() const;

private:
    std::chrono::system_clock::time_point m_lastPinCheck;
    std::vector<std::unique_ptr<Pin>> m_pins; // cones to mark the measurement
    opencover::coVRPlugin *m_plugin;


protected:
    std::unique_ptr<opencover::ui::Menu> m_gui;
    Scaling m_scale;
    int m_id;
    bool placing = false;
    double oldDist = 0.0;
    virtual void scalingChanged();
    osg::Matrix getPinPos(int index);
    void addPin();
    bool pinPlacing();
    bool pinMoving();
};


#endif // MEASURE_DIMENSION_H
