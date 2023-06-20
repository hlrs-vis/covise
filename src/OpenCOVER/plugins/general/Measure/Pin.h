#ifndef MEASURE_PIN_H
#define MEASURE_PIN_H

#include <osg/MatrixTransform>
#include <osg/ref_ptr>

#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/osg/OSGVruiNode.h>
#include <OpenVRUI/sginterface/vruiHit.h>

#include <memory>

#include <cover/ui/Group.h>
#include <cover/ui/VectorEditField.h>

#include <chrono>

class Pin : public vrui::coAction
{
public:
    bool placing = true;
    bool moveMarker = false;

    Pin(double coneSize, int id, int dimensionID, opencover::ui::Group *parent);
    Pin(const Pin &other) = delete;
    Pin(Pin &&other) = delete;
    Pin &operator=(const Pin &) = delete;
    Pin &operator=(Pin &&) = delete;

    virtual ~Pin();
    virtual int hit(vrui::vruiHit *hit);
    virtual void miss();
    void update();
    void setPos(osg::Matrix &mat);
    void resize();
    float getDist(osg::Vec3 &a);
    osg::Matrix getMat() const;
    void setIcon(int i);
    void setConeSize(float size);
    int getDimensionID() const;
    std::chrono::system_clock::time_point getTimeOfLastChange() const;
private:
    double coneSize = 150; 
    const int id = 0, dimensionId = 0;
    osg::ref_ptr<osg::MatrixTransform> pos;
    std::unique_ptr<vrui::OSGVruiNode> vNode;
    std::unique_ptr<vrui::coTrackerButtonInteraction> interactionA; ///< interaction for first button
    bool moveStarted = false;
    osg::ref_ptr<osg::MatrixTransform> sc;
    osg::ref_ptr<osg::Node> geo;
    osg::ref_ptr<osg::Switch> icons;
    osg::Matrix startPos;
    osg::Matrix invStartHand;
    opencover::ui::VectorEditField *positionInput;
    std::chrono::system_clock::time_point timeOfLastChange;


};

#endif // MEASURE_PIN_H
