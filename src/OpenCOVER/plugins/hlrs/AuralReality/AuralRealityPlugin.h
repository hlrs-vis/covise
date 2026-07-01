/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AURAL_REALITY_PLUGIN_H
#define _AURAL_REALITY_PLUGIN_H

#include <PluginUtil/coSensor.h>
#include <cover/coInteractor.h>
#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <map>
#include <memory>
#include <string>
#include <grpc/grpc.h>
#include <grpcpp/channel.h>

#include <cover/ui/Menu.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

#include "CustomTransformInteractor.h"

#include "tmt_service.grpc.pb.h"

struct SpeakerProperties
{
    float dispersion_horizontal = 0.0;
    float dispersion_vertical = 0.0;
    float cutoff_frequency_low = 0.0;
    float cutoff_frequency_high = 0.0;
    float maximum_sound_pressure_level = 0.0;
    float power_handling = 0.0;
};

class Selectable
{
public:
    void select()
    {
        if (m_isSelected)
            return;
        m_isSelected = true;
        updateSelection();
    }
    void deselect()
    {
        if (!m_isSelected)
            return;
        m_isSelected = false;
        updateSelection();
    }

    bool isSelected() const { return m_isSelected; }

protected:
    virtual void updateSelection() { };
    bool m_isSelected = false;
};

class Selection
{
    typedef Selectable *SelectablePtr;

public:
    void selectSingle(SelectablePtr selectable)
    {
        for (auto i : m_selected)
        {
            if (i != selectable)
            {
                i->deselect();
                m_selected.erase(i);
            }
        }

        addToSelection(selectable);
    }

    void removeFromSelection(SelectablePtr selectable)
    {
        if (m_selected.find(selectable) != m_selected.end())
        {
            selectable->deselect();
            m_selected.erase(selectable);
        }
    }

    void toggleSelection(SelectablePtr selectable)
    {
        if (selectable->isSelected())
        {
            removeFromSelection(selectable);
        }
        else
        {
            addToSelection(selectable);
        }
    }

    void addToSelection(SelectablePtr selectable)
    {
        if (m_selected.find(selectable) == m_selected.end())
        {
            selectable->select();
            m_selected.insert(selectable);
        }
    }

    const std::set<SelectablePtr> getSelected() const
    {
        return m_selected;
    }

protected:
    std::set<SelectablePtr> m_selected;
};

class SelectableSensor : public coPickSensor
{
private:
    Selection *selection;
    Selectable *selectable;

public:
    SelectableSensor(Selection *s, Selectable *s2, osg::Node *n)
        : coPickSensor(n, false, vrui::coInteraction::ButtonA, vrui::coInteraction::Medium)
        , selection(s)
        , selectable(s2)
    {
    }
    ~SelectableSensor()
    {
        if (active)
            disactivate();
    }
    void activate() override
    {
        selection->addToSelection(selectable);
    }

    void disactivate() override
    {
        // selection->removeFromSelection(selectable);
    }
};

class Speaker : public Selectable
{
public:
    Speaker(const std::string &id);
    ~Speaker();
    void preFrame();

    const std::string &getId() const
    {
        return id;
    }

    void setTransform(osg::Matrix transform)
    {
        this->transform->setMatrix(transform);
        interactor.updateTransform(offset * transform);
    }
    osg::Matrix getTransform() const
    {
        return transform->getMatrix();
    }

    void setProperties(SpeakerProperties properties)
    {
        this->properties = properties;
    }
    const SpeakerProperties &getProperties() const
    {
        return properties;
    }

protected:
    virtual void updateSelection()
    {
        if (isSelected())
        {
            interactor.show();
            interactor.enableIntersection();
        }
        else
        {
            interactor.hide();
            interactor.disableIntersection();
        }
    }

private:
    std::string id;
    CustomTransformInteractor interactor;
    osg::ref_ptr<osg::MatrixTransform> transform;
    SpeakerProperties properties;

    osg::Matrix offset;
    osg::Matrix offset_i;

    SelectableSensor *sensor;
};

class Trajectory;
class TrajectoryPoint : public Selectable
{
    friend class Trajectory;

public:
    TrajectoryPoint(Trajectory *trajectory_);
    ~TrajectoryPoint();
    void preFrame();

    void setTransforms(const osg::Matrix &anchor_, const osg::Vec3 &controlPointIn_, const osg::Vec3 &controlPointOut_)
    {
        anchor = anchor_;
        controlPointIn = controlPointIn_;
        controlPointOut = controlPointOut_;

        anchorNode->setMatrix(anchor);
        controlPointInNode->setMatrix(osg::Matrix::translate(controlPointIn));
        controlPointOutNode->setMatrix(osg::Matrix::translate(controlPointOut));

        anchorInteractor.updateTransform(anchorNode->getMatrix());
        controlPointInInteractor.updateTransform(controlPointInNode->getMatrix());
        controlPointOutInteractor.updateTransform(controlPointOutNode->getMatrix());
    }

protected:
    virtual void updateSelection()
    {
        if (isSelected())
        {
            anchorInteractor.show();
            anchorInteractor.enableIntersection();

            controlPointInInteractor.show();
            controlPointInInteractor.enableIntersection();

            controlPointOutInteractor.show();
            controlPointOutInteractor.enableIntersection();
        }
        else
        {
            anchorInteractor.hide();
            anchorInteractor.disableIntersection();

            controlPointInInteractor.hide();
            controlPointInInteractor.disableIntersection();

            controlPointOutInteractor.hide();
            controlPointOutInteractor.disableIntersection();
        }
    }

private:
    Trajectory *trajectory;

    osg::ref_ptr<osg::Group> groupNode;
    osg::ref_ptr<osg::MatrixTransform> anchorNode;
    osg::ref_ptr<osg::MatrixTransform> controlPointInNode;
    osg::ref_ptr<osg::MatrixTransform> controlPointOutNode;

    CustomTransformInteractor anchorInteractor;
    CustomTransformInteractor controlPointInInteractor;
    CustomTransformInteractor controlPointOutInteractor;

    osg::Matrix anchor;
    osg::Vec3 controlPointIn;
    osg::Vec3 controlPointOut;

    SelectableSensor *sensor;
};

class Trajectory : public Selectable
{

public:
    Trajectory();
    ~Trajectory();
    void preFrame()
    {
        for (auto &p : points)
        {
            p->preFrame();
        }
    }

    virtual void updateSelection()
    {
        for (auto &p : points)
        {
            if (isSelected())
                p->select();
            else
                p->deselect();
        }
    }

    void pointChanged();
    void rebuildGeometry();

    std::vector<std::shared_ptr<TrajectoryPoint>> points;
    osg::ref_ptr<osg::Geode> linesGeode;
    osg::ref_ptr<osg::Geometry> linesGeometry;
    osg::ref_ptr<osg::Geometry> bezierGeometry;
};

class AuralRealityPlugin : public opencover::coVRPlugin,
                           public opencover::ui::Owner
{
    friend class Speaker;
    friend class Trajectory;
    friend class TrajectoryPoint;

public:
    AuralRealityPlugin();
    ~AuralRealityPlugin();
    static AuralRealityPlugin *instance() { return plugin; };
    virtual void preFrame() override;

private:
    static AuralRealityPlugin *plugin;
    virtual bool update() override;

    void sync();
    void syncSpeakers();
    void fetchSpeaker(std::shared_ptr<Speaker> speaker);
    void pushSpeaker(const Speaker *speaker);
    void pushSpeaker(const std::string &id);
    void createSpeaker();

    std::map<std::string, std::shared_ptr<Speaker>> speakers;
    std::map<std::string, std::shared_ptr<Trajectory>> trajectories;

    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<auralreality::TMTService::Stub> service;

    opencover::ui::Menu *menu;

    boost::uuids::random_generator uuid_generator;
    Selection selection;
};

#endif
