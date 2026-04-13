/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AURAL_REALITY_PLUGIN_H
#define _AURAL_REALITY_PLUGIN_H

#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/coInteractor.h>
#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>
#include <osg/MatrixTransform>
#include <map>
#include <memory>
#include <string>
#include <grpc/grpc.h>
#include <grpcpp/channel.h>

#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>

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

class Speaker
{
public:
    Speaker(const std::string &id);
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

private:
    std::string id;
    opencover::coVR3DTransRotInteractor interactor;
    osg::ref_ptr<osg::MatrixTransform> transform;
    SpeakerProperties properties;

    osg::Matrix offset;
    osg::Matrix offset_i;
};

class AuralRealityPlugin : public opencover::coVRPlugin,
                           public opencover::ui::Owner
{
    friend class Speaker;

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

    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<auralreality::TMTService::Stub> service;

    opencover::ui::Menu *menu;

    boost::uuids::random_generator uuid_generator;
};

#endif
