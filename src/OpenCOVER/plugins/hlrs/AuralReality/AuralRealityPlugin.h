/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AURAL_REALITY_PLUGIN_H
#define _AURAL_REALITY_PLUGIN_H

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
#include "Selection.h"
#include "Speaker.h"
#include "Trajectory.h"

#include "tmt_service.grpc.pb.h"

class AuralRealityPlugin : public opencover::coVRPlugin,
                           public opencover::ui::Owner
{
public:
    AuralRealityPlugin();
    ~AuralRealityPlugin();
    static AuralRealityPlugin *instance() { return plugin; };
    virtual void preFrame() override;

    Selection &selection() { return m_selection; }

    void sync();
    void syncSpeakers();
    void fetchSpeaker(std::shared_ptr<Speaker> speaker);
    void pushSpeaker(const Speaker *speaker);
    void pushSpeaker(const std::string &id);
    void createSpeaker();

private:
    static AuralRealityPlugin *plugin;
    virtual bool update() override;

    std::map<std::string, std::shared_ptr<Speaker>> speakers;
    std::map<std::string, std::shared_ptr<Trajectory>> trajectories;

    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<auralreality::TMTService::Stub> service;

    opencover::ui::Menu *menu;

    boost::uuids::random_generator uuid_generator;
    Selection m_selection;
};

#endif
