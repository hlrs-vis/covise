/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AuralRealityPlugin.h"
#include "arrpc/arrpc_future.h"
#include "arrpc/arrpc_status.h"
#include "arrpc/message_header.pb.h"
#include "arrpc/utils.pb.h"

#include <absl/strings/str_format.h>
#include <chrono>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>
#include <boost/uuid/uuid_io.hpp>

#include <map>
#include <memory>
#include <osg/PolygonOffset>
#include <osg/ShapeDrawable>
#include <utility>

#include <osg/MatrixTransform>
#include <osg/TexEnv>
#include <osg/Material>
#include <osg/io_utils>

#include <cover/ui/Action.h>

#include <arrpc/message_header.pb.h>
#include <zmq.hpp>
#include <zmq_addon.hpp>

AuralRealityPlugin *AuralRealityPlugin::plugin = NULL;

using namespace opencover;
namespace ar = auralreality;
namespace ui = opencover::ui;

AuralRealityPlugin::AuralRealityPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("AuralRealityPlugin", cover->ui)
    , socket(context, zmq::socket_type::dealer)
    , channel(socket)
    , client(channel)

{
    plugin = this;

    menu = new ui::Menu("Aural Reality", this);
    menu->setText("Aural Reality");

    auto new_speaker = new ui::Action(menu, "New speaker");
    new_speaker->setCallback([this]()
        { createSpeaker(); });

    socket.connect("tcp://127.0.0.1:9999");

    ar::Id request;
    request.set_id("hello");
    ar::Id request2;
    request2.set_id("hello2");

    client.Ping(request).then([](rpc::RpcResult<ar::Id> res)
        {
            auto id = res.value();
            auto i = id.id();
            std::cout << "Ping 1 response: " << i << std::endl; });

    client.Ping(request2).then([](rpc::RpcResult<ar::Id> res)
        {
            auto id = res.value();
            auto i = id.id();
            std::cout << "Ping 2 response: " << i << std::endl; });

    sync();

    auto t = std::make_shared<Trajectory>();
    trajectories["test"] = t;

    for (int i = 3; i < 6; i++)
    {
        auto p = std::make_shared<TrajectoryPoint>(t.get());
        p->setTransforms(osg::Matrix::translate(i, 3, 1), osg::Vec3(i, 2, 1.2), osg::Vec3(i, 4, 0.8));
        t->points.push_back(p);
    }

    t->updateSelection();
    t->rebuildGeometry();

    for (auto &[_, s] : speakers)
        m_selection.addToSelection(s.get());
}

bool AuralRealityPlugin::update()
{
    return false;
}

AuralRealityPlugin::~AuralRealityPlugin()
{
}

void AuralRealityPlugin::preFrame()
{
    channel.poll();

    for (auto &[_, s] : speakers)
        s->preFrame();

    for (auto &[_, t] : trajectories)
        t->preFrame();
}

template <typename Key, typename Value>
bool contains(std::map<Key, Value> map, const Key &key)
{
    return map.find(key) != map.end();
}

template <typename T>
std::function<void(rpc::RpcResult<T>)> handleError(std::function<void(T t)> callback)
{
    return [callback](rpc::RpcResult<T> res)
    {
        if (!res.ok())
        {
            std::cerr << "RPC failed: " << res.status().message << std::endl
                      << "Error code: " << res.status().code << std::endl;
            return;
        }
        else
        {
            callback(res.take_value());
        }
    };
}

void AuralRealityPlugin::sync()
{
    syncSpeakers();
}
void AuralRealityPlugin::syncSpeakers()
{
    client.GetSpeakerIds().then(handleError<ar::IdList>(
        [&](ar::IdList response)
        {
        for (const auto &id : response.ids())
        {
            std::cout << " Found speaker " << id << std::endl;
            if (!contains(speakers, id))
            {
                speakers[id] = std::make_shared<Speaker>(id);
            }
            fetchSpeaker(speakers[id]);

            // TODO: delete removed speakers
        } }));
}

osg::Matrix tmt_transform_to_matrix(const ar::Transform &t)
{
    auto p = t.position();
    auto r = t.rotation();

    osg::Matrix m1;
    osg::Matrix m2;
    osg::Matrix m3;

    m1.makeTranslate(p.x(), p.y(), p.z());

    switch (r.content_case())
    {
    case ar::Rotation::ContentCase::kEuler:
    {
        auto euler = r.euler();
        m2.makeRotate(
            euler.x(), osg::Vec3(1, 0, 0),
            euler.y(), osg::Vec3(0, 1, 0),
            euler.z(), osg::Vec3(0, 0, 1));
        break;
    }
    case ar::Rotation::ContentCase::kQuaternion:
    {
        auto quat = r.quaternion();
        m2.makeRotate(osg::Quat(quat.x(), quat.y(), quat.z(), quat.w()));
        break;
    }
    case ar::Rotation::ContentCase::CONTENT_NOT_SET:
    default:
        m2.makeIdentity();
        break;
    }

    float s = t.has_scale() ? t.scale() : 1.0;
    m3.makeScale(s, s, s);

    return m2 * m1 * m3; // TODO: check order ;)
}

void matrix_to_tmt_transform(const osg::Matrix &m, ar::Transform *result)
{
    osg::Vec3d translation;
    osg::Quat rotation;
    osg::Vec3d scale;
    osg::Quat so;
    m.decompose(translation, rotation, scale, so);

    ar::Vector *position = result->mutable_position();
    position->set_x(translation.x());
    position->set_y(translation.y());
    position->set_z(translation.z());

    ar::Rotation *rot = result->mutable_rotation();
    ar::Quaternion *quat = rot->mutable_quaternion();
    quat->set_x(rotation.x());
    quat->set_y(rotation.y());
    quat->set_z(rotation.z());
    quat->set_w(rotation.w());

    // TODO: scale?
    if (scale.length2() != 1)
    {
        if (scale.x() == scale.y() && scale.x() == scale.z())
        {
            // Only allow uniform scales, ignore otherwise
            result->set_scale(scale.x());
        }
    }
}

void AuralRealityPlugin::fetchSpeaker(std::shared_ptr<Speaker> speaker)
{
    ar::Id request;
    request.set_id(speaker->getId());

    client.GetSpeaker(request).then(handleError<ar::Speaker>([&](ar::Speaker response)
        {
            speaker->setTransform(tmt_transform_to_matrix(response.transform()));

            SpeakerProperties p;
            p.dispersion_horizontal = response.dispersion_horizontal();
            p.dispersion_vertical = response.dispersion_vertical();
            p.cutoff_frequency_low = response.cutoff_frequency_low();
            p.cutoff_frequency_high = response.cutoff_frequency_high();
            p.maximum_sound_pressure_level = response.maximum_sound_pressure_level();
            p.power_handling = response.power_handling();
            speaker->setProperties(p);

            std::cout << " Updated speaker " << speaker->getId() << std::endl; }));
}

void AuralRealityPlugin::pushSpeaker(const std::string &id)
{
    if (speakers.find(id) == speakers.end())
    {
        return;
    }

    pushSpeaker(speakers[id].get());
}

void AuralRealityPlugin::pushSpeaker(const Speaker *speaker)
{
    ar::Speaker request, response;
    request.set_id(speaker->getId());
    matrix_to_tmt_transform(speaker->getTransform(), request.mutable_transform());

    auto p = speaker->getProperties();
    request.set_dispersion_horizontal(p.dispersion_horizontal);
    request.set_dispersion_vertical(p.dispersion_vertical);
    request.set_cutoff_frequency_low(p.cutoff_frequency_low);
    request.set_cutoff_frequency_high(p.cutoff_frequency_high);
    request.set_maximum_sound_pressure_level(p.maximum_sound_pressure_level);
    request.set_power_handling(p.power_handling);

    client.UpdateSpeaker(request).then(handleError<ar::Speaker>([](ar::Speaker response)
        {
            // TODO: parse response?
        }));
}

osg::Matrix unscale(osg::Matrix v)
{
    osg::Vec3d translation;
    osg::Quat rotation;
    osg::Vec3d scale;
    osg::Quat so;
    v.decompose(translation, rotation, scale, so);
    return osg::Matrix::rotate(rotation) * osg::Matrix::translate(translation);
}

void AuralRealityPlugin::createSpeaker()
{
    std::string id = boost::uuids::to_string(uuid_generator());
    speakers[id] = std::make_shared<Speaker>(id);

    // auto m = VRSceneGraph::instance()->getTransform()->getMatrix();
    auto m = cover->getInvBaseMat();
    m = unscale(m);
    // m = osg::Matrix::rotate(m.getRotate()) * osg::Matrix::translate(m.getTrans());
    m.preMultTranslate(osg::Vec3(0, 2, 0));
    speakers[id]->setTransform(m);

    pushSpeaker(id);
}

COVERPLUGIN(AuralRealityPlugin)
