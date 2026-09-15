/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AuralRealityPlugin.h"
#include "Trajectory.h"
#include "arrpc/arrpc_future.h"
#include "arrpc/arrpc_status.h"
#include "arrpc/message_header.pb.h"
#include "arrpc/quaternion.pb.h"
#include "arrpc/rotation.pb.h"
#include "arrpc/utils.pb.h"
#include "arrpc/vector.pb.h"

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

osg::Quat from_proto(const ar::Quaternion &quat)
{
    return osg::Quat(quat.x(), quat.y(), quat.z(), quat.w());
}

osg::Quat from_proto(const ar::Rotation &r)
{
    switch (r.content_case())
    {
    case ar::Rotation::ContentCase::kEuler:
    {
        auto euler = r.euler();
        return osg::Quat(
            euler.x(), osg::Vec3(1, 0, 0),
            euler.y(), osg::Vec3(0, 1, 0),
            euler.z(), osg::Vec3(0, 0, 1));
    }
    case ar::Rotation::ContentCase::kQuaternion:
    {
        return from_proto(r.quaternion());
    }
    case ar::Rotation::ContentCase::CONTENT_NOT_SET:
    default:
        return osg::Quat();
    }
}
inline osg::Vec3 from_proto(const ar::Vector &v)
{
    return osg::Vec3(v.x(), v.y(), v.z());
}

osg::Matrix from_proto(const ar::Transform &t)
{
    float s = t.has_scale() ? t.scale() : 1.0;
    return osg::Matrix::rotate(from_proto(t.rotation())) * osg::Matrix::translate(from_proto(t.position())) * osg::Matrix::scale(s, s, s);
}

void to_proto(const osg::Vec3 &vec, ar::Vector *out)
{
    out->set_x(vec.x());
    out->set_y(vec.y());
    out->set_z(vec.z());
}
void to_proto(const osg::Quat &quat, ar::Rotation *out)
{
    auto q = out->mutable_quaternion();
    q->set_w(quat.w());
    q->set_x(quat.x());
    q->set_y(quat.y());
    q->set_z(quat.z());
}
void to_proto(const osg::Matrix &m, ar::Transform *result)
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

    socket.connect("tcp://127.0.0.1:17419");

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

    fetchAll();

    // auto t = std::make_shared<Trajectory>();
    // trajectories["test"] = t;
    //
    // for (int i = 3; i < 6; i++)
    // {
    //     auto p = std::make_shared<TrajectoryPoint>(t.get());
    //     p->setTransforms(osg::Matrix::translate(i, 3, 1), osg::Vec3(i, 2, 1.2), osg::Vec3(i, 4, 0.8));
    //     t->points.push_back(p);
    // }
    //
    // t->updateSelection();
    // t->rebuildGeometry();

    // for (auto &[_, s] : speakers)
    //     m_selection.addToSelection(s.get());
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
            std::cerr << "RPC failed: " << res.status().message << std::endl;
            return;
        }
        else
        {
            callback(res.take_value());
        }
    };
}

template <typename T>
std::function<void(rpc::RpcResult<T>)> handleError()
{
    return handleError<T>([](T t) { });
}

void AuralRealityPlugin::fetchAll()
{
    fetchSpeakers();
    fetchTrajectories();
}

void AuralRealityPlugin::fetchSpeakers()
{
    client.GetSpeakerIds().then(handleError<ar::IdList>(
        [&](ar::IdList response)
        {
        for (const auto &id : response.ids())
        {
            if (!contains(speakers, id))
            {
                speakers[id] = std::make_shared<Speaker>(id);
            }
            fetchSpeaker(speakers[id]);

            // TODO: delete removed
        } }));
}

void AuralRealityPlugin::fetchTrajectories()
{
    client.GetTrajectoryIds().then(handleError<ar::IdList>(
        [&](ar::IdList response)
        {
        for (const auto &id : response.ids())
        {
            if (!contains(trajectories, id))
            {
                trajectories[id] = std::make_shared<Trajectory>(id);
            }
            fetchTrajectory(trajectories[id]);

            // TODO: delete removed 
        } }));
}

osg::Matrix tmt_anchor_and_rotation_to_matrix(const ar::Vector &p, const ar::Rotation &r)
{
    osg::Matrix m1;
    osg::Matrix m2;

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

    return m2 * m1;
}

void AuralRealityPlugin::fetchSpeaker(std::shared_ptr<Speaker> speaker)
{
    ar::Id request;
    request.set_id(speaker->getId());

    client.GetSpeaker(request).then(handleError<ar::Speaker>([&, speaker](ar::Speaker response)
        { speaker->setTransform(from_proto(response.transform())); }));
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
    to_proto(speaker->getTransform(), request.mutable_transform());

    client.UpdateSpeaker(request).then(handleError<ar::Speaker>([](ar::Speaker response)
        {
            // TODO: parse response?
        }));
}

void AuralRealityPlugin::fetchTrajectory(std::shared_ptr<Trajectory> trajectory)
{
    ar::Id request;
    request.set_id(trajectory->getId());

    client.GetTrajectory(request).then(handleError<ar::Trajectory>([&, trajectory](ar::Trajectory response)
        {
            trajectory->setTransform(from_proto(response.transform()));
            trajectory->points.clear();
            trajectory->closed = response.closed();

            for (auto point : response.points())
            {
                auto p = std::make_shared<TrajectoryPoint>(trajectory.get());
                p->setTransforms(
                        from_proto(point.anchor()),
                        from_proto(point.control_before()),
                        from_proto(point.control_after()),
                        from_proto(point.rotation())
                    );
                // p->setTime(point.time());
                trajectory->points.push_back(p);
            }

            trajectory->updateSelection();
            trajectory->rebuildGeometry(); }));
}

void AuralRealityPlugin::pushTrajectory(const std::string &id)
{
    if (trajectories.find(id) == trajectories.end())
    {
        return;
    }

    pushTrajectory(trajectories[id].get());
}

void AuralRealityPlugin::pushTrajectory(const Trajectory *trajectory)
{
    ar::Trajectory request, response;
    request.set_id(trajectory->getId());
    to_proto(trajectory->getTransform(), request.mutable_transform());

    request.set_closed(trajectory->closed);
    for (const auto &point : trajectory->points)
    {
        auto p = request.add_points();
        to_proto(point->getAnchor(), p->mutable_anchor());
        to_proto(point->getRotation(), p->mutable_rotation());
        to_proto(point->getControlPointIn(), p->mutable_control_before());
        to_proto(point->getControlPointOut(), p->mutable_control_after());
        // TODO: time
    }

    client.UpdateTrajectory(request).then(handleError<ar::Trajectory>());
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
