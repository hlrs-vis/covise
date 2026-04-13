/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AuralRealityPlugin.h"

#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>
#include <boost/uuid/uuid_io.hpp>

#include <grpc/grpc.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <map>
#include <grpcpp/security/credentials.h>
#include <osg/MatrixTransform>
#include <osg/TexEnv>
#include <osg/Material>
#include <utility>

AuralRealityPlugin *AuralRealityPlugin::plugin = NULL;

using namespace opencover;
namespace ar = auralreality;
namespace ui = opencover::ui;

Speaker::Speaker(const std::string &id)
    : id(id)
    , interactor(osg::Matrix::identity(), 1000, vrui::coInteraction::ButtonA, "hand", "speakerInteractor", vrui::coInteraction::Medium)
{
    interactor.show();
    interactor.enableIntersection();

    offset.makeTranslate(0, 0, 0.4);
    offset_i.invert(offset);

    transform = new osg::MatrixTransform;
    cover->getObjectsRoot()->addChild(transform);

    auto transform2 = new osg::MatrixTransform;
    osg::Matrix m;
    // m.makeScale(0.001, 0.001, 0.001);
    transform2->setMatrix(m);
    transform->addChild(transform2);

    // Attach the speaker icon to the transform
    auto icon = coVRFileManager::instance()->loadFile("share/covise/icons/speaker.glb", nullptr, transform2);

    osg::StateSet *ss = icon->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF | osg::StateAttribute::OVERRIDE);

    osg::ref_ptr<osg::Material> mat = new osg::Material;
    mat->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
    ss->setAttributeAndModes(mat, osg::StateAttribute::ON);
}

void Speaker::preFrame()
{
    interactor.preFrame();

    if (interactor.isRunning())
    {
        osg::Matrix m = offset_i * interactor.getMatrix();
        transform->setMatrix(m);

        AuralRealityPlugin::instance()->pushSpeaker(id);
    }
}

AuralRealityPlugin::AuralRealityPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("AuralRealityPlugin", cover->ui)

{
    plugin = this;

    menu = new ui::Menu("Aural Reality", this);
    menu->setText("Aural Reality");

    auto new_speaker = new ui::Button(menu, "New speaker");
    new_speaker->setCallback([this](bool state)
        { createSpeaker(); });

    // speakers["foo"] = std::make_shared<Speaker>("foo");

    // connect
    channel = grpc::CreateChannel("[::]:9999", grpc::InsecureChannelCredentials());
    service = ar::TMTService::NewStub(channel);

    ar::Id request, response;
    request.set_id("hello");

    grpc::ClientContext context;
    grpc::Status status = service->Ping(&context, request, &response);
    if (status.ok())
    {
        std::cerr << "RPC pong: " << response.id() << std::endl;
    }
    else
    {
        std::cerr << "RPC failed: " << status.error_message() << std::endl
                  << "Error code: " << status.error_code() << std::endl;
    }

    sync();
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
    for (auto &[_, s] : speakers)
        s->preFrame();
}

template <typename Key, typename Value>
bool contains(std::map<Key, Value> map, const Key &key)
{
    return map.find(key) != map.end();
}

void AuralRealityPlugin::sync()
{
    syncSpeakers();
}
void AuralRealityPlugin::syncSpeakers()
{
    ar::Empty request;
    ar::IdList response;
    grpc::ClientContext context;
    grpc::Status status = service->GetSpeakerIds(&context, request, &response);

    if (!status.ok())
    {
        std::cerr << "RPC failed: " << status.error_message() << std::endl
                  << "Error code: " << status.error_code() << std::endl;
        return;
    }

    for (const auto &id : response.ids())
    {
        std::cout << " Found speaker " << id << std::endl;
        if (!contains(speakers, id))
        {
            speakers[id] = std::make_shared<Speaker>(id);
        }
        fetchSpeaker(speakers[id]);
    }

    // TODO: delete removed speakers
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
    }

    float s = t.has_scale() ? t.scale() : 1.0;
    m3.makeScale(s, s, s);

    return m1 * m2 * m3; // TODO: check order ;)
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
    ar::Speaker response;
    request.set_id(speaker->getId());
    grpc::ClientContext context;
    grpc::Status status = service->GetSpeaker(&context, request, &response);

    if (!status.ok())
    {
        std::cerr << "RPC failed: " << status.error_message() << std::endl
                  << "Error code: " << status.error_code() << std::endl;
        return;
    }

    speaker->setTransform(tmt_transform_to_matrix(response.transform()));

    SpeakerProperties p;
    p.dispersion_horizontal = response.dispersion_horizontal();
    p.dispersion_vertical = response.dispersion_vertical();
    p.cutoff_frequency_low = response.cutoff_frequency_low();
    p.cutoff_frequency_high = response.cutoff_frequency_high();
    p.maximum_sound_pressure_level = response.maximum_sound_pressure_level();
    p.power_handling = response.power_handling();
    speaker->setProperties(p);

    std::cout << " Updated speaker " << speaker->getId() << std::endl;
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

    grpc::ClientContext context;
    grpc::Status status = service->UpdateSpeaker(&context, request, &response);

    if (!status.ok())
    {
        std::cerr << "RPC failed: " << status.error_message() << std::endl
                  << "Error code: " << status.error_code() << std::endl;
        return;
    }

    // TODO: parse response again?
}

void AuralRealityPlugin::createSpeaker()
{
    std::string id = boost::uuids::to_string(uuid_generator());
    speakers[id] = std::make_shared<Speaker>(id);
    pushSpeaker(id);
}

COVERPLUGIN(AuralRealityPlugin)
