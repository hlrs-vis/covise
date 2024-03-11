#include "EnnovatisDevice.h"

#include <ennovatis/building.h>
#include <ennovatis/rest.h>
#include <utils/threadworker.h>

#include <cover/coVRFileManager.h>

#include <nlohmann/json.hpp>
#include <osg/Material>
#include <osg/ref_ptr>

using namespace opencover;
using json = nlohmann::json;

namespace {
float h = 1.f;
float w = 2.f;
constexpr float default_height = 100.f;
utils::ThreadWorker<std::string> rest_worker;

/**
 * @brief Fetches the channels from a given channel group and building.
 * 
 * This function fetches the channels from the specified channel group and building
 * and populates the REST request object with last used channelid. Results will be available in the rest_worker by accessing futures over threads.
 * 
 * @param group The channel group to fetch channels from.
 * @param b The building to fetch channels from.
 * @param req The REST request object to populate with fetched channels.
 */
void fetchChannels(const ennovatis::ChannelGroup &group, const ennovatis::Building &b, ennovatis::rest_request req)
{
    auto input = b.getChannels(group);
    for (auto &channel: input) {
        req.channelId = channel.id;
        rest_worker.addThread(std::async(std::launch::async, ennovatis::rest::fetch_data, req));
    }
}

/**
 * @brief Adds a cylinder between two points.
 * Source: http://www.thjsmith.com/40/cylinder-between-two-points-opengl-c
 * 
 * @param start The starting point of the cylinder.
 * @param end The ending point of the cylinder.
 * @param radius The radius of the cylinder.
 * @param cylinderColor The color of the cylinder.
 * @param group The group to which the cylinder will be added.
 */
void addCylinderBetweenPoints(osg::Vec3 start, osg::Vec3 end, float radius, osg::Vec4 cyclinderColor, osg::Group *group)
{
    osg::ref_ptr geode = new osg::Geode;
    osg::Vec3 center;
    float height;
    osg::ref_ptr<osg::Cylinder> cylinder;
    osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
    osg::ref_ptr<osg::Material> pMaterial;

    height = (start - end).length();
    center = osg::Vec3((start.x() + end.x()) / 2, (start.y() + end.y()) / 2, (start.z() + end.z()) / 2);

    // This is the default direction for the cylinders to face in OpenGL
    osg::Vec3 z = osg::Vec3(0, 0, 1);

    // Get diff between two points you want cylinder along
    osg::Vec3 p = start - end;

    // Get CROSS product (the axis of rotation)
    osg::Vec3 t = z ^ p;

    // Get angle. length is magnitude of the vector
    double angle = acos((z * p) / p.length());

    // Create a cylinder between the two points with the given radius
    cylinder = new osg::Cylinder(center, radius, height);
    cylinder->setRotation(osg::Quat(angle, osg::Vec3(t.x(), t.y(), t.z())));

    cylinderDrawable = new osg::ShapeDrawable(cylinder);
    geode->addDrawable(cylinderDrawable);

    // Set the color of the cylinder that extends between the two points.
    pMaterial = new osg::Material;
    pMaterial->setDiffuse(osg::Material::FRONT, cyclinderColor);
    geode->getOrCreateStateSet()->setAttribute(pMaterial, osg::StateAttribute::OVERRIDE);

    // Add the cylinder between the two points to an existing group
    group->addChild(geode);
}
} // namespace

EnnovatisDevice::EnnovatisDevice(const ennovatis::Building &building, std::shared_ptr<ennovatis::rest_request> req)
: m_buildingInfo(BuildingInfo(&building)), m_request(req)
{
    m_deviceGroup = new osg::Group();
    m_deviceGroup->setName(m_buildingInfo.building->getId() + ".");

    osg::MatrixTransform *matTrans = new osg::MatrixTransform();
    osg::Matrix mat;
    mat.makeTranslate(
        osg::Vec3(m_buildingInfo.building->getLat(), m_buildingInfo.building->getLon(), default_height + h));
    matTrans->setMatrix(mat);

    m_BBoard = new coBillboard();
    m_BBoard->setNormal(osg::Vec3(0, -1, 0));
    m_BBoard->setAxis(osg::Vec3(0, 0, 1));
    m_BBoard->setMode(coBillboard::AXIAL_ROT);

    matTrans->addChild(m_BBoard);
    m_deviceGroup->addChild(matTrans);
    m_TextGeode = nullptr;
    init(3.f);
}

EnnovatisDevice::~EnnovatisDevice()
{}

osg::Vec4 EnnovatisDevice::getColor(float val, float max)
{
    osg::Vec4 colHigh = osg::Vec4(1, 0.1, 0, 1.0);
    osg::Vec4 colLow = osg::Vec4(0, 1, 0.5, 1.0);
    float valN = val / max;

    osg::Vec4 col(colHigh.r() * valN + colLow.r() * (1 - valN), colHigh.g() * valN + colLow.g() * (1 - valN),
                  colHigh.b() * valN + colLow.b() * (1 - valN), colHigh.a() * valN + colLow.a() * (1 - valN));
    return col;
}

void EnnovatisDevice::fetchData()
{
    if (!m_InfoVisible)
        return;
    fetchChannels(*m_channelGroup, *m_buildingInfo.building, *m_request);
}

void EnnovatisDevice::init(float r)
{
    if (m_geoBars) {
        m_deviceGroup->removeChild(m_geoBars);
        m_geoBars = nullptr;
    }

    m_rad = r;
    w = m_rad * 20;
    h = m_rad * 21;

    // RGB Colors 1,1,1 = white, 0,0,0 = black
    osg::Vec4 color(0.753, 0.443, 0.816, 1.f);
    const osg::Vec3f bottom(m_buildingInfo.building->getLat(), m_buildingInfo.building->getLon(),
                            -m_buildingInfo.building->getHeight());
    osg::Vec3f top(bottom);
    top.z() += default_height;

    addCylinderBetweenPoints(bottom, top, m_rad, color, m_deviceGroup.get());
}

void EnnovatisDevice::update()
{
    if (!m_InfoVisible)
        return;
    if (rest_worker.checkStatus() && rest_worker.poolSize() > 0) {
        m_buildingInfo.channelResponse.clear();
        for (auto &t: rest_worker.threadsList()) {
            try {
                std::string requ = t.get();
                m_buildingInfo.channelResponse.push_back(requ);
            } catch (const std::exception &e) {
                std::cout << e.what() << "\n";
            }
        }
        rest_worker.clear();
        showInfo();
    }
}

void EnnovatisDevice::activate()
{
    if (m_TextGeode) {
        m_BBoard->removeChild(m_TextGeode);
        m_TextGeode = nullptr;
        m_InfoVisible = false;
    } else {
        m_InfoVisible = true;
        fetchData();
    }
}

void EnnovatisDevice::disactivate()
{}

void EnnovatisDevice::showInfo()
{
    osg::ref_ptr<osg::MatrixTransform> matShift = new osg::MatrixTransform();
    osg::Matrix ms;
    int charSize = 2;
    ms.makeTranslate(osg::Vec3(w / 2, 0, h));
    matShift->setMatrix(ms);

    osg::ref_ptr<osgText::Text> textBoxTitle = new osgText::Text();
    textBoxTitle->setAlignment(osgText::Text::LEFT_TOP);
    textBoxTitle->setAxisAlignment(osgText::Text::XZ_PLANE);
    textBoxTitle->setColor(osg::Vec4(1, 1, 1, 1));
    textBoxTitle->setText(m_buildingInfo.building->getName(), osgText::String::ENCODING_UTF8);
    textBoxTitle->setCharacterSize(charSize);
    textBoxTitle->setFont(coVRFileManager::instance()->getFontFile("DroidSans-Bold.ttf"));
    textBoxTitle->setMaximumWidth(w);
    textBoxTitle->setPosition(osg::Vec3(m_rad - w / 2., 0, h * 0.9));

    osg::ref_ptr<osgText::Text> textBoxContent = new osgText::Text();
    textBoxContent->setAlignment(osgText::Text::LEFT_TOP);
    textBoxContent->setAxisAlignment(osgText::Text::XZ_PLANE);
    textBoxContent->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    textBoxContent->setLineSpacing(1.25);
    // textBoxContent->setText(" > name:\n > description:\n > type:\n > unit:", osgText::String::ENCODING_UTF8);
    textBoxContent->setCharacterSize(charSize);
    textBoxContent->setFont(coVRFileManager::instance()->getFontFile(NULL));
    textBoxContent->setMaximumWidth(w * 2. / 3.);
    textBoxContent->setPosition(osg::Vec3(m_rad - w / 2.f, 0, h * 0.75));

    osg::ref_ptr<osgText::Text> textBoxValues = new osgText::Text();
    textBoxValues->setAlignment(osgText::Text::LEFT_TOP);
    textBoxValues->setAxisAlignment(osgText::Text::XZ_PLANE);
    textBoxValues->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    textBoxValues->setLineSpacing(1);

    std::string textvalues("");
    textvalues += m_buildingInfo.building->to_string() + "\n";
    auto channelsIt = m_buildingInfo.building->getChannels(*m_channelGroup).begin();
    const auto &responses = m_buildingInfo.channelResponse;
    for (size_t i = 0; i < m_buildingInfo.channelResponse.size(); ++i) {
        textvalues += (*channelsIt).to_string() + "\n";
        ++channelsIt;
        textvalues += "Response:\n";
        json j = json::parse(responses[i]);
        textvalues += j.dump(4) + "\n";
    }
    std::cout << "TextValues:"
              << "\n";
    std::cout << textvalues << "\n";

    textBoxValues->setText(textvalues);
    textBoxValues->setCharacterSize(charSize);
    textBoxValues->setFont(coVRFileManager::instance()->getFontFile(NULL));
    textBoxValues->setMaximumWidth(w / 3.);
    textBoxValues->setPosition(osg::Vec3(m_rad + w / 6., 0, h * 0.75));

    osg::Vec4 colVec(0., 0., 0., 0.2);
    osg::ref_ptr<osg::Material> mat = new osg::Material();
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, colVec);
    mat->setAmbient(osg::Material::FRONT_AND_BACK, colVec);

    osg::ref_ptr<osg::Box> box = new osg::Box(osg::Vec3(m_rad, 0.04 * m_rad, h / 2.f), w, 0, h);
    osg::ref_ptr<osg::ShapeDrawable> sdBox = new osg::ShapeDrawable(box);
    sdBox->setColor(colVec);
    osg::ref_ptr<osg::StateSet> boxState = sdBox->getOrCreateStateSet();
    boxState->setAttribute(mat.get(), osg::StateAttribute::PROTECTED);
    sdBox->setStateSet(boxState);

    osg::ref_ptr<osg::StateSet> textStateT = textBoxTitle->getOrCreateStateSet();
    textBoxTitle->setStateSet(textStateT);
    osg::ref_ptr<osg::StateSet> textStateC = textBoxContent->getOrCreateStateSet();
    textBoxContent->setStateSet(textStateC);
    osg::ref_ptr<osg::StateSet> textStateV = textBoxValues->getOrCreateStateSet();
    textBoxValues->setStateSet(textStateV);

    osg::ref_ptr<osg::Geode> geo = new osg::Geode();
    geo->setName("TextBox");
    geo->addDrawable(textBoxTitle);
    geo->addDrawable(textBoxContent);
    geo->addDrawable(textBoxValues);
    geo->addDrawable(sdBox);

    matShift->addChild(geo);
    m_TextGeode = new osg::Group();
    m_TextGeode->setName("TextGroup");
    m_TextGeode->addChild(matShift);
    m_BBoard->addChild(m_TextGeode);
}