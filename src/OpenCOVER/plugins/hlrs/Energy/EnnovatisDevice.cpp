#include "EnnovatisDevice.h"

#include "ennovatis/building.h"
#include "ennovatis/rest.h"
#include "utils/threadworker.h"

#include <cover/coVRFileManager.h>

#include <nlohmann/json.hpp>
#include <osg/Material>
#include <osg/Group>
#include <osg/ref_ptr>

using namespace opencover;
using json = nlohmann::json;

namespace {
float h = 1.f;
float w = 2.f;
constexpr float default_height = 1000.f;
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
void fetchChannels(const ennovatis::ChannelGroup &group, const ennovatis::Building &b, ennovatis::RESTRequest req)
{
    auto input = b.getChannels(group);
    for (auto &channel: input) {
        req.channelId = channel.id;
        rest_worker.addThread(std::async(std::launch::async, ennovatis::fetchEnnovatisData, req));
    }
}
} // namespace

EnnovatisDevice::EnnovatisDevice(const ennovatis::Building &building)
: m_buildingInfo(BuildingInfo(&building))
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

void EnnovatisDevice::fetchData(const ennovatis::ChannelGroup &group, const ennovatis::RESTRequest &req)
{
    fetchChannels(group, *m_buildingInfo.building, req);
}

void EnnovatisDevice::init(float r)
{
    if (m_geoBars) {
        m_deviceGroup->removeChild(m_geoBars);
        m_geoBars = nullptr;
    }

    m_rad = r;
    w = m_rad * 10;
    h = m_rad * 11;

    osg::ref_ptr<osg::Cylinder> cyl = new osg::Cylinder(
        osg::Vec3(m_buildingInfo.building->getLat(), m_buildingInfo.building->getLon(), default_height/2), m_rad, -default_height);
    osg::Vec4 colVec(0.1, 0.1, 0.1, 1.f);
    osg::Vec4 colVecLimit(1.f, 1.f, 1.f, 1.f);

    osg::ref_ptr<osg::ShapeDrawable> shapeD = new osg::ShapeDrawable(cyl);
    osg::ref_ptr<osg::Material> mat = new osg::Material();
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, colVec);
    mat->setAmbient(osg::Material::FRONT_AND_BACK, colVec);
    mat->setEmission(osg::Material::FRONT_AND_BACK, colVec);
    mat->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    mat->setColorMode(osg::Material::EMISSION);

    osg::ref_ptr<osg::StateSet> state = shapeD->getOrCreateStateSet();
    state->setAttribute(mat.get(), osg::StateAttribute::PROTECTED);
    state->setNestRenderBins(false);

    shapeD->setStateSet(state);
    shapeD->setUseDisplayList(false);
    shapeD->setColor(colVec);

    m_geoBars = new osg::Geode();
    m_geoBars->setName(m_buildingInfo.building->getId());
    m_geoBars->addDrawable(shapeD);

    m_deviceGroup->addChild(m_geoBars.get());
}

void EnnovatisDevice::update()
{
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
    } else
        m_InfoVisible = true;
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
    textBoxContent->setText(" > name:\n > description:\n > type:\n > unit:", osgText::String::ENCODING_UTF8);
    textBoxContent->setCharacterSize(charSize);
    textBoxContent->setFont(coVRFileManager::instance()->getFontFile(NULL));
    textBoxContent->setMaximumWidth(w * 2. / 3.);
    textBoxContent->setPosition(osg::Vec3(m_rad - w / 2.f, 0, h * 0.75));

    osg::ref_ptr<osgText::Text> textBoxValues = new osgText::Text();
    textBoxValues->setAlignment(osgText::Text::LEFT_TOP);
    textBoxValues->setAxisAlignment(osgText::Text::XZ_PLANE);
    textBoxValues->setColor(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    textBoxValues->setLineSpacing(1.25);

    std::string textvalues("");
    textvalues += m_buildingInfo.building->to_string() + "\n";
    // TODO: iterate through channelResponse and display the data in a text box
    for (auto &channel: m_buildingInfo.channelResponse) {
        json j = json::parse(channel);
        textvalues += j.dump(4);
        // textvalues += " > " + j["name"].get<std::string>() + ":\n";
        // textvalues += " > " + j["description"].get<std::string>() + ":\n";
        // textvalues += " > " + j["type"].get<std::string>() + ":\n";
        // textvalues += " > " + j["unit"].get<std::string>() + ":\n";
    }

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