/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************

#include "Oct.h"
#include <config/CoviseConfig.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRShader.h>
#include <cover/coVRTui.h>
#include <cover/coVRMSController.h>
#include <cover/coVRConfig.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/AlphaFunc>
#include <osg/Point>
#include <osg/PointSprite>
#include <osg/TemplatePrimitiveFunctor>
#include <osg/TemplatePrimitiveIndexFunctor>
#include <osg/io_utils>
#include <osg/TexEnv>
#include <osg/TexGen>

using namespace osg;
using namespace covise;
using namespace opencover;

constexpr int MAX_POINTS = 30000000;

static FileHandler handler = {nullptr, OctPlugin::load, OctPlugin::unload, "csv"};
OctPlugin *OctPlugin::m_plugin = nullptr;

COVERPLUGIN(OctPlugin)

// Constructor
OctPlugin::OctPlugin()
    : ui::Owner("OCT", cover->ui)
    , m_octMenu("Oct", this)
    , m_colorMenu(&m_octMenu, "ColorMenu")
    , m_coordTerms{{{&m_octMenu, "X"}, {&m_octMenu, "Y"}, {&m_octMenu, "Z"}}}
    , m_colorTerm(&m_octMenu, "Color")
    , m_animationSpeedMulti(&m_octMenu, "AnimationSpeedMultiplier")
    , m_pointSizeSlider(&m_octMenu, "PointSize")
    , m_colorMapSelector(&m_octMenu, "ColorMap")
    , m_reloadBtn(&m_octMenu, "Reload")
    , m_timeScaleIndicator(&m_octMenu, "TimeScaleIndicator")
    , m_delimiter(&m_octMenu, "Delimiter")
    , m_offset(&m_octMenu, "CeaderOffset")
    , m_colorsGroup(&m_octMenu, "Colors")
    , m_colorBar(&m_colorMenu)
{
    m_coordTerms[0].setValue(coCoviseConfig::getEntry("COVER.Plugin.Oct.X"));
    m_coordTerms[1].setValue(coCoviseConfig::getEntry("COVER.Plugin.Oct.Y"));
    m_coordTerms[2].setValue(coCoviseConfig::getEntry("COVER.Plugin.Oct.Z"));
    m_colorTerm.setValue(coCoviseConfig::getEntry("COVER.Plugin.Oct.Color"));
    m_timeScaleIndicator.setValue(coCoviseConfig::getEntry("COVER.Plugin.Oct.TimeScaleIndicator"));
    m_delimiter.setValue(coCoviseConfig::getEntry("COVER.Plugin.Oct.Delimiter"));
    m_offset.setValue(coCoviseConfig::getEntry("COVER.Plugin.Oct.HeaderOffset"));
    if(m_delimiter.value().empty())
        m_delimiter.setValue(";");
    m_animationSpeedMulti.setShared(true);
    m_animationSpeedMulti.setBounds(0, 1000);
    m_animationSpeedMulti.setValue(1);
    m_animationSpeedMulti.setCallback([this](ui::Slider::ValueType value, bool released) {
        if(released && m_pointCloud)
        {
            coVRAnimationManager::instance()->setNumTimesteps(m_pointCloud->getOrCreateVertexBufferObject()->getArray(0)->getNumElements() / value, this);
        }
    });

    m_colorMaps = readColorMaps();
    for(const auto&map : m_colorMaps)
        m_colorMapSelector.append(map.first);
    m_colorMapSelector.select(0);

    m_pointSizeSlider.setBounds(0, 20);
    m_pointSizeSlider.setValue(4);
    m_pointSizeSlider.setCallback([this](ui::Slider::ValueType val, bool release)
                                  {
                                      if (m_pointCloud)
                                      {
                                          dynamic_cast<osg::Point *>(m_pointCloud->getStateSet()->getAttribute(osg::StateAttribute::Type::POINT))->setSize(val);
                                      }
                                  });

    m_reloadBtn.setCallback([this](bool b)
                            {
                                if(m_currentGeode)
                                {
                                    auto parent = m_currentGeode->getParent(0);
                                    auto filename = m_currentGeode->getName();
                                    unloadFile();
                                    load(filename.c_str(), parent, nullptr);
                                }
                            });
}

const OctPlugin *OctPlugin::instance() const
{
    return m_plugin;
}

bool OctPlugin::init()
{
    if (m_plugin)
        return false;
    m_plugin = this;
    m_pointSizeSlider.setValue(coCoviseConfig::getFloat("COVER.Plugin.PointCloud.PointSize", pointSize()));

    coVRFileManager::instance()->registerFileHandler(&handler);
    return true;
}

OctPlugin::~OctPlugin()
{

    coVRFileManager::instance()->unregisterFileHandler(&handler);
}

int OctPlugin::load(const char *filename, osg::Group *loadParent, const char *covise_key)
{
    osg::Group *g = new osg::Group;
    loadParent->addChild(g);
    if (filename != NULL)
    {
        g->setName(filename);
    }
    assert(m_plugin);
    m_plugin->createGeodes(g, filename);
    return 1;
}

int OctPlugin::unload(const char *filename, const char *covise_key)
{
    return m_plugin->unloadFile();
}

void setStateSet(osg::Geometry *geo, float pointSize)
{
  /*  auto stateset = new osg::StateSet();
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateset->setMode(GL_DEPTH_TEST, StateAttribute::ON);
    stateset->setMode(GL_ALPHA_TEST, StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OFF);
    osg::AlphaFunc *alphaFunc = new AlphaFunc(AlphaFunc::GREATER, 0.5);
    stateset->setAttributeAndModes(alphaFunc, StateAttribute::ON);

    auto pointstate = new osg::Point();
    pointstate->setSize(pointSize);
    stateset->setAttributeAndModes(pointstate, StateAttribute::ON);
    osg::PointSprite *sprite = new osg::PointSprite();
    stateset->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::ON);
    geo->setStateSet(stateset);
    */


    // after test move stateset higher up in the tree
    auto* stateset = new StateSet();
    //stateset->setMode(GL_PROGRAM_POINT_SIZE_EXT, StateAttribute::ON);
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateset->setMode(GL_DEPTH_TEST, StateAttribute::ON);
    stateset->setMode(GL_ALPHA_TEST, StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OFF);
    AlphaFunc* alphaFunc = new AlphaFunc(AlphaFunc::GREATER, 0.5);
    stateset->setAttributeAndModes(alphaFunc, StateAttribute::ON);

    osg::Point* pointstate = new osg::Point();
    pointstate->setSize(pointSize);
    stateset->setAttributeAndModes(pointstate, StateAttribute::ON);

    osg::PointSprite* sprite = new osg::PointSprite();
    stateset->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::ON);

    const char* mapName = opencover::coVRFileManager::instance()->getName("share/covise/icons/particle.png");
    if (mapName != NULL)
    {
        osg::Image* image = osgDB::readImageFile(mapName);
        osg::Texture2D* tex = new osg::Texture2D(image);

        tex->setTextureSize(image->s(), image->t());
        tex->setInternalFormat(GL_RGBA);
        tex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        tex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
        stateset->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
        osg::TexEnv* texEnv = new osg::TexEnv;
        texEnv->setMode(osg::TexEnv::MODULATE);
        stateset->setTextureAttributeAndModes(0, texEnv, osg::StateAttribute::ON);

        osg::ref_ptr<osg::TexGen> texGen = new osg::TexGen();
        stateset->setTextureAttributeAndModes(0, texGen.get(), osg::StateAttribute::OFF);

    }
    geo->setStateSet(stateset);
}

bool OctPlugin::compileSymbol(DataTable &symbols, const std::string& symbol, Expression &expr)
{
    expr().register_symbol_table(symbols.symbols());
    if(!expr.parser.compile(symbol, expr()))
    {
        std::cerr << "failed to parse symbol " << symbol << std::endl;
        return false;
    }
    return true;
}

osg::Geometry *OctPlugin::createOsgPoints(DataTable &symbols)
{
    // compile parser
    std::array<Expression, 4> stringExpressions;

    for (size_t i = 0; i < stringExpressions.size(); i++)
    {
        if(!compileSymbol(symbols, i < 3 ? m_coordTerms[i].value() : m_colorTerm.value(), stringExpressions[i]))
            return nullptr;
    }
    //create geometry
    auto geo = new osg::Geometry();
    geo->setUseDisplayList(false);
    geo->setSupportsDisplayList(false);
    geo->setUseVertexBufferObjects(true);
    auto vertexBufferArray = geo->getOrCreateVertexBufferObject();
    auto colors = new Vec4Array();
    auto points = new Vec3Array();


    //calculate coords and color
    std::vector<float> scalarData(symbols.size());
    float minScalar = 0, maxScalar = 0;
    for (size_t i = 0; i < symbols.size(); i++)
    {
        osg::Vec3 coords;
        for (size_t j = 0; j < 3; j++)
        {
            coords[j] = stringExpressions[j]().value();
        }
        points->push_back(coords);
        scalarData[i] = stringExpressions[3]().value();
        minScalar = std::min(minScalar, scalarData[i]);
        maxScalar = std::max(maxScalar, scalarData[i]);
        symbols.advance();
    }
    const auto &colorMap = m_colorMaps[m_colorMapSelector.selectedItem()];

    for (size_t i = 0; i < symbols.size(); i++)
        colors->push_back(covise::getColor(scalarData[i], colorMap, minScalar, maxScalar));

    osg::Vec3 bottomLeft, hpr, offset;
    if (coVRMSController::instance()->isMaster() && coVRConfig::instance()->numScreens() > 0) {
        auto hudScale = covise::coCoviseConfig::getFloat("COVER.Plugin.ColorBar.HudScale", 0.5); // half screen height
        const auto &s0 = coVRConfig::instance()->screens[0];
        hpr = s0.hpr;
        auto sz = osg::Vec3(s0.hsize, 0., s0.vsize);
        osg::Matrix mat;
        MAKE_EULER_MAT_VEC(mat, hpr);
        bottomLeft = s0.xyz - sz * mat * 0.5;
        auto minsize = std::min(s0.hsize, s0.vsize);
        bottomLeft += osg::Vec3(minsize, 0., minsize) * mat * 0.02;
        offset = osg::Vec3(s0.vsize/2.5, 0 , 0) * mat * hudScale;
    }
    m_colorBar.setName("Power");
    m_colorBar.show(true);
    m_colorBar.update(m_colorTerm.value(), minScalar, maxScalar, colorMap.a.size(), colorMap.r.data(), colorMap.g.data(), colorMap.b.data(), colorMap.a.data());
    m_colorBar.setHudPosition(bottomLeft, hpr, offset[0]/480);
    m_colorBar.show(true);


    vertexBufferArray->setArray(0, points);
    vertexBufferArray->setArray(1, colors);
    points->setBinding(osg::Array::BIND_PER_VERTEX);
    colors->setBinding(osg::Array::BIND_PER_VERTEX);
    // bind color per vertex
    geo->setVertexArray(points);
    geo->setColorArray(colors);
    osg::Vec3Array* normals = new osg::Vec3Array;
    normals->push_back(osg::Vec3(0.0f, -1.0f, 0.0f));
    geo->setNormalArray(normals, osg::Array::BIND_OVERALL);
    geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, symbols.size()));

    setStateSet(geo, pointSize());

    return geo;
}

void OctPlugin::createGeodes(Group *parent, const std::string &filename)
{
    auto pointShader = opencover::coVRShaderList::instance()->get("Points");
    int offset = 0;
    try
    {
        offset = std::stoi(m_offset.value());
    }
    catch(const std::exception& e)
    {
        std::cerr << "header offset must be an integer" << std::endl;
        return;
    }

    DataTable dataTable(filename, m_timeScaleIndicator.value(), m_delimiter.value()[0], offset);
    m_pointCloud = createOsgPoints(dataTable);
    m_currentGeode = new osg::Geode();
    m_currentGeode->setName(filename);
    parent->addChild(m_currentGeode);
    if(!m_pointCloud)
        return;
    m_currentGeode->addDrawable(m_pointCloud);
    if (pointShader != nullptr)
    {
        pointShader->apply(m_currentGeode, m_pointCloud);
    }
    coVRAnimationManager::instance()->setNumTimesteps(dataTable.size(), this);
}

void OctPlugin::setTimestep(int t) 
{
    if(m_pointCloud)
    {
        m_pointCloud->setPrimitiveSet(0, new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, (t + 1) * m_animationSpeedMulti.value()));
        std::cerr << "setPrimitiveSet : " << (t + 1) * m_animationSpeedMulti.value() << std::endl;
    }
}

float OctPlugin::pointSize() const
{
    return m_pointSizeSlider.value();
}

int OctPlugin::unloadFile()
{
    if(m_currentGeode && m_currentGeode->getNumParents() > 0)
    {
        m_currentGeode->getParent(0)->removeChild(m_currentGeode);
        m_pointCloud = nullptr;
        m_currentGeode = nullptr;
        return 0;
    }
    return -1;
}

