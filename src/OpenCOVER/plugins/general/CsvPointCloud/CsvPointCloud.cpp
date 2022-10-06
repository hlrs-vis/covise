/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************

#include "CsvPointCloud.h"
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

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/VrmlNamespace.h>

#include <boost/filesystem.hpp>

using namespace osg;
using namespace covise;
using namespace opencover;
using namespace vrml;

constexpr int MAX_POINTS = 30000000;

static FileHandler handler = {nullptr, CsvPointCloudPlugin::load, CsvPointCloudPlugin::unload, "csv"};
CsvPointCloudPlugin *CsvPointCloudPlugin::m_plugin = nullptr;

COVERPLUGIN(CsvPointCloudPlugin)


class MachineNode;
std::vector<MachineNode*> machineNodes;

static VrmlNode* creator(VrmlScene* scene);

class PLUGINEXPORT MachineNode : public vrml::VrmlNodeChild
{
public:
    static VrmlNode* creator(VrmlScene* scene)
    {
        return new MachineNode(scene);
    }
    MachineNode(VrmlScene* scene):VrmlNodeChild(scene), m_index(machineNodes.size()) {
        
        std::cerr << "vrml Machine node created" << std::endl;
        machineNodes.push_back(this);
    }
    ~MachineNode()
    {
        machineNodes.erase(machineNodes.begin() + m_index);
    }
    // Define the fields of XCar nodes
    static VrmlNodeType* defineType(VrmlNodeType* t = 0)
    {
        static VrmlNodeType* st = 0;

        if (!t)
        {
            if (st)
                return st; // Only define the type once.
            t = st = new VrmlNodeType("CsvPointCloud", creator);
        }

        VrmlNodeChild::defineType(t); // Parent class


        t->addEventOut("x", VrmlField::SFVEC3F);
        t->addEventOut("y", VrmlField::SFVEC3F);
        t->addEventOut("z", VrmlField::SFVEC3F);


        return t;
    }
    virtual VrmlNodeType* nodeType() const { return defineType(); };
    VrmlNode* cloneMe() const
    {
        return new MachineNode(*this);
    }
    void move(VrmlSFVec3f &position)
    {
        auto t = System::the->time();
        constexpr float scale = 1;
        eventOut(t, "x", VrmlSFVec3f{ 0, 0, position.x() * scale});
        eventOut(t, "y", VrmlSFVec3f{ position.y() * scale, 0, 0});
        eventOut(t, "z", VrmlSFVec3f{ 0, position.z() * scale, 0 });
    }

private:
    size_t m_index = 0;
};

VrmlNode* creator(VrmlScene* scene)
{
    return new MachineNode(scene);
}

namespace fs = boost::filesystem;

// Constructor
CsvPointCloudPlugin::CsvPointCloudPlugin()
    : ui::Owner("CsvPointCloud", cover->ui)
    , m_CsvPointCloudMenu(new ui::Menu("CsvPointCloud", this))
    , m_dataScale(new ui::EditField(m_CsvPointCloudMenu, "Scale"))
    , m_colorMenu(new ui::Menu(m_CsvPointCloudMenu, "ColorMenu"))
    , m_coordTerms{ {new ui::EditField(m_CsvPointCloudMenu, "X"), new ui::EditField(m_CsvPointCloudMenu, "Y"), new ui::EditField(m_CsvPointCloudMenu, "Z")} }
    , m_machinePositionsTerms{{new ui::EditField(m_CsvPointCloudMenu, "Right"), new ui::EditField(m_CsvPointCloudMenu, "Forward"), new ui::EditField(m_CsvPointCloudMenu, "Up")}}
    , m_colorTerm(new ui::EditField(m_CsvPointCloudMenu, "Color"))
    , m_pointSizeSlider(new ui::Slider(m_CsvPointCloudMenu, "PointSize"))
    , m_numPointsSlider(new ui::Slider(m_CsvPointCloudMenu, "NumPoints"))
    , m_colorMapSelector(*m_CsvPointCloudMenu)
    , m_reloadBtn(new ui::Button(m_CsvPointCloudMenu, "Reload"))
    , m_timeScaleIndicator(new ui::EditField(m_CsvPointCloudMenu, "TimeScaleIndicator"))
    , m_pointReductionCriteria(new ui::EditField(m_CsvPointCloudMenu, "PointReductionCriteria"))
    , m_delimiter(new ui::EditField(m_CsvPointCloudMenu, "Delimiter"))
    , m_offset(new ui::EditField(m_CsvPointCloudMenu, "HeaderOffset"))
    , m_colorsGroup(new ui::Group(m_CsvPointCloudMenu, "Colors"))
    , m_colorBar(new opencover::ColorBar(m_colorMenu))
    , m_editFields{ m_dataScale,  m_coordTerms[0] ,  m_coordTerms[1],  m_coordTerms[2], m_machinePositionsTerms[0] ,  m_machinePositionsTerms[1],  m_machinePositionsTerms[2], m_colorTerm ,m_timeScaleIndicator ,m_delimiter, m_offset, m_pointReductionCriteria }
{
    
    m_dataScale->setValue("1");
    for (auto ef : m_editFields)
        ef->setShared(true);

    if(m_delimiter->value().empty())
        m_delimiter->setValue(";");

    m_pointSizeSlider->setBounds(0, 20);
    m_pointSizeSlider->setValue(4);
    m_pointSizeSlider->setCallback([this](ui::Slider::ValueType val, bool release)
                                  {
                                      if (m_pointCloud)
                                      {
                                          dynamic_cast<osg::Point *>(m_pointCloud->getStateSet()->getAttribute(osg::StateAttribute::Type::POINT))->setSize(val);
                                      }
                                  });
    m_pointSizeSlider->setShared(true);
    
    m_numPointsSlider->setShared(true);


    m_reloadBtn->setCallback([this](bool state)
                            {
            (void)state;
            if(m_currentGeode)
                                {
                                    auto parent = m_currentGeode->getParent(0);
                                    auto filename = m_currentGeode->getName();
                                    unloadFile(filename);
                                    load(filename.c_str(), parent, nullptr);
                                }
                            });
    m_reloadBtn->setShared(true);

    m_sliders = { new ui::Slider(m_CsvPointCloudMenu, "Xx"), new ui::Slider(m_CsvPointCloudMenu, "Yy"), new ui::Slider(m_CsvPointCloudMenu, "Zz") };
    for (auto slider : m_sliders)
    {
        slider->setBounds(-500, 500);
        slider->setCallback([this](double val, bool b) {
            osg::Vec3 v;
            for (size_t i = 0; i < 3; i++)
            {
                v[i] = m_sliders[i]->value();
            }
            osg::Matrix m;
            m.makeTranslate(v);
            //m(0, 3) = m_machinePositions[t].x();
            m_transform->setMatrix(m);
            });
    }
}

const CsvPointCloudPlugin *CsvPointCloudPlugin::instance() const
{
    return m_plugin;
}

bool CsvPointCloudPlugin::init()
{
    if (m_plugin)
        return false;
    m_plugin = this;
    m_pointSizeSlider->setValue(coCoviseConfig::getFloat("COVER.Plugin.PointCloud.PointSize", pointSize()));

    coVRFileManager::instance()->registerFileHandler(&handler);
    VrmlNamespace::addBuiltIn(MachineNode::defineType());
    return true;
}

CsvPointCloudPlugin::~CsvPointCloudPlugin()
{

    coVRFileManager::instance()->unregisterFileHandler(&handler);
}

int CsvPointCloudPlugin::load(const char *filename, osg::Group *loadParent, const char *covise_key)
{
    osg::MatrixTransform* t = new osg::MatrixTransform;
    osg::Group *g = new osg::Group;
    loadParent->addChild(t);
    t->addChild(g);
    if (filename)
    {
        g->setName(filename);
    }
    assert(m_plugin);
    m_plugin->m_transform = t;
    m_plugin->createGeodes(g, filename);
    return 1;
}

int CsvPointCloudPlugin::unload(const char *filename, const char *covise_key)
{
    return m_plugin->unloadFile(filename);
}

void setStateSet(osg::Geometry *geo, float pointSize)
{
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

bool CsvPointCloudPlugin::compileSymbol(DataTable &symbols, const std::string& symbol, Expression &expr)
{
    expr().register_symbol_table(symbols.symbols());
    if(!expr.parser.compile(symbol, expr()))
    {
        std::cerr << "failed to parse symbol " << symbol << std::endl;
        return false;
    }
    return true;
}

void CsvPointCloudPlugin::readSettings(const std::string& filename)
{
    auto settingsFileName = filename.substr(0, filename.find_last_of('.')) + ".txt";

    auto fn = coVRFileManager::instance()->findOrGetFile(settingsFileName);
    std::fstream f(fn);
    std::string line;


    while (std::getline(f, line))
    {
        std::string name = line.substr(0, line.find_first_of(" "));
        std::string value = line.substr(line.find_first_of('"') + 1, line.find_last_of('"') - line.find_first_of('"') - 1);
        auto setting = std::find_if(m_editFields.begin(), m_editFields.end(), [name](ui::EditField* ef) {return ef->name() == name; });
        if (setting != m_editFields.end())
            (*setting)->setValue(value);
    }
    
}

void CsvPointCloudPlugin::writeSettings(const std::string& filename)
{
    auto settingsFileName = filename.substr(0, filename.find_last_of('.')) + ".txt";

    auto fn = coVRFileManager::instance()->findOrGetFile(settingsFileName);
    std::ofstream f(fn);

    for (const auto ef : m_editFields)
    {
        f << ef->name() << " \"" << ef->value() << "\"\n";
    }
}

template<typename T>
T read(std::ifstream& f)
{
    T t;
    f.read((char*)&t, sizeof(T));
    return t;
}

template<typename T>
void write(std::ofstream& f, const T& t)
{
    f.write((const char*)&t, sizeof(T));
}

float parseScale(const std::string& scale)
{
    exprtk::symbol_table<float> symbol_table;
    typedef exprtk::expression<float>   expression_t;
    typedef exprtk::parser<float>       parser_t;

    std::string expression_string = "z := x - (3 * y)";


    expression_t expression;
    expression.register_symbol_table(symbol_table);

    parser_t parser;

    if (!parser.compile(scale, expression))
    {
        std::cerr << "parseScale failed" << std::endl;
        return 1;
    }
    return expression.value();
}

osg::Geometry *CsvPointCloudPlugin::createOsgPoints(DataTable &symbols, std::ofstream& f)
{
    // compile parser
    std::array<Expression, 4> stringExpressions;

    for (size_t i = 0; i < stringExpressions.size(); i++)
    {
        if(!compileSymbol(symbols, i < 3 ? m_coordTerms[i]->value() : m_colorTerm->value(), stringExpressions[i]))
            return nullptr;
    }

    auto colors = new Vec4Array();
    auto points = new Vec3Array();


    //calculate coords and color
    std::vector<float> scalarData(symbols.size());
    float minScalar = 0, maxScalar = 0;
    auto scale = parseScale(m_dataScale->value());
    for (size_t i = 0; i < symbols.size(); i++)
    {
        osg::Vec3 coords;
        for (size_t j = 0; j < 3; j++)
        {
            coords[j] = stringExpressions[j]().value() * scale;
        }
        points->push_back(coords);
        scalarData[i] = stringExpressions[3]().value();
        minScalar = std::min(minScalar, scalarData[i]);
        maxScalar = std::max(maxScalar, scalarData[i]);
        symbols.advance();
    }
    for (size_t i = 0; i < symbols.size(); i++)
        colors->push_back(m_colorMapSelector.getColor(scalarData[i], minScalar, maxScalar));
    
    //write cache file
    write(f, symbols.size());
    write(f, minScalar);
    write(f, maxScalar);
    f.write((const char*)&(*points)[0], symbols.size() * sizeof(osg::Vec3));
    f.write((const char*) &(*colors)[0], symbols.size() * sizeof(osg::Vec4));

    return createOsgPoints(points, colors, minScalar, maxScalar);
}

osg::Geometry* CsvPointCloudPlugin::createOsgPoints(Vec3Array* points, Vec4Array* colors, float minColor, float maxColor)
{
    //create geometry
    auto geo = new osg::Geometry();
    geo->setUseDisplayList(false);
    geo->setSupportsDisplayList(false);
    geo->setUseVertexBufferObjects(true);
    auto vertexBufferArray = geo->getOrCreateVertexBufferObject();
    
    osg::Vec3 bottomLeft, hpr, offset;
    if (coVRMSController::instance()->isMaster() && coVRConfig::instance()->numScreens() > 0) {
        auto hudScale = covise::coCoviseConfig::getFloat("COVER.Plugin.ColorBar.HudScale", 0.5); // half screen height
        const auto& s0 = coVRConfig::instance()->screens[0];
        hpr = s0.hpr;
        auto sz = osg::Vec3(s0.hsize, 0., s0.vsize);
        osg::Matrix mat;
        MAKE_EULER_MAT_VEC(mat, hpr);
        bottomLeft = s0.xyz - sz * mat * 0.5;
        auto minsize = std::min(s0.hsize, s0.vsize);
        bottomLeft += osg::Vec3(minsize, 0., minsize) * mat * 0.02;
        offset = osg::Vec3(s0.vsize / 2.5, 0, 0) * mat * hudScale;
    }
    m_colorBar->setName("Power");
    m_colorBar->show(true);
    m_colorBar->update(m_colorTerm->value(), minColor, maxColor, m_colorMapSelector.selectedMap().a.size(), m_colorMapSelector.selectedMap().r.data(), m_colorMapSelector.selectedMap().g.data(), m_colorMapSelector.selectedMap().b.data(), m_colorMapSelector.selectedMap().a.data());
    m_colorBar->setHudPosition(bottomLeft, hpr, offset[0] / 480);
    m_colorBar->show(true);


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
    geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, points->size()));
    geo->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, 0));

    setStateSet(geo, pointSize());

    return geo;
}



std::vector<VrmlSFVec3f> CsvPointCloudPlugin::readMachinePositions(DataTable& symbols) {
    
    symbols.reset();
    std::vector<VrmlSFVec3f> retval;
    // compile parser
    std::array<Expression, 3> stringExpressions;
    auto scale = parseScale(m_dataScale->value());

    for (size_t i = 0; i < stringExpressions.size(); i++)
    {
        if (!compileSymbol(symbols,m_machinePositionsTerms[i]->value(), stringExpressions[i]))
            return retval;
    }
    for (size_t i = 0; i < symbols.size(); i++)
    {
        retval.emplace_back( -1 * stringExpressions[0]().value() * scale, stringExpressions[1]().value() * scale, stringExpressions[2]().value() * scale);
        symbols.advance();
    }
    return retval;
}

std::vector<unsigned int> CsvPointCloudPlugin::readReducedPoints(DataTable& symbols) {

    symbols.reset();
    std::vector<unsigned int> retval;
    // compile parser
    Expression stringExpression;

    if (!compileSymbol(symbols, m_pointReductionCriteria->value(), stringExpression))
        return retval;

    for (size_t i = 0; i < symbols.size(); i++)
    {
        if (stringExpression().value())
            retval.push_back(i);
        symbols.advance();
    }
    return retval;
}

void CsvPointCloudPlugin::createGeodes(Group *parent, const std::string &filename)
{
    readSettings(filename);

    auto pointShader = opencover::coVRShaderList::instance()->get("Points");
    int offset = 0;
    try
    {
        offset = std::stoi(m_offset->value());
    }
    catch(const std::exception& e)
    {
        std::cerr << "header offset must be an integer" << std::endl;
        return;
    }
    size_t size = 0;
    if (auto cache = cacheFileUpToData(filename))
    {
        std::cerr << "using cache" << std::endl;
        size= read<size_t>(*cache);
        auto min = read<float>(*cache);
        auto max = read<float>(*cache);
        auto points = new Vec3Array(size);
        auto colors = new Vec4Array(size);
        cache->read((char*)&(*points)[0], size * sizeof(osg::Vec3));
        cache->read((char*)&(*colors)[0], size * sizeof(osg::Vec4));
        m_pointCloud = createOsgPoints(points, colors, min, max);
        m_machinePositions.resize(size);
        cache->read((char*)m_machinePositions.data(), size * sizeof(vrml::VrmlSFVec3f));
        auto numNonReducedPoints = read<size_t>(*cache);
        m_pointsToNotReduce.resize(numNonReducedPoints);
        cache->read((char*)m_pointsToNotReduce.data(), numNonReducedPoints * sizeof(unsigned int));

    }
    else {
        auto cacheFileName = filename.substr(0, filename.find_last_of('.')) + ".cache";

        std::ofstream f(cacheFileName, std::ios::binary);
        writeCacheFileHeader(f);
        DataTable dataTable(filename, m_timeScaleIndicator->value(), m_delimiter->value()[0], offset);
        size = dataTable.size();
        m_pointCloud = createOsgPoints(dataTable, f);
        m_machinePositions = readMachinePositions(dataTable);
        m_pointsToNotReduce = readReducedPoints(dataTable);
        f.write((const char*)m_machinePositions.data(), m_machinePositions.size() * sizeof(vrml::VrmlSFVec3f));
        write(f, m_pointsToNotReduce.size());
        f.write((const char*)m_pointsToNotReduce.data(), m_pointsToNotReduce.size() * sizeof(unsigned int));

    }
    m_numPointsSlider->setBounds(0, size);
    m_numPointsSlider->setValue(size);

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
    coVRAnimationManager::instance()->setNumTimesteps(size, this);
    coVRAnimationManager::instance()->setAnimationSkipMax(5000);
}

void CsvPointCloudPlugin::setTimestep(int t) 
{
    //show points until t
    if(m_pointCloud)
    {
        size_t start = std::max(ui::Slider::ValueType{ 0 }, t - m_numPointsSlider->value());
        size_t i = 0;
        for(; i < m_pointsToNotReduce.size(); i++)
        {
            if (m_pointsToNotReduce[i] >= start)
                break;

        }

        m_pointCloud->setPrimitiveSet(1, new osg::DrawElementsUInt(osg::PrimitiveSet::POINTS, i, m_pointsToNotReduce.data()));
        m_pointCloud->setPrimitiveSet(0, new osg::DrawArrays(osg::PrimitiveSet::POINTS, start, t + 1 - start));
    }
    //move machine axis
    if (m_machinePositions.size() > t)
    {
        for (auto machineNode : machineNodes)
        {
            machineNode->move(m_machinePositions[t]);
        }
        //move the workpiece with the machine table
        osg::Vec3 v{ 0, -1 * m_machinePositions[t].y(), 0 };
        osg::Matrix m;
        m.makeTranslate(v);
        if(m_transform)
            m_transform->setMatrix(m);
    }
}

float CsvPointCloudPlugin::pointSize() const
{
    return m_pointSizeSlider->value();
}

int CsvPointCloudPlugin::unloadFile(const std::string& filename)
{
    if(m_currentGeode && m_currentGeode->getNumParents() > 0)
    {
        writeSettings(filename);
        m_currentGeode->getParent(0)->removeChild(m_currentGeode);
        m_pointCloud = nullptr;
        m_currentGeode = nullptr;
        m_transform = nullptr;
        return 0;
    }
    return -1;
}

std::string readString(std::ifstream& f)
{
    std::string str;
    size_t size;
    f.read((char*)&size, sizeof(size));
    str.resize(size);
    f.read(&str[0], size);
    return str;
}

void writeString(std::ofstream& f, const std::string& s)
{
    size_t size = s.size();
    f.write((const char*)&size, sizeof(size));
    f.write(&s[0], size);
}

std::unique_ptr<std::ifstream> CsvPointCloudPlugin::cacheFileUpToData(const std::string& filename)
{
    auto settingsFileName = filename.substr(0, filename.find_last_of('.')) + ".txt";
    auto cacheFileName = filename.substr(0, filename.find_last_of('.')) + ".cache";
    if (fs::exists(cacheFileName) && fs::last_write_time(cacheFileName) > fs::last_write_time(settingsFileName) && fs::last_write_time(cacheFileName) > fs::last_write_time(filename))
    {
        std::unique_ptr<std::ifstream> f{ new std::ifstream{ cacheFileName, std::ios::binary } };
        auto date = readString(*f);
        auto time = readString(*f);
        if (date != __DATE__ || time != __TIME__)
            return nullptr;
        for (const auto editField : m_editFields)
        {
            auto s = readString(*f);
            if (s != editField->value())
                return nullptr;
        }
        return f;
    }
    return nullptr;
}

void CsvPointCloudPlugin::writeCacheFileHeader(std::ofstream& f)
{
    writeString(f, __DATE__);
    writeString(f, __TIME__);
    for (const auto editField : m_editFields)
    {
        writeString(f, editField->value());
    }
}
