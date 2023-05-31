/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************

#include "ColorMapShader.h"
#include "CsvPointCloud.h"
#include "RenderObject.h"
#include "SurfacePrimitiveSet.h"

#include <OpenVRUI/osg/mathUtils.h>
#include <config/CoviseConfig.h>
#include <util/string_util.h>

#include <cover/VRSceneGraph.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRConfig.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRShader.h>
#include <cover/coVRTui.h>
#include <osg/AlphaFunc>
#include <osg/Material>
#include <osg/Point>
#include <osg/PointSprite>
#include <osg/TemplatePrimitiveFunctor>
#include <osg/TemplatePrimitiveIndexFunctor>
#include <osg/TexEnv>
#include <osg/TexGen>
#include <osg/io_utils>

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/VrmlNamespace.h>

#include <boost/filesystem.hpp>

using namespace osg;
using namespace covise;
using namespace opencover;
using namespace vrml;




CsvPointCloudPlugin *CsvPointCloudPlugin::m_plugin = nullptr;

COVERPLUGIN(CsvPointCloudPlugin)

class MachineNode;
std::vector<MachineNode *> machineNodes;

static VrmlNode *creator(VrmlScene *scene);

class PLUGINEXPORT MachineNode : public vrml::VrmlNodeChild
{
public:
    static VrmlNode *creator(VrmlScene *scene)
    {
        return new MachineNode(scene);
    }
    MachineNode(VrmlScene *scene) : VrmlNodeChild(scene), m_index(machineNodes.size())
    {

        std::cerr << "vrml Machine node created" << std::endl;
        machineNodes.push_back(this);
    }
    ~MachineNode()
    {
        machineNodes.erase(machineNodes.begin() + m_index);
    }
    // Define the fields of XCar nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0)
    {
        static VrmlNodeType *st = 0;

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
    virtual VrmlNodeType *nodeType() const { return defineType(); };
    VrmlNode *cloneMe() const
    {
        return new MachineNode(*this);
    }
    void move(VrmlSFVec3f &position)
    {
        auto t = System::the->time();
        eventOut(t, "x", VrmlSFVec3f{-position.x(), 0, 0});
        eventOut(t, "y", VrmlSFVec3f{0, 0, -position.y()});
        eventOut(t, "z", VrmlSFVec3f{0, position.z(), 0});
    }

private:
    size_t m_index = 0;
};

VrmlNode *creator(VrmlScene *scene)
{
    return new MachineNode(scene);
}

namespace fs = boost::filesystem;

CsvRenderObject renderObject;

// Constructor
CsvPointCloudPlugin::CsvPointCloudPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , m_numThreads(std::thread::hardware_concurrency())
    , ui::Owner("CsvPointCloud10", cover->ui)
    , m_CsvPointCloudMenu(new ui::Menu("CsvPointCloud11", this))
    , m_dataSelector(new ui::SelectionList(m_CsvPointCloudMenu, "ScalarData"))
    , m_pointSizeSlider(new ui::Slider(m_CsvPointCloudMenu, "PointSize"))
    , m_numPointsSlider(new ui::Slider(m_CsvPointCloudMenu, "NumPoints"))
    , m_moveMachineBtn(new ui::Button(m_CsvPointCloudMenu, "MoveMachine"))
    , m_showSurfaceBtn(new ui::Button(m_CsvPointCloudMenu, "ShowSurface"))
    , m_advancedBtn(new ui::Button(m_CsvPointCloudMenu, "Advanced"))
    , m_dataScale(new ui::EditField(m_CsvPointCloudMenu, "Scale"))
    , m_coordTerms{{new ui::EditField(m_CsvPointCloudMenu, "X"), new ui::EditField(m_CsvPointCloudMenu, "Y"), new ui::EditField(m_CsvPointCloudMenu, "Z")}}
    , m_machinePositionsTerms{{new ui::EditField(m_CsvPointCloudMenu, "Right"), new ui::EditField(m_CsvPointCloudMenu, "Forward"), new ui::EditField(m_CsvPointCloudMenu, "Up")}}
    , m_colorTerm(new ui::EditField(m_CsvPointCloudMenu, "Color"))
    , m_colorMapSelector(*m_CsvPointCloudMenu)
    , m_timeScaleIndicator(new ui::EditField(m_CsvPointCloudMenu, "TimeScaleIndicator"))
    , m_pointReductionCriteria(new ui::EditField(m_CsvPointCloudMenu, "PointReductionCriterium"))
    , m_numPontesPerCycle(new ui::EditField(m_CsvPointCloudMenu, "NumPointsPerCycle"))
    , m_delimiter(new ui::EditField(m_CsvPointCloudMenu, "Delimiter"))
    , m_offset(new ui::EditField(m_CsvPointCloudMenu, "HeaderOffset"))
    , m_editFields{m_dataScale, m_coordTerms[0], m_coordTerms[1], m_coordTerms[2], m_machinePositionsTerms[0], m_machinePositionsTerms[1], m_machinePositionsTerms[2], m_colorTerm, m_timeScaleIndicator, m_delimiter, m_offset, m_pointReductionCriteria, m_numPontesPerCycle}
    , m_applyBtn(new ui::Button(m_CsvPointCloudMenu, "Apply"))
    , m_colorInteractor(new CsvInteractor())
{
    m_dataSelector->setShared(true);
    m_showSurfaceBtn->setState(false);
    m_showSurfaceBtn->setShared(true);
    m_showSurfaceBtn->setCallback([this](bool state)
                                  { setTimestep(m_lastTimestep); });
    m_moveMachineBtn->setState(true);
    m_moveMachineBtn->setShared(true);
    m_colorInteractor->incRefCount();
    coVRAnimationManager::instance()->setAnimationSkipMax(5000);
    m_dataScale->setValue("1");
    for (auto ef : m_editFields)
        ef->setShared(true);

    if (m_delimiter->value().empty())
        m_delimiter->setValue(";");

    m_pointSizeSlider->setBounds(0, 20);
    m_pointSizeSlider->setValue(4);
    m_pointSizeSlider->setCallback([this](ui::Slider::ValueType val, bool release)
                                   {
                                      if (m_points)
                                          static_cast<Point *>(m_points->getStateSet()->getAttribute(StateAttribute::Type::POINT))->setSize(val);
                                    });
    m_pointSizeSlider->setShared(true);

    m_numPointsSlider->setShared(true);
    m_numPointsSlider->setBounds(0, 1);
    m_numPointsSlider->setValue(1);

    m_applyBtn->setCallback([this](bool state)
                            {
                                (void)state;
                                if(m_currentGeode)
                                {
                                    auto parent = m_currentGeode->getParent(0);
                                    auto filename = m_currentGeode->getName();
                                    unloadFile(filename);
                                    load(filename.c_str(), parent, nullptr);
                                } });
    m_applyBtn->setShared(true);

    m_colorTerm->setCallback([this](const std::string &text)
                             {
        auto term = updateDataSelector(text);
        loadData(term);
 });

    m_colorMapSelector.setCallback([this](const ColorMap &map)
                                   {
            m_colorInteractor->setColorMap(map);
            updateColorMap();
        });
    m_colorInteractor->setColorMap(m_colorMapSelector.selectedMap());

    if(!cover->visMenu)
    {
        cover->visMenu = new ui::Menu("Oct", this);
    }
    cover->addPlugin("ColorBars");
    m_advancedBtn->setCallback([this](bool b)
        {
            for (auto ef : m_editFields)
                ef->setVisible(b);
            m_applyBtn->setVisible(b);
        });
    for (auto ef : m_editFields)
        ef->setVisible(false);
    m_applyBtn->setVisible(false);
    m_advancedBtn->setState(false);
}

std::string CsvPointCloudPlugin::updateDataSelector(const std::string& term)
{
    auto scalars = split(term, ';');

    if (scalars.size() % 2 != 0)
    {
        std::cerr << "color term must be a ';' separeted list of Names and math terms" << std::endl;
        return "";
    }
    std::vector<std::string> names;
    std::vector<std::string> terms;
    for (size_t i = 0; i < scalars.size();)
    {
        names.push_back(scalars[i++]);
        terms.push_back(scalars[i++]);
    }
    m_dataSelector->setList(names);
    m_dataSelector->setCallback([this, terms](int index) {
        loadData(terms[index]);
        });
    size_t index = m_dataSelector->selectedIndex();
    if (index >= names.size())
    {
        index = 0;
        m_dataSelector->select(0);
    }
    return terms[index];
}

void CsvPointCloudPlugin::loadData(const std::string& term)
{
    if (m_dataTable)
    {
        m_colorInteractor->setName(term);
        auto colors = getScalarData(*m_dataTable, term);

        if (m_points)
            m_points->setVertexAttribArray(DataAttrib, colors.data);
        if (m_surface)
            m_surface->setVertexAttribArray(DataAttrib, colors.data);

        updateColorMap();
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

    m_handler[0] = FileHandler{nullptr, CsvPointCloudPlugin::load, CsvPointCloudPlugin::unload, "csv"};
    m_handler[1] = FileHandler{ nullptr, CsvPointCloudPlugin::load, CsvPointCloudPlugin::unload, "oct"};

    coVRFileManager::instance()->registerFileHandler(&m_handler[0]);
    coVRFileManager::instance()->registerFileHandler(&m_handler[1]);
    VrmlNamespace::addBuiltIn(MachineNode::defineType());
    return true;
}

CsvPointCloudPlugin::~CsvPointCloudPlugin()
{

    coVRFileManager::instance()->unregisterFileHandler(&m_handler[0]);
    coVRFileManager::instance()->unregisterFileHandler(&m_handler[1]);
    opencover::coVRPluginList::instance()->removeObject(renderObject.getName(), false);
    for (auto ef : m_editFields)
        ef->setVisible(true);
    m_applyBtn->setVisible(true);

}

int CsvPointCloudPlugin::load(const char *filename, Group *loadParent, const char *covise_key)
{
    MatrixTransform *t = new MatrixTransform;
    Group *g = new Group;
    g->setNodeMask(g->getNodeMask() & ~Isect::Intersection & ~Isect::Pick);
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

void setDefaultMaterial(StateSet *geoState)
{

    geoState->setNestRenderBins(false);

    ref_ptr<Material> globalmtl = new Material;
    globalmtl->setColorMode(Material::AMBIENT_AND_DIFFUSE);
    globalmtl->setAmbient(Material::FRONT_AND_BACK, Vec4(0.2f, 0.2f, 0.2f, 1.0));
    globalmtl->setDiffuse(Material::FRONT_AND_BACK, Vec4(0.8f, 0.8f, 0.8f, 1.0));
    globalmtl->setSpecular(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
    globalmtl->setEmission(Material::FRONT_AND_BACK, Vec4(0.0f, 0.0f, 0.0f, 1.0));
    globalmtl->setShininess(Material::FRONT_AND_BACK, 5.0f);

    geoState->setAttributeAndModes(globalmtl.get(), StateAttribute::ON);

}

void applyPointState(StateSet *stateset, int pointSize)
{

    AlphaFunc *alphaFunc = new AlphaFunc(AlphaFunc::GREATER, 0.5);
    stateset->setAttributeAndModes(alphaFunc, StateAttribute::ON);

    Point *pointstate = new Point();
    pointstate->setSize(pointSize);
    stateset->setAttributeAndModes(pointstate, StateAttribute::ON);

    PointSprite *sprite = new PointSprite();
    stateset->setTextureAttributeAndModes(0, sprite, StateAttribute::ON);
    stateset->setMode(GL_POINT_SMOOTH, osg::StateAttribute::ON);
    const char *mapName = opencover::coVRFileManager::instance()->getName("share/covise/icons/particle.png");
    if (mapName != NULL)
    {
        Image *image = osgDB::readImageFile(mapName);
        Texture2D *tex = new Texture2D(image);

        tex->setTextureSize(image->s(), image->t());
        tex->setInternalFormat(GL_RGBA);
        tex->setFilter(Texture2D::MIN_FILTER, Texture2D::LINEAR);
        tex->setFilter(Texture2D::MAG_FILTER, Texture2D::LINEAR);
        stateset->setTextureAttributeAndModes(0, tex, StateAttribute::ON);
        TexEnv *texEnv = new TexEnv;
        texEnv->setMode(TexEnv::MODULATE);
        stateset->setTextureAttributeAndModes(0, texEnv, StateAttribute::ON);

        ref_ptr<TexGen> texGen = new TexGen();
        stateset->setTextureAttributeAndModes(0, texGen.get(), StateAttribute::OFF);
        std::cerr << "read image file " << mapName << std::endl;
    }
}

bool CsvPointCloudPlugin::compileSymbol(DataTable &symbols, const std::string &symbol, Expression &expr)
{
    expr().register_symbol_table(symbols.symbols());
    if (!expr.parser.compile(symbol, expr()))
    {
        std::cerr << "failed to parse symbol " << symbol << std::endl;
        return false;
    }
    return true;
}

void CsvPointCloudPlugin::readSettings(const std::string &filename)
{
    auto fn = filename.substr(0, filename.find_last_of('.')) + ".txt";
    m_readSettingsTime = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::fstream f(fn);
    if (!f.is_open())
    {
        std::cerr << "csvPointCloud: could not read settings file " << fn << std::endl;
        return;
    }
    std::string line;

    while (std::getline(f, line))
    {
        std::string name = line.substr(0, line.find_first_of(" "));
        std::string value = line.substr(line.find_first_of('"') + 1, line.find_last_of('"') - line.find_first_of('"') - 1);
        auto setting = std::find_if(m_editFields.begin(), m_editFields.end(), [name](ui::EditField *ef)
                                    { return ef->name() == name; });
        if (setting != m_editFields.end())
            (*setting)->setValue(value);
        else if (name == m_pointSizeSlider->name())
            m_pointSizeSlider->setValue(std::stof(value));
        else if (name == m_numPointsSlider->name())
        {
            m_numPointsSlider->setBounds(0, std::stoi(value));
            m_numPointsSlider->setValue(std::stoi(value));
        }
        else if (name == "AnimationSpeed")
        {
            coVRAnimationManager::instance()->setAnimationSpeed(std::stof(value));
            m_animSpeedSet = true;
        }
        else if (name == "AnimationSkip")
        {
            coVRAnimationManager::instance()->setAnimationSkip(std::stoi(value));
            m_animSkipSet = true;
        }
    }
}

void CsvPointCloudPlugin::writeSettings(const std::string &filename)
{

    auto fn = filename.substr(0, filename.find_last_of('.')) + ".txt";
    // Settings file has changed since last read, don't override
    if (fs::last_write_time(fn) > m_readSettingsTime)
        return;

    std::ofstream f(fn);

    for (const auto ef : m_editFields)
    {
        f << ef->name() << " \"" << ef->value() << "\"\n";
    }
    f << m_pointSizeSlider->name() << " \"" << m_pointSizeSlider->value() << "\"\n";
    f << m_numPointsSlider->name() << " \"" << m_numPointsSlider->value() << "\"\n";
    if (m_animSpeedSet)
        f << "AnimationSpeed"
          << " \"" << coVRAnimationManager::instance()->getAnimationSpeed() << "\"\n";
    if (m_animSkipSet)
        f << "AnimationSkip"
          << " \"" << coVRAnimationManager::instance()->getAnimationSkip() << "\"\n";
}

float parseScale(const std::string &scale)
{
    exprtk::symbol_table<float> symbol_table;
    typedef exprtk::expression<float> expression_t;
    typedef exprtk::parser<float> parser_t;

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

void CsvPointCloudPlugin::updateColorMap()
{
    auto cm = m_colorInteractor->getColorMap();
    if (m_points)
        applyPointShader(m_currentGeode, m_points, cm, m_minColor, m_maxColor);
    if (m_surface)
        applySurfaceShader(m_currentGeode, m_surface, cm, m_minColor, m_maxColor);

    opencover::coVRPluginList::instance()->removeObject("CsvPointCloud4", false);
    opencover::coVRPluginList::instance()->newInteractor(&renderObject, m_colorInteractor);
}

CsvPointCloudPlugin::ScalarData CsvPointCloudPlugin::getScalarData(DataTable &symbols, const std::string& term)
{
    if (term.empty())
        return ScalarData{};
    renderObject.setObjName(term);
    size_t numColorsPerThread = symbols.size() / m_numThreads;
    std::vector<std::future<ScalarData>> futures;
    for (size_t i = 0; i < m_numThreads; i++)
    {
        auto begin = i * numColorsPerThread;
        auto end = i == m_numThreads - 1 ? symbols.size() : (i + 1) * numColorsPerThread;
        futures.push_back(std::async(std::launch::async, [this, symbols, &term,  begin, end]()
                                     {
                                        ScalarData data;
                                        data.data = new FloatArray();
                                        data.data->setBinding(Array::BIND_PER_VERTEX);
                                        std::array<float, 3> currentMachineSpeed;
                                        resetMachineSpeed(currentMachineSpeed);
                                        auto symbolsFragment = symbols;
                                        addMachineSpeedSymbols(symbolsFragment, currentMachineSpeed);
                                        Expression colorExporession, reductionCriterium;

                                        if (!compileSymbol(symbolsFragment, m_pointReductionCriteria->value(), reductionCriterium))
                                            return data;
                                        if (!compileSymbol(symbolsFragment, term, colorExporession))
                                            return data;
                                        for (size_t i = begin; i < end; i++)
                                        {
                                            auto scalar = colorExporession().value();
                                            data.data->push_back(scalar);
                                            data.min = std::min(data.min, scalar);
                                            data.max = std::max(data.max, scalar);
                                            advanceMachineSpeed(currentMachineSpeed, i);
                                            symbolsFragment.setCurrentValues(i);
                                        } 
                                        return data; 
                                    }));
    }
    ScalarData data;
    data.data = new FloatArray();
    data.data->setBinding(Array::BIND_PER_VERTEX);
    for (size_t i = 0; i < m_numThreads; i++)
    {
        auto dataFragment = futures[i].get();
        data.data->insert(data.data->end(), dataFragment.data->begin(), dataFragment.data->end());
        i == 0 ? data.min = dataFragment.min : data.min = std::min(data.min, dataFragment.min);
        i == 0 ? data.max = dataFragment.max : data.max = std::max(data.max, dataFragment.max);
    }

    m_colorInteractor->setMinMax(data.min, data.max);
    m_minColor = data.min;
    m_maxColor = data.max;

    return data;
}

ref_ptr<Vec3Array> CsvPointCloudPlugin::getCoords(DataTable &symbols)
{
    auto scale = parseScale(m_dataScale->value());
    size_t numColorsPerThread = symbols.size() / m_numThreads;
    std::vector<std::future<std::pair<ref_ptr<Vec3Array>, std::vector<size_t>>>> futures;
    for (size_t i = 0; i < m_numThreads; i++)
    {
        auto begin = i * numColorsPerThread;
        auto end = i == m_numThreads - 1 ? symbols.size() : (i + 1) * numColorsPerThread;
        futures.push_back(std::async(std::launch::async, [this, symbols, scale, begin, end]()
                                     {
            auto pair = std::make_pair<ref_ptr<Vec3Array>, std::vector<size_t>>(new Vec3Array, std::vector<size_t>{});
            auto &reducedIndices = pair.second;
            auto &coords = pair.first;
            coords->setBinding(Array::BIND_PER_VERTEX);
            std::array<float, 3> currentMachineSpeed;
            resetMachineSpeed(currentMachineSpeed);
            auto symbolsFragment = symbols;
            addMachineSpeedSymbols(symbolsFragment, currentMachineSpeed);
            Expression colorExporession, reductionCriterium;

            if (!compileSymbol(symbolsFragment, m_pointReductionCriteria->value(), reductionCriterium))
                return pair;
            std::array<Expression, 3> coordExpressions;

            for (size_t i = 0; i < coordExpressions.size(); i++)
            {
                if (!compileSymbol(symbolsFragment, m_coordTerms[i]->value(), coordExpressions[i]))
                    return pair;
            }
            for (size_t i = begin; i < end; i++)
            {
                Vec3 coord;
                for (size_t j = 0; j < 3; j++)
                    coord[j] = coordExpressions[j]().value() * scale;
                
                if (reductionCriterium().value())
                {
                    reducedIndices.push_back(i);
                } 
                coords->push_back(coord);
                advanceMachineSpeed(currentMachineSpeed, i);
                symbolsFragment.setCurrentValues(i);
            }
            return pair; }));
    }
    ref_ptr<Vec3Array> coords = new Vec3Array();
    coords->setBinding(Array::BIND_PER_VERTEX);
    m_reducedIndices.clear();
    for (size_t i = 0; i < m_numThreads; i++)
    {
        auto coordsFragment = futures[i].get();
        coords->insert(coords->end(), coordsFragment.first->begin(), coordsFragment.first->end());
        m_reducedIndices.insert(m_reducedIndices.end(), coordsFragment.second.begin(), coordsFragment.second.end());
    }
    return coords;
}

constexpr unsigned int allPointsPrimitiveIndex = 0;
constexpr unsigned int reducedPointsPrimitiveIndex = 1;

Vec3 getNormal(const Vec3Array& vertices, size_t vertexIndex, size_t numPointsPerCycle)
{
    std::array<Vec3, 4> neigbors = {vertexIndex >= 1 ? vertices[vertexIndex - 1] : vertices[vertexIndex],
                                    vertexIndex  + 1 < vertices.size() ? vertices[vertexIndex + 1] : vertices[vertexIndex],
                                    vertexIndex + numPointsPerCycle < vertices.size() ? vertices[vertexIndex + numPointsPerCycle] : vertices[vertexIndex],
                                    vertexIndex >= numPointsPerCycle ? vertices[vertexIndex - numPointsPerCycle] : vertices[vertexIndex]};
    Vec3 normal;

    for (size_t i = 0; i < neigbors.size(); i++)
    {
        auto last = i == 0 ? 3 : i - 1;
        auto x = vertices[vertexIndex] - neigbors[i] ^ vertices[vertexIndex] - neigbors[last];
        x.normalize();
        normal += x;
    }

    return normal;
}

ref_ptr<Vec3Array> calculateNormals(ref_ptr<Vec3Array> &vertices, size_t numPointsPerCycle)
{
    ref_ptr<Vec3Array> normals = new Vec3Array;
    
    for (size_t i = 0; i < vertices->size() - numPointsPerCycle - 1; i++)
        normals->push_back(getNormal(*vertices, i, numPointsPerCycle));
    return normals;
}

void CsvPointCloudPlugin::createGeometries(DataTable &symbols)
{
    try
    {
        m_numPointsPerCycle = std::stoi(m_numPontesPerCycle->value());
    }
    catch(const std::exception& ){}
    
    auto coords = getCoords(symbols);
    auto term = updateDataSelector(m_colorTerm->value());
    auto colors = getScalarData(symbols, term);

    if (!colors.data || !coords)
        return;

    auto normals = calculateNormals(coords, m_numPointsPerCycle);
    ref_ptr<StateSet> stateSet = VRSceneGraph::instance()->loadDefaultGeostate();
    ref_ptr<StateSet> stateSet2 = VRSceneGraph::instance()->loadDefaultGeostate();

    

    applyPointState(stateSet, pointSize());
    m_points = createOsgGeometry(coords, colors.data, normals, stateSet);

    m_surface = createOsgGeometry(coords, colors.data, normals, stateSet2);

    m_points->insertPrimitiveSet(allPointsPrimitiveIndex, new DrawArrays(PrimitiveSet::POINTS, 0, coords->size()));
    ref_ptr<SurfacePrimitiveSet> primitives = new SurfacePrimitiveSet(osg::PrimitiveSet::POINTS);
    primitives->insert(primitives->begin(), m_reducedIndices.begin(), m_reducedIndices.end());
    m_points->insertPrimitiveSet(reducedPointsPrimitiveIndex, primitives);

    primitives = new SurfacePrimitiveSet(osg::PrimitiveSet::QUADS);
    
    for (size_t i = 0; i < coords->size() - m_numPointsPerCycle - 1; i++)
    {
        primitives->push_back(i);
        primitives->push_back(i + m_numPointsPerCycle);
        primitives->push_back(i + m_numPointsPerCycle + 1);
        primitives->push_back(i + 1);
    }
    primitives->setRange(0, primitives->getNumPrimitives());
    m_surface->addPrimitiveSet(primitives);
}

ref_ptr<Geometry> CsvPointCloudPlugin::createOsgGeometry(ref_ptr<Vec3Array> &vertices, ref_ptr<FloatArray> &colors, ref_ptr<Vec3Array>& normals, ref_ptr<StateSet> &state)
{
   
    osg::ref_ptr<osg::Geometry> geo = new Geometry();

    geo->setVertexArray(vertices);
    geo->setVertexAttribArray(DataAttrib, colors);
    geo->setNormalArray(normals.get(), Array::BIND_PER_VERTEX);
    geo->setUseDisplayList(false);
    geo->setSupportsDisplayList(false);
    geo->setUseVertexBufferObjects(true);
    geo->setStateSet(state);
    return geo;
}

std::vector<VrmlSFVec3f> CsvPointCloudPlugin::readMachinePositions(DataTable &symbols)
{
    size_t numColorsPerThread = symbols.size() / m_numThreads;
    std::vector<std::future<bool>> futures;
    auto scale = parseScale(m_dataScale->value());
    std::vector<VrmlSFVec3f> retval(symbols.size());
    for (size_t i = 0; i < m_numThreads; i++)
    {
        auto begin = i * numColorsPerThread;
        auto end = i == m_numThreads - 1 ? symbols.size() : (i + 1) * numColorsPerThread;
        futures.push_back(std::async(std::launch::async, [this, &retval, symbols, scale, begin, end]()
                                     {
            std::array<Expression, 3> stringExpressions;
                                        auto symbolsFragment = symbols;

            for (size_t i = 0; i < stringExpressions.size(); i++)
            {
                if (!compileSymbol(symbolsFragment, m_machinePositionsTerms[i]->value(), stringExpressions[i]))
                    return false;
            }
            for (size_t i = begin; i < end; i++)
            {
                retval[i] = VrmlSFVec3f(stringExpressions[0]().value() * scale, stringExpressions[1]().value() * scale, stringExpressions[2]().value() * scale);
                symbolsFragment.setCurrentValues(i);
            }
            return true;
        }));
    }
    for (const auto& f : futures)
        f.wait();

    return retval;
}

void CsvPointCloudPlugin::createGeodes(Group *parent, const std::string &filename)
{
    readSettings(filename);

    int offset = 0;
    try
    {
        offset = std::stoi(m_offset->value());
    }
    catch (const std::exception &)
    {
        std::cerr << "header offset must be an integer" << std::endl;
        return;
    }
    size_t size = 0;

    if (!m_dataTable)
    {
        auto binaryFile = filename.substr(0, filename.find_last_of('.')) + ".oct";
        if (filename == binaryFile)
            m_dataTable.reset(new DataTable(binaryFile));
        else
        {
            m_dataTable.reset(new DataTable(filename, m_timeScaleIndicator->value(), m_delimiter->value()[0], offset));
            m_dataTable->writeToFile(binaryFile);
        }
        addMachineSpeedSymbols(*m_dataTable, m_currentMachineSpeeds);
    }
    if (m_dataTable->size() == 0)
        return;
    m_machinePositions = readMachinePositions(*m_dataTable);
    size = m_dataTable->size();
    createGeometries(*m_dataTable);



    m_numPointsSlider->setBounds(0, size);
    if (m_numPointsSlider->value() == 1)
        m_numPointsSlider->setValue(size);
    m_numPointsSlider->setIntegral(true);

    m_currentGeode = new Geode();
    m_currentGeode->setName(filename);
    parent->addChild(m_currentGeode);

    if (m_points)
        m_currentGeode->addDrawable(m_points.get());
    if (m_surface)
        m_currentGeode->addDrawable(m_surface.get());
    updateColorMap();

    coVRAnimationManager::instance()->setNumTimesteps(size, this);
}

void CsvPointCloudPlugin::addMachineSpeedSymbols(DataTable &symbols, std::array<float, 3> &currentMachineSpeed)
{
    for (size_t i = 0; i < currentMachineSpeed.size(); i++)
        symbols.symbols().add_variable(m_machineSpeedNames[i], currentMachineSpeed[i]);
}

void CsvPointCloudPlugin::setTimestep(int t)
{
    
    if (m_lastTimestep > t)
        m_reducedPointsBetween = 0;

    if (m_lastNumFullDrawnPoints != (size_t)m_numPointsSlider->value())
        m_reducedPointsBetween = 0;


    size_t start = std::max(ui::Slider::ValueType{0}, t - m_numPointsSlider->value());
    for (;; ++m_reducedPointsBetween)
    {
        if (m_reducedPointsBetween >= m_reducedIndices.size() || m_reducedIndices[m_reducedPointsBetween] > start)
            break;
    }

    size_t count = std::min(t, (int)m_numPointsSlider->value());
    if (m_points)
    {
        static_cast<DrawArrays *>(m_points->getPrimitiveSet(allPointsPrimitiveIndex))->setFirst(start);
        static_cast<DrawArrays *>(m_points->getPrimitiveSet(allPointsPrimitiveIndex))->setCount(count);

        auto reducedPontsPrimitives = static_cast<SurfacePrimitiveSet*>(m_points->getPrimitiveSet(reducedPointsPrimitiveIndex));
        reducedPontsPrimitives->setRange(0, m_reducedPointsBetween);
    }
    if(m_surface)
    {
        auto surfacePrimitives = static_cast<SurfacePrimitiveSet*>(m_surface->getPrimitiveSet(0));
        if(m_showSurfaceBtn->state() && start + m_numPointsSlider->value() < surfacePrimitives->getNumPrimitives())
            surfacePrimitives->setRange(start, count);
        else
            surfacePrimitives->setRange(0, 0);
    }

    // move machine axis
    if (m_moveMachineBtn->state() && m_machinePositions.size() > t)
    {
        for (auto machineNode : machineNodes)
        {
            machineNode->move(m_machinePositions[t]);
        }
        // move the workpiece with the machine table
        Vec3 v{-m_machinePositions[t].x(), 0, 0};
        Matrix m;
        m.makeTranslate(v);
        if (m_transform)
            m_transform->setMatrix(m);
    }
    m_lastTimestep = t;
    m_lastNumFullDrawnPoints = m_numPointsSlider->value();
}

float CsvPointCloudPlugin::pointSize() const
{
    return m_pointSizeSlider->value();
}

int CsvPointCloudPlugin::unloadFile(const std::string &filename)
{
    if (m_currentGeode && m_currentGeode->getNumParents() > 0)
    {
        writeSettings(filename);
        m_currentGeode->getParent(0)->removeChild(m_currentGeode);
        m_points = nullptr;
        m_currentGeode = nullptr;
        m_transform = nullptr;

        return 0;
    }
    return -1;
}

void CsvPointCloudPlugin::resetMachineSpeed(std::array<float, 3> &machineSpeed)
{
    std::fill(machineSpeed.begin(), machineSpeed.end(), 0);
}

void CsvPointCloudPlugin::advanceMachineSpeed(std::array<float, 3> &machineSpeed, size_t i)
{
    if (i > 0)
    {
        auto speed = m_machinePositions[i];
        speed.subtract(&m_machinePositions[i - 1]);
        machineSpeed[0] = speed.x();
        machineSpeed[1] = speed.y();
        machineSpeed[2] = speed.z();
    }
}

bool CsvPointCloudPlugin::update()
{
    float min, max;
    int steps;
    m_colorInteractor->getFloatScalarParam("min", min);
    m_colorInteractor->getFloatScalarParam("max", max);
    m_colorInteractor->getIntScalarParam("steps", steps);
    if (min != m_minColor || max != m_maxColor || steps != m_numColorSteps)
    {
        m_minColor = min;
        m_maxColor = max;
        m_numColorSteps = steps;
        updateColorMap();
        return true;
    }
    return false;
}
