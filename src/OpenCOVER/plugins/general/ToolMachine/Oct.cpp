#include "Oct.h"

#include <cover/VRSceneGraph.h>
#include <PluginUtil/coShaderUtil.h>
#include <OpcUaClient/opcuaClient.h>
#include <osg/Point>
using namespace opencover;

constexpr unsigned int allPointsPrimitiveIndex = 0;
constexpr unsigned int reducedPointsPrimitiveIndex = 1;

Oct::Oct(opencover::ui::Group *group, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode)
: Tool(group, toolHeadNode, tableNode)
, m_pointSizeSlider(new ui::Slider(group, "pointSize"))
, m_showSurfaceBtn(new ui::Button(group, "showSurface"))

{
    m_pointSizeSlider->setBounds(0, 20);
    m_pointSizeSlider->setValue(4);
    m_pointSizeSlider->setCallback([this](ui::Slider::ValueType v, bool r)
    {
        for(auto &section : m_sections)
            static_cast<osg::Point *>(section.points->getStateSet()->getAttribute(osg::StateAttribute::Type::POINT))->setSize(m_pointSizeSlider->value());
    });
    m_attributeName->setCallback([this](int i){
        m_opcuaAttribId = m_client->observeNode(m_attributeName->selectedItem());
    });
    m_numSectionsSlider->setCallback([this](ui::Slider::ValueType val, bool b){
        Section::Visibility vis = Section::Visible;
        if(!m_showSurfaceBtn->state())
            vis = Section::PointsOnly;
        if(val < 0)
        {
            for(auto &section : m_sections)
                section.show(vis);
            return;
        }

        for (size_t i = 0; i < m_sections.size() - val; i++)
        {
            m_sections[i].show(Section::Invisible);
        }

        for (size_t i = m_sections.size() - val; i < m_sections.size(); i++)
        {
            m_sections[i].show(vis);
        }
        
    });
    m_showSurfaceBtn->setState(false);
    m_showSurfaceBtn->setCallback([this](bool state){
        for(auto &section : m_sections)
        {
            if(section.status == Section::PointsOnly && state)
                section.show(Section::Visible);
            if(section.status == Section::Visible && !state)
                section.show(Section::PointsOnly);
        }

    });
}

void Oct::setOffset(const std::string &name)
{
    m_offsetName = name;
    m_opcuaOffsetId = m_client->observeNode(name);
}

void Oct::setScale(float scale)
{
    m_opcUaToVrmlScale = scale;
}

constexpr size_t maxSectionSize = 5000;
constexpr size_t numSections = 2000;


void Oct::clear()
{
    m_sections.clear();
}

void Oct::applyShader(const covise::ColorMap& map, float min, float max)
{
    for(auto &section : m_sections)
        opencover::applyShader(section.points, map, min, max, "OctPoints");
}

// constexpr auto phaseOffset = 4.01426;
const auto phaseOffset = osg::PI;


void Oct::addPoints(const std::string &valueName, const osg::Vec3 &toolHeadPos, const osg::Vec3 &up, float r)
{
    std::vector<UA_Float> offset, data;
    if(!m_client->isConnected())
        return;
    auto array = m_client->getArray<UA_Float>(m_offsetName);
    offset = std::move(array.data);
    auto array2 = m_client->getArray<UA_Float>(m_attributeName->selectedItem());
    data = std::move(array2.data);
    if(data.empty())
        data.push_back(0);
        // std::fill(data.begin(), data.end(), 0);
    float increment = -2 *osg::PI /offset.size();
    if(m_sections.empty())
        addSection();
    auto *section = &m_sections.back();

    correctLastUpdate(toolHeadPos);

    m_lastUpdate = Update{offset.size(), toolHeadPos};
    
    // std::cerr << std::endl;
    for (size_t i = 0; i < offset.size(); i++)
    {
        float theta = increment * i;
        osg::Vec3 v{
            toolHeadPos.x() + r * (float)cos(theta + phaseOffset),
            toolHeadPos.y() - offset[i] * m_opcUaToVrmlScale,
            toolHeadPos.z() + r * (float)sin(theta + phaseOffset) };
        // std::cerr << "vert[" << v.x() << ", " << v.y() << ", " << v.z() << "]" << std::endl;
        // if(!section->append(v, data.size() == offset.size()? data[i] : data[0]))
        if(!section->append(v, i))
        {
            section->createSurface();
            section = &addSection();
            section->append(v, offset[i]);
        }
    }
    // std::cerr << std::endl;
}

void Oct::correctLastUpdate(const osg::Vec3 &toolHeadPos)
{
    if(m_sections.size() == 1 && m_sections[0].vertices->empty())
        return;
    auto *section = &m_sections.back();
    auto begin = section->vertices->size() - m_lastUpdate.numValues;
    if(begin < 0)
    {
        section = &m_sections[m_sections.size() - 2];
        begin += section->vertices->size();
        section->changed = true;
    }
    auto end = std::min(begin + m_lastUpdate.numValues, maxSectionSize);
    float angleIncrement = 2 *osg::PI / m_lastUpdate.numValues;
    auto posIncrement = (toolHeadPos - m_lastUpdate.pos) / m_lastUpdate.numValues;
    for (size_t i = 0; i < m_lastUpdate.numValues; i++)
    {
        (*section->vertices)[begin] += posIncrement * i;
        begin++;
        if(begin >= maxSectionSize)
        {
            begin = 0;
            section = &m_sections.back();
            section->changed = true;
        }
    }

}

Oct::Section::Section(double pointSize, const covise::ColorMap& map, float min, float max, osg::MatrixTransform* parent)
: m_parent(parent)
, points(new osg::Geometry)
, surface(new osg::Geometry)
, vertices(new osg::Vec3Array)
, species(new osg::FloatArray)
, pointPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS))
, reducedPointPrimitiveSet(new osg::DrawElementsUInt(osg::PrimitiveSet::POINTS))
{
    osg::ref_ptr<osg::StateSet> stateSet = VRSceneGraph::instance()->loadDefaultPointstate(pointSize);
    points->setVertexArray(vertices);
    points->setVertexAttribArray(DataAttrib, species, osg::Array::BIND_PER_VERTEX);
    points->setUseDisplayList(false);
    points->setSupportsDisplayList(false);
    points->setUseVertexBufferObjects(true);
    points->setStateSet(stateSet);
    pointPrimitiveSet->setDataVariance(osg::Object::DYNAMIC);
    points->insertPrimitiveSet(allPointsPrimitiveIndex, pointPrimitiveSet);
    points->insertPrimitiveSet(reducedPointsPrimitiveIndex, reducedPointPrimitiveSet);
    parent->addChild(points);
    points->setName("OctPoints");
    opencover::applyShader(points, map, min, max, "OctPoints");

    stateSet = VRSceneGraph::instance()->loadDefaultGeostate();
    surface->setVertexArray(vertices);
    surface->setVertexAttribArray(DataAttrib, species, osg::Array::BIND_PER_VERTEX);
    surface->setUseDisplayList(false);
    surface->setSupportsDisplayList(false);
    surface->setUseVertexBufferObjects(true);
    surface->setStateSet(stateSet);

    parent->addChild(surface);
    surface->setName("OctSurface");
    opencover::applyShader(surface, map, min, max, "MapColorsAttrib");
}

Oct::Section::~Section()
{
    show(Invisible);
}

bool Oct::Section::append(const osg::Vec3 &pos, float val)
{
    if(vertices->size() >= maxSectionSize)
        return false;
    vertices->push_back(pos);
    species->push_back(val);
    changed = true;
    return true;
}

osg::Vec3 getNormal(const osg::Vec3Array& vertices, size_t vertexIndex, size_t numPointsPerCycle)
{
    using namespace osg;
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

osg::ref_ptr<osg::Vec3Array> calculateNormals(osg::ref_ptr<osg::Vec3Array> &vertices, size_t numPointsPerCycle)
{
    using namespace osg;
    
    ref_ptr<Vec3Array> normals = new Vec3Array;
    
    for (size_t i = 0; i < vertices->size() - numPointsPerCycle - 1; i++)
        normals->push_back(getNormal(*vertices, i, numPointsPerCycle));
    return normals;
}

void Oct::Section::createSurface()
{
    surfacePrimitiveSet = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS);
    surfacePrimitiveSet->setDataVariance(osg::Object::DYNAMIC);
    constexpr size_t numPointsPerCycle = 200;
    for (size_t i = 0; i < vertices->size() - numPointsPerCycle - 1; i++)
    {
        surfacePrimitiveSet->push_back(i);
        surfacePrimitiveSet->push_back(i + numPointsPerCycle);
        surfacePrimitiveSet->push_back(i + numPointsPerCycle + 1);
        surfacePrimitiveSet->push_back(i + 1);
    }
    surface->addPrimitiveSet(surfacePrimitiveSet);
    auto normals = calculateNormals(vertices, numPointsPerCycle);
    surface->setVertexArray(vertices);
    surface->setNormalArray(normals, osg::Array::BIND_PER_VERTEX);
    surface->setVertexAttribArray(DataAttrib, species, osg::Array::BIND_PER_VERTEX);
}

void Oct::Section::show(Visibility s)
{
    if(status == s)
        return;

    if(s & PointsOnly && !(status & PointsOnly))
    {
        m_parent->addChild(points);
    }
    if(s & SurfaceOnly && !(status & SurfaceOnly))
    {
        m_parent->addChild(surface);
    }
    if(!(s & PointsOnly) && status & PointsOnly)
    {
        m_parent->removeChild(points);
    }
    if(!(s & SurfaceOnly) && status & SurfaceOnly)
    {
        m_parent->removeChild(surface);
    }
    status = s;
}




void Oct::updateGeo(bool paused)
{
    if(paused)
        return;
    auto toolHeadPos = toolHeadInTableCoords();
    addPoints(m_attributeName->selectedItem(), toolHeadPos, osg::Z_AXIS, 0.005);
    // addPoints(m_attributeName->selectedItem(), toolHeadPos, osg::Z_AXIS, 0.0015);


    for(auto &section : m_sections)
    {
        if(section.changed)
        {
            section.changed = false;
            section.points->setVertexArray(section.vertices);
            section.points->setVertexAttribArray(DataAttrib, section.species, osg::Array::BIND_PER_VERTEX);
            section.pointPrimitiveSet->setFirst(0);
            section.pointPrimitiveSet->setCount(section.vertices->size());
        }
    }
}

Oct::Section &Oct::addSection()
{
    auto &section = m_sections.emplace_back(m_pointSizeSlider->value(), m_colorMapSelector->selectedMap(), m_minAttribute->number(), m_maxAttribute->number(), m_tableNode);
    if(m_sections.size() > numSections)
        m_sections.pop_front();
    if(m_numSectionsSlider->value() > 0 && m_sections.size() > m_numSectionsSlider->value())
        m_sections.front().show(Section::Invisible);
    if(!m_showSurfaceBtn->state())
        section.show(Section::PointsOnly);
    return section;
}

std::vector<std::string> Oct::getAttributes()
{
    return m_client->availableNumericalArrays();
}