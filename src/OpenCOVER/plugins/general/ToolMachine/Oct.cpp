#include "Oct.h"

#include <cover/VRSceneGraph.h>
#include <PluginUtil/coShaderUtil.h>
#include <DataClient/DataClient.h>
#include <osg/Point>
#include <osg/VertexArrayState>

#include "../CsvPointCloud/OctUtils.h"

using namespace opencover;

constexpr unsigned int allPointsPrimitiveIndex = 0;
constexpr unsigned int reducedPointsPrimitiveIndex = 1;

Oct::Oct(opencover::ui::Group *group, config::File &file, osg::MatrixTransform *toolHeadNode, osg::MatrixTransform *tableNode)
: Tool(group, file, toolHeadNode, tableNode)
, m_pointSizeSlider(new ui::Slider(group, "pointSize"))
, m_showSurfaceBtn(new ui::Button(group, "showSurface"))
, m_switchVecScalar(new ui::Button(group, "toggleData"))

{
    m_pointSizeSlider->setBounds(0, 20);
    m_pointSizeSlider->setValue(4);
    m_pointSizeSlider->setCallback([this](ui::Slider::ValueType v, bool r)
    {
        for(auto &section : m_sections)
            static_cast<osg::Point *>(section.points->getStateSet()->getAttribute(osg::StateAttribute::Type::POINT))->setSize(m_pointSizeSlider->value());
    });

    m_numSectionsSlider->setUpdater([this](){
        Section::Visibility vis = Section::Visible;
        auto val = m_numSectionsSlider->getValue();
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

constexpr size_t maxSectionSize = 50000;
constexpr size_t numSections = 500;


void Oct::clear()
{
    m_sections.clear();
}

void Oct::applyShader(const opencover::ColorMap& map)
{
    for(auto &section : m_sections)
        opencover::applyShader(section.points, map, "OctPoints");
}

// constexpr auto phaseOffset = 4.01426;
const auto phaseOffset = osg::PI_4 + osg::PI_2 + 15 / 100 *osg::PI;


void Oct::addPoints(const opencover::dataclient::MultiDimensionalArray<double> &data, const osg::Vec3 &toolHeadPos, const osg::Vec3 &up, float r)
{
    std::vector<double> offset, vector;
    double scalar;
    if(!m_client->isConnected())
        return;
    scalar = data.data[3];
    vector.resize(200);
    offset.resize(200);
    std::copy(data.data.begin() + 4, data.data.begin() + 204, offset.begin());
    std::copy(data.data.begin() + 204, data.data.begin() + 404, vector.begin());

    float increment = -2 *osg::PI /offset.size();
    if(m_sections.empty())
        addSection(offset.size());
    auto *section = &m_sections.back();

    correctLastUpdate(toolHeadPos);
    m_lastUpdatePos = toolHeadPos;
    
    for (size_t i = 0; i < offset.size(); i++)
    {
        float theta = increment * i;
        osg::Vec3 v{
            toolHeadPos.x() + r * (float)cos(theta + phaseOffset),
            toolHeadPos.y() + (float)offset[i] * m_opcUaToVrmlScale,
            toolHeadPos.z() + r * (float)sin(theta + phaseOffset) };
        // if(!section->append(v, i))
        if(!section->append(v, m_switchVecScalar->state() ? vector[i] : scalar))
        {
            section->createSurface();
            section = &addSection(offset.size());
            section->startIndex = i;
            section->append(v, m_switchVecScalar->state() ? vector[i] : scalar);
        }
    }
}

bool Oct::Section::lastIsOutsidePoint()
{
    auto pointIndex = vertices->size() - 1;
    if(pointIndex < vertsPerCircle)
        return false;
    auto speed = (*vertices)[pointIndex] - (*vertices)[pointIndex - vertsPerCircle];
    float increment = -2 *osg::PI /vertsPerCircle;
    auto angle = ((startIndex + pointIndex) % vertsPerCircle) *increment + phaseOffset; 
    auto dx = speed.x();
    auto dy = speed.z();
    return std::abs((cos(angle) * dx + sin(angle) * dy)/sqrt(std::pow(dx, 2) + std::pow(dy, 2))) < 0.1;
}

void Oct::correctLastUpdate(const osg::Vec3 &toolHeadPos)
{
    if(m_sections.size() == 1 && m_sections[0].vertices->empty())
        return;
    auto *section = &m_sections.back();
    long begin = section->vertices->size() - section->vertsPerCircle;
    if(begin < 0)
    {
        section = &m_sections[m_sections.size() - 2];
        begin += section->vertices->size();
        section->changed = true;
    }
    auto end = std::min(begin + section->vertsPerCircle, maxSectionSize);
    float angleIncrement = 2 *osg::PI / section->vertsPerCircle;
    auto posIncrement = (toolHeadPos - m_lastUpdatePos) / section->vertsPerCircle;
    for (size_t i = 0; i < section->vertsPerCircle; i++)
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

Oct::Section::Section(size_t vertsPerCircle, double pointSize, const opencover::ColorMap& map, osg::MatrixTransform* parent)
: m_parent(parent)
, points(new osg::Geometry)
, surface(new osg::Geometry)
, vertices(new osg::Vec3Array)
, species(new osg::FloatArray)
, pointPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS))
, reducedPointPrimitiveSet(new osg::DrawElementsUInt(osg::PrimitiveSet::POINTS))
, vertsPerCircle(vertsPerCircle)
{
    assert(vertsPerCircle > 0);
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
    opencover::applyShader(points, map, "OctPoints");

    stateSet = VRSceneGraph::instance()->loadDefaultGeostate();
    surface->setVertexArray(vertices);
    surface->setVertexAttribArray(DataAttrib, species, osg::Array::BIND_PER_VERTEX);
    surface->setUseDisplayList(false);
    surface->setSupportsDisplayList(false);
    surface->setUseVertexBufferObjects(true);
    surface->setStateSet(stateSet);
    parent->addChild(surface);
    surface->setName("OctSurface");
    opencover::applyShader(surface, map, "MapColorsAttrib");
}

Oct::Section::~Section()
{
    show(Invisible);
    m_parent->removeChild(points);
}

bool Oct::Section::append(const osg::Vec3 &pos, float val)
{
    if(vertices->size() >= maxSectionSize)
        return false;
    vertices->push_back(pos);
    species->push_back(val);
    if(lastIsOutsidePoint())
        reducedPointPrimitiveSet->push_back(vertices->size() - 1);
    changed = true;
    return true;
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
    auto normals = oct::calculateNormals(vertices, numPointsPerCycle);
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
        // m_parent->addChild(points);
        points->insertPrimitiveSet(allPointsPrimitiveIndex, pointPrimitiveSet);
    }
    if(s & SurfaceOnly && !(status & SurfaceOnly))
    {
        m_parent->addChild(surface);
    }
    if(!(s & PointsOnly) && status & PointsOnly)
    {
        // m_parent->removeChild(points);
        points->removePrimitiveSet(allPointsPrimitiveIndex);

    }
    if(!(s & SurfaceOnly) && status & SurfaceOnly)
    {
        m_parent->removeChild(surface);
    }
    status = s;
}

void Oct::updateGeo(bool paused, const opencover::dataclient::MultiDimensionalArray<double> &data)
{
    if(paused)
        return;
    auto toolHeadPos = toolHeadInTableCoords();
    addPoints(data, toolHeadPos, osg::Z_AXIS, 0.0015);


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

Oct::Section &Oct::addSection(size_t numVerts)
{
    auto &section = m_sections.emplace_back(numVerts, m_pointSizeSlider->value(), m_colorMapSelector->colorMap(), m_tableNode);
    if(m_sections.size() > numSections)
        m_sections.pop_front();
    if(m_numSectionsSlider->getValue() > 0 && m_sections.size() > m_numSectionsSlider->getValue())
        m_sections[m_sections.size() - m_numSectionsSlider->getValue()].show(Section::Invisible);
    if(!m_showSurfaceBtn->state())
        section.show(Section::PointsOnly);
    return section;
}

std::vector<std::string> Oct::getAttributes()
{
    return m_client->availableNumericalArrays();
}

void Oct::attributeChanged(float value)
{
    //this tool only works in array mode
}
