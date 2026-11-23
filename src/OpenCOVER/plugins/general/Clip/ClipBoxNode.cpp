#include "ClipBoxNode.h"
#include "utils.h"
#include <cover/coVRPluginSupport.h>
#include <osg/Depth>

namespace
{
// ghost cube colors (unselected = orange, selected = red)
const osg::Vec3 CLIP_GHOST_NORMAL(1.0f, 0.55f, 0.1f);
const osg::Vec3 CLIP_GHOST_SELECTED(1.0f, 0.1f, 0.1f);

// name prefix of the plugin's box nodes, so visitors can find / skip them
const std::string CLIP_BOX_PREFIX = "ClipBox_";

// unique box node name from a single running counter
static std::string nextBoxName()
{
    static int counter = 0;
    return CLIP_BOX_PREFIX + std::to_string(counter++);
}
}

ClipBoxNode::ClipBoxNode(const std::string &name)
    : osg::MatrixTransform()
{
    setName(name);
    buildGeometry();

    // selectable/movable but no collision, walking, touching or ray
    // intersection (the box is purely visual + data for the shader)
    setNodeMask(opencover::Isect::Visible | opencover::Isect::Pick);
}
// highlight the box while it is selected (the selection plugin's
// selectionChanged listener toggles this on the selected box)
void ClipBoxNode::setSelected(bool selected)
{
    m_selected = selected;
    if (selected)
        setGhostColor(CLIP_GHOST_SELECTED, 0.45f, 0.6f);
    else
        setGhostColor(CLIP_GHOST_NORMAL, 0.25f, 0.35f);
}

bool ClipBoxNode::isSelected() const { return m_selected; }

void ClipBoxNode::setSize(const osg::Vec3 &size)
{
    m_size = size;
    rebuildMatrix();
}

// set the box's extent in its parent space with an explicit min/max
// (center-preserving when the box already has a size)
void ClipBoxNode::setAABB(const osg::Vec3 &min, const osg::Vec3 &max)
{
    m_size = max - min;
    setMatrix(osg::Matrix::scale(m_size) * osg::Matrix::translate((min + max) * 0.5f));
}

const osg::Vec3 &ClipBoxNode::size() const { return m_size; }

// the box's world-space AABB: transform the unit cube's corners through
// the full parent chain (halted at `haltAt`, usually the objects root)
void ClipBoxNode::worldAABB(osg::Node *haltAt, osg::Vec3 &min, osg::Vec3 &max) const
{
    osg::NodePathList paths = getParentalNodePaths(haltAt);
    if (paths.empty())
    {
        min = max = osg::Vec3(0.0f, 0.0f, 0.0f);
        return;
    }
    const osg::Matrix m = localToWorldBelowRoot(paths.front());
    min.set(FLT_MAX, FLT_MAX, FLT_MAX);
    max.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = 0; i < 8; ++i)
    {
        osg::Vec3 c((i & 1) ? 0.5f : -0.5f, (i & 2) ? 0.5f : -0.5f, (i & 4) ? 0.5f : -0.5f);
        // NOTE: `c * m` (preMult), NOT `m * c`: in this OSG build (Matrixd)
        // the `Matrix * Vec3` operator is a column-vector product with a
        // perspective divide 1/(t.z*cz + 1), which distorts every point of
        // a matrix carrying a z-translation (box AABB came out warped and
        // 1.58x inflated). `Vec3 * Matrix` is the row-vector convention
        // OSG matrices are built in and divides by exactly 1 for affine
        // matrices.
        c = c * m;
        min = osg::Vec3(std::min(min.x(), c.x()), std::min(min.y(), c.y()), std::min(min.z(), c.z()));
        max = osg::Vec3(std::max(max.x(), c.x()), std::max(max.y(), c.y()), std::max(max.z(), c.z()));
    }
}

void ClipBoxNode::buildGeometry()
{
    // unit cube, 8 corners
    static const float C[8][3] = {
        { -0.5f, -0.5f, -0.5f }, { 0.5f, -0.5f, -0.5f },
        { 0.5f, 0.5f, -0.5f }, { -0.5f, 0.5f, -0.5f },
        { -0.5f, -0.5f, 0.5f }, { 0.5f, -0.5f, 0.5f },
        { 0.5f, 0.5f, 0.5f }, { -0.5f, 0.5f, 0.5f }
    };

    static const int EDGES[12][2] = {
        { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },
        { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },
        { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }
    };
    static const int FACES[6][4] = {
        { 0, 1, 2, 3 }, { 4, 5, 6, 7 },
        { 0, 1, 5, 4 }, { 2, 3, 7, 6 },
        { 0, 3, 7, 4 }, { 1, 2, 6, 5 }
    };

    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array();
    for (const auto &c : C)
        verts->push_back(osg::Vec3(c[0], c[1], c[2]));

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry();
    geom->setVertexArray(verts.get());
    for (const auto &e : EDGES)
        geom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, e[0], 2));
    for (const auto &f : FACES)
    {
        // quad = two triangles (a, b, c) + (a, c, d)
        osg::ref_ptr<osg::DrawElementsUInt> idx = new osg::DrawElementsUInt(GL_TRIANGLES, 6);
        (*idx)[0] = f[0];
        (*idx)[1] = f[1];
        (*idx)[2] = f[2];
        (*idx)[3] = f[0];
        (*idx)[4] = f[2];
        (*idx)[5] = f[3];
        geom->addPrimitiveSet(idx);
    }

    m_stateSet = geom->getOrCreateStateSet();
    m_material = new osg::Material();
    setGhostColor(CLIP_GHOST_NORMAL, 0.25f, 0.35f);
    m_stateSet->setAttributeAndModes(m_material.get(), osg::StateAttribute::ON);
    m_stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    m_stateSet->setMode(GL_BLEND, osg::StateAttribute::ON);
    m_stateSet->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
    m_stateSet->setAttributeAndModes(new osg::Depth(osg::Depth::LESS, 0.0f, 1.0f, false), osg::StateAttribute::ON);
    m_stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    osg::ref_ptr<osg::Geode> geode = new osg::Geode();
    geode->addDrawable(geom.get());
    addChild(geode.get());
}

void ClipBoxNode::setGhostColor(const osg::Vec3 &rgb, float alpha, float emissionAlpha)
{
    if (!m_material)
        return;
    osg::Vec4 c(rgb.x(), rgb.y(), rgb.z(), alpha);
    m_material->setDiffuse(osg::Material::FRONT, c);
    m_material->setAmbient(osg::Material::FRONT, c);
    m_material->setEmission(osg::Material::FRONT, osg::Vec4(rgb.x(), rgb.y(), rgb.z(), emissionAlpha));
}

// matrix = scale(size); the translation is the local origin, so the box
// occupies [origin, origin+size] in parent space (bottom-anchored);
// the Move plugin composes its moves on top of the current matrix
void ClipBoxNode::rebuildMatrix()
{
    osg::Matrix m = getMatrix();
    osg::Vec3 trans, scale, soScale;
    osg::Quat rot, soRot;
    m.decompose(trans, rot, scale, soRot);
    setMatrix(osg::Matrix::scale(m_size) * osg::Matrix::translate(trans));
}
