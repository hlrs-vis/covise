#include "Clip.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coVRLighting.h>
#include <cover/coVRIntersectionInteractorManager.h>
#include <cover/coVRIntersectionInteractor.h>
#include <cover/ui/Button.h>
#include <cover/ui/Group.h>
#include <cover/ui/Label.h>
#include <cover/ui/Menu.h>
#include <cover/ui/View.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/NodeVisitor>
#include <osg/Material>
#include <osg/Texture>
#include <osg/ShapeDrawable>
#include <osg/Program>
#include <osg/Shader>
#include <osg/Uniform>
#include <osg/StateSet>
#include <osg/StateAttribute>
#include <osg/Depth>

#include <algorithm>
#include <cfloat>
#include <cstdio>

using namespace opencover;

// ghost cube colors (unselected = orange, selected = red)
const osg::Vec3 CLIP_GHOST_NORMAL(1.0f, 0.55f, 0.1f);
const osg::Vec3 CLIP_GHOST_SELECTED(1.0f, 0.1f, 0.1f);

namespace
{
// the maximum number of boxes the shader can test at once (fixed uniform
// array size); boxes beyond that are ignored
constexpr int CLIP_MAX_BOXES = 16;
// default size (width, height, depth) of a manually placed box, in scene units
constexpr float CLIP_DEFAULT_SIZE[3] = {10.0f, 2.0f, 10.0f};

// name prefix of the plugin's box nodes, so visitors can find / skip them
const std::string CLIP_BOX_PREFIX = "ClipBox_";

// unique box node name from a single running counter
static std::string nextBoxName()
{
    static int counter = 0;
    return CLIP_BOX_PREFIX + std::to_string(counter++);
}
}

// ---------------------------------------------------------------------------
// scene-graph visitors (file scope: used by the ClipPlugin methods too)
// ---------------------------------------------------------------------------

// collect all geodes in a subtree
class GeodeAllVisitor : public osg::NodeVisitor
{
public:
    explicit GeodeAllVisitor(std::vector<osg::Geode *> &result)
        : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
        , m_result(result)
    {
    }

    void apply(osg::Geode &geode) override { m_result.push_back(&geode); }

    void apply(osg::Node &n) override
    {
        // never collect our own box nodes
        if (n.getName().find(CLIP_BOX_PREFIX) == 0)
            return;
        traverse(n);
    }

private:
    std::vector<osg::Geode *> &m_result;
};

// find the first node carrying an exact name inside a subtree
class NameSearchVisitor : public osg::NodeVisitor
{
public:
    explicit NameSearchVisitor(const std::string &name)
        : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
        , m_name(name)
    {
    }

    osg::Node *node = nullptr;

    void apply(osg::Node &n) override
    {
        if (node)
            return;
        if (n.getName().find(CLIP_BOX_PREFIX) == 0)
            return;
        if (n.getName() == m_name)
        {
            node = &n;
            return;
        }
        traverse(n);
    }

private:
    const std::string &m_name;
};

// ---------------------------------------------------------------------------
// local-to-root matrix WITHOUT osg::computeLocalToWorld: the objects root is
// a ClipNode (an osg::Camera), and the free osg::computeLocalToWorld skips
// absolute/root cameras -- once the root has a parent (i.e. after scene setup)
// it silently drops the whole chain and returns identity.  Instead walk the
// path manually and apply each node's virtual computeLocalToWorldMatrix, which
// is correct for both MatrixTransform and Camera and stable across frames.
// The path runs root->...->node; the root itself maps root space to itself.
// ---------------------------------------------------------------------------
osg::Matrix localToWorldBelowRoot(const osg::NodePath &path)
{
    osg::Matrix m;
    for (unsigned int i = 1; i < path.size(); ++i)   // skip path[0] = root (maps root->root)
    {
        osg::Transform *t = path[i]->asTransform();
        if (t)
            t->computeLocalToWorldMatrix(m, 0);   // virtual: handles RELATIVE/ABSOLUTE + per-class matrix
    }
    return m;
}

// ---------------------------------------------------------------------------
// ClipBoxNode: a movable, axis-aligned box in the scene. The geometry is a
// unit cube ([-0.5, 0.5]^3) rendered as a ghost (edges + translucent faces);
// the box's world AABB is always derived from its world transform, so moving
// it with the Move plugin just works.
// ---------------------------------------------------------------------------
class ClipBoxNode : public osg::MatrixTransform
{
public:
    explicit ClipBoxNode(const std::string &name)
        : osg::MatrixTransform()
    {
        setName(name);
        buildGeometry();

        // selectable/movable but no collision, walking, touching or ray
        // intersection (the box is purely visual + data for the shader)
        setNodeMask(Isect::Visible | Isect::Pick);
    }
    // highlight the box while it is selected (the selection plugin's
    // selectionChanged listener toggles this on the selected box)
    void setSelected(bool selected)
    {
        m_selected = selected;
        if (selected)
            setGhostColor(CLIP_GHOST_SELECTED, 0.45f, 0.6f);
        else
            setGhostColor(CLIP_GHOST_NORMAL, 0.25f, 0.35f);
    }

    bool isSelected() const { return m_selected; }

    void setSize(const osg::Vec3 &size)
    {
        m_size = size;
        rebuildMatrix();
    }

    const osg::Vec3 &size() const { return m_size; }

    // the box's world-space AABB: transform the unit cube's corners through
    // the full parent chain (halted at `haltAt`, usually the objects root)
    void worldAABB(osg::Node *haltAt, osg::Vec3 &min, osg::Vec3 &max) const
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
private:
    void buildGeometry()
    {
        // unit cube, 8 corners
        static const float C[8][3] = {
            {-0.5f, -0.5f, -0.5f}, {0.5f, -0.5f, -0.5f},
            {0.5f, 0.5f, -0.5f}, {-0.5f, 0.5f, -0.5f},
            {-0.5f, -0.5f, 0.5f}, {0.5f, -0.5f, 0.5f},
            {0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}};

        static const int EDGES[12][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            {0, 4}, {1, 5}, {2, 6}, {3, 7}};
        static const int FACES[6][4] = {
            {0, 1, 2, 3}, {4, 5, 6, 7},
            {0, 1, 5, 4}, {2, 3, 7, 6},
            {0, 3, 7, 4}, {1, 2, 6, 5}};

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

    void setGhostColor(const osg::Vec3 &rgb, float alpha, float emissionAlpha)
    {
        if (!m_material)
            return;
        osg::Vec4 c(rgb.x(), rgb.y(), rgb.z(), alpha);
        m_material->setDiffuse(osg::Material::FRONT, c);
        m_material->setAmbient(osg::Material::FRONT, c);
        m_material->setEmission(osg::Material::FRONT, osg::Vec4(rgb.x(), rgb.y(), rgb.z(), emissionAlpha));
    }

    osg::StateSet *m_stateSet = nullptr;
    osg::ref_ptr<osg::Material> m_material;
    bool m_selected = false;
    osg::Vec3 m_size{1.0f, 1.0f, 1.0f};

    // matrix = scale(size); the translation is the local origin, so the box
    // occupies [origin, origin+size] in parent space (bottom-anchored);
    // the Move plugin composes its moves on top of the current matrix
    void rebuildMatrix()
    {
        osg::Matrix m = getMatrix();
        osg::Vec3 trans, scale, soScale;
        osg::Quat rot, soRot;
        m.decompose(trans, rot, scale, soRot);
        setMatrix(osg::Matrix::scale(m_size) * osg::Matrix::translate(trans));
    }
};

// ---------------------------------------------------------------------------
// the clip shader (GLSL 150). Attached to every geode of a clipped target it:
//   - passes the vertex's ObjectsRoot-space position (computed with the
//     per-geode `clipWorldMatrix` uniform, which is what makes the boxes work
//     in the same space as the objects) and its normal,
//   - tests the world position of every fragment against all clip boxes and
//     discards the fragments that fall inside any box.
//
// Because the fragment shader takes over the fixed-function pipeline it also
// reproduces the target's own look: material (or vertex color) times an
// optional base texture, lit by the scene headlight (ambient + Lambert).
// The fixed-function attributes OSG always provides (osg_Vertex, osg_Normal,
// osg_Color, osg_TexCoord0) are safe to declare: OSG binds missing geometry
// attributes to constant defaults (normal = (0,0,0), color = (1,1,1,0),
// texcoord = (0,0)), which is exactly what the `clipHasNormal` /
// `clipHasVertexColor` / `clipHasTexture` guards are for.
// ---------------------------------------------------------------------------
// LEGACY GLSL (NO #version directive) - REQUIRED in this OSG 3.6.5 build.
// That build was compiled with OSG_GL_FIXED_FUNCTION_AVAILABLE=ON, which means:
//   * State sets _useModelViewAndProjectionUniforms = false, so the
//     osg_ModelViewMatrix / osg_ProjectionMatrix / osg_ModelViewProjectionMatrix
//     uniforms are NEVER applied (RenderLeaf only applies them when the flag is
//     true). Using them leaves them all-zero and projects every vertex to the
//     origin.
//   * State sets _useVertexAttributeAliasing = false, so OSG feeds the standard
//     fixed-function client arrays, i.e. the BUILT-IN gl_Vertex / gl_Normal /
//     gl_Color / gl_MultiTexCoord0 attributes (NOT the osg_* aliases).
//   * A "#version 150" (>= 140) program has the pre-140 built-ins removed
//     (gl_ModelViewProjectionMatrix, gl_Vertex, gl_MultiTexCoord0, ...), so it
//     CANNOT use the fixed-function matrices and FAILS to compile (error C7616);
//     OSG then silently falls back to no program -> nothing gets cut.
// The only configuration that both compiles AND receives real matrix + vertex
// data in this build is legacy GLSL: built-in attributes +
// gl_ModelViewProjectionMatrix; both requirements hold for the shaders below.
const char *CLIP_VERTEX_SHADER =
    "uniform mat4 clipWorldMatrix;\n"
    "varying vec4 clipRootPos;\n"
    "varying vec3 clipNormal;\n"
    "varying vec4 clipColor;\n"
    "varying vec2 clipTexCoord;\n"
    "void main()\n"
    "{\n"
    "    clipRootPos = clipWorldMatrix * gl_Vertex;\n"
    "    // proper normal transform (inverse transpose), safe for scaled targets\n"
    "    clipNormal = transpose(inverse(mat3(clipWorldMatrix))) * gl_Normal;\n"
    "    clipColor = gl_Color;\n"
    "    clipTexCoord = gl_MultiTexCoord0.xy;\n"
    "    // gl_ModelViewProjectionMatrix is the fixed-function built-in (object ->\n"
    "    // clip). Positioning uses the *local* vertex; clipRootPos (which has\n"
    "    // clipWorldMatrix applied) is only for the box test in the fragment shader.\n"
    "    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
    "}\n";

// Legacy GLSL fragment shader (see CLIP_VERTEX_SHADER for why legacy is required).
// NOTE the box loop: the upper bound must be a *constant* (16) so the loop index
// is a valid constant-index-expression for the uniform array access clipBoxMin[i]
// (GLSL well-formed-loop rule). clipBoxCount is enforced with an early break.
const char *CLIP_FRAGMENT_SHADER =
    "uniform int clipBoxCount;\n"
    "uniform vec3 clipBoxMin[16];\n"
    "uniform vec3 clipBoxMax[16];\n"
    "uniform vec4 clipLightPos;\n"
    "uniform vec3 clipLightDiffuse;\n"
    "uniform vec3 clipLightAmbient;\n"
    "uniform vec3 clipMatDiffuse;\n"
    "uniform vec3 clipMatAmbient;\n"
    "uniform int clipHasTexture;\n"
    "uniform int clipHasNormal;\n"
    "uniform int clipHasVertexColor;\n"
    "uniform sampler2D clipBaseTexture;\n"
    "varying vec4 clipRootPos;\n"
    "varying vec3 clipNormal;\n"
    "varying vec4 clipColor;\n"
    "varying vec2 clipTexCoord;\n"
    "void main()\n"
"{\n"
"    // ---- the cut: discard fragments inside any clip box ----\n"
"    int inBox = 0;\n"
"    for (int i = 0; i < 16; ++i)\n"
"    {\n"
"        if (i >= clipBoxCount)\n"
"            break;\n"
"        if (clipRootPos.x > clipBoxMin[i].x && clipRootPos.x < clipBoxMax[i].x &&\n"
"            clipRootPos.y > clipBoxMin[i].y && clipRootPos.y < clipBoxMax[i].y &&\n"
"            clipRootPos.z > clipBoxMin[i].z && clipRootPos.z < clipBoxMax[i].z)\n"
"            inBox = 1;\n"
"    }\n"
"    if (inBox == 1)\n"
"        discard;\n"
    "    // ---- reproduce the target's own look (fixed-function equivalent:\n"
    "    // (matAmbient*lightAmbient + matDiffuse*lightDiffuse*NdotL) * vtxColor * tex)\n"
    "    vec4 tcol = vec4(1.0);\n"
    "    if (clipHasTexture == 1)\n"
    "        tcol = texture2D(clipBaseTexture, clipTexCoord);\n"
    "    vec3 vc = (clipHasVertexColor == 1) ? clipColor.rgb : vec3(1.0);\n"
    "    vec3 N = clipNormal;\n"
    "    if (clipHasNormal == 0 || dot(N, N) < 1e-8)\n"
    "        N = vec3(0.0, 0.0, 1.0);\n"
    "    else\n"
    "        N = normalize(N);\n"
    "    vec3 L;\n"
    "    if (clipLightPos.w != 0.0)\n"
    "        L = normalize(clipLightPos.xyz - clipRootPos.xyz); // point light (COVER headlight)\n"
    "    else\n"
    "        L = -normalize(clipLightPos.xyz); // directional light (xyz = direction OF the light, flip it)\n"
    "    float diff = max(dot(N, L), 0.0);\n"
    "    vec3 lit = clipMatAmbient * clipLightAmbient + clipMatDiffuse * clipLightDiffuse * diff;\n"
    "    gl_FragColor = vec4(lit * vc * tcol.rgb, 1.0);\n"
    "}\n";

// ---------------------------------------------------------------------------
// plugin implementation
// ---------------------------------------------------------------------------
ClipPlugin::ClipPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("Clip", cover->ui)
{
}

ClipPlugin::~ClipPlugin()
{
    coVRSelectionManager::instance()->removeListener(this);
}

bool ClipPlugin::init()
{
    std::fprintf(stderr, "CLIP: plugin loaded (v2: shader box cutting)\n");

    m_menu = new ui::Menu("ClipMenu", this);
    m_menu->setText("Clip objects");

    m_group = new ui::Group(m_menu, "ClipObjects");
    m_group->setText("Cut holes with boxes");

    m_targetBtn = new ui::Button(m_group, "PickTarget");
    m_targetBtn->setText("Pick target (e.g. terrain)");
    m_targetBtn->setCallback([this](bool state) {
        if (state)
            setMode(Mode::PickTarget);
        else if (m_mode == Mode::PickTarget)
            setMode(Mode::None);
    });

    m_boxBtn = new ui::Button(m_group, "BoxFromObject");
    m_boxBtn->setText("Box from picked object");
    m_boxBtn->setCallback([this](bool state) {
        if (state)
            setMode(Mode::BoxFromObject);
        else if (m_mode == Mode::BoxFromObject)
            setMode(Mode::None);
    });

    m_addBoxBtn = new ui::Button(m_group, "AddBox");
    m_addBoxBtn->setText("Add box at hit point");
    m_addBoxBtn->setCallback([this](bool state) {
        // momentary action: the click toggled the checkbox on, create the box
        // (if there is a hit point) and immediately switch it off again
        if (!state)
            return;
        osg::Vec3 pos;
        if (currentHitPos(pos))
            addBoxAt(pos);
        else
            m_statusLabel->setText("Add box: move the interactor onto a surface first, then press again");
        m_addBoxBtn->setState(false);
    });

    m_removeBoxBtn = new ui::Button(m_group, "RemoveLastBox");
    m_removeBoxBtn->setText("Remove last box");
    m_removeBoxBtn->setCallback([this](bool) {
        removeLastBox();
    });

    m_showBoxesBtn = new ui::Button(m_group, "ShowBoxes");
    m_showBoxesBtn->setText("Show box ghosts");
    m_showBoxesBtn->setState(m_showBoxes);
    m_showBoxesBtn->setCallback([this](bool state) {
        if (state != m_showBoxes)
            setBoxesVisible(state);
    });

    m_clearBtn = new ui::Button(m_group, "Clear");
    m_clearBtn->setText("Clear all");
    m_clearBtn->setCallback([this](bool) {
        clearAll();
    });

    m_statusLabel = new ui::Label(m_group, "Status");
    m_statusLabel->setText("Targets: (none) | Boxes: 0");

    auto *loadSetupBtn = new ui::Button(m_group, "LoadSetup");
    loadSetupBtn->setText("Apply setup from config");
    loadSetupBtn->setCallback([this](bool) {
        loadSetup();
    });

    m_cfgTargets = configStringArray("Setup", "targets", {}, config::Flag::PerModel);
    m_cfgTargets->setUpdater([this]() {
        loadSetup();
    });
    m_cfgShowBoxes = configBool("Setup", "showBoxes", true, config::Flag::PerModel);
    m_showBoxes = m_cfgShowBoxes->value();
    m_showBoxesBtn->setState(m_showBoxes);
    m_cfgShowBoxes->setUpdater([this](const bool &v) {
        setBoxesVisible(v);
    });

    coVRSelectionManager::instance()->addListener(this);

    loadSetup();

    return true;
}

bool ClipPlugin::destroy()
{
    clearAll();
    coVRSelectionManager::instance()->removeListener(this);
    return true;
}

// ---------------------------------------------------------------------------
// scene-graph helpers
// ---------------------------------------------------------------------------

osg::Node *ClipPlugin::topObjectOf(osg::Node *node) const
{
    if (!node)
        return nullptr;
    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return nullptr;

    std::vector<osg::Node *> chain;
    osg::Node *n = node;
    while (n)
    {
        if (n->getName().find(CLIP_BOX_PREFIX) == 0)
            return nullptr; // never clip our own boxes
        chain.push_back(n);
        if (n->getNumParents() == 1 && n->getParent(0) == root)
            break;
        if (n->getNumParents() == 0)
            return nullptr;
        n = n->getParent(0);
    }
    if (chain.empty() || !(chain.back()->getNumParents() == 1 && chain.back()->getParent(0) == root))
        return nullptr;

    for (size_t j = chain.size(); j-- > 0;)
        if (!coVRSelectionManager::instance()->isHelperNode(chain[j]))
            return chain[j];
    return chain.back();
}

osg::Node *ClipPlugin::innerObjectOf(osg::Node *child) const
{
    if (!child)
        return nullptr;
    osg::Node *n = child;
    osg::Group *g = n->asGroup();
    while (g && coVRSelectionManager::instance()->isHelperNode(n) && g->getNumChildren() > 0)
    {
        n = g->getChild(0);
        g = n->asGroup();
    }
    return n;
}

bool ClipPlugin::matchName(const std::string &cfgName, const std::string &nodeName)
{
    if (cfgName.empty() || nodeName.empty())
        return false;
    if (cfgName == nodeName)
        return true;

    // file-loaded objects are named by their (url) path: "terrain" or
    // "terrain.obj" should match "/some/path/terrain.obj"
    const std::string base = cfgName.substr(cfgName.find_last_of("/\\") + 1);
    if (base == nodeName)
        return true;
    if (base.size() > 4 && base.size() <= nodeName.size() &&
        nodeName.compare(nodeName.size() - base.size(), base.size(), base) == 0)
        return true;

    // allow a plain prefix as well ("build" -> "building1.obj")
    return cfgName.size() >= 3 && nodeName.find(cfgName) == 0;
}

osg::Node *ClipPlugin::topNodeOf(const std::string &name) const
{
    if (name.empty())
        return nullptr;

    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return nullptr;

    osg::Node *top = nullptr;
    for (unsigned int i = 0; i < root->getNumChildren(); ++i)
    {
        osg::Node *child = root->getChild(i);
        if (!child)
            continue;
        // a moved object sits INSIDE a helper that is the direct child of the
        // root: match (and return) the object inside it, not the helper
        osg::Node *inner = innerObjectOf(child);
        if (!inner || inner->getName().find(CLIP_BOX_PREFIX) == 0)
            continue;
        const std::string n = inner->getName();
        if (n == name)
        {
            top = inner;
            break;
        }
        if (!top && matchName(name, n))
            top = inner;
    }
    return top;
}

osg::Geode *ClipPlugin::findGeodeForName(const std::string &name) const
{
    if (name.empty())
        return nullptr;

    // 1) a top-level object carrying the name (helper-aware)
    osg::Node *top = topNodeOf(name);
    if (top)
    {
        if (top->asGeode())
            return top->asGeode();
        std::vector<osg::Geode *> list;
        GeodeAllVisitor v2(list);
        top->accept(v2);
        if (!list.empty())
            return list.front();
    }

    // 2) a named sub-node (a geode/group named after an OBJ material group,
    //    a photogrammetry tile, ...) inside any top-level object
    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return nullptr;
    for (unsigned int i = 0; i < root->getNumChildren(); ++i)
    {
        osg::Node *child = root->getChild(i);
        if (!child || child->getName().find(CLIP_BOX_PREFIX) == 0)
            continue;
        NameSearchVisitor v(name);
        child->accept(v);
        if (!v.node)
            continue;
        if (v.node->asGeode())
            return v.node->asGeode();
        std::vector<osg::Geode *> list;
        GeodeAllVisitor gv(list);
        v.node->accept(gv);
        if (!list.empty())
            return list.front();
    }
    return nullptr;
}

std::string ClipPlugin::displayNameOf(osg::Node *node) const
{
    osg::Node *top = topObjectOf(node);
    if (node && node != top && !node->getName().empty())
        return node->getName();
    return top ? top->getName() : "";
}

// the node that is actually clipped for a target name: the top-level object
// when the name matches it, or the named sub-node carrying the name
osg::Node *ClipPlugin::findClipNode(const std::string &name) const
{
    osg::Node *top = topNodeOf(name);
    if (!top)
        return nullptr;
    if (top->getName() == name)
        return top;
    NameSearchVisitor v(name);
    top->accept(v);
    return v.node;
}

void ClipPlugin::geodeWorldAABB(const osg::Geode *geode, osg::Vec3 &min, osg::Vec3 &max)
{
    min.set(FLT_MAX, FLT_MAX, FLT_MAX);
    max.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    if (!geode)
        return;

    osg::Group *root = cover->getObjectsRoot();
    for (unsigned int i = 0; i < geode->getNumDrawables(); ++i)
    {
        const osg::BoundingSphere bs = geode->getDrawable(i)->getBound();
        const osg::Vec3 c = bs.center();
        const float r = bs.radius();
        osg::NodePathList paths = geode->getParentalNodePaths(root);
        if (paths.empty())
            continue;
        const osg::Matrix m = localToWorldBelowRoot(paths.front());
        for (int a = 0; a < 8; ++a)
        {
            osg::Vec3 corner(c.x() + ((a & 1) ? r : -r), c.y() + ((a & 2) ? r : -r), c.z() + ((a & 4) ? r : -r));
            // `corner * m` (preMult), NOT `m * corner` -- see worldAABB: the
            // column-vector product with perspective divide would warp the
            // bound for any geode whose chain carries a translation.
            corner = corner * m;
            min = osg::Vec3(std::min(min.x(), corner.x()), std::min(min.y(), corner.y()), std::min(min.z(), corner.z()));
            max = osg::Vec3(std::max(max.x(), corner.x()), std::max(max.y(), corner.y()), std::max(max.z(), corner.z()));
        }
    }
    if (min.x() > max.x() || min.y() > max.y() || min.z() > max.z())
        min = max = osg::Vec3(0.0f, 0.0f, 0.0f); // empty geode: fall back to the origin
}

bool ClipPlugin::currentHitPos(osg::Vec3 &pos)
{
    auto *ii = coVRIntersectionInteractorManager::the()->getCurrentIntersectionInteractor();
    if (!ii || !ii->isIntersected())
        return false;
    pos = ii->getHitPos();
    return true;
}

void ClipPlugin::setMode(Mode m)
{
    m_mode = m;
    m_targetBtn->setState(m_mode == Mode::PickTarget);
    m_boxBtn->setState(m_mode == Mode::BoxFromObject);
}

// ---------------------------------------------------------------------------
// clip shader injection
// ---------------------------------------------------------------------------

bool ClipPlugin::geodeHasVertexColor(const osg::Geode *geode)
{
    for (unsigned int i = 0; i < geode->getNumDrawables(); ++i)
    {
        auto *geom = geode->getDrawable(i)->asGeometry();
        if (geom && geom->getColorArray())
            return true;
    }
    return false;
}

osg::Texture *ClipPlugin::geodeTexture(osg::Geode *geode, int &unit)
{
    // the outline shader's pattern: the texture sits on unit 0 of one of the
    // drawables' (or the geode's) state sets
    for (int searchUnit = 0; searchUnit < 8; ++searchUnit)
    {
        osg::StateSet *ss = geode->getStateSet();
        osg::Texture *t = ss ? dynamic_cast<osg::Texture *>(ss->getTextureAttribute(searchUnit, osg::StateAttribute::TEXTURE)) : nullptr;
        if (t)
        {
            unit = searchUnit;
            return t;
        }
        for (unsigned int i = 0; i < geode->getNumDrawables(); ++i)
        {
            ss = geode->getDrawable(i)->getStateSet();
            t = ss ? dynamic_cast<osg::Texture *>(ss->getTextureAttribute(searchUnit, osg::StateAttribute::TEXTURE)) : nullptr;
            if (t)
            {
                unit = searchUnit;
                return t;
            }
        }
    }
    return nullptr;
}

bool ClipPlugin::geodeHasNormal(const osg::Geode *geode)
{
    for (unsigned int i = 0; i < geode->getNumDrawables(); ++i)
    {
        auto *geom = geode->getDrawable(i)->asGeometry();
        if (geom && !geom->getNormalArray())
            return false;
    }
    return true;
}

void ClipPlugin::applyClipToGeode(osg::Geode *geode)
{
    if (!geode || m_geodeClip.count(geode))
        return;

    // one shared program instance, attached to every clipped geode
    if (!m_program)
    {
        m_program = new osg::Program();
        auto *vs = new osg::Shader(osg::Shader::VERTEX);
        auto *fs = new osg::Shader(osg::Shader::FRAGMENT);
        vs->setShaderSource(CLIP_VERTEX_SHADER);
        fs->setShaderSource(CLIP_FRAGMENT_SHADER);
        m_program->addShader(vs);
        m_program->addShader(fs);

        // the shared, per-frame uniforms (same instance on all geodes)
        m_boxMin = new osg::Uniform(osg::Uniform::FLOAT_VEC3, "clipBoxMin", 16);
        m_boxMax = new osg::Uniform(osg::Uniform::FLOAT_VEC3, "clipBoxMax", 16);
        m_boxCount = new osg::Uniform(osg::Uniform::INT, "clipBoxCount", 1);
        m_boxCount->setElement(0, 0);
        m_lightPos = new osg::Uniform("clipLightPos", osg::Vec4(0.0f, -1.0f, 1.0f, 0.0f));
        m_lightDiffuse = new osg::Uniform("clipLightDiffuse", osg::Vec3(1.0f, 1.0f, 1.0f));
        m_lightAmbient = new osg::Uniform("clipLightAmbient", osg::Vec3(0.3f, 0.3f, 0.3f));
    }

    GeodeClipSave save;

    // sample the geode's appearance BEFORE touching its state
    // (Material has a protected destructor -> heap-allocated default)
    auto *defaultMat = new osg::Material();
    save.diffuse = defaultMat->getDiffuse(osg::Material::FRONT);
    save.ambient = defaultMat->getAmbient(osg::Material::FRONT);
    osg::StateSet *ss = geode->getStateSet();
    if (ss)
    {
        auto *mat = dynamic_cast<osg::Material *>(ss->getAttribute(osg::StateAttribute::MATERIAL));
        if (mat)
        {
            save.diffuse = mat->getDiffuse(osg::Material::FRONT);
            save.ambient = mat->getAmbient(osg::Material::FRONT);
        }
        save.texUnit = 0;
        save.texture = geodeTexture(geode, save.texUnit);
        if (save.texture)
        {
            save.hadTextureAttribute = ss->getTextureAttribute(save.texUnit, osg::StateAttribute::TEXTURE) != nullptr;
            save.wasDrawableTexture = !save.hadTextureAttribute;
        }
        save.depth = dynamic_cast<osg::Depth *>(ss->getAttribute(osg::StateAttribute::DEPTH));
        save.hadDepth = ss->getMode(GL_DEPTH_TEST) == osg::StateAttribute::ON;
    }
    save.vertexColor = geodeHasVertexColor(geode);
    save.hasNormal = geodeHasNormal(geode);

    ss = geode->getOrCreateStateSet();

    // everything the shader replaces is switched off; the original material /
    // texture / depth are saved above for an exact restore
    {
        auto *mat = dynamic_cast<osg::Material *>(ss->getAttribute(osg::StateAttribute::MATERIAL));
        if (mat)
        {
            save.material = mat;
            ss->removeAttribute(osg::StateAttribute::MATERIAL);
        }
    }
    if (save.hadTextureAttribute)
    {
        save.texture = dynamic_cast<osg::Texture *>(ss->getTextureAttribute(save.texUnit, osg::StateAttribute::TEXTURE));
        ss->removeTextureAttribute(save.texUnit, osg::StateAttribute::TEXTURE);
    }
    if (save.hadDepth)
    {
        save.depth = dynamic_cast<osg::Depth *>(ss->getAttribute(osg::StateAttribute::DEPTH));
        ss->removeMode(GL_DEPTH_TEST);
    }

    // rebind the original texture onto the geode's own state set, where the
    // shader can see it (drawable-level textures are not visible from a
    // geode-level shader)
    if (save.texture)
        ss->setTextureAttributeAndModes(save.texUnit, save.texture, osg::StateAttribute::ON);

    // per-geode uniforms
    // NOTE: must be FLOAT_MAT4 (osg::Matrixf) - the shader declares `uniform mat4`
    // (float). An osg::Matrix (Matrixd) uniform would be applied via
    // glUniformMatrix4dv and fail with GL_INVALID_OPERATION on a float mat4.
    save.worldMatrix = new osg::Uniform("clipWorldMatrix", osg::Matrixf());
    save.matDiffuse = new osg::Uniform("clipMatDiffuse", osg::Vec3(save.diffuse.x(), save.diffuse.y(), save.diffuse.z()));
    save.matAmbient = new osg::Uniform("clipMatAmbient", osg::Vec3(save.ambient.x(), save.ambient.y(), save.ambient.z()));
    save.hasTexture = new osg::Uniform("clipHasTexture", save.texture ? 1 : 0);
    save.hasNormalUniform = new osg::Uniform("clipHasNormal", save.hasNormal ? 1 : 0);
    save.hasVertexColor = new osg::Uniform("clipHasVertexColor", save.vertexColor ? 1 : 0);
    if (save.texture)
        save.texUnitUniform = new osg::Uniform("clipBaseTexture", save.texUnit);

    // attach: shared program + shared + per-geode uniforms
    ss->setAttributeAndModes(m_program.get(), osg::StateAttribute::ON);
    ss->addUniform(m_boxMin.get());
    ss->addUniform(m_boxMax.get());
    ss->addUniform(m_boxCount.get());
    ss->addUniform(m_lightPos.get());
    ss->addUniform(m_lightDiffuse.get());
    ss->addUniform(m_lightAmbient.get());
    ss->addUniform(save.worldMatrix.get());
    ss->addUniform(save.matDiffuse.get());
    ss->addUniform(save.matAmbient.get());
    ss->addUniform(save.hasTexture.get());
    ss->addUniform(save.hasNormalUniform.get());
    ss->addUniform(save.hasVertexColor.get());
    if (save.texUnitUniform)
        ss->addUniform(save.texUnitUniform.get());

    m_geodeClip[geode] = save;

}

void ClipPlugin::removeClipFromGeode(osg::Geode *geode)
{
    auto it = m_geodeClip.find(geode);
    if (it == m_geodeClip.end())
        return;
    GeodeClipSave save = it->second;
    m_geodeClip.erase(it);

    osg::StateSet *ss = geode->getOrCreateStateSet();
    ss->removeAttribute(osg::StateAttribute::PROGRAM);
    // remove the geode-level texture binding we added for the shader (the
    // original drawable-level binding, if any, is left untouched)
    if (save.texture)
        ss->removeTextureAttribute(save.texUnit, osg::StateAttribute::TEXTURE);

    // restore what we saved
    if (save.material)
        ss->setAttributeAndModes(save.material, osg::StateAttribute::ON);
    if (save.hadTextureAttribute && save.texture)
        ss->setTextureAttributeAndModes(save.texUnit, save.texture, osg::StateAttribute::ON);
    if (save.hadDepth)
    {
        ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
        if (save.depth)
            ss->setAttributeAndModes(save.depth, osg::StateAttribute::ON);
    }
}

// ---------------------------------------------------------------------------
// targets
// ---------------------------------------------------------------------------

void ClipPlugin::addTarget(const std::string &name)
{
    if (name.empty() || m_targetIndex.count(name))
        return;

    // the node that is actually clipped: the top-level object itself when the
    // name matches it, or a named sub-node (a photogrammetry tile, an OBJ
    // material group) when the name points inside an object
    osg::Node *top = topNodeOf(name);
    osg::Node *clip = top;
    if (!top || top->getName() != name)
    {
        osg::Geode *geode = findGeodeForName(name);
        if (!geode)
        {
            m_statusLabel->setText(("No such object: " + name).c_str());
            return;
        }
        top = topObjectOf(geode);
        if (!top)
        {
            return;
        }
        clip = geode;
    }

    Target t;
    t.name = name;
    t.topNode = top;
    t.clipNode = clip;
    {
        GeodeAllVisitor gv(t.geodes);
        clip->accept(gv);
    }
    if (t.geodes.empty())
    {
        m_statusLabel->setText(("No geodes in: " + name).c_str());
        return;
    }

    size_t index = m_targets.size();
    m_targets.push_back(t);
    m_targetIndex[name] = index;

    // apply the clip shader to all (currently loaded) geodes
    for (osg::Geode *g : m_targets.back().geodes)
        applyClipToGeode(g);

    m_manual = true;
    updateUi();
    if (cover->debugLevel(3))
        std::fprintf(stderr, "CLIP: target = %s (%zu geodes)\n", name.c_str(), m_targets.back().geodes.size());
}

void ClipPlugin::removeTarget(const std::string &name)
{
    auto it = m_targetIndex.find(name);
    if (it == m_targetIndex.end())
        return;

    Target &t = m_targets[it->second];
    for (osg::Geode *g : t.geodes)
        removeClipFromGeode(g);
    m_targets.erase(m_targets.begin() + it->second);
    m_targetIndex.erase(it);

    // renumber the index map
    for (size_t i = 0; i < m_targets.size(); ++i)
        m_targetIndex[m_targets[i].name] = i;

    m_manual = true;
    updateUi();
    if (cover->debugLevel(3))
        std::fprintf(stderr, "CLIP: removed target = %s\n", name.c_str());
}

// keep one target's geode list in sync with the live scene (paged tiles are
// added/expired by the DatabasePager while the plugin runs). Returns true if
// anything changed.
bool ClipPlugin::reconcileTargetGeodes(Target &target)
{
    // re-resolve the nodes: the cached ones may have been wrapped by the Move
    // plugin or replaced on reload
    osg::Node *top = target.topNode.get();
    osg::Node *liveTop = top ? topObjectOf(top) : nullptr;
    if (liveTop)
        target.topNode = liveTop;
    else
        top = target.topNode = topNodeOf(target.name);
    if (!top)
        return false;

    osg::Node *clip = target.clipNode.get();
    // the cached clip node must still belong to the same top-level object
    if (!clip || topObjectOf(clip) != top)
        clip = target.clipNode = (target.topNode->getName() == target.name) ? target.topNode.get() : findClipNode(target.name);
    if (!clip)
        return false;

    std::vector<osg::Geode *> liveGeodes;
    GeodeAllVisitor gv(liveGeodes);
    clip->accept(gv);

    // apply the clip to new geodes
    for (osg::Geode *g : liveGeodes)
    {
        if (!m_geodeClip.count(g))
        {
            applyClipToGeode(g);
            target.geodes.push_back(g);
        }
    }
    // restore geodes that are gone (expired pages, removed drawables)
    for (size_t i = target.geodes.size(); i-- > 0;)
    {
        bool alive = std::find(liveGeodes.begin(), liveGeodes.end(), target.geodes[i]) != liveGeodes.end();
        if (!alive)
        {
            removeClipFromGeode(target.geodes[i]);
            target.geodes.erase(target.geodes.begin() + i);
        }
    }
    return liveGeodes.size() != target.geodes.size();
}

void ClipPlugin::reconcileTargets()
{
    for (Target &t : m_targets)
        reconcileTargetGeodes(t);
}

// ---------------------------------------------------------------------------
// boxes
// ---------------------------------------------------------------------------

void ClipPlugin::boxFromObject(osg::Geode *geode, const std::string &name)
{
    (void)name;
    if (!geode)
        return;

    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return;

    osg::Vec3 min, max;
    geodeWorldAABB(geode, min, max);

    // the box sits in the same space as the objects: create it at the objects
    // root with a transform that maps the unit cube onto the picked AABB
    auto box = new ClipBoxNode(nextBoxName());
    osg::Matrix m = osg::Matrix::scale((max - min)) * osg::Matrix::translate((min + max) / 2.0f);
    box->setMatrix(m);
    box->setSize(max - min);
    root->addChild(box);

    m_boxes.push_back(box);
    m_manual = true;
    updateUi();
    if (cover->debugLevel(3))
        std::fprintf(stderr, "CLIP: box from object %s: min=(%g,%g,%g) max=(%g,%g,%g)\n", name.c_str(), min.x(), min.y(), min.z(), max.x(), max.y(), max.z());
}

void ClipPlugin::addBoxAt(const osg::Vec3 &point)
{
    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return;

    auto box = new ClipBoxNode(nextBoxName());
    osg::Vec3 size(CLIP_DEFAULT_SIZE[0], CLIP_DEFAULT_SIZE[1], CLIP_DEFAULT_SIZE[2]);
    // bottom-anchored: the point is the box's lower corner
    osg::Matrix m = osg::Matrix::scale(size) * osg::Matrix::translate(osg::Vec3(point.x() + size.x() * 0.5f, point.y() + size.y() * 0.5f, point.z() + size.z() * 0.5f));
    box->setMatrix(m);
    box->setSize(size);
    root->addChild(box);

    m_boxes.push_back(box);
    m_manual = true;
    updateUi();
    if (cover->debugLevel(3))
        std::fprintf(stderr, "CLIP: box at (%g,%g,%g) (default size (%g,%g,%g))\n", point.x(), point.y(), point.z(), size.x(), size.y(), size.z());
}

void ClipPlugin::removeLastBox()
{
    if (m_boxes.empty())
        return;
    auto box = m_boxes.back();
    if (box->getNumParents() > 0)
        box->getParent(0)->removeChild(box.get());
    m_boxes.pop_back();
    m_manual = true;
    updateUi();
}

void ClipPlugin::setBoxesVisible(bool visible)
{
    if (m_showBoxes == visible)
        return;
    m_showBoxes = visible;
    if (m_showBoxesBtn)
        m_showBoxesBtn->setState(visible);
    if (m_cfgShowBoxes)
        *m_cfgShowBoxes = visible;
    for (const auto &box : m_boxes)
        box->setNodeMask(visible ? (Isect::Visible | Isect::Pick) : 0);
}

void ClipPlugin::clearAll()
{
    // remove all boxes
    while (!m_boxes.empty())
    {
        auto box = m_boxes.back();
        if (box->getNumParents() > 0)
            box->getParent(0)->removeChild(box.get());
        m_boxes.pop_back();
    }
    // remove all targets (restores their geodes); copy the names first,
    // removeTarget() erases from m_targetIndex
    std::vector<std::string> names;
    names.reserve(m_targetIndex.size());
    for (const auto &kv : m_targetIndex)
        names.push_back(kv.first);
    for (const auto &n : names)
        removeTarget(n);
    m_manual = false;
    updateUi();
}

// ---------------------------------------------------------------------------
// per-frame sync
// ---------------------------------------------------------------------------

void ClipPlugin::preFrame()
{

    // keep the box list in sync with the scene (boxes may have been removed
    // by the user / the Move plugin may have wrapped them)
    std::vector<std::string> boxNames;
    for (const auto &box : m_boxes)
        if (box && box->getNumParents() > 0)
            boxNames.push_back(box->getName());
    for (size_t i = m_boxes.size(); i-- > 0;)
    {
        if (!m_boxes[i] || m_boxes[i]->getNumParents() == 0)
            m_boxes.erase(m_boxes.begin() + i);
    }
    // newly added boxes (e.g. restored from a reload) are picked up by name
    {
        osg::Group *root = cover->getObjectsRoot();
    if (root)
    {
        for (unsigned int i = 0; i < root->getNumChildren(); ++i)
        {
            auto *inner = dynamic_cast<ClipBoxNode *>(innerObjectOf(root->getChild(i)));
            if (inner && std::find(boxNames.begin(), boxNames.end(), inner->getName()) == boxNames.end())
            {
                m_boxes.push_back(inner);
                boxNames.push_back(inner->getName());
            }
        }
    }
    }

    // keep the box ghosts' visibility in sync (a reload may re-create them)
    for (const auto &box : m_boxes)
    {
        unsigned int mask = box->getNodeMask();
        bool visible = (mask & Isect::Visible) != 0;
        if (visible != m_showBoxes)
            box->setNodeMask(m_showBoxes ? (Isect::Visible | Isect::Pick) : 0);
    }

    // keep the geode lists of paged targets in sync, then push the uniforms
    reconcileTargets();
    syncClipUniforms();
}

void ClipPlugin::syncClipUniforms()
{
    if (m_geodeClip.empty())
        return;

    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return;


    // ---- shared uniforms: the box AABBs (root space) + the scene light ----
    int n = 0;
    for (const auto &box : m_boxes)
    {
        if (n >= CLIP_MAX_BOXES)
            break;
        osg::Vec3 min, max;
        box->worldAABB(root, min, max);
        m_boxMin->setElement(n, min);
        m_boxMax->setElement(n, max);
        ++n;
    }
    if (n < CLIP_MAX_BOXES)
    {
        // pad with a degenerate box that never cuts
        osg::Vec3 none(0.0f, 0.0f, 0.0f);
        for (int i = n; i < CLIP_MAX_BOXES; ++i)
        {
            m_boxMin->setElement(i, none);
            m_boxMax->setElement(i, none);
        }
    }
    m_boxCount->setElement(0, n);



    // the scene headlight ("Sun"): a directional light
    {
        auto *lighting = coVRLighting::instance();
        if (lighting && lighting->headlight)
        {
            osg::Light *l = lighting->headlight->getLight();
            if (l)
            {
                const osg::Vec4 &p = l->getPosition();
                const osg::Vec4 &d = l->getDiffuse();
                const osg::Vec4 &a = l->getAmbient();
                m_lightPos->set(p);
                m_lightDiffuse->set(osg::Vec3(d.x(), d.y(), d.z()));
                m_lightAmbient->set(osg::Vec3(a.x(), a.y(), a.z()));
            }
        }
    }

    // ---- per-geode uniform: this geode's local-to-root matrix ----
    for (auto &kv : m_geodeClip)
    {
        osg::Geode *g = kv.first;
        if (!g)
            continue;
        osg::NodePathList paths = g->getParentalNodePaths(root);
        if (paths.empty())
            continue;
        kv.second.worldMatrix->set(osg::Matrixf(localToWorldBelowRoot(paths.front())));
    }

}

// ---------------------------------------------------------------------------
// setup (config)
// ---------------------------------------------------------------------------

void ClipPlugin::loadSetup()
{
    if (!m_cfgTargets)
        return;

    // a (re)loaded setup means the config is in charge again
    m_manual = false;
    syncSetup();
}

void ClipPlugin::syncSetup()
{
    if (m_manual || !m_cfgTargets)
        return;

    // apply every configured target that is not active yet (by name), and
    // drop targets that are no longer configured
    const auto &cfgTargets = m_cfgTargets->value();

    for (const auto &t : m_targets)
    {
        if (std::find(cfgTargets.begin(), cfgTargets.end(), t.name) == cfgTargets.end())
            removeTarget(t.name);
    }
    for (const std::string &name : cfgTargets)
    {
        if (name.empty() || m_targetIndex.count(name))
            continue;
        addTarget(name);
    }

    // the configured showBoxes flag
    if (m_cfgShowBoxes)
        setBoxesVisible(m_cfgShowBoxes->value());
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

void ClipPlugin::updateUi()
{
    if (!m_statusLabel)
        return;

    std::string text = "Targets: ";
    if (m_targets.empty())
        text += "(none)";
    else
    {
        for (size_t i = 0; i < m_targets.size(); ++i)
            text += (i ? ", " : "") + m_targets[i].name;
    }
    text += " | Boxes: " + std::to_string(m_boxes.size());
    m_statusLabel->setText(text);
}

// ---------------------------------------------------------------------------
// coSelectionListener + scene change hooks
// ---------------------------------------------------------------------------

bool ClipPlugin::selectionChanged()
{
    // only act while a Clip pick mode is active; otherwise leave the
    // Selection/Move plugins' selection untouched (boxes are ordinary
    // selectable objects, so the user can also pick them with the Selection
    // plugin and move them with the Move plugin)
    if (m_mode == Mode::None)
    {
        // highlight the selected box(es), if any
        auto selection = coVRSelectionManager::instance()->getSelectionList();
        for (const auto &box : m_boxes)
        {
            bool sel = false;
            for (const auto &s : selection)
                if (s == box.get())
                    sel = true;
            box->setSelected(sel);
        }
        return true;
    }

    auto selection = coVRSelectionManager::instance()->getSelectionList();
    if (selection.empty())
    {
        // deselected while a pick mode was active: keep the mode, wait for
        // the next selection
        return true;
    }

    osg::Node *node = selection.back().get();
    auto *geode = dynamic_cast<osg::Geode *>(node);
    if (!geode && node)
    {
        // the selection may have grabbed a wrapper; find the first geode inside
        std::vector<osg::Geode *> found;
        GeodeAllVisitor visitor(found);
        node->accept(visitor);
        if (!found.empty())
            geode = found[0];
    }
    if (!geode)
        return true;

    // ignore our own box nodes as pick targets
    if (dynamic_cast<ClipBoxNode *>(topObjectOf(geode)))
        return true;

    std::string name = displayNameOf(geode);
    if (name.empty())
        return true;

    switch (m_mode)
    {
    case Mode::PickTarget:
        addTarget(name);
        break;
    case Mode::BoxFromObject:
        boxFromObject(geode, name);
        break;
    default:
        break;
    }

    m_manual = true;
    updateUi();
    return true;
}

bool ClipPlugin::pickedObjChanged()
{
    return true;
}

void ClipPlugin::addNodeFromPlugin(osg::Node *node, const RenderObject *ro, coVRPlugin *addingPlugin)
{
    (void)node;
    (void)ro;
    (void)addingPlugin;
    // a new COVISE object may match the configured setup; syncSetup()
    // re-resolves the objects by name
    syncSetup();
}

void ClipPlugin::removeNode(osg::Node *node, bool isGroup, osg::Node *realNode)
{
    (void)isGroup;
    (void)realNode;
    (void)node;
    // a removed geode is dropped by reconcileTargets() on the next frame
    // (it restores the clip shader state of expired pages)
}

void ClipPlugin::removeObject(const char *objName, bool replaceFlag)
{
    (void)replaceFlag;
    if (!objName)
        return;

    // the config names the object, but the scene top-level node may be named
    // differently (file objects carry their path); resolve before comparing
    const std::string name = objName;
    for (const auto &t : m_targets)
    {
        osg::Node *obj = topNodeOf(name);
        if (!obj)
            continue;
        if (t.topNode.get() == obj || t.name == name ||
            (obj->getName() == t.name) ||
            matchName(t.name, name) || matchName(name, t.name))
            removeTarget(t.name);
    }
}

COVERPLUGIN(ClipPlugin)
