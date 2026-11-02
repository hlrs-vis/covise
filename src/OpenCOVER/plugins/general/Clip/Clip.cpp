#include "Clip.h"

#include <cover/coVRPluginSupport.h>
#include <cover/ui/Button.h>
#include <cover/ui/Group.h>
#include <cover/ui/Label.h>
#include <cover/ui/Menu.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/View.h>

#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/NodeVisitor>
#include <osg/ColorMask>
#include <osg/CullFace>
#include <osg/Depth>
#include <osg/Stencil>
#include <osg/StateSet>
#include <osg/DisplaySettings>
#include <cover/coVRConfig.h>
#include <GL/gl.h>

#include <algorithm>
#include <iostream>

using namespace opencover;

namespace
{

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

private:
    std::vector<osg::Geode *> &m_result;
};

// masks stripped from a hidden void: no collision/walk/touch, so lasers,
// navigation and walking pass through the invisible object; Pick stays set so
// Selection/Move can still grab it
const unsigned int HIDDEN_STRIP_MASK = Isect::Collision | Isect::Walk | Isect::Touch;

// shadow clones never participate in picking, collisions, walking, updates or
// the selection update of the object
void setShadowFlags(osg::Node *node)
{
    node->setNodeMask(node->getNodeMask() & ~(Isect::Collision | Isect::Intersection | Isect::Pick | Isect::Update | Isect::Walk | Isect::Touch));
}

// renders only into the depth buffer: a clone of the target written before the
// void passes, so the voids can be depth-tested against the target and only the
// part of a void that is actually in front of (or inside) the target writes to
// the stencil. Color off, no stencil, depth write on.
void prepareTargetShadow(osg::Node *shadow)
{
    osg::StateSet *ss = shadow->getOrCreateStateSet();
    ss->setRenderBinDetails(-3, "RenderBin");
    ss->setNestRenderBins(false);
    ss->setAttributeAndModes(new osg::ColorMask(false, false, false, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
}

// shared state of both void passes: nothing is visible (color/depth write
// off), the geometry is depth-tested against the target shadow so only the
// part of the void in front of (or inside) the target writes to the stencil.
void prepareVoidShadowCommon(osg::Node *shadow, int bin, unsigned int stencilBit)
{
    osg::StateSet *ss = shadow->getOrCreateStateSet();
    ss->setRenderBinDetails(bin, "RenderBin");
    ss->setNestRenderBins(false);
    ss->setAttributeAndModes(new osg::ColorMask(false, false, false, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

    // only the fragments that pass the depth test against the target shadow
    // (i.e. are in front of the target) write to the stencil. REPLACE writes
    // ref & writeMask, so ref == the bit we want to set leaves the other bit
    // (from the sibling pass) untouched
    osg::Stencil *stencil = new osg::Stencil();
    stencil->setFunction(osg::Stencil::ALWAYS, (int)stencilBit, 0);
    stencil->setOperation(osg::Stencil::KEEP, osg::Stencil::KEEP, osg::Stencil::REPLACE);
    stencil->setWriteMask(stencilBit);
    ss->setAttributeAndModes(stencil, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

    // test against the target shadow but do not write depth, so the invisible
    // clones cannot cull the real object in the main pass
    ss->setAttributeAndModes(new osg::Depth(osg::Depth::LESS, 0.0, 1.0, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

    ss->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    ss->setMode(GL_CULL_FACE, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
}

// the near-surface (front faces) of the void: culls back faces, i.e. renders
// only the front shell of the object, and writes stencil bit 0 where that
// shell is in front of the target (depth shadow)
void prepareVoidShadowNear(osg::Node *shadow)
{
    prepareVoidShadowCommon(shadow, -2, 0x1);
    shadow->getOrCreateStateSet()->setAttributeAndModes(new osg::CullFace(osg::CullFace::BACK), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
}

// the far-surface (back faces) of the void: culls front faces, i.e. renders
// only the back shell of the object, and writes stencil bit 1 where that
// shell is in front of the target (depth shadow)
void prepareVoidShadowFar(osg::Node *shadow)
{
    prepareVoidShadowCommon(shadow, -1, 0x2);
    shadow->getOrCreateStateSet()->setAttributeAndModes(new osg::CullFace(osg::CullFace::FRONT), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
}

// the helper node type encoded in the node's "TYPE <n>" description
// (coVRSelectionManager::getHelperType() is private, so replicate it)
int helperTypeOf(osg::Node *node)
{
    for (unsigned int j = 0; node && j < node->getNumDescriptions(); ++j)
        if (strncmp(node->getDescription(j).c_str(), "TYPE ", 5) == 0)
            return atoi(node->getDescription(j).c_str() + 5);
    return 100;
}

// true if a helper node of `helperType` (SELECTION => currently selected / being
// moved; MOVE => already moved once) sits between `node` and the objects root.
// The Move helper is permanent, so only the SELECTION helper marks active
// editing. Helpers nest in type order (MOVE closest to the geode, SELECTION
// above it), so the whole chain has to be walked.
bool hasHelperAncestor(osg::Node *node, int helperType = -1)
{
    osg::Group *root = cover->getObjectsRoot();
    osg::Group *g = node ? node->asGroup() : nullptr;
    while (g && g != root)
    {
        if (coVRSelectionManager::instance()->isHelperNode(g) &&
            (helperType < 0 || helperTypeOf(g) == helperType))
            return true;
        osg::Group *p = g->getNumParents() > 0 ? g->getParent(0)->asGroup() : nullptr;
        if (!p)
            break;
        g = p;
    }
    return false;
}

} // namespace

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
    m_menu = new ui::Menu("ClipMenu", this);
    m_menu->setText("Clip objects");

    m_group = new ui::Group(m_menu, "ClipObjects");
    m_group->setText("Cut holes with objects");

    m_targetBtn = new ui::Button(m_group, "PickTarget");
    m_targetBtn->setText("Pick target (e.g. terrain)");
    m_targetBtn->setCallback([this](bool state) {
        if (state)
            setMode(Mode::PickTarget);
    });

    m_addVoidBtn = new ui::Button(m_group, "AddVoid");
    m_addVoidBtn->setText("Add void object");
    m_addVoidBtn->setCallback([this](bool state) {
        if (state)
            setMode(Mode::AddVoid);
    });

    m_removeVoidBtn = new ui::Button(m_group, "RemoveVoid");
    m_removeVoidBtn->setText("Remove void object");
    m_removeVoidBtn->setCallback([this](bool state) {
        if (state)
            setMode(Mode::RemoveVoid);
    });

    m_hideVoidsBtn = new ui::Button(m_group, "HideVoids");
    m_hideVoidsBtn->setText("Hide void objects");
    m_hideVoidsBtn->setState(m_hideVoids);
    m_hideVoidsBtn->setCallback([this](bool state) {
        if (state != m_hideVoids)
            setHideVoids(state);
    });

    m_clearBtn = new ui::Button(m_group, "Clear");
    m_clearBtn->setText("Clear all");
    m_clearBtn->setCallback([this](bool) {
        clearAll();
    });

    m_targetLabel = new ui::Label(m_group, "Target");
    m_targetLabel->setText("Target: (none)");

    m_voidsLabel = new ui::Label(m_group, "Voids");
    m_voidsLabel->setText("Voids: (none)");

    m_voidsList = new ui::SelectionList(m_group, "VoidList");
    m_voidsList->setText("Select void (removes it)");
    m_voidsList->setVisible(false, ui::View::VR);
    m_voidsList->setCallback([this](int index) {
        if (index >= 0 && index < (int)m_voidNames.size())
            removeVoid(m_voidNames[index]);
    });

    auto *loadSetupBtn = new ui::Button(m_group, "LoadSetup");
    loadSetupBtn->setText("Apply setup from config");
    loadSetupBtn->setCallback([this](bool) {
        loadSetup();
    });

    // setup can also be (re)applied from the toml config file
    m_cfgTarget = configString("Setup", "target", "", config::Flag::PerModel);
    m_cfgTarget->setUpdater([this](const std::string &) {
        loadSetup();
    });
    m_cfgVoids = configStringArray("Setup", "voids", {}, config::Flag::PerModel);
    m_cfgVoids->setUpdater([this]() {
        loadSetup();
    });
    m_cfgHide = configBool("Setup", "hideVoids", true, config::Flag::PerModel);
    m_hideVoids = m_cfgHide->value();
    m_hideVoidsBtn->setState(m_hideVoids);
    m_cfgHide->setUpdater([this](const bool &v) {
        setHideVoids(v);
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

std::string ClipPlugin::objectNameOf(osg::Node *node) const
{
    osg::Node *top = topNodeOf(node);
    return top ? top->getName() : "";
}

osg::Node *ClipPlugin::topNodeOf(osg::Node *node) const
{
    if (!node)
        return nullptr;

    osg::Group *root = cover->getObjectsRoot();
    osg::Node *n = node;
    while (n)
    {
        // never climb into a shadow clone (it is a sibling of the object, so we
        // must not mistake it for the object when started from inside it)
        if (n->getName().find("_ClipShadow") != std::string::npos)
            return nullptr;
        // skip selection helper nodes (they are inserted between the object and
        // its parent at edit time)
        if (coVRSelectionManager::instance()->isHelperNode(n))
        {
            n = n->getNumParents() > 0 ? n->getParent(0) : nullptr;
            continue;
        }
        // a direct child of the objects root is a top-level object, whatever its
        // type (a file object can be a bare Geode or a Group)
        if (root && n->getNumParents() == 1 && n->getParent(0) == root)
            return n;
        n = n->getNumParents() > 0 ? n->getParent(0) : nullptr;
    }
    return node;
}

bool ClipPlugin::matchName(const std::string &cfgName, const std::string &nodeName)
{
    if (cfgName.empty() || nodeName.empty())
        return false;
    if (cfgName == nodeName)
        return true;

    // file-loaded objects are named by their (url) path: "terrain" or "terrain.obj"
    // should match "/some/path/terrain.obj"
    const std::string base = cfgName.substr(cfgName.find_last_of("/\\") + 1);
    if (base == nodeName)
        return true;
    if (base.size() > 4 && base.size() <= nodeName.size() &&
        nodeName.compare(nodeName.size() - base.size(), base.size(), base) == 0)
        return true;

    // allow a plain prefix as well ("build" -> "building1.obj")
    return cfgName.size() >= 3 && nodeName.find(cfgName) == 0;
}

std::string ClipPlugin::resolveName(const std::string &name) const
{
    if (name.empty())
        return "";

    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return "";

    std::string prefixMatch;
    for (unsigned int i = 0; i < root->getNumChildren(); ++i)
    {
        osg::Node *child = root->getChild(i);
        if (!child)
            continue;
        // never match our own shadow clones (they are siblings of the objects)
        if (child->getName().find("_ClipShadow") != std::string::npos)
            continue;
        const std::string n = child->getName();
        if (matchName(name, n))
            return n;
        if (prefixMatch.empty() && name.size() >= 3 && n.find(name) == 0)
            prefixMatch = n;
    }
    return prefixMatch;
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
        // never match our own shadow clones (they are siblings of the objects)
        if (child->getName().find("_ClipShadow") != std::string::npos)
            continue;
        const std::string n = child->getName();
        if (n == name)
        {
            top = child;
            break;
        }
        if (!top && matchName(name, n))
            top = child;
    }
    return top;
}

std::vector<osg::Geode *> ClipPlugin::geodesOf(const std::string &name) const
{
    std::vector<osg::Geode *> result;
    osg::Node *top = topNodeOf(name);
    if (!top)
        return result;
    GeodeAllVisitor visitor(result);
    top->accept(visitor);
    return result;
}

void ClipPlugin::setMode(Mode m)
{
    if (m_mode == m)
        m = Mode::None;

    m_mode = m;
    m_targetBtn->setState(m_mode == Mode::PickTarget);
    m_addVoidBtn->setState(m_mode == Mode::AddVoid);
    m_removeVoidBtn->setState(m_mode == Mode::RemoveVoid);
}

void ClipPlugin::pickTarget(osg::Geode *geode, const std::string &name)
{
    if (!geode || name.empty())
        return;
    if (m_targetTopNode.get() == topNodeOf(name))
        return;

    if (cover->debugLevel(3))
        std::cerr << "Clip: target = " << name << " (" << geodesOf(name).size() << " geodes)\n";
    // reset the previous target's rendering before switching
    restoreTarget();

    m_targetName = name;
    m_targetTopNode = topNodeOf(name);
    m_targetGeodes = geodesOf(name);

    // save each geode's original depth/stencil so restoreTarget() can bring it
    // back exactly (e.g. an object may already carry its own attributes)
    m_targetSaved.clear();
    for (osg::Geode *g : m_targetGeodes)
    {
        GeodeSavedState st;
        st.geode = g;
        osg::StateSet *ss = g->getOrCreateStateSet();
        st.depth = ss->getAttribute(osg::StateAttribute::DEPTH);
        st.stencil = ss->getAttribute(osg::StateAttribute::STENCIL);
        m_targetSaved.push_back(st);
    }
    m_targetSavedState = true;

    rebuildTargetShadow();
    updateMainPass();
    refreshStatus();
}

void ClipPlugin::restoreTarget()
{
    if (m_targetSavedState)
    {
        for (auto &st : m_targetSaved)
        {
            if (!st.geode)
                continue;
            osg::StateSet *ss = st.geode->getOrCreateStateSet();
            if (st.depth)
                ss->setAttributeAndModes(st.depth, osg::StateAttribute::ON);
            else
                ss->removeAttribute(osg::StateAttribute::DEPTH);
            if (st.stencil)
                ss->setAttributeAndModes(st.stencil, osg::StateAttribute::ON);
            else
                ss->removeAttribute(osg::StateAttribute::STENCIL);
        }
    }
    m_targetSavedState = false;
    m_targetSaved.clear();
}

void ClipPlugin::pickVoid(osg::Geode *geode, const std::string &name)
{
    if (!geode || name.empty() || name == m_targetName || std::find(m_voidNames.begin(), m_voidNames.end(), name) != m_voidNames.end())
        return;

    if (cover->debugLevel(3))
        std::cerr << "Clip: void = " << name << " (" << geodesOf(name).size() << " geodes)\n";
    m_voidNames.push_back(name);
    createVoidShadow(name, geode);
    applyVoidVisibility(name);
    updateVoidsUi();
    updateMainPass();
}

void ClipPlugin::removeVoid(const std::string &name)
{
    auto it = std::find(m_voidNames.begin(), m_voidNames.end(), name);
    if (it == m_voidNames.end())
        return;

    m_voidNames.erase(it);
    removeVoidShadow(name);
    updateVoidsUi();
    updateMainPass();
}

void ClipPlugin::clearAll()
{
    // restore the target geodes and drop the shadows
    for (const auto &name : m_voidNames)
        removeVoidShadow(name);
    restoreTarget();
    removeTargetShadow();
    m_targetName.clear();
    m_targetTopNode = nullptr;
    m_targetGeodes.clear();
    m_voidNames.clear();
    m_manual = true; // don't let the config re-apply right after a manual clear
    updateVoidsUi();
    updateMainPass();
    refreshStatus();
}

void ClipPlugin::loadSetup()
{
    if (!m_cfgTarget || !m_cfgVoids)
        return;

    // a (re)loaded setup means the config is in charge again
    m_manual = false;
    syncSetup();
}

void ClipPlugin::syncSetup()
{
    if (m_manual || !m_cfgTarget || !m_cfgVoids)
        return;

    const std::string targetPrefix = m_cfgTarget->value();

    // target
    if (!targetPrefix.empty())
    {
        const std::string name = resolveName(targetPrefix);
        if (!name.empty() && m_targetName != name)
        {
            auto it = std::find(m_voidNames.begin(), m_voidNames.end(), name);
            if (it != m_voidNames.end())
            {
                m_voidNames.erase(it);
                removeVoidShadow(name);
            }
            std::vector<osg::Geode *> geodes = geodesOf(name);
            if (!geodes.empty())
                pickTarget(geodes[0], name);
        }
    }
    // voids
    for (const auto &prefix : m_cfgVoids->value())
    {
        if (prefix.empty())
            continue;
        const std::string name = resolveName(prefix);
        if (name.empty() || name == m_targetName)
            continue;
        if (std::find(m_voidNames.begin(), m_voidNames.end(), name) != m_voidNames.end())
            continue;
        std::vector<osg::Geode *> geodes = geodesOf(name);
        if (!geodes.empty())
            pickVoid(geodes[0], name);
    }
}

void ClipPlugin::setHideVoids(bool hide)
{
    if (m_hideVoids == hide)
        return;
    m_hideVoids = hide;
    if (m_hideVoidsBtn)
        m_hideVoidsBtn->setState(hide);
    if (m_cfgHide)
        *m_cfgHide = hide;
    for (const auto &name : m_voidNames)
        applyVoidVisibility(name);
}

void ClipPlugin::applyVoidVisibility(const std::string &name)
{
    auto it = m_voidShadows.find(name);
    if (it == m_voidShadows.end())
        return;
    Shadow *shadow = it->second;
    if (shadow->geodes.empty() || !shadow->topNode.get())
        return;

    // a void that is currently selected / being moved (a SELECTION helper node
    // sits above it) stays visible, so it can be edited in the scene.
    // The permanent MOVE helper is ignored: it only means the object was moved.
    const bool hidden = m_hideVoids && !hasHelperAncestor(shadow->topNode.get(), coVRSelectionManager::SELECTION);
    if (shadow->appliedHidden == hidden)
        return;
    shadow->appliedHidden = hidden;

    if (hidden)
    {
        // hide every geode: write no color, no depth, ignore stencil. Lasers /
        // walking pass through (Collision/Walk/Touch removed from the mask),
        // but Pick stays set so Selection/Move can still grab the object. The
        // selection outline makes it visible while it is being edited.
        for (size_t i = 0; i < shadow->geodes.size(); ++i)
        {
            osg::Geode *geode = shadow->geodes[i];
            if (!geode)
                continue;

            // capture the original state first so "unhide" restores it exactly
            // (must happen before the hide attributes below are applied; the
            // capture is per-geode, so a geode added later is captured when it
            // is hidden the first time)
            if (i < shadow->saved.size() && !shadow->saved[i].captured)
            {
                shadow->saved[i].geode = geode;
                shadow->saved[i].colorMask = geode->getStateSet() ? geode->getStateSet()->getAttribute(osg::StateAttribute::COLORMASK) : nullptr;
                shadow->saved[i].depth = geode->getStateSet() ? geode->getStateSet()->getAttribute(osg::StateAttribute::DEPTH) : nullptr;
                shadow->saved[i].stencil = geode->getStateSet() ? geode->getStateSet()->getAttribute(osg::StateAttribute::STENCIL) : nullptr;
                shadow->saved[i].nodeMask = geode->getNodeMask();
                shadow->saved[i].captured = true;
            }

            osg::StateSet *ss = geode->getOrCreateStateSet();
            ss->setAttributeAndModes(new osg::ColorMask(false, false, false, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            ss->setAttributeAndModes(new osg::Depth(osg::Depth::ALWAYS, 0.0, 1.0, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            osg::Stencil *stencil = new osg::Stencil();
            stencil->setFunction(osg::Stencil::ALWAYS, 0, 0);
            ss->setAttributeAndModes(stencil, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            geode->setNodeMask(geode->getNodeMask() & ~HIDDEN_STRIP_MASK);
        }
    }
    else
    {
        for (size_t i = 0; i < shadow->geodes.size(); ++i)
        {
            if (i >= shadow->saved.size() || !shadow->saved[i].captured)
                continue;
            osg::Geode *geode = shadow->geodes[i];
            if (!geode || geode != shadow->saved[i].geode)
                continue;
            osg::StateSet *ss = geode->getOrCreateStateSet();
            if (shadow->saved[i].colorMask)
                ss->setAttributeAndModes(shadow->saved[i].colorMask, osg::StateAttribute::ON);
            else
                ss->removeAttribute(osg::StateAttribute::COLORMASK);
            if (shadow->saved[i].depth)
                ss->setAttributeAndModes(shadow->saved[i].depth, osg::StateAttribute::ON);
            else
                ss->removeAttribute(osg::StateAttribute::DEPTH);
            if (shadow->saved[i].stencil)
                ss->setAttributeAndModes(shadow->saved[i].stencil, osg::StateAttribute::ON);
            else
                ss->removeAttribute(osg::StateAttribute::STENCIL);
            if (shadow->saved[i].nodeMask != 0)
                geode->setNodeMask(shadow->saved[i].nodeMask);
        }
    }
}

ClipPlugin::Shadow *ClipPlugin::shadowFor(const std::string &name)
{
    auto it = m_voidShadows.find(name);
    return (it != m_voidShadows.end()) ? it->second : nullptr;
}

void ClipPlugin::removeTargetShadow()
{
    if (m_targetShadow)
    {
        if (m_targetShadow->parent.get() && m_targetShadow->node.get())
            m_targetShadow->parent->removeChild(m_targetShadow->node.get());
        delete m_targetShadow;
        m_targetShadow = nullptr;
    }
}

void ClipPlugin::rebuildTargetShadow()
{
    removeTargetShadow();
    osg::Node *top = m_targetTopNode.get();
    if (!top || top->getNumParents() == 0)
        return;
    osg::Group *parent = top->getParent(0)->asGroup();
    if (!parent)
        return;

    auto *shadow = new Shadow();
    shadow->name = m_targetName;
    shadow->topNode = top;
    shadow->geodes = m_targetGeodes;
    shadow->parent = parent;

    // a deep copy of the whole top node (a group, or a bare geode if the object
    // is a single geode); it is a sibling of `top`, so it carries top's local
    // matrix and aligns exactly, following every (helper) transform
    shadow->node = top->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
    if (!shadow->node.get())
    {
        delete shadow;
        return;
    }
    shadow->node->setName(m_targetName + "_ClipShadow");
    setShadowFlags(shadow->node.get());
    prepareTargetShadow(shadow->node.get());
    parent->addChild(shadow->node.get());
    m_targetShadow = shadow;
}

void ClipPlugin::removeVoidShadow(const std::string &name)
{
    auto it = m_voidShadows.find(name);
    if (it == m_voidShadows.end())
        return;

    Shadow *shadow = it->second;

    // restore the geodes' original rendering state
    for (size_t i = 0; i < shadow->geodes.size(); ++i)
    {
        if (i >= shadow->saved.size() || !shadow->saved[i].captured || !shadow->saved[i].geode)
            continue;
        osg::Geode *geode = shadow->geodes[i];
        if (!geode || geode != shadow->saved[i].geode)
            continue;
        osg::StateSet *ss = geode->getOrCreateStateSet();
        if (shadow->saved[i].colorMask)
            ss->setAttributeAndModes(shadow->saved[i].colorMask, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::COLORMASK);
        if (shadow->saved[i].depth)
            ss->setAttributeAndModes(shadow->saved[i].depth, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::DEPTH);
        if (shadow->saved[i].stencil)
            ss->setAttributeAndModes(shadow->saved[i].stencil, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::STENCIL);
        if (shadow->saved[i].nodeMask != 0)
            geode->setNodeMask(shadow->saved[i].nodeMask);
    }

    if (shadow->parent.get() && shadow->node.get())
        shadow->parent->removeChild(shadow->node.get());
    if (shadow->parent.get() && shadow->nodeFar.get())
        shadow->parent->removeChild(shadow->nodeFar.get());
    delete shadow;
    m_voidShadows.erase(it);
}

void ClipPlugin::createVoidShadow(const std::string &name, osg::Geode *geode)
{
    (void)geode;
    osg::Node *top = topNodeOf(name);
    if (!top || top->getNumParents() == 0 || geodesOf(name).empty())
        return;

    auto *shadow = new Shadow();
    shadow->name = name;
    shadow->topNode = top;
    shadow->geodes = geodesOf(name);
    shadow->parent = top->getParent(0)->asGroup();
    // pre-size the saved-state bookkeeping (captured on the first hide)
    shadow->saved.resize(shadow->geodes.size());
    for (size_t i = 0; i < shadow->geodes.size(); ++i)
        shadow->saved[i].geode = shadow->geodes[i];
    makeVoidClones(shadow, name);
    m_voidShadows[name] = shadow;
}

void ClipPlugin::makeVoidClones(Shadow *shadow, const std::string &name)
{
    // two deep copies of the whole object, siblings of the top node so they
    // follow every (helper) transform exactly like the real object: one
    // rendered with its front shell (near, bin -2, stencil bit 0) and one with
    // its back shell (far, bin -1, stencil bit 1)
    osg::Group *parent = shadow->parent.get();
    if (!parent)
        return;

    shadow->node = shadow->topNode->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
    if (!shadow->node.get())
        return;
    shadow->node->setName(name + "_ClipShadow");
    setShadowFlags(shadow->node.get());
    prepareVoidShadowNear(shadow->node.get());
    parent->addChild(shadow->node.get());

    shadow->nodeFar = shadow->topNode->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
    if (!shadow->nodeFar.get())
        return;
    shadow->nodeFar->setName(name + "_ClipShadowFar");
    setShadowFlags(shadow->nodeFar.get());
    prepareVoidShadowFar(shadow->nodeFar.get());
    parent->addChild(shadow->nodeFar.get());
}

void ClipPlugin::reattachVoidShadow(const std::string &name)
{
    // the object's top node was replaced (COVISE update): drop the clones and
    // re-create them as siblings of the current top node
    Shadow *shadow = shadowFor(name);
    if (!shadow)
        return;

    if (shadow->parent.get() && shadow->node.get())
        shadow->parent->removeChild(shadow->node.get());
    if (shadow->parent.get() && shadow->nodeFar.get())
        shadow->parent->removeChild(shadow->nodeFar.get());
    shadow->node = nullptr;
    shadow->nodeFar = nullptr;

    shadow->topNode = topNodeOf(name);
    shadow->geodes = geodesOf(name);
    shadow->saved.clear();
    shadow->saved.resize(shadow->geodes.size());
    for (size_t i = 0; i < shadow->geodes.size(); ++i)
        shadow->saved[i].geode = shadow->geodes[i];
    shadow->parent = shadow->topNode.get() ? shadow->topNode->getNumParents() > 0 ? shadow->topNode->getParent(0)->asGroup() : nullptr : nullptr;

    if (shadow->parent.get())
        makeVoidClones(shadow, name);

    // the new geodes have their original (unmodified) attributes, so forget
    // the saved ones and let applyVoidVisibility() capture fresh ones
    shadow->appliedHidden = false;
}

void ClipPlugin::syncShadows()
{
    // Void and target shadows are deep copies of the whole top-level object,
    // placed as siblings of that object. The clone carries the top node's local
    // matrix, so it aligns with the object and follows every transform. When a
    // (selection/move) helper is inserted or removed, the object's parent
    // changes; re-parenting the clones under the object's current parent keeps
    // them on the object. This works for single-geode files as well as for
    // multi-geode objects (e.g. photogrammetry .osgb with many tile geodes).

    // iterate a copy: removing a void (deleted object) changes m_voidNames
    std::vector<std::string> voidNames(m_voidNames.begin(), m_voidNames.end());
    for (const auto &name : voidNames)
    {
        Shadow *shadow = shadowFor(name);
        if (!shadow)
            continue;

        osg::Node *top = topNodeOf(name);
        if (!top || geodesOf(name).empty())
        {
            // object not in the scene (e.g. mid-replace, or deleted via the
            // SGBrowser, which does not fire the plugin remove hooks): keep the
            // last state for a few frames, then clean up
            if (++m_missingFrames[name] < 30)
                continue;
            removeVoid(name);
            continue;
        }
        m_missingFrames.erase(name);

        if (top != shadow->topNode.get())
        {
            // the object's top node was replaced (COVISE object update):
            // re-clone and re-capture the original state of the new geodes
            reattachVoidShadow(name);
            applyVoidVisibility(name);
            continue;
        }

        // the top node's parent may have changed (a helper was inserted or
        // removed) -> move the clones with it
        osg::Group *parent = top->getNumParents() > 0 ? top->getParent(0)->asGroup() : nullptr;
        if (parent && parent != shadow->parent.get())
        {
            if (shadow->parent.get() && shadow->node.get())
                shadow->parent->removeChild(shadow->node.get());
            if (shadow->parent.get() && shadow->nodeFar.get())
                shadow->parent->removeChild(shadow->nodeFar.get());
            shadow->parent = parent;
            if (shadow->node.get())
                parent->addChild(shadow->node.get());
            if (shadow->nodeFar.get())
                parent->addChild(shadow->nodeFar.get());
        }

        // show the void while it is being edited, hide it otherwise
        applyVoidVisibility(name);
    }

    // target: same handling
    if (m_targetName.empty() || !m_targetTopNode.get())
        return;

    osg::Node *top = topNodeOf(m_targetName);
    if (!top || geodesOf(m_targetName).empty())
    {
        // object not in the scene: keep the last state for a few frames, then
        // clean up
        if (++m_targetMissingFrames >= 30)
        {
            removeTargetShadow();
            m_targetName.clear();
            m_targetTopNode = nullptr;
            m_targetGeodes.clear();
            updateMainPass();
            refreshStatus();
        }
        return;
    }
    m_targetMissingFrames = 0;

    if (top != m_targetTopNode.get())
    {
        // the object was replaced by a COVISE update: rebuild the clone and
        // re-apply the main-pass overrides on the new geodes
        restoreTarget();
        m_targetTopNode = top;
        m_targetGeodes = geodesOf(m_targetName);
        m_targetSaved.clear();
        for (osg::Geode *g : m_targetGeodes)
        {
            GeodeSavedState st;
            st.geode = g;
            osg::StateSet *ss = g->getOrCreateStateSet();
            st.depth = ss->getAttribute(osg::StateAttribute::DEPTH);
            st.stencil = ss->getAttribute(osg::StateAttribute::STENCIL);
            m_targetSaved.push_back(st);
        }
        m_targetSavedState = true;
        rebuildTargetShadow();
        updateMainPass();
        return;
    }

    if (m_targetShadow && m_targetShadow->node.get())
    {
        osg::Group *parent = top->getNumParents() > 0 ? top->getParent(0)->asGroup() : nullptr;
        if (parent && parent != m_targetShadow->parent.get())
        {
            if (m_targetShadow->parent.get())
                m_targetShadow->parent->removeChild(m_targetShadow->node.get());
            m_targetShadow->parent = parent;
            m_targetShadow->parent->addChild(m_targetShadow->node.get());
        }
    }
}

void ClipPlugin::preFrame()
{
    syncSetup();
    syncShadows();

    // TEMPORARY debug dump (CLIP_DEBUG_DUMP=file): writes state once per second
    {
        static int dbgFrames = 0;
        static const char *dbgFile = getenv("CLIP_DEBUG_DUMP");
        static bool printedOnce = false;
        if (dbgFile && ++dbgFrames % 30 == 0)
        {
            std::FILE *f = fopen(dbgFile, "w");
            if (f)
            {
                fprintf(f, "== frame %d ==\n", dbgFrames);
                fprintf(f, "cfg target='%s' cfg voids:", m_cfgTarget ? m_cfgTarget->value().c_str() : "(null)");
                if (m_cfgVoids)
                    for (const auto &v : m_cfgVoids->value())
                        fprintf(f, " [%s]", v.c_str());
                fprintf(f, "\n");
                fprintf(f, "m_targetName='%s' m_manual=%d m_hideVoids=%d\n", m_targetName.c_str(), (int)m_manual, (int)m_hideVoids);
                fprintf(f, "voidNames: ");
                for (const auto &v : m_voidNames)
                    fprintf(f, "[%s] ", v.c_str());
                fprintf(f, "\n");

                osg::Group *root = cover->getObjectsRoot();
                fprintf(f, "objectsRoot top-level children:");
                if (root)
                    for (unsigned i = 0; i < root->getNumChildren(); ++i)
                        fprintf(f, " [%s]", root->getChild(i)->getName().c_str());
                fprintf(f, "\n");

                fprintf(f, "GL context stencil bits: ");
                {
                    osg::DisplaySettings *ds = osg::DisplaySettings::instance().get();
                    fprintf(f, "minStencilBits=%u windowBits=", ds->getMinimumNumStencilBits());
                    // actual context bits
                    bool haveCtx = false;
                    if (coVRConfig::instance() && !coVRConfig::instance()->windows.empty())
                    {
                        if (coVRConfig::instance()->windows[0].context.valid())
                        {
                            osg::GraphicsContext *gc = coVRConfig::instance()->windows[0].context.get();
                            gc->makeCurrent();
                            GLint bits = 0;
                            glGetIntegerv(GL_STENCIL_BITS, &bits);
                            fprintf(f, "%d", (int)bits);
                            haveCtx = true;
                        }
                    }
                    if (!haveCtx)
                        fprintf(f, "n/a");
                    fprintf(f, "\n");
                }

                fprintf(f, "-- target (%zu geodes) --\n", m_targetGeodes.size());
                for (size_t i = 0; i < m_targetGeodes.size(); ++i)
                {
                    osg::Geode *g = m_targetGeodes[i];
                    if (!g)
                    {
                        fprintf(f, "  [%zu] (null)\n", i);
                        continue;
                    }
                    osg::StateSet *ss = g->getStateSet();
                    const osg::Stencil *st = ss ? dynamic_cast<const osg::Stencil *>(ss->getAttribute(osg::StateAttribute::STENCIL)) : nullptr;
                    const osg::Depth *dp = ss ? dynamic_cast<const osg::Depth *>(ss->getAttribute(osg::StateAttribute::DEPTH)) : nullptr;
                    fprintf(f, "  [%zu] '%s' mask=0x%x STENCIL=%s DEPTH=%s\n", i, g->getName().c_str(), (unsigned)g->getNodeMask(), st ? "yes" : "no", dp ? "yes" : "no");
                    if (st)
                        fprintf(f, "       stencil func=%d ref=%d fmask=0x%x wmask=0x%x\n", (int)st->getFunction(), (int)st->getFunctionRef(), (unsigned)st->getFunctionMask(), (unsigned)st->getWriteMask());
                }
                fprintf(f, "-- target shadow --\n");
                if (m_targetShadow && m_targetShadow->node.get())
                {
                    osg::StateSet *ss = m_targetShadow->node->getStateSet();
                    const osg::ColorMask *cm = ss ? dynamic_cast<const osg::ColorMask *>(ss->getAttribute(osg::StateAttribute::COLORMASK)) : nullptr;
                    int bin = ss && ss->useRenderBinDetails() ? ss->getBinNumber() : -99;
                    fprintf(f, "  '%s' bin=%d colormask=%s mask=0x%x parents=%d\n", m_targetShadow->node->getName().c_str(), bin, cm ? (cm->getRedMask() ? "on" : "off") : "none", (unsigned)m_targetShadow->node->getNodeMask(), (int)m_targetShadow->node->getNumParents());
                }
                else
                    fprintf(f, "  (none)\n");

                fprintf(f, "-- void shadows (%zu) --\n", m_voidShadows.size());
                for (auto &kv : m_voidShadows)
                {
                    Shadow *sh = kv.second;
                    fprintf(f, "  [%s] geodes=%zu hidden=%d\n", kv.first.c_str(), sh->geodes.size(), (int)sh->appliedHidden);
                    if (sh->node.get())
                    {
                        osg::StateSet *ss = sh->node->getStateSet();
                        const osg::Stencil *st = ss ? dynamic_cast<const osg::Stencil *>(ss->getAttribute(osg::StateAttribute::STENCIL)) : nullptr;
                        int bin = ss && ss->useRenderBinDetails() ? ss->getBinNumber() : -99;
                        fprintf(f, "    near '%s' bin=%d stencil=%s ref=%d wmask=0x%x parents=%d\n", sh->node->getName().c_str(), bin, st ? "yes" : "no", st ? (int)st->getFunctionRef() : -1, st ? (unsigned)st->getWriteMask() : 0, (int)sh->node->getNumParents());
                    }
                    else
                        fprintf(f, "    near (none)\n");
                    if (sh->nodeFar.get())
                    {
                        osg::StateSet *ss = sh->nodeFar->getStateSet();
                        const osg::Stencil *st = ss ? dynamic_cast<const osg::Stencil *>(ss->getAttribute(osg::StateAttribute::STENCIL)) : nullptr;
                        int bin = ss && ss->useRenderBinDetails() ? ss->getBinNumber() : -99;
                        fprintf(f, "    far  '%s' bin=%d stencil=%s ref=%d wmask=0x%x parents=%d\n", sh->nodeFar->getName().c_str(), bin, st ? "yes" : "no", st ? (int)st->getFunctionRef() : -1, st ? (unsigned)st->getWriteMask() : 0, (int)sh->nodeFar->getNumParents());
                    }
                    else
                        fprintf(f, "    far  (none)\n");
                }
                fclose(f);
                if (!printedOnce)
                {
                    fprintf(stderr, "CLIP: debug dump -> %s\n", dbgFile);
                    printedOnce = true;
                }
            }
        }
    }
}

void ClipPlugin::applyMainPass(osg::StateSet *ss, bool on)
{
    if (!ss)
        return;

    // the target shadow writes the target's own depth in an early bin, so the
    // real target must use LEQUAL (not the default LESS) or it would fail its
    // own depth and disappear (black)
    ss->setAttributeAndModes(new osg::Depth(osg::Depth::LEQUAL, 0.0, 1.0, true), osg::StateAttribute::ON);

    if (on)
    {
        // cut only where bit 0 is set and bit 1 is clear (value 1): the target
        // lies between the void's near and far surface, i.e. inside the void.
        // Where the void only occludes the target (value 3) or nothing writes
        // (value 0) the target is kept.
        osg::Stencil *stencil = new osg::Stencil();
        stencil->setFunction(osg::Stencil::NOTEQUAL, 1, 3); // both bits: cut where value==1
        stencil->setOperation(osg::Stencil::KEEP, osg::Stencil::KEEP, osg::Stencil::KEEP);
        stencil->setWriteMask(0);
        ss->setAttributeAndModes(stencil, osg::StateAttribute::ON);
    }
    else
    {
        ss->removeAttribute(osg::StateAttribute::STENCIL);
    }
}

void ClipPlugin::updateMainPass()
{
    bool cut = !m_targetName.empty() && !m_voidNames.empty();
    for (osg::Geode *geode : m_targetGeodes)
        if (geode)
            applyMainPass(geode->getOrCreateStateSet(), cut);
}

bool ClipPlugin::selectionChanged()
{
    // only act while a Clip pick mode is active; otherwise leave the
    // Selection/Move plugins' selection untouched (the persistent selection
    // highlight doubles as a marker for the hidden void objects)
    if (m_mode == Mode::None)
        return true;

    auto selection = coVRSelectionManager::instance()->getSelectionList();
    if (selection.empty())
    {
        // deselected while a pick mode was active: keep the mode, wait for the
        // next selection
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
    if (geode)
    {
        std::string name = objectNameOf(geode);
        switch (m_mode)
        {
        case Mode::PickTarget:
            pickTarget(geode, name);
            break;
        case Mode::AddVoid:
            pickVoid(geode, name);
            break;
        case Mode::RemoveVoid:
            if (!name.empty())
                removeVoid(name);
            break;
        default:
            break;
        }
    }

    // a manual pick takes over from the config setup; the pick mode stays
    // active until the button is pressed again
    m_manual = true;
    refreshStatus();
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
    // a new COVISE object may match the configured setup, or be the replacement
    // of a replaced target/void; syncShadows() re-resolves the objects by name
    // every frame, so only trigger a (re-)application of the config setup here
    syncSetup();
}

void ClipPlugin::removeNode(osg::Node *node, bool isGroup, osg::Node *realNode)
{
    (void)isGroup;
    (void)realNode;
    if (!node)
        return;

    // a replaced or deleted object removes its geodes from the scene; the stale
    // geode pointers are dropped now (removeObject() may have already cleaned
    // up). syncShadows() re-resolves by name and re-attaches once a
    // replacement appears again.
    if (auto *geode = dynamic_cast<osg::Geode *>(realNode))
    {
        for (auto *g : m_targetGeodes)
            if (g == geode)
                g = nullptr;
        for (auto &kv : m_voidShadows)
            if (kv.second)
                for (auto *g : kv.second->geodes)
                    if (g == geode)
                        g = nullptr; // force re-attach on the next frame
    }
}

void ClipPlugin::removeObject(const char *objName, bool replaceFlag)
{
    (void)replaceFlag;
    if (!objName)
        return;

    // the config names the object, but the scene top-level node may be named
    // differently (file objects carry their path); resolve before comparing
    const std::string name = objName;

    auto vit = std::find(m_voidNames.begin(), m_voidNames.end(), name);
    if (vit == m_voidNames.end())
        for (auto it = m_voidNames.begin(); it != m_voidNames.end(); ++it)
            if (matchName(*it, name) || matchName(name, *it))
            {
                vit = it;
                break;
            }
    if (vit != m_voidNames.end())
    {
        const std::string voidName = *vit;
        m_voidNames.erase(vit);
        removeVoidShadow(voidName);
        updateVoidsUi();
        updateMainPass();
    }

    if (m_targetName == name)
    {
        removeTargetShadow();
        restoreTarget();
        m_targetName.clear();
        m_targetTopNode = nullptr;
        m_targetGeodes.clear();
        updateMainPass();
        refreshStatus();
    }
}

void ClipPlugin::updateVoidsUi()
{
    std::vector<std::string> names(m_voidNames);
    if (m_voidsList)
        m_voidsList->setList(names);
    refreshStatus();
}

void ClipPlugin::refreshStatus()
{
    if (m_targetLabel)
        m_targetLabel->setText("Target: " + (m_targetName.empty() ? std::string("(none)") : m_targetName));
    if (m_voidsLabel)
    {
        std::string text = "Voids: ";
        if (m_voidNames.empty())
            text += "(none)";
        else
        {
            for (size_t i = 0; i < m_voidNames.size(); ++i)
                text += (i ? ", " : "") + m_voidNames[i];
        }
        m_voidsLabel->setText(text);
    }
}

COVERPLUGIN(ClipPlugin)
