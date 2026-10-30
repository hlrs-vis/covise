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

// find the first geode in a subtree
class GeodeFirstVisitor : public osg::NodeVisitor
{
public:
    explicit GeodeFirstVisitor(osg::Geode * &result)
        : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
        , m_result(result)
    {
    }

    void apply(osg::Geode &geode) override
    {
        if (!m_result)
            m_result = &geode;
    }

private:
    osg::Geode * &m_result;
};

// masks stripped from a hidden void: no collision/walk/touch, so lasers,
// navigation and walking pass through the invisible object; Pick stays set so
// Selection/Move can still grab it
const unsigned int HIDDEN_STRIP_MASK = Isect::Collision | Isect::Walk | Isect::Touch;

void setShadowFlags(osg::Node *node)
{
    node->setNodeMask(node->getNodeMask() & ~(Isect::Collision | Isect::Intersection | Isect::Pick | Isect::Update | Isect::Walk | Isect::Touch));
}

// renders only into the depth buffer: a clone of the target written before the
// void pass, so the voids can be depth-tested against the target and only the
// part of a void that is actually in front of (or inside) the target writes to
// the stencil. Color off, no stencil, depth write on.
void prepareTargetShadow(osg::Node *shadow)
{
    osg::StateSet *ss = shadow->getOrCreateStateSet();
    ss->setRenderBinDetails(-2, "RenderBin");
    ss->setNestRenderBins(false);
    ss->setAttributeAndModes(new osg::ColorMask(false, false, false, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
}

// renders only into the stencil buffer, depth-tested against the target shadow:
// writes 1 where the void object is in front of (or inside) the target, so the
// target is cut out there but not where the void is hidden behind the target.
// Color and depth write are off so it does not show up or disturb the scene.
void prepareVoidShadow(osg::Node *shadow)
{
    osg::StateSet *ss = shadow->getOrCreateStateSet();
    ss->setRenderBinDetails(-1, "RenderBin");
    ss->setNestRenderBins(false);
    ss->setAttributeAndModes(new osg::ColorMask(false, false, false, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

    osg::Stencil *stencil = new osg::Stencil();
    stencil->setFunction(osg::Stencil::ALWAYS, 1, 1);
    // only the fragments that pass the depth test against the target shadow
    // (i.e. are in front of the target) write to the stencil
    stencil->setOperation(osg::Stencil::KEEP, osg::Stencil::KEEP, osg::Stencil::REPLACE);
    stencil->setWriteMask(1);
    ss->setAttributeAndModes(stencil, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

    // test against the target shadow but do not write depth, so the invisible
    // clone cannot cull the real object in the main pass
    ss->setAttributeAndModes(new osg::Depth(osg::Depth::LESS, 0.0, 1.0, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);

    ss->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
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
    while (node && node != root)
    {
        if (coVRSelectionManager::instance()->isHelperNode(node) &&
            (helperType < 0 || helperTypeOf(node) == helperType))
            return true;
        if (node->getNumParents() == 0)
            break;
        node = node->getParent(0);
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
    if (!node)
        return "";

    osg::Group *root = cover->getObjectsRoot();

    // walk up the parent chain, skipping selection helper groups and our own shadows
    osg::Node *n = node;
    while (n)
    {
        osg::Group *group = n->asGroup();
        if (group)
        {
            if (group->getName().find("_ClipShadow") != std::string::npos)
                return "";
            if (coVRSelectionManager::instance()->isHelperNode(n))
            {
                n = n->getNumParents() > 0 ? n->getParent(0) : nullptr;
                continue;
            }
            // plain file-loaded object: the top-level node is a direct child of the objects root
            if (root && n->getNumParents() == 1 && n->getParent(0) == root)
                return n->getName();
            // covise object: <name>_Mat MatrixTransform wraps the <name>_Geom geode
            const std::string &nm = group->getName();
            if (nm.size() > 4 && nm.compare(nm.size() - 4, 4, "_Mat") == 0)
                return nm.substr(0, nm.size() - 4);
        }
        n = n->getNumParents() > 0 ? n->getParent(0) : nullptr;
    }

    return node->getName();
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
        const std::string n = root->getChild(i)->getName();
        if (matchName(name, n))
            return n;
        if (prefixMatch.empty() && name.size() >= 3 && n.find(name) == 0)
            prefixMatch = n;
    }
    return prefixMatch;
}

osg::Geode *ClipPlugin::geodeOf(const std::string &name) const
{
    if (name.empty())
        return nullptr;

    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return nullptr;

    // find the top-level node (exact match, then basename/prefix)
    osg::Node *top = nullptr;
    for (unsigned int i = 0; i < root->getNumChildren(); ++i)
    {
        osg::Node *child = root->getChild(i);
        const std::string n = child->getName();
        if (n == name)
        {
            top = child;
            break;
        }
        if (!top && matchName(name, n))
            top = child;
    }
    if (!top)
        return nullptr;

    if (auto *geode = dynamic_cast<osg::Geode *>(top))
        return geode;

    osg::Geode *found = nullptr;
    GeodeFirstVisitor visitor(found);
    top->accept(visitor);
    return found;
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
    if (m_targetGeode.get() == geode && m_targetName == name)
        return;

    if (cover->debugLevel(3))
        std::cerr << "Clip: target = " << name << "\n";
    // reset the previous target's rendering before switching
    restoreTarget();

    m_targetName = name;
    m_targetGeode = geode;

    // save the geode's original depth/stencil so restoreTarget() can bring it
    // back exactly (e.g. an object may already carry its own attributes)
    osg::StateSet *ss = geode->getOrCreateStateSet();
    m_targetSavedDepth = ss->getAttribute(osg::StateAttribute::DEPTH);
    m_targetSavedStencil = ss->getAttribute(osg::StateAttribute::STENCIL);
    m_targetSavedState = true;

    rebuildTargetShadow(geode);
    updateMainPass();
    refreshStatus();
}

void ClipPlugin::restoreTarget()
{
    if (m_targetGeode.get() && m_targetSavedState)
    {
        osg::StateSet *ss = m_targetGeode->getOrCreateStateSet();
        if (m_targetSavedDepth)
            ss->setAttributeAndModes(m_targetSavedDepth, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::DEPTH);
        if (m_targetSavedStencil)
            ss->setAttributeAndModes(m_targetSavedStencil, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::STENCIL);
    }
    m_targetSavedState = false;
    m_targetSavedDepth = nullptr;
    m_targetSavedStencil = nullptr;
}

void ClipPlugin::pickVoid(osg::Geode *geode, const std::string &name)
{
    if (!geode || name.empty() || name == m_targetName || std::find(m_voidNames.begin(), m_voidNames.end(), name) != m_voidNames.end())
        return;

    if (cover->debugLevel(3))
        std::cerr << "Clip: void = " << name << "\n";
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
    // restore the target geode and drop the shadows
    for (const auto &name : m_voidNames)
        removeVoidShadow(name);
    restoreTarget();
    removeTargetShadow();
    m_targetName.clear();
    m_targetGeode = nullptr;
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
            if (auto *geode = geodeOf(name))
                pickTarget(geode, name);
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
        if (auto *geode = geodeOf(name))
            pickVoid(geode, name);
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
    if (!shadow || !shadow->geode || shadow->geode->getNumParents() == 0)
        return;

    // a void that is currently selected / being moved (a SELECTION helper node
    // sits above its geode) stays visible, so it can be edited in the scene.
    // The permanent MOVE helper is ignored: it only means the object was moved.
    const bool hidden = m_hideVoids && !hasHelperAncestor(shadow->geode, coVRSelectionManager::SELECTION);
    if (shadow->appliedHidden == hidden)
        return;
    shadow->appliedHidden = hidden;

    osg::Geode *geode = shadow->geode;
    osg::StateSet *ss = geode->getOrCreateStateSet();

    if (hidden)
    {
        // save the geode's original rendering state once so we can restore it
        if (!shadow->savedColorMask)
            shadow->savedColorMask = ss->getAttribute(osg::StateAttribute::COLORMASK);
        if (!shadow->savedDepth)
            shadow->savedDepth = ss->getAttribute(osg::StateAttribute::DEPTH);
        if (!shadow->savedStencil)
            shadow->savedStencil = ss->getAttribute(osg::StateAttribute::STENCIL);
        if (shadow->savedMask == 0)
            shadow->savedMask = geode->getNodeMask();

        // hide: write no color, no depth, ignore stencil. Lasers/walking pass
        // through (Collision/Walk/Touch removed from the mask), but Pick stays
        // set so Selection/Move can still grab the object. The selection
        // outline makes it visible while it is being edited.
        ss->setAttributeAndModes(new osg::ColorMask(false, false, false, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        ss->setAttributeAndModes(new osg::Depth(osg::Depth::ALWAYS, 0.0, 1.0, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        osg::Stencil *stencil = new osg::Stencil();
        stencil->setFunction(osg::Stencil::ALWAYS, 0, 0);
        ss->setAttributeAndModes(stencil, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
        geode->setNodeMask(geode->getNodeMask() & ~HIDDEN_STRIP_MASK);
    }
    else
    {
        if (shadow->savedColorMask)
            ss->setAttributeAndModes(shadow->savedColorMask, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::COLORMASK);
        if (shadow->savedDepth)
            ss->setAttributeAndModes(shadow->savedDepth, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::DEPTH);
        if (shadow->savedStencil)
            ss->setAttributeAndModes(shadow->savedStencil, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::STENCIL);
        if (shadow->savedMask != 0)
            geode->setNodeMask(shadow->savedMask);
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

void ClipPlugin::rebuildTargetShadow(osg::Geode *geode)
{
    removeTargetShadow();
    if (!geode || geode->getNumParents() == 0)
        return;

    auto *parent = geode->getParent(0)->asGroup();
    if (!parent)
        return;

    auto *shadow = new Shadow();
    shadow->name = m_targetName;
    shadow->parent = parent;
    shadow->geode = geode;
    shadow->node = geode->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
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

    // restore the geode's original rendering state
    if (shadow->geode && shadow->geode->getNumParents() > 0)
    {
        osg::StateSet *ss = shadow->geode->getOrCreateStateSet();
        if (shadow->savedColorMask)
            ss->setAttributeAndModes(shadow->savedColorMask, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::COLORMASK);
        if (shadow->savedDepth)
            ss->setAttributeAndModes(shadow->savedDepth, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::DEPTH);
        if (shadow->savedStencil)
            ss->setAttributeAndModes(shadow->savedStencil, osg::StateAttribute::ON);
        else
            ss->removeAttribute(osg::StateAttribute::STENCIL);
        if (shadow->savedMask != 0)
            shadow->geode->setNodeMask(shadow->savedMask);
    }

    if (shadow->parent.get() && shadow->node.get())
        shadow->parent->removeChild(shadow->node.get());
    delete shadow;
    m_voidShadows.erase(it);
}

void ClipPlugin::createVoidShadow(const std::string &name, osg::Geode *geode)
{
    if (!geode || geode->getNumParents() == 0)
        return;

    osg::Group *parent = geode->getParent(0)->asGroup();
    if (!parent)
        return;

    auto *shadow = new Shadow();
    shadow->name = name;
    shadow->parent = parent;
    shadow->geode = geode;
    shadow->savedMask = geode->getNodeMask();
    shadow->node = geode->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
    if (!shadow->node.get())
    {
        delete shadow;
        return;
    }

    shadow->node->setName(name + "_ClipShadow");
    setShadowFlags(shadow->node.get());
    prepareVoidShadow(shadow->node.get());
    parent->addChild(shadow->node.get());
    m_voidShadows[name] = shadow;
}

void ClipPlugin::syncShadows()
{
    // The Move/Selection plugins wrap geodes in helper MatrixTransforms at
    // edit time and COVISE updates replace geodes. A shadow clone is a sibling
    // of the geode (under the same current parent), so re-parenting it under
    // the geode's current parent keeps it exactly on the object for any
    // (helper) transform in between. Re-attach only when something changed.

    // iterate a copy: removing a void (deleted object) changes m_voidNames
    std::vector<std::string> voidNames(m_voidNames.begin(), m_voidNames.end());
    for (const auto &name : voidNames)
    {
        Shadow *shadow = shadowFor(name);
        if (!shadow)
            continue;

        osg::Geode *geode = geodeOf(name);
        if (!geode || geode->getNumParents() == 0)
        {
            // object not in the scene (e.g. mid-replace, or deleted via the
            // SGBrowser, which does not fire the plugin remove hooks): keep
            // the last state for a few frames, then clean up
            if (++m_missingFrames[name] < 30)
                continue;
            removeVoid(name);
            continue;
        }
        m_missingFrames.erase(name);

        if (geode != shadow->geode)
        {
            // the geode was replaced (COVISE object update): re-clone and
            // re-capture the original state of the new geode
            reattachVoidShadow(name);
            applyVoidVisibility(name);
            continue;
        }

        // the geode's parent may have changed (a helper was inserted or
        // removed above it) -> move the shadow with it. The clone carries the
        // geode's local matrix, so it stays on the object after re-parenting.
        osg::Group *parent = geode->getParent(0)->asGroup();
        if (parent && parent != shadow->parent.get())
        {
            if (shadow->parent.get() && shadow->node.get())
                shadow->parent->removeChild(shadow->node.get());
            shadow->parent = parent;
            if (shadow->node.get())
                shadow->parent->addChild(shadow->node.get());
        }

        // show the void while it is being edited, hide it otherwise
        applyVoidVisibility(name);
    }

    // target: same handling
    if (m_targetName.empty() || !m_targetGeode.get())
        return;

    osg::Geode *geode = geodeOf(m_targetName);
    if (!geode || geode->getNumParents() == 0)
    {
        // object not in the scene (e.g. mid-replace, or deleted via the
        // SGBrowser): keep the last state for a few frames, then clean up
        if (++m_targetMissingFrames >= 30)
        {
            removeTargetShadow();
            m_targetName.clear();
            m_targetGeode = nullptr;
            updateMainPass();
            refreshStatus();
        }
        return;
    }
    m_targetMissingFrames = 0;

    if (geode != m_targetGeode.get())
    {
        // the geode was replaced by a COVISE update: rebuild the clone and
        // re-apply the main-pass overrides on the new geode
        restoreTarget();
        m_targetGeode = geode;
        osg::StateSet *ss = geode->getOrCreateStateSet();
        m_targetSavedDepth = ss->getAttribute(osg::StateAttribute::DEPTH);
        m_targetSavedStencil = ss->getAttribute(osg::StateAttribute::STENCIL);
        m_targetSavedState = true;
        rebuildTargetShadow(geode);
        updateMainPass();
        return;
    }

    if (m_targetShadow && m_targetShadow->node.get())
    {
        osg::Group *parent = geode->getParent(0)->asGroup();
        if (parent && parent != m_targetShadow->parent.get())
        {
            if (m_targetShadow->parent.get())
                m_targetShadow->parent->removeChild(m_targetShadow->node.get());
            m_targetShadow->parent = parent;
            m_targetShadow->parent->addChild(m_targetShadow->node.get());
        }
    }
}

void ClipPlugin::reattachVoidShadow(const std::string &name)
{
    // clone a fresh shadow next to the object's current geode (used after a
    // COVISE object update replaced the geode)
    Shadow *shadow = shadowFor(name);
    if (!shadow)
        return;

    osg::Geode *geode = geodeOf(name);
    if (!geode || geode->getNumParents() == 0)
        return;

    if (shadow->parent.get() && shadow->node.get())
        shadow->parent->removeChild(shadow->node.get());

    shadow->geode = geode;
    shadow->savedMask = geode->getNodeMask();
    osg::Group *parent = geode->getParent(0)->asGroup();
    if (!parent)
        return;
    shadow->parent = parent;

    shadow->node = geode->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
    if (!shadow->node.get())
        return;

    shadow->node->setName(name + "_ClipShadow");
    setShadowFlags(shadow->node.get());
    prepareVoidShadow(shadow->node.get());
    parent->addChild(shadow->node.get());

    // the new geode has its original (unmodified) attributes, so forget the
    // saved ones and let applyVoidVisibility() capture fresh ones
    shadow->savedColorMask = nullptr;
    shadow->savedDepth = nullptr;
    shadow->savedStencil = nullptr;
    shadow->appliedHidden = false;
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

                fprintf(f, "-- target geode %p --\n", (void *)m_targetGeode.get());
                if (m_targetGeode.get())
                {
                    osg::Geode *g = m_targetGeode.get();
                    fprintf(f, "  name='%s' mask=0x%x parents=%d\n", g->getName().c_str(), (unsigned)g->getNodeMask(), (int)g->getNumParents());
                    osg::StateSet *ss = g->getStateSet();
                    if (ss)
                    {
                        const osg::Stencil *st = dynamic_cast<const osg::Stencil *>(ss->getAttribute(osg::StateAttribute::STENCIL));
                        const osg::Depth *dp = dynamic_cast<const osg::Depth *>(ss->getAttribute(osg::StateAttribute::DEPTH));
                        fprintf(f, "  STENCIL=%s DEPTH=%s useRBD=%d\n", st ? "yes" : "no", dp ? "yes" : "no", (int)ss->useRenderBinDetails());
                        if (st)
                            fprintf(f, "    stencil func=%d ref=%d fmask=0x%x wmask=0x%x\n", (int)st->getFunction(), (int)st->getFunctionRef(), (unsigned)st->getFunctionMask(), (unsigned)st->getWriteMask());
                    }
                    else
                        fprintf(f, "  no state set\n");
                }
                fprintf(f, "-- target shadow --\n");
                if (m_targetShadow && m_targetShadow->node.get())
                    fprintf(f, "  '%s' mask=0x%x parents=%d\n", m_targetShadow->node->getName().c_str(), (unsigned)m_targetShadow->node->getNodeMask(), (int)m_targetShadow->node->getNumParents());
                else
                    fprintf(f, "  (none)\n");

                fprintf(f, "-- void shadows (%zu) --\n", m_voidShadows.size());
                for (auto &kv : m_voidShadows)
                {
                    Shadow *sh = kv.second;
                    fprintf(f, "  [%s]\n", kv.first.c_str());
                    if (sh->node.get())
                    {
                        osg::StateSet *ss = sh->node->getStateSet();
                        const osg::Stencil *st = ss ? dynamic_cast<const osg::Stencil *>(ss->getAttribute(osg::StateAttribute::STENCIL)) : nullptr;
                        fprintf(f, "    shadow '%s' mask=0x%x parents=%d stencil=%s\n", sh->node->getName().c_str(), (unsigned)sh->node->getNodeMask(), (int)sh->node->getNumParents(), st ? "yes" : "no");
                    }
                    else
                        fprintf(f, "    shadow (none)\n");
                    if (sh->geode)
                    {
                        osg::StateSet *ss = sh->geode->getStateSet();
                        const osg::ColorMask *cm = ss ? dynamic_cast<const osg::ColorMask *>(ss->getAttribute(osg::StateAttribute::COLORMASK)) : nullptr;
                        fprintf(f, "    geode %p mask=0x%x hidden=%d colormask=%s\n", (void *)sh->geode, (unsigned)sh->geode->getNodeMask(), (int)sh->appliedHidden, cm ? (cm->getRedMask() ? "on" : "off") : "none");
                    }
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
        // draw only where no void object wrote to the stencil (EQUAL 0), which
        // cuts out the void from the target
        osg::Stencil *stencil = new osg::Stencil();
        stencil->setFunction(osg::Stencil::EQUAL, 0, 1);
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
    if (m_targetGeode)
        applyMainPass(m_targetGeode->getOrCreateStateSet(), cut);
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
        osg::Geode *found = nullptr;
        GeodeFirstVisitor visitor(found);
        node->accept(visitor);
        geode = found;
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
    // of a replaced target/void; syncShadows() re-resolves the geodes by name
    // every frame, so only trigger a (re-)application of the config setup here
    syncSetup();
}

void ClipPlugin::removeNode(osg::Node *node, bool isGroup, osg::Node *realNode)
{
    (void)isGroup;
    (void)realNode;
    if (!node)
        return;

    // a replaced or deleted object removes its geode from the scene; the stale
    // geode pointer is dropped now (removeObject() may have already cleaned up).
    // syncShadows() re-resolves by name and re-attaches once a replacement
    // geode appears again.
    if (auto *geode = dynamic_cast<osg::Geode *>(realNode))
    {
        if (m_targetGeode.get() == geode)
        {
            removeTargetShadow();
            m_targetGeode = nullptr;
            updateMainPass();
        }
        for (auto &kv : m_voidShadows)
            if (kv.second && kv.second->geode == geode)
                kv.second->geode = nullptr; // force re-attach on the next frame
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
        m_targetGeode = nullptr;
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
