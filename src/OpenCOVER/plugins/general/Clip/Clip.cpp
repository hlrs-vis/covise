#include "Clip.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/ui/Button.h>
#include <cover/ui/Group.h>
#include <cover/ui/Label.h>
#include <cover/ui/Menu.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/View.h>

#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/NodeVisitor>
#include <osg/PagedLOD>
#include <osg/ColorMask>
#include <osg/CullFace>
#include <osg/Depth>
#include <osg/Stencil>
#include <osg/StateSet>

#include <osgUtil/CullVisitor>
#include <osgUtil/RenderBin>
#include <osgUtil/RenderStage>

#include <algorithm>
#include <cstdarg>
#include <cstdio>

#include <iostream>
#include <tuple>

using namespace opencover;

namespace
{

// ---- CLIP_DEBUG diagnostics (gated by the CLIP_DEBUG env var) ----
bool clipDebug()
{
    static int v = getenv("CLIP_DEBUG") ? 1 : 0;
    return v;
}

void dbg(const char *fmt, ...)
{
    if (!clipDebug())
        return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
}

// frame counter bumped in preFrame(), so cull callbacks can report
// per-frame without touching the RenderInfo
inline int &clipFrameCounter()
{
    static int f = 0;
    return f;
}

void dbgBounds(const char *label, const osg::Node *n)
{
    if (!clipDebug() || !n)
        return;
    const osg::BoundingSphere &bs = n->getBound();
    if (bs.valid())
        fprintf(stderr, "CLIP: %s bound c=(%.2f, %.2f, %.2f) r=%.2f\n", label, bs._center.x(), bs._center.y(), bs._center.z(), bs._radius);
    else
        fprintf(stderr, "CLIP: %s bound EMPTY\n", label);
}

// prints the full chain from a node up to (and including) the objects root
// so a failing climb (e.g. pickVoid ABORT) can be read directly from the log
void dbgParentChain(const char *label, osg::Node *n, osg::Group *root)
{
    if (!clipDebug())
        return;
    fprintf(stderr, "CLIP: %s parent chain from '%s' [%s]:\n", label, n ? n->getName().c_str() : "(null)", n ? n->className() : "");
    int i = 0;
    osg::Node *c = n;
    while (c && i < 60)
    {
        const char *tag = "";
        if (c == root)
            tag = "  <== OBJECTS_ROOT";
        else if (c->getName().find("_ClipShadow") != std::string::npos)
            tag = "  <== CLIP SHADOW";
        else if (coVRSelectionManager::instance()->isHelperNode(c))
            tag = "  <== SELECTION HELPER";
        fprintf(stderr, "  [%d] %s '%s' mask=0x%x parents=%u%s\n", i, c->className(), c->getName().c_str(), c->getNodeMask(), (unsigned)c->getNumParents(), tag);
        if (c->getNumParents() == 0)
        {
            fprintf(stderr, "  [%d] (no parent - STOP, root reached: %s)\n", i + 1, c == root ? "yes" : "NO - walk would ABORT");
            break;
        }
        c = c->getParent(0);
        ++i;
    }
    if (i >= 60)
        fprintf(stderr, "  ... (chain longer than 60, truncated)\n");
}

// reports which render stage/bin a shadow subtree is being culled into;
// prints only on the first cull of a frame or when the stage/bin changed
// (which drawables end up where is a per-drawable matter, but the bin the
// shadow ROOT is culled into already reveals override leaks)
class ClipCullReport : public osg::NodeCallback
{
public:
    explicit ClipCullReport(const char *tag)
        : m_tag(tag)
    {
    }

    void operator()(osg::Node *node, osg::NodeVisitor *nv) override
    {
        if (clipDebug() && node)
        {
            osgUtil::CullVisitor *cv = dynamic_cast<osgUtil::CullVisitor *>(nv);
            osgUtil::RenderBin *bin = cv ? cv->getCurrentRenderBin() : nullptr;
            osgUtil::RenderStage *stage = cv ? cv->getCurrentRenderStage() : nullptr;
            osg::Camera *cam = stage ? stage->getCamera() : nullptr;
            int frame = clipFrameCounter();
            int binNum = bin ? bin->getBinNum() : -12345;
            void *stagePtr = static_cast<void *>(stage);
            void *camPtr = static_cast<void *>(cam);
            if (frame != m_lastFrame || binNum != m_lastBin || stagePtr != m_lastStage || camPtr != m_lastCam)
            {
                m_lastFrame = frame;
                m_lastBin = binNum;
                m_lastStage = stagePtr;
                m_lastCam = camPtr;
                fprintf(stderr, "CLIP cull %s: frame %d stage=%p cam='%s' bin=%d\n", m_tag, frame, stagePtr, cam ? cam->getName().c_str() : "?", binNum);
            }
        }
        // OSG 3.6.5: a cull callback REPLACES the traversal -- the
        // children still have to be visited manually
        if (node && nv)
            node->traverse(*nv);
    }

private:
    const char *m_tag;
    int m_lastFrame = -1;
    int m_lastBin = 0;
    void *m_lastStage = nullptr;
    void *m_lastCam = nullptr;
};

// count drawables in a subtree that carry their own render bin details
// (a leak of such a drawable out of the shadow's pre-pass bin would move the
// geometry into the main pass or a wrong pre-pass)
void dbgBinLeaks(const char *tag, osg::Node *root)
{
    if (!clipDebug() || !root)
        return;
    struct DV : osg::NodeVisitor
    {
        int total = 0;
        int withBin = 0;
        std::vector<std::string> detail;
        DV()
            : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
        {
        }
        void apply(osg::Geode &g) override
        {
            for (unsigned int i = 0; i < g.getNumDrawables(); ++i)
            {
                osg::Drawable *d = g.getDrawable(i);
                if (!d)
                    continue;
                total++;
                osg::StateSet *ss = d->getStateSet();
                if (ss && ss->useRenderBinDetails() && !ss->getBinName().empty())
                {
                    withBin++;
                    if (detail.size() < 8)
                    {
                        char buf[48];
                        snprintf(buf, sizeof buf, "[%d]", ss->getBinNumber());
                        detail.push_back(std::string(d->getName().empty() ? "(anon)" : d->getName().c_str()) + buf);
                    }
                }
            }
        }
    } dv;
    root->accept(dv);
    fprintf(stderr, "CLIP: %s leak-check: drawables=%d withExplicitBin=%d%s\n", tag, dv.total, dv.withBin, dv.detail.empty() ? "" : ":");
    for (const auto &s : dv.detail)
        fprintf(stderr, "  %s\n", s.c_str());
}

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

    // shadow clones are siblings of the mirrored node INSIDE a moved object's
    // (helper-wrapped) subtree, so prune them: their geodes must never be
    // collected as live-object geodes (state restore, rebuild counts, ...)
    void apply(osg::Node &n) override
    {
        if (n.getName().find("_ClipShadow") != std::string::npos)
            return;
        traverse(n);
    }

private:
    std::vector<osg::Geode *> &m_result;
};

// find the first node (geode or group) carrying an exact name inside a subtree
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
        // never resolve into one of our own shadow clones: prune the subtree
        // (a shadow clone contains geodes named like the live object)
        if (n.getName().find("_ClipShadow") != std::string::npos)
            return;
        if (n.getName() == m_name)
        {
            node = &n;
            return;
        }
        // descend (the previous version never called traverse(), so it only
        // ever saw the top node it was called on and found no sub-node)
        traverse(n);
    }

private:
    const std::string &m_name;
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
    // OVERRIDE_PROTECTED (not the default USE) mode: the deep copy inherits each
    // target drawable's own render bin. The VRML plugin puts an explicit bin on
    // every drawable (setRenderingHint(OPAQUE_BIN) -> bin 0, USE mode), and a
    // USE-mode descendant bin OVERRIDES the enclosing root bin in the
    // CullVisitor (pushStateSet: useRenderBinDetails is true for it). That would
    // drag the whole target shadow out of the -3 pre-pass into the main pass,
    // so the void stencil passes saw an empty depth buffer and never cut.
    // OVERRIDE mode (the PROTECTED bit) suppresses every non-protected
    // descendant bin, pinning all shadow geometry to this bin.
    ss->setRenderBinDetails(-3, "RenderBin", osg::StateSet::OVERRIDE_PROTECTED_RENDERBIN_DETAILS);
    ss->setNestRenderBins(false);
    ss->setAttributeAndModes(new osg::ColorMask(false, false, false, false), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    ss->setMode(GL_DEPTH_TEST, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
    if (clipDebug())
        shadow->setCullCallback(new ClipCullReport("targetShadow"));
}

// shared state of both void passes: nothing is visible (color/depth write
// off), the geometry is depth-tested against the target shadow so only the
// part of the void in front of (or inside) the target writes to the stencil.
void prepareVoidShadowCommon(osg::Node *shadow, int bin, unsigned int stencilBit)
{
    osg::StateSet *ss = shadow->getOrCreateStateSet();
    // OVERRIDE_PROTECTED mode, same reason as prepareTargetShadow(): the
    // clones' drawables may carry their own (USE-mode) bin hints, which would
    // otherwise pull them out of this pre-pass bin.
    ss->setRenderBinDetails(bin, "RenderBin", osg::StateSet::OVERRIDE_PROTECTED_RENDERBIN_DETAILS);
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
    if (clipDebug())
        shadow->setCullCallback(new ClipCullReport("voidNear"));
}

// the far-surface (back faces) of the void: culls front faces, i.e. renders
// only the back shell of the object, and writes stencil bit 1 where that
// shell is in front of the target (depth shadow)
void prepareVoidShadowFar(osg::Node *shadow)
{
    prepareVoidShadowCommon(shadow, -1, 0x2);
    shadow->getOrCreateStateSet()->setAttributeAndModes(new osg::CullFace(osg::CullFace::FRONT), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
    if (clipDebug())
        shadow->setCullCallback(new ClipCullReport("voidFar"));
}

// ---- PagedLOD (paged terrain) shadow support ----
//
// A DEEP_COPY of a PagedLOD is still a *live* paged PagedLOD: it carries the
// same file names and (shallow-copied) DatabaseRequest state, so the
// DatabasePager pages it exactly like the original and loads the same tile
// files. The pager, however, attaches a loaded tile to only ONE PagedLOD, so
// the live target and the shadow then diverge in LOD state (one holds the
// high-res tile, the other stays on the low-res proxy). The depth shadow no
// longer matches the live target's depth, and the main pass renders the
// target black.
//
// neutralizePagedLODTree() therefore turns every PagedLOD in a shadow clone
// into a static snapshot of the geometry that is loaded *right now*: no more
// external paging, no shared DatabaseRequest state. The already-loaded
// children stay and keep being rendered (that is the depth the shadow
// provides).
// remove shadow-clone subtrees from a freshly deep-copied node. For a moved
// object the shadow clones live INSIDE the object's (helper-wrapped) subtree
// as siblings of the mirrored node (at any depth), so a clone of a subtree
// that contains them would duplicate them (nested shadows, leaked render
// bins). Live objects never carry such children, so pruning them from the
// copy is safe. Recurses: a clone of a clone would carry the name too.
void pruneShadowClones(osg::Node *node)
{
    osg::Group *g = node ? node->asGroup() : nullptr;
    if (!g)
        return;
    for (unsigned int i = 0; i < g->getNumChildren();)
    {
        osg::Node *c = g->getChild(i);
        if (c && c->getName().find("_ClipShadow") != std::string::npos)
        {
            g->removeChild(c);
            continue;
        }
        ++i;
    }
    for (unsigned int i = 0; i < g->getNumChildren(); ++i)
        pruneShadowClones(g->getChild(i));
}

void neutralizePagedLODTree(osg::Node *node)
{
    struct V : osg::NodeVisitor
    {
        V() : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN) {}
        void apply(osg::PagedLOD &p) override
        {
            p.setDisableExternalChildrenPaging(true);
            for (unsigned int i = 0; i < p.getNumFileNames(); ++i)
            {
                // drop the (shared) DatabaseRequest ref; the live PagedLOD
                // keeps its own. Also clear the file name so paging could
                // never fire for this node again.
                p.getDatabaseRequest(i) = nullptr;
                p.setFileName(i, "");
            }
            traverse(p);
        }
    } v;
    node->accept(v);
}

// signature of all PagedLODs in a subtree (their child counts, in tree
// order). Empty when the subtree holds no PagedLOD, i.e. for a static object
// it never changes. Used to detect when a paged target advanced to a new LOD
// level (a tile got loaded/expired) so the static shadow can be rebuilt to
// mirror the current level.
std::string PagedLODTreeSignature(osg::Node *node)
{
    std::string sig;
    struct V : osg::NodeVisitor
    {
        explicit V(std::string &s) : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN), sig(s) {}
        void apply(osg::PagedLOD &p) override
        {
            sig += "P" + std::to_string(p.getNumChildren()) + ";";
            traverse(p);
        }
        std::string &sig;
    } v(sig);
    node->accept(v);
    return sig;
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

// the real object a node belongs to: walk up to the boundary (the node whose
// parent is the objects root) and then back down, skipping helper nodes.
// Normally the boundary IS the object (a direct child of the root). But the
// Move/Selection plugins wrap a MOVED object in a MatrixTransform helper that
// becomes the direct child of the root (insertHelperNode does
// root.replaceChild(object, helper)), so for a moved object the boundary is
// the helper and the object is the first non-helper node below it. Returns
// null if the node is inside one of our shadow clones or not under the root.
osg::Node *objectOf(osg::Node *node)
{
    if (!node)
        return nullptr;
    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return nullptr;

    // chain[0] = node ... chain.back() = boundary (direct child of the root)
    std::vector<osg::Node *> chain;
    osg::Node *n = node;
    while (n)
    {
        if (n->getName().find("_ClipShadow") != std::string::npos)
            return nullptr;
        chain.push_back(n);
        if (n->getNumParents() == 1 && n->getParent(0) == root)
            break;
        if (n->getNumParents() == 0)
            return nullptr; // not under the objects root
        n = n->getParent(0);
    }
    if (chain.empty() || !(chain.back()->getNumParents() == 1 && chain.back()->getParent(0) == root))
        return nullptr;

    // from the boundary (topmost) down to the node: the first non-helper is
    // the object (the helper, if any, sits between the root and the object)
    for (size_t j = chain.size(); j-- > 0;)
        if (!coVRSelectionManager::instance()->isHelperNode(chain[j]))
            return chain[j];
    return chain.back(); // degenerate: every node on the chain is a helper
}

// for a direct child of the objects root: if it is a helper, return the real
// object inside it (helpers nest, so descend through all of them); otherwise
// the child itself. Null for a bare child without an object beneath.
osg::Node *innerObjectOf(osg::Node *child)
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
    std::fprintf(stderr, "CLIP: plugin loaded (build 2026-11-06: move-wrapped objects)\n");

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
    // delegate to the shared helper: it climbs to the objects-root boundary
    // and back down, skipping helper nodes (a MOVED object is wrapped in a
    // helper that is the direct child of the root, so the climb must not stop
    // at the helper). Returns the real top-level object, or null if the node
    // is inside one of our shadow clones / not under the root.
    osg::Node *top = objectOf(node);
    return top ? top : node;
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
        // a moved object sits INSIDE a helper that is the direct child of the
        // root: match the object's name (inside the helper), not the helper's
        // (empty) name
        osg::Node *inner = innerObjectOf(child);
        if (!inner || inner->getName().find("_ClipShadow") != std::string::npos)
            continue;
        const std::string n = inner->getName();
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
        // a moved object sits INSIDE a helper that is the direct child of the
        // root: match (and return) the object inside it, not the helper
        osg::Node *inner = innerObjectOf(child);
        if (!inner || inner->getName().find("_ClipShadow") != std::string::npos)
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

osg::Node *ClipPlugin::topObjectOf(osg::Node *node) const
{
    if (!node)
        return nullptr;
    // see topNodeOf(osg::Node*): objectOf() handles the Move-plugin-wrapped
    // case (helper as the direct child of the root) and returns null when the
    // picked node is inside one of our shadow clones or not under the root.
    return objectOf(node);
}

osg::Node *ClipPlugin::findNamedNode(const std::string &name) const
{
    if (name.empty())
        return nullptr;

    osg::Group *root = cover->getObjectsRoot();
    if (!root)
        return nullptr;

    // 1) a top-level object carrying the name (a moved object is wrapped in a
    //    helper that is the direct child of the root, so look inside it)
    for (unsigned int i = 0; i < root->getNumChildren(); ++i)
    {
        osg::Node *child = root->getChild(i);
        if (!child)
            continue;
        osg::Node *inner = innerObjectOf(child);
        if (!inner || inner->getName().find("_ClipShadow") != std::string::npos)
            continue;
        if (inner->getName() == name)
            return inner;
    }
    // 2) a named sub-node (e.g. a geode named after an OBJ material group,
    //    or a VRML object node) inside any top-level object
    for (unsigned int i = 0; i < root->getNumChildren(); ++i)
    {
        osg::Node *child = root->getChild(i);
        if (!child)
            continue;
        if (child->getName().find("_ClipShadow") != std::string::npos)
            continue;
        NameSearchVisitor v(name);
        child->accept(v);
        if (v.node)
            return v.node;
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

osg::Node *ClipPlugin::resolveTop(const osg::ref_ptr<osg::Node> &cached, const std::string &name) const
{
    if (cached.get())
    {
        osg::Group *root = cover->getObjectsRoot();
        // a still-valid cached top node: it is a direct child of the root, or
        // (a MOVED object) wrapped in a helper that is the direct child of the
        // root. objectOf() climbs to the boundary and back down, skipping
        // helpers, so it covers both; null means the node is gone or inside a
        // shadow clone.
        if (objectOf(cached.get()) &&
            root && cached.get()->getNumParents() == 1 &&
            (cached.get()->getParent(0) == root || coVRSelectionManager::instance()->isHelperNode(cached.get()->getParent(0))))
        {
            if (cached.get()->getName() == name)
                return cached.get();
            // the top node may be named by the file path while the name is a
            // sub-node name (e.g. an OBJ material group): confirm the subtree
            // actually contains the name
            NameSearchVisitor v(name);
            cached->accept(v);
            if (v.node)
                return cached.get();
        }
    }
    return topNodeOf(name);
}

osg::Node *ClipPlugin::resolveMirrorNode(osg::Node *top, const std::string &name) const
{
    if (!top)
        return nullptr;
    // whole-object target: the top node itself carries the name
    if (top->getName() == name)
        return top;
    // sub-node target (a named geode/group, e.g. a photogrammetry tile or an
    // OBJ material group): it must still exist under the object, otherwise the
    // part the user picked is gone and there is nothing to mirror
    osg::Node *byName = findNamedNode(name);
    if (byName && topObjectOf(byName) == top)
        return byName;
    return nullptr;
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
    if (!geode || name.empty() || name == m_targetName)
        return;
    osg::Node *top = topObjectOf(geode);
    if (!top)
        return;

    // the node the shadow will mirror: the picked node's named part (a
    // photogrammetry tile, an OBJ material group) or the whole object
    osg::Node *target = resolveMirrorNode(top, name);
    if (!target)
        target = top;

    std::vector<osg::Geode *> targetGeodes;
    {
        GeodeAllVisitor gv(targetGeodes);
        target->accept(gv);
    }
    if (targetGeodes.empty())
    {
        dbg("CLIP: pickTarget('%s') ABORT: no geodes under the picked node (top='%s')\n", name.c_str(), top->getName().c_str());
        return;
    }
    if (cover->debugLevel(3))
        std::cerr << "Clip: target = " << name << " (" << targetGeodes.size() << " geodes)\n";
    dbg("CLIP: pickTarget('%s') top='%s' shadow='%s' geodes=%d\n", name.c_str(), top->getName().c_str(), target->getName().c_str(), (int)targetGeodes.size());
    dbgBounds("CLIP: target top node", top);
    dbgBounds("CLIP: target shadow node", target);
    // reset the previous target's rendering before switching
    restoreTarget();

    m_targetName = name;
    m_targetTopNode = top;
    // derive the geode set and save each geode's original depth/stencil so
    // restoreTarget() can bring it back exactly (a freshly loaded paged tile
    // is still untouched here, so its current state IS the original state)
    reconcileTargetGeodes();

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

    osg::Node *top = topObjectOf(geode);
    if (!top || top->getNumParents() == 0)
    {
        dbg("CLIP: pickVoid('%s') ABORT: no top-level object above the picked node\n", name.c_str());
        dbgParentChain("pickVoid ABORT", geode, cover->getObjectsRoot());
        return;
    }

    if (cover->debugLevel(3))
        std::cerr << "Clip: void = " << name << " (" << geodesOf(name).size() << " geodes)\n";
    m_voidNames.push_back(name);
    createVoidShadow(name, geode, top);
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

    // the target/void names from the config may name a top-level object or a
    // named sub-node of one (e.g. an OBJ material group); resolve both. The
    // returned geode feeds the pick; the matched node decides the NAME (the
    // first geode of a top-level object would otherwise rename a whole-object
    // setup to one of its parts)
    auto resolveEntry = [this](const std::string &prefix) -> std::tuple<osg::Node *, osg::Geode *, osg::Node *>
    {
        osg::Group *root = cover->getObjectsRoot();
        if (!root)
            return {nullptr, nullptr, nullptr};
        for (unsigned int i = 0; i < root->getNumChildren(); ++i)
        {
            osg::Node *child = root->getChild(i);
            if (!child || child->getName().find("_ClipShadow") != std::string::npos)
                continue;
            osg::Node *match = nullptr;
            if (matchName(prefix, child->getName()))
                match = child;
            else
            {
                NameSearchVisitor v(prefix);
                child->accept(v);
                match = v.node;
            }
            if (!match)
                continue;
            osg::Node *top = topObjectOf(match);
            if (!top)
                top = child;
            osg::Geode *geode = nullptr;
            if (match->asGeode())
                geode = match->asGeode();
            else
            {
                std::vector<osg::Geode *> glist;
                GeodeAllVisitor gv(glist);
                match->accept(gv);
                if (!glist.empty())
                    geode = glist.front();
            }
            return {top, geode, match};
        }
        return {nullptr, nullptr, nullptr};
    };

    // target
    if (!targetPrefix.empty())
    {
        auto r = resolveEntry(targetPrefix);
        if (std::get<1>(r))
        {
            // name the entry from the MATCHED node (a top-level object keeps
            // its object name, a named sub-node keeps its part name) — not
            // from an arbitrary geode inside it
            const std::string name = displayNameOf(std::get<2>(r));
            if (m_targetName != name)
            {
                auto it = std::find(m_voidNames.begin(), m_voidNames.end(), name);
                if (it != m_voidNames.end())
                {
                    m_voidNames.erase(it);
                    removeVoidShadow(name);
                }
                pickTarget(std::get<1>(r), name);
            }
        }
    }
    // voids
    for (const auto &prefix : m_cfgVoids->value())
    {
        if (prefix.empty())
            continue;
        auto r = resolveEntry(prefix);
        if (!std::get<1>(r))
            continue;
        const std::string name = displayNameOf(std::get<2>(r));
        if (name.empty() || name == m_targetName)
            continue;
        if (std::find(m_voidNames.begin(), m_voidNames.end(), name) != m_voidNames.end())
            continue;
        pickVoid(std::get<1>(r), name);
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
    m_targetPagedLODSig.clear();
}

void ClipPlugin::rebuildTargetShadow()
{
    removeTargetShadow();
    osg::Node *top = resolveTop(m_targetTopNode, m_targetName);
    if (!top)
        return;

    // a (top-level) object that was picked; for sub-object picks (a named
    // geode, e.g. a photogrammetry tile or an OBJ material group) the clone
    // mirrors only that sub-node, so paging of the rest of the object does
    // not force shadow rebuilds
    osg::Node *target = resolveMirrorNode(top, m_targetName);
    if (!target)
        target = top;
    if (target->getNumParents() == 0)
        return;
    // the clone is a sibling of the mirrored node (a geode for sub-object
    // picks, the top-level group otherwise), so it follows that parent's
    // transforms exactly
    osg::Group *parent = target->getParent(0)->asGroup();
    if (!parent)
        return;

    auto *shadow = new Shadow();
    shadow->name = m_targetName;
    shadow->topNode = top;
    shadow->targetTopNode = target;
    shadow->geodes = m_targetGeodes;
    shadow->parent = parent;

    // a deep copy of the node (a group, or a bare geode if the object is a
    // single geode); it is a sibling of `target`, so it carries target's
    // local matrix and aligns exactly, following every (helper) transform.
    //
    // For a paged object (photogrammetry terrain) the clone is a *static*
    // snapshot of the currently loaded geometry: every PagedLOD in it is
    // neutralized so the DatabasePager does not page it as a second, live
    // PagedLOD (that would desync it from the live target and break the depth
    // shadow -> black tile). syncShadows() rebuilds the shadow whenever the
    // target's PagedLOD structure changes (a new LOD level was loaded).
    shadow->node = target->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
    if (!shadow->node.get())
    {
        delete shadow;
        return;
    }
    pruneShadowClones(shadow->node.get());
    neutralizePagedLODTree(shadow->node.get());
    shadow->pagedLODSig = PagedLODTreeSignature(target);
    m_targetPagedLODSig = shadow->pagedLODSig;
    shadow->node->setName(m_targetName + "_ClipShadow");
    setShadowFlags(shadow->node.get());
    prepareTargetShadow(shadow->node.get());
    parent->addChild(shadow->node.get());
    m_targetShadow = shadow;
    {
        // report the clone's OWN geode count (geodesOf() only matches direct
        // ObjectsRoot children, which is wrong for sub-object targets)
        std::vector<osg::Geode *> cloneGeodes;
        GeodeAllVisitor gv(cloneGeodes);
        shadow->node->accept(gv);
        dbg("CLIP: rebuildTargetShadow: clone geodes=%d (mirror='%s' parents=%d)\n", (int)cloneGeodes.size(), target->getName().c_str(), (int)target->getNumParents());
    }
    dbgBounds("CLIP: targetShadow clone", shadow->node.get());
    dbgBinLeaks("targetShadow", shadow->node.get());
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

void ClipPlugin::createVoidShadow(const std::string &name, osg::Geode *geode, osg::Node *topNode)
{
    // decide which node to clone: normally the (top-level) object the node
    // belongs to. When the picked node is a *named sub-node* (e.g. a geode
    // named after a material group of a multi-group OBJ, where the top-level
    // node is named by the file path), clone only that sub-node so the void
    // is exactly the shape the user picked
    osg::Node *target = topNode;
    if (geode && topNode && !geode->getName().empty() && geode != topNode)
    {
        osg::Node *byName = findNamedNode(name);
        if (byName)
            target = byName;
    }

    std::vector<osg::Geode *> topGeodes;
    if (target)
    {
        GeodeAllVisitor gv(topGeodes);
        target->accept(gv);
    }
    if (!target || target->getNumParents() == 0 || topGeodes.empty())
    {
        dbg("CLIP: createVoidShadow('%s') ABORT: target=%s parents=%d geodes=%d (picked geode='%s' top='%s')\n",
            name.c_str(), target ? "ok" : "null", target ? (int)target->getNumParents() : -1,
            (int)topGeodes.size(), geode ? geode->getName().c_str() : "null",
            topNode ? topNode->getName().c_str() : "null");
        return;
    }

    auto *shadow = new Shadow();
    shadow->name = name;
    shadow->topNode = topNode;
    shadow->targetTopNode = target;
    shadow->geodes = topGeodes;
    shadow->parent = target->getParent(0)->asGroup();
    // pre-size the saved-state bookkeeping (captured on the first hide)
    shadow->saved.resize(shadow->geodes.size());
    for (size_t i = 0; i < shadow->geodes.size(); ++i)
        shadow->saved[i].geode = shadow->geodes[i];
    makeVoidClones(shadow, name);
    m_voidShadows[name] = shadow;
    dbg("CLIP: createVoidShadow('%s') geodes=%d\n", name.c_str(), (int)shadow->geodes.size());
    dbgBounds("CLIP: void top node", topNode);
    dbgBounds("CLIP: voidShadow near clone", shadow->node.get());
    dbgBounds("CLIP: voidShadow far clone", shadow->nodeFar.get());
    dbgBinLeaks("voidNear", shadow->node.get());
    dbgBinLeaks("voidFar", shadow->nodeFar.get());
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

    osg::Node *src = shadow->targetTopNode.get() ? shadow->targetTopNode.get() : shadow->topNode.get();
    if (!src)
        return;
    shadow->node = src->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
    if (!shadow->node.get())
        return;
    pruneShadowClones(shadow->node.get());
    neutralizePagedLODTree(shadow->node.get());
    shadow->pagedLODSig = PagedLODTreeSignature(src);
    shadow->node->setName(name + "_ClipShadow");
    setShadowFlags(shadow->node.get());
    prepareVoidShadowNear(shadow->node.get());
    parent->addChild(shadow->node.get());

    shadow->nodeFar = src->clone(osg::CopyOp::DEEP_COPY_ALL)->asNode();
    if (!shadow->nodeFar.get())
        return;
    pruneShadowClones(shadow->nodeFar.get());
    neutralizePagedLODTree(shadow->nodeFar.get());
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

    // the object's current (replaced) top node and the sub-node it refers to
    osg::Node *top = resolveTop(shadow->topNode, name);
    if (!top)
        return;
    shadow->topNode = top;
    osg::Node *target = findNamedNode(name);
    if (!target || topObjectOf(target) != top)
        target = top;
    shadow->targetTopNode = target;
    std::vector<osg::Geode *> curGeodes;
    {
        GeodeAllVisitor gv(curGeodes);
        target->accept(gv);
    }
    shadow->geodes = curGeodes;
    // carry over the saved state of the geodes that survived the re-attach, so
    // a hidden paged void keeps its ORIGINAL attributes (not the hide
    // overrides) when its clones are rebuilt on a new LOD level; new geodes
    // get fresh uncaptured entries (their current state IS the original one)
    std::vector<GeodeSavedState> oldSaved = shadow->saved;
    shadow->saved.clear();
    shadow->saved.resize(shadow->geodes.size());
    for (size_t i = 0; i < shadow->geodes.size(); ++i)
    {
        shadow->saved[i].geode = shadow->geodes[i];
        for (const auto &st : oldSaved)
        {
            if (st.geode == shadow->geodes[i])
            {
                shadow->saved[i] = st;
                break;
            }
        }
    }
    shadow->parent = shadow->targetTopNode.get() ? (shadow->targetTopNode->getNumParents() > 0 ? shadow->targetTopNode->getParent(0)->asGroup() : nullptr) : nullptr;

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

        // the node the clones should mirror (and its geodes): the object
        // itself or the named sub-node (OBJ material group, VRML tile) it
        // refers to
        osg::Node *top = resolveTop(shadow->topNode, name);
        osg::Node *target = resolveMirrorNode(top, name);
        std::vector<osg::Geode *> curGeodes;
        if (target)
        {
            GeodeAllVisitor gv(curGeodes);
            target->accept(gv);
        }

        if (!top || curGeodes.empty())
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

        if (top != shadow->topNode.get() || target != shadow->targetTopNode.get())
        {
            // the object's top node was replaced (COVISE object update), or the
            // named sub-node moved/was replaced: re-clone and re-capture the
            // original state of the current geodes
            reattachVoidShadow(name);
            applyVoidVisibility(name);
            continue;
        }

        // the geode set drifted (a paged tile loaded or expired, or a partial
        // update changed the object's geodes): shadow->geodes may hold
        // pointers the pager has already freed, so re-clone before anything
        // touches them
        bool geodesDrifted = curGeodes.size() != shadow->geodes.size();
        if (!geodesDrifted)
            for (osg::Geode *g : curGeodes)
                if (std::find(shadow->geodes.begin(), shadow->geodes.end(), g) == shadow->geodes.end())
                {
                    geodesDrifted = true;
                    break;
                }
        if (geodesDrifted)
        {
            reattachVoidShadow(name);
            applyVoidVisibility(name);
            continue;
        }

        if (!shadow->pagedLODSig.empty() && shadow->pagedLODSig != PagedLODTreeSignature(target))
        {
            // a paged void object advanced to a new LOD level: the (static)
            // near/far clones would no longer mirror it, so rebuild them. The
            // live object is normally hidden while clipped, so a new geode is
            // hidden in the very frame it is loaded.
            reattachVoidShadow(name);
            applyVoidVisibility(name);
            continue;
        }

        // the mirror node's parent may have changed (a helper was inserted or
        // removed, or the object was re-parented) -> move the clones with it
        osg::Group *parent = target->getNumParents() > 0 ? target->getParent(0)->asGroup() : nullptr;
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

    // CLIP_DEBUG: periodic camera-cull-mask vs. shadow-node-mask diagnostic
    // (the void cull reports are silent when a clone is culled out by its node
    // mask, so the masks have to be compared explicitly)
    dbgReportCamera();

    // target: same handling
    if (m_targetName.empty() || !m_targetTopNode.get())
        return;

    osg::Node *top = resolveTop(m_targetTopNode, m_targetName);
    // the target name is either the top-level object's name (whole-object
    // target) or a named sub-node (a photogrammetry tile, an OBJ material
    // group); a sub-node that is gone means the target is gone
    osg::Node *target = resolveMirrorNode(top, m_targetName);
    std::vector<osg::Geode *> curTargetGeodes;
    if (target)
    {
        GeodeAllVisitor gv(curTargetGeodes);
        target->accept(gv);
    }
    if (!target || curTargetGeodes.empty())
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
        // the object was replaced by a COVISE update: the old subtree (and its
        // geodes) may already be freed, so purge the dead geodes from the
        // lists FIRST (reconcileTargetGeodes() does not touch the attributes
        // of geodes that are no longer in the tree), then re-derive the geode
        // set on the new top node and rebuild the shadow clone
        m_targetTopNode = top;
        reconcileTargetGeodes();
        rebuildTargetShadow();
        updateMainPass();
        return;
    }
    if (m_targetShadow && m_targetShadow->targetTopNode.get() && m_targetShadow->targetTopNode.get() != target)
    {
        // the target is a sub-node (e.g. a photogrammetry tile) and the node
        // it refers to was replaced (a paged reload of that tile): the old
        // geodes may already be freed, so reconcile the geode set (purge the
        // dead, save the new) before rebuilding and touching attributes
        reconcileTargetGeodes();
        rebuildTargetShadow();
        updateMainPass();
        return;
    }

    // keep the geode set in sync with the (possibly paged) live object: newly
    // loaded tiles join the set while still untouched (so their saved state is
    // the original one), expired tiles are dropped before their (freed) state
    // is touched again
    const bool geodesChanged = reconcileTargetGeodes();

    if (m_targetShadow && !m_targetPagedLODSig.empty() && m_targetPagedLODSig != PagedLODTreeSignature(target))
    {
        // a paged target advanced to a new LOD level (a tile was loaded or
        // expired): the (static) depth shadow would no longer mirror the live
        // target. Rebuild it on the new level (rebuildTargetShadow() deep-copies
        // the current children and neutralizes their PagedLODs); the geodes'
        // clip overrides are re-applied by updateMainPass() below, their
        // original state stays in m_targetSaved.
        rebuildTargetShadow();
        updateMainPass();
        return;
    }

    if (m_targetShadow && m_targetShadow->node.get())
    {
        osg::Group *parent = target->getNumParents() > 0 ? target->getParent(0)->asGroup() : nullptr;
        if (parent && parent != m_targetShadow->parent.get())
        {
            if (m_targetShadow->parent.get())
                m_targetShadow->parent->removeChild(m_targetShadow->node.get());
            m_targetShadow->parent = parent;
            m_targetShadow->parent->addChild(m_targetShadow->node.get());
        }
    }

    // a new tile may have loaded at the same LOD level (structure unchanged)
    // and joined the geode set; apply the main pass to the grown set (cheap
    // no-op for the geodes that already have the attributes). If nothing
    // changed, the main pass was already applied on the same set in the
    // previous frame -- skip the attribute churn
    if (geodesChanged)
        updateMainPass();

    // CLIP_DEBUG: periodic state snapshot so a failing session can be
    // diagnosed from the console alone
    if (clipDebug())
    {
        static int frame = 0;
        if (++frame % 600 != 0) // every 10 s at 60 fps
            return;
        osg::Geode *tg = m_targetGeodes.empty() ? nullptr : m_targetGeodes[0];
        if (tg)
        {
            osg::StateSet *ss = tg->getStateSet();
            osg::Stencil *st = ss ? (osg::Stencil *)ss->getAttribute(osg::StateAttribute::STENCIL) : nullptr;
            osg::Depth *dp = ss ? (osg::Depth *)ss->getAttribute(osg::StateAttribute::DEPTH) : nullptr;
            fprintf(stderr, "CLIP state: targetGeodes=%d geode[0]='%s' mask=0x%x stencil=%s depth=%s\n",
                    (int)m_targetGeodes.size(), tg->getName().c_str(), tg->getNodeMask(),
                    st ? (st->getFunction() == osg::Stencil::NOTEQUAL ? "NOTEQUAL" : (st->getFunction() == osg::Stencil::ALWAYS ? "ALWAYS" : "OTHER")) : "none",
                    dp ? "set" : "none");
        }
        if (m_targetShadow && m_targetShadow->node.get())
        {
            osg::Node *n = m_targetShadow->node.get();
            int parents = n->getNumParents();
            osg::Node *top = m_targetShadow->targetTopNode.get();
            fprintf(stderr, "CLIP state: targetShadow '%s' parents=%d sameParent=%d topParents=%d\n",
                    n->getName().c_str(), parents,
                    (parents && top && top->getNumParents() && top->getParent(0) == n->getParent(0)) ? 1 : 0,
                    top ? top->getNumParents() : -1);
        }
        for (const auto &kv : m_voidShadows)
            if (kv.second && kv.second->node.get())
                fprintf(stderr, "CLIP state: void '%s' near=%p far=%p geodes=%d shadow='%s'\n", kv.first.c_str(), kv.second->node.get(), kv.second->nodeFar.get(), (int)kv.second->geodes.size(),
                        kv.second->targetTopNode.get() ? kv.second->targetTopNode->getName().c_str() : "(null)");
    }
}

void ClipPlugin::preFrame()
{
    if (clipDebug())
        clipFrameCounter()++;
    syncSetup();
    syncShadows();
}

void ClipPlugin::dbgReportCamera()
{
    if (!clipDebug())
        return;
    int frame = clipFrameCounter();
    if (frame - m_dbgLastDiagFrame < 300)
        return; // every ~5 s at 60 fps
    m_dbgLastDiagFrame = frame;

    osg::Camera *cam = coVRConfig::instance()->channels.empty() ? nullptr : coVRConfig::instance()->channels[0].camera.get();
    if (!cam)
        return;
    const unsigned int camMask = cam->getCullMask();
    fprintf(stderr, "CLIP diag frame %d: cam='%s' cullMask=0x%x hideVoids=%d voidNames=%zu\n",
            frame, cam->getName().c_str(), camMask, (int)m_hideVoids, m_voidNames.size());
    for (const std::string &n : m_voidNames)
        fprintf(stderr, "  voidName='%s'\n", n.c_str());

    auto reportClone = [&](const char *what, osg::Node *node, osg::Node *topNode)
    {
        if (!node)
        {
            fprintf(stderr, "  %s: node=NULL (shadow not built!) topNode=%p\n", what, (void *)topNode);
            return;
        }
        const unsigned int nm = node->getNodeMask();
        const unsigned int topNm = topNode ? topNode->getNodeMask() : 0;
        const int parents = node->getNumParents();
        osg::Node *par = parents ? node->getParent(0) : nullptr;
        fprintf(stderr, "  %s: node=%p mask=0x%x topMask=0x%x parents=%d parent='%s' intersectsCamMask=%d\n",
                what, (void *)node, nm, topNm, parents,
                par ? par->getName().c_str() : "?",
                (nm & camMask) ? 1 : 0);
    };

    if (m_targetShadow)
    {
        reportClone("targetShadow", m_targetShadow->node.get(), m_targetShadow->topNode.get());
    }
    for (const auto &kv : m_voidShadows)
    {
        const Shadow *s = kv.second;
        if (!s)
            continue;
        reportClone(("voidNear[" + kv.first + "]").c_str(), s->node.get(), s->topNode.get());
        reportClone(("voidFar[" + kv.first + "]").c_str(), s->nodeFar.get(), s->topNode.get());
    }
    if (m_voidShadows.empty())
        fprintf(stderr, "  (NO void shadows in m_voidShadows -> createVoidShadow never succeeded)\n");

    // also dump the scene structure: every direct child of ObjectsRoot with its
    // node mask -- shows whether the clones exist as siblings at all
    osg::Group *root = cover->getObjectsRoot();
    if (root)
    {
        fprintf(stderr, "  ObjectsRoot children=%u:\n", root->getNumChildren());
        for (unsigned int i = 0; i < root->getNumChildren(); ++i)
        {
            osg::Node *c = root->getChild(i);
            if (!c)
                continue;
            fprintf(stderr, "    child[%u]='%s' mask=0x%x\n", i, c->getName().c_str(), c->getNodeMask());
        }
    }
}

void ClipPlugin::postFrame()
{
    if (!clipDebug() || m_dbgPosted)
        return;
    if (m_targetName.empty() || m_voidNames.empty() || !m_targetShadow || !m_targetShadow->node.get())
        return;
    m_dbgPosted = true;

    osg::Camera *cam = coVRConfig::instance()->channels.empty() ? nullptr : coVRConfig::instance()->channels[0].camera.get();
    if (!cam)
    {
        dbg("CLIP postFrame: no main camera\n");
        return;
    }
    dbg("CLIP postFrame: main cam '%s' renderOrder=%s clearMask=0x%x\n",
        cam->getName().c_str(),
        cam->getRenderOrder() == osg::Camera::PRE_RENDER ? "PRE" : (cam->getRenderOrder() == osg::Camera::POST_RENDER ? "POST" : "DEFAULT"),
        cam->getClearMask());
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
    // every target geode carries the LEQUAL depth and (while a void is active)
    // the stencil cut. This is called both on selection changes and per frame
    // for newly loaded (paged) tiles, so it applies the attributes on the
    // whole current set -- a cheap no-op for the geodes that already have them
    for (osg::Geode *geode : m_targetGeodes)
        if (geode)
            applyMainPass(geode->getOrCreateStateSet(), cut);
}

bool ClipPlugin::reconcileTargetGeodes()
{
    // keep m_targetGeodes / m_targetSaved in sync with the target's live
    // geodes. A paged target (photogrammetry terrain) loads new tiles while
    // the plugin is active and the pager can also EXPIRE tiles; the pager's
    // Group::removeChildren() frees an expired subtree immediately and the
    // plugin hooks are not involved. A raw geode pointer is only valid while
    // the geode is still in the live object's tree, so a geode that vanished
    // from it must be dropped from the lists BEFORE any of its attributes are
    // touched again.
    if (m_targetName.empty())
        return false;

    std::vector<osg::Geode *> currentGeodes;
    {
        osg::Node *top = resolveTop(m_targetTopNode, m_targetName);
        osg::Node *target = resolveMirrorNode(top, m_targetName);
        if (target)
        {
            GeodeAllVisitor gv(currentGeodes);
            target->accept(gv);
        }
    }
    std::set<osg::Geode *> alive(currentGeodes.begin(), currentGeodes.end());

    // purge geodes that are gone from the scene (expired tiles); their saved
    // entries are dropped as well so restoreTarget() never touches freed
    // memory. A purged geode is NOT restored first: the pager freed it
    // (Group::removeChildren dropped the last ref), so it must not be
    // touched at all -- its clip attributes die with it, which is fine, the
    // geometry is gone
    const size_t savedCount = m_targetSaved.size();
    bool changed = false;
    for (size_t i = 0; i < m_targetGeodes.size(); ++i)
    {
        osg::Geode *g = m_targetGeodes[i];
        if (g && alive.count(g))
            continue;
        if (!g)
            continue; // already dropped
        m_targetGeodes[i] = nullptr;
        if (i < savedCount)
            m_targetSaved[i].geode = nullptr;
        changed = true;
    }

    // append geodes that appeared since the last call (newly loaded tiles).
    // They are not in the saved list yet, and no clip override is applied to
    // them -- their current state IS the original state, so it can be saved.
    for (osg::Geode *g : currentGeodes)
    {
        if (std::find(m_targetGeodes.begin(), m_targetGeodes.end(), g) != m_targetGeodes.end())
            continue;
        m_targetGeodes.push_back(g);
        GeodeSavedState st;
        st.geode = g;
        osg::StateSet *ss = g->getOrCreateStateSet();
        st.depth = ss->getAttribute(osg::StateAttribute::DEPTH);
        st.stencil = ss->getAttribute(osg::StateAttribute::STENCIL);
        m_targetSaved.push_back(st);
        changed = true;
    }

    // belt & braces: all scene-graph mutations of the pager (attaching loaded
    // tiles, expiring old ones) happen in DatabasePager::updateTraverse, i.e.
    // on the main thread AFTER our preFrame() -- so nothing can be freed or
    // added while we run. The check only guards against a geode that died in
    // the same frame between the traversal and here; if that ever happens the
    // list is dropped and rebuilt from scratch next frame
    for (osg::Geode *g : m_targetGeodes)
        if (g && !alive.count(g))
        {
            m_targetGeodes.clear();
            m_targetSaved.clear();
            m_targetSavedState = false;
            return true;
        }

    if (std::none_of(m_targetGeodes.begin(), m_targetGeodes.end(), [](osg::Geode *g) { return g; }))
        m_targetSavedState = false;
    else
        m_targetSavedState = true;

    return changed;
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
        // the name the user should see (and that the plugin resolves again):
        // the name of the picked node itself for named sub-nodes (a
        // photogrammetry tile, an OBJ material group), else the top-level
        // object name
        std::string name = displayNameOf(geode);
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

    // a replaced or deleted object removes its geodes from the scene; the
    // stale geode pointers are dropped now (removeObject() may have already
    // cleaned up). syncShadows() re-resolves by name and re-attaches once a
    // replacement appears again.
    if (auto *geode = dynamic_cast<osg::Geode *>(realNode))
    {
        for (size_t i = 0; i < m_targetGeodes.size(); ++i)
        {
            if (m_targetGeodes[i] == geode)
            {
                m_targetGeodes[i] = nullptr;
                if (i < m_targetSaved.size())
                    m_targetSaved[i].geode = nullptr; // restoreTarget() must not touch it
            }
        }
        for (auto &kv : m_voidShadows)
            if (kv.second)
                for (size_t i = 0; i < kv.second->geodes.size(); ++i)
                    if (kv.second->geodes[i] == geode)
                        kv.second->geodes[i] = nullptr; // force re-attach on the next frame
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
    if (vit == m_voidNames.end())
    {
        // the removed node may be the top-level object while the stored void
        // name is one of its sub-nodes (e.g. an OBJ material group)
        osg::Node *obj = topNodeOf(name);
        if (obj)
            for (auto it = m_voidNames.begin(); it != m_voidNames.end(); ++it)
            {
                osg::Node *vn = findNamedNode(*it);
                if (vn && topObjectOf(vn) == obj)
                {
                    vit = it;
                    break;
                }
            }
    }
    if (vit != m_voidNames.end())
    {
        const std::string voidName = *vit;
        m_voidNames.erase(vit);
        removeVoidShadow(voidName);
        updateVoidsUi();
        updateMainPass();
    }

    bool targetGone = (m_targetName == name);
    if (!targetGone && m_targetTopNode.get() && m_targetTopNode.get() == topNodeOf(name))
        targetGone = true;
    if (targetGone)
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
