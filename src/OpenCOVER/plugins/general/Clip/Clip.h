/*
//  Author : Marko Djuric
//  Date   : 28.10.2026

  Cuts holes into a target object (e.g. terrain) where selected void objects
  (e.g. buildings) overlap it. Implemented with the GL stencil buffer.

  The goal is to remove only the part of the target that a void object
  *occupies* (its footprint), while the target *behind* the void stays visible.
  To tell "inside the void" apart from "hidden behind the void", the void
  contributes two stencil bits (a depth-slab test along the view ray):

  - bin -3: a "shadow" clone of the target is rendered into the depth buffer
  - bin -2: the void's *near* (front-facing) surface renders into the stencil
            buffer (depth-tested against the target shadow); where it is in
            front of the target it sets bit 0
  - bin -1: the void's *far* (back-facing) surface renders into the stencil
            buffer (depth-tested against the target shadow); where it is in
            front of the target it sets bit 1
  - bin 0 : the whole scene renders normally; the target carries a stencil test
            (NOT_EQUAL 1) and is therefore not drawn only where bit 0 is set
            and bit 1 is clear -- i.e. the target lies between the void's near
            and far surface (inside the void). Terrain that the void merely
            occludes (both bits set -> value 3) is kept.

  The shadows are deep copies of the whole top-level object (not just one
  geode), placed as siblings of that object. That keeps every tile/geode of a
  (possibly multi-geode, e.g. photogrammetry .osgb) object in place, follows
  its (helper) transforms, and lets the render-bin/stencil state be set once
  on the clone's root so it propagates to all children.

  Void objects are hidden in the main pass (color/depth write off) while
  remaining pickable, so they stay editable (Move, Selection, ...).

  No custom GLSL is required, which keeps the target's own shader (e.g. the
  heightmap terrain generator) intact.
*/

#ifndef CLIP_PLUGIN_H
#define CLIP_PLUGIN_H

#include <cover/ui/Owner.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRSelectionManager.h>
#include <OpenConfig/value.h>
#include <OpenConfig/array.h>

#include <osg/StateAttribute>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace opencover
{
namespace ui
{
class Button;
class Group;
class Label;
class Menu;
class SelectionList;
}
}

using namespace opencover;

class ClipPlugin : public coVRPlugin, public ui::Owner, public coSelectionListener
{
public:
    ClipPlugin();
    ~ClipPlugin() override;

    bool init() override;
    bool destroy() override;
    void preFrame() override;

    // coSelectionListener
    bool selectionChanged() override;
    bool pickedObjChanged() override;

    void addNodeFromPlugin(osg::Node *node, const RenderObject *ro, coVRPlugin *addingPlugin) override;
    void removeNode(osg::Node *node, bool isGroup, osg::Node *realNode) override;
    void removeObject(const char *objName, bool replaceFlag) override;

private:
    enum class Mode
    {
        None,
        PickTarget,
        AddVoid,
        RemoveVoid
    };

    struct GeodeSavedState
    {
        osg::Geode *geode = nullptr;
        osg::ref_ptr<osg::StateAttribute> colorMask;
        osg::ref_ptr<osg::StateAttribute> depth;
        osg::ref_ptr<osg::StateAttribute> stencil;
        unsigned int nodeMask = 0;
        bool captured = false; // true once the original state has been captured
    };

    struct Shadow
    {
        std::string name;
        osg::ref_ptr<osg::Node> topNode; // the object's current top-level node (helper-aware)
        std::vector<osg::Geode *> geodes; // all geodes of the object (non-owning)
        // the parent the clones live under (a sibling of `topNode`); re-parented
        // when a helper is inserted/removed above the object
        osg::ref_ptr<osg::Group> parent;
        // primary clone (a deep copy of the whole top node): for the target it is
        // the depth shadow; for a void it is the near-surface (front faces) clone
        osg::ref_ptr<osg::Node> node;
        // void only: the far-surface (back faces) clone, rendered in a different
        // bin to set the other stencil bit
        osg::ref_ptr<osg::Node> nodeFar;
        // the object geodes' original rendering state, saved so hiding can be
        // reverted (parallel to `geodes`)
        std::vector<GeodeSavedState> saved;
        int savedCount = 0; // how many geodes `saved` currently describes
        bool appliedHidden = false; // current visibility state of the geodes
    };

    // ---- target / void bookkeeping ----
    void setMode(Mode m);
    void pickTarget(osg::Geode *geode, const std::string &name);
    void pickVoid(osg::Geode *geode, const std::string &name);
    void removeVoid(const std::string &name);
    void clearAll();
    void loadSetup();
    // resolve the configured target/void prefixes to the loaded objects and apply
    void syncSetup();
    // toggle whether void objects are hidden in the main pass (still pickable)
    void setHideVoids(bool hide);
    // apply (or revert) the hide state on one void's geodes
    void applyVoidVisibility(const std::string &name);
    // per-frame: keep the shadow clones in sync with their (possibly moved,
    // wrapped or replaced) objects and apply the hide state
    void syncShadows();

    // ---- shadow (stencil/depth pre-pass) management ----
    void removeTargetShadow();
    void rebuildTargetShadow();
    void removeVoidShadow(const std::string &name);
    void createVoidShadow(const std::string &name, osg::Geode *geode);
    // (re-)create the near/far clones at the scene root for the whole object
    void makeVoidClones(Shadow *shadow, const std::string &name);
    // re-create the clones for the object's current (replaced) top node
    void reattachVoidShadow(const std::string &name);
    void updateMainPass();
    void applyMainPass(osg::StateSet *ss, bool on);
    // put the target geodes back to their original depth/stencil state (no clip)
    void restoreTarget();

    // ---- helpers ----
    // name of the top-level scene object a node belongs to
    std::string objectNameOf(osg::Node *node) const;
    // the top-level scene node a node belongs to (skips selection helpers)
    osg::Node *topNodeOf(osg::Node *node) const;
    // resolve a configured name/prefix to the object's top-level node name, "" if not found
    std::string resolveName(const std::string &name) const;
    // the top-level node of an object name, "" if not found
    osg::Node *topNodeOf(const std::string &name) const;
    // all geodes of an object (empty if the object is absent)
    std::vector<osg::Geode *> geodesOf(const std::string &name) const;
    // match a configured name against a top-level node name (file-loaded objects
    // are named by their URL, so the basename of the path is compared)
    static bool matchName(const std::string &cfgName, const std::string &nodeName);
    Shadow *shadowFor(const std::string &name);
    void updateVoidsUi();
    void refreshStatus();

    Mode m_mode = Mode::None;
    std::string m_targetName;
    osg::ref_ptr<osg::Node> m_targetTopNode;
    std::vector<osg::Geode *> m_targetGeodes; // the target's geodes the main pass is applied to
    std::vector<std::string> m_voidNames;
    bool m_manual = false; // user has interacted -> config no longer auto-applies until "Load Setup"
    bool m_hideVoids = true; // void objects are hidden in the main pass but stay pickable

    Shadow *m_targetShadow = nullptr;
    std::map<std::string, Shadow *> m_voidShadows; // keyed by object name

    // per-object counters: frames in a row the object's geodes have been absent
    // from the scene; a replaced object is gone for at most one frame, so a
    // few dozen frames means it was really deleted (file objects removed via
    // the SGBrowser do not fire the plugin remove hooks, so clean up here)
    std::map<std::string, int> m_missingFrames;
    int m_targetMissingFrames = 0;

    // the target geodes' original depth/stencil attributes, saved before the
    // clip overrides are applied so they can be restored on remove/clear
    bool m_targetSavedState = false;
    std::vector<GeodeSavedState> m_targetSaved;

    ui::Menu *m_menu = nullptr;
    ui::Group *m_group = nullptr;
    ui::Button *m_targetBtn = nullptr;
    ui::Button *m_addVoidBtn = nullptr;
    ui::Button *m_removeVoidBtn = nullptr;
    ui::Button *m_clearBtn = nullptr;
    ui::Button *m_hideVoidsBtn = nullptr;
    ui::Label *m_targetLabel = nullptr;
    ui::Label *m_voidsLabel = nullptr;
    ui::SelectionList *m_voidsList = nullptr;

    std::unique_ptr<ConfigString> m_cfgTarget;
    std::unique_ptr<ConfigStringArray> m_cfgVoids;
    std::unique_ptr<config::Value<bool>> m_cfgHide;
};

#endif
