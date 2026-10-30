/*
//  Author : Marko Djuric
//  Date   : 28.10.2026

  Cuts holes into a target object (e.g. terrain) where selected void objects
  (e.g. buildings) overlap it. Implemented with the GL stencil buffer:

  - bin -2: a "shadow" clone of the target is rendered into the depth buffer
  - bin -1: "shadow" clones of all void objects render into the stencil buffer
            (depth-tested against the target shadow, so only the part of a void
            in front of / inside the target writes stencil)
  - bin 0 : the whole scene renders normally; the target additionally carries a
            stencil test (EQUAL 0) and is thus not drawn where the voids wrote.

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

    struct Shadow
    {
        std::string name;
        osg::ref_ptr<osg::Node> node; // clone inserted next to the original
        osg::ref_ptr<osg::Group> parent;
        osg::Geode *geode = nullptr; // non-owning; the geode the clone was taken from
        // the void geode's original rendering state, saved so hiding can be reverted
        osg::ref_ptr<osg::StateAttribute> savedColorMask;
        osg::ref_ptr<osg::StateAttribute> savedDepth;
        osg::ref_ptr<osg::StateAttribute> savedStencil;
        unsigned int savedMask = 0; // original node mask (0 = not captured)
        bool appliedHidden = false; // current visibility state of the geode
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
    // apply (or revert) the hide state on one void's geode
    void applyVoidVisibility(const std::string &name);
    // per-frame: keep the shadow clones in sync with their (possibly moved,
    // wrapped or replaced) objects and apply the hide state
    void syncShadows();

    // ---- shadow (stencil/depth pre-pass) management ----
    void removeTargetShadow();
    void rebuildTargetShadow(osg::Geode *geode);
    void removeVoidShadow(const std::string &name);
    void createVoidShadow(const std::string &name, osg::Geode *geode);
    // re-create the clone next to the object's current (replaced) geode
    void reattachVoidShadow(const std::string &name);
    void updateMainPass();
    void applyMainPass(osg::StateSet *ss, bool on);
    // put the target geode back to its original depth/stencil state (no clip)
    void restoreTarget();

    // ---- helpers ----
    // name of the top-level scene object a node belongs to
    std::string objectNameOf(osg::Node *node) const;
    // resolve a configured name/prefix to the object's top-level node, "" if not found
    std::string resolveName(const std::string &name) const;
    // find the geode of a top-level object (exact or basename/prefix match)
    osg::Geode *geodeOf(const std::string &name) const;
    // match a configured name against a top-level node name (file-loaded objects
    // are named by their URL, so the basename of the path is compared)
    static bool matchName(const std::string &cfgName, const std::string &nodeName);
    Shadow *shadowFor(const std::string &name);
    void updateVoidsUi();
    void refreshStatus();

    Mode m_mode = Mode::None;
    std::string m_targetName;
    osg::ref_ptr<osg::Geode> m_targetGeode;
    std::vector<std::string> m_voidNames;
    bool m_manual = false; // user has interacted -> config no longer auto-applies until "Load Setup"
    bool m_hideVoids = true; // void objects are hidden in the main pass but stay pickable

    Shadow *m_targetShadow = nullptr;
    std::map<std::string, Shadow *> m_voidShadows; // keyed by object name

    // per-object counters: frames in a row the object's geode has been absent
    // from the scene; a replaced object is gone for at most one frame, so a
    // few dozen frames means it was really deleted (file objects removed via
    // the SGBrowser do not fire the plugin remove hooks, so clean up here)
    std::map<std::string, int> m_missingFrames;
    int m_targetMissingFrames = 0;

    // the target geode's original depth/stencil attributes, saved before the
    // clip overrides are applied so they can be restored on remove/clear
    bool m_targetSavedState = false;
    osg::ref_ptr<osg::StateAttribute> m_targetSavedDepth;
    osg::ref_ptr<osg::StateAttribute> m_targetSavedStencil;

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
