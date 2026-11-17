/*
//  Author : Marko Djuric
//  Date   : 28.10.2026

Cuts axis-aligned, world-space rectangular voids out of one or more target
objects (e.g. photogrammetry terrain). This is v2 of the plugin: instead of
the old GL stencil/depth-shadow machinery (clones of void objects, three
pre-pass render bins), the cut is done per pixel in a GLSL 150 shader that is
injected into every geode of a clipped target:

  - the vertex shader passes the vertex's *root-space* position and its
    normal through,
  - the fragment shader tests that position against up to 16 axis-aligned
    boxes (fed as `clipBoxMin[i]` / `clipBoxMax[i]` uniforms) and discards
    the fragments that fall inside any box.

Discarding (rather than fading) keeps the target in the opaque render bin,
so depth writing and sorting stay as they were and whatever is behind the
hole shows through.

The boxes are first-class scene nodes (`ClipBoxNode`, a `MatrixTransform`
with ghost cube geometry) under the ObjectsRoot, so they are ordinary
movable/selectable objects:

  - auto mode: picking an object creates a box matching its world AABB,
  - manual mode: a default 10 x 2 x 10 box is created with its bottom corner
    at the current intersection hit point.

A box's AABB is maintained from its world transform every frame; the shader
uniforms (shared `clipBox*` uniforms + a per-geode `clipWorldMatrix`) are
refreshed in `preFrame()`. Because the boxes live in the same space as the
objects, moving a box (or the target) with the Move plugin updates the cut
automatically.

The injected fragment shader takes over the fixed-function pipeline, so it
reproduces the target's own look: material (or vertex color) times an
optional base texture, lit by the scene headlight (ambient + Lambert).
Everything the shader overrides is saved per geode and restored exactly when
the target is removed or the plugin is destroyed.

Multiple targets and multiple boxes are supported; the setup (targets +
boxes) can be stored in the plugin's config file and re-applied on (re)load.
*/

#ifndef CLIP_PLUGIN_H
#define CLIP_PLUGIN_H

#include <cover/ui/Owner.h>
#include <cover/coVRPlugin.h>
#include <cover/coVRSelectionManager.h>
#include <OpenConfig/value.h>
#include <OpenConfig/array.h>

#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Material>
#include <osg/Depth>
#include <osg/Texture>
#include <osg/Uniform>
#include <osg/Program>

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
class EditField;
class Group;
class Label;
class Menu;
}
}

using namespace opencover;

namespace osg
{
class Group;
}

class ClipBoxNode;

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
        BoxFromObject
    };

    // a clipped target: its top-level object node + the node that is
    // actually clipped (the whole object or a named sub-node such as a
    // photogrammetry tile) + the geodes the clip shader is applied to
    struct Target
    {
        std::string name;
        osg::ref_ptr<osg::Node> topNode;
        osg::ref_ptr<osg::Node> clipNode; // geodes are collected under this one
        std::vector<osg::Geode *> geodes; // live geodes (non-owning)
    };

    // ---- scene-graph helpers (helper-wrapped / paged objects) ----
    void setMode(Mode m);

    // the (top-level) object a node belongs to: the highest ancestor (skipping
    // selection/move helpers) whose parent is the ObjectsRoot; nullptr if none
    osg::Node *topObjectOf(osg::Node *node) const;
    // for a direct child of the objects root: the real object inside a helper
    // (if wrapped), else the child itself
    osg::Node *innerObjectOf(osg::Node *child) const;
    // match a configured name against a top-level node name (file-loaded
    // objects are named by their URL, so basenames and prefixes are compared)
    static bool matchName(const std::string &cfgName, const std::string &nodeName);
    // the top-level node of an object name (helper-aware); nullptr if not found
    osg::Node *topNodeOf(const std::string &name) const;
    // a geode carrying (or inside) the given name; nullptr if not found
    osg::Geode *findGeodeForName(const std::string &name) const;
    // the node that is actually clipped for a target name (the top-level
    // object itself, or the named sub-node carrying the name)
    osg::Node *findClipNode(const std::string &name) const;
    // the display name of a picked node: its top-level object name, or the
    // name of the picked node itself when it is a named sub-node
    std::string displayNameOf(osg::Node *node) const;
    // a geode's world AABB in ObjectsRoot space (drawable bounding boxes
    // transformed through the full parent chain)
    static void geodeWorldAABB(const osg::Geode *geode, osg::Vec3 &min, osg::Vec3 &max);
    // the current intersection hit point (already in COVER space); false if
    // no interactor has a current hit
    static bool currentHitPos(osg::Vec3 &pos);

    // ---- targets ----
    void addTarget(const std::string &name);
    void removeTarget(const std::string &name);
    // keep each target's geode list in sync with the live scene (paged tiles
    // are added/expired while the plugin runs); applies the clip shader to new
    // geodes and restores geodes that disappeared
    void reconcileTargets();
    bool reconcileTargetGeodes(Target &target);

    // ---- boxes ----
    // create a box matching the picked object's world AABB
    void boxFromObject(osg::Geode *geode, const std::string &name);
    // create a default box with its bottom corner at the given point
    void addBoxAt(const osg::Vec3 &point);
    // create a box with an explicit world AABB (root space)
    void addBoxAABB(const osg::Vec3 &min, const osg::Vec3 &max);
    void removeLastBox();
    void setBoxesVisible(bool visible);
    void clearAll();
    // the currently selected box (first box in the Selection plugin's
    // selection); nullptr if none is selected
    ClipBoxNode *selectedBox() const;
    // show the `Resize box` fields for `box` (or hide them when null) and
    // sync the X/Y/Z size values with the box's current world extent
    void updateBoxSizeUi(ClipBoxNode *box = nullptr);
    // set the box's world extent to the given size, keeping its center fixed
    // (called from the resize X/Y/Z fields)
    void setBoxSize(ClipBoxNode *box, const osg::Vec3 &size);
    static bool parseBoxString(const std::string &s, osg::Vec3 &lo, osg::Vec3 &hi);

    // ---- clip shader injection ----
    // attach the shared clip program + uniforms (and restore later) to one
    // geode, or detach it and restore the original state
    void applyClipToGeode(osg::Geode *geode);
    void removeClipFromGeode(osg::Geode *geode);
    // the material/vertex-color/texture/normal properties of a geode's
    // drawables, sampled once when the clip is applied
    static bool geodeHasVertexColor(const osg::Geode *geode);
    static osg::Texture *geodeTexture(osg::Geode *geode, int &unit);
    static bool geodeHasNormal(const osg::Geode *geode);
    // per-frame: rebuild the shared box arrays and light uniforms from the
    // current box list / scene lighting, and the per-geode world matrices
    void syncClipUniforms();

    // ---- setup (config) ----
    void loadSetup();
    void syncSetup();
    // keep the configured `boxes` in sync with the scene: configured boxes
    // are created, moved/resized in place (matched by center proximity), and
    // boxes that used to be configured but no longer are removed. Boxes the
    // user created manually are kept (they are not part of the config).
    void syncBoxes();

    void updateUi();

    Mode m_mode = Mode::None;
    bool m_manual = false; // user interacted -> config no longer auto-applies

    std::vector<Target> m_targets;
    std::map<std::string, size_t> m_targetIndex; // target name -> index into m_targets

    std::vector<osg::ref_ptr<ClipBoxNode> > m_boxes;
    bool m_showBoxes = true;

    // shared shader program (one instance, attached to all clipped geodes)
    osg::ref_ptr<osg::Program> m_program;
    // per-frame shared uniforms (same instance attached to every geode)
    osg::ref_ptr<osg::Uniform> m_boxMin;
    osg::ref_ptr<osg::Uniform> m_boxMax;
    osg::ref_ptr<osg::Uniform> m_boxCount;
    osg::ref_ptr<osg::Uniform> m_lightPos;
    osg::ref_ptr<osg::Uniform> m_lightDiffuse;
    osg::ref_ptr<osg::Uniform> m_lightAmbient;

    // per-geode state: everything the clip shader overrides is saved here so
    // it can be restored exactly (material, texture, depth, vertex color)
    struct GeodeClipSave
    {
        osg::ref_ptr<osg::Uniform> worldMatrix; // this geode's clipWorldMatrix
        osg::ref_ptr<osg::Uniform> matDiffuse; // this geode's clipMatDiffuse
        osg::ref_ptr<osg::Uniform> matAmbient; // this geode's clipMatAmbient
        osg::ref_ptr<osg::Uniform> texUnitUniform; // this geode's clipBaseTexture
        osg::ref_ptr<osg::Uniform> hasTexture; // this geode's clipHasTexture
        osg::ref_ptr<osg::Uniform> hasNormalUniform; // this geode's clipHasNormal
        osg::ref_ptr<osg::Uniform> hasVertexColor; // this geode's clipHasVertexColor
        osg::ref_ptr<osg::Material> material; // original material (if any)
        osg::ref_ptr<osg::Texture> texture; // original texture at `texUnit` (if any)
        int texUnit = 0;
        bool hadTextureAttribute = false; // texture was bound at the geode level
        bool wasDrawableTexture = false; // texture was bound on a drawable (not the geode)
        osg::ref_ptr<osg::Depth> depth; // original depth (if any)
        bool hadDepth = false;
        osg::Vec4 diffuse{1.0f, 1.0f, 1.0f, 1.0f};
        osg::Vec4 ambient{1.0f, 1.0f, 1.0f, 1.0f};
        bool vertexColor = false;
        bool hasNormal = true;
    };
    std::map<osg::Geode *, GeodeClipSave> m_geodeClip; // geode -> clip is applied

    ui::Menu *m_menu = nullptr;
    ui::Group *m_group = nullptr;
    ui::Button *m_targetBtn = nullptr;
    ui::Button *m_boxBtn = nullptr;
    ui::Button *m_addBoxBtn = nullptr;
    ui::Button *m_removeBoxBtn = nullptr;
    ui::Button *m_clearBtn = nullptr;
    ui::Button *m_showBoxesBtn = nullptr;
    ui::Label *m_statusLabel = nullptr;

    // resize box: numeric world size (X/Y/Z) of the currently selected box.
    // The fields track the box selected with the Selection plugin; editing
    // them resizes that box (center kept fixed).
    ui::Label *m_boxSizeLabel = nullptr;
    ui::EditField *m_boxSizeX = nullptr;
    ui::EditField *m_boxSizeY = nullptr;
    ui::EditField *m_boxSizeZ = nullptr;
    // the box the fields currently describe + a re-entrancy guard so
    // refreshing the fields does not re-trigger their own callbacks
    ClipBoxNode *m_boxSizeUiBox = nullptr;
    bool m_boxSizeUpdating = false;
    // last world AABB pushed into the fields: while the tracked box does not
    // move (e.g. while the user is typing), the fields are not overwritten
    osg::Vec3 m_boxSizeUiLo{0, 0, 0}, m_boxSizeUiHi{0, 0, 0};
    bool m_boxSizeUiHasAABB = false;

    std::unique_ptr<ConfigStringArray> m_cfgTargets;
    std::unique_ptr<ConfigStringArray> m_cfgBoxes;
    std::unique_ptr<ConfigBool> m_cfgShowBoxes;
    // names of the boxes that were (re)created/updated from the config; only
    // those are removed when they disappear from the config (boxes the user
    // created manually are kept)
    std::set<std::string> m_cfgBoxApplied;
};

#endif
