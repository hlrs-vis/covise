/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef KITE_PLUGIN_H
#define KITE_PLUGIN_H

#include <cover/coVRPlugin.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/BoundingBox>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Quat>
#include <osg/ShapeDrawable>
#include <osg/Vec3>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>

#include <string>
#include <vector>

class KitePlugin : public opencover::coVRPlugin
{
public:
    KitePlugin();
    ~KitePlugin() override;

    bool init() override;
    void preFrame() override;

private:
    bool loadModel(const std::string &path);
    void parseCsv(const std::string &path);
    void updateTransform(int frameIndex);

    struct Frame
    {
        double t = 0.0;
        osg::Vec3 pos;
        double roll = 0.0;
        double pitch = 0.0;
        double yaw = 0.0;
    };

    osg::ref_ptr<osg::MatrixTransform> m_transform;
    osg::ref_ptr<osg::Node> m_model;
    std::vector<Frame> m_frames;
    std::string m_csvPath;

    bool m_useNedFrame = false;
    bool m_frameAuto = true;
    osg::Quat m_modelOffset;

    double m_rollSign = 1.0;
    double m_pitchSign = 1.0;
    double m_yawSign = 1.0;

    double m_rollOffsetDeg = 0.0;
    double m_pitchOffsetDeg = 0.0;
    double m_yawOffsetDeg = 0.0;

    bool m_smoothPose = true;
    double m_posSmoothAlpha = 0.25;
    double m_rotSmoothAlpha = 0.20;
    bool m_haveSmoothedPose = false;
    int m_lastFrameIndex = -1;
    osg::Vec3 m_smoothedPos;
    osg::Quat m_smoothedRot;

    bool m_despikeOrientation = true;
    bool m_despikeYawOnly = true;
    double m_despikeJumpDeg = 120.0;
    double m_despikeSettleDeg = 25.0;

    // ----------------------------
    // Rope rendering (procedural)
    // ----------------------------
    struct RopeSegment
    {
        osg::ref_ptr<osg::MatrixTransform> xform; // translate + rotate
        osg::ref_ptr<osg::MatrixTransform> scale; // scale radius + length
    };

    struct RopeVisual
    {
        osg::ref_ptr<osg::Group> root;
        std::vector<RopeSegment> segments;
    };

    void initRopes();
    void updateRopes();

    void ensureSegments(RopeVisual &rope, size_t needed);

    osg::Vec3 localToWorld(const osg::Vec3 &pLocal) const;
    static osg::Quat quatFromZToDir(const osg::Vec3 &dir);

    // --- Rope curve helpers ---
    static std::vector<osg::Vec3> sagCurvePoints(const osg::Vec3 &a, const osg::Vec3 &b,
                                                 int samples, float sagUnits);
    static void appendPolylineAsLines(osg::Vec3Array *arr, const std::vector<osg::Vec3> &pts);
    static std::vector<osg::Vec3> clipPolylineTailByLength(const std::vector<osg::Vec3> &pts, float tailLenUnits);
    void addSaggedLine(osg::Vec3Array *arr,
                       const osg::Vec3 &a,
                       const osg::Vec3 &b,
                       int samples,
                       float sagUnits) const;

    // --- Anchor snapping (to attach bridles to the mesh) ---
    void snapBridleAnchorsToSurface(const osg::BoundingBox &bb);
    bool snapPointToModelSurfaceMultiAxis(osg::Node *model, const osg::Vec3 &pModel,
                                          float rayUp, float rayDown,
                                          osg::Vec3 &hitModel) const;

    void createGround();
    void createGroundStation();
    osg::Vec3 groundStationWorld() const;
    void updateGroundAnchor();

    // Rope config
    bool m_ropeEnabled = true;

    int m_ropeSamplesMain = 32;
    int m_ropeSamplesBridle = 10;

    float m_ropeRadius = 0.015f;

    float m_ropeSagFactor = 0.1f;
    float m_ropeSagMax = 6.0f;

    // ----- Tether visualization tuning -----
    bool m_tetherFade = true; // draw far part faint, near part strong
    float m_tetherNearLen_m = 80.0f; // last N meters drawn strong (near the kite)

    float m_tetherSagBoost = 4.0f; // extra sag for long tether segments
    float m_tetherSagMaxLong_m = 40.0f; // cap for long-tether sag (meters)

    // separate styles for far/near tether
    float m_tetherLineWidthNear = 3.0f;
    float m_tetherLineWidthFar = 1.0f;
    float m_tetherAlphaNear = 0.90f;
    float m_tetherAlphaFar = 0.15f;

    osg::Vec3 m_groundRefCsv = osg::Vec3(0.f, 0.f, 0.f);
    bool m_haveGroundRef = false;
    float m_worldScale = 1.0f;

    // --- Physical sizing based on TU Delft V3 numbers (meters) ---
    float m_span_m = 8.32f;
    float m_chord_m = 2.63f;
    float m_bridleHeight_m = 9.60f;

    // CoG percentages from TU Delft table
    float m_cogX_frac = 0.54f;
    float m_cogZ_frac = 0.343f;

    // Derived: model-units per meter
    float m_unitsPerMeter = 1.0f;

    // Bridle layout
    int m_bridleN_span = 10;
    bool m_twoRows = true;
    float m_frontRow_frac = 0.25f;
    float m_rearRow_frac = 0.75f;
    float m_underside_frac = 0.05f;

    // Key point (knot / KCU attach) in kite-local coords
    osg::Vec3 m_knotLocal;
    osg::Vec3 m_cogLocal;
    bool m_haveCogLocal = false;
    bool m_knotFollowTether = true;
    bool m_useTwoStageKnot = true;
    float m_knotSplit_m = 1.2f;
    osg::Vec3 m_knotFrontLocal;
    osg::Vec3 m_knotRearLocal;

    std::vector<osg::Vec3> m_attachLocal;
    osg::Vec3 m_junctionLocal = osg::Vec3(0.f, 0.f, -0.2f);
    osg::Vec3 m_groundPos = osg::Vec3(0.f, 0.f, 0.f);
    osg::BoundingBox m_modelBB;
    bool m_haveModelBB = false;

    // optional: visualize-safe clamp
    bool m_clampAboveGround = true; // config toggle later if you want
    float m_groundZUnits = 0.f; // your ground plane height in *units*
    float m_globalPosScale = 1.f; // optional global scaling (instead of meanR trick)

    // Rope as line geometry.
    osg::ref_ptr<osg::Geode> m_ropeGeode;
    // --- Tether: draw full + focus tail separately ---
    osg::ref_ptr<osg::Geometry> m_tetherFullGeom;
    osg::ref_ptr<osg::Vec3Array> m_tetherFullVerts;

    osg::ref_ptr<osg::Geometry> m_tetherGeom;
    osg::ref_ptr<osg::Geometry> m_bridleGeom;
    osg::ref_ptr<osg::Vec3Array> m_tetherVerts;
    osg::ref_ptr<osg::Vec3Array> m_bridleVerts;

    osg::ref_ptr<osg::Vec4Array> m_tetherFullColors;
    osg::ref_ptr<osg::Vec4Array> m_tetherFocusColors;
    osg::ref_ptr<osg::Vec4Array> m_bridleColors;

    bool m_drawFullTether = true; // always draw a faint full tether
    float m_fullTetherAlpha = 0.25f; // faint
    float m_focusTetherAlpha = 1.0f; // bright

    // --- Anchor snapping ---
    bool m_snapAnchorsToSurface = true;
    float m_anchorLift_m = 0.03f;
    float m_snapRayUp_m = 2.0f;
    float m_snapRayDown_m = 4.0f;

    float m_lineWidth = 2.0f;
    bool m_focusMode = false;
    float m_focusTetherLen_m = 40.0f;

    // Bridle layout params (fractions of bbox).
    int m_numBridles = 10;
    float m_attachYMinFrac = 0.15f;
    float m_attachYMaxFrac = 0.85f;
    float m_attachZFromMaxFrac = 0.02f;
    float m_attachXFrac = 0.50f;

    float m_junctionXFrac = 0.50f;
    float m_junctionZFromMaxFrac = 0.30f;

    osg::ref_ptr<osg::Group> m_ropeRoot;
    RopeVisual m_mainRope;
    std::vector<RopeVisual> m_bridleRopes;

    osg::ref_ptr<osg::Geode> m_groundGeode;
    osg::ref_ptr<osg::Geode> m_stationGeode;
    osg::ref_ptr<osg::MatrixTransform> m_groundXform;

    bool m_showGround = true;
    float m_groundSize_m = 600.0f;
    bool m_groundSizeAuto = true;
    float m_groundZOffset_m = 0.0f;
    float m_targetTether_m = 300.0f;

    // Optional geodata (terrain/VRML/etc)
    std::string m_geoModelPath;
    osg::ref_ptr<osg::MatrixTransform> m_geoXform;
    osg::ref_ptr<osg::Node> m_geoModel;
    osg::Vec3 m_geoPos = osg::Vec3(0.f, 0.f, 0.f);
    float m_geoScale = 1.0f;
    float m_geoYawDeg = 0.0f;
    bool m_geoAutoCenter = false;
    bool m_geoAutoGround = false;
    osg::BoundingBox m_geoBBox;
    bool m_haveGeoBBox = false;
};

#endif // KITE_PLUGIN_H
