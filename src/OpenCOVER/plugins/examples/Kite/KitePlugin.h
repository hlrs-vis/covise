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
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Quat>
#include <osg/ShapeDrawable>
#include <osg/Vec3>

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
    osg::Quat m_modelOffset;

    double m_rollSign = 1.0;
    double m_pitchSign = 1.0;
    double m_yawSign = 1.0;

    double m_rollOffsetDeg = 0.0;
    double m_pitchOffsetDeg = 0.0;
    double m_yawOffsetDeg = 0.0;

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
    static std::vector<osg::Vec3> sagCurvePoints(const osg::Vec3 &a, const osg::Vec3 &b,
                                                 int samples, float sagMeters);

    void createGround();
    void createGroundStation();
    osg::Vec3 groundStationWorld() const;
    void updateGroundAnchor();

    // Rope config
    bool m_ropeEnabled = true;

    int m_ropeSamplesMain = 28;
    int m_ropeSamplesBridle = 14;

    float m_ropeRadius = 0.015f;

    float m_ropeSagFactor = 0.06f;
    float m_ropeSagMax = 8.0f;

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
    bool m_useTwoStageKnot = true;
    float m_knotSplit_m = 1.2f;
    osg::Vec3 m_knotFrontLocal;
    osg::Vec3 m_knotRearLocal;

    std::vector<osg::Vec3> m_attachLocal;
    osg::Vec3 m_junctionLocal = osg::Vec3(0.f, 0.f, -0.2f);
    osg::Vec3 m_groundPos = osg::Vec3(0.f, 0.f, 0.f);

    // Rope as line geometry.
    osg::ref_ptr<osg::Geode> m_ropeGeode;
    osg::ref_ptr<osg::Geometry> m_tetherGeom;
    osg::ref_ptr<osg::Geometry> m_bridleGeom;
    osg::ref_ptr<osg::Vec3Array> m_tetherVerts;
    osg::ref_ptr<osg::Vec3Array> m_bridleVerts;

    float m_lineWidth = 2.0f;
    bool m_focusMode = true;
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
    float m_groundZOffset_m = 0.0f;
    float m_targetTether_m = 300.0f;
};

#endif // KITE_PLUGIN_H
