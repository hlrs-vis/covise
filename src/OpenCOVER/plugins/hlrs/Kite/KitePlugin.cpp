/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KitePlugin.h"

#include <config/CoviseConfig.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>

#include <osg/Geode>
#include <osg/Group>
#include <osg/Array>
#include <osg/Depth>
#include <osg/LineWidth>
#include <osg/Matrix>
#include <osg/Shape>           // <-- FIX: Cylinder is declared here
#include <osg/ShapeDrawable>   // <-- FIX
#include <osg/StateSet>
#include <osg/Vec4>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>
#include <osgDB/ReadFile>
#include <osgDB/Options>
#include <osgDB/Registry>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <osg/ComputeBoundsVisitor>
#include <osg/BoundingBox>

using namespace covise;
using namespace opencover;

KitePlugin::KitePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

KitePlugin::~KitePlugin()
{
    if (m_transform && m_transform->getNumParents() > 0)
        m_transform->getParent(0)->removeChild(m_transform);

    if (m_ropeRoot && m_ropeRoot->getNumParents() > 0)
        m_ropeRoot->getParent(0)->removeChild(m_ropeRoot);

    if (m_groundXform && m_groundXform->getNumParents() > 0)
        m_groundXform->getParent(0)->removeChild(m_groundXform);
    if (m_groundGeode && m_groundGeode->getNumParents() > 0)
        m_groundGeode->getParent(0)->removeChild(m_groundGeode);
    if (m_stationGeode && m_stationGeode->getNumParents() > 0)
        m_stationGeode->getParent(0)->removeChild(m_stationGeode);
    if (m_geoXform && m_geoXform->getNumParents() > 0)
        m_geoXform->getParent(0)->removeChild(m_geoXform);
}

static osg::BoundingBox computeLocalBBox(osg::Node *n)
{
    osg::ComputeBoundsVisitor cbv;
    n->accept(cbv);
    return cbv.getBoundingBox();
}

static osg::Matrix geoTransformMatrix(const osg::Vec3 &pos, float scale, float yawDeg)
{
    const double d2r = M_PI / 180.0;
    return osg::Matrix::scale(osg::Vec3(scale, scale, scale)) *
           osg::Matrix::rotate(yawDeg * d2r, osg::Vec3(0, 0, 1)) *
           osg::Matrix::translate(pos);
}


bool KitePlugin::init()
{
    auto modelEntry = coCoviseConfig::getEntry("value", "COVER.Plugin.KitePlugin.Model", "");
    m_csvPath = coCoviseConfig::getEntry("value", "COVER.Plugin.KitePlugin.CSV", "");

    auto toLowerCopy = [](std::string s) {
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        return s;
    };
    auto applyFrameMode = [&](const std::string &mode) {
        const std::string val = toLowerCopy(mode);
        if (val == "ned")
        {
            m_useNedFrame = true;
            m_frameAuto = false;
            return true;
        }
        if (val == "enu")
        {
            m_useNedFrame = false;
            m_frameAuto = false;
            return true;
        }
        if (val == "auto")
        {
            m_frameAuto = true;
            return true;
        }
        return false;
    };

    const std::string nedEntry = coCoviseConfig::getEntry("value", "COVER.Plugin.KitePlugin.NED", "");
    if (!nedEntry.empty())
    {
        m_useNedFrame = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.NED", m_useNedFrame);
        m_frameAuto = false;
    }
    const std::string frameEntry = coCoviseConfig::getEntry("value", "COVER.Plugin.KitePlugin.Frame", "");
    if (!frameEntry.empty())
        applyFrameMode(frameEntry);
    m_rollSign = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.RollSign", m_rollSign);
    m_pitchSign = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.PitchSign", m_pitchSign);
    m_yawSign = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.YawSign", m_yawSign);
    m_rollOffsetDeg = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.RollOffset", m_rollOffsetDeg);
    m_pitchOffsetDeg = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.PitchOffset", m_pitchOffsetDeg);
    m_yawOffsetDeg = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.YawOffset", m_yawOffsetDeg);
    m_smoothPose = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.SmoothPose", m_smoothPose);
    m_posSmoothAlpha = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.PositionSmoothing", (float)m_posSmoothAlpha);
    m_rotSmoothAlpha = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.RotationSmoothing", (float)m_rotSmoothAlpha);
    m_despikeOrientation = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.DespikeOrientation", m_despikeOrientation);
    m_despikeYawOnly = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.DespikeYawOnly", m_despikeYawOnly);
    m_despikeJumpDeg = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.DespikeJumpDeg", (float)m_despikeJumpDeg);
    m_despikeSettleDeg = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.DespikeSettleDeg", (float)m_despikeSettleDeg);
    m_knotFollowTether = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.KnotFollowTether", m_knotFollowTether);

    // Rope config (optional)
    m_ropeEnabled = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.RopeEnabled", m_ropeEnabled);
    // m_ropeRadius = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.RopeRadius", m_ropeRadius);
    m_ropeRadius = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.RopeRadius", m_ropeRadius);
    if (m_ropeRadius <= 0.f)
        m_ropeRadius = 0.05f; // force visible default

    m_ropeSagFactor = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.RopeSagFactor", m_ropeSagFactor);
    m_ropeSagMax = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.RopeSagMax", m_ropeSagMax);
    m_ropeSamplesMain = (int)coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.RopeSamplesMain", (float)m_ropeSamplesMain);
    m_ropeSamplesBridle = (int)coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.RopeSamplesBridle", (float)m_ropeSamplesBridle);

    m_drawFullTether = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.DrawFullTether", m_drawFullTether);
    m_fullTetherAlpha = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.FullTetherAlpha", m_fullTetherAlpha);
    m_focusTetherAlpha = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.FocusTetherAlpha", m_focusTetherAlpha);

    m_tetherFade = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.TetherFade", m_tetherFade);
    m_tetherNearLen_m = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.TetherNearLen_m", m_tetherNearLen_m);

    m_tetherSagBoost = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.TetherSagBoost", m_tetherSagBoost);
    m_tetherSagMaxLong_m = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.TetherSagMaxLong_m", m_tetherSagMaxLong_m);

    m_tetherLineWidthNear = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.TetherLineWidthNear", m_tetherLineWidthNear);
    m_tetherLineWidthFar = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.TetherLineWidthFar", m_tetherLineWidthFar);

    m_tetherAlphaNear = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.TetherAlphaNear", m_tetherAlphaNear);
    m_tetherAlphaFar = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.TetherAlphaFar", m_tetherAlphaFar);

    m_snapAnchorsToSurface = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.SnapAnchorsToSurface", m_snapAnchorsToSurface);
    m_anchorLift_m = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.AnchorLift_m", m_anchorLift_m);
    m_snapRayUp_m = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.SnapRayUp_m", m_snapRayUp_m);
    m_snapRayDown_m = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.SnapRayDown_m", m_snapRayDown_m);

    m_groundPos.x() = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.GroundPosX", m_groundPos.x());
    m_groundPos.y() = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.GroundPosY", m_groundPos.y());
    m_groundPos.z() = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.GroundPosZ", m_groundPos.z());
    m_showGround = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.ShowGround", m_showGround);
    m_worldScale = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.WorldScale", m_worldScale);
    m_targetTether_m = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.TargetTether_m", m_targetTether_m);
    const std::string groundSizeEntry = coCoviseConfig::getEntry("value", "COVER.Plugin.KitePlugin.GroundSize_m", "");
    if (!groundSizeEntry.empty())
    {
        m_groundSize_m = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.GroundSize_m", m_groundSize_m);
        m_groundSizeAuto = false;
    }
    m_focusMode = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.FocusMode", m_focusMode);
    m_focusTetherLen_m = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.FocusTetherLen_m", m_focusTetherLen_m);

    m_geoModelPath = coCoviseConfig::getEntry("value", "COVER.Plugin.KitePlugin.GeoModel", "");
    m_geoPos.x() = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.GeoPosX", m_geoPos.x());
    m_geoPos.y() = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.GeoPosY", m_geoPos.y());
    m_geoPos.z() = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.GeoPosZ", m_geoPos.z());
    m_geoScale = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.GeoScale", m_geoScale);
    m_geoYawDeg = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.GeoYawDeg", m_geoYawDeg);
    m_geoAutoCenter = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.GeoAutoCenter", m_geoAutoCenter);
    m_geoAutoGround = coCoviseConfig::isOn("COVER.Plugin.KitePlugin.GeoAutoGround", m_geoAutoGround);

    m_junctionLocal.x() = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.JunctionLocalX", m_junctionLocal.x());
    m_junctionLocal.y() = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.JunctionLocalY", m_junctionLocal.y());
    m_junctionLocal.z() = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.JunctionLocalZ", m_junctionLocal.z());

    auto envToBool = [](const char *name, bool value) {
        if (const char *env = std::getenv(name))
        {
            std::string val(env);
            std::transform(val.begin(), val.end(), val.begin(),
                           [](unsigned char c){ return std::tolower(c); });
            if (val == "1" || val == "true" || val == "yes" || val == "on")
                value = true;
            else if (val == "0" || val == "false" || val == "no" || val == "off")
                value = false;
        }
        return value;
    };
    auto envToDouble = [](const char *name, double value) {
        if (const char *env = std::getenv(name))
        {
            char *end = nullptr;
            double v = std::strtod(env, &end);
            if (end && end != env)
                value = v;
        }
        return value;
    };
    auto envToFloat = [](const char *name, float value) {
        if (const char *env = std::getenv(name))
        {
            char *end = nullptr;
            double v = std::strtod(env, &end);
            if (end && end != env)
                value = (float)v;
        }
        return value;
    };

    if (modelEntry.empty())
    {
        modelEntry = coCoviseConfig::getEntry("value", "COVER.Plugin.KitePlugin.Model", "");
    }
    if (m_csvPath.empty())
    {
        m_csvPath = coCoviseConfig::getEntry("value", "COVER.Plugin.KitePlugin.CSV", "");
    }
    if (m_geoModelPath.empty())
    {
        m_geoModelPath = coCoviseConfig::getEntry("value", "COVER.Plugin.KitePlugin.GeoModel", "");
    }

    if (std::getenv("KITE_NED"))
    {
        m_useNedFrame = envToBool("KITE_NED", m_useNedFrame);
        m_frameAuto = false;
    }
    if (const char *env = std::getenv("KITE_FRAME"))
        applyFrameMode(env);

    m_rollSign = envToDouble("KITE_ROLL_SIGN", m_rollSign);
    m_pitchSign = envToDouble("KITE_PITCH_SIGN", m_pitchSign);
    m_yawSign = envToDouble("KITE_YAW_SIGN", m_yawSign);
    m_rollOffsetDeg = envToDouble("KITE_ROLL_OFFSET", m_rollOffsetDeg);
    m_pitchOffsetDeg = envToDouble("KITE_PITCH_OFFSET", m_pitchOffsetDeg);
    m_yawOffsetDeg = envToDouble("KITE_YAW_OFFSET", m_yawOffsetDeg);
    m_smoothPose = envToBool("KITE_SMOOTH_POSE", m_smoothPose);
    m_posSmoothAlpha = envToDouble("KITE_POSITION_SMOOTHING", m_posSmoothAlpha);
    m_rotSmoothAlpha = envToDouble("KITE_ROTATION_SMOOTHING", m_rotSmoothAlpha);
    m_despikeOrientation = envToBool("KITE_DESPIKE_ORIENTATION", m_despikeOrientation);
    m_despikeYawOnly = envToBool("KITE_DESPIKE_YAW_ONLY", m_despikeYawOnly);
    m_despikeJumpDeg = envToDouble("KITE_DESPIKE_JUMP_DEG", m_despikeJumpDeg);
    m_despikeSettleDeg = envToDouble("KITE_DESPIKE_SETTLE_DEG", m_despikeSettleDeg);
    m_knotFollowTether = envToBool("KITE_KNOT_FOLLOW_TETHER", m_knotFollowTether);

    m_posSmoothAlpha = std::max(0.0, std::min(1.0, m_posSmoothAlpha));
    m_rotSmoothAlpha = std::max(0.0, std::min(1.0, m_rotSmoothAlpha));

    m_ropeEnabled = envToBool("KITE_ROPE", m_ropeEnabled);
    m_ropeRadius = envToFloat("KITE_ROPE_RADIUS", m_ropeRadius);
    m_ropeSagFactor = envToFloat("KITE_ROPE_SAG_FACTOR", m_ropeSagFactor);
    m_ropeSagMax = envToFloat("KITE_ROPE_SAG_MAX", m_ropeSagMax);
    m_focusMode = envToBool("KITE_FOCUS_MODE", m_focusMode);
    m_focusTetherLen_m = envToFloat("KITE_FOCUS_TETHER_LEN", m_focusTetherLen_m);

    m_showGround = envToBool("KITE_SHOW_GROUND", m_showGround);
    m_groundPos.x() = envToFloat("KITE_GROUND_POS_X", m_groundPos.x());
    m_groundPos.y() = envToFloat("KITE_GROUND_POS_Y", m_groundPos.y());
    m_groundPos.z() = envToFloat("KITE_GROUND_POS_Z", m_groundPos.z());
    if (std::getenv("KITE_GROUND_SIZE_M"))
    {
        m_groundSize_m = envToFloat("KITE_GROUND_SIZE_M", m_groundSize_m);
        m_groundSizeAuto = false;
    }
    m_geoPos.x() = envToFloat("KITE_GEO_POS_X", m_geoPos.x());
    m_geoPos.y() = envToFloat("KITE_GEO_POS_Y", m_geoPos.y());
    m_geoPos.z() = envToFloat("KITE_GEO_POS_Z", m_geoPos.z());
    m_geoScale = envToFloat("KITE_GEO_SCALE", m_geoScale);
    m_geoYawDeg = envToFloat("KITE_GEO_YAW_DEG", m_geoYawDeg);
    m_geoAutoCenter = envToBool("KITE_GEO_AUTO_CENTER", m_geoAutoCenter);
    m_geoAutoGround = envToBool("KITE_GEO_AUTO_GROUND", m_geoAutoGround);

    double modelRollDeg = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.ModelOffsetRoll", 0.0);
    double modelPitchDeg = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.ModelOffsetPitch", 0.0);
    double modelYawDeg = coCoviseConfig::getFloat("COVER.Plugin.KitePlugin.ModelOffsetYaw", 0.0);
    modelRollDeg = envToDouble("KITE_MODEL_OFFSET_ROLL", modelRollDeg);
    modelPitchDeg = envToDouble("KITE_MODEL_OFFSET_PITCH", modelPitchDeg);
    modelYawDeg = envToDouble("KITE_MODEL_OFFSET_YAW", modelYawDeg);

    const double d2r = M_PI / 180.0;
    m_modelOffset = osg::Quat(modelYawDeg * d2r, osg::Vec3(0, 0, 1)) *
                    osg::Quat(modelPitchDeg * d2r, osg::Vec3(0, 1, 0)) *
                    osg::Quat(modelRollDeg * d2r, osg::Vec3(1, 0, 0));

    if (modelEntry.empty())
    {
        fprintf(stderr, "KitePlugin: no model path configured (KitePlugin.Model)\n");
        return false;
    }

    m_transform = new osg::MatrixTransform();
    m_groundXform = new osg::MatrixTransform();
    cover->getObjectsRoot()->addChild(m_groundXform.get());

    if (!loadModel(modelEntry))
    {
        fprintf(stderr, "KitePlugin: could not load model '%s'\n", modelEntry.c_str());
        return false;
    }
    fprintf(stderr, "KitePlugin: loaded model '%s'\n", modelEntry.c_str());

    if (m_model)
    {
        osg::BoundingBox bb = computeLocalBBox(m_model.get());
        fprintf(stderr,
                "KitePlugin: MODEL LOCAL BBOX:\n"
                "  min = (%.6f, %.6f, %.6f)\n"
                "  max = (%.6f, %.6f, %.6f)\n"
                "  size= (%.6f, %.6f, %.6f)\n"
                "  center=(%.6f, %.6f, %.6f)\n",
                bb.xMin(), bb.yMin(), bb.zMin(),
                bb.xMax(), bb.yMax(), bb.zMax(),
                bb.xMax() - bb.xMin(), bb.yMax() - bb.yMin(), bb.zMax() - bb.zMin(),
                bb.center().x(), bb.center().y(), bb.center().z());

        m_modelBB = bb;
        m_haveModelBB = true;
    }

    if (!m_geoModelPath.empty())
    {
        std::string resolved = m_geoModelPath;
        if (const char *searched = coVRFileManager::instance()->getName(m_geoModelPath.c_str()))
            resolved = searched;

        if (!m_geoXform)
            m_geoXform = new osg::MatrixTransform();
        cover->getObjectsRoot()->addChild(m_geoXform.get());

        auto getDir = [](const std::string &path) {
            const size_t pos = path.find_last_of("/\\");
            return (pos == std::string::npos) ? std::string() : path.substr(0, pos);
        };
        const std::string geoBase = getDir(resolved);
        if (!geoBase.empty())
        {
            auto &paths = osgDB::Registry::instance()->getDataFilePathList();
            if (std::find(paths.begin(), paths.end(), geoBase) == paths.end())
                paths.push_back(geoBase);
        }

        osg::ref_ptr<osgDB::Options> geoOptions = new osgDB::Options();
        geoOptions->setDatabasePath(geoBase);

        // Prefer direct read so VRML Inline paths resolve relative to the .wrl directory.
        m_geoModel = osgDB::readNodeFile(resolved, geoOptions.get());
        if (m_geoModel)
        {
            m_geoXform->addChild(m_geoModel.get());
        }
        else
        {
            unsigned int geoChildrenBefore = m_geoXform->getNumChildren();
            osg::Node *geoLoaded = coVRFileManager::instance()->loadFile(resolved.c_str(), nullptr, m_geoXform.get());
            if (geoLoaded)
            {
                m_geoModel = geoLoaded;
            }
            else if (m_geoXform->getNumChildren() > geoChildrenBefore)
            {
                m_geoModel = m_geoXform->getChild(m_geoXform->getNumChildren() - 1);
            }
        }

        if (m_geoModel)
        {
            const double d2r = M_PI / 180.0;
            m_geoXform->setMatrix(geoTransformMatrix(m_geoPos, m_geoScale, m_geoYawDeg));

            osg::BoundingBox gbb = computeLocalBBox(m_geoModel.get());
            if (gbb.valid())
            {
                m_geoBBox = gbb;
                m_haveGeoBBox = true;
                fprintf(stderr,
                        "KitePlugin: loaded geodata '%s'\n"
                        "  geo bbox min=(%.3f %.3f %.3f) max=(%.3f %.3f %.3f)\n",
                        resolved.c_str(),
                        gbb.xMin(), gbb.yMin(), gbb.zMin(),
                        gbb.xMax(), gbb.yMax(), gbb.zMax());
            }
            else
            {
                fprintf(stderr,
                        "KitePlugin: loaded geodata '%s' but bbox is invalid (likely missing inlines)\n",
                        resolved.c_str());
            }
        }
        else
        {
            fprintf(stderr, "KitePlugin: could not load geodata '%s'\n", resolved.c_str());
        }
    }

    cover->getObjectsRoot()->addChild(m_transform);

    // Init ropes after transform exists
    initRopes();

    if (!m_csvPath.empty())
    {
        parseCsv(m_csvPath);
        if (!m_frames.empty())
        {
            coVRAnimationManager::instance()->setNumTimesteps((int)m_frames.size(), this);
            coVRAnimationManager::instance()->enableAnimation(true);
        }
        if (m_useNedFrame)
            fprintf(stderr, "KitePlugin: interpreting CSV positions/orientation as NED and converting to ENU\n");
    }
    else
    {
        fprintf(stderr, "KitePlugin: no CSV path configured (KitePlugin.CSV), showing static model\n");
    }

    return true;
}


void KitePlugin::preFrame()
{
    if (!m_frames.empty())
    {
        int frame = coVRAnimationManager::instance()->getAnimationFrame();
        updateTransform(frame);
    }

    updateRopes();

    // ---- DEBUG: world positions for visual sanity ----
    static int dbgCount = 0;
    if ((dbgCount++ % 300) == 0 && m_transform)
    {
        const osg::Vec3 kitePosW = m_transform->getMatrix().getTrans();
        const osg::Vec3 knotW = localToWorld(m_knotLocal);

        osg::Vec3 a0W(0.f, 0.f, 0.f);
        osg::Vec3 aMW(0.f, 0.f, 0.f);
        if (!m_attachLocal.empty())
        {
            a0W = localToWorld(m_attachLocal.front());
            aMW = localToWorld(m_attachLocal[m_attachLocal.size() / 2]);
        }

        const osg::Vec3 groundW = groundStationWorld();
        fprintf(stderr,
                "KitePlugin DBG world: kitePos=(%.2f %.2f %.2f) |r|=%.2f\n"
                "  ground=(%.2f %.2f %.2f)\n"
                "  knotW =(%.2f %.2f %.2f) dist(ground->knot)=%.2f\n"
                "  a0W   =(%.2f %.2f %.2f) dist(knot->a0)=%.2f\n"
                "  aMW  =(%.2f %.2f %.2f) dist(knot->aMid)=%.2f\n",
                kitePosW.x(), kitePosW.y(), kitePosW.z(), kitePosW.length(),
                groundW.x(), groundW.y(), groundW.z(),
                knotW.x(), knotW.y(), knotW.z(), (knotW - groundW).length(),
                a0W.x(), a0W.y(), a0W.z(), (a0W - knotW).length(),
                aMW.x(), aMW.y(), aMW.z(), (aMW - knotW).length());
    }
}

COVERPLUGIN(KitePlugin)
