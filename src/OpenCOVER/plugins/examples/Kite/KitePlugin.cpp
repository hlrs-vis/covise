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
}

static osg::BoundingBox computeLocalBBox(osg::Node *n)
{
    osg::ComputeBoundsVisitor cbv;
    n->accept(cbv);
    return cbv.getBoundingBox();
}


bool KitePlugin::init()
{
    auto modelEntry = coCoviseConfig::getEntry("value", "KitePlugin.Model", "");
    m_csvPath = coCoviseConfig::getEntry("value", "KitePlugin.CSV", "");

    m_useNedFrame = coCoviseConfig::isOn("KitePlugin.NED", m_useNedFrame);
    m_rollSign = coCoviseConfig::getFloat("KitePlugin.RollSign", m_rollSign);
    m_pitchSign = coCoviseConfig::getFloat("KitePlugin.PitchSign", m_pitchSign);
    m_yawSign = coCoviseConfig::getFloat("KitePlugin.YawSign", m_yawSign);
    m_rollOffsetDeg = coCoviseConfig::getFloat("KitePlugin.RollOffset", m_rollOffsetDeg);
    m_pitchOffsetDeg = coCoviseConfig::getFloat("KitePlugin.PitchOffset", m_pitchOffsetDeg);
    m_yawOffsetDeg = coCoviseConfig::getFloat("KitePlugin.YawOffset", m_yawOffsetDeg);

    // Rope config (optional)
    m_ropeEnabled = coCoviseConfig::isOn("KitePlugin.RopeEnabled", m_ropeEnabled);
    // m_ropeRadius = coCoviseConfig::getFloat("KitePlugin.RopeRadius", m_ropeRadius);
    m_ropeRadius = coCoviseConfig::getFloat("KitePlugin.RopeRadius", m_ropeRadius);
    if (m_ropeRadius <= 0.f)
        m_ropeRadius = 0.05f; // force visible default

    m_ropeSagFactor = coCoviseConfig::getFloat("KitePlugin.RopeSagFactor", m_ropeSagFactor);
    m_ropeSagMax = coCoviseConfig::getFloat("KitePlugin.RopeSagMax", m_ropeSagMax);
    m_ropeSamplesMain = (int)coCoviseConfig::getFloat("KitePlugin.RopeSamplesMain", (float)m_ropeSamplesMain);
    m_ropeSamplesBridle = (int)coCoviseConfig::getFloat("KitePlugin.RopeSamplesBridle", (float)m_ropeSamplesBridle);

    m_drawFullTether = coCoviseConfig::isOn("KitePlugin.DrawFullTether", m_drawFullTether);
    m_fullTetherAlpha = coCoviseConfig::getFloat("KitePlugin.FullTetherAlpha", m_fullTetherAlpha);
    m_focusTetherAlpha = coCoviseConfig::getFloat("KitePlugin.FocusTetherAlpha", m_focusTetherAlpha);

    m_tetherFade = coCoviseConfig::isOn("KitePlugin.TetherFade", m_tetherFade);
    m_tetherNearLen_m = coCoviseConfig::getFloat("KitePlugin.TetherNearLen_m", m_tetherNearLen_m);

    m_tetherSagBoost = coCoviseConfig::getFloat("KitePlugin.TetherSagBoost", m_tetherSagBoost);
    m_tetherSagMaxLong_m = coCoviseConfig::getFloat("KitePlugin.TetherSagMaxLong_m", m_tetherSagMaxLong_m);

    m_tetherLineWidthNear = coCoviseConfig::getFloat("KitePlugin.TetherLineWidthNear", m_tetherLineWidthNear);
    m_tetherLineWidthFar = coCoviseConfig::getFloat("KitePlugin.TetherLineWidthFar", m_tetherLineWidthFar);

    m_tetherAlphaNear = coCoviseConfig::getFloat("KitePlugin.TetherAlphaNear", m_tetherAlphaNear);
    m_tetherAlphaFar = coCoviseConfig::getFloat("KitePlugin.TetherAlphaFar", m_tetherAlphaFar);

    m_snapAnchorsToSurface = coCoviseConfig::isOn("KitePlugin.SnapAnchorsToSurface", m_snapAnchorsToSurface);
    m_anchorLift_m = coCoviseConfig::getFloat("KitePlugin.AnchorLift_m", m_anchorLift_m);
    m_snapRayUp_m = coCoviseConfig::getFloat("KitePlugin.SnapRayUp_m", m_snapRayUp_m);
    m_snapRayDown_m = coCoviseConfig::getFloat("KitePlugin.SnapRayDown_m", m_snapRayDown_m);

    m_groundPos.x() = coCoviseConfig::getFloat("KitePlugin.GroundPosX", m_groundPos.x());
    m_groundPos.y() = coCoviseConfig::getFloat("KitePlugin.GroundPosY", m_groundPos.y());
    m_groundPos.z() = coCoviseConfig::getFloat("KitePlugin.GroundPosZ", m_groundPos.z());
    m_worldScale = coCoviseConfig::getFloat("KitePlugin.WorldScale", m_worldScale);
    m_targetTether_m = coCoviseConfig::getFloat("KitePlugin.TargetTether_m", m_targetTether_m);
    m_focusMode = coCoviseConfig::isOn("KitePlugin.FocusMode", m_focusMode);
    m_focusTetherLen_m = coCoviseConfig::getFloat("KitePlugin.FocusTetherLen_m", m_focusTetherLen_m);

    m_junctionLocal.x() = coCoviseConfig::getFloat("KitePlugin.JunctionLocalX", m_junctionLocal.x());
    m_junctionLocal.y() = coCoviseConfig::getFloat("KitePlugin.JunctionLocalY", m_junctionLocal.y());
    m_junctionLocal.z() = coCoviseConfig::getFloat("KitePlugin.JunctionLocalZ", m_junctionLocal.z());

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
        if (const char *env = std::getenv("KITE_MODEL"))
            modelEntry = env;
    }
    if (m_csvPath.empty())
    {
        if (const char *env = std::getenv("KITE_CSV"))
            m_csvPath = env;
    }

    m_useNedFrame = envToBool("KITE_NED", m_useNedFrame);
    if (const char *env = std::getenv("KITE_FRAME"))
    {
        std::string val(env);
        std::transform(val.begin(), val.end(), val.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (val == "ned")
            m_useNedFrame = true;
        else if (val == "enu")
            m_useNedFrame = false;
    }

    m_rollSign = envToDouble("KITE_ROLL_SIGN", m_rollSign);
    m_pitchSign = envToDouble("KITE_PITCH_SIGN", m_pitchSign);
    m_yawSign = envToDouble("KITE_YAW_SIGN", m_yawSign);
    m_rollOffsetDeg = envToDouble("KITE_ROLL_OFFSET", m_rollOffsetDeg);
    m_pitchOffsetDeg = envToDouble("KITE_PITCH_OFFSET", m_pitchOffsetDeg);
    m_yawOffsetDeg = envToDouble("KITE_YAW_OFFSET", m_yawOffsetDeg);

    m_ropeEnabled = envToBool("KITE_ROPE", m_ropeEnabled);
    m_ropeRadius = envToFloat("KITE_ROPE_RADIUS", m_ropeRadius);
    m_ropeSagFactor = envToFloat("KITE_ROPE_SAG_FACTOR", m_ropeSagFactor);
    m_ropeSagMax = envToFloat("KITE_ROPE_SAG_MAX", m_ropeSagMax);
    m_focusMode = envToBool("KITE_FOCUS_MODE", m_focusMode);
    m_focusTetherLen_m = envToFloat("KITE_FOCUS_TETHER_LEN", m_focusTetherLen_m);

    double modelRollDeg = coCoviseConfig::getFloat("KitePlugin.ModelOffsetRoll", 0.0);
    double modelPitchDeg = coCoviseConfig::getFloat("KitePlugin.ModelOffsetPitch", 0.0);
    double modelYawDeg = coCoviseConfig::getFloat("KitePlugin.ModelOffsetYaw", 0.0);
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

bool KitePlugin::loadModel(const std::string &path)
{
    std::string resolved = path;
    if (const char *searched = coVRFileManager::instance()->getName(path.c_str()))
        resolved = searched;

    unsigned int childrenBefore = m_transform->getNumChildren();

    osg::Node *loaded = coVRFileManager::instance()->loadFile(resolved.c_str(), nullptr, m_transform.get());
    if (loaded)
    {
        m_model = loaded;
        return true;
    }

    if (m_transform->getNumChildren() > childrenBefore)
    {
        m_model = m_transform->getChild(m_transform->getNumChildren() - 1);
        fprintf(stderr, "KitePlugin: model loaded via handler into transform (%u children)\n", m_transform->getNumChildren());
        return true;
    }

    m_model = osgDB::readNodeFile(resolved);
    if (!m_model)
        return false;

    m_transform->addChild(m_model);
    return true;
}

void KitePlugin::parseCsv(const std::string &path)
{
    std::ifstream in(path.c_str());
    if (!in.good())
    {
        fprintf(stderr, "KitePlugin: cannot open CSV '%s'\n", path.c_str());
        return;
    }
    fprintf(stderr, "KitePlugin: parsing CSV '%s'\n", path.c_str());

    auto split = [](const std::string &s, char delim) {
        std::vector<std::string> out;
        std::stringstream ss(s);
        std::string tok;
        while (std::getline(ss, tok, delim))
            out.push_back(tok);
        return out;
    };

    std::string headerLine;
    if (!std::getline(in, headerLine))
        return;

    char delim = (headerLine.find(';') != std::string::npos) ? ';' : ',';
    auto headers = split(headerLine, delim);

    std::unordered_map<std::string, size_t> idx;
    for (size_t i = 0; i < headers.size(); ++i)
        idx[headers[i]] = i;

    const char *required[] = {"time", "kite_pos_east", "kite_pos_north", "kite_height",
                              "kite_0_roll", "kite_0_pitch", "kite_0_yaw"};
    for (auto key : required)
    {
        if (idx.find(key) == idx.end())
        {
            fprintf(stderr, "KitePlugin: CSV missing column '%s'\n", key);
            return;
        }
    }

    auto trim = [](const std::string &s) {
        size_t b = 0;
        while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b])))
            ++b;
        size_t e = s.size();
        while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1])))
            --e;
        return s.substr(b, e - b);
    };
    auto parseNumber = [&](const std::string &s, double &out) {
        std::string t = trim(s);
        if (t.empty())
            return false;
        char *end = nullptr;
        out = std::strtod(t.c_str(), &end);
        if (end == t.c_str())
            return false;
        while (end && *end && std::isspace(static_cast<unsigned char>(*end)))
            ++end;
        return end && *end == '\0';
    };

    std::string line;
    size_t skipped = 0;
    while (std::getline(in, line))
    {
        if (line.empty())
            continue;

        auto toks = split(line, delim);
        if (toks.size() <= idx["kite_0_yaw"])
        {
            ++skipped;
            continue;
        }

        Frame f;
        double v = 0.0;
        if (!parseNumber(toks[idx["time"]], f.t)) { ++skipped; continue; }
        if (!parseNumber(toks[idx["kite_pos_east"]], v)) { ++skipped; continue; }
        f.pos.x() = (float)v;
        if (!parseNumber(toks[idx["kite_pos_north"]], v)) { ++skipped; continue; }
        f.pos.y() = (float)v;
        if (!parseNumber(toks[idx["kite_height"]], v)) { ++skipped; continue; }
        f.pos.z() = (float)v;
        if (!parseNumber(toks[idx["kite_0_roll"]], f.roll)) { ++skipped; continue; }
        if (!parseNumber(toks[idx["kite_0_pitch"]], f.pitch)) { ++skipped; continue; }
        if (!parseNumber(toks[idx["kite_0_yaw"]], f.yaw)) { ++skipped; continue; }

        m_frames.push_back(f);
    }

    fprintf(stderr, "KitePlugin: loaded %zu frames (skipped %zu) from '%s'\n",
            m_frames.size(), skipped, path.c_str());

    if (!m_frames.empty())
    {
        // DO NOT shift XY here and DO NOT apply target tether scaling here.
        // Keep positions in meters exactly as in the CSV.
        // Ground station is (0,0,0) in this coordinate system if the CSV is truly "relative to station".
        // If the dataset actually has an offset, we can fix that later using GPS columns.

        // Debug: find min/max height in raw meters
        double minZ = 1e30, maxZ = -1e30;
        size_t minZi = 0;
        for (size_t i = 0; i < m_frames.size(); ++i)
        {
            const double z = m_frames[i].pos.z();
            if (z < minZ)
            {
                minZ = z;
                minZi = i;
            }
            if (z > maxZ)
                maxZ = z;
        }

        fprintf(stderr,
                "KitePlugin: RAW height stats (meters): minZ=%.3f at i=%zu (t=%.3f), maxZ=%.3f\n",
                minZ, minZi, m_frames[minZi].t, maxZ);
    }
}

void KitePlugin::updateTransform(int frameIndex)
{
    if (!m_transform || m_frames.empty())
        return;

    frameIndex = std::max(0, std::min(frameIndex, (int)m_frames.size() - 1));
    const auto &f = m_frames[frameIndex];

    const double d2r = M_PI / 180.0;
    const double rollDeg = f.roll * m_rollSign + m_rollOffsetDeg;
    const double pitchDeg = f.pitch * m_pitchSign + m_pitchOffsetDeg;
    const double yawDeg = f.yaw * m_yawSign + m_yawOffsetDeg;

    const osg::Quat roll(rollDeg * d2r, osg::Vec3(1, 0, 0));
    const osg::Quat pitch(pitchDeg * d2r, osg::Vec3(0, 1, 0));
    const osg::Quat yaw(yawDeg * d2r, osg::Vec3(0, 0, 1));

    osg::Quat q = yaw * pitch * roll;
    // f.pos is in METERS (CSV units). Convert to scene units here.
    osg::Vec3 pos = f.pos * (m_unitsPerMeter * m_worldScale * m_globalPosScale);

    if (m_useNedFrame)
    {
        pos = osg::Vec3(pos.y(), pos.x(), -pos.z());
        static const osg::Quat nedToEnu =
            osg::Quat(90.0 * d2r, osg::Vec3(0, 0, 1)) *
            osg::Quat(180.0 * d2r, osg::Vec3(1, 0, 0));
        q = nedToEnu * q;
    }

    if (m_haveModelBB)
    {
        const float zGround = m_groundPos.z();
        const float meshBottom = pos.z() + m_modelBB.zMin();
        if (meshBottom < zGround - 1e-3f)
        {
            fprintf(stderr,
                    "KitePlugin WARN: below ground: frame=%d posZ=%.2f meshBottom=%.2f ground=%.2f rawHeight(m)=%.3f\n",
                    frameIndex, pos.z(), meshBottom, zGround, f.pos.z());
        }
    }

    // Optional: clamp so the *mesh* never goes below the ground plane.
    // Ground plane is at m_groundPos.z() (world units), but you also keep a groundXform.
    // Here we assume your ground plane is z = m_groundPos.z().
    if (m_clampAboveGround && m_haveModelBB)
    {
        const float zGround = m_groundPos.z(); // in units
        const float minNeeded = zGround - m_modelBB.zMin(); // because zMin is negative
        if (pos.z() < minNeeded)
            pos.z() = minNeeded;
    }

    q = m_modelOffset * q;

    osg::Matrix m = osg::Matrix::translate(pos) * osg::Matrix::rotate(q);
    m_transform->setMatrix(m);
}

// ----------------- Rope helpers -----------------

osg::Vec3 KitePlugin::localToWorld(const osg::Vec3 &pLocal) const
{
    return m_transform->getMatrix().preMult(pLocal);
}

osg::Quat KitePlugin::quatFromZToDir(const osg::Vec3 &dir)
{
    osg::Vec3 d = dir;
    if (d.length2() < 1e-12)
        return osg::Quat();
    d.normalize();
    osg::Quat q;
    q.makeRotate(osg::Vec3(0, 0, 1), d);
    return q;
}

std::vector<osg::Vec3> KitePlugin::sagCurvePoints(const osg::Vec3 &a, const osg::Vec3 &b,
                                                  int samples, float sagUnits)
{
    samples = std::max(2, samples);

    const osg::Vec3 up(0, 0, 1);
    const osg::Vec3 ab = b - a;
    const float L = ab.length();

    std::vector<osg::Vec3> pts;
    pts.reserve(samples);

    if (L < 1e-6f)
    {
        pts.push_back(a);
        pts.push_back(b);
        return pts;
    }

    const osg::Vec3 d = ab / L;
    const float verticalness = std::fabs(d * up);
    // IMPORTANT: keep some sag even if nearly vertical (visual + still plausible)
    const float sagScale = std::max(0.25f, 1.0f - verticalness);
    const float sag = sagUnits * sagScale;

    for (int i = 0; i < samples; ++i)
    {
        const float t = float(i) / float(samples - 1);
        osg::Vec3 p = a + ab * t;

        // max sag at mid (sin curve) - purely visual
        const float s = std::sin(float(M_PI) * t);
        p -= up * (sag * s);

        pts.push_back(p);
    }

    return pts;
}

void KitePlugin::addSaggedLine(osg::Vec3Array *arr,
                               const osg::Vec3 &a,
                               const osg::Vec3 &b,
                               int samples,
                               float sagUnits) const
{
    if (!arr)
        return;

    samples = std::max(2, samples);

    if (samples <= 2 || sagUnits <= 1e-6f)
    {
        arr->push_back(a);
        arr->push_back(b);
        return;
    }

    auto pts = sagCurvePoints(a, b, samples, sagUnits);
    for (size_t i = 1; i < pts.size(); ++i)
    {
        arr->push_back(pts[i - 1]);
        arr->push_back(pts[i]);
    }
}

void KitePlugin::appendPolylineAsLines(osg::Vec3Array *arr, const std::vector<osg::Vec3> &pts)
{
    if (!arr || pts.size() < 2)
        return;

    const float eps2 = 1e-8f;
    for (size_t i = 0; i + 1 < pts.size(); ++i)
    {
        const osg::Vec3 a = pts[i];
        const osg::Vec3 b = pts[i + 1];
        if ((b - a).length2() < eps2)
            continue; // avoid overdraw "bold" segments

        arr->push_back(a);
        arr->push_back(b);
    }
}

// Clips the *tail* (end) of a polyline by arc-length, returning only the last tailLenUnits.
// Keeps curvature consistent (important for "focus tether" mode).
std::vector<osg::Vec3> KitePlugin::clipPolylineTailByLength(const std::vector<osg::Vec3> &pts, float tailLenUnits)
{
    if (pts.size() < 2 || tailLenUnits <= 0.f)
        return pts;

    // walk backwards accumulating arc length
    float acc = 0.f;
    for (int i = (int)pts.size() - 1; i > 0; --i)
    {
        const osg::Vec3 a = pts[i - 1];
        const osg::Vec3 b = pts[i];
        const float seg = (b - a).length();

        if (acc + seg >= tailLenUnits)
        {
            const float need = tailLenUnits - acc;
            const float t = (seg > 1e-6f) ? (need / seg) : 0.f;
            const osg::Vec3 start = b + (a - b) * t; // interpolate towards previous

            std::vector<osg::Vec3> out;
            out.reserve((pts.size() - (size_t)(i - 1)) + 1);
            out.push_back(start);
            for (size_t k = (size_t)i; k < pts.size(); ++k)
                out.push_back(pts[k]);
            return out;
        }
        acc += seg;
    }

    // tail longer than total length: return whole thing
    return pts;
}

bool KitePlugin::snapPointToModelSurfaceMultiAxis(osg::Node *model, const osg::Vec3 &pModel,
                                                  float rayUp, float rayDown,
                                                  osg::Vec3 &hitModel) const
{
    if (!model)
        return false;

    // Try 6 axis directions; pick closest hit to pModel
    const osg::Vec3 axes[3] = { osg::Vec3(0, 0, 1), osg::Vec3(1, 0, 0), osg::Vec3(0, 1, 0) };

    bool found = false;
    float bestD2 = 1e30f;
    osg::Vec3 bestHit;

    auto tryRay = [&](const osg::Vec3 &dir) {
        osg::Vec3 start = pModel + dir * rayUp;
        osg::Vec3 end = pModel - dir * rayDown;

        osg::ref_ptr<osgUtil::LineSegmentIntersector> inter =
            new osgUtil::LineSegmentIntersector(osgUtil::Intersector::MODEL, start, end);

        osgUtil::IntersectionVisitor iv(inter.get());
        model->accept(iv);

        if (!inter->containsIntersections())
            return;

        // Closest intersection along segment (first in set)
        const auto &isect = *inter->getIntersections().begin();
        osg::Vec3 hit = isect.getLocalIntersectPoint();
        float d2 = (hit - pModel).length2();
        if (d2 < bestD2)
        {
            bestD2 = d2;
            bestHit = hit;
            found = true;
        }
    };

    for (int a = 0; a < 3; ++a)
    {
        tryRay(axes[a]);
        tryRay(-axes[a]);
    }

    if (!found)
        return false;

    hitModel = bestHit;
    return true;
}

void KitePlugin::snapBridleAnchorsToSurface(const osg::BoundingBox &bb)
{
    (void)bb;
    if (!m_snapAnchorsToSurface || !m_model || m_attachLocal.empty())
        return;

    const float liftUnits = m_anchorLift_m * m_unitsPerMeter;
    const float rayUpUnits = m_snapRayUp_m * m_unitsPerMeter;
    const float rayDownUnits = m_snapRayDown_m * m_unitsPerMeter;

    int snapped = 0;
    for (auto &p : m_attachLocal)
    {
        osg::Vec3 hit;
        if (snapPointToModelSurfaceMultiAxis(m_model.get(), p, rayUpUnits, rayDownUnits, hit))
        {
            // lift slightly in +Z model axis to avoid z-fighting/gaps
            p = hit + osg::Vec3(0.f, 0.f, liftUnits);
            ++snapped;
        }
    }

    fprintf(stderr, "KitePlugin: snapped %d/%zu bridle anchors to surface (multi-axis)\n",
            snapped, m_attachLocal.size());
}

void KitePlugin::ensureSegments(RopeVisual &rope, size_t needed)
{
    if (!rope.root)
        rope.root = new osg::Group();

    while (rope.segments.size() < needed)
    {
        RopeSegment seg;
        seg.xform = new osg::MatrixTransform();
        seg.scale = new osg::MatrixTransform();

        // Cylinder is defined in osg/Shape
        osg::ref_ptr<osg::Cylinder> cyl = new osg::Cylinder(osg::Vec3(0, 0, 0), 1.0f, 1.0f);
        osg::ref_ptr<osg::ShapeDrawable> drawable = new osg::ShapeDrawable(cyl);

        osg::ref_ptr<osg::Geode> geode = new osg::Geode();
        geode->addDrawable(drawable);

        seg.scale->addChild(geode);
        seg.xform->addChild(seg.scale);
        rope.root->addChild(seg.xform);

        rope.segments.push_back(seg);
    }

    for (size_t i = 0; i < rope.segments.size(); ++i)
        rope.segments[i].xform->setNodeMask(i < needed ? ~0u : 0u);
}

void KitePlugin::initRopes()
{
    if (!m_ropeEnabled || !m_model)
        return;

    osg::BoundingBox bb = computeLocalBBox(m_model.get());
    const float spanY = bb.yMax() - bb.yMin();
    const float chordX = bb.xMax() - bb.xMin();
    const float spanZ = bb.zMax() - bb.zMin();

    // Scale: model-units per meter (use span as most reliable).
    m_unitsPerMeter = (m_span_m > 0.f) ? (spanY / m_span_m) : 1.0f;

    // CoG estimate using TU Delft CoG fractions.
    osg::Vec3 cogLocal;
    cogLocal.x() = bb.xMin() + m_cogX_frac * chordX;
    cogLocal.y() = 0.0f;
    cogLocal.z() = bb.zMin() + m_cogZ_frac * spanZ;

    // Knot / KCU attach point: bridle height below CoG (down is toward zMin).
    m_knotLocal = cogLocal;
    m_knotLocal.z() -= m_bridleHeight_m * m_unitsPerMeter;

    const float knotSplit = m_knotSplit_m * m_unitsPerMeter * 0.5f;
    m_knotFrontLocal = m_knotLocal + osg::Vec3(-knotSplit, 0.f, 0.f);
    m_knotRearLocal = m_knotLocal + osg::Vec3(knotSplit, 0.f, 0.f);

    // Build bridle anchor points on underside.
    m_attachLocal.clear();

    const float y0 = bb.yMin() + 0.15f * spanY;
    const float y1 = bb.yMax() - 0.15f * spanY;
    const float zUnder = bb.zMin() + m_underside_frac * spanZ;

    auto curvedZ = [&](float t) {
        const float u = 2.f * t - 1.f;
        const float shape = 1.f - u * u;
        return zUnder + 0.15f * spanZ * shape;
    };
    auto curvedX = [&](float baseX, float t) {
        const float u = 2.f * t - 1.f;
        const float shape = u * u;
        return baseX - 0.08f * chordX * shape;
    };
    auto addRow = [&](float chordFrac) {
        const float baseX = bb.xMin() + chordFrac * chordX;
        for (int i = 0; i < m_bridleN_span; ++i)
        {
            const float t = (m_bridleN_span == 1) ? 0.5f : (float)i / (float)(m_bridleN_span - 1);
            const float y = y0 + t * (y1 - y0);
            const float x = curvedX(baseX, t);
            const float z = curvedZ(t);
            m_attachLocal.emplace_back(x, y, z);
        }
    };

    // Front row always.
    addRow(m_frontRow_frac);
    if (m_twoRows)
        addRow(m_rearRow_frac);

    // Snap anchor points onto the actual mesh surface so bridles look attached
    snapBridleAnchorsToSurface(bb);

    // ---- DEBUG: model / bridle geometry sanity ----
    {
        const float spanX = bb.xMax() - bb.xMin();
        fprintf(stderr,
                "KitePlugin DBG bbox: x[%.3f..%.3f] y[%.3f..%.3f] z[%.3f..%.3f]\n"
                "KitePlugin DBG spans: spanX=%.3f spanY=%.3f spanZ=%.3f center=(%.3f %.3f %.3f)\n",
                bb.xMin(), bb.xMax(), bb.yMin(), bb.yMax(), bb.zMin(), bb.zMax(),
                spanX, spanY, spanZ,
                bb.center().x(), bb.center().y(), bb.center().z());

        fprintf(stderr,
                "KitePlugin DBG unitsPerMeter=%.3f bridleHeight_m=%.2f bridleHeight_units=%.2f\n",
                m_unitsPerMeter, m_bridleHeight_m, m_unitsPerMeter * m_bridleHeight_m);

        fprintf(stderr,
                "KitePlugin DBG cogLocal=(%.3f %.3f %.3f) knotLocal=(%.3f %.3f %.3f)\n",
                cogLocal.x(), cogLocal.y(), cogLocal.z(),
                m_knotLocal.x(), m_knotLocal.y(), m_knotLocal.z());

        if (!m_attachLocal.empty())
        {
            const auto &a0 = m_attachLocal.front();
            const auto &aM = m_attachLocal[m_attachLocal.size() / 2];
            const auto &aL = m_attachLocal.back();
            fprintf(stderr,
                    "KitePlugin DBG anchorsLocal: N=%zu\n"
                    "  first=(%.3f %.3f %.3f)\n"
                    "  mid  =(%.3f %.3f %.3f)\n"
                    "  last =(%.3f %.3f %.3f)\n",
                    m_attachLocal.size(),
                    a0.x(), a0.y(), a0.z(),
                    aM.x(), aM.y(), aM.z(),
                    aL.x(), aL.y(), aL.z());
        }
    }

    m_groundSize_m = std::max(200.0f, 2.2f * m_targetTether_m);

    m_ropeGeode = new osg::Geode();

    // ---- FULL tether (faint) ----
    m_tetherFullGeom = new osg::Geometry();
    m_tetherFullVerts = new osg::Vec3Array();
    m_tetherFullColors = new osg::Vec4Array();
    m_tetherFullGeom->setVertexArray(m_tetherFullVerts.get());
    m_tetherFullGeom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 0));
    m_tetherFullGeom->setDataVariance(osg::Object::DYNAMIC);
    m_tetherFullColors->push_back(osg::Vec4(1.f, 1.f, 1.f, m_fullTetherAlpha));
    m_tetherFullGeom->setColorArray(m_tetherFullColors.get(), osg::Array::BIND_OVERALL);
    m_ropeGeode->addDrawable(m_tetherFullGeom.get());

    // ---- FOCUS tether (bright) ----
    m_tetherGeom = new osg::Geometry();
    m_tetherVerts = new osg::Vec3Array();
    m_tetherFocusColors = new osg::Vec4Array();
    m_tetherGeom->setVertexArray(m_tetherVerts.get());
    m_tetherGeom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 0));
    m_tetherGeom->setDataVariance(osg::Object::DYNAMIC);
    m_tetherFocusColors->push_back(osg::Vec4(1.f, 1.f, 1.f, m_focusTetherAlpha));
    m_tetherGeom->setColorArray(m_tetherFocusColors.get(), osg::Array::BIND_OVERALL);
    m_ropeGeode->addDrawable(m_tetherGeom.get());

    // ---- Bridles ----
    m_bridleGeom = new osg::Geometry();
    m_bridleVerts = new osg::Vec3Array();
    m_bridleColors = new osg::Vec4Array();
    m_bridleGeom->setVertexArray(m_bridleVerts.get());
    m_bridleGeom->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, 0));
    m_bridleGeom->setDataVariance(osg::Object::DYNAMIC);
    m_bridleColors->push_back(osg::Vec4(1.f, 1.f, 1.f, 1.f));
    m_bridleGeom->setColorArray(m_bridleColors.get(), osg::Array::BIND_OVERALL);
    m_ropeGeode->addDrawable(m_bridleGeom.get());

    // Styling: no lighting, blend on
    osg::StateSet *ss = m_ropeGeode->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::ON);
    ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    // Use thinner default to reduce "bold" look
    ss->setAttributeAndModes(new osg::LineWidth(1.0f), osg::StateAttribute::ON);

    if (!m_ropeRoot)
        m_ropeRoot = new osg::Group();
    m_ropeRoot->addChild(m_ropeGeode.get());
    cover->getObjectsRoot()->addChild(m_ropeRoot.get());

    fprintf(stderr,
            "KitePlugin: unitsPerMeter=%.3f (spanY=%.3f m_span=%.3f)\n"
            "KitePlugin: cogLocal=(%.3f %.3f %.3f)\n"
            "KitePlugin: knotLocal=(%.3f %.3f %.3f) (bridleHeight=%.2fm)\n"
            "KitePlugin: bridle anchors=%zu (twoRows=%d)\n",
            m_unitsPerMeter, spanY, m_span_m,
            cogLocal.x(), cogLocal.y(), cogLocal.z(),
            m_knotLocal.x(), m_knotLocal.y(), m_knotLocal.z(), m_bridleHeight_m,
            m_attachLocal.size(), (int)m_twoRows);

    createGround();
    createGroundStation();
    updateGroundAnchor();
}

void KitePlugin::createGround()
{
    if (!m_showGround)
        return;

    if (!m_groundXform)
    {
        m_groundXform = new osg::MatrixTransform();
        cover->getObjectsRoot()->addChild(m_groundXform.get());
    }

    if (m_groundGeode)
        return;

    const float s = m_groundSize_m * m_unitsPerMeter * 0.5f;

    osg::ref_ptr<osg::Geometry> geom = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> v = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> c = new osg::Vec4Array;
    osg::ref_ptr<osg::DrawElementsUInt> idx = new osg::DrawElementsUInt(GL_TRIANGLES);

    v->push_back(osg::Vec3(-s, -s, 0.f));
    v->push_back(osg::Vec3( s, -s, 0.f));
    v->push_back(osg::Vec3( s,  s, 0.f));
    v->push_back(osg::Vec3(-s,  s, 0.f));

    idx->push_back(0); idx->push_back(1); idx->push_back(2);
    idx->push_back(0); idx->push_back(2); idx->push_back(3);

    c->push_back(osg::Vec4(0.15f, 0.15f, 0.15f, 0.35f));

    geom->setVertexArray(v.get());
    geom->addPrimitiveSet(idx.get());
    geom->setColorArray(c.get(), osg::Array::BIND_OVERALL);
    osg::StateSet *ss = geom->getOrCreateStateSet();
    ss->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::ON);
    ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    osg::ref_ptr<osg::Depth> depth = new osg::Depth;
    depth->setWriteMask(false);
    ss->setAttributeAndModes(depth.get(), osg::StateAttribute::ON);
    ss->setMode(GL_LINE_SMOOTH, osg::StateAttribute::ON);

    m_groundGeode = new osg::Geode;
    m_groundGeode->setName("KitePlugin_Ground");
    m_groundGeode->addDrawable(geom.get());

    // Grid for scale cues.
    osg::ref_ptr<osg::Geometry> grid = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> gv = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> gc = new osg::Vec4Array;
    gc->push_back(osg::Vec4(0.3f, 0.3f, 0.3f, 0.35f));
    grid->setVertexArray(gv.get());
    grid->setColorArray(gc.get(), osg::Array::BIND_OVERALL);

    const float half = s;
    const float step = 20.0f * m_unitsPerMeter;
    for (float x = -half; x <= half; x += step)
    {
        gv->push_back(osg::Vec3(x, -half, 0.f));
        gv->push_back(osg::Vec3(x,  half, 0.f));
    }
    for (float y = -half; y <= half; y += step)
    {
        gv->push_back(osg::Vec3(-half, y, 0.f));
        gv->push_back(osg::Vec3( half, y, 0.f));
    }
    grid->addPrimitiveSet(new osg::DrawArrays(GL_LINES, 0, gv->size()));

    osg::StateSet *gss = grid->getOrCreateStateSet();
    gss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    gss->setMode(GL_BLEND, osg::StateAttribute::ON);
    gss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    m_groundGeode->addDrawable(grid.get());

    // attach to ground transform
    m_groundXform->addChild(m_groundGeode.get());

    // Place groundXform at world z-offset, and also define ground station origin there.
    const float zW = m_groundZOffset_m * m_unitsPerMeter;
    m_groundPos = osg::Vec3(0.f, 0.f, zW);
    m_groundXform->setMatrix(osg::Matrix::translate(m_groundPos));
}

void KitePlugin::createGroundStation()
{
    if (!m_groundXform)
    {
        m_groundXform = new osg::MatrixTransform();
        cover->getObjectsRoot()->addChild(m_groundXform.get());
        m_groundXform->setMatrix(osg::Matrix::translate(m_groundPos));
    }

    if (m_stationGeode)
        return;

    const float r = 0.5f * m_unitsPerMeter;

    osg::ref_ptr<osg::Sphere> sph = new osg::Sphere(osg::Vec3(0, 0, 0), r);
    osg::ref_ptr<osg::ShapeDrawable> sd = new osg::ShapeDrawable(sph.get());
    sd->setColor(osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));

    m_stationGeode = new osg::Geode;
    m_stationGeode->setName("KitePlugin_GroundStation");
    m_stationGeode->addDrawable(sd.get());

    // attach to same ground transform
    m_groundXform->addChild(m_stationGeode.get());
}

osg::Vec3 KitePlugin::groundStationWorld() const
{
    // m_groundPos is already in world coords (after our setup)
    return m_groundPos;
}

void KitePlugin::updateGroundAnchor()
{
    if (!m_groundXform)
        return;

    m_groundXform->setMatrix(osg::Matrix::translate(groundStationWorld()));
}

void KitePlugin::updateRopes()
{
    if (!m_ropeEnabled || !m_transform)
        return;

    const osg::Vec3 knotWorld = localToWorld(m_knotLocal);
    const osg::Vec3 knotFrontWorld = localToWorld(m_knotFrontLocal);
    const osg::Vec3 knotRearWorld = localToWorld(m_knotRearLocal);

    const osg::Vec3 g = m_groundPos;
    const osg::Vec3 k = knotWorld;

    // ---------- FULL tether (faint) ----------
    if (m_tetherFullVerts && m_tetherFullGeom)
    {
        m_tetherFullVerts->clear();

        const float L_units = (k - g).length();
        const float L_m = (m_unitsPerMeter > 1e-6f) ? (L_units / m_unitsPerMeter) : L_units;
        const float sag_m = std::min(m_ropeSagFactor * L_m, m_ropeSagMax);
        const float sag_units = sag_m * m_unitsPerMeter;

        const int samplesMain = std::max(8, m_ropeSamplesMain);
        std::vector<osg::Vec3> fullPts = sagCurvePoints(g, k, samplesMain, sag_units);

        if (m_drawFullTether)
            appendPolylineAsLines(m_tetherFullVerts.get(), fullPts);

        static_cast<osg::DrawArrays *>(m_tetherFullGeom->getPrimitiveSet(0))->setCount((int)m_tetherFullVerts->size());
        m_tetherFullVerts->dirty();
        m_tetherFullGeom->dirtyDisplayList();
        m_tetherFullGeom->dirtyBound();
    }

    // ---------- FOCUS tether tail (bright) ----------
    if (m_tetherVerts && m_tetherGeom)
    {
        m_tetherVerts->clear();

        if (m_focusMode)
        {
            const float L_units = (k - g).length();
            const float L_m = (m_unitsPerMeter > 1e-6f) ? (L_units / m_unitsPerMeter) : L_units;
            const float sag_m = std::min(m_ropeSagFactor * L_m, m_ropeSagMax);
            const float sag_units = sag_m * m_unitsPerMeter;

            const int samplesMain = std::max(8, m_ropeSamplesMain);
            std::vector<osg::Vec3> fullPts = sagCurvePoints(g, k, samplesMain, sag_units);

            // Instead of complex arc-length clipping: choose last segment by parameter,
            // but keep the *entire* curve visible via full tether already.
            // This avoids "partial sag" artifacts and keeps the focus tail stable.
            const float tailUnits = m_focusTetherLen_m * m_unitsPerMeter;
            float dist = 0.f;
            int startIdx = 0;
            for (int i = (int)fullPts.size() - 1; i > 0; --i)
            {
                dist += (fullPts[i] - fullPts[i - 1]).length();
                if (dist >= tailUnits)
                {
                    startIdx = i - 1;
                    break;
                }
            }

            std::vector<osg::Vec3> tailPts;
            tailPts.reserve(fullPts.size() - (size_t)startIdx);
            for (size_t i = (size_t)startIdx; i < fullPts.size(); ++i)
                tailPts.push_back(fullPts[i]);

            appendPolylineAsLines(m_tetherVerts.get(), tailPts);
        }
        else
        {
            // if not in focus mode, hide focus tether (full tether already shows it)
        }

        static_cast<osg::DrawArrays *>(m_tetherGeom->getPrimitiveSet(0))->setCount((int)m_tetherVerts->size());
        m_tetherVerts->dirty();
        m_tetherGeom->dirtyDisplayList();
        m_tetherGeom->dirtyBound();
    }

    // ---------- Bridles (sag) ----------
    if (m_bridleVerts && m_bridleGeom)
    {
        m_bridleVerts->clear();

        const size_t N = m_attachLocal.size();
        const size_t half = (N / 2);
        const int samplesBr = std::max(6, m_ropeSamplesBridle);

        for (size_t i = 0; i < N; ++i)
        {
            const osg::Vec3 aWorld = localToWorld(m_attachLocal[i]);

            const osg::Vec3 kWorld =
                (m_useTwoStageKnot && i < half) ? knotFrontWorld :
                (m_useTwoStageKnot)             ? knotRearWorld  :
                                                  knotWorld;

            const float brL_units = (aWorld - kWorld).length();
            const float brL_m = (m_unitsPerMeter > 1e-6f) ? (brL_units / m_unitsPerMeter) : brL_units;
            const float brSag_m = std::min(m_ropeSagFactor * brL_m, m_ropeSagMax);
            const float brSag_units = brSag_m * m_unitsPerMeter;

            std::vector<osg::Vec3> brPts = sagCurvePoints(kWorld, aWorld, samplesBr, brSag_units);
            appendPolylineAsLines(m_bridleVerts.get(), brPts);
        }

        static_cast<osg::DrawArrays *>(m_bridleGeom->getPrimitiveSet(0))->setCount((int)m_bridleVerts->size());
        m_bridleVerts->dirty();
        m_bridleGeom->dirtyDisplayList();
        m_bridleGeom->dirtyBound();
    }
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
