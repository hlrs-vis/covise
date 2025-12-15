/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KitePlugin.h"

#include <config/CoviseConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRPluginSupport.h>

#include <osgDB/ReadFile>
#include <osg/Matrix>
#include <osg/Quat>
#include <osg/ShapeDrawable>
#include <osg/LineWidth>
#include <osg/Math>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace covise;
using namespace opencover;

KitePlugin::KitePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

KitePlugin::~KitePlugin()
{
    if (m_transform && m_transform->getNumParents() > 0)
    {
        m_transform->getParent(0)->removeChild(m_transform);
    }
    if (m_tetherGeode && m_tetherGeode->getNumParents() > 0)
    {
        m_tetherGeode->getParent(0)->removeChild(m_tetherGeode);
    }
    if (m_groundGeode && m_groundGeode->getNumParents() > 0)
    {
        m_groundGeode->getParent(0)->removeChild(m_groundGeode);
    }
    if (m_windGeode && m_windGeode->getNumParents() > 0)
    {
        m_windGeode->getParent(0)->removeChild(m_windGeode);
    }
}

bool KitePlugin::init()
{
    // Model path can be provided via config or environment variable.
    auto modelEntry = coCoviseConfig::getEntry("value", "KitePlugin.Model", "");
    m_csvPath = coCoviseConfig::getEntry("value", "KitePlugin.CSV", "");
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

    if (modelEntry.empty())
    {
        fprintf(stderr, "KitePlugin: no model path configured (KitePlugin.Model)\n");
        return false;
    }

    m_transform = new osg::MatrixTransform();

    if (!loadModel(modelEntry))
    {
        fprintf(stderr, "KitePlugin: could not load model '%s'\n", modelEntry.c_str());
        return false;
    }
    else
    {
        fprintf(stderr, "KitePlugin: loaded model '%s'\n", modelEntry.c_str());
    }

    cover->getObjectsRoot()->addChild(m_transform);

    // Tether line from ground to kite
    m_tetherGeode = new osg::Geode();
    m_tetherGeom = new osg::Geometry();
    auto tetherVerts = new osg::Vec3Array(2);
    (*tetherVerts)[0].set(0.f, 0.f, 0.f);
    (*tetherVerts)[1].set(0.f, 0.f, 0.f);
    m_tetherGeom->setVertexArray(tetherVerts);
    m_tetherGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2));
    auto tetherColors = new osg::Vec4Array(1);
    (*tetherColors)[0].set(0.f, 0.7f, 0.2f, 1.f);
    m_tetherGeom->setColorArray(tetherColors, osg::Array::BIND_OVERALL);
    m_tetherWidth = new osg::LineWidth(2.0f);
    auto tetherSS = m_tetherGeom->getOrCreateStateSet();
    tetherSS->setAttribute(m_tetherWidth, osg::StateAttribute::ON);
    m_tetherGeode->addDrawable(m_tetherGeom);
    cover->getObjectsRoot()->addChild(m_tetherGeode);

    // Ground station marker
    m_groundGeode = new osg::Geode();
    auto groundMarker = new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(0.f, 0.f, 0.f), 0.2f));
    groundMarker->setColor(osg::Vec4(0.2f, 0.2f, 0.2f, 1.f));
    m_groundGeode->addDrawable(groundMarker);
    cover->getObjectsRoot()->addChild(m_groundGeode);

    // Ground wind arrow
    m_windGeode = new osg::Geode();
    m_windGeom = new osg::Geometry();
    auto windVerts = new osg::Vec3Array(2);
    (*windVerts)[0].set(0.f, 0.f, 0.f);
    (*windVerts)[1].set(0.f, 0.f, 0.f);
    m_windGeom->setVertexArray(windVerts);
    m_windGeom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2));
    auto windColors = new osg::Vec4Array(1);
    (*windColors)[0].set(0.1f, 0.4f, 0.9f, 1.f);
    m_windGeom->setColorArray(windColors, osg::Array::BIND_OVERALL);
    auto windSS = m_windGeom->getOrCreateStateSet();
    windSS->setAttribute(new osg::LineWidth(2.0f), osg::StateAttribute::ON);
    m_windGeode->addDrawable(m_windGeom);
    cover->getObjectsRoot()->addChild(m_windGeode);

    if (!m_csvPath.empty())
    {
        parseCsv(m_csvPath);
        if (!m_frames.empty())
        {
            coVRAnimationManager::instance()->setNumTimesteps(static_cast<int>(m_frames.size()), this);
            coVRAnimationManager::instance()->enableAnimation(true);
        }
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

    // Try via file manager first (allows handlers to kick in).
    osg::Node *loaded = coVRFileManager::instance()->loadFile(resolved.c_str(), nullptr, m_transform.get());
    if (loaded)
    {
        m_model = loaded;
        return true;
    }

    // Some handlers attach directly to the parent and return nullptr.
    if (m_transform->getNumChildren() > childrenBefore)
    {
        m_model = m_transform->getChild(m_transform->getNumChildren() - 1);
        fprintf(stderr, "KitePlugin: model loaded via handler into transform (%u children)\n", m_transform->getNumChildren());
        return true;
    }

    // Fallback: direct osgDB load.
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

    auto optionalIdx = [&](const char *key) -> int {
        auto it = idx.find(key);
        if (it == idx.end())
            return -1;
        return static_cast<int>(it->second);
    };

    std::string line;
    size_t parsed = 0, skipped = 0;
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
        f.pos.x() = v;
        if (!parseNumber(toks[idx["kite_pos_north"]], v)) { ++skipped; continue; }
        f.pos.y() = v;
        if (!parseNumber(toks[idx["kite_height"]], v)) { ++skipped; continue; }
        f.pos.z() = v;
        if (!parseNumber(toks[idx["kite_0_roll"]], f.roll)) { ++skipped; continue; }
        if (!parseNumber(toks[idx["kite_0_pitch"]], f.pitch)) { ++skipped; continue; }
        if (!parseNumber(toks[idx["kite_0_yaw"]], f.yaw)) { ++skipped; continue; }

        int id;
        if ((id = optionalIdx("ground_tether_length")) >= 0 && id < static_cast<int>(toks.size()) && parseNumber(toks[id], v))
            f.tetherLength = v;
        if ((id = optionalIdx("ground_tether_force")) >= 0 && id < static_cast<int>(toks.size()) && parseNumber(toks[id], v))
        {
            f.tetherForce = v;
            if (v > m_maxForce)
                m_maxForce = v;
        }
        if ((id = optionalIdx("ground_tether_reelout_speed")) >= 0 && id < static_cast<int>(toks.size()) && parseNumber(toks[id], v))
            f.tetherReeloutSpeed = v;
        if ((id = optionalIdx("ground_wind_velocity")) >= 0 && id < static_cast<int>(toks.size()) && parseNumber(toks[id], v))
            f.windSpeed = v;
        if ((id = optionalIdx("ground_upwind_direction")) >= 0 && id < static_cast<int>(toks.size()) && parseNumber(toks[id], v))
            f.windDir = v;
        if ((id = optionalIdx("kite_elevation")) >= 0 && id < static_cast<int>(toks.size()) && parseNumber(toks[id], v))
            f.kiteElevation = v;
        if ((id = optionalIdx("kite_azimuth")) >= 0 && id < static_cast<int>(toks.size()) && parseNumber(toks[id], v))
            f.kiteAzimuth = v;
        if ((id = optionalIdx("kite_heading")) >= 0 && id < static_cast<int>(toks.size()) && parseNumber(toks[id], v))
            f.kiteHeading = v;
        if ((id = optionalIdx("kite_course")) >= 0 && id < static_cast<int>(toks.size()) && parseNumber(toks[id], v))
            f.kiteCourse = v;
        m_frames.push_back(f);
        ++parsed;
    }

    fprintf(stderr, "KitePlugin: loaded %zu frames (skipped %zu) from '%s'\n", m_frames.size(), skipped, path.c_str());
}

void KitePlugin::updateTransform(int frameIndex)
{
    if (!m_transform || m_frames.empty())
        return;

    frameIndex = std::max(0, std::min(frameIndex, static_cast<int>(m_frames.size() - 1)));
    const auto &f = m_frames[frameIndex];

    const double d2r = M_PI / 180.0;
    osg::Quat q;
    q.makeRotate(f.roll * d2r, osg::Vec3(1, 0, 0),
                 f.pitch * d2r, osg::Vec3(0, 1, 0),
                 f.yaw * d2r, osg::Vec3(0, 0, 1));

    osg::Matrix m = osg::Matrix::rotate(q) * osg::Matrix::translate(f.pos);
    m_transform->setMatrix(m);

    // Update tether geometry to connect ground to kite position.
    if (m_tetherGeom.valid())
    {
        auto verts = static_cast<osg::Vec3Array *>(m_tetherGeom->getVertexArray());
        if (verts && verts->size() >= 2)
        {
            (*verts)[0].set(0.f, 0.f, 0.f);
            (*verts)[1] = f.pos;
            verts->dirty();
            m_tetherGeom->dirtyDisplayList();
            m_tetherGeom->dirtyBound();
        }

        // Color/width based on force and slack/overstretch.
        const double dist = f.pos.length();
        const double tetherLen = f.tetherLength > 0.0 ? f.tetherLength : dist;
        double slack = 0.0;
        if (tetherLen > 1e-6)
            slack = (tetherLen - dist) / tetherLen; // positive: slack, negative: overstretch

        float r = 0.0f, g = 0.7f, b = 0.2f;
        if (slack > 0.05)
        {
            r = 0.2f; g = 0.4f; b = 1.0f; // slack -> bluish
        }
        else if (slack < -0.05)
        {
            r = 1.0f; g = 0.1f; b = 0.1f; // overstretch -> red
        }
        else
        {
            // map force to green→yellow→red
            double normF = (m_maxForce > 0.0) ? std::min(1.0, f.tetherForce / m_maxForce) : 0.0;
            r = static_cast<float>(normF);
            g = static_cast<float>(1.0 - 0.5 * normF);
            b = 0.1f;
        }
        auto colors = static_cast<osg::Vec4Array *>(m_tetherGeom->getColorArray());
        if (colors && !colors->empty())
        {
            (*colors)[0].set(r, g, b, 1.0f);
            colors->dirty();
        }
        if (m_tetherWidth.valid())
        {
            float width = 2.0f + static_cast<float>(std::min(3.0, (m_maxForce > 0.0 ? f.tetherForce / m_maxForce : 0.0) * 3.0));
            m_tetherWidth->setWidth(width);
        }
    }

    // Update ground wind arrow.
    if (m_windGeom.valid())
    {
        auto verts = static_cast<osg::Vec3Array *>(m_windGeom->getVertexArray());
        if (verts && verts->size() >= 2)
        {
            float scale = static_cast<float>(std::max(0.1, f.windSpeed * 0.1)); // scale down wind speed
            double dirRad = f.windDir * M_PI / 180.0;
            osg::Vec3 tip(std::sin(dirRad) * scale, std::cos(dirRad) * scale, 0.0f);
            (*verts)[0].set(0.f, 0.f, 0.f);
            (*verts)[1] = tip;
            verts->dirty();
            m_windGeom->dirtyDisplayList();
            m_windGeom->dirtyBound();
        }
    }
}

void KitePlugin::preFrame()
{
    if (!m_frames.empty())
    {
        int frame = coVRAnimationManager::instance()->getAnimationFrame();
        updateTransform(frame);
    }
}

COVERPLUGIN(KitePlugin)
