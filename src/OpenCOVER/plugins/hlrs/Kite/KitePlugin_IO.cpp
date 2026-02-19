/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KitePlugin.h"

#include <cover/coVRFileManager.h>

#include <osgDB/ReadFile>
#include <osg/Matrix>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unordered_map>

using namespace opencover;

namespace
{
osg::Matrix geoTransformMatrix(const osg::Vec3 &pos, float scale, float yawDeg)
{
    const double d2r = M_PI / 180.0;
    return osg::Matrix::scale(osg::Vec3(scale, scale, scale)) *
           osg::Matrix::rotate(yawDeg * d2r, osg::Vec3(0, 0, 1)) *
           osg::Matrix::translate(pos);
}
} // namespace

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

    const auto itElev = idx.find("kite_elevation");
    const auto itDist = idx.find("kite_distance");
    const int idxElev = (itElev != idx.end()) ? (int)itElev->second : -1;
    const int idxDist = (itDist != idx.end()) ? (int)itDist->second : -1;
    const bool autoDetectFrame = m_frameAuto && idxElev >= 0 && idxDist >= 0;
    int autoSamples = 0;
    int autoSame = 0;
    int autoOpp = 0;
    const int autoMaxSamples = 200;

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

        if (autoDetectFrame && autoSamples < autoMaxSamples)
        {
            const size_t need = (size_t)std::max(idxElev, idxDist);
            if (toks.size() > need)
            {
                double elev = 0.0;
                double dist = 0.0;
                if (parseNumber(toks[idxElev], elev) && parseNumber(toks[idxDist], dist))
                {
                    const double pred = std::sin(elev) * dist;
                    const double h = f.pos.z();
                    if (std::fabs(pred) > 1e-3 && std::fabs(h) > 1e-3)
                    {
                        ++autoSamples;
                        if ((pred > 0.0) == (h > 0.0))
                            ++autoSame;
                        else
                            ++autoOpp;
                    }
                }
            }
        }

        m_frames.push_back(f);
    }

    fprintf(stderr, "KitePlugin: loaded %zu frames (skipped %zu) from '%s'\n",
            m_frames.size(), skipped, path.c_str());

    if (autoDetectFrame && autoSamples >= 10)
    {
        const double sameRatio = (double)autoSame / (double)autoSamples;
        const double oppRatio = (double)autoOpp / (double)autoSamples;
        const bool detectedNed = (oppRatio >= 0.7);
        const bool detectedEnu = (sameRatio >= 0.7);

        if (detectedNed || detectedEnu)
        {
            const bool newUseNed = detectedNed;
            if (m_useNedFrame != newUseNed)
            {
                fprintf(stderr,
                        "KitePlugin: auto frame detect (elev vs height) samples=%d same=%d opp=%d -> %s (override)\n",
                        autoSamples, autoSame, autoOpp, newUseNed ? "NED" : "ENU");
            }
            else
            {
                fprintf(stderr,
                        "KitePlugin: auto frame detect (elev vs height) samples=%d same=%d opp=%d -> %s\n",
                        autoSamples, autoSame, autoOpp, newUseNed ? "NED" : "ENU");
            }
            m_useNedFrame = newUseNed;
        }
    }

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

    if (m_geoXform && m_geoModel && m_geoAutoCenter && m_haveGeoBBox && !m_frames.empty())
    {
        const auto &f0 = m_frames.front();
        osg::Vec3 pos = f0.pos * (m_unitsPerMeter * m_worldScale * m_globalPosScale);
        if (m_useNedFrame)
            pos = osg::Vec3(pos.y(), pos.x(), -pos.z());
        pos += m_groundPos;

        const double d2r = M_PI / 180.0;
        osg::Matrix rot = osg::Matrix::rotate(m_geoYawDeg * d2r, osg::Vec3(0, 0, 1));
        osg::Vec3 centerLocal = m_geoBBox.center();
        osg::Vec3 centerScaled = centerLocal * m_geoScale;
        osg::Vec3 centerRot = rot.preMult(centerScaled);

        osg::Vec3 newPos = m_geoPos;
        newPos.x() += pos.x() - centerRot.x();
        newPos.y() += pos.y() - centerRot.y();
        if (m_geoAutoGround)
            newPos.z() += m_groundPos.z() - (m_geoBBox.zMin() * m_geoScale);

        m_geoPos = newPos;
        m_geoXform->setMatrix(geoTransformMatrix(m_geoPos, m_geoScale, m_geoYawDeg));

        fprintf(stderr,
                "KitePlugin: geo auto-center applied: pos=(%.2f %.2f %.2f) target=(%.2f %.2f %.2f)\n",
                m_geoPos.x(), m_geoPos.y(), m_geoPos.z(),
                pos.x(), pos.y(), pos.z());
    }
}

