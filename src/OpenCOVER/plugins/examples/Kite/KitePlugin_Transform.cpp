/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KitePlugin.h"

#include <osg/Matrix>

#include <algorithm>
#include <cmath>
#include <cstdio>

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

    // CSV trajectory is ground-station relative; move whole setup in world.
    pos += m_groundPos;

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

    osg::Matrix m = osg::Matrix::rotate(q) * osg::Matrix::translate(pos);
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

