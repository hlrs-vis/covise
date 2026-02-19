/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "KitePlugin.h"

#include <cover/coVRPluginSupport.h>

#include <osg/Array>
#include <osg/Geode>
#include <osg/Group>
#include <osg/LineWidth>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/StateSet>
#include <osg/Vec4>
#include <osg/ComputeBoundsVisitor>

#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>

#include <algorithm>
#include <cmath>
#include <cstdio>

using namespace opencover;

namespace
{
osg::BoundingBox computeLocalBBox(osg::Node *n)
{
    osg::ComputeBoundsVisitor cbv;
    n->accept(cbv);
    return cbv.getBoundingBox();
}
} // namespace

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

    if (m_groundSizeAuto)
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

