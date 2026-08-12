/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PlaceLabel.h"

#include <cover/coVRFileManager.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>

#include <osg/CopyOp>
#include <osg/LOD>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/PolygonOffset>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Geometry>
#include <osg/Geode>
#include <osgText/Font>
#include <osgText/Text>
#include <osg/Array>
#include <osg/AlphaFunc>

using namespace opencover;

static osg::Vec4 WHITE(1, 1, 1, 1);
static osg::Vec4 BLACK(0, 0, 0, 1);
static osg::Vec4 GRAY(1, 1, 1, 0.5);

static osg::Vec4 YELLOW(0.91, 0.68, 0, 1);
static osg::Vec4 GREEN(0, 0.533, 0.278, 1);

class DistanceScale : public osg::Transform
{
public:
    DistanceScale()
    {
    }

    DistanceScale(const DistanceScale &pat, const osg::CopyOp &copyop)
        : Transform(pat, copyop)
    {
    }

    virtual osg::Object *cloneType() const
    {
        return new DistanceScale();
    }
    virtual osg::Object *clone(const osg::CopyOp &copyop) const
    {
        return new DistanceScale(*this, copyop);
    }
    virtual bool isSameKindAs(const osg::Object *obj) const
    {
        return dynamic_cast<const DistanceScale *>(obj) != NULL;
    }
    virtual const char *className() const
    {
        return "DistanceScale";
    }
    virtual const char *libraryname() const
    {
        return "OpenCOVER";
    }

    virtual void accept(osg::NodeVisitor &nv)
    {
        if (nv.getVisitorType() == osg::NodeVisitor::CULL_VISITOR)
        {
            osg::CullStack *cs = dynamic_cast<osg::CullStack *>(&nv);
            if (cs)
            {
                osg::Vec3 eyePoint = cs->getEyeLocal();
                double d = eyePoint.length();
                double f = (d - d0) / (d1 - d0);
                double s = s0 + std::clamp(f, 0.0, 1.0) * (s1 - s0);
                _cachedMatrix.makeScale(s, s, s);
                _cachedInvMatrix.invert(_cachedMatrix);
            }
        }

        // now do the proper accept
        Transform::accept(nv);
    }

    bool computeLocalToWorldMatrix(osg::Matrix &matrix, osg::NodeVisitor *) const
    {
        if (_referenceFrame == RELATIVE_RF)
        {
            matrix.preMult(_cachedMatrix);
        }
        else // absolute
        {
            matrix = _cachedMatrix;
        }
        return true;
    }

    bool computeWorldToLocalMatrix(osg::Matrix &matrix, osg::NodeVisitor *) const
    {
        if (_referenceFrame == RELATIVE_RF)
        {
            matrix.postMult(_cachedInvMatrix);
        }
        else // absolute
        {
            matrix = _cachedInvMatrix;
        }
        return true;
    }

    void setDistances(double d0_, double s0_, double d1_, double s1_)
    {
        d0 = d0_;
        d1 = d1_;
        s0 = s0_;
        s1 = s1_;
    }

protected:
    virtual ~DistanceScale()
    {
    }

    mutable osg::Matrix _cachedMatrix;
    mutable osg::Matrix _cachedInvMatrix;

    double d0, s0, d1, s1;
};

PlaceLabel::PlaceLabel(const std::string &value, const osg::Vec3 &position, osg::ref_ptr<osg::Group> parent, const std::string &font, int size)
    : value(value)
    , position(position)
    , size(size)
{
    float s = size >= 3 ? 4.f : size >= 2 ? 2.f
                                          : 1.f;

    transform = new osg::MatrixTransform;
    parent->addChild(transform);

    lod = new osg::LOD;
    transform->addChild(lod);

    auto ds = new DistanceScale;
    float m = pow(s, 2);
    ds->setDistances(10000, 1.0, 10000 * m, m);
    lod->addChild(ds, 100, 10000 * m);

    // billboarding label
    billboard = new coBillboard();
    billboard->setNodeMask(billboard->getNodeMask() & ~Isect::Intersection & ~Isect::Pick);
    billboard->setMode(coBillboard::POINT_ROT_VIEWER);
    billboard->setAxis(osg::Vec3(0, 1, 0));
    billboard->setNormal(osg::Vec3(0, 0, 1));
    ds->addChild(billboard);

    geode = new osg::Geode();
    geode->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    billboard->addChild(geode);

    text = new osgText::Text();
    text->setAlignment(osgText::Text::CENTER_BASE_LINE);
    text->setColor(BLACK);
    const char *fn = coVRFileManager::instance()->getName("share/covise/fonts/DroidSans-Bold.ttf");
    text->setFont(fn);
    text->setCharacterSize(fontSize * s);
    // text->setFontResolution(128, 128);
    text->setText(value, osgText::String::ENCODING_UTF8);
    text->setPosition(osg::Vec3(0, 0, 0));
    // text->setBackdropType(osgText::Text::OUTLINE);
    // text->setBackdropColor(osg::Vec4(1, 1, 1, 1));
    geode->addDrawable(text);

    auto ss = geode->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    ss->setAttributeAndModes(new osg::PolygonOffset(-2.0f, -2.0f));

    // Add background
    float padding = 8.f;

    auto bg = new osg::Geometry();

    auto bb = text->getBoundingBox();
    float x0 = bb.xMin() - padding;
    float x1 = bb.xMax() + padding;
    float y0 = bb.yMin() - padding;
    float y1 = bb.yMax() + padding;

    auto verts = new osg::Vec3Array();
    verts->push_back(osg::Vec3(x0, y0, 0));
    verts->push_back(osg::Vec3(x1, y0, 0));
    verts->push_back(osg::Vec3(x1, y1, 0));
    verts->push_back(osg::Vec3(x0, y1, 0));
    bg->setVertexArray(verts);

    // constant color
    auto colors = new osg::Vec4Array();
    colors->push_back(YELLOW);
    bg->setColorArray(colors);
    bg->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::ref_ptr<osg::DrawElementsUInt> indices = new osg::DrawElementsUInt(GL_TRIANGLES);
    indices->push_back(0);
    indices->push_back(1);
    indices->push_back(2);
    indices->push_back(0);
    indices->push_back(2);
    indices->push_back(3);
    bg->addPrimitiveSet(indices);

    // Put it into the same screen-space “plane” as the text.
    // You can also set its rendering order via render bin if you need.
    auto bgGeode = new osg::Geode();
    bgGeode->addDrawable(bg);

    ss = bgGeode->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    ss->setAttributeAndModes(new osg::PolygonOffset(-1.0f, -1.0f));
    billboard->addChild(bgGeode);

    reposition();
}

void PlaceLabel::reposition()
{
    float distanceScale = 1.0f;
    transform->setMatrix(osg::Matrix::scale(distanceScale, distanceScale, distanceScale) * osg::Matrix::translate(position + osg::Vec3(0, 0, 100)));
}
