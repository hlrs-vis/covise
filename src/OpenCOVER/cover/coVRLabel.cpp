/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRLabel.h"
#include "coVRPluginSupport.h"
#include "coVRFileManager.h"
#include "VRSceneGraph.h"

#include "coVRConfig.h"
#include "VRViewer.h"

#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/StateSet>
#include <osg/Material>
#include <osg/Geometry>
#include <osg/BoundingBox>
#include <osg/Geode>
#include <osgText/Font>
#include <osgText/Text>
#include <osg/Array>
#include <osg/AlphaFunc>
#include <osg/Version>
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
#define getBound getBoundingBox
#endif

using namespace opencover;
coVRLabel::coVRLabel(const char *name, float fontsize, float lineLen, osg::Vec4 fgc, osg::Vec4 bgc)
{

    //fprintf(stderr,"===== new coVRLabel\n");

    offset = lineLen;
    osg::ref_ptr<osgText::Font> font = coVRFileManager::instance()->loadFont(NULL);

    // unlighted geostate
    osg::Material *mtl = new osg::Material;
    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.9f, 0.9f, 0.9f, 1.0));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 1.0f, 1.0f, 1.0));
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);

    // position dsc
    posTransform = new osg::MatrixTransform;

    // billboarding label
    billboard = new coBillboard();
    billboard->setNodeMask(billboard->getNodeMask() & ~Isect::Intersection & ~Isect::Pick);
    //   billboard->setMode(coBillboard::AXIAL_ROT);
    billboard->setMode(coBillboard::POINT_ROT_WORLD);
    //   billboard->setMode(coBillboard::POINT_ROT_EYE);

    osg::Vec3 zaxis(0, 1, 0);
    billboard->setAxis(zaxis);
    osg::Vec3 normal(0, 0, 1);
    billboard->setNormal(normal);

    label = new osg::Geode();

    //labelString->setMode(PFSTR_JUSTIFY, PFSTR_CENTER);
    //labelString->setColor(fgc[0], fgc[1], fgc[2], fgc[3]);
    //labelString->setGState(linegeostate);
    //labelString->setFont(font);
    //labelString->setString(name);
    //m.makeScale(fontsize, fontsize, fontsize);
    //labelString->setMat(m);

    text = new osgText::Text();
    text->setDataVariance(osg::Object::DYNAMIC);
    text->setAlignment(osgText::Text::CENTER_BASE_LINE);
    text->setColor(osg::Vec4(fgc[0], fgc[1], fgc[2], fgc[3]));
    text->setFont(font);
    text->setCharacterSize(fontsize);
    text->setText(name, osgText::String::ENCODING_UTF8);
    text->setPosition(osg::Vec3(0, lineLen, 0));
    label->addDrawable(text);

    osg::StateSet *linegeostate = label->getOrCreateStateSet();
    linegeostate->setAttributeAndModes(mtl, osg::StateAttribute::ON);
    linegeostate->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::GEQUAL, 0.1f);

    if (fgc.a() < 1.0f)
    {
        linegeostate->setMode(GL_BLEND, osg::StateAttribute::ON);
        linegeostate->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    }
    else
    {
        linegeostate->setMode(GL_BLEND, osg::StateAttribute::OFF);
        linegeostate->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    }

    linegeostate->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);

    ///const pfBox *bb;
    ///bb=labelString->getBBox();

    osg::BoundingBox bb = text->getBound();
    float h = 0.1 * (bb.yMax() - bb.yMin());

    //fprintf(stderr,"===== h=%f\n", h);
    // line startpoint
    osg::Vec3 p0 = osg::Vec3(0.0, 0.0, 0.0);

    //fprintf(stderr,"p0:[%f %f %f]\n", p0[0], p0[1], p0[2]);

    // line endpoint
    osg::Vec3 p1 = osg::Vec3(0.2, lineLen, 0.2);

    //fprintf(stderr,"p1:[%f %f %f]\n", p1[0], p1[1], p1[2]);

    lc = new osg::Vec3Array();
    lc->push_back(p0);
    lc->push_back(p1);

    osg::Vec4Array *fgcolor = new osg::Vec4Array();
    fgcolor->push_back(osg::Vec4(fgc[0], fgc[1], fgc[2], fgc[3]));

    lineGeoset = new osg::Geometry();
    lineGeoset->setColorArray(fgcolor);
    lineGeoset->setColorBinding(osg::Geometry::BIND_OVERALL);
    osg::DrawArrays *primitives = new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2);
    lineGeoset->setVertexArray(lc);
    lineGeoset->addPrimitiveSet(primitives);
    lineGeoset->setStateSet(linegeostate);

    osg::Vec4Array *bgcolor = new osg::Vec4Array();
    bgcolor->push_back(osg::Vec4(bgc[0], bgc[1], bgc[2], bgc[3]));
    bgcolor->push_back(osg::Vec4(bgc[0], bgc[1], bgc[2], bgc[3]));
    bgcolor->push_back(osg::Vec4(bgc[0], bgc[1], bgc[2], bgc[3]));
    bgcolor->push_back(osg::Vec4(bgc[0], bgc[1], bgc[2], bgc[3]));

    quadGeoset = new osg::Geometry();

    osg::StateSet *quadgeostate = quadGeoset->getOrCreateStateSet();
    quadgeostate->setAttributeAndModes(mtl, osg::StateAttribute::ON);
    quadgeostate->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    if (bgc.a() < 1.0f)
    {
        quadgeostate->setMode(GL_BLEND, osg::StateAttribute::ON);
        quadgeostate->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    }
    else
    {
        quadgeostate->setMode(GL_BLEND, osg::StateAttribute::OFF);
        quadgeostate->setRenderingHint(osg::StateSet::OPAQUE_BIN);
    }

    quadgeostate->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);

    quadGeoset->setColorArray(bgcolor);
    quadGeoset->setColorBinding(osg::Geometry::BIND_OVERALL);

    primitives = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4);

    osg::Vec3Array *normala = new osg::Vec3Array(1);
    (*normala)[0].set(0.0f, 0.0f, 1.0f);
    quadGeoset->setNormalArray(normala);
    quadGeoset->setNormalBinding(osg::Geometry::BIND_OVERALL);
    qc = new osg::Vec3Array();
    qc->push_back(osg::Vec3(bb.xMin() - h, bb.yMin() - h, -0.001 * cover->getSceneSize()));
    qc->push_back(osg::Vec3(bb.xMax() + h, bb.yMin() - h, -0.001 * cover->getSceneSize()));
    qc->push_back(osg::Vec3(bb.xMax() + h, bb.yMax() + h, -0.001 * cover->getSceneSize()));
    qc->push_back(osg::Vec3(bb.xMin() - h, bb.yMax() + h, -0.001 * cover->getSceneSize()));
    quadGeoset->setVertexArray(qc);
    quadGeoset->addPrimitiveSet(primitives);

    // scene graph
    geode = new osg::Geode();
    cover->getScene()->addChild(posTransform);
    posTransform->addChild(billboard.get());
    billboard->addChild(label);
    billboard->addChild(geode);
    //billboard->addChild(VRSceneGraph::instance()->loadAxisGeode(1));
    geode->addDrawable(lineGeoset);
    geode->addDrawable(quadGeoset);
}

// lookForParent looks for the first parent
// of a given type
template <class T>
static T *
lookForParent(osg::Group *child)
{
    int no_parents = child->getNumParents();
    int index;
    for (index = 0; index < no_parents; ++index)
    {
        osg::Group *parent = child->getParent(index);
        if (dynamic_cast<T *>(parent))
        {
            return dynamic_cast<T *>(parent);
        }
    }
    return NULL;
}

// reAttachTo changes the parent of *this
// Assume that without the call to this function
// the matrix would be correct if the parent
// were cover->getScene().
// The matrix is changed so as not to
// move the label.

//!!!!!!!!!!!!
// Does not work correctly!!!!!
//!!!!!!!!!!!!
void
coVRLabel::reAttachTo(osg::Group *anchor)
{
    if (!anchor)
    {
        return;
    }

    osg::Matrix postMul;
    postMul.makeIdentity();

    osg::MatrixTransform *parent = dynamic_cast<osg::MatrixTransform *>(anchor);
    while (parent != NULL)
    {
        osg::Matrix parentMat = parent->getMatrix();
        postMul.mult(postMul, parentMat);
        parent = lookForParent<osg::MatrixTransform>(parent);
    }

    osg::Matrix invPostMul;
    invPostMul.invert(postMul);

    osg::Matrix mat = posTransform->getMatrix();
    mat.mult(mat, invPostMul);
    posTransform->setMatrix(mat);

    osg::Group *previousParent = posTransform->getParent(0);
    if (anchor != posTransform->getParent(0))
    {
        anchor->addChild(posTransform);
        if (previousParent)
            previousParent->removeChild(posTransform);
    }
}

coVRLabel::~coVRLabel()
{
    //fprintf(stderr,"===== delete coVRLabel\n");

    if (label && label->getNumParents())
    {
        osg::Group *parent = dynamic_cast<osg::Group *>(label->getParent(0));
        if (parent)
        {
            parent->removeChild(label);
        }
    }

    if (geode && geode->getNumParents())
    {
        osg::Group *parent = dynamic_cast<osg::Group *>(geode->getParent(0));
        parent->removeChild(geode);
    }

    if (billboard.get() && billboard->getNumParents())
    {
        osg::Group *parent = dynamic_cast<osg::Group *>(billboard->getParent(0));
        parent->removeChild(billboard.get());
    }

    if (posTransform && posTransform->getNumParents())
    {
        osg::Group *parent = dynamic_cast<osg::Group *>(posTransform->getParent(0));
        parent->removeChild(posTransform);
    }
}

void
coVRLabel::setPosition(const osg::Vec3 &pos)
{
    position = pos;
    keepPositionInScene = false;
    update();
}

void coVRLabel::setPositionInScene(const osg::Vec3 &pos)
{
    position = pos;
    keepPositionInScene = true;
    update();
}

void
coVRLabel::keepDistanceFromCamera(bool enable, float distance)
{
    moveToCam = enable;
    distanceFromCamera = distance;
}

void
coVRLabel::update()
{
    //fprintf(stderr,"===== coVRLabel::setPosition\n");

    //fprintf(stderr,"setPos=[%f %f %f]\n", p[0], p[1], p[2]);

    //float depthFactor;
    //float viewerDist = -cover->getViewerMat().getTrans().y();
    osg::Vec3 pos = position;

    if (keepPositionInScene)
    {
        pos *= VRSceneGraph::instance()->scaleFactor();
        pos = cover->getXformMat().preMult(pos);
    }

    if (moveToCam)
    {
        pos = moveToCamera(pos, distanceFromCamera);
    }

    osg::Vec3 viewerVec = cover->getViewerMat().getTrans();
    osg::Vec3 viewerToPos = pos - viewerVec;

    float depthFactor = viewerToPos.length() / viewerVec.length();
    osg::Matrix m;
    m.makeTranslate(pos[0], pos[1], pos[2] /*+(offset*depthFactor)*/);
    //float depthFactor = cover->getInteractorScale(pos)*cover->getScale();

    //depthFactor = fabs((pos[1] + viewerDist)/viewerDist);
    if (!coVRConfig::instance()->orthographic())
    {
        m.preMult(osg::Matrix::scale(depthFactor, depthFactor, depthFactor));
    }
    posTransform->setMatrix(m);
}

void
coVRLabel::setLineLen(float l)
{
    //fprintf(stderr,"===== coVRLabel::setLineLen\n");

    offset = l;

    // line startpoint
    //const osg::BoundingBox bb=label->getBoundingBox();
    //float h=0.1*bb.yMax()-bb.yMin();

    // line startpoint
    osg::Vec3 p0 = osg::Vec3(0.0, 0, 0.0);

    // line endpoint
    osg::Vec3 p1 = osg::Vec3(0.0, l, 0.0);
    text->setPosition(osg::Vec3(0, l, 0));

    //cerr << bb.zMin() << "  " <<bb.zMax() << "  " << h << endl;
    (*lc)[0].set(p0);
    (*lc)[1].set(p1);
    lineGeoset->setVertexArray(lc);
    lineGeoset->dirtyDisplayList();
    geode->dirtyBound();
}

void
coVRLabel::setString(const char *name)
{
    //fprintf(stderr,"===== coVRLabel::setString\n");

    text->setText(name, osgText::String::ENCODING_UTF8);

    //const osg::BoundingBox bb=label->getBoundingBox();
    osg::BoundingBox bb = text->getBound();
    float h = (0.1 * (bb.yMax() - bb.yMin()));

    //cerr << bb.xMin() << " x " <<bb.xMax() << "  " << h << endl;
    //cerr << bb.yMin() << " y " <<bb.yMax() << "  " << h << endl;
    //cerr << bb.zMin() << " z " <<bb.zMax() << "  " << h << endl;
    //(*qc)[3].set(osg::Vec3(bb.xMin()-h, bb.yMin()-h, -0.001*cover->getSceneSize()));
    //(*qc)[2].set(osg::Vec3(bb.xMax()+h, bb.yMin()-h, -0.001*cover->getSceneSize()));
    //(*qc)[1].set(osg::Vec3(bb.xMax()+h, bb.yMax()+h, -0.001*cover->getSceneSize()));
    //(*qc)[0].set(osg::Vec3(bb.xMin()-h, bb.yMax()+h, -0.001*cover->getSceneSize()));

    qc = new osg::Vec3Array();
    qc->push_back(osg::Vec3(bb.xMin() - h, bb.yMin() - h, -0.001 * cover->getSceneSize()));
    qc->push_back(osg::Vec3(bb.xMax() + h, bb.yMin() - h, -0.001 * cover->getSceneSize()));
    qc->push_back(osg::Vec3(bb.xMax() + h, bb.yMax() + h, -0.001 * cover->getSceneSize()));
    qc->push_back(osg::Vec3(bb.xMin() - h, bb.yMax() + h, -0.001 * cover->getSceneSize()));

    quadGeoset->setVertexArray(qc);
    quadGeoset->dirtyDisplayList();
    geode->dirtyBound();
}

void
coVRLabel::setFGColor(osg::Vec4 fgc)
{
    //text->setColor(osg::Vec4(fgc[0], fgc[1], fgc[2], fgc[3]));
    text->setColor(fgc);

    osg::Vec4Array *fgcolor = new osg::Vec4Array();
    fgcolor->push_back(osg::Vec4(fgc[0], fgc[1], fgc[2], fgc[3]));
    lineGeoset->setColorArray(fgcolor);
}

void
coVRLabel::show()
{
    //fprintf(stderr,"===== coVRLabel::show\n");

    //posTransform->setTravMask(PFTRAV_DRAW, 1, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
    //posTransform->setTravMask(PFTRAV_ISECT, 1,PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
    //pfPrint(posTransform, PFTRAV_SELF|PFTRAV_DESCEND, PFPRINT_VB_DEBUG, NULL);
    if (!posTransform->containsNode(billboard.get()))
    {
        posTransform->addChild(billboard.get());
        ;
    }
}

void
coVRLabel::hide()
{
    //fprintf(stderr,"===== coVRLabel::hide %x\n", posTransform);
    //posTransform->setTravMask(PFTRAV_DRAW, 0, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
    //posTransform->setTravMask(PFTRAV_ISECT, 0,PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
    //pfPrint(posTransform, PFTRAV_SELF|PFTRAV_DESCEND, PFPRINT_VB_DEBUG, NULL);
    if (posTransform && posTransform->containsNode(billboard.get()))
    {
        posTransform->removeChild(billboard.get());
    }
}

void coVRLabel::showLine()
{
    if (!geode->containsDrawable(lineGeoset))
    {
        geode->addDrawable(lineGeoset);
    }
}

void coVRLabel::hideLine()
{
    if (geode->containsDrawable(lineGeoset))
    {
        geode->removeDrawable(lineGeoset);
    }
}
void coVRLabel::setRotMode(coBillboard::RotationMode mode)
{
    //_mode = mode;
    billboard->setMode(mode);
}

osg::Vec3f coVRLabel::moveToCamera(const osg::Vec3f &point, float dist)
{
    osg::Vec3 newPos;

    if (coVRConfig::instance()->orthographic())
    {
        newPos = point;
        newPos[1] = VRViewer::instance()->getViewerPos()[1] + dist * VRSceneGraph::instance()->scaleFactor();
    }
    else
    {
        osg::Vec3 newLabelMoveVec = point - VRViewer::instance()->getViewerPos();
        newLabelMoveVec.normalize();
        newPos = VRViewer::instance()->getViewerPos() + newLabelMoveVec * dist * VRSceneGraph::instance()->scaleFactor();
    }
    return newPos;
}
