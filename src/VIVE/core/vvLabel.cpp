/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvLabel.h"
#include "vvPluginSupport.h"
#include "vvFileManager.h"
#include "vvSceneGraph.h"

#include "vvConfig.h"
#include "vvViewer.h"

#include <vsg/maths/mat4.h>
#include <vsg/nodes/MatrixTransform.h>

using namespace vive;
vvLabel::vvLabel(const char *name, float fontsize, float lineLen, vsg::vec4 fgc, vsg::vec4 bgc)
{

    //fprintf(stderr,"===== new vvLabel\n");
    moveToCam=false; // change default back to normal behavior (Uwe 2016)
    offset = lineLen;
    auto font = vvFileManager::instance()->loadFont(NULL);

    // unlighted geostate
    /*osg::Material* mtl = new osg::Material;
    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, vsg::vec4(0.9f, 0.9f, 0.9f, 1.0));
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, vsg::vec4(0.9f, 0.9f, 0.9f, 1.0));
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, vsg::vec4(0.9f, 0.9f, 0.9f, 1.0));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, vsg::vec4(1.0f, 1.0f, 1.0f, 1.0));
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);*/

    // position dsc
    posTransform = vsg::MatrixTransform::create();

    // billboarding label
    billboard = new vvBillboard();
    //billboard->setNodeMask(billboard->getNodeMask() & ~Isect::Intersection & ~Isect::Pick);
    //   billboard->setMode(vvBillboard::AXIAL_ROT);
    billboard->setMode(vvBillboard::POINT_ROT_WORLD);
    //   billboard->setMode(vvBillboard::POINT_ROT_EYE);

    vsg::vec3 zaxis(0, 1, 0);
    billboard->setAxis(zaxis);
    vsg::vec3 normal(0, 0, 1);
    billboard->setNormal(normal);

    /*label = new osg::Geode();

    //labelString->setMode(PFSTR_JUSTIFY, PFSTR_CENTER);
    //labelString->setColor(fgc[0], fgc[1], fgc[2], fgc[3]);
    //labelString->setGState(linegeostate);
    //labelString->setFont(font);
    //labelString->setString(name);
    //m= vsg::scale(fontsize, fontsize, fontsize);
    //labelString->setMat(m);

    text = new osgText::Text();
    text->setAlignment(osgText::Text::CENTER_BASE_LINE);
    text->setColor(vsg::vec4(fgc[0], fgc[1], fgc[2], fgc[3]));
    text->setFont(font);
    text->setCharacterSize(fontsize);
    text->setText(name, osgText::String::ENCODING_UTF8);
    text->setPosition(vsg::vec3(0, lineLen, 0));

    label->addDrawable(text);
    label->getOrCreateStateSet()->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

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
    vsg::vec3 p0 = vsg::vec3(0.0, 0.0, 0.0);

    //fprintf(stderr,"p0:[%f %f %f]\n", p0[0], p0[1], p0[2]);

    // line endpoint
    vsg::vec3 p1 = vsg::vec3(0.2, lineLen, 0.2);

    //fprintf(stderr,"p1:[%f %f %f]\n", p1[0], p1[1], p1[2]);

    lc = new vsg::vec3Array();
    lc->push_back(p0);
    lc->push_back(p1);

    vsg::vec4Array *fgcolor = new vsg::vec4Array();
    fgcolor->push_back(vsg::vec4(fgc[0], fgc[1], fgc[2], fgc[3]));

    lineGeoset = new vsg::Node();
    lineGeoset->setColorArray(fgcolor);
    lineGeoset->setColorBinding(vsg::Node::BIND_OVERALL);
    osg::DrawArrays *primitives = new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2);
    lineGeoset->setVertexArray(lc);
    lineGeoset->addPrimitiveSet(primitives);
    lineGeoset->setStateSet(linegeostate);

    vsg::vec4Array *bgcolor = new vsg::vec4Array();
    bgcolor->push_back(vsg::vec4(bgc[0], bgc[1], bgc[2], bgc[3]));
    bgcolor->push_back(vsg::vec4(bgc[0], bgc[1], bgc[2], bgc[3]));
    bgcolor->push_back(vsg::vec4(bgc[0], bgc[1], bgc[2], bgc[3]));
    bgcolor->push_back(vsg::vec4(bgc[0], bgc[1], bgc[2], bgc[3]));

    quadGeoset = new vsg::Node();

    osg::StateSet *quadgeostate = quadGeoset->getOrCreateStateSet();
    quadgeostate->setAttributeAndModes(mtl, osg::StateAttribute::ON);
    quadgeostate->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    if (bgc.a() < 0.9f)
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
    quadGeoset->setColorBinding(vsg::Node::BIND_OVERALL);

    primitives = new osg::DrawArrays(osg::PrimitiveSet::QUADS, 0, 4);

    vsg::vec3Array *normala = new vsg::vec3Array(1);
    (*normala)[0].set(0.0f, 0.0f, 1.0f);
    quadGeoset->setNormalArray(normala);
    quadGeoset->setNormalBinding(vsg::Node::BIND_OVERALL);
    qc = new vsg::vec3Array();
    qc->push_back(vsg::vec3(bb.xMin() - h, bb.yMin() - h, -0.1 * fontsize));
    qc->push_back(vsg::vec3(bb.xMax() + h, bb.yMin() - h, -0.1 * fontsize));
    qc->push_back(vsg::vec3(bb.xMax() + h, bb.yMax() + h, -0.1 * fontsize));
    qc->push_back(vsg::vec3(bb.xMin() - h, bb.yMax() + h, -0.1 * fontsize));
    quadGeoset->setVertexArray(qc);
    quadGeoset->addPrimitiveSet(primitives);

    // scene graph
    geode = new osg::Geode();
    vv->getScene()->addChild(posTransform);
    posTransform->addChild(billboard.get());
    billboard->addChild(label);
    billboard->addChild(geode);
    //billboard->addChild(vvSceneGraph::instance()->loadAxisGeode(1));
    geode->addDrawable(lineGeoset);
    geode->addDrawable(quadGeoset);*/
}

// lookForParent looks for the first parent
// of a given type
template <class T>
static T *
lookForParent(vsg::Group *child)
{
    /*int no_parents = child->getNumParents();
    int index;
    for (index = 0; index < no_parents; ++index)
    {
        vsg::Group *parent = child->getParent(index);
        if (dynamic_cast<T *>(parent))
        {
            return dynamic_cast<T *>(parent);
        }
    }*/
    return NULL;
}

// reAttachTo changes the parent of *this
// Assume that without the call to this function
// the matrix would be correct if the parent
// were vv->getScene().
// The matrix is changed so as not to
// move the label.

//!!!!!!!!!!!!
// Does not work correctly!!!!!
//!!!!!!!!!!!!
void
vvLabel::reAttachTo(vsg::Group *anchor)
{
    if (!anchor)
    {
        return;
    }

   /* vsg::dmat4 postMul;
    postMul.makeIdentity();

    vsg::MatrixTransform *parent = dynamic_cast<vsg::MatrixTransform *>(anchor);
    while (parent != NULL)
    {
        vsg::dmat4 parentMat = parent->matrix;
        postMul.mult(postMul, parentMat);
        parent = lookForParent<vsg::MatrixTransform>(parent);
    }

    vsg::dmat4 invPostMul;
    invPostMul.invert(postMul);

    vsg::dmat4 mat = posTransform->matrix;
    mat.mult(mat, invPostMul);
    posTransform->matrix = (mat);

    vsg::Group *previousParent = posTransform->getParent(0);
    if (anchor != posTransform->getParent(0))
    {
        anchor->addChild(posTransform);
        if (previousParent)
            previousParent->removeChild(posTransform);
    }*/
}

vvLabel::~vvLabel()
{
    //fprintf(stderr,"===== delete vvLabel\n");

    /*if (label && label->getNumParents())
    {
        vsg::Group *parent = dynamic_cast<vsg::Group *>(label->getParent(0));
        if (parent)
        {
            parent->removeChild(label);
        }
    }

    if (geode && geode->getNumParents())
    {
        vsg::Group *parent = dynamic_cast<vsg::Group *>(geode->getParent(0));
        parent->removeChild(geode);
    }

    if (billboard.get() && billboard->getNumParents())
    {
        vsg::Group *parent = dynamic_cast<vsg::Group *>(billboard->getParent(0));
        parent->removeChild(billboard.get());
    }

    if (posTransform && posTransform->getNumParents())
    {
        vsg::Group *parent = dynamic_cast<vsg::Group *>(posTransform->getParent(0));
        parent->removeChild(posTransform);
    }*/
}

void
vvLabel::setPosition(const vsg::dvec3 &pos)
{
    position = pos;
    keepPositionInScene = false;
    update();
}

void vvLabel::setPositionInScene(const vsg::dvec3 &pos)
{
    position = pos;
    keepPositionInScene = true;
    update();
}

void
vvLabel::keepDistanceFromCamera(bool enable, float distance)
{
    moveToCam = enable;
    distanceFromCamera = distance;
}

void
vvLabel::update()
{
    //fprintf(stderr,"===== vvLabel::setPosition\n");

    //fprintf(stderr,"setPos=[%f %f %f]\n", p[0], p[1], p[2]);

    //float depthFactor;
    //float viewerDist = -vv->getViewerMat().getTrans().y();
    vsg::dvec3 pos = position;

    if (keepPositionInScene)
    {
        pos *= vvSceneGraph::instance()->scaleFactor();
        pos = vv->getXformMat() * pos;
    }

    if (moveToCam)
    {
        pos = moveToCamera(pos, distanceFromCamera);
    }

    vsg::dmat4 m;
    m= vsg::translate(pos[0], pos[1], pos[2] /*+(offset*depthFactor)*/);
    //float depthFactor = vv->getInteractorScale(pos)*vv->getScale();

    //depthFactor = fabs((pos[1] + viewerDist)/viewerDist);
    if (!vvConfig::instance()->orthographic() && depthScale)
    {
        vsg::dvec3 viewerVec = getTrans(vv->getViewerMat());
        vsg::dvec3 viewerToPos = pos - viewerVec;

        double depthFactor = length(viewerToPos) / length(viewerVec);
        m = vsg::scale(depthFactor, depthFactor, depthFactor)*m;
    }
    posTransform->matrix = (m);
}

void
vvLabel::setLineLen(float l)
{
    //fprintf(stderr,"===== vvLabel::setLineLen\n");

    offset = l;

    // line startpoint
    //const osg::BoundingBox bb=label->getBoundingBox();
    //float h=0.1*bb.yMax()-bb.yMin();

    // line startpoint
    vsg::vec3 p0 = vsg::vec3(0.0, 0, 0.0);

    // line endpoint
    vsg::vec3 p1 = vsg::vec3(0.0, l, 0.0);
  /*  text->setPosition(vsg::vec3(0, l, 0));

    //cerr << bb.zMin() << "  " <<bb.zMax() << "  " << h << endl;
    (*lc)[0].set(p0);
    (*lc)[1].set(p1);
	lc->dirty();
    lineGeoset->setVertexArray(lc);
    lineGeoset->dirtyDisplayList();
    geode->dirtyBound();*/
}

void
vvLabel::setString(const char *name)
{
    //fprintf(stderr,"===== vvLabel::setString\n");

   /* text->setText(name, osgText::String::ENCODING_UTF8);

    //const osg::BoundingBox bb=label->getBoundingBox();
    osg::BoundingBox bb = text->getBound();
    float h = (0.1 * (bb.yMax() - bb.yMin()));

    //cerr << bb.xMin() << " x " <<bb.xMax() << "  " << h << endl;
    //cerr << bb.yMin() << " y " <<bb.yMax() << "  " << h << endl;
    //cerr << bb.zMin() << " z " <<bb.zMax() << "  " << h << endl;
    //(*qc)[3].set(vsg::vec3(bb.xMin()-h, bb.yMin()-h, -0.001*vv->getSceneSize()));
    //(*qc)[2].set(vsg::vec3(bb.xMax()+h, bb.yMin()-h, -0.001*vv->getSceneSize()));
    //(*qc)[1].set(vsg::vec3(bb.xMax()+h, bb.yMax()+h, -0.001*vv->getSceneSize()));
    //(*qc)[0].set(vsg::vec3(bb.xMin()-h, bb.yMax()+h, -0.001*vv->getSceneSize()));

    qc = new vsg::vec3Array();
    qc->push_back(vsg::vec3(bb.xMin() - h, bb.yMin() - h, -0.1 * text->getCharacterHeight()));
    qc->push_back(vsg::vec3(bb.xMax() + h, bb.yMin() - h, -0.1 * text->getCharacterHeight()));
    qc->push_back(vsg::vec3(bb.xMax() + h, bb.yMax() + h, -0.1 * text->getCharacterHeight()));
    qc->push_back(vsg::vec3(bb.xMin() - h, bb.yMax() + h, -0.1 * text->getCharacterHeight()));

    quadGeoset->setVertexArray(qc);
    quadGeoset->dirtyDisplayList();
    geode->dirtyBound();*/
}

void
vvLabel::setFGColor(vsg::vec4 fgc)
{
    //text->setColor(vsg::vec4(fgc[0], fgc[1], fgc[2], fgc[3]));
   /* text->setColor(fgc);

    vsg::vec4Array *fgcolor = new vsg::vec4Array();
    fgcolor->push_back(vsg::vec4(fgc[0], fgc[1], fgc[2], fgc[3]));
    lineGeoset->setColorArray(fgcolor);*/
}

void
vvLabel::show()
{
    //fprintf(stderr,"===== vvLabel::show\n");

    //posTransform->setTravMask(PFTRAV_DRAW, 1, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
    //posTransform->setTravMask(PFTRAV_ISECT, 1,PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
    //pfPrint(posTransform, PFTRAV_SELF|PFTRAV_DESCEND, PFPRINT_VB_DEBUG, NULL);
    //if (!posTransform->containsNode(billboard.get()))
    {
        posTransform->addChild(billboard);
    }
}

void
vvLabel::hide()
{
    //fprintf(stderr,"===== vvLabel::hide %x\n", posTransform);
    //posTransform->setTravMask(PFTRAV_DRAW, 0, PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
    //posTransform->setTravMask(PFTRAV_ISECT, 0,PFTRAV_SELF | PFTRAV_DESCEND, PF_SET);
    //pfPrint(posTransform, PFTRAV_SELF|PFTRAV_DESCEND, PFPRINT_VB_DEBUG, NULL);
    //if (posTransform && posTransform->children.find(billboard))
    {
        vv->removeChild(posTransform,billboard);
    }
}

void vvLabel::showLine()
{
    /*if (!geode->containsDrawable(lineGeoset))
    {
        geode->addDrawable(lineGeoset);
    }*/
}

void vvLabel::hideLine()
{
    /*if (geode->containsDrawable(lineGeoset))
    {
        geode->removeDrawable(lineGeoset);
    }*/
}
void vvLabel::setRotMode(vvBillboard::RotationMode mode)
{
    //_mode = mode;
    billboard->setMode(mode);
}

vsg::dvec3 vvLabel::moveToCamera(const vsg::dvec3 &point, float dist)
{
    vsg::dvec3 newPos;

    if (vvConfig::instance()->orthographic())
    {
        newPos = point;
        newPos[1] = vvViewer::instance()->getViewerPos()[1] + dist * vvSceneGraph::instance()->scaleFactor();
    }
    else
    {
        vsg::dvec3 newLabelMoveVec = point - vvViewer::instance()->getViewerPos();
        normalize(newLabelMoveVec);
        newPos = (vvViewer::instance()->getViewerPos() + newLabelMoveVec) * (double)dist * (double)vvSceneGraph::instance()->scaleFactor();
    }
    return newPos;
}
