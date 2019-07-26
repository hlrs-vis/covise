/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "coVRBoxOfInterest.h"
#include "VariantPlugin.h"
#include <osg/Shape>
#include <osg/LineSegment>
#include <osgWidget/Box>
#include <osg/ShapeDrawable>
#include <osg/PolygonMode>

#include <net/tokenbuffer.h>
#include <cover/coVRFileManager.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRSelectionManager.h>
#include <osg/Material>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/LightModel>

using namespace vrui;
using namespace opencover;
using covise::TokenBuffer;

//-----------------------------------------------------------
coVRBoxOfInterest::coVRBoxOfInterest(VariantPlugin *plug, coTrackerButtonInteraction *_interactionA)
    : plugin(plug)
{

    parent = cover->getObjectsRoot();

    boiNode = new osg::MatrixTransform;
    boiNode->setName("myBox");
    bMt = new osg::MatrixTransform;
    bMt->setName("Boxgeometrie");

    osg::Vec3 center;
    center.set(0, 0, 0);
    length.set(10, 10, 10);
    osg::Vec3Array *points = new osg::Vec3Array;
    points->push_back(center + osg::Vec3(-length.x(), -length.y(), -length.z())); //0
    points->push_back(center + osg::Vec3(length.x(), -length.y(), -length.z())); //1
    points->push_back(center + osg::Vec3(length.x(), length.y(), -length.z())); //2
    points->push_back(center + osg::Vec3(-length.x(), length.y(), -length.z())); //3
    points->push_back(center + osg::Vec3(-length.x(), -length.y(), length.z())); //4
    points->push_back(center + osg::Vec3(length.x(), -length.y(), length.z())); //5
    points->push_back(center + osg::Vec3(length.x(), length.y(), length.z())); //6
    points->push_back(center + osg::Vec3(-length.x(), length.y(), length.z())); //7

    //  osg::Geode *quads = new osg::Geode();
    //  createQuads(quads,points);

    //Creating Box
    b = new osg::Box;
    b->set(center, length);

    osg::ShapeDrawable *sd = new osg::ShapeDrawable(b);
    osg::Geode *box = new osg::Geode();
    box->setName("Box");
    sd->setColor(osg::Vec4(0., 0., 1., 0.2));
    box->addDrawable(sd);
    box->setNodeMask(box->getNodeMask() & ~Isect::Intersection);
    osg::StateSet *stateSet = box->getOrCreateStateSet();
    loadTransparentGeostate(stateSet);

    bMt->addChild(box);

    //Creating Lines
    osg::Geode *lines = new osg::Geode();
    createLines(lines, points);
    //  boiNode->addChild(quads);
    bMt->addChild(lines);
    boiNode->addChild(bMt);
    bSphere = new interactorSpheres(boiNode, center, length, _interactionA);
    // createClipNode("myCNNode");
    cover->getObjectsRoot()->addChild(boiNode);
}
//-----------------------------------------------------------
coVRBoxOfInterest::~coVRBoxOfInterest()
{
    cout << "destr:" << this << ":" << boiNode << endl << endl << endl;
    this->showHide(false);
    this->parent->removeChild(boiNode);
    this->parent = 0;
    delete bSphere;
}
//-----------------------------------------------------------
osg::ClipNode *coVRBoxOfInterest::createClipNode(std::string cnName)
{
    osg::ClipNode *mycn = new osg::ClipNode;
    // cover->getObjectsRoot()->addChild(boiNode);

    for (int i = 0; i < cover->getNumClipPlanes(); i++)
    {
        cp[i] = cover->getClipPlane(i);
        mycn->addClipPlane(cp[i].get());
    }

    mycn->setName(cnName);
    cover->getObjectsRoot()->addChild(mycn);
    return mycn;
}
//-----------------------------------------------------------
void coVRBoxOfInterest::showHide(bool state)
{

    std::string path = coVRSelectionManager::generatePath(boiNode);
    TokenBuffer tb;
    tb << path;
    std::string pPath = path.substr(0, path.find_last_of(";"));
    tb << pPath;
    if (state)
        cover->sendMessage(plugin, "SGBrowser", PluginMessageTypes::SGBrowserShowNode, tb.getData().length(), tb.getData().data());
    else
        cover->sendMessage(plugin, "SGBrowser", PluginMessageTypes::SGBrowserHideNode, tb.getData().length(), tb.getData().data());
}
//-----------------------------------------------------------
//void coVRBoxOfInterest::setCoord(osg::Vec3 center,osg::Vec3 length )
//{
//
//     float xpos[]= {center.x() ,  center.x()            ,   center.x()+length.x(),  center.x()           ,   center.x()-length.x(),center.x()           ,center.x()           } ;
//     float ypos[]= {center.y() ,  center.y()-length.y() ,   center.y()          ,  center.y()+length.y() ,   center.y()           ,center.y()           ,center.y()           } ;
//     float zpos[]= {center.z() ,  center.z()            ,   center.z()          ,  center.z()            ,   center.z()           ,center.z()-length.z(),center.z()+length.z()} ;
//     float size[]= {2          ,                     1  ,                      1,                       1,                       1,                    1,                    1} ;
//     bSphere->setCoords(9,xpos,ypos,zpos,size);
//
//}
//-----------------------------------------------------------
void coVRBoxOfInterest::createLines(osg::Geode *node, osg::Vec3Array *vertices)
{
    node->setName("Lines");
    osg::Geometry *lineGeometry = new osg::Geometry();
    osg::StateSet *ss = lineGeometry->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    node->addDrawable(lineGeometry);
    lineGeometry->setVertexArray(vertices);
    osg::DrawElementsUInt *lineBase1 = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    lineBase1->push_back(0);
    lineBase1->push_back(5);
    lineBase1->push_back(1);
    lineBase1->push_back(4);
    osg::DrawElementsUInt *lineBase2 = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    lineBase2->push_back(1);
    lineBase2->push_back(6);
    lineBase2->push_back(2);
    lineBase2->push_back(5);
    osg::DrawElementsUInt *lineBase3 = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    lineBase3->push_back(2);
    lineBase3->push_back(7);
    lineBase3->push_back(3);
    lineBase3->push_back(6);
    osg::DrawElementsUInt *lineBase4 = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    lineBase4->push_back(0);
    lineBase4->push_back(7);
    lineBase4->push_back(4);
    lineBase4->push_back(3);
    osg::DrawElementsUInt *lineBase5 = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    lineBase5->push_back(0);
    lineBase5->push_back(2);
    lineBase5->push_back(1);
    lineBase5->push_back(3);
    osg::DrawElementsUInt *lineBase6 = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    lineBase6->push_back(4);
    lineBase6->push_back(6);
    lineBase6->push_back(5);
    lineBase6->push_back(7);
    osg::DrawElementsUInt *lineBase7 = new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
    lineBase7->push_back(0);
    lineBase7->push_back(1);
    lineBase7->push_back(1);
    lineBase7->push_back(2);
    lineBase7->push_back(2);
    lineBase7->push_back(3);
    lineBase7->push_back(3);
    lineBase7->push_back(0);
    lineBase7->push_back(4);
    lineBase7->push_back(5);
    lineBase7->push_back(5);
    lineBase7->push_back(6);
    lineBase7->push_back(6);
    lineBase7->push_back(7);
    lineBase7->push_back(7);
    lineBase7->push_back(4);
    lineBase7->push_back(0);
    lineBase7->push_back(4);
    lineBase7->push_back(1);
    lineBase7->push_back(5);
    lineBase7->push_back(2);
    lineBase7->push_back(6);
    lineBase7->push_back(3);
    lineBase7->push_back(7);

    lineGeometry->addPrimitiveSet(lineBase1);
    lineGeometry->addPrimitiveSet(lineBase2);
    lineGeometry->addPrimitiveSet(lineBase3);
    lineGeometry->addPrimitiveSet(lineBase4);
    lineGeometry->addPrimitiveSet(lineBase5);
    lineGeometry->addPrimitiveSet(lineBase6);
    lineGeometry->addPrimitiveSet(lineBase7);
}
//-----------------------------------------------------------
void coVRBoxOfInterest::createQuads(osg::Geode *node, osg::Vec3Array *vertices)
{
    node->setName("Quads");
    osg::Geometry *quadGeometry = new osg::Geometry();
    node->addDrawable(quadGeometry);
    quadGeometry->setVertexArray(vertices);
    osg::DrawElementsUInt *quad1 = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    quad1->push_back(0);
    quad1->push_back(1);
    quad1->push_back(2);
    quad1->push_back(3);
    osg::DrawElementsUInt *quad2 = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    quad2->push_back(1);
    quad2->push_back(2);
    quad2->push_back(6);
    quad2->push_back(5);
    osg::DrawElementsUInt *quad3 = new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    quad3->push_back(2);
    quad3->push_back(3);
    quad3->push_back(7);
    quad3->push_back(6);
    //    osg::DrawElementsUInt* quad4 =   new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0);
    //    quad1->push_back(0); quad4->push_back(1);    quad1->push_back(2);    quad1->push_back(3);

    quadGeometry->addPrimitiveSet(quad1);
    quadGeometry->addPrimitiveSet(quad2);
    quadGeometry->addPrimitiveSet(quad3);
    osg::StateSet *stateSet = node->getOrCreateStateSet();
    loadTransparentGeostate(stateSet);
}
//-----------------------------------------------------------
void coVRBoxOfInterest::loadTransparentGeostate(osg::StateSet *stateSet)
{
    osg::Material *material = new osg::Material();
    //material->setColorMode(osg::Material::DIFFUSE);
    //material->setAmbient     (osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 0.5f));
    //material->setDiffuse     (osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f));
    //material->setSpecular    (osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f));
    //material->setEmission    (osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f));
    //material->setShininess   (osg::Material::FRONT_AND_BACK, 16.0f);
    material->setAlpha(osg::Material::FRONT_AND_BACK, 0.5f);

    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::ALWAYS, 1.0);

    osg::BlendFunc *blendFunc = new osg::BlendFunc();
    blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    stateSet->setMode(GL_BLEND, osg::StateAttribute::ON);

    osg::LightModel *defaultLm;
    defaultLm = new osg::LightModel();
    defaultLm->setLocalViewer(true);
    defaultLm->setTwoSided(true);
    defaultLm->setColorControl(osg::LightModel::SINGLE_COLOR);

    stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateSet->setNestRenderBins(false);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    stateSet->setAttributeAndModes(alphaFunc, osg::StateAttribute::OFF);
    //stateSet->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    //stateSet->setAttributeAndModes(defaultLm, osg::StateAttribute::ON);
}
//-----------------------------------------------------------
void coVRBoxOfInterest::setMatrix(osg::Matrix mat)
{
    boiNode->setMatrix(mat);
}
//-----------------------------------------------------------
void coVRBoxOfInterest::setScale(osg::Matrix startMat, osg::Vec3 scale, TRANS direction)
{

    bMt->setMatrix(startMat * startMat.scale(scale));
    bSphere->updateSpherePos(scale, length, direction);
}
//-----------------------------------------------------------
bool coVRBoxOfInterest::isSensorActiv(std::string sensorName)
{
    return bSphere->isSensorActiv(sensorName);
}
//-----------------------------------------------------------
osg::Matrix coVRBoxOfInterest::getMat()
{
    return boiNode->getMatrix();
}
//-----------------------------------------------------------
osg::Matrix coVRBoxOfInterest::getinvMat()
{
    osg::Matrix tmp;
    tmp.invert(boiNode->getMatrix());
    return tmp;
}
//-----------------------------------------------------------
void coVRBoxOfInterest::setStartMatrix()
{
    bSphere->setStartMatrix();
}
//-----------------------------------------------------------
osg::MatrixTransform *coVRBoxOfInterest::getBoxGeoMt()
{
    return bMt;
}
//-----------------------------------------------------------
osg::MatrixTransform *coVRBoxOfInterest::getBoxCenterMt()
{
    return bSphere->getCenterMt();
}
//-----------------------------------------------------------
osg::MatrixTransform *coVRBoxOfInterest::getBoxaaMt()
{
    return bSphere->getaaMt();
}
//-----------------------------------------------------------
osg::MatrixTransform *coVRBoxOfInterest::getBoxbbMt()
{
    return bSphere->getbbMt();
}
//-----------------------------------------------------------
osg::MatrixTransform *coVRBoxOfInterest::getBoxccMt()
{
    return bSphere->getccMt();
}
//-----------------------------------------------------------
osg::MatrixTransform *coVRBoxOfInterest::getBoxddMt()
{
    return bSphere->getddMt();
}
//-----------------------------------------------------------
osg::MatrixTransform *coVRBoxOfInterest::getBoxeeMt()
{
    return bSphere->geteeMt();
}
//-----------------------------------------------------------
osg::MatrixTransform *coVRBoxOfInterest::getBoxffMt()
{
    return bSphere->getffMt();
}
//-----------------------------------------------------------
osg::Vec3 coVRBoxOfInterest::getLength()
{
    return length;
}
//-----------------------------------------------------------
void coVRBoxOfInterest::setLentgh(osg::Vec3 scale)
{
    length.set(length.x() * scale.x(), length.y() * scale.y(), length.z() * scale.z());
}
//-----------------------------------------------------------

void coVRBoxOfInterest::attachClippingPlanes(osg::Node *varNode, osg::ClipNode *mycn)
{
    osg::MatrixTransform *mt = dynamic_cast<osg::MatrixTransform *>(varNode);
    osg::Node::ParentList parents = mycn->getParents();
    if (mt)
    {
        int numChilds = mt->getNumChildren();
        for (int i = 0; i < numChilds; i++)
        {
            osg::Node *child = mt->getChild(i);
            if (child->getName() != "Label")
                mycn->addChild(child);
        }
        //     mt->removeChildren(0, numChilds);
        for (unsigned int i = 0; i <= mt->getNumChildren(); i++)
        {
            osg::Node *child = mt->getChild((numChilds - 1) - i);
            if (child->getName() != "Label")
                mt->removeChild(child);
        }

        mt->addChild(mycn);

        for (osg::Node::ParentList::iterator parent = parents.begin(); parent != parents.end(); ++parent)
        {
            // cover->getObjectsRoot()->addChild(mycn);
            //(*parent)->addChild(mycn); //clipping node
            //mycn->addChild(varNode);
            (*parent)->removeChild(mycn);
        }
    }
    //TODO: Der Clipping Node sollte zwischen MT und den Childs eingebaut werden!!!
}
//-----------------------------------------------------------

void coVRBoxOfInterest::releaseClippingPlanes(osg::Node *varNode, osg::ClipNode *mycn)
{
    osg::Node::ParentList parents = varNode->getParents();
    osg::MatrixTransform *mt = dynamic_cast<osg::MatrixTransform *>(varNode);
    for (osg::Node::ParentList::iterator parent = parents.begin(); parent != parents.end(); ++parent)
    {
        int numChilds = mycn->getNumChildren();
        for (int i = 0; i < numChilds; i++)
        {
            osg::Node *mychild = mycn->getChild(i);
            cout << "ChildName: " << mychild->getName().c_str() << endl;
            mt->addChild(mychild);

            //cover->getObjectsRoot()->removeChild(mycn);
        }

        mycn->removeChildren(0, numChilds);
        (*parent)->addChild(mycn);
    }
    mt->removeChild(mycn);
}
//-----------------------------------------------------------

void coVRBoxOfInterest::updateClippingPlanes()
{
    osg::Matrix centerMat = this->getGlobalMat(this->getBoxCenterMt());
    osg::Vec3 centerVec = centerMat.getTrans();

    osg::Matrix aaMat = this->getGlobalMat(this->getBoxaaMt());
    osg::Matrix ccMat = this->getGlobalMat(this->getBoxccMt());
    osg::Matrix bbMat = this->getGlobalMat(this->getBoxbbMt());
    osg::Matrix ddMat = this->getGlobalMat(this->getBoxddMt());
    osg::Matrix eeMat = this->getGlobalMat(this->getBoxeeMt());
    osg::Matrix ffMat = this->getGlobalMat(this->getBoxffMt());

    osg::Vec3d aaVec = aaMat.getTrans();
    osg::Vec3d ccVec = ccMat.getTrans();
    osg::Vec3d bbVec = bbMat.getTrans();
    osg::Vec3d ddVec = ddMat.getTrans();
    osg::Vec3d eeVec = eeMat.getTrans();
    osg::Vec3d ffVec = ffMat.getTrans();

    osg::Vec3d diraa = (aaVec - centerVec);
    osg::Vec3d dircc = (ccVec - centerVec);
    osg::Vec3d dirbb = (bbVec - centerVec);
    osg::Vec3d dirdd = (ddVec - centerVec);
    osg::Vec3d diree = (eeVec - centerVec);
    osg::Vec3d dirff = (ffVec - centerVec);

    osg::Vec3d vecaa = -(centerVec + diraa);
    osg::Vec3d veccc = -(centerVec + dircc);
    osg::Vec3d vecbb = -(centerVec + dirbb);
    osg::Vec3d vecdd = -(centerVec + dirdd);
    osg::Vec3d vecee = -(centerVec + diree);
    osg::Vec3d vecff = -(centerVec + dirff);

    osg::Vec3d tmpaa = diraa / diraa.length();
    float abst_aa = vecaa * tmpaa;
    osg::Vec3d tmpcc = dircc / dircc.length();
    float abst_cc = veccc * tmpcc;
    osg::Vec3d tmpbb = dirbb / dirbb.length();
    float abst_bb = vecbb * tmpbb;
    osg::Vec3d tmpdd = dirdd / dirdd.length();
    float abst_dd = vecdd * tmpdd;
    osg::Vec3d tmpee = diree / diree.length();
    float abst_ee = vecee * tmpee;
    osg::Vec3d tmpff = dirff / dirff.length();
    float abst_ff = vecff * tmpff;

    osg::Plane plane[6];
    plane[0].set(-tmpaa, -abst_aa);
    plane[1].set(-tmpcc, -abst_cc);
    plane[2].set(-tmpbb, -abst_bb);
    plane[3].set(-tmpdd, -abst_dd);
    plane[4].set(-tmpee, -abst_ee);
    plane[5].set(-tmpff, -abst_ff);

    for (int i=0; i<6; ++i)
    {
        if (cp[i])
            cp[i]->setClipPlane(plane[i]);
    }
}
//-----------------------------------------------------------

osg::Matrix coVRBoxOfInterest::getGlobalMat(osg::Node *node)
{
    osg::Node *currentNode = node;
    osg::Matrix startBaseMat;
    osg::Matrix dcsMat;
    startBaseMat.makeIdentity();
    //osg::MatrixTransform* mtmat = boiNode;
    while (currentNode != NULL)
    {
        if (dynamic_cast<osg::MatrixTransform *>(currentNode))
        {
            dcsMat = ((osg::MatrixTransform *)currentNode)->getMatrix();
            startBaseMat.postMult(dcsMat);
        }
        if (currentNode->getNumParents() > 0 && currentNode->getParent(0) != cover->getObjectsRoot())
            currentNode = currentNode->getParent(0);
        else
            currentNode = NULL;
    }
    osg::Matrix CompleteMat = startBaseMat;
    return CompleteMat;
}
//-----------------------------------------------------------
interactorSpheres::interactorSpheres(osg::Node *n, osg::Vec3 center, osg::Vec3 length, coTrackerButtonInteraction *_interactionA)
{

    mt = new osg::MatrixTransform;
    centerMt = new osg::MatrixTransform;
    aaMt = new osg::MatrixTransform;
    bbMt = new osg::MatrixTransform;
    ccMt = new osg::MatrixTransform;
    ddMt = new osg::MatrixTransform;
    eeMt = new osg::MatrixTransform;
    ffMt = new osg::MatrixTransform;
    mt->setName("InteractorSpheres");
    centerMt->setName("CenterSphere");
    aaMt->setName("aaSphere");
    bbMt->setName("bbSphere");
    ccMt->setName("ccSphere");
    bbMt->setName("ddSphere");
    eeMt->setName("eeSphere");
    ffMt->setName("ffSphere");
    osg::MatrixTransform *m = dynamic_cast<osg::MatrixTransform *>(n);
    if (m)
    {

        hint = new osg::TessellationHints();
        float ratio = 0.5;
        hint->setDetailRatio(ratio);

        //Center Sphere
        centerSphere = new osg::Sphere(center, 2.);
        osg::ShapeDrawable *centerSphereDrawable = new osg::ShapeDrawable(centerSphere, hint.get());
        centerSphereDrawable->setColor(osg::Vec4(1., 0., 0., 1.0f));
        osg::Geode *centerGeode = new osg::Geode();
        osg::StateSet *stateSet = centerGeode->getOrCreateStateSet();
        setStateSet(stateSet);

        centerSensor = new mySensor(centerGeode, "center", _interactionA, centerSphereDrawable);
        centerGeode->setName("centerSphere");
        centerGeode->addDrawable(centerSphereDrawable);
        centerMt->addChild(centerGeode);
        mt->addChild(centerMt);

        //aa-Sphere (see header)
        //TODOaaSphere = new osg::Sphere(osg::Vec3(center.x(), center.y()-length.y(), center.z()), 1.);
        aaSphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.);
        osg::ShapeDrawable *aaSphereDrawable = new osg::ShapeDrawable(aaSphere, hint.get());
        aaSphereDrawable->setColor(osg::Vec4(1., 0., 0., 1.0f));
        osg::Geode *aaGeode = new osg::Geode();
        stateSet = aaGeode->getOrCreateStateSet();
        setStateSet(stateSet);

        aaSensor = new mySensor(aaGeode, "aa", _interactionA, aaSphereDrawable);
        aaGeode->setName("aaSphere");
        aaGeode->addDrawable(aaSphereDrawable);
        aaMt->addChild(aaGeode);
        aaMt->setMatrix(aaMt->getMatrix().translate(osg::Vec3(center.x(), center.y() - length.y(), center.z())));
        mt->addChild(aaMt);

        //bb-Sphere (see header)
        bbSphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.);
        osg::ShapeDrawable *bbSphereDrawable = new osg::ShapeDrawable(bbSphere, hint.get());
        bbSphereDrawable->setColor(osg::Vec4(1., 0., 0., 1.0f));
        osg::Geode *bbGeode = new osg::Geode();
        stateSet = bbGeode->getOrCreateStateSet();
        setStateSet(stateSet);

        bbSensor = new mySensor(bbGeode, "bb", _interactionA, bbSphereDrawable);
        bbGeode->setName("bbSphere");
        bbGeode->addDrawable(bbSphereDrawable);
        bbMt->addChild(bbGeode);
        bbMt->setMatrix(bbMt->getMatrix().translate(osg::Vec3(center.x() - length.x(), center.y(), center.z())));
        mt->addChild(bbMt);

        //cc-Sphere (see header)
        ccSphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.);
        osg::ShapeDrawable *ccSphereDrawable = new osg::ShapeDrawable(ccSphere, hint.get());
        ccSphereDrawable->setColor(osg::Vec4(1., 0., 0., 1.0f));
        osg::Geode *ccGeode = new osg::Geode();
        stateSet = ccGeode->getOrCreateStateSet();
        setStateSet(stateSet);

        ccSensor = new mySensor(ccGeode, "cc", _interactionA, ccSphereDrawable);
        ccGeode->setName("ccSphere");
        ccGeode->addDrawable(ccSphereDrawable);
        ccMt->addChild(ccGeode);
        ccMt->setMatrix(ccMt->getMatrix().translate(osg::Vec3(center.x(), center.y() + length.y(), center.z())));
        mt->addChild(ccMt);

        //dd-Sphere (see header)
        ddSphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.);
        osg::ShapeDrawable *ddSphereDrawable = new osg::ShapeDrawable(ddSphere, hint.get());
        ddSphereDrawable->setColor(osg::Vec4(1., 0., 0., 1.0f));
        osg::Geode *ddGeode = new osg::Geode();
        stateSet = ddGeode->getOrCreateStateSet();
        setStateSet(stateSet);

        ddSensor = new mySensor(ddGeode, "dd", _interactionA, ddSphereDrawable);
        ddGeode->setName("ddSphere");
        ddGeode->addDrawable(ddSphereDrawable);
        ddMt->addChild(ddGeode);
        ddMt->setMatrix(ddMt->getMatrix().translate(osg::Vec3(center.x() + length.x(), center.y(), center.z())));
        mt->addChild(ddMt);

        //ee-Sphere (see header)
        eeSphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.);
        osg::ShapeDrawable *eeSphereDrawable = new osg::ShapeDrawable(eeSphere, hint.get());
        eeSphereDrawable->setColor(osg::Vec4(1., 0., 0., 1.0f));
        osg::Geode *eeGeode = new osg::Geode();
        stateSet = eeGeode->getOrCreateStateSet();
        setStateSet(stateSet);

        eeSensor = new mySensor(eeGeode, "ee", _interactionA, eeSphereDrawable);
        eeGeode->setName("eeSphere");
        eeGeode->addDrawable(eeSphereDrawable);
        eeMt->addChild(eeGeode);
        eeMt->setMatrix(eeMt->getMatrix().translate(osg::Vec3(center.x(), center.y(), center.z() + length.z())));
        mt->addChild(eeMt);

        //ff-Sphere (see header)
        ffSphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.);
        osg::ShapeDrawable *ffSphereDrawable = new osg::ShapeDrawable(ffSphere, hint.get());
        ffSphereDrawable->setColor(osg::Vec4(1., 0., 0., 1.0f));
        osg::Geode *ffGeode = new osg::Geode();
        stateSet = ffGeode->getOrCreateStateSet();
        setStateSet(stateSet);

        ffSensor = new mySensor(ffGeode, "ff", _interactionA, ffSphereDrawable);
        ffGeode->setName("ffSphere");
        ffGeode->addDrawable(ffSphereDrawable);
        ffMt->addChild(ffGeode);
        ffMt->setMatrix(ffMt->getMatrix().translate(osg::Vec3(center.x(), center.y(), center.z() - length.z())));
        mt->addChild(ffMt);

        m->addChild(mt);
    }
}
//-----------------------------------------------------------
interactorSpheres::~interactorSpheres()
{
    delete centerSensor;
    delete aaSensor;
    delete bbSensor;
    delete ccSensor;
    delete ddSensor;
    delete eeSensor;
    delete ffSensor;
}
void interactorSpheres::setStateSet(osg::StateSet *stateSet)
{
    osg::Material *material = new osg::Material();
    material->setColorMode(osg::Material::DIFFUSE);
    //material->setAmbient     (osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f));
    //material->setSpecular    (osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f));
    //material->setEmission    (osg::Material::FRONT_AND_BACK, osg::Vec4(1.0f, 0.0f, 0.0f, 0.5f));
    //material->setShininess   (osg::Material::FRONT_AND_BACK, 16.0f);
    // material->setAlpha       (osg::Material::FRONT_AND_BACK,  0.5f);

    //osg::AlphaFunc* alphaFunc = new osg::AlphaFunc();
    // alphaFunc->setFunction(osg::AlphaFunc::ALWAYS,1.0);

    //  osg::BlendFunc* blendFunc = new osg::BlendFunc();
    //  blendFunc->setFunction(osg::BlendFunc::SRC_ALPHA, osg::BlendFunc::ONE_MINUS_SRC_ALPHA);
    //  stateSet->setMode(GL_BLEND, osg::StateAttribute::OFF);

    osg::LightModel *defaultLm;
    defaultLm = new osg::LightModel();
    defaultLm->setLocalViewer(true);
    defaultLm->setTwoSided(true);
    defaultLm->setColorControl(osg::LightModel::SINGLE_COLOR);

    //stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateSet->setAttributeAndModes(material, osg::StateAttribute::ON);
    // stateSet->setMode(GL_LIGHTING,osg::StateAttribute::OFF);
    // stateSet->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);
    //stateSet->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(defaultLm, osg::StateAttribute::ON);
}
//-----------------------------------------------------------
void interactorSpheres::updateSpheres(osg::Vec3 center, osg::Vec3)
{
    centerSphere->set(center, 2.);
}
//-----------------------------------------------------------

bool interactorSpheres::isSensorActiv(std::string sensorName)
{
    if (sensorName == centerSensor->getSensorName())
    {
        if (centerSensor->isSensorActive())
            return true;
    }
    if (sensorName == aaSensor->getSensorName())
    {
        if (aaSensor->isSensorActive())
            return true;
    }
    if (sensorName == bbSensor->getSensorName())
    {
        if (bbSensor->isSensorActive())
            return true;
    }
    if (sensorName == ccSensor->getSensorName())
    {
        if (ccSensor->isSensorActive())
            return true;
    }
    if (sensorName == ddSensor->getSensorName())
    {
        if (ddSensor->isSensorActive())
            return true;
    }
    if (sensorName == eeSensor->getSensorName())
    {
        if (eeSensor->isSensorActive())
            return true;
    }
    if (sensorName == ffSensor->getSensorName())
    {
        if (ffSensor->isSensorActive())
            return true;
    }
    return false;
}
//-----------------------------------------------------------
void interactorSpheres::updateSpherePos(osg::Vec3 scaleVec, osg::Vec3 size, TRANS direction)
{
    float xSize = size.x();
    float ySize = size.y();
    float zSize = size.z();
    osg::Vec3 transVec(xSize - (scaleVec.x() * xSize), ySize - (scaleVec.y() * ySize), zSize - (scaleVec.z() * zSize));

    // centerMt->setMatrix(starCenterMat * centerMt->getMatrix().scale(tmpVec));
    if (direction == XTRANS)
    {
        aaMt->setMatrix(startaaMat * startaaMat.translate(transVec));
        ccMt->setMatrix(startccMat * startccMat.translate(-transVec));
    }
    if (direction == YTRANS)
    {
        bbMt->setMatrix(startbbMat * startbbMat.translate(transVec));
        ddMt->setMatrix(startddMat * startddMat.translate(-transVec));
    }
    if (direction == ZTRANS)
    {
        eeMt->setMatrix(starteeMat * starteeMat.translate(-transVec));
        ffMt->setMatrix(startffMat * startffMat.translate(transVec));
    }
}
//-----------------------------------------------------------
void interactorSpheres::setStartMatrix()
{
    startCenterMat = centerMt->getMatrix();
    startaaMat = aaMt->getMatrix();
    startbbMat = bbMt->getMatrix();
    startccMat = ccMt->getMatrix();
    startddMat = ddMt->getMatrix();
    starteeMat = eeMt->getMatrix();
    startffMat = ffMt->getMatrix();
}
//-----------------------------------------------------------
osg::MatrixTransform *interactorSpheres::getaaMt()
{
    return aaMt;
}
//-----------------------------------------------------------
osg::MatrixTransform *interactorSpheres::getCenterMt()
{
    return centerMt;
}
//-----------------------------------------------------------
osg::MatrixTransform *interactorSpheres::getbbMt()
{
    return bbMt;
}
//-----------------------------------------------------------
osg::MatrixTransform *interactorSpheres::getccMt()
{
    return ccMt;
}
//-----------------------------------------------------------
osg::MatrixTransform *interactorSpheres::getddMt()
{
    return ddMt;
}
//-----------------------------------------------------------
osg::MatrixTransform *interactorSpheres::geteeMt()
{
    return eeMt;
}
//-----------------------------------------------------------
osg::MatrixTransform *interactorSpheres::getffMt()
{
    return ffMt;
}
//-----------------------------------------------------------
//-----------------------------------------------------------
mySensor::mySensor(osg::Node *node, std::string name, coTrackerButtonInteraction *_interactionA, osg::ShapeDrawable *cSphDr)
    : coPickSensor(node)
{
    sensorName = name;
    isActive = false;
    _interA = _interactionA;
    shapDr = cSphDr;
    VariantPlugin::plugin->sensorList.append(this);
}
mySensor::~mySensor()
{
    if (VariantPlugin::plugin->sensorList.find(this))
        VariantPlugin::plugin->sensorList.remove();
}
//-----------------------------------------------------------
void mySensor::activate()
{
    isActive = true;
    cout << "---Activate--" << sensorName.c_str() << endl;
    coInteractionManager::the()->registerInteraction(_interA);
    shapDr->setColor(osg::Vec4(1., 1., 0., 1.0f));
    //TODO change color
}
//-----------------------------------------------------------
void mySensor::disactivate()
{
    cout << "---Disactivate--" << sensorName.c_str() << endl;
    isActive = false;
    coInteractionManager::the()->unregisterInteraction(_interA);
    shapDr->setColor(osg::Vec4(1., 0., 0., 1.0f));
    //TODO change color
}
//-----------------------------------------------------------

std::string mySensor::getSensorName()
{
    return sensorName;
}

bool mySensor::isSensorActive()
{
    if (isActive)
        return true;
    else
        return false;
}
//-----------------------------------------------------------

//------------------------------------------------------------------------------------------------------------------------------
void interactorSpheres::printMatrix(osg::Matrix ma)
{
    cout << "/----------------------- " << endl;
    cout << ma(0, 0) << " " << ma(0, 1) << " " << ma(0, 2) << " " << ma(0, 3) << endl;
    cout << ma(1, 0) << " " << ma(1, 1) << " " << ma(1, 2) << " " << ma(1, 3) << endl;
    cout << ma(2, 0) << " " << ma(2, 1) << " " << ma(2, 2) << " " << ma(2, 3) << endl;
    cout << ma(3, 0) << " " << ma(3, 1) << " " << ma(3, 2) << " " << ma(3, 3) << endl;
    cout << "/-----------------------  " << endl;
}
