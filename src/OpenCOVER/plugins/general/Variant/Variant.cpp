/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <osg/MatrixTransform>
#include "Variant.h"
#include <net/tokenbuffer.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRSelectionManager.h>
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/PluginMessageTypes.h>
#include "VariantPlugin.h"
using namespace covise;
using namespace opencover;

//------------------------------------------------------------------------------
Variant *Variant::variantClass = NULL;
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

Variant::Variant(std::string var_Name, osg::Node *node, osg::Node::ParentList pa, ui::Menu *Variant_menu, coTUITab *VariantPluginTab, int numVar, QDomDocument *xml, QDomElement *qDE_V, coVRBoxOfInterest *boi, bool default_state)
{
    myboi = boi;
    variantClass = this;
    parents = pa;
    varName = var_Name;
    VarNode = new osg::MatrixTransform;
    VarNode->setName("Variant:"+var_Name);

    if (node)
    {
        attachNode(node);
    }

    origin_matrix = VarNode->getMatrix();
    createVRLabel();
    ui = new VariantUI(var_Name, Variant_menu, VariantPluginTab);
    ui->setPosTUIItems(numVar);
    //XML
    xmlfile = xml;
    qDE_Variant = qDE_V;

    QDomElement qDE_Variant_item = xmlfile->createElement(varName.c_str());
    QDomElement qDE_Variant_item_visible = xmlfile->createElement("visible");
    qDE_Variant_item_visible.setAttribute("state", default_state);
    qDE_Variant_item.appendChild(qDE_Variant_item_visible);
    QDomElement qDE_Variant_translations = xmlfile->createElement("transform");
    qDE_Variant_translations.setAttribute("X", 0);
    qDE_Variant_translations.setAttribute("Y", 0);
    qDE_Variant_translations.setAttribute("Z", 0);
    qDE_Variant_item.appendChild(qDE_Variant_translations);
    qDE_Variant->appendChild(qDE_Variant_item);
    std::string cnName = "_clNode_";
    cnName.append(varName);
    cn = boi->createClipNode(cnName);
}
//------------------------------------------------------------------------------

Variant::~Variant()
{
    delete VarLabel;
    delete ui;
}
//------------------------------------------------------------------------------

void Variant::AddToScenegraph()
{
    for (osg::Node::ParentList::iterator parent = parents.begin(); parent != parents.end(); ++parent)
        (*parent)->addChild(VarNode);
}
//------------------------------------------------------------------------------

void Variant::removeFromScenegraph(osg::Node *node)
{
    for (osg::Node::ParentList::iterator parent = parents.begin(); parent != parents.end(); ++parent)
    {
        cout << "NodeName " << node->getName() << endl;
        (*parent)->addChild(node);
        (*parent)->removeChild(VarNode);
        cout << "ParentName: " << (*parent)->getName() << endl;
    }
}
//------------------------------------------------------------------------------

void Variant::releaseNode(osg::Node *node)
{
    attachedNodesList.remove(node);
}

//------------------------------------------------------------------------------

void Variant::attachNode(osg::Node *node)
{
    if (node)
    {
        node->ref();
        while (node->getNumParents() > 0)
            node->getParent(0)->removeChild(node);
        attachedNodesList.push_back(node);
        cover->getObjectsRoot()->removeChild(node);
        VarNode->addChild(node);
        node->unref();
    }
}

//------------------------------------------------------------------------------

osg::Node *Variant::getNode()
{
    return VarNode;
}
//------------------------------------------------------------------------------

osg::Matrix Variant::getOriginMatrix()
{
    return origin_matrix;
}
//------------------------------------------------------------------------------

int Variant::numNodes()
{
    return attachedNodesList.size();
}
//------------------------------------------------------------------------------

std::list<osg::Node *> Variant::getAttachedNodes()
{
    return attachedNodesList;
}
//------------------------------------------------------------------------------

void Variant::createVRLabel()
{
    //Create VRLabel
    osg::MatrixTransform *mtn = new osg::MatrixTransform;
    mtn->setName("Variant:Label");
    mtn->setMatrix((mtn->getMatrix()).scale(0.1, 0.1, 0.1));
    VarLabel = new coVRLabel(varName.c_str(), 5, 10.0, osg::Vec4(1, 1, 1, 1), osg::Vec4(0.1, 0.1, 0.1, 1));
    VarLabel->reAttachTo(mtn);
    VarLabel->setPosition(VarNode->getBound()._center);
    VarNode->addChild(mtn);
}
//------------------------------------------------------------------------------

void Variant::showVRLabel()
{
    VarLabel->show();
}

void Variant::hideVRLabel()
{
    VarLabel->hide();
}

void Variant::printMatrix(osg::MatrixTransform *mt)
{
    osg::Matrix ma = mt->getMatrix();
    cout << "/----------------------- " << endl;
    cout << ma(0, 0) << " " << ma(0, 1) << " " << ma(0, 2) << " " << ma(0, 3) << endl;
    cout << ma(1, 0) << " " << ma(1, 1) << " " << ma(1, 2) << " " << ma(1, 3) << endl;
    cout << ma(2, 0) << " " << ma(2, 1) << " " << ma(2, 2) << " " << ma(2, 3) << endl;
    cout << ma(3, 0) << " " << ma(3, 1) << " " << ma(3, 2) << " " << ma(3, 3) << endl;
    cout << "/-----------------------  " << endl;
}

#ifdef VRUI
void Variant::menuEvent(coMenuItem *item)
{
    coCheckboxMenuItem *m = dynamic_cast<coCheckboxMenuItem *>(item);
    if (m)
    {
        bool selected = m->getState();
        std::string vName = this->getVarname();
        TokenBuffer tb;
        tb << vName;
        if (selected)
            cover->sendMessage(this, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantShow, tb.get_length(), tb.get_data());
        else
            cover->sendMessage(this, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantHide, tb.get_length(), tb.get_data());
    }
}
#endif

void Variant::tabletEvent(coTUIElement *item)
{
    if (item == ui->getTUI_Item())
    {
        coTUIToggleButton *el = dynamic_cast<coTUIToggleButton *>(item);
        bool selected = el->getState();
        std::string vName = this->getVarname();
        TokenBuffer tb;
        tb << vName;
        if (selected)
            cover->sendMessage(this, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantShow, tb.getData().length(), tb.getData().data());
        else
            cover->sendMessage(this, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantHide, tb.getData().length(), tb.getData().data());
    }
    else if (item == ui->getRadioTUI_Item())
    {
        coTUIToggleButton *el = dynamic_cast<coTUIToggleButton *>(item);
        VariantPlugin::plugin->HideAllVariants();
        bool selected = el->getState();
        if (!selected)
        {
            el->setState(true);
        }
        std::string vName = this->getVarname();
        TokenBuffer tb;
        tb << vName;
        cover->sendMessage(this, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantShow, tb.getData().length(), tb.getData().data());
    }
    //EditFloatField Button:
    else
    {
        setOriginTransMatrix();
    }
}

void Variant::setOriginTransMatrix()
{
    osg::Vec3d vec = ui->getTransVec();
    VarNode->setMatrix(origin_matrix.translate(vec));
    setQDomElemTRANS(VarNode->getMatrix().getTrans());
}
//------------------------------------------------------------------------------------------------------------------------------

void Variant::setOriginTransMatrix(osg::Vec3d vec)
{
    VarNode->setMatrix(origin_matrix.translate(vec));
    setQDomElemTRANS(VarNode->getMatrix().getTrans());
}
//------------------------------------------------------------------------------------------------------------------------------

void Variant::attachClippingPlane()
{
    myboi->attachClippingPlanes(VarNode, cn);
}
//------------------------------------------------------------------------------------------------------------------------------

void Variant::releaseClippingPlane()
{
    myboi->releaseClippingPlanes(VarNode, cn);
}
//------------------------------------------------------------------------------------------------------------------------------

void Variant::setQDomElemTRANS(osg::Vec3d vec)
{

    QDomNodeList qdl = xmlfile->elementsByTagName("transform");
    for (int i = 0; i < qdl.size(); i++)
    {
        if (qdl.item(i).parentNode().toElement().tagName() == varName.c_str())
        {
            qdl.item(i).toElement().setAttribute("X", vec.x());
            qdl.item(i).toElement().setAttribute("Y", vec.y());
            qdl.item(i).toElement().setAttribute("Z", vec.z());
        }
    }
}

int Variant::numParents()
{
    return parents.size();
}
