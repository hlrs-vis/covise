/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*********************************************************************************\
 **                                                            2009 HLRS         **
 **                                                                              **
 ** Description:  Show/Hide of VariantPlugins, defined in Collect Module               **
 **                                                                              **
 **                                                                              **
 ** Author: A.Gottlieb                                                           **
 **                                                                              **
 ** Jul-09  v1                                                                   **
 **                                                                              **
 **                                                                              **
\*********************************************************************************/

#include "VariantPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRFileManager.h>
#include <osg/CullStack>
#include <iostream>
#include <cover/coTabletUI.h>
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/PluginMessageTypes.h>
#include <net/tokenbuffer.h>
#include <osg/Node>
#include <algorithm>
#include <map>
#include <vector>
#include <iterator>
#include <numeric>
#include "VrmlNodeVariant.h"
#include <vrml97/vrml/VrmlNamespace.h>
#include <util/string_util.h>

using namespace covise;
using namespace opencover;

VariantPlugin *VariantPlugin::plugin = NULL;

VariantMarker::VariantMarker(std::string EntryName)
{
    float scale = coCoviseConfig::getFloat("scale", EntryName, -1.0);

    std::string markerNames = coCoviseConfig::getEntry("markerNames", EntryName);
    auto markerVec = split(markerNames, ';', true);
    for (const auto &m: markerVec)
    {
        markerSet.insert(MarkerTracking::instance()->getMarker(m));
    }

    std::string variants = coCoviseConfig::getEntry("variants", EntryName);
    auto varVec = split(variants, ';', true);
    for (const auto &v: varVec)
    {
        variantSet.insert(v);
    }
}

VariantMarker::~VariantMarker() = default;

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

VariantPlugin::VariantPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("VariantPlugin", cover->ui)
{
    assert(plugin == NULL);
    plugin = this;

    interActing = false;
    _interactionA = new vrui::coTrackerButtonInteraction(vrui::coInteraction::ButtonA, "MoveMode", vrui::coInteraction::Medium);
    tmpVec.set(1, 1, 1);
    boi = NULL;


    coCoviseConfig::ScopeEntries variantEntries = coCoviseConfig::getScopeEntries("COVER.Plugin.Variant.Marker");
    for (const auto& varMarkerName : variantEntries)
    {
        std::string EntryName = std::string("COVER.Plugin.Variant.Marker.") + varMarkerName.first;
        variantMarkers.emplace_back(EntryName);
    }

    
    VrmlNamespace::addBuiltIn(VrmlNodeVariant::defineType());
}
//------------------------------------------------------------------------------------------------------------------------------

bool VariantPlugin::init()
{
    cover->addPlugin("SGBrowser"); // required for hiding/showing nodes
    //variant_menu = new ui::Menu(this, "Variants");

    variant_menu = new ui::Menu("Variants", this);

    options_menu = new ui::Menu(variant_menu, "Options");
    showHideLabels = new ui::Button(options_menu, "ShowLabels");
    showHideLabels->setText("Show labels");
    showHideLabels->setState(false);
    showHideLabels->setCallback([this](bool state){
        if (state)
        {
            showAllLabel();
        }
        else
        {
            hideAllLabel();
        }
        setQDomElemLabels(state);
        tui_showLabel->setState(state);
    });

    roi_menu = new ui::Menu(variant_menu, "RegionOfInterest");
    roi_menu->setText("Region of interest");
    define_roi = new ui::Button(roi_menu, "Define");
    define_roi->setState(false);
    define_roi->setCallback([this](bool state){
        if (state && firsttime)
        {
            float initSize = getTypicalSice() * 0.01;
            boi->setMatrix(boi->getMat() * boi->getMat().scale(initSize, initSize, initSize));
            boi->setStartMatrix();
            printMatrix(boi->getMat());
            boi->updateClippingPlanes();
            firsttime = false;
        }
        boi->showHide(state);
    });

    active_roi = new ui::Button(roi_menu, "Active");
    active_roi->setState(false);
    active_roi->setCallback([this](bool state){
        // float initSize = cover->getBBox(cover->getObjectsRoot()).radius() * 0.1;
        std::list<Variant *>::iterator it;
        if (state)
        {
            for (it = varlist.begin(); it != varlist.end(); it++)
            {
                //boi->attachClippingPlanes((*it)->getNode());
                (*it)->attachClippingPlane();
            }
        }
        else
        {
            for (it = varlist.begin(); it != varlist.end(); it++)
            {
                //boi->releaseClippingPlanes((*it)->getNode());
                (*it)->releaseClippingPlane();
            }
        }
        //boi->setMatrix(osg::Vec3(10,10,10),osg::Vec3(1,1,1));
    });

    coVRSelectionManager::instance()->addListener(this);
    //tuTab
    VariantPluginTab = new coTUITab("Variants", coVRTui::instance()->mainFolder->getID());
    VariantPluginTab->setPos(0, 0);
    coTUILabel *lbl_VariantPlugins = new coTUILabel("Variants", VariantPluginTab->getID());
    lbl_VariantPlugins->setPos(0, 0);
    coTUIToggleButton *lbl_X = new coTUIToggleButton("X-trans", VariantPluginTab->getID());
    lbl_X->setPos(1, 0);
    lbl_X->setEventListener(this);
    coTUIToggleButton *lbl_Y = new coTUIToggleButton("Y-trans", VariantPluginTab->getID());
    lbl_Y->setPos(2, 0);
    lbl_Y->setEventListener(this);
    coTUIToggleButton *lbl_Z = new coTUIToggleButton("Z-trans", VariantPluginTab->getID());
    lbl_Z->setPos(3, 0);
    lbl_Z->setEventListener(this);
    coTUILabel *space = new coTUILabel("                     ", VariantPluginTab->getID());
    space->setPos(4, 0);
    saveXML = new coTUIFileBrowserButton("Save", VariantPluginTab->getID());
    saveXML->setPos(5, 0);
    saveXML->setEventListener(this);
    saveXML->setFilterList("*.xml");
    readXML = new coTUIFileBrowserButton("Read", VariantPluginTab->getID());
    readXML->setPos(5, 1);
    readXML->setEventListener(this);
    readXML->setFilterList("*.xml");

    tui_showLabel = new coTUIToggleButton("Show Labels", VariantPluginTab->getID());
    tui_showLabel->setPos(6, 0);
    tui_showLabel->setState(false);
    tui_showLabel->setEventListener(this);

    tui_header_trans[lbl_X->getName()] = lbl_X;
    tui_header_trans[lbl_Y->getName()] = lbl_Y;
    tui_header_trans[lbl_Z->getName()] = lbl_Z;

    //XML Section

    xmlfile = new QDomDocument;
    qDE_Variant = xmlfile->createElement("Variant");
    qDE_Variant.setAttribute("name", "Settings for Variant-Plugin");
    xmlfile->appendChild(qDE_Variant);
    QDomElement xmloptions = xmlfile->createElement("options");
    qDE_Variant.appendChild(xmloptions);
    QDomElement xmlshowLabels = xmlfile->createElement("showLabels");
    xmloptions.appendChild(xmlshowLabels);
    setQDomElemLabels(true);
    //Box of Interest
    boi = new coVRBoxOfInterest(plugin, _interactionA);
    boi->showHide(false);
    firsttime = true;

    return true;
}
//------------------------------------------------------------------------------------------------------------------------------
// this is called if the plugin is removed at runtime

VariantPlugin::~VariantPlugin()
{
    //fprintf ( stderr,"VariantPlugin::~VariantPlugin\n" );

#ifdef VRUI
    delete showHideLabels;
    delete options_menu;
    delete define_roi;
    delete roi_menue;
    delete roi;
    delete variant_menu;
    delete button;
    delete variants;
    delete variants_menu;
    //
    delete options;
#endif

    delete VariantPluginTab;
    delete boi;
    plugin = nullptr;
}
//------------------------------------------------------------------------------------------------------------------------------

void
VariantPlugin::preFrame()
{
    const VariantMarker *toActivate = nullptr;
    for (const auto& variantMarker : variantMarkers)
    {
        for (const auto &m: variantMarker.markerSet)
        {
            if (m->isVisible())
            {
                if (!toActivate)
                    toActivate = &variantMarker;
                if (activatedMarker == &variantMarker)
                    toActivate = nullptr;
            }
        }
    }
    if (toActivate)
    {
        std::cerr << "Variant: Activating new variants via Marker: " << toActivate->variants << std::endl;
        setVariant(toActivate->variants);
        if(toActivate->scale!=-1)
            cover->setScale(toActivate->scale);
        activatedMarker = toActivate;
    }

    sensorList.update();

    static osg::Matrix invStartHand;
    static osg::Matrix startPos;

    //tmpVec = osg::Vec3(1,1,1);

    int state = cover->getPointerButton()->getState(); //Button States are defined in /covise/src/renderer/OpenCOVER/device/VRTracker.h

    if (_interactionA->isRunning())
    {
        //if centersphere is selected, do translation
        if (boi->isSensorActiv("center"))
        {
            if (!interActing)
            {
                invStartHand.invert(cover->getPointerMat() * cover->getInvBaseMat());
                startPos = boi->getMat();
                interActing = true;
            }
            else
            {
                osg::Matrix transMat = startPos * invStartHand * (cover->getPointerMat() * cover->getInvBaseMat());
                boi->setMatrix(transMat);
                boi->updateClippingPlanes();
            }
        }
        //if aa is selected do scale
        if (boi->isSensorActiv("aa"))
        {
            if (!interActing)
            {
                invStartHand.invert(cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                startPos = boi->getBoxGeoMt()->getMatrix();
                boi->setStartMatrix();
                interActing = true;
                tmpVec.set(1, 1, 1.);
            }
            else
            {
                osg::Matrix transMat = (invStartHand * (cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat()));
                tmpVec.set(1, (1 - (transMat.getTrans().y() / boi->getLength().y())), 1);
                boi->setScale(startPos, tmpVec, XTRANS);
                boi->updateClippingPlanes();
            }
        }
        //if bb is selected do scale
        if (boi->isSensorActiv("bb"))
        {
            if (!interActing)
            {
                invStartHand.invert(cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                startPos = boi->getBoxGeoMt()->getMatrix();
                boi->setStartMatrix();
                interActing = true;
                tmpVec.set(1, 1, 1.);
            }
            else
            {
                osg::Matrix transMat = invStartHand * (cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                tmpVec.set((1 - (transMat.getTrans().x() / boi->getLength().x())), 1, 1);
                boi->setScale(startPos, tmpVec, YTRANS);
                boi->updateClippingPlanes();
            }
        }
        //if cc is selected do scale
        if (boi->isSensorActiv("cc"))
        {
            if (!interActing)
            {
                invStartHand.invert(cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                startPos = boi->getBoxGeoMt()->getMatrix();
                boi->setStartMatrix();
                interActing = true;
                tmpVec.set(1, 1, 1.);
            }
            else
            {
                osg::Matrix transMat = invStartHand * (cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                tmpVec.set(1, (1 - (-transMat.getTrans().y() / boi->getLength().y())), 1);
                boi->setScale(startPos, tmpVec, XTRANS);
                boi->updateClippingPlanes();
            }
        }
        //if dd is selected do scale
        if (boi->isSensorActiv("dd"))
        {
            if (!interActing)
            {
                invStartHand.invert(cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                startPos = boi->getBoxGeoMt()->getMatrix();
                boi->setStartMatrix();
                interActing = true;
                tmpVec.set(1, 1, 1.);
            }
            else
            {
                osg::Matrix transMat = invStartHand * (cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                tmpVec.set((1 - (-transMat.getTrans().x() / boi->getLength().x())), 1, 1);
                boi->setScale(startPos, tmpVec, YTRANS);
                boi->updateClippingPlanes();
            }
        }
        //if ee is selected do scale
        if (boi->isSensorActiv("ee"))
        {
            if (!interActing)
            {
                invStartHand.invert(cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                startPos = boi->getBoxGeoMt()->getMatrix();
                boi->setStartMatrix();
                interActing = true;
                tmpVec.set(1, 1, 1.);
            }
            else
            {
                osg::Matrix transMat = invStartHand * (cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                tmpVec.set(1, 1, (1 - (-transMat.getTrans().z() / boi->getLength().z())));
                boi->setScale(startPos, tmpVec, ZTRANS);
                boi->updateClippingPlanes();
            }
        }
        //if ff is selected do scale
        if (boi->isSensorActiv("ff"))
        {
            if (!interActing)
            {
                invStartHand.invert(cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                startPos = boi->getBoxGeoMt()->getMatrix();
                boi->setStartMatrix();
                interActing = true;
                tmpVec.set(1, 1, 1.);
            }
            else
            {
                osg::Matrix transMat = invStartHand * (cover->getPointerMat() * cover->getInvBaseMat() * boi->getinvMat());
                tmpVec.set(1, 1, (1 - (transMat.getTrans().z() / boi->getLength().z())));
                boi->setScale(startPos, tmpVec, ZTRANS);
                boi->updateClippingPlanes();
            }
        }
        // boi->
    }

    if (_interactionA->wasStopped() && state == false)
    {
        interActing = false;
        //cout<<"tmp"<<tmpVec.y()<<endl;
        boi->setLentgh(tmpVec);
        //cout<<"Length nach der Interaktion: "<<boi->getLength().y()<<endl;
        tmpVec = osg::Vec3(1, 1, 1);
    }
    //every Frame/////////////////////////////////////////////////////////////
    //update Sphere size
    //osg::Matrix initMat;
    //if (cover->getPointerButton()->wasPressed())
    //{
    //initMat.set(boi->getBoxCenterMt()->getMatrix());
    //}
    //float size = (1 / cover->getScale())*10 ;
    //cout<<"Scale "<<cover->getScale()<<"  Scale2 "<<size<<endl;
    //boi->getBoxCenterMt()->setMatrix(initMat * boi->getBoxCenterMt()->getMatrix().scale(osg::Vec3(size,size,size)));
}

//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

bool VariantPlugin::selectionChanged()
{
    //std::cerr << "VariantPlugin::selectionChanged info: called" << std::endl;

    return true;
}
//------------------------------------------------------------------------------------------------------------------------------

bool VariantPlugin::pickedObjChanged()
{
    return true;
}
//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::addNode(osg::Node *node, const RenderObject *render)
{
    if (render != NULL)
    {
        if (render->getAttribute("VARIANT") != NULL)
        {
            std::string var_att(render->getAttribute("VARIANT"));

            if (var_att != "NULL" && var_att != "")
            {
                bool set_default = false;
                bool default_state = false;
                if (render->getAttribute("VARIANT_VISIBLE"))
                {
                    set_default = true;
                    if (std::string(render->getAttribute("VARIANT_VISIBLE")) == "off")
                        default_state = false;
                    else
                        default_state = true;
                }
                if(var_att.length()>3 && var_att.compare(var_att.length()-3,3,"_on")==0)
                {
                   var_att = var_att.substr(0,var_att.length()-3);
                   default_state = true;
                   set_default = true;
                }
                if(var_att.length()>4 && var_att.compare(var_att.length()-4,4,"_off")==0)
                {
                   var_att = var_att.substr(0,var_att.length()-4);
                   default_state = false;
                   set_default = true;
                }
                if (set_default)
                {
                    //std::cerr << "Variant " << var_att << ", default=" << default_state << std::endl;
                }
                Variant *var = getVariant(var_att);
                if (!var) //create new menu item
                {
                    osg::Node::ParentList parents;
                    if (node)
                        parents = node->getParents();
                    var = new Variant(this, var_att, node, parents, variant_menu, VariantPluginTab, varlist.size() + 1,
                                      xmlfile, &qDE_Variant, boi, set_default ? default_state : true);
                    varlist.push_back(var);
                    var->AddToScenegraph();
                    var->hideVRLabel();
                    if(set_default && default_state==false)
                    {
                       osg::Node *n = var->getNode();
                       if (n)
                       {
                           std::string path = coVRSelectionManager::generatePath(n);
                           std::string pPath = path.substr(0, path.find_last_of(";"));
                           TokenBuffer tb2;
                           tb2 << path;
                           tb2 << pPath;
                           cover->sendMessage(plugin, "SGBrowser", PluginMessageTypes::SGBrowserHideNode, tb2.getData().length(), tb2.getData().data());
                           setMenuItem(var, (false));
                           setQDomElemState(var, false);
                       }
                    }
                }
                else
                {
                    var->attachNode(node);
                }
                varmap[node] = var;
            }
        }
    }
}

//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::removeNode(osg::Node *node, bool /*isGroup*/, osg::Node * /*realNode*/)
{
    Variant *var = getVariantbyAttachedNode(node);
    if (var)
    {
        cout << "Varname " << var->getVarname().c_str() << endl;
        cout << "Number of Parents " << var->numParents() << endl;
        var->releaseNode(node);
        varmap.erase(node);
        if (var->numNodes() == 0)
        {
            auto it = std::find(varlist.begin(), varlist.end(), var);
            if (it != varlist.end())
            {
                varlist.erase(it);
            }
            var->removeFromScenegraph(node);
            delete var;
        }
    }
}
//------------------------------------------------------------------------------------------------------------------------------

Variant *VariantPlugin::getVariant(std::string varName)
{
    std::list<Variant *>::iterator varlIter;

    for (varlIter = varlist.begin(); varlIter != varlist.end(); varlIter++)
    {
        if ((*varlIter)->getVarname() == varName)
        {
            return *varlIter;
        }
    }
    return NULL;
}
//------------------------------------------------------------------------------------------------------------------------------

Variant *VariantPlugin::getVariant(osg::Node *varNode)
{
    std::list<Variant *>::iterator varlIter;

    for (varlIter = varlist.begin(); varlIter != varlist.end(); varlIter++)
    {
        if ((*varlIter)->getNode() == varNode)
        {
            return *varlIter;
        }
    }
    return NULL;
}

void VariantPlugin::setVariant(std::string var)
{
        VariantPlugin::plugin->HideAllVariants();
        std::stringstream ss(var);
        std::string out;
        while(std::getline(ss,out,';')) {
            TokenBuffer tb;
            tb << out;
            cover->sendMessage(this, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantShow, tb.getData().length(), tb.getData().data());
        }
}
//------------------------------------------------------------------------------------------------------------------------------

Variant *VariantPlugin::getVariantbyAttachedNode(osg::Node *node)
{
    auto it = varmap.find(node);
    if (it == varmap.end())
        return nullptr;

    return it->second;
}

//------------------------------------------------------------------------------------------------------------------------------

#ifdef VRUI
void VariantPlugin::menuEvent(coMenuItem *item)
{

    coCheckboxMenuItem *m = dynamic_cast<coCheckboxMenuItem *>(item);
    if (m)
    {

        if (m == showHideLabels)
        {
            if (m->getState())
            {
                showAllLabel();
            }
            else
            {
                hideAllLabel();
            }
            setQDomElemLabels(m->getState());
            tui_showLabel->setState(m->getState());
        }
        if (m == define_roi)
        {
            if (m->getState())
            {
                if (firsttime)
                {
                    float initSize = getTypicalSice() * 0.01;
                    boi->setMatrix(boi->getMat() * boi->getMat().scale(initSize, initSize, initSize));
                    boi->setStartMatrix();
                    printMatrix(boi->getMat());
                    boi->updateClippingPlanes();
                    firsttime = false;
                }
                boi->showHide(true);
            }
            else
            {
                boi->showHide(false);
            }
        }
        if (m == active_roi)
        {
            // float initSize = cover->getBBox(cover->getObjectsRoot()).radius() * 0.1;

            std::list<Variant *>::iterator it;

            if (m->getState())
            {
                for (it = varlist.begin(); it != varlist.end(); it++)
                {
                    //boi->attachClippingPlanes((*it)->getNode());
                    (*it)->attachClippingPlane();
                }
            }
            else
            {
                for (it = varlist.begin(); it != varlist.end(); it++)
                {
                    //boi->releaseClippingPlanes((*it)->getNode());
                    (*it)->releaseClippingPlane();
                }
            }
            //boi->setMatrix(osg::Vec3(10,10,10),osg::Vec3(1,1,1));
        }
    }
}
#endif
//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::tabletEvent(coTUIElement *elem)
{
    coTUIToggleButton *t = dynamic_cast<coTUIToggleButton *>(elem); //button for shifting VariantPlugins into several directions
    if (t)
    {
        if (tui_header_trans.find(t->getName()) != tui_header_trans.end())
        {
            TRANS dir;
            if (std::string(t->getName()) == std::string("X-trans"))
            {
                dir = XTRANS;
            }
            else if (std::string(t->getName()) == std::string("Y-trans"))
            {
                dir = YTRANS;
            }
            else if (std::string(t->getName()) == std::string("Z-trans"))
            {
                dir = ZTRANS;
            }
            bool state = t->getState();
            if (state)
            {
                makeTranslations(dir);
            }
            else
            {
                clearTranslations(dir);
            }
        }
        if (t == tui_showLabel)
        {
            bool state = t->getState();
            if (state)
            {
                showAllLabel();
            }
            else
            {
                hideAllLabel();
            }
            setQDomElemLabels(state);
            showHideLabels->setState(state);
        }
    }
    coTUIFileBrowserButton *u = dynamic_cast<coTUIFileBrowserButton *>(elem);
    if (u == saveXML)
    {
        saveXmlFile();
    }
    if (u == readXML)
    {
        readXmlFile();
    }
}
//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::message(int toWhom, int type, int len, const void *buf)
{
    if (type != PluginMessageTypes::VariantHide && type != PluginMessageTypes::VariantShow)
        return;

    TokenBuffer tb((char *)buf, len);
    std::string VariantName;
    tb >> VariantName;
    Variant *var = getVariant(VariantName);
    if (var)
    {
        osg::Node *n = var->getNode();
        if (n)
        {
            std::string path = coVRSelectionManager::generatePath(n);
            std::string pPath = path.substr(0, path.find_last_of(";"));
            TokenBuffer tb2;
            tb2 << path;
            tb2 << pPath;
            if (type == PluginMessageTypes::VariantHide)
            {
                cover->sendMessage(plugin, "SGBrowser", PluginMessageTypes::SGBrowserHideNode, tb2.getData().length(), tb2.getData().data());
                setMenuItem(var, (false));
                setQDomElemState(var, false);
            }
            else
            {
                if (VrmlNodeVariant::instance())
                    VrmlNodeVariant::instance()->setVariant(VariantName);
                cover->sendMessage(plugin, "SGBrowser", PluginMessageTypes::SGBrowserShowNode, tb2.getData().length(), tb2.getData().data());
                setMenuItem(var, (true));
                setQDomElemState(var, true);
            }
        }
        else
        {
            cerr << "Node of Variant " << VariantName << " not found" << endl;
        }
    }
    else
    {
        cerr << "Variant " << VariantName << " not found" << endl;
    }
}
//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::setMenuItem(Variant *var, bool state)
{
    var->ui->getVRUI_Item()->setState(state);
    var->ui->getTUI_Item()->setState(state);
}
//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::updateTUItemPos()
{
    std::list<Variant *>::iterator it;
    int i = 1;
    for (it = varlist.begin(); it != varlist.end(); it++)
    {
        (*it)->ui->getTUI_Item()->setPos(0, i);
        (*it)->ui->getXTransItem()->setPos(1, i);
        (*it)->ui->getYTransItem()->setPos(2, i);
        (*it)->ui->getZTransItem()->setPos(3, i);
    }
}
//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::clearTranslations(TRANS dir)
{
    osg::Vec3d transVec(0, 0, 0);
    switch (dir)
    {
    case XTRANS:
        transVec.set(1, 0, 0);
        break;
    case YTRANS:
        transVec.set(0, 1, 0);
        break;
    case ZTRANS:
        transVec.set(0, 0, 1);
        break;
    }
    std::list<Variant *>::iterator it;
    for (it = varlist.begin(); it != varlist.end(); it++)
    {
        osg::MatrixTransform *mtn = dynamic_cast<osg::MatrixTransform *>((*it)->getNode());
        if (mtn)
        {
            osg::Matrix mat = (mtn->getMatrix()).translate(0, 0, 0);
            osg::Matrix mat2 = (mtn->getMatrix() * osg::Matrix::translate(osg::Vec3d(0, 0, 0)));
            osg::Vec3d v1 = mat.getTrans();
            osg::Vec3d v2 = mat2.getTrans();
            osg::Vec3d vr = v1 - v2;
            vr.set(vr.x() * transVec.x(), vr.y() * transVec.y(), vr.z() * transVec.z());
            mtn->setMatrix(mat2 * osg::Matrix::translate(vr));
            (*it)->ui->setTransVec(mtn->getMatrix().getTrans());
        }
    }
}
//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::makeTranslations(TRANS dir)
{
    osg::Vec3d transVec(0, 0, 0);
    osg::Vec3d oVec;
    std::list<Variant *>::iterator it;
    for (it = varlist.begin(); it != varlist.end(); it++)
    {
        osg::MatrixTransform *mtn = dynamic_cast<osg::MatrixTransform *>((*it)->getNode());
        if (mtn)
        {
            mtn->setMatrix(mtn->getMatrix() * osg::Matrix::translate(transVec));
            oVec = (mtn->getMatrix()).getTrans();
        }
        (*it)->ui->setTransVec(oVec);
        (*it)->setQDomElemTRANS(oVec);
        transVec.operator+=(createTransVec((*it), dir));
    }
}
//------------------------------------------------------------------------------------------------------------------------------

osg::Vec3d VariantPlugin::createTransVec(Variant *var, TRANS dir)
{
    float Xmax, Ymax, Zmax;
    float Xmin, Ymin, Zmin;
    osg::Vec3d vec;
    Xmax = 0;
    Ymax = 0;
    Zmax = 0;
    Xmin = 0;
    Ymin = 0;
    Zmin = 0;
    osg::Node *node = var->getNode();
    switch (dir)
    {
    case XTRANS:
        Xmax = getBoundingBox(node).xMax();
        Xmin = getBoundingBox(node).xMin();
        break;
    case YTRANS:
        Ymax = getBoundingBox(node).yMax();
        Ymin = getBoundingBox(node).yMin();
        break;
    case ZTRANS:
        Zmax = getBoundingBox(node).zMax();
        Zmin = getBoundingBox(node).zMin();
        break;
    }

    vec.set(Xmax - Xmin, Ymax - Ymin, Zmax - Zmin);
    if (vec.length() < 1E-4)
    {
        switch (dir)
        {
        case XTRANS:
            vec.set(getTypicalSice(), 0, 0);
            break;
        case YTRANS:
            vec.set(0, getTypicalSice(), 0);
            break;
        case ZTRANS:
            vec.set(0, 0, getTypicalSice());
            break;
        }
    }
    return vec * 1.1;
}
//------------------------------------------------------------------------------------------------------------------------------

osg::BoundingBox VariantPlugin::getBoundingBox(osg::Node *node)
{
    box.init();
    VariantPluginNodeVisitor myVis(this, &box);
    node->traverse(myVis);
    return box;
}
//------------------------------------------------------------------------------------------------------------------------------

float VariantPlugin::getTypicalSice()
{
    std::vector<float> values;
    osg::ClipNode *ojr = cover->getObjectsRoot();
    cout << "SceneSize:" << cover->getSceneSize() << endl;
    for (unsigned int i = 0; i < ojr->getNumChildren(); i++)
    {
        osg::Node *child = ojr->getChild(i);
        cout << "-.-.NodeName:" << child->getName() << "---";
        if (child->getName() != "myBox")
        {
            values.push_back(child->getBound()._radius);
            cout << *(std::max_element(values.begin(), values.end())) << endl;
        }
    }

    float maxVal = *(std::max_element(values.begin(), values.end()));
    values.clear();
    return maxVal;
}

//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::hideAllLabel()
{
    std::list<Variant *>::iterator it;
    for (it = varlist.begin(); it != varlist.end(); it++)
        (*it)->hideVRLabel();
}
//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::showAllLabel()
{
    std::list<Variant *>::iterator it;
    for (it = varlist.begin(); it != varlist.end(); it++)
        (*it)->showVRLabel();
}
//------------------------------------------------------------------------------------------------------------------------------

int VariantPlugin::saveXmlFile()
{
    std::string filename = saveXML->getSelectedPath().c_str();
    size_t spos = filename.find("file://");
    if (spos != std::string::npos)
    {
        filename.erase(spos, 7);
    }
    QFile file(filename.c_str());
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return 0;
    QTextStream out(&file);
    out << xmlfile->toString();
    return 1;
}
//------------------------------------------------------------------------------------------------------------------------------

int VariantPlugin::readXmlFile()
{
    QString errmessage;
    int errorLine, errorColumn;
    std::string filename = readXML->getSelectedPath().c_str();
    size_t spos = filename.find("file://");
    if (spos != std::string::npos)
    {
        filename.erase(spos, 7);
    }
    QFile file(filename.c_str());
    if (!file.open(QIODevice::ReadOnly))
        return -1;
    if (!xmlfile->setContent(&file, false, &errmessage, &errorLine, &errorColumn))
    {
        file.close();
        return -2;
    }
    file.close();
    parseXML(xmlfile);
    return 1;
}
//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::setQDomElemState(Variant *var, bool state)
{

    QDomNodeList qdl = xmlfile->elementsByTagName("visible");
    for (int i = 0; i < qdl.size(); i++)
    {
        if (qdl.item(i).parentNode().toElement().tagName() == var->getVarname().c_str())
            qdl.item(i).toElement().setAttribute("state", state);
    }
}
//------------------------------------------------------------------------------------------------------------------------------

int VariantPlugin::parseXML(QDomDocument *qxmlDoc)
{
    QDomElement root = qxmlDoc->documentElement();
    if (root.tagName() != "Variant")
        return 0;
    QDomNode n = root.firstChild();
    while (!n.isNull())
    {
        QDomElement e = n.toElement();
        //---------
        if (e.tagName() == "options")
        {
            QDomNode n1 = n.firstChild();
            if (n1.toElement().hasAttribute("showLabels"))
            {
                QString state = n1.toElement().attribute("showLabels");
                if (state == "0")
                {
                    hideAllLabel();
                    showHideLabels->setState(false);
                    tui_showLabel->setState(false);
                }
                else
                {
                    showAllLabel();
                    showHideLabels->setState(true);
                    tui_showLabel->setState(true);
                }
            }
        }
        //---------
        else
        {
            QDomNode n1 = n.firstChild();
            std::string varName = n.toElement().tagName().toStdString();
            Variant *var = getVariant(varName);
            while ((var != NULL) && !n1.isNull())
            {
                std::string tagName = n1.toElement().tagName().toStdString();
                //--
                if (tagName == "visible")
                {
                    QString state = n1.toElement().attribute("state");
                    std::string path = coVRSelectionManager::generatePath(var->getNode());
                    TokenBuffer tb;
                    tb << path;
                    std::string pPath = path.substr(0, path.find_last_of(";"));
                    tb << pPath;

                    if (state != "0")
                        cover->sendMessage(plugin, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantShow, tb.getData().length(), tb.getData().data());
                    else
                        cover->sendMessage(plugin, coVRPluginSupport::TO_ALL, PluginMessageTypes::VariantHide, tb.getData().length(), tb.getData().data());
                }
                //--
                if (tagName == "transform")
                {
                    QString x = n1.toElement().attribute("X");
                    QString y = n1.toElement().attribute("Y");
                    QString z = n1.toElement().attribute("Z");
                    osg::Vec3d vec;
                    vec.set(x.toDouble(), y.toDouble(), z.toDouble());
                    var->setOriginTransMatrix(vec);
                    var->ui->setTransVec(vec);
                }
                n1 = n1.nextSibling();
            }
        }
        n = n.nextSibling();
        cout << "TagName:    " << e.tagName().toStdString() << endl;
    }
    return 1;
}

//------------------------------------------------------------------------------------------------------------------------------

void VariantPlugin::setQDomElemLabels(bool state)
{

    QDomNodeList qdl = xmlfile->elementsByTagName("showLabels");
    for (int i = 0; i < qdl.size(); i++)
    {
        qdl.item(i).toElement().setAttribute("showLabels", state);
    }
}
//------------------------------------------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------------------------------------------

VariantPluginNodeVisitor::VariantPluginNodeVisitor(coVRPlugin *plug, osg::BoundingBox *box)
    : osg::NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
    , plugin(plug)
{
    bbox = box;
}

void VariantPluginNodeVisitor::apply(osg::Node &node)
{
    //cout << "traverseName " << node.getName() << endl;
    osg::Geode *geo = dynamic_cast<osg::Geode *>(&node);
    if (geo)
    {
        bbox->expandBy(geo->getBoundingBox());
    }
    if ((&node)->getName() != "Label")
        traverse(node);
}

void VariantPlugin::printMatrix(osg::Matrix ma)
{
    cout << "/----------------------- " << endl;
    cout << ma(0, 0) << " " << ma(0, 1) << " " << ma(0, 2) << " " << ma(0, 3) << endl;
    cout << ma(1, 0) << " " << ma(1, 1) << " " << ma(1, 2) << " " << ma(1, 3) << endl;
    cout << ma(2, 0) << " " << ma(2, 1) << " " << ma(2, 2) << " " << ma(2, 3) << endl;
    cout << ma(3, 0) << " " << ma(3, 1) << " " << ma(3, 2) << " " << ma(3, 3) << endl;
    cout << "/-----------------------  " << endl;
}

void VariantPlugin::HideAllVariants()
{
    std::list<Variant *>::iterator varlIter;
    for (varlIter = varlist.begin(); varlIter != varlist.end(); varlIter++)
    {
        std::string vName = (*varlIter)->getVarname();
        TokenBuffer tb;
        tb << vName;
        cover->sendMessage(this, "Variant", PluginMessageTypes::VariantHide, tb.getData().length(), tb.getData().data());
    }
}

COVERPLUGIN(VariantPlugin)
