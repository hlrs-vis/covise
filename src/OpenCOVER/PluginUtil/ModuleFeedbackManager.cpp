/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ModuleFeedbackManager.h"

#include <net/message.h>
#include <net/message_types.h>
#include <config/CoviseConfig.h>
#include <grmsg/coGRObjRegisterMsg.h>
#include <grmsg/coGRObjVisMsg.h>

#include <cover/coInteractor.h>
#include <cover/VRPinboard.h>
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>

#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>

#include <util/coExport.h>

#include "ColorBar.h"

#include <osgDB/WriteFile>

#include <sstream>
#include <osg/Switch>
#include <osg/Sequence>

using namespace std;
using namespace grmsg;
using namespace covise;
using namespace vrui;
using namespace opencover;

void
writeSgToFile()
{
    fprintf(stderr, "ModuleFeedbackManager writeSgToFile\n");
    std::string filename = covise::coCoviseConfig::getEntry("value", "COVER.SaveFile", "/var/tmp/OpenCOVER.osg");
    if (osgDB::writeNodeFile(*(VRSceneGraph::instance()->getScene()), filename))
    {
        cerr << "Data written to '" << filename << "'." << std::endl;
    }
}
// ----------------------------------------------------------------------
// construction / destruction
// ----------------------------------------------------------------------

ModuleFeedbackManager::ModuleFeedbackManager(RenderObject *containerObject, coInteractor *inter, const char *pluginName)
    : inter_(inter)
{
    if (cover->debugLevel(3))
    {
        if (containerObject)
            fprintf(stderr, "ModuleFeedbackManager::ModuleFeedbackManager1(containerObject =%s geomObject=%s)\n", containerObject->getName(), inter->getObject()->getName());
        else
            fprintf(stderr, "ModuleFeedbackManager::ModuleFeedbackManager1(containerObject =NULL geomObject=%s)\n", inter->getObject()->getName());
    }
    inter_->incRefCount();

    pName_ = string(pluginName, strlen(pluginName));
    //pName_ = new char[strlen(pluginName)+1];
    //strcpy(pName_, pluginName);

    containerObjectName_ = "";
    attrObjectName_ = "";

    if (containerObject)
        moduleName_ = ModuleName(containerObject->getName());
    else
        moduleName_ = ModuleName(inter->getObject()->getName());

    if (containerObject)
        containerObjectName_ = containerObject->getName();
    else
        containerObjectName_ = "";

    if (containerObject && containerObject->getAttribute("OBJECTNAME"))
        attrObjectName_ = containerObject->getAttribute("OBJECTNAME");
    else if (inter->getObject()->getAttribute("OBJECTNAME"))
        attrObjectName_ = inter->getObject()->getAttribute("OBJECTNAME");
    else
        attrObjectName_ = "";

    if (inter->getObject()->getAttribute("PART"))
        attrPartName_ = inter->getObject()->getAttribute("PART");
    else
        attrPartName_ = "";

    geomObjectName_ = inter->getObject()->getName();

    // register object at UI
    if (containerObject)
    {
        containerObjectName_ = containerObject->getName();
        registerObjAtUi(containerObjectName_); // for 2D parts Collect needs to be registered not RWCovise
    }
    else
    {
        registerObjAtUi(geomObjectName_);
    }

    // create menu
    createMenu();

    myNode_ = findMyNode();
    if (myNode_ && myNode_->getNumParents() > 0)
        myNodesParent_ = myNode_->getParent(0);
}

ModuleFeedbackManager::ModuleFeedbackManager(RenderObject *containerObject, RenderObject *geomObject, const char *pluginName)
{
    if (cover->debugLevel(3))
    {
        if (containerObject)
            fprintf(stderr, "ModuleFeedbackManager::ModuleFeedbackManager2(containerObject=%s, geomObject=%s)\n", containerObject->getName(), geomObject->getName());
        else
            fprintf(stderr, "ModuleFeedbackManager::ModuleFeedbackManager2(containerObject=NULL, geomObject=%s)\n", geomObject->getName());
    }

    pName_ = string(pluginName, strlen(pluginName));
    //pName_ = new char[strlen(pluginName)+1];
    //strcpy(pName_, pluginName);

    containerObjectName_ = "";
    attrObjectName_ = "";

    if (containerObject)
        moduleName_ = ModuleName(containerObject->getName());
    else
        moduleName_ = ModuleName(geomObject->getName());

    if (geomObject->getAttribute("OBJECTNAME"))
        attrObjectName_ = geomObject->getAttribute("OBJECTNAME");
    else if (containerObject && containerObject->getAttribute("OBJECTNAME"))
        attrObjectName_ = containerObject->getAttribute("OBJECTNAME");
    else
        attrObjectName_ = "";

    if (geomObject->getAttribute("PART"))
        attrPartName_ = geomObject->getAttribute("PART");
    else
        attrPartName_ = "";

    geomObjectName_ = geomObject->getName();

    if (containerObject)
    {
        containerObjectName_ = containerObject->getName();
        registerObjAtUi(containerObjectName_); // for 2D parts Collect needs to be registered not RWCovise
    }
    else
    {
        registerObjAtUi(geomObjectName_);
    }

    if (geomObject->getAttribute("OBJECTNAME"))
    {
        attrObjectName_ = geomObject->getAttribute("OBJECTNAME");
    }
    else if (containerObject && containerObject->getAttribute("OBJECTNAME"))
    {
        attrObjectName_ = containerObject->getAttribute("OBJECTNAME");
    }
    else
    {
        attrObjectName_ = "";
    }
    //fprintf(stderr,"attrObjectName_=%s\n", attrObjectName_.c_str());
    if (geomObject->getAttribute("PART"))
        attrPartName_ = geomObject->getAttribute("PART");
    else
        attrPartName_ = "";

    inter_ = NULL;

    // create the menu
    createMenu();
}

ModuleFeedbackManager::~ModuleFeedbackManager()
{
    parentMenu_->remove(menuItem_);

    if (colorBar_)
        delete colorBar_;

    delete hideCheckbox_;

    delete syncCheckbox_;

    if (newButton_)
        delete newButton_;

    if (deleteButton_)
        delete deleteButton_;

    if (executeCheckbox_)
        delete executeCheckbox_;

    delete colorsButton_;

    delete menu_;

    delete menuItem_;

    if (parentMenu_ != coviseMenu_)
    {
        // erst checken ob es leer ist !!!TODO!!!

        //delete caseMenu_;

        //fprintf(stderr,"deleting case menu item\n");

        //delete caseMenuItem_;
    }
}

// -----------------------------------------------------------------
// protected function to create the menu
// called in constructor
// -----------------------------------------------------------------
void ModuleFeedbackManager::createMenu()
{
    //fprintf(stderr,"ModuleFeedbackManager::createMenu\n");
    // get the menu from coPinboard
    VRMenu *covise = VRPinboard::instance()->namedMenu("COVISE");
    if (!covise)
    {
        VRPinboard::instance()->addMenu("COVISE", VRPinboard::instance()->mainMenu->getCoMenu());
        covise = VRPinboard::instance()->namedMenu("COVISE");
        cover->addSubmenuButton("COVISE...", NULL, "COVISE", false, NULL, -1, this);
    }
    coviseMenu_ = (coMenu *)covise->getCoMenu();
    caseName_ = "Covise";
    parentMenu_ = coviseMenu_;

    geometryCaseDCS_ = NULL;

    // create a submenu item in the "Covise" menu
    // and a corresponding menu

    if (attrObjectName_ != "")
        visMenuName_ = attrObjectName_;
    else if (attrPartName_ != "")
        visMenuName_ = attrPartName_;
    else
        visMenuName_ = moduleName_;

    visItemName_ = visMenuName_;
    visItemName_ += "...";

    menuItem_ = new coSubMenuItem(visItemName_.c_str());
    //menu_ = new coRowMenu(visMenuName_.c_str(),coviseMenu_, cover->getMaxMenuItems());
    menu_ = new coRowMenu(visMenuName_.c_str(), coviseMenu_);
    menuItem_->setMenu(menu_);

    coviseMenu_->add(menuItem_);

    // hide geometry
    hideCheckbox_ = new coCheckboxMenuItem("Hide", false);
    hideCheckbox_->setMenuListener(this);

    // sync interaction
    syncCheckbox_ = new coCheckboxMenuItem("Sync", true);
    syncCheckbox_->setMenuListener(this);

    // new module, for complex modules only adn disable for gui
    bool cfdgui = covise::coCoviseConfig::isOn("COVERConfig.CfdGui", false);
    newButton_ = NULL;
    deleteButton_ = NULL;
    int len = strlen(moduleName_.c_str());
    if (len >= 4 && !cfdgui && string("Comp") == moduleName_.c_str() + len - 4)
    {
        newButton_ = new coButtonMenuItem("New");
        newButton_->setMenuListener(this);

        deleteButton_ = new coButtonMenuItem("Delete");
        deleteButton_->setMenuListener(this);
    }

    // if ExecuteOnChange is not on, provide execute button
    executeCheckbox_ = NULL;
    if (!covise::coCoviseConfig::isOn("COVERConfig.EXECUTE_ON_CHANGE", true))
    {
        executeCheckbox_ = new coCheckboxMenuItem("Execute", false);
        executeCheckbox_->setMenuListener(this);
    }

    inExecute_ = false;

    // colors button
    colorsButton_ = new coSubMenuItem("Colors...");
    colorsButton_->setMenuListener(this);

    // colors submenu
    colorBar_ = NULL;

    // append the items to the menu
    menu_->add(hideCheckbox_);
    menu_->add(syncCheckbox_);
    if (newButton_)
        menu_->add(newButton_);
    if (deleteButton_)
        menu_->add(deleteButton_);
    if (executeCheckbox_)
        menu_->add(executeCheckbox_);
    menu_->add(colorsButton_);
}

// -----------------------------------------------------------------
// protected function to register the object at ui
// called in constructor
// -----------------------------------------------------------------
void ModuleFeedbackManager::registerObjAtUi(string name)
{
    initialObjectName_ = name;

    if (coVRMSController::instance()->isMaster())
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "\nModuleFeedbackManager::registerObjAtUi %s\n", initialObjectName_.c_str());
        coGRObjRegisterMsg regMsg(initialObjectName_.c_str(), NULL);
        Message grmsg;
        grmsg.type = COVISE_MESSAGE_UI;
        grmsg.data = (char *)(regMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

// -----------------------------------------------------------------
// compare functions
// -----------------------------------------------------------------
// returns true if inter comes from the same module
bool
ModuleFeedbackManager::compare(coInteractor *inter)
{
    if (inter_)
        return inter_->isSameModule(inter);
    else
        return compare(inter->getObject()->getName());
}

// returns true if inter comes from the same module
// name is an object name...
bool
ModuleFeedbackManager::compare(const char *objectName)
{
    //    fprintf(stderr,"ModuleFeedbackManager::compare objectName=%s\n", objectName);
    //    if (containerObjectName_ != "")
    //       fprintf(stderr,"...compare with containerObjectName_ %s\n", containerObjectName_.c_str());
    //       //cerr << "containerObjectName_=" << containerObjectName_ << endl;
    //    else
    //       fprintf(stderr,"... containerObjectName_=NULL\n");
    //    fprintf(stderr,"...compare with geomObjectName_ %s\n", geomObjectName_.c_str());
    //    if (attrObjectName_ != "")
    //       fprintf(stderr,"...compare with attrObjectName_ %s\n", attrObjectName_.c_str());

    // check container module name or geomObject module name
    if ((containerObjectName_ != "" && (ModuleName(objectName) == ModuleName(containerObjectName_.c_str())))
        || (ModuleName(objectName) == ModuleName(geomObjectName_.c_str()))
        || (ModuleName(objectName) == ModuleName(attrObjectName_.c_str())) //naja knnte man auch gleich strcmp nehmen
        )
    {
        //fprintf(stderr,"compare sucessful\n\n");
        return true;
    }
    else
        return false;
}

bool
ModuleFeedbackManager::comparePlugin(const char *name)
{
    return (strcmp(pName_.c_str(), name) == 0);
}

void
ModuleFeedbackManager::preFrame()
{
    //fprintf(stderr,"ModuleFeedbackManager::preFrame for object=%s plugin=%s\n", initialObjectName_.c_str(), pName_.c_str());
}

// -----------------------------------------------------------------
// update functions
// -----------------------------------------------------------------
void
ModuleFeedbackManager::update(RenderObject *containerObject, coInteractor *inter)
{

    if (cover->debugLevel(3))
    {
        if (containerObject)
            fprintf(stderr, "ModuleFeedbackManager::update containerObject=%s geomObject=%s\n", containerObject->getName(), inter->getObject()->getName());
        else
            fprintf(stderr, "ModuleFeedbackManager::update containerObject=NULL geomObject=%s\n", inter->getObject()->getName());
    }

    if (inter_ != inter)
    {
        inter_->decRefCount();
        inter_ = inter;
        inter_->incRefCount();
    }

    if (containerObject)
    {
        moduleName_ = ModuleName(containerObject->getName());
        containerObjectName_ = containerObject->getName();
    }
    else
    {
        containerObjectName_ = "";
        moduleName_ = ModuleName(inter->getObject()->getName());
    }

    geomObjectName_ = inter->getObject()->getName();

    if (inter->getObject()->getAttribute("OBJECTNAME"))
        attrObjectName_ = inter->getObject()->getAttribute("OBJECTNAME");
    else if (containerObject && containerObject->getAttribute("OBJECTNAME"))
        attrObjectName_ = containerObject->getAttribute("OBJECTNAME");
    else
        attrObjectName_ = "";

    if (inter->getObject()->getAttribute("PART"))
        attrPartName_ = inter->getObject()->getAttribute("PART");
    else
        attrPartName_ = "";

    //updateMenuNames();
    updateColorBar(containerObject);
}

void
ModuleFeedbackManager::update(RenderObject *containerObject, RenderObject *geomObject)
{

    if (containerObject)
    {

        if (cover->debugLevel(3))
            fprintf(stderr, "ModuleFeedbackManager::update for containerObject=%s geomObject=%s\n", containerObject->getName(), geomObject->getName());

        moduleName_ = ModuleName(containerObject->getName());
        containerObjectName_ = containerObject->getName();
    }
    else
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "ModuleFeedbackManager::update for containerObject=NULL geomObject=%s\n", geomObject->getName());
        moduleName_ = ModuleName(geomObject->getName());
        containerObjectName_ = "";
    }

    geomObjectName_ = geomObject->getName();
    if (geomObject->getAttribute("OBJECTNAME"))
        attrObjectName_ = geomObject->getAttribute("OBJECTNAME");
    else if (containerObject && containerObject->getAttribute("OBJECTNAME"))
        attrObjectName_ = containerObject->getAttribute("OBJECTNAME");
    else
        attrObjectName_ = "";

    if (geomObject->getAttribute("PART"))
        attrPartName_ = geomObject->getAttribute("PART");
    else
        attrPartName_ = "";

    //updateMenuNames();
    updateColorBar(containerObject);
}

// empty implementation of 3D tex functionality
void
ModuleFeedbackManager::update3DTex(string, cp3dplane *, const char *)
{
}

void
ModuleFeedbackManager::updateMenuNames()
{
    // if the user changed the module title
    // we have to update the submenu kitem and the menu title

    if (attrObjectName_ != "")
        visMenuName_ = attrObjectName_;
    else if (attrPartName_ != "")
        visMenuName_ = attrPartName_;
    else
        visMenuName_ = moduleName_;

    visItemName_ = visMenuName_;
    visItemName_ += "...";

    // update the label of the submenu item
    menuItem_->setName(visItemName_.c_str());

    // update the submenu title
    menu_->updateTitle(visMenuName_.c_str());

    if (inExecute_)
    {
        inExecute_ = false;
        if (executeCheckbox_)
            executeCheckbox_->setState(false);
    }
}

void
ModuleFeedbackManager::addColorbarInteractor(coInteractor *i)
{
    if (colorBar_)
        colorBar_->addInter(i);
    else
        std::cerr << "ModuleFeedbackManager::addColorbarInteractor: don't have a colorbar" << std::endl;
}

void
ModuleFeedbackManager::updateColorBar(RenderObject *containerObject)
{

    RenderObject *colorObj = NULL;
    const char *colormapString = NULL;

    if (containerObject)
    {
        colorObj = containerObject->getColors();
        if (!colorObj)
            colorObj = containerObject->getTexture();
        if (colorObj)
        {
            colormapString = colorObj->getAttribute("COLORMAP");
            if (colormapString == NULL && colorObj->isSet())
            {
                size_t noElems = colorObj->getNumElements();
                for (size_t elem = 0; elem < noElems; ++elem)
                {
                    colormapString = colorObj->getElement(elem)->getAttribute("COLORMAP");
                    if (colormapString)
                        break;
                }
            }
            // compute colorbar
            std::string tmpStr = ModuleName(geomObjectName_.c_str());
            const char *modname = tmpStr.c_str();
            float min = 0.0;
            float max = 1.0;
            int numColors;
            float *r = NULL;
            float *g = NULL;
            float *b = NULL;
            float *a = NULL;
            char *species = NULL;
            if (colormapString)
            {
                //cerr << "ModuleFeedbackManager::updateColorBar(..) COLORMAPSTRING " << colormapString << endl;
                ColorBar::parseAttrib(colormapString, species, min, max, numColors, r, g, b, a);
            }
            else
            {
                species = new char[16];
                strcpy(species, "NoColors");
                numColors = 2;
                min = 0.0;
                min = 1.0;
                r = new float[2];
                g = new float[2];
                b = new float[2];
                a = new float[2];
                r[0] = 0.0;
                g[0] = 0.0;
                b[0] = 0.0;
                a[0] = 1.0;
                r[1] = 1.0;
                g[1] = 1.0;
                b[1] = 1.0;
                a[1] = 1.0;
            }

            // color bar
            if (colorBar_)
            {
                colorBar_->update(species, min, max, numColors, r, g, b, a);
            }
            else
            {
                //cerr << "ModuleFeedbackManager::updateColorBar(..) MODNAME <" << modname << ">" << endl;
                colorBar_ = new ColorBar(colorsButton_, menu_,
                                         modname, species, min, max, numColors, r, g, b, a);
            }
            delete[] species;
            delete[] r;
            delete[] g;
            delete[] b;
            delete[] a;
        }
        else // container without colors
        {
            if (colorBar_) // the previous object had colors
                delete colorBar_;
            colorBar_ = NULL;
        }
    }
    else // no container
    {
        if (colorBar_) //the previous object had colors
            delete colorBar_;
        colorBar_ = NULL;
    }
}

// --------------------------------------------------------------------------
// EventListener
// --------------------------------------------------------------------------
void
ModuleFeedbackManager::menuEvent(coMenuItem *item)
{
    //fprintf(stderr,"ModuleFeedbackManager::menuEvent\n");
    if (item == hideCheckbox_)
    {
        hideGeometry(hideCheckbox_->getState());
        sendHideMsg(hideCheckbox_->getState());
    }
    else if (item == newButton_) // copy this module and execute it
    {
        // copy this module and execute it
        inter_->copyModuleExec();
    }
    else if (item == deleteButton_) // delete this module
    {
        // delete this module
        inter_->deleteModule();
    }
    else if (item == executeCheckbox_)
    {
        inExecute_ = true;
        inter_->executeModule();
    }
}

// hides geometry
// needed for menuevent
void
ModuleFeedbackManager::hideGeometry(bool hide)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "ModuleFeedbackManager::hideGeometry hide=%d\n", hide);

    osg::Node *node = findMyNode();
    if (node)
        myNode_ = node;
    else
        node = myNode_.get();
    if (node)
    {
        osg::MatrixTransform *dcs = dynamic_cast<osg::MatrixTransform *>(node);
        if (!dcs)
        {
            fprintf(stderr, "ERROR node is not the dcs\n");
            //writeSgToFile();
            return;
        }
        if (hide)
        {
            if (node->getNumParents() > 0)
            {
                myNodesParent_ = node->getParent(0);
                myNodesParent_->removeChild(node);
            }
            //osgDB::writeNodeFile(*dcs,"invisible.osg");
        }
        else
        {
            if (myNodesParent_.get())
                myNodesParent_->addChild(myNode_);
            //osgDB::writeNodeFile(*dcs,"visible.osg");
        }
    }
    else
    {
        fprintf(stderr, "ModuleFeedbackManager::hideGeometry could not find node\n");
    }
}

osg::Node *
ModuleFeedbackManager::findMyNode()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "ModuleFeedbackManager::findMyNode\n");
    osg::Geode *geode;
    osg::Group *group;
    if (attrObjectName_ != "")
    {
        //fprintf(stderr,"looking for attrObjectName_=%s\n", attrObjectName_.c_str());
        geode = VRSceneGraph::instance()->findFirstNode<osg::Geode>(attrObjectName_.c_str());
        if (geode != NULL)
        {
            //fprintf(stderr,"found geode with attrObjectName_=%s\n", attrObjectName_.c_str());
            return (geode);
        }
        group = VRSceneGraph::instance()->findFirstNode<osg::Group>(attrObjectName_.c_str());
        if (group != NULL)
        {
            //fprintf(stderr,"found group with attrObjectName_=%s\n", attrObjectName_.c_str());
            return (group);
        }
    }
    if (attrPartName_ != "")
    {
        //fprintf(stderr,"looking for attrPartName_=%s\n", attrPartName_.c_str());
        geode = VRSceneGraph::instance()->findFirstNode<osg::Geode>(attrPartName_.c_str());
        if (geode != NULL)
        {
            //fprintf(stderr,"found geode with attrPartName_=%s\n", attrPartName_.c_str());
            return (geode);
        }
        group = VRSceneGraph::instance()->findFirstNode<osg::Group>(attrPartName_.c_str());
        if (group != NULL)
        {
            //fprintf(stderr,"found group with attrPartName_=%s\n", attrPartName_.c_str());
            return (group);
        }
    }

    if (containerObjectName_ != "")
    {
        //fprintf(stderr,"looking for containerObjectName_=%s\n", containerObjectName_.c_str());
        geode = VRSceneGraph::instance()->findFirstNode<osg::Geode>(containerObjectName_.c_str());
        if (geode != NULL)
        {
            //fprintf(stderr,"found geode with containerObjectName_=%s\n", containerObjectName_.c_str());
            return (geode);
        }

        group = VRSceneGraph::instance()->findFirstNode<osg::Group>(containerObjectName_.c_str());
        if (group)
        {
            //fprintf(stderr,"found group with containerObjectName_=%s\n", containerObjectName_.c_str());
            return (group);
        }
    }

    //fprintf(stderr,"looking for moduleName_=%s\n", moduleName_.c_str());
    group = VRSceneGraph::instance()->findFirstNode<osg::Group>(moduleName_.c_str());
    if (group)
    {
        //fprintf(stderr,"found group with moduleName_=%s\n", moduleName_.c_str());
        return (group);
    }

    //fprintf(stderr,"looking for geomObjectName_=%s\n", geomObjectName_.c_str());
    geode = VRSceneGraph::instance()->findFirstNode<osg::Geode>(geomObjectName_.c_str());
    if (geode != NULL)
    {
        //fprintf(stderr,"found geode with geomObjectName_=%s\n", geomObjectName_.c_str());
        return (geode);
    }
    else
    {
        //fprintf(stderr,"ModuleFeedbackManager::findMyNode could not find node\n");
        //writeSgToFile();
    }
    return (NULL);
}

std::vector<osg::Geode *>
ModuleFeedbackManager::findMyGeode()
{
    //fprintf(stderr,"ModuleFeedbackManager::findMyGeode\n");
    std::vector<osg::Geode *> geodelist;
    osg::Node *node;

    node = findMyNode();
    //pfPrint(node, PFTRAV_SELF | PFTRAV_DESCEND, PFPRINT_VB_NOTICE, stderr);
    if (!node)
    {
        fprintf(stderr, "ModuleFeedbackManager::findMyGeode ERROR didn't find my node\n");
        return (geodelist);
    }
    return findRecMyGeode(node);
}

std::vector<osg::Geode *>
ModuleFeedbackManager::findRecMyGeode(osg::Node *node)
{
    osg::Geode *geode;
    osg::Switch *sequence;
    osg::MatrixTransform *dcs;
    osg::Group *group;
    std::vector<osg::Geode *> tmp_geodelist, geodelist;
    // check if node is a sequence
    sequence = dynamic_cast<osg::Switch *>(node);
    if (sequence)
    {
        //fprintf(stderr,"node is a sequence\n");
        for (unsigned int i = 0; i < sequence->getNumChildren(); i++)
        {
            // seq-dcs-geode
            dcs = dynamic_cast<osg::MatrixTransform *>(sequence->getChild(i));
            if (dcs && dcs->getNumChildren() > 0)
            {
                tmp_geodelist = findRecMyGeode(dcs->getChild(0));
            }
            if (!tmp_geodelist.empty())
                std::copy(tmp_geodelist.begin(), tmp_geodelist.end(), std::back_inserter(geodelist));
        }
        return (geodelist);
    }

    // check if node is a group
    group = dynamic_cast<osg::Group *>(node);
    if (group)
    {
        //fprintf(stderr,"node is a group %d\n", group->getNumChildren());
        for (unsigned int i = 0; i < group->getNumChildren(); i++)
        {
            // seq-dcs-geode
            dcs = dynamic_cast<osg::MatrixTransform *>(group->getChild(i));
            //fprintf(stderr, "type of child %s\n", group->getChild(i)->className());
            if (dynamic_cast<osg::Group *>(group->getChild(i)))
                tmp_geodelist = findRecMyGeode(group->getChild(i));
            if ((geode = dynamic_cast<osg::Geode *>(group->getChild(i))))
                geodelist.push_back(geode);
            if (dcs && dcs->getNumChildren() > 0)
            {
                //fprintf(stderr, "findRecMyGeode\n");
                tmp_geodelist = findRecMyGeode(dcs->getChild(0));
            }
            if (!tmp_geodelist.empty())
            {
                //fprintf(stderr, "copy\n");
                std::copy(tmp_geodelist.begin(), tmp_geodelist.end(), std::back_inserter(geodelist));
            }
        }
        return (geodelist);
    }

    // check if node is a geode
    geode = dynamic_cast<osg::Geode *>(node);
    if (geode)
    {
        //fprintf(stderr,"node is a geode\n");
        geodelist.push_back(geode);
    }

    return (geodelist);
}

void
ModuleFeedbackManager::setHideFromGui(bool hide)
{
    //fprintf(stderr,"ModuleFeedbackManager::setHideFromGui %d\n", hide);
    hideCheckbox_->setState(hide, true);
    hideGeometry(hide);
}
void
ModuleFeedbackManager::setCaseFromGui(const char *casename)
{
    //fprintf(stderr,"ModuleFeedbackManager::setCaseFromGui case=%s menuItem=%s \n", casename, geomObjectName_.c_str());

    caseName_ = casename;

    // first time we have to create the menu item and the menu
    if (parentMenu_ == coviseMenu_)
    {
        coviseMenu_->remove(menuItem_);

        // the we have to create a new case submenu item in the main pinboard and a case submenu
        string caseMenuName = casename;
        string caseMenuItemName = casename;
        caseMenuItemName += "...";

        // do we already have this case in the main menu?
        bool found = false;
        coMenuItemVector allItems = cover->getMenu()->getAllItems();
        int i = 0;
        while (i < allItems.size())
        {
            //fprintf(stderr,"checking item %s\n", allItems[i]->getName());
            // remove the item from the Covise Menu
            if (strcmp(allItems[i]->getName(), caseMenuItemName.c_str()) == 0)
            {
                //fprintf(stderr,"found case %s\n", allItems[i]->getName());
                caseMenuItem_ = (coSubMenuItem *)allItems[i];
                caseMenu_ = (coRowMenu *)caseMenuItem_->getMenu();
                parentMenu_ = caseMenu_;
                found = true;
                break;
            }
            i++;
        }
        if (!found)
        {
            caseMenuItem_ = new coSubMenuItem(caseMenuItemName.c_str()); //"Modellkabine..."
            //caseMenu_= new coRowMenu(caseMenuName.c_str(), cover->getMenu(), cover->getMaxMenuItems());        // "Modellkabine"
            caseMenu_ = new coRowMenu(caseMenuName.c_str(), cover->getMenu()); // "Modellkabine"
            caseMenuItem_->setMenu(caseMenu_);
            parentMenu_ = caseMenu_;
            cover->getMenu()->add(caseMenuItem_);
            //fprintf(stderr,"ModuleFeedbackManager::setCase added item %s to main menu\n", caseMenuItem_->getName());
        }

        caseMenu_->add(menuItem_); // add "Tracer_1..." to "Modellkabine"
        menu_->setParent(caseMenu_);

        geometryCaseDCS_ = VRSceneGraph::instance()->findFirstNode<osg::MatrixTransform>(casename);
        if (geometryCaseDCS_ == NULL)
        {
            //fprintf(stderr,"creating geometryCaseDCS_\n");
            // firsttime we create also a case DCS
            //fprintf(stderr,"ModuleFeedbackManager::setCaseFromGui create case DCS\n");
            geometryCaseDCS_ = new osg::MatrixTransform();
            geometryCaseDCS_->ref();
            geometryCaseDCS_->setName(casename);
            cover->getObjectsRoot()->addChild(geometryCaseDCS_.get());
        }

        // remove dcs from objectsRoot and add it to case
        // as VRSceneGraph::addNode renames the geode <objectname>_geom and the dcs <objectName>
        // we directly find the dcs
        osg::Node *myNode = findMyNode();
        if (!myNode)
        {
            fprintf(stderr, "ModuleFeedbackManager::setCaseFromGui ERROR: Didn't find my Node\n");
            return;
        }
        //else
        //{
        //	  if (!myNode->getName().empty())
        //		  fprintf(stderr,"found my node %s\n", myNode->getName().c_str());
        //	  else
        //		  fprintf(stderr,"found my node but doesn't have a name\n");
        //}

        osg::MatrixTransform *dcs = dynamic_cast<osg::MatrixTransform *>(myNode);
        if (!dcs)
        {
            fprintf(stderr, "node is not a dcs adding it to case anyway\n");
            return;
        }
        addNodeToCase(myNode);
    }
    else
    {
        // nothing to do at the moment, because we assume it can change only once
    }

    //fprintf(stderr,"ModuleFeedbackManager::setCaseFromGui done\n");
}

void
ModuleFeedbackManager::setNameFromGui(const char *newName)
{
    //fprintf(stderr,"ModuleFeedbackManager::setNameFromGui coviseObjectName=%s newName=%s \n",  geomObjectName_.c_str(), newName);

    visMenuName_ = newName;
    visItemName_ = visMenuName_;
    visItemName_ += "...";

    // update the label of the submenu item
    menuItem_->setName(visItemName_.c_str());

    // update the submenu title
    menu_->updateTitle(visMenuName_.c_str());

    std::vector<osg::Geode *> geodes = findMyGeode();
    if (geodes.size() > 0)
    {
        for (int i = 0; i < geodes.size(); i++)
        {
            osg::Geode *geode = geodes[i];
            if (geode->getNumDescriptions())
            {
                // if there is already a description which begins with SCGR_ replace it
                std::vector<std::string> dl = geode->getDescriptions();
                for (int i = 0; i < dl.size(); i++)
                {
                    std::string descr = dl[i];
                    if (descr.find("_SCGR_") != string::npos)
                    {
                        dl[i] = visMenuName_ + "_SCGR_";
                    }
                }
                geode->setDescriptions(dl);
            }
            else // add a description
                geode->addDescription(visMenuName_ + "_SCGR_");
        }
    }
}

void
ModuleFeedbackManager::sendHideMsg(bool hide)
{
    //fprintf(stderr,"-------------------ModuleFeedbackManager::sendHideMsg for %s\n", initialObjectName_.c_str());
    if (coVRMSController::instance()->isMaster())
    {
        coGRObjVisMsg visMsg(coGRMsg::GEO_VISIBLE, initialObjectName_.c_str(), !hide);
        Message grmsg;
        grmsg.type = COVISE_MESSAGE_UI;
        grmsg.data = (char *)(visMsg.c_str());
        grmsg.length = strlen(grmsg.data) + 1;
        cover->sendVrbMessage(&grmsg);
    }
}

/* set Transformation of a geode*/
void
ModuleFeedbackManager::setMatrix(float *row0, float *row1, float *row2, float *row3)
{
    //fprintf(stderr,"ModuleFeedbackManager::setMatrix\n");
    // get the parent of goede -> should be dcs
    osg::Node *node = findMyNode();
    //fprintf(stderr,"GeneralGeometryInteraction::setMatrix of object %s\n", geode->getName());
    if (node->getNumParents() < 1)
    {
        //fprintf(stderr, "ERROR geode has no parent -> cannot set dcs\n");
        return;
    }
    osg::MatrixTransform *dcs = dynamic_cast<osg::MatrixTransform *>(node->getParent(0));
    if (!dcs)
    {
        //fprintf(stderr, "ERROR parent of geode no dcs -> cannot set transformation");
        return;
    }

    //set the matrix of the dcs
    osg::Matrix m;
    m.set(row0[0], row0[1], row0[2], row0[3],
          row1[0], row1[1], row1[2], row1[3],
          row2[0], row2[1], row2[2], row2[3],
          row3[0], row3[1], row3[2], 1.0);

    dcs->setMatrix(m);
}

// auxiliary function is for suggesting a menu name:
// it looks recursively for the OBJECTNAME attribute
const char *
ObjectName(RenderObject *obj)
{
    const char *ret = obj->getAttribute("OBJECTNAME");
    if (ret)
    {
        return ret;
    }
    if (obj->isGeometry())
    {
        RenderObject *geom = ((RenderObject *)obj)->getGeometry();
        return ObjectName(geom);
    }
    else if (obj->isSet())
    {
        size_t num_elem = obj->getNumElements();
        for (size_t elem = 0; elem < num_elem; ++elem)
        {
            ret = ObjectName(obj->getElement(elem));
            if (ret)
            {
                return ret;
            }
        }
    }
    return NULL;
}

// get the menu name calculated upon construction
string
ModuleFeedbackManager::getMenuName() const
{
    return menuName_;
}

// get the module name+instance (for example Tracer_1)
string
ModuleFeedbackManager::ModuleName(const char *objectName) const
{
    //cerr << "ModuleFeedbackManager::ModuleName(..) objectName <"<< objectName << ">" << endl;
    //fprintf(stderr,"ModuleFeedbackManager::ModuleName(objectName=%s)\n", objectName);
    string moduleName;
    char *buf;

    buf = new char[strlen(objectName) + 1];
    strcpy(buf, objectName); // RWCovise_1_OUT_01

    char *tmp = strstr(buf, "_OUT_");
    if (tmp)
    {
        tmp[0] = '\0'; // RWCovise_1
        moduleName += buf;
    }
    else
        moduleName += objectName;

    //fprintf(stderr,"\tmodulename=%s\n", moduleName.c_str());
    return moduleName;
}

void
ModuleFeedbackManager::addNodeToCase(osg::Node *node)
{
    // der Knoten ist das DCS
    if (cover->debugLevel(3))
    {
        if (!node->getName().empty())
            fprintf(stderr, "ModuleFeedbackManager::addNodeToCase %s\n", node->getName().c_str());
        else
            fprintf(stderr, "ModuleFeedbackManager::addNodeToCase\n");
    }

    if (geometryCaseDCS_.get())
    {
        // is this node already under case DCS?
        osg::Node *child = node;
        if (node->getNumParents() > 0)
        {
            osg::Node *parent = node->getParent(0);
            while (parent && (parent != geometryCaseDCS_.get()))
            {
                if (parent == cover->getObjectsRoot())
                {
                    geometryCaseDCS_->addChild(child);
                    cover->getObjectsRoot()->removeChild(child);
                    break;
                }
                child = parent;
                if (child->getNumParents() > 0)
                    parent = child->getParent(0);
                else
                {
                    //// das ist ein fehler und sollte noch behoben werden, dr, fprintf(stderr, "WARNING: cannot put node %s into case\n",node->getName().c_str());
                    return;
                }
            }
        }
    }
    //else
    //{
    //      // vor der ersten setCase msg
    //      fprintf(stderr,"ModuleFeedbackManager::addNodeToCase add node to case ERROR: geometryCaseDCS_=NULL\n");
    //}
}

bool ModuleFeedbackManager::getSyncState() { return syncCheckbox_->getState(); };
