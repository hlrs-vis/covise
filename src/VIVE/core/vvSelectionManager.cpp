/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvSelectionManager.h"
#include "vvPluginSupport.h"
#include "vvIntersection.h"
#include <OpenVRUI/coNavInteraction.h>

#include <vsg/nodes/MatrixTransform.h>

#include <net/tokenbuffer.h>
#include <util/string_util.h>
#include <vrb/client/VRBMessage.h>
#include <net/message.h>
#include <net/message_types.h>

using namespace vive;
using namespace vrui;


vvSelectionManager *vvSelectionManager::s_instance = NULL;

vvSelectionManager *vvSelectionManager::instance()
{
    if (!s_instance)
        s_instance = new vvSelectionManager();
    return s_instance;
}

vvSelectionManager::vvSelectionManager()
{
    assert(!s_instance);

    selectionInteractionA = new coNavInteraction(coInteraction::ButtonA, "Selection", coInteraction::High);
    selectedNodeList.clear();
    selectedParentList.clear();
    selectionNodeList.clear();
    SelOnOff = 1;
    SelWire = 3;
    SelRed = 0.0f;
    SelGreen = 0.1f;
    SelBlue = 1.0f;

    updateManager = vv->getUpdateManager();
    updateManager->add(this);
}

vvSelectionManager::~vvSelectionManager()
{
    delete selectionInteractionA;
    updateManager->remove(this);

    s_instance = NULL;
}
/*
osg::BoundingSphere vvSelectionManager::getBoundingSphere(vsg::Node *objRoot)
{
    osg::BoundingSphere bsphere;

    osg::BoundingBox bb;
    bb.init();
    vsg::Node *currentNode = NULL;

    if (selectedNodeList.empty())
    {
        vsg::Group *root = dynamic_cast<vsg::Group *>(objRoot);
        if (root)
        {
            for (unsigned int i = 0; i < root->children.size(); i++)
            {
                currentNode = root->children[i];
                const vsg::MatrixTransform *transform = currentNode->asTransform();
                if ((!transform || transform->getReferenceFrame() == vsg::MatrixTransform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
                {
                    bb.expandBy(vv->getBBox(currentNode));
                }
            }

            if (bb.valid())
            {

                bsphere._center = bb.center();
                bsphere._radius = 0.0f;
                for (unsigned int i = 0; i < root->children.size(); i++)
                {
                    currentNode = root->children[i];
                    const vsg::MatrixTransform *transform = currentNode->asTransform();
                    if ((!transform || transform->getReferenceFrame() == vsg::MatrixTransform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
                    {
                        bsphere.expandRadiusBy( vv->getBBox(currentNode));
                    }
                }
            }
        }
    }
    else
    {
        vsg::dmat4 startBaseMat, trans;
        std::list<vsg::ref_ptr<vsg::Node> >::iterator iter = selectedNodeList.begin();
        while (iter != selectedNodeList.end())
        {
            currentNode = (*iter).get();
            const vsg::MatrixTransform *transform = currentNode->asTransform();
            if ((!transform || transform->getReferenceFrame() == vsg::MatrixTransform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
            {
                osg::BoundingSphere bs, bs_new;

                bs = currentNode->getBound();

                startBaseMat.makeIdentity();
                if (currentNode->getNumParents() > 0)
                    currentNode = currentNode->getParent(0);
                else
                    currentNode = NULL;
                while (currentNode != NULL && (currentNode->getName() != "OBJECTS_ROOT"))
                {
                    if (dynamic_cast<vsg::MatrixTransform *>(currentNode))
                    {
                        trans = ((vsg::MatrixTransform *)currentNode)->matrix;
                        startBaseMat.postMult(trans);
                    }
                    if (currentNode->getNumParents() > 0)
                        currentNode = currentNode->getParent(0);
                    else
                        currentNode = NULL;
                }
                bs_new = bs;
                bs_new._center = startBaseMat.preMult(bs._center);
                bb.expandBy(bs_new);
            }
            iter++;
        }
        if (bb.valid())
        {
            bsphere._center = bb.center();
            bsphere._radius = 0.0f;
            iter = selectedNodeList.begin();
            while (iter != selectedNodeList.end())
            {
                currentNode = (*iter).get();
                const vsg::MatrixTransform *transform = currentNode->asTransform();
                if ((!transform || transform->getReferenceFrame() == vsg::MatrixTransform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
                {
                    osg::BoundingSphere bs, bs_new;

                    bs = currentNode->getBound();

                    startBaseMat.makeIdentity();
                    if (currentNode->getNumParents() > 0)
                        currentNode = currentNode->getParent(0);
                    else
                        currentNode = NULL;
                    while (currentNode != NULL && (currentNode->getName() != "OBJECTS_ROOT"))
                    {
                        if (dynamic_cast<vsg::MatrixTransform *>(currentNode))
                        {
                            trans = ((vsg::MatrixTransform *)currentNode)->matrix;
                            startBaseMat.postMult(trans);
                        }
                        if (currentNode->getNumParents() > 0)
                            currentNode = currentNode->getParent(0);
                        else
                            currentNode = NULL;
                    }
                    bs_new = bs;
                    bs_new._center = startBaseMat.preMult(bs._center);
                    bsphere.expandRadiusBy(bs_new);
                }
                iter++;
            }
        }
    }
    return bsphere;
}*/
bool vvSelectionManager::update()
{

    if (selectionInteractionA->wasStarted())
    {
        if (!vv->getIntersectedNode())
        {
            clearSelection();
            pickedObjChanged();
            return true;
        }
        const vsg::Intersector::NodePath &intersectedNodePath = vv->getIntersectedNodePath();
        bool isSceneNode = false;
        for (auto iter = intersectedNodePath.begin();
             iter != intersectedNodePath.end();
             ++iter)
        {
            if ((*iter) == vv->getObjectsRoot())
            {
                isSceneNode = true;
                break;
            }
        }

        if (isSceneNode)
        {
            auto iter = intersectedNodePath.end();
            --iter;
            --iter;
            const vsg::Group *parent = dynamic_cast<const vsg::Group *>(*iter);
            while (parent && isHelperNode(parent))
            {
                iter--;
                parent = dynamic_cast<const vsg::Group*>(*iter);
            }
            if (parent)
            {
                clearSelection();
                addSelection(parent, vv->getIntersectedNode());
                pickedObjChanged();
            }
            else
            {
                cerr << "parent not found" << endl;
            }
        }
    }
    return true;
}

void vvSelectionManager::addListener(coSelectionListener *l)
{
    if (l && find(listenerList.begin(), listenerList.end(), l) == listenerList.end())
        listenerList.push_back(l);
}

void vvSelectionManager::removeListener(coSelectionListener *l)
{
    list<coSelectionListener *>::iterator it = find(listenerList.begin(), listenerList.end(), l);
    if (it != listenerList.end())
        listenerList.erase(it);
}

void vvSelectionManager::selectionChanged()
{
    for (list<coSelectionListener *>::iterator item = listenerList.begin(); item != listenerList.end();)
    {
        if (!(*item)->selectionChanged())
            item = listenerList.erase(item);
        else
            ++item;
    }
}

void vvSelectionManager::pickedObjChanged()
{
    for (list<coSelectionListener *>::iterator item = listenerList.begin(); item != listenerList.end();)
    {
        if (!(*item)->pickedObjChanged())
            item = listenerList.erase(item);
        else
            ++item;
    }
}

void vvSelectionManager::removeNode(vsg::Node *node)
{
    if (selectedNodeList.empty())
        return;

    std::list<vsg::ref_ptr<vsg::Node> >::iterator childIter = selectedNodeList.begin();
    std::list<vsg::ref_ptr<vsg::Group> >::iterator parentIter = selectedParentList.begin();
    std::list<vsg::ref_ptr<vsg::Group> >::iterator nodeIter = selectionNodeList.begin();

    while (childIter != selectedNodeList.end()) //&& !selectedNodeList.empty())
    {
        if ((node == (*childIter).get()) || (node == (*parentIter).get()) || haveToDelete((*parentIter).get(), node))
        {
            if ((*nodeIter).get())
            {
                vsg::Group *mygroup = NULL;
                vsg::Node *child = NULL;
                /*if ((*nodeIter).get()->getNumParents())
                {
                    vsg::Node *parent = (*nodeIter).get()->getParent(0);
                    mygroup = dynamic_cast<vsg::Group *>(parent);
                }
                if ((*nodeIter).get()->children.size())
                {
                    child = (*nodeIter).get()->getChild(0);
                }
                if (child && mygroup && (*nodeIter).get())
                {
                    mygroup->replaceChild((*nodeIter).get(), child);
                    (*nodeIter).get()->removeChild(child);
                }*/
            }
            childIter = selectedNodeList.erase(childIter);
            parentIter = selectedParentList.erase(parentIter);
            nodeIter = selectionNodeList.erase(nodeIter);
        }
        else
        {

            childIter++;
            parentIter++;
            nodeIter++;
        }
    }
}

bool vvSelectionManager::haveToDelete(vsg::Node *parent, vsg::Node *node)
{
   /* if ((parent->getNumParents() > 1) || (parent->getNumParents() == 0))
        return false;
    else
    {
        if (parent->getParent(0) == node)
            return true;
        else
            return haveToDelete(parent->getParent(0), node);
    }*/
    return false;
}

#if 0
void vvSelectionManager::selectionCallback(void *, buttonSpecCell *spec)
{
    //fprintf(stderr,"vvSelectionManager::selectionCallback\n");
    if (spec->state == 1.0)
    {
        vvSelectionManager::instance()->setSelectionOnOff(1);
    }
    else
    {
        vvSelectionManager::instance()->setSelectionOnOff(0);
    }
}
#endif

void vvSelectionManager::setSelectionColor(float R, float G, float B)
{
    SelRed = R;
    SelGreen = G;
    SelBlue = B;
}

void vvSelectionManager::setSelectionWire(int mode)
{
    SelWire = mode;
}

void vvSelectionManager::setSelectionOnOff(int mode)
{
    if (mode)
    {
        vvSelectionManager::instance()->setSelectionWire(3);
        vvSelectionManager::instance()->setSelectionColor(0, 0, 1);

        if (!selectionInteractionA->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(selectionInteractionA);
            vvIntersection::instance()->isectAllNodes(true);
        }
    }
    else
    {

        if (selectionInteractionA->isRegistered())
        {
            coInteractionManager::the()->unregisterInteraction(selectionInteractionA);
            vvIntersection::instance()->isectAllNodes(false);
        }
    }
}

void vvSelectionManager::showhideSelection(int mode)
{
    SelOnOff = mode;
}

void vvSelectionManager::receiveAdd(covise::TokenBuffer &messageData)
{
    std::string parentPath; 
    std::string nodePath; 

    messageData >> parentPath;
    messageData >> nodePath;

    vsg::Node *parent = validPath(parentPath);
    vsg::Node *node = validPath(nodePath);
    vsg::Group *parentG = dynamic_cast<vsg::Group *>(parent);

    addSelection(parentG, node, false);
    pickedObjChanged();
}

void vvSelectionManager::addSelection(const vsg::Group *parent, const vsg::Node *selectedNode, bool send)
{
    if ((!parent) || (!selectedNode))
        return;

    if (send)
    {
        covise::TokenBuffer tb;
        tb << vrb::ADD_SELECTION;
        tb <<generatePath(parent);
        tb << generatePath(selectedNode);
        covise::Message msg;
        msg.type = covise::COVISE_MESSAGE_VRB_MESSAGE;
        vv->sendVrbMessage(&msg);
    }

    vsg::Group *selectionNode = NULL;

    /*vsg::Node *existNode = getHelperNode(parent, selectedNode, SELECTION);
   if(existNode)
      selectionNode = existNode->asGroup();
   else*/

   /* if (SelOnOff)
    {
        // create a material
        osg::Material* selMaterial = new osg::Material();
        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, vsg::vec4f(SelRed, SelGreen, SelBlue, 1.0f));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, vsg::vec4f(SelRed, SelGreen, SelBlue, 1.0f));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, vsg::vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);

        if (SelWire == 0) // FILLED
        {
            // apply material
            selectionNode = new vsg::Group();
            selectionNode->setName("Selection:Filled");
            osg::StateSet *ss = selectionNode->getOrCreateStateSet();
            osg::PolygonMode *polymode = new osg::PolygonMode();
            polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::FILL);
            ss->setAttribute(selMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
            ss->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
            // disable shader
            ss->setAttributeAndModes(new osg::Program(), osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            ss->setMode(GL_LIGHTING, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            ss->setMode(GL_NORMALIZE, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            ss->setMode(GL_BLEND, osg::StateAttribute::OVERRIDE | osg::StateAttribute::ON);
            ss->setMode(osg::StateAttribute::PROGRAM, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
            ss->setMode(osg::StateAttribute::VERTEXPROGRAM, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
            ss->setMode(osg::StateAttribute::FRAGMENTPROGRAM, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
            ss->setMode(osg::StateAttribute::TEXTURE, osg::StateAttribute::OVERRIDE | osg::StateAttribute::OFF);
        }
        else if (SelWire == 1) // WIREFRAME_SELECTION_COLOR
        {
            // apply material
            selectionNode = new vsg::Group();
            selectionNode->setName("Selection:Wire-Sel");
            osg::StateSet *ss = selectionNode->getOrCreateStateSet();
            osg::PolygonMode *polymode = new osg::PolygonMode();
            polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
            ss->setAttribute(selMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
            ss->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
        }
        else if (SelWire == 2) // WIREFRAME_OBJECT_COLOR
        {
            // apply material
            selectionNode = new vsg::Group();
            selectionNode->setName("Selection:Wire-Obj");
            osg::StateSet *ss = selectionNode->getOrCreateStateSet();
            osg::PolygonMode *polymode = new osg::PolygonMode();
            polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
            ss->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
        }
        else if (SelWire == 3) // OUTLINE
        {
            // apply material
            osgFX::Outline *out = new osgFX::Outline;
            out->setColor(vsg::vec4f(SelRed, SelGreen, SelBlue, 1.0f));
            out->setWidth(4);
            selectionNode = static_cast<vsg::Group *>(out);
        }
    }
    else
    {
        selectionNode = new vsg::Group();
        selectionNode->setName("Selection:None");
    }*/

    /*insertHelperNode(parent, selectedNode, selectionNode, SELECTION);

    selectedNodeList.push_back(selectedNode);
    selectedParentList.push_back(parent);
    selectionNodeList.push_back(selectionNode);
    selectionChanged();*/
}

void vvSelectionManager::receiveClear()
{
    clearSelection(false);
}

void vvSelectionManager::clearSelection(bool send)
{
    if (send)
    {
        covise::TokenBuffer tb;
        tb << vrb::CLEAR_SELECTION;
        covise::Message msg;
        msg.type = covise::COVISE_MESSAGE_VRB_MESSAGE;
        vv->sendVrbMessage(&msg);
    }

    while (!selectionNodeList.empty())
    {
        vsg::ref_ptr<vsg::Group> mySelectionNode = selectionNodeList.front();
       /* if (mySelectionNode->getNumParents())
        {
            vsg::Node *parent = mySelectionNode->getParent(0);
            vsg::Node *child = NULL;
            if (mySelectionNode->children.size() > 0)
                child = mySelectionNode->children{0];
            vsg::Group *mygroup = dynamic_cast<vsg::Group*>(parent);
            if (child && mygroup && mySelectionNode.get())
            {
                mygroup->replaceChild(mySelectionNode.get(), child);
                vvPluginSupport::removeChild(mySelectionNode.get(), child);
            }
        }*/
        selectionNodeList.pop_front();
    }

    selectedNodeList.clear();
    selectedParentList.clear();
    selectionChanged();
}

vsg::Group *vvSelectionManager::getHelperNode(vsg::Group *parent, vsg::Node *child, HelperNodeType type)
{

    if ((!parent) || (!child))
        return NULL;

    int index = type;

    vsg::Group *helpParent = NULL;
    vsg::Group *helpParent2 = NULL;

    /*for (unsigned int i = 0; i < child->getNumParents(); i++)
    {
        helpParent = child->getParent(i);
        if (helpParent == parent)
            return NULL;
        else
        {
            helpParent2 = helpParent;
            while (isHelperNode(helpParent))
            {
                helpParent = helpParent->getParent(0);
            }
            if (helpParent == parent)
            {
                while (helpParent2 && isHelperNode(helpParent2) && (index != getHelperType(helpParent2)))
                {
                    helpParent2 = helpParent2->getParent(0);
                }
                if (helpParent2 && (index == getHelperType(helpParent2)))
                    return helpParent2;
                else
                    return NULL;
            }
        }
    }*/
    return NULL;
}

void vvSelectionManager::insertHelperNode(vsg::Group *parent, vsg::Node *child, vsg::Group *insertNode, HelperNodeType type, bool show)
{
    if ((!parent) || (!child))
        return;

    int index = type;
    if (!hasType(insertNode))
    {
        std::string strType = "TYPE ";
        std::string strNumber = "";
        std::stringstream ss;
        ss << index;
        ss >> strNumber;
        strType.append(strNumber);
        //insertNode->addDescription(strType);
    }

    bool equal = false;
    vsg::Node *myChild = NULL;
    vsg::Group *myParent = NULL;
    vsg::Group *helpParent = NULL;
    vsg::Group *helpParent2 = NULL;
    vsg::Node *helpChild = NULL;

    /*for (unsigned int i = 0; i < child->getNumParents(); i++)
    {
        helpParent = child->getParent(i);
        if (helpParent == parent)
        {
            myChild = child;
            myParent = parent;
        }
        else
        {
            helpParent2 = helpParent;
            while (isHelperNode(helpParent))
            {
                helpParent = helpParent->getParent(0);
            }
            if (helpParent == parent)
            {
                helpChild = child;
                while (helpParent2 && isHelperNode(helpParent2) && (getHelperType(helpParent2) >= index))
                {
                    helpChild = helpParent2;
                    helpParent2 = helpParent2->getParent(0);
                }
                if (helpChild && (index == getHelperType(helpChild)))
                    equal = true;
                else
                    equal = false;
                myChild = helpChild;
                myParent = helpParent2;
            }
        }
    }*/
    /*
    if (insertNode && myParent && myChild)
    {
        if (equal)
        {
            vsg::Group *equalNode = dynamic_cast<vsg::Group*>(myChild);
            insertNode->addChild(equalNode->children[0]);

            if (index == SHOWHIDE)
            {
                vsg::Switch *mySwitch = (vsg::Switch *)insertNode;
                mySwitch->setChildValue(equalNode->getChild(0), show);
                myParent->replaceChild(myChild, mySwitch);
            }
            else
                myParent->replaceChild(myChild, insertNode);
            //equalNode->removeChild(0,1);
        }
        else
        {
            if (index == SHOWHIDE)
            {
                osg::Switch *mySwitch = (osg::Switch *)insertNode;
                mySwitch->addChild(myChild, show);
                myParent->replaceChild(myChild, mySwitch);
            }
            else
            {
                insertNode->addChild(myChild);
                myParent->replaceChild(myChild, insertNode);
            }
        }
        if (!isHelperNode(insertNode))
            markAsHelperNode(insertNode);
    }*/
}

void vvSelectionManager::markAsHelperNode(vsg::Node *node)
{
    node->setValue("SELECTIONHELPER","");
}

bool vvSelectionManager::isHelperNode(const vsg::Node *node)
{
    std::string val;
    return node->getValue("SELECTIONHELPER", val);
}

bool vvSelectionManager::hasType(vsg::Node *node)
{
    /*for (unsigned int j = 0; node && j < node->getNumDescriptions(); j++)
    {
        if (strncmp(node->getDescription(j).c_str(), "TYPE", 4) == 0)
        {
            return true;
        }
    }*/
    return false;
}

int vvSelectionManager::getHelperType(vsg::Node *node)
{
   /*
    for (unsigned int j = 0; node && j < node->getNumDescriptions(); j++)
    {
        if (strncmp(node->getDescription(j).c_str(), "TYPE ", 5) == 0)
        {
            return atoi(node->getDescription(j).c_str() + 5);
        }
    }*/
    return 100;
}
std::string vvSelectionManager::generateNames(vsg::Node *node)
{
    std::string name = "";
    std::string newstr = "";
    std::string oldstr = "";
    vsg::Group *parent = NULL;

    bool arrive = false;

   /* while (!arrive)
    {
        name = node->getName();
        if (node == vv->getObjectsRoot())
        {
            newstr = "OBJECTS_ROOT";
            if (!oldstr.empty())
            {
                oldstr.append("\n");
                oldstr.append(newstr);
            }
            else
            {
                oldstr = newstr;
            }

            arrive = true;
        }
        else
        {
            if (node->getNumParents())
            {
                newstr = node->getName();
                parent = node->getParent(0)->asGroup();
                while (isHelperNode(parent))
                {
                    node = parent;
                    parent = parent->getParent(0)->asGroup();
                }

                if (!oldstr.empty())
                {
                    oldstr.append("\n");
                    oldstr.append(newstr);
                }
                else
                {
                    oldstr = newstr;
                }

                node = parent;
            }
            else
            {
                arrive = true;
            }

        } //end else

    } //end while*/

    return oldstr;
}
std::string vvSelectionManager::generatePath(const vsg::Node *node)
{
    std::string name = "";
    std::string newpath = "";
    std::string oldpath = "";
    int index = 0;
    vsg::Group *parent = NULL;
    vsg::Node *realNode = NULL;
    bool arrive = false;
    fprintf(stderr, "TODO:\n");

    while (!arrive)
    {
        node->getValue("name",name);
        if (node == vv->getObjectsRoot())
        {
            newpath = "ROOT";
            if (!oldpath.empty())
            {
                newpath.append(";");
                newpath.append(oldpath);
            }
            arrive = true;
        }
        else
        {
            /*if (node->getNumParents())
            {

                parent = node->getParent(0)->asGroup();
                realNode = node;
                while (isHelperNode(parent))
                {

                    node = parent;
                    parent = parent->getParent(0)->asGroup();
                }

                for (unsigned int i = 0; parent && i < parent->children.size(); i++)
                {
                    if (parent->children[i] == node)
                    {
                        index = i;
                    }
                }

                //CoOP
                if (node->getParent(0) == vv->getObjectsRoot())
                {
                    newpath = realNode->getName();
                }
                else
                {
                    std::stringstream ss;
                    ss << index;
                    ss >> newpath;
                }

                if (!oldpath.empty())
                {
                    newpath.append(";");
                    newpath.append(oldpath);
                }

                node = parent;
            }
            else
            {
                arrive = true;
            }*/

        } //end else

        oldpath = newpath;

    } //end while

    return newpath;
}

vsg::Node *vvSelectionManager::validPath(std::string path)
{
    vsg::Node *returnValue = NULL;

    char delims[] = ";";
    char *result = NULL;
    int i = 0;
    vsg::Group *help = NULL;
    vsg::Node *child = NULL;
    vsg::Group *parent = NULL;

    char *pathchar = new char[path.length() + 1];
    strcpy(pathchar, path.c_str());

    result = strtok(pathchar, delims);

    while (result != NULL)
    {
        if (!strcmp(result, "ROOT"))
        {
            returnValue = vv->getObjectsRoot();
        }
        else
        {
            i = atoi(result);

            while (isHelperNode(returnValue))
            {
                if (returnValue)
                {
                    help = dynamic_cast<vsg::Group*>(returnValue);
                    if (help)
                        returnValue = help->children[0];
                    else
                        returnValue = NULL;
                }
            }
            if (returnValue)
            {
                help = dynamic_cast<vsg::Group*>(returnValue);
                if (help)
                {
                    //CoOP
                    if (isdigit(result[0]) && (((unsigned int)i) < help->children.size()))
                    {
                        returnValue = help->children[i];
                        //else
                        //ERROR Node was deleted?
                    }
                    else
                    {
                        std::string name = std::string(result);
                        for (unsigned int j = 0; j < help->children.size(); j++)
                        {
                            child = help->children[j];
                            while (isHelperNode(child))
                            {
                                parent = dynamic_cast<vsg::Group*>(child);
                                if (parent)
                                    child = parent->children[0];
                                else
                                    child = NULL;
                            }
                            if (child)
                            {
                                std::string n;
                                child->getValue("name", name);
                                if (n == name)
                                    returnValue = child;
                            }
                        }
                    }
                }
                else
                    returnValue = NULL;
            }
        }
        result = strtok(NULL, delims);
    }

    //path leads to a helper node
    while (isHelperNode(returnValue))
    {
        help = dynamic_cast<vsg::Group*>(returnValue);
        if (help!=NULL && help->children.size()>0)
            returnValue = help->children[0];
        else
            returnValue = NULL;
    }
    delete[] pathchar;

    return returnValue;
}
