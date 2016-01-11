/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRSelectionManager.h"
#include "coVRPluginSupport.h"
#include "coIntersection.h"
#include <OpenVRUI/coNavInteraction.h>

#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/Switch>
#include <osgFX/Outline>
#include <osg/MatrixTransform>

#include <util/string_util.h>

using namespace opencover;
using namespace vrui;
coVRSelectionManager *coVRSelectionManager::instance()
{
    static coVRSelectionManager *singleton = NULL;
    if (!singleton)
        singleton = new coVRSelectionManager();
    return singleton;
}

coVRSelectionManager::coVRSelectionManager()
{

    selectionInteractionA = new coNavInteraction(coInteraction::ButtonA, "Selection", coInteraction::High);
    selectedNodeList.clear();
    selectedParentList.clear();
    selectionNodeList.clear();
    SelOnOff = 1;
    SelWire = 3;
    SelRed = 0.0f;
    SelGreen = 0.1f;
    SelBlue = 1.0f;

    updateManager = cover->getUpdateManager();
    updateManager->add(this);
}
coVRSelectionManager::~coVRSelectionManager()
{
    delete selectionInteractionA;
    updateManager->remove(this);
}

osg::BoundingSphere coVRSelectionManager::getBoundingSphere(osg::Node *objRoot)
{
    osg::BoundingSphere bsphere;

    osg::BoundingBox bb;
    bb.init();
    osg::Node *currentNode = NULL;

    if (selectedNodeList.empty())
    {
        osg::Group *root = objRoot->asGroup();
        if (root)
        {
            for (unsigned int i = 0; i < root->getNumChildren(); i++)
            {
                currentNode = root->getChild(i);
                const osg::Transform *transform = currentNode->asTransform();
                if ((!transform || transform->getReferenceFrame() == osg::Transform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
                {
                    bb.expandBy(/*currentNode->getBound()*/ cover->getBBox(currentNode));
                }
            }

            if (bb.valid())
            {

                bsphere._center = bb.center();
                bsphere._radius = 0.0f;
                for (unsigned int i = 0; i < root->getNumChildren(); i++)
                {
                    currentNode = root->getChild(i);
                    const osg::Transform *transform = currentNode->asTransform();
                    if ((!transform || transform->getReferenceFrame() == osg::Transform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
                    {
                        bsphere.expandRadiusBy(/*currentNode->getBound()*/ cover->getBBox(currentNode));
                    }
                }
            }
        }
    }
    else
    {
        osg::Matrix startBaseMat, trans;
        std::list<osg::ref_ptr<osg::Node> >::iterator iter = selectedNodeList.begin();
        while (iter != selectedNodeList.end())
        {
            currentNode = (*iter).get();
            const osg::Transform *transform = currentNode->asTransform();
            if ((!transform || transform->getReferenceFrame() == osg::Transform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
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
                    if (dynamic_cast<osg::MatrixTransform *>(currentNode))
                    {
                        trans = ((osg::MatrixTransform *)currentNode)->getMatrix();
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
                const osg::Transform *transform = currentNode->asTransform();
                if ((!transform || transform->getReferenceFrame() == osg::Transform::RELATIVE_RF) && strncmp(currentNode->getName().c_str(), "Avatar ", 7) != 0)
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
                        if (dynamic_cast<osg::MatrixTransform *>(currentNode))
                        {
                            trans = ((osg::MatrixTransform *)currentNode)->getMatrix();
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
}
bool coVRSelectionManager::update()
{

    if (selectionInteractionA->wasStarted())
    {
        if (!cover->getIntersectedNode())
        {
            clearSelection();
            pickedObjChanged();
            return true;
        }
        const osg::NodePath &intersectedNodePath = cover->getIntersectedNodePath();
        bool isSceneNode = false;
        for (std::vector<osg::Node *>::const_iterator iter = intersectedNodePath.begin();
             iter != intersectedNodePath.end();
             ++iter)
        {
            if ((*iter) == cover->getObjectsRoot())
            {
                isSceneNode = true;
                break;
            }
        }

        if (isSceneNode)
        {
            std::vector<osg::Node *>::const_iterator iter = intersectedNodePath.end();
            --iter;
            --iter;
            osg::Group *parent = (*iter)->asGroup();
            while (parent && isHelperNode(parent))
            {
                iter--;
                parent = (*iter)->asGroup();
            }
            if (parent)
            {
                clearSelection();
                addSelection(parent, cover->getIntersectedNode());
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

void coVRSelectionManager::addListener(coSelectionListener *l)
{
    if (l && find(listenerList.begin(), listenerList.end(), l) == listenerList.end())
        listenerList.push_back(l);
}

void coVRSelectionManager::removeListener(coSelectionListener *l)
{
    list<coSelectionListener *>::iterator it = find(listenerList.begin(), listenerList.end(), l);
    if (it != listenerList.end())
        listenerList.erase(it);
}

void coVRSelectionManager::selectionChanged()
{
    for (list<coSelectionListener *>::iterator item = listenerList.begin(); item != listenerList.end();)
    {
        if (!(*item)->selectionChanged())
            item = listenerList.erase(item);
        else
            ++item;
    }
}

void coVRSelectionManager::pickedObjChanged()
{
    for (list<coSelectionListener *>::iterator item = listenerList.begin(); item != listenerList.end();)
    {
        if (!(*item)->pickedObjChanged())
            item = listenerList.erase(item);
        else
            ++item;
    }
}

void coVRSelectionManager::removeNode(osg::Node *node)
{
    if (selectedNodeList.empty())
        return;

    std::list<osg::ref_ptr<osg::Node> >::iterator childIter = selectedNodeList.begin();
    std::list<osg::ref_ptr<osg::Group> >::iterator parentIter = selectedParentList.begin();
    std::list<osg::ref_ptr<osg::Group> >::iterator nodeIter = selectionNodeList.begin();

    while (childIter != selectedNodeList.end()) //&& !selectedNodeList.empty())
    {
        if ((node == (*childIter).get()) || (node == (*parentIter).get()) || haveToDelete((*parentIter).get(), node))
        {
            if ((*nodeIter).get())
            {
                osg::Group *mygroup = NULL;
                osg::Node *child = NULL;
                if ((*nodeIter).get()->getNumParents())
                {
                    osg::Node *parent = (*nodeIter).get()->getParent(0);
                    mygroup = parent->asGroup();
                }
                if ((*nodeIter).get()->getNumChildren())
                {
                    child = (*nodeIter).get()->getChild(0);
                }
                if (child && mygroup && (*nodeIter).get())
                {
                    mygroup->replaceChild((*nodeIter).get(), child);
                    (*nodeIter).get()->removeChild(child);
                }
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

bool coVRSelectionManager::haveToDelete(osg::Node *parent, osg::Node *node)
{
    if ((parent->getNumParents() > 1) || (parent->getNumParents() == 0))
        return false;
    else
    {
        if (parent->getParent(0) == node)
            return true;
        else
            return haveToDelete(parent->getParent(0), node);
    }
}

void coVRSelectionManager::selectionCallback(void *, buttonSpecCell *spec)
{
    //fprintf(stderr,"coVRSelectionManager::selectionCallback\n");
    if (spec->state == 1.0)
    {
        coVRSelectionManager::instance()->setSelectionOnOff(1);
    }
    else
    {
        coVRSelectionManager::instance()->setSelectionOnOff(0);
    }
}

void coVRSelectionManager::setSelectionColor(float R, float G, float B)
{
    SelRed = R;
    SelGreen = G;
    SelBlue = B;
}

void coVRSelectionManager::setSelectionWire(int mode)
{
    SelWire = mode;
}

void coVRSelectionManager::setSelectionOnOff(int mode)
{
    if (mode)
    {
        coVRSelectionManager::instance()->setSelectionWire(3);
        coVRSelectionManager::instance()->setSelectionColor(0, 0, 1);

        if (!selectionInteractionA->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(selectionInteractionA);
            coIntersection::instance()->isectAllNodes(true);
        }
    }
    else
    {

        if (selectionInteractionA->isRegistered())
        {
            coInteractionManager::the()->unregisterInteraction(selectionInteractionA);
            coIntersection::instance()->isectAllNodes(false);
        }
    }
}

void coVRSelectionManager::showhideSelection(int mode)
{
    SelOnOff = mode;
}

void coVRSelectionManager::receiveAdd(const char *messageData)
{
    if (!messageData)
        return;

    std::string str(messageData);
    std::vector<std::string> tokens = split(str, '?');

    std::string &parentPath = tokens[0];
    std::string &nodePath = tokens[1];

    osg::Node *parent = validPath(parentPath);
    osg::Node *node = validPath(nodePath);
    osg::Group *parentG = parent->asGroup();

    addSelection(parentG, node, false);
    pickedObjChanged();
}

void coVRSelectionManager::addSelection(osg::Group *parent, osg::Node *selectedNode, bool send)
{
    if ((!parent) || (!selectedNode))
        return;

    if (send)
    {
        std::string msg = generatePath(parent);
        msg.append("?");
        msg.append(generatePath(selectedNode));
        char *charmsg = (char *)msg.c_str();
        int len = strlen(charmsg) + 1;
        cover->sendBinMessage("ADD_SELECTION", charmsg, len);
    }

    osg::Group *selectionNode = NULL;

    /*osg::Node *existNode = getHelperNode(parent, selectedNode, SELECTION);
   if(existNode)
      selectionNode = existNode->asGroup();
   else*/

    if (SelOnOff)
    {
        // create a material
        osg::Material *selMaterial = new osg::Material();
        selMaterial->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4f(SelRed, SelGreen, SelBlue, 1.0f));
        selMaterial->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4f(SelRed, SelGreen, SelBlue, 1.0f));
        selMaterial->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
        selMaterial->setShininess(osg::Material::FRONT_AND_BACK, 10.f);
        selMaterial->setColorMode(osg::Material::OFF);

        if (SelWire == 0) // FILLED
        {
            // apply material
            selectionNode = new osg::Group();
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
            selectionNode = new osg::Group();
            osg::StateSet *ss = selectionNode->getOrCreateStateSet();
            osg::PolygonMode *polymode = new osg::PolygonMode();
            polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
            ss->setAttribute(selMaterial, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED);
            ss->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
        }
        else if (SelWire == 2) // WIREFRAME_OBJECT_COLOR
        {
            // apply material
            selectionNode = new osg::Group();
            osg::StateSet *ss = selectionNode->getOrCreateStateSet();
            osg::PolygonMode *polymode = new osg::PolygonMode();
            polymode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
            ss->setAttributeAndModes(polymode, osg::StateAttribute::OVERRIDE | osg::StateAttribute::PROTECTED | osg::StateAttribute::ON);
        }
        else if (SelWire == 3) // OUTLINE
        {
            // apply material
            osgFX::Outline *out = new osgFX::Outline;
            out->setColor(osg::Vec4f(SelRed, SelGreen, SelBlue, 1.0f));
            out->setWidth(4);
            selectionNode = static_cast<osg::Group *>(out);
        }
    }
    else
    {
        selectionNode = new osg::Group();
    }

    /*if(!existNode)*/
    /*{*/
    insertHelperNode(parent, selectedNode, selectionNode, SELECTION);

    selectedNodeList.push_back(selectedNode);
    selectedParentList.push_back(parent);
    selectionNodeList.push_back(selectionNode);
    selectionChanged();
    /*}*/
}

void coVRSelectionManager::receiveClear()
{
    clearSelection(false);
}

void coVRSelectionManager::clearSelection(bool send)
{
    if (send)
    {
        cover->sendBinMessage("CLEAR_SELECTION", "", 0);
    }

    while (!selectionNodeList.empty())
    {
        osg::ref_ptr<osg::Group> mySelectionNode = selectionNodeList.front();
        if (mySelectionNode->getNumParents())
        {
            osg::Node *parent = mySelectionNode->getParent(0);
            osg::Node *child = NULL;
            if (mySelectionNode->getNumChildren() > 0)
                child = mySelectionNode->getChild(0);
            osg::Group *mygroup = parent->asGroup();
            if (child && mygroup && mySelectionNode.get())
            {
                mygroup->replaceChild(mySelectionNode.get(), child);
                mySelectionNode->removeChild(child);
            }
        }
        selectionNodeList.pop_front();
    }

    selectedNodeList.clear();
    selectedParentList.clear();
    selectionChanged();
}

osg::Group *coVRSelectionManager::getHelperNode(osg::Group *parent, osg::Node *child, HelperNodeType type)
{

    if ((!parent) || (!child))
        return NULL;

    int index = type;

    osg::Group *helpParent = NULL;
    osg::Group *helpParent2 = NULL;

    for (unsigned int i = 0; i < child->getNumParents(); i++)
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
    }
    return NULL;
}

void coVRSelectionManager::insertHelperNode(osg::Group *parent, osg::Node *child, osg::Group *insertNode, HelperNodeType type, bool show)
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
        insertNode->addDescription(strType);
    }

    bool equal = false;
    osg::Node *myChild = NULL;
    osg::Group *myParent = NULL;
    osg::Group *helpParent = NULL;
    osg::Group *helpParent2 = NULL;
    osg::Node *helpChild = NULL;

    for (unsigned int i = 0; i < child->getNumParents(); i++)
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
    }

    if (insertNode && myParent && myChild)
    {
        if (equal)
        {
            osg::Group *equalNode = myChild->asGroup();
            insertNode->addChild(equalNode->getChild(0));

            if (index == SHOWHIDE)
            {
                osg::Switch *mySwitch = (osg::Switch *)insertNode;
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
    }
}

void coVRSelectionManager::markAsHelperNode(osg::Node *node)
{
    node->addDescription("SELECTIONHELPER");
}

bool coVRSelectionManager::isHelperNode(osg::Node *node)
{
    for (unsigned int j = 0; node && j < node->getNumDescriptions(); j++)
    {
        if (strncmp(node->getDescription(j).c_str(), "SELECTIONHELPER", 15) == 0)
        {
            return true;
        }
    }
    return false;
}

bool coVRSelectionManager::hasType(osg::Node *node)
{
    for (unsigned int j = 0; node && j < node->getNumDescriptions(); j++)
    {
        if (strncmp(node->getDescription(j).c_str(), "TYPE", 4) == 0)
        {
            return true;
        }
    }
    return false;
}

int coVRSelectionManager::getHelperType(osg::Node *node)
{

    for (unsigned int j = 0; node && j < node->getNumDescriptions(); j++)
    {
        if (strncmp(node->getDescription(j).c_str(), "TYPE ", 5) == 0)
        {
            return atoi(node->getDescription(j).c_str() + 5);
        }
    }
    return 100;
}
std::string coVRSelectionManager::generateNames(osg::Node *node)
{
    std::string name = "";
    std::string newstr = "";
    std::string oldstr = "";
    osg::Group *parent = NULL;

    bool arrive = false;

    while (!arrive)
    {
        name = node->getName();
        if (node == cover->getObjectsRoot())
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

    } //end while

    return oldstr;
}
std::string coVRSelectionManager::generatePath(osg::Node *node)
{
    std::string name = "";
    std::string newpath = "";
    std::string oldpath = "";
    int index = 0;
    osg::Group *parent = NULL;
    osg::Node *realNode = NULL;
    bool arrive = false;

    while (!arrive)
    {
        name = node->getName();
        if (node == cover->getObjectsRoot())
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
            if (node->getNumParents())
            {

                parent = node->getParent(0)->asGroup();
                realNode = node;
                while (isHelperNode(parent))
                {

                    node = parent;
                    parent = parent->getParent(0)->asGroup();
                }

                for (unsigned int i = 0; parent && i < parent->getNumChildren(); i++)
                {
                    if (parent->getChild(i) == node)
                    {
                        index = i;
                    }
                }

                //CoOP
                if (node->getParent(0) == cover->getObjectsRoot())
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
            }

        } //end else

        oldpath = newpath;

    } //end while

    return newpath;
}

osg::Node *coVRSelectionManager::validPath(std::string path)
{
    osg::Node *returnValue = NULL;

    char delims[] = ";";
    char *result = NULL;
    int i = 0;
    osg::Group *help = NULL;
    osg::Node *child = NULL;
    osg::Group *parent = NULL;

    char *pathchar = new char[path.length() + 1];
    strcpy(pathchar, path.c_str());

    result = strtok(pathchar, delims);

    while (result != NULL)
    {
        if (!strcmp(result, "ROOT"))
        {
            returnValue = cover->getObjectsRoot();
        }
        else
        {
            i = atoi(result);

            while (isHelperNode(returnValue))
            {
                if (returnValue)
                {
                    help = returnValue->asGroup();
                    if (help)
                        returnValue = help->getChild(0);
                    else
                        returnValue = NULL;
                }
            }
            if (returnValue)
            {
                help = returnValue->asGroup();
                if (help)
                {
                    //CoOP
                    if (isdigit(result[0]) && (((unsigned int)i) < help->getNumChildren()))
                    {
                        returnValue = help->getChild(i);
                        //else
                        //ERROR Node was deleted?
                    }
                    else
                    {
                        std::string name = std::string(result);
                        for (unsigned int j = 0; j < help->getNumChildren(); j++)
                        {
                            child = help->getChild(j);
                            while (isHelperNode(child))
                            {
                                parent = child->asGroup();
                                if (parent)
                                    child = parent->getChild(0);
                                else
                                    child = NULL;
                            }
                            if (child)
                            {
                                if (child->getName() == name)
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
        help = returnValue->asGroup();
        if (help)
            returnValue = help->getChild(0);
        else
            returnValue = NULL;
    }
    delete[] pathchar;

    return returnValue;
}
