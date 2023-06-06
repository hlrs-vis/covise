/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coInteractor.h>
#include <cover/coVRFileManager.h>

#include <osg/Group>
#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Material>
#include <osg/Switch>
#include <osg/TexGenNode>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osgFX/Outline>

#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFrame.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coNavInteraction.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coFlatButtonGeometry.h>
#include <OpenVRUI/coRectButtonGeometry.h>
#include <OpenVRUI/coMouseButtonInteraction.h>
#include <cover/coBillboard.h>
#include <virvo/vvtoolshed.h>
#include <cover/VRVruiRenderInterface.h>

#include <PluginUtil/PluginMessageTypes.h>

#include "Schweissen.h"

using namespace osg;

Schweissen *Schweissen::plugin = NULL;
float Schweissbrenner::basevalue = 10;
const string MARKER = "Schweissen: ";
vrml::Player *Schweissen::player = NULL;

SchweissbrennerSensor::SchweissbrennerSensor(Schweissbrenner *m, osg::Node *n)
    : coPickSensor(n)
{
    mySchweissbrenner = m;
    setThreshold(50);
    //threshold = 50*50;
}

SchweissbrennerSensor::~SchweissbrennerSensor()
{
    if (active)
        disactivate();
}

void SchweissbrennerSensor::activate()
{
    Schweissen::plugin->setcurrentSchweissbrenner(mySchweissbrenner);
    active = 1;
    //mySchweissbrenner->setIcon(1);
}

void SchweissbrennerSensor::disactivate()
{
    if (!Schweissen::plugin->interactionA->isRunning())
        Schweissen::plugin->setcurrentSchweissbrenner(NULL);
    active = 0;
    //mySchweissbrenner->setIcon(0);
}

Schweissen::Schweissen()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool Schweissen::init()
{

    if (Schweissen::plugin != NULL)
        return false;

    Schweissen::plugin = this;

    //  cubeToolTip = NULL;
    menuSelected = false;
    //selectedSchweissenId = -1;

    // default button size
    float buttonSize[] = { 80, 15 };

    // schweissbrenner menu positions
    float x[] = { 0, 60 };
    float y[] = { 60, 45, 30, 15 };
    float z = 1;
    float labelSize = 8;

    // get host name
    char temp[20];
    if (gethostname(temp, 20) == -1)
    {
        myHost.append("NoName");
    }
    else
    {
        myHost.append(temp);
    }

    // create new schweissbrenner menu
    schweissbrennerMenuItem = new coSubMenuItem("Schweissen");

    // create row menu to add items
    schweissbrennerMenu = new coRowMenu("Schweissbrenner");
    schweissbrennerMenuItem->setMenu(schweissbrennerMenu);

    // create a new schweissbrenner
    schweissbrennerMenuCheckbox = new coCheckboxMenuItem("neuer Schweissbrenner", false);
    schweissbrennerMenu->add(schweissbrennerMenuCheckbox);
    schweissbrennerMenuCheckbox->setMenuListener(this);

    // hide all schweissbrenners
    hideMenuCheckbox = new coCheckboxMenuItem("zeige Schweissbrenner", true);
    schweissbrennerMenu->add(hideMenuCheckbox);
    hideMenuCheckbox->setMenuListener(this);

    // add delete all button
    deleteAllButton = new coButtonMenuItem("Alle loeschen");
    schweissbrennerMenu->add(deleteAllButton);
    deleteAllButton->setMenuListener(this);

    // get the root node for COVISE objects
    mainNode = new osg::Group;
    mainNode->setName("SchweissenMainNode");
    cover->getObjectsRoot()->addChild((osg::Node *)mainNode.get());
    //setScaleAll(scaleMenuPoti->getValue());            // set the Main node

    //create a pop up menu for schweissbrenner interaction
    schweissbrennerHandle = new coPopupHandle("SchweissenControl");
    schweissbrennerFrame = new coFrame("UI/Frame");
    schweissbrennerPanel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    schweissbrennerLabel = new coLabel;
    schweissbrennerLabel->setString(MARKER);
    schweissbrennerLabel->setPos(x[0], y[0], z + 1);
    schweissbrennerLabel->setFontSize(labelSize);
    schweissbrennerHandle->addElement(schweissbrennerFrame);
    schweissbrennerDeleteButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Schweissen/delete"), this);
    schweissbrennerPlayButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Schweissen/play"), this);
    schweissbrennerRecordButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Schweissen/record"), this);
    colorPoti = new coValuePoti("Color", this, "Volume/valuepoti-bg");
    schweissbrennerPlayButton->setSize(40);
    schweissbrennerRecordButton->setSize(40);
    schweissbrennerDeleteButton->setSize(40);
    schweissbrennerPlayButton->setPos(x[0], y[1], z);
    schweissbrennerPlayButton->setVisible(false);
    schweissbrennerRecordButton->setVisible(false);
    schweissbrennerRecordButton->setPos(x[0], y[2], z);
    schweissbrennerDeleteButton->setPos(x[0], y[3], z);
    colorPoti->setPos(x[1], y[2], z);
    colorPoti->setMin(0);
    colorPoti->setMax(1);
    schweissbrennerPanel->addElement(schweissbrennerLabel);
    schweissbrennerPanel->addElement(schweissbrennerPlayButton);
    schweissbrennerPanel->addElement(schweissbrennerRecordButton);
    schweissbrennerPanel->addElement(schweissbrennerDeleteButton);
    schweissbrennerPanel->addElement(colorPoti);
    schweissbrennerPanel->setScale(5);
    schweissbrennerPanel->resize();
    schweissbrennerFrame->addElement(schweissbrennerPanel);
    schweissbrennerHandle->setVisible(false);

    currentSchweissbrenner = NULL;
    previousSchweissbrenner = NULL;

    // add schweissbrenner menu to the main menu
    cover->getMenu()->add(schweissbrennerMenuItem);

    moving = false;
    interactionA = new coNavInteraction(coInteraction::ButtonA, "SchweissenPlacement", coInteraction::Medium);
    interactionC = new coNavInteraction(coInteraction::ButtonC, "SchweissenPlacement", coInteraction::Medium);
    SchweissbrennerNode = coVRFileManager::instance()->loadIcon("schweissbrenner");

    handTransform = new osg::MatrixTransform();

    handTransform->setNodeMask(handTransform->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));
    cover->getScene()->addChild(handTransform);

    return true;
}

string Schweissen::getMyHost()
{
    return myHost;
}

void Schweissen::setMMString(char *dst, string src)
{
    assert(src.size() < 20);
    strcpy(dst, src.c_str());
}

/*
 * Set pre determined schweissbrenners if structure exists
 *
 */
/*void Schweissen::setSchweissens(string structure)
{
  SchweissbrennerMessage mm;
  mm.id = maxid;
  setMMString(mm.host, getMyHost());
  setMMString(mm.filename, structure);
  mm.token = 6;
  cover->sendMessage(this,
    coVRPluginSupport::TO_SAME,
    0,
    sizeof(SchweissbrennerMessage),
    &mm);
}*/

// this is called if the plugin is removed at runtime
Schweissen::~Schweissen()
{
    vector<Schweissbrenner *>::iterator it = schweissbrenner.begin();
    for (; it < schweissbrenner.end(); it++)
    {
        delete (*it);
    }
    cover->getObjectsRoot()->removeChild((osg::Node *)mainNode.get());
    removeMenuEntry();
    delete interactionA;
    delete interactionC;
}

// hide all schweissbrenners (make transparent)
void Schweissen::setVisible(bool vis)
{
    vector<Schweissbrenner *>::iterator it = schweissbrenner.begin();
    for (; it < schweissbrenner.end(); it++)
    {
        (*it)->setVisible(vis);
    }
}
osg::Node *findNamedNode(const std::string &searchName,
                         osg::Node *currNode)
{
    osg::Group *currGroup;
    osg::Node *foundNode;

    // check to see if we have a valid (non-NULL) node.
    // if we do have a null node, return NULL.
    if (!currNode)
    {
        return NULL;
    }

    // We have a valid node, check to see if this is the node we
    // are looking for. If so, return the current node.
    if (currNode->getName() == searchName)
    {
        return currNode;
    }

    // We have a valid node, but not the one we are looking for.
    // Check to see if it has children (non-leaf node). If the node
    // has children, check each of the child nodes by recursive call.
    // If one of the recursive calls returns a non-null value we have
    // found the correct node, so return this node.
    // If we check all of the children and have not found the node,
    // return NULL
    currGroup = currNode->asGroup(); // returns NULL if not a group.
    if (currGroup)
    {
        for (unsigned int i = 0; i < currGroup->getNumChildren(); i++)
        {
            foundNode = findNamedNode(searchName, currGroup->getChild(i));
            if (foundNode)
                return foundNode; // found a match!
        }
        return NULL; // We have checked each child node - no match found.
    }
    else
    {
        return NULL; // leaf node, no match
    }
}
Schweissbrenner::Schweissbrenner(int i, string host, osg::Group *node, float initscale)
{
    id = i;
    pos = new osg::MatrixTransform();
    _scaleVal = 1.0f;

    // create a billboard to attach the text to
    billText = new osg::Billboard;
    billText->setNormal(osg::Vec3(0.0f, 0.0f, 1.0f));
    billText->setAxis(osg::Vec3(0.0f, 0.0f, 1.0f));
    billText->setMode(osg::Billboard::AXIAL_ROT);

    // create label for the the schweissbrenner
    osgText::Text *schweissbrennerLabel = new osgText::Text;
    schweissbrennerLabel->setDataVariance(Object::DYNAMIC);
    schweissbrennerLabel->setCharacterSize(basevalue * cover->getScale());

    // set host name
    hname = host;

    // convert int to string for the label
    std::ostringstream label;
    if (label << id)
    {
        schweissbrennerLabel->setText(label.str(), osgText::String::ENCODING_UTF8);
    }
    schweissbrennerLabel->setAxisAlignment(osgText::Text::XZ_PLANE);

    // Set the text to render with alignment anchor
    schweissbrennerLabel->setDrawMode(osgText::Text::TEXT);
    schweissbrennerLabel->setAlignment(osgText::Text::CENTER_BOTTOM);
    //schweissbrennerLabel->setPosition( osg::Vec3(0.0f, 0.0f,-(basevalue * cover->getScale())));
    schweissbrennerLabel->setColor(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    billText->addDrawable(schweissbrennerLabel);
    billText->setPosition(0, osg::Vec3(0.0f, 0.0f, -(basevalue * cover->getScale())));
    mat = NULL;
    //geo = new osg::Geode;
    scale = new osg::MatrixTransform;
    osg::Node *n = coVRFileManager::instance()->loadIcon("schweissbrenner");
    if (n)
    {

        //out = new osgFX::Outline;
        //out->setWidth(0);
        //out->setColor(osg::Vec4f(1, 0, 0, 1.0f));
        //out->addChild(n);
        scale->addChild(n);
        osg::Node *griff = findNamedNode("Griff-FACES", n);
        if (griff != NULL)
        {
            osg::Geode *g = dynamic_cast<osg::Geode *>(griff);
            if (g != NULL)
            {
                osg::Drawable *d = g->getDrawable(0);
                if (d != NULL)
                {
                    mat = dynamic_cast<osg::Material *>(d->getStateSet()->getAttribute(osg::StateAttribute::MATERIAL));
                }
            }
        }
    }

    // call set color (with the default color loaded from the config file)
    //setColor(0.5);    // set all cones to the same color by default
    if (id != 0)
        setColor((id % 10) / 10.0f); // use different colors for cones by default (up to 10, then repeat)

    /*// set rendering hints
  osg::TessellationHints* hints = new osg::TessellationHints();
  hints->setDetailRatio(0.2f);

  float size = basevalue * cover->getScale();
  osg::Vec3 center(0.0f, 0.0f, -0.75f * size);
  cone = new osg::Cone(center, size / 5.0f , size);
  osg::ShapeDrawable* conedraw = new osg::ShapeDrawable(cone);
  conedraw->setTessellationHints(hints);
  osg::StateSet* state = conedraw->getOrCreateStateSet();
  state->setAttributeAndModes(mat);
  conedraw->setStateSet(state);
  conedraw->setUseDisplayList(false);

  // set main scale
  osg::Matrix matrix = scale->getMatrix();
  matrix.makeScale(initscale, initscale,initscale);
  scale->setMatrix(matrix);

  // attach nodes to the mainNode
  geo->addDrawable(conedraw);

  //geo->addDrawable(schweissbrennerLabel);
  scale->addChild(geo);*/
    //scale->addChild(billText);
    pos->addChild(scale);
    node->addChild(pos);
    mySensor = new SchweissbrennerSensor(this, n);
    Schweissen::plugin->sensorList.append(mySensor);
}

void Schweissbrenner::setVisible(bool vis)
{
    if (vis)
    {
        scale->setNodeMask(~0);
        billText->setNodeMask(~0);
    }
    else
    {
        scale->setNodeMask(0);
        billText->setNodeMask(0);
    }
}

int Schweissbrenner::getID()
{
    return id;
}

string Schweissbrenner::getHost()
{
    return hname;
}

void Schweissbrenner::setBaseSize(float basescale)
{
    basevalue = basescale;
}

void Schweissbrenner::setColor(float hue)
{
    float r, g, b;
    _hue = hue;
    vvToolshed::HSBtoRGB(_hue, 1, 1, &r, &g, &b);
    osg::Vec4 color(r, g, b, 1.0);
    if (mat != NULL)
        mat->setDiffuse(osg::Material::FRONT_AND_BACK, color);
}

float Schweissbrenner::getColor()
{
    return _hue;
}

void Schweissbrenner::setSelectedHost(const char *host)
{
    _selectedhname.clear();
    if (host != NULL)
    {
        _selectedhname.append(host);
    }
}

void Schweissbrenner::setAmbient(osg::Vec4 spec)
{
    if (spec[0] == 1.0)
    {
        //out->setWidth(1.0);
    }
    else
    {
        //out->setWidth(0.0);
    }
    if (mat != NULL)
        mat->setAmbient(osg::Material::FRONT_AND_BACK, spec);
}

/////// NEED TO COMPLETE ///////
Schweissbrenner::~Schweissbrenner()
{
    pos->getParent(0)->removeChild(pos);
    if (Schweissen::plugin->sensorList.find(mySensor))
        Schweissen::plugin->sensorList.remove();
    delete mySensor;
}

void Schweissbrenner::setMat(osg::Matrix &mat)
{
    pos->setMatrix(mat);
}

void Schweissbrenner::setScale(float scaleset)
{
    osg::Matrix mat;
    mat.makeScale(scaleset, scaleset, scaleset);
    //scale->setMatrix(mat);
    _scaleVal = scaleset;
}

float Schweissbrenner::getScale()
{
    return _scaleVal;
}

osg::Matrix Schweissbrenner::getPos()
{
    return pos->getMatrix();
}

// check if the schweissbrenner is available for selection
bool Schweissbrenner::isSelectable(string shost)
{
    if (matches(shost) || (_selectedhname.empty()))
    {
        return true;
    }
    else
    {
        return false;
    }
}

// check if the schweissbrenner matches
bool Schweissbrenner::matches(string shost)
{
    return (_selectedhname == shost);
}

/*float Schweissbrenner::getDist(osg::Vec3 &a)
{
   osg::Vec3 b,diff;
   osg::Matrix mat;
   pos->getMatrix(mat);
   mat.getRow(3,b);
   diff = a - b;
   return diff.dot(diff);
}*/

void Schweissen::preFrame()
{
    sensorList.update();
    static osg::Matrix startPos;
    static osg::Matrix invStartHand;
    osg::Matrix scaleHand;
    osg::Matrix invScaleHand;
    float s = cover->getScale();
    scaleHand = osg::Matrix::scale(s, s, s);
    invScaleHand = osg::Matrix::scale(1.0 / s, 1.0 / s, 1.0 / s);
    handTransform->setMatrix(scaleHand * cover->getPointerMat());

    SchweissbrennerMessage mm;

    if (!interactionA->isRunning() && !interactionC->isRunning())
    {
        //check if the object was previously moving
        if (currentSchweissbrenner)
        {
            if (moving && !schweissbrennerHandle->isVisible())
            {
                // send message giving up control of the schweissbrenner
                mm.id = currentSchweissbrenner->getID();
                setMMString(mm.host, currentSchweissbrenner->getHost());
                setMMString(mm.selecthost, getMyHost());
                mm.token = 5;
                cover->sendMessage(this,
                                   coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::Marker1,
                                   sizeof(SchweissbrennerMessage),
                                   &mm);
            }
        }
        moving = false;
    }
    if ((currentSchweissbrenner) || (menuSelected))
    {
        if (!interactionA->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(interactionA);
        }
    }
    else
    {
        if (interactionA->isRegistered())
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
    }
    if ((currentSchweissbrenner))
    {
        if (!interactionC->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(interactionC);
        }
    }
    else
    {
        if (interactionC->isRegistered())
        {
            coInteractionManager::the()->unregisterInteraction(interactionC);
            setcurrentSchweissbrenner(NULL);
        }
    }

    if (interactionA->wasStarted())
    {
        if (currentSchweissbrenner) // start moving the current Schweissen
        {
            invStartHand.invert(cover->getPointerMat());
            osg::Matrix tmp;
            currentSchweissbrenner->getMat(tmp);
            startPos = tmp * cover->getBaseMat();
            //check if another schweissbrenner is being selected to move (releases lock on old schweissbrenner)
            if (currentSchweissbrenner != previousSchweissbrenner)
                schweissbrennerHandle->setVisible(false);

            // send move Message place lock selected schweissbrenner
            if (currentSchweissbrenner->isSelectable(getMyHost()))
            {
                mm.id = currentSchweissbrenner->getID();
                setMMString(mm.host, currentSchweissbrenner->getHost());
                setMMString(mm.selecthost, getMyHost());
                mm.token = 0;
                currentSchweissbrenner->getMat(mm.mat);
                cover->sendMessage(this,
                                   coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::Marker0,
                                   sizeof(SchweissbrennerMessage),
                                   &mm);
            }
            moving = true;
        }
        else if (menuSelected) // create a new Schweissen
        {
            mm.id = getLowestUnusedSchweissenID();
            setMMString(mm.host, getMyHost());

            mm.mat = scaleHand * cover->getPointerMat() * cover->getInvBaseMat();
            mm.token = 0;
            cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Marker0,
                               sizeof(SchweissbrennerMessage),
                               &mm);
            schweissbrennerMenuCheckbox->setState(false);

            menuSelected = 0;
            if (!menuSelected && handTransform->getNumChildren() > 0)
                handTransform->removeChild(SchweissbrennerNode.get());
        }
    }
    if (interactionC->wasStarted())
    {
        if (currentSchweissbrenner)
        {
            // check if the schweissbrenner is selectable
            if (currentSchweissbrenner->isSelectable(getMyHost()))
            {
                // check if selecting another schweissbrenner
                if (currentSchweissbrenner != previousSchweissbrenner)
                {
                    mm.id = currentSchweissbrenner->getID();
                    setMMString(mm.host, currentSchweissbrenner->getHost());
                    setMMString(mm.selecthost, getMyHost());
                    mm.token = 2;
                    cover->sendMessage(this,
                                       coVRPluginSupport::TO_SAME,
                                       PluginMessageTypes::Marker1,
                                       sizeof(SchweissbrennerMessage),
                                       &mm);
                }
                else // close menu
                {
                    schweissbrennerHandle->setVisible(false);
                }
            }
        }
    }
    if (interactionA->isRunning())
    {
        if (currentSchweissbrenner)
        {
            if (!moving) // start moving the current Schweissen
            {
                invStartHand.invert(cover->getPointerMat());
                osg::Matrix tmp;
                currentSchweissbrenner->getMat(tmp);
                startPos = tmp * cover->getBaseMat();
                moving = true;
            }
            osg::Matrix dMat = invStartHand * cover->getPointerMat();
            osg::Matrix current;
            osg::Matrix tmp;
            tmp = startPos * dMat;
            current = tmp * cover->getInvBaseMat();

            // send move Message
            if (currentSchweissbrenner->isSelectable(getMyHost()))
            {
                mm.id = currentSchweissbrenner->getID();
                setMMString(mm.host, currentSchweissbrenner->getHost());
                setMMString(mm.selecthost, getMyHost());
                mm.token = 0;
                //currentSchweissbrenner->getMat(mm.mat);
                mm.mat = current;
                cover->sendMessage(this,
                                   coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::Marker0,
                                   sizeof(SchweissbrennerMessage),
                                   &mm);
            }
        }
    }

    // check if someone closes the schweissbrenner panel
    if ((!schweissbrennerHandle->isVisible()) && (previousSchweissbrenner != NULL))
    {
        // send message giving up control of the schweissbrenner
        if (previousSchweissbrenner->isSelectable(getMyHost()))
        {
            mm.id = previousSchweissbrenner->getID();
            setMMString(mm.host, previousSchweissbrenner->getHost());
            setMMString(mm.selecthost, getMyHost());
            mm.token = 5;
            cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Marker1,
                               sizeof(SchweissbrennerMessage),
                               &mm);
        }
    }
}

/** Use this when creating new schweissbrenners to ensure that the ID list is always continuous.
*/
int Schweissen::getLowestUnusedSchweissenID()
{
    int id = 0;

    while (isIDInUse(id))
    {
        ++id;
    }
    return id;
}

bool Schweissen::isIDInUse(int id)
{
    vector<Schweissbrenner *>::iterator it = schweissbrenner.begin();
    for (; it < schweissbrenner.end(); it++)
    {
        if ((*it)->getID() == id)
            return true;
    }
    return false;
}

void Schweissbrenner::getMat(osg::Matrix &m)
{
    m = pos->getMatrix();
}

void
Schweissen::setcurrentSchweissbrenner(Schweissbrenner *m)
{
    if ((currentSchweissbrenner == NULL) || (m == NULL))
        currentSchweissbrenner = m;
}

void
Schweissen::removeMenuEntry()
{
    delete schweissbrennerMenuItem;
    delete schweissbrennerMenu;
    delete schweissbrennerMenuCheckbox;
    delete hideMenuCheckbox;
    //delete scaleMenuPoti;
    delete deleteAllButton;
    delete schweissbrennerHandle;
    delete schweissbrennerFrame;
    delete schweissbrennerPanel;
    delete schweissbrennerLabel;
    delete schweissbrennerDeleteButton;
    delete schweissbrennerPlayButton;
    delete schweissbrennerRecordButton;
    delete colorPoti;
}

void Schweissen::setScaleAll(float scaleset)
{
    vector<Schweissbrenner *>::iterator it = schweissbrenner.begin();
    for (; it < schweissbrenner.end(); it++)
    {
        (*it)->setScale(scaleset);
    }
}

// adjust individual maker colors
void Schweissen::potiValueChanged(float, float newvalue, coValuePoti *, int)
{
    if (previousSchweissbrenner != NULL)
    {
        SchweissbrennerMessage mm;
        setMMString(mm.host, previousSchweissbrenner->getHost());
        setMMString(mm.selecthost, getMyHost());
        mm.id = previousSchweissbrenner->getID();
        mm.token = 3;
        mm.color = newvalue;
        cover->sendMessage(this,
                           coVRPluginSupport::TO_SAME,
                           PluginMessageTypes::Marker1,
                           sizeof(SchweissbrennerMessage),
                           &mm);
    }
}

void Schweissen::menuEvent(coMenuItem *item)
{
    if (item == schweissbrennerMenuCheckbox)
    {
        menuSelected = schweissbrennerMenuCheckbox->getState();
        if (menuSelected && handTransform->getNumChildren() == 0)
            handTransform->addChild(SchweissbrennerNode.get());
        else if (!menuSelected && handTransform->getNumChildren() > 0)
            handTransform->removeChild(SchweissbrennerNode.get());
    }
    else if (item == hideMenuCheckbox)
    {
        setVisible(hideMenuCheckbox->getState());
    }
    /*else if (item == scaleMenuPoti)
  {
    setScaleAll(scaleMenuPoti->getValue());
  }*/
    else if (item == deleteAllButton)
    {
        deleteAllSchweissens();
    }
}

/*
 * Delete all schweissbrenners
 *
 */
void Schweissen::deleteAllSchweissens()
{
    //check if plugin exists
    if (plugin == NULL)
    {
        fprintf(stderr, "Schweissen constuctor never called\n");
        return;
    }

    SchweissbrennerMessage mm;
    mm.token = 4;
    cover->sendMessage(this,
                       coVRPluginSupport::TO_SAME,
                       PluginMessageTypes::Marker1,
                       sizeof(SchweissbrennerMessage),
                       &mm);
}

// check for button events
void Schweissen::buttonEvent(coButton *cobutton)
{
    SchweissbrennerMessage mm;

    if (cobutton == schweissbrennerRecordButton)
    {
    }
    else if (cobutton == schweissbrennerPlayButton)
    {
    }
    else if (cobutton == schweissbrennerDeleteButton)
    {
        if (previousSchweissbrenner != NULL)
        {
            mm.id = previousSchweissbrenner->getID();
            setMMString(mm.host, previousSchweissbrenner->getHost());
            setMMString(mm.selecthost, getMyHost());
            mm.token = 1;
            cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Marker1,
                               sizeof(SchweissbrennerMessage),
                               &mm);
            setcurrentSchweissbrenner(NULL); //currentSchweissbrenner=NULL;
        }
    }
}

void Schweissen::message(int toWhom, int type, int len, const void *buf)
{

    (void)len;

    if (type != PluginMessageTypes::Marker0 && type != PluginMessageTypes::Marker1)
        return;

    SchweissbrennerMessage *mm = (SchweissbrennerMessage *)buf;
    switch (mm->token)
    {
    case 0: // MOVE/ADD
    {
        Schweissbrenner *curr = NULL;
        vector<Schweissbrenner *>::iterator it = schweissbrenner.begin();
        for (; it < schweissbrenner.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    // need to update schweissbrenner
                    (*it)->setSelectedHost(mm->selecthost);
                    (*it)->setAmbient(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
                    curr = *it;
                    break;
                }
            }
        }

        if (curr == NULL)
        {
            curr = new Schweissbrenner(mm->id, mm->host, mainNode.get(), 1.0);
            schweissbrenner.push_back(curr);
        }

        curr->setMat(mm->mat);
        break;
    }

    case 1: // Remove a schweissbrenner
    {
        vector<Schweissbrenner *>::iterator it = schweissbrenner.begin();
        for (; it < schweissbrenner.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    if (getMyHost() == mm->selecthost)
                    {
                        previousSchweissbrenner = NULL;
                        // close schweissbrennerHandle
                        schweissbrennerLabel->setString(MARKER);
                        schweissbrennerHandle->setVisible(false);
                    }
                    delete *it;
                    schweissbrenner.erase(it);
                    break;
                }
            }
        }
        break;
    }

    case 2: // schweissbrenner selected
    {

        vector<Schweissbrenner *>::iterator it = schweissbrenner.begin();
        for (; it < schweissbrenner.end(); it++)
        {

            // check for newly selected schweissbrenner
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                // check for match and selectable
                if ((*it)->isSelectable(mm->selecthost))
                {
                    // set selected hostname
                    (*it)->setSelectedHost(mm->selecthost);
                    colorPoti->setValue((*it)->getColor());
                    (*it)->setAmbient(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));

                    // make panel visible only for user of the schweissbrenner
                    if (getMyHost() == mm->selecthost)
                    {
                        setcurrentSchweissbrenner(*it); //currentSchweissbrenner = *it;
                        // convert int to string for the label
                        std::ostringstream label;
                        if (label << ((*it)->getID() + 1))
                        {
                            schweissbrennerLabel->setString(MARKER + label.str());
                        }
                        previousSchweissbrenner = *it;
                        schweissbrennerHandle->setVisible(true);
                    }
                }
            }
            else if ((*it)->matches(mm->selecthost))
            {
                (*it)->setSelectedHost(NULL);
                (*it)->setAmbient(osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
            }
        }
        break;
    }

    case 3: // Change schweissbrenner color
    {
        vector<Schweissbrenner *>::iterator it = schweissbrenner.begin();
        for (; it < schweissbrenner.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    (*it)->setColor(mm->color);
                    break;
                }
            }
        }
        break;
    }

    case 4: // Deletes all Schweissens
    {
        vector<Schweissbrenner *>::iterator it;
        int size = schweissbrenner.size();
        for (int i = 0; i < size; i++)
        {

            it = schweissbrenner.begin();
            /*cout << "TransForm: ";
        for(int i = 0;i < 16;i++) {
         cout << ((*it)->getPos().ptr())[i] << ",";
          }
        cout << endl <<"Color " << (*it)->getColor() << endl << endl;
        */
            delete *it;
            schweissbrenner.erase(it);
            // close schweissbrennerHandle
            schweissbrennerLabel->setString(MARKER);
            schweissbrennerHandle->setVisible(false);
        }
        previousSchweissbrenner = NULL;
        setcurrentSchweissbrenner(NULL);
        break;
    }

    case 5: // Release current lock on a schweissbrenner
    {
        vector<Schweissbrenner *>::iterator it = schweissbrenner.begin();
        for (; it < schweissbrenner.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    // reset color of currently selected schweissbrenner
                    (*it)->setSelectedHost(NULL);
                    (*it)->setAmbient(osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
                    //check if selectable should be reset
                    if ((getMyHost() == mm->selecthost) && (*it == previousSchweissbrenner))
                    {
                        schweissbrennerLabel->setString(MARKER);
                        schweissbrennerHandle->setVisible(false);
                        previousSchweissbrenner = NULL;
                    }
                    break;
                }
            }
        }
        break;
    }
    }
}

COVERPLUGIN(Schweissen)
