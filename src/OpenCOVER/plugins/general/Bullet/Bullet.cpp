/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coInteractor.h>

#include <osg/Group>
#include <osg/Node>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/Material>
#include <osg/Switch>
#include <osg/TexGenNode>
#include <osg/Geode>
#include <osg/ShapeDrawable>

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
#include <cover/VRVruiRenderInterface.h>
#include <OpenVRUI/osg/mathUtils.h>

#include <PluginUtil/PluginMessageTypes.h>

#include "Bullet.h"

#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLUni.hpp>

using namespace osg;

Bullet *Bullet::plugin = NULL;
float BulletProbe::basevalue = 10;
const string BULLET = "Bullet: ";
vrml::Player *Bullet::player = NULL;

static FileHandler handlers[] = {
    { NULL,
      Bullet::sloadBullet,
      Bullet::unloadBullet,
      "bullet" }
};

Bullet *Bullet::instance()
{
    return plugin;
}

BulletSensor::BulletSensor(BulletProbe *m, osg::Node *n)
    : coPickSensor(n)
{
    myBulletProbe = m;
    setThreshold(50);
    //threshold = 50*50;
    Bullet::plugin->sensorList.append(this);
}

BulletSensor::~BulletSensor()
{
    if (active)
        disactivate();
    if (Bullet::plugin->sensorList.find(this))
        Bullet::plugin->sensorList.remove();
}

void BulletSensor::activate()
{
    Bullet::plugin->setCurrentBullet(myBulletProbe);
    active = 1;
    //myBulletProbe->setIcon(1);
}

void BulletSensor::disactivate()
{
    Bullet::plugin->setCurrentBullet(NULL);
    active = 0;
    //myBulletProbe->setIcon(0);
}

Bullet::Bullet()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fileName = "test.bullet";

    coVRFileManager::instance()->registerFileHandler(&handlers[0]);
}

bool Bullet::init()
{

	XMLCh *t1 = NULL;
    if (Bullet::plugin != NULL)
        return false;

    impl = xercesc::DOMImplementationRegistry::getDOMImplementation(t1 = xercesc::XMLString::transcode("Core")); xercesc::XMLString::release(&t1);

    Bullet::plugin = this;

    //  cubeToolTip = NULL;
    menuSelected = false;
    //selectedBulletId = -1;

    // default button size
    float buttonSize[] = { 80, 15 };

    // Bullet menu positions
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

    // create new Bullet menu
    BulletMenuItem = new coSubMenuItem("Bullet");

    // create row menu to add items
    BulletMenu = new coRowMenu("Bullet Options");
    BulletMenuItem->setMenu(BulletMenu);

    // create a new Bullet
    BulletMenuCheckbox = new coCheckboxMenuItem("NewBullet", false);
    BulletMenu->add(BulletMenuCheckbox);
    BulletMenuCheckbox->setMenuListener(this);

    // hide all Bullets
    hideMenuCheckbox = new coCheckboxMenuItem("Display Bullets", true);
    BulletMenu->add(hideMenuCheckbox);
    hideMenuCheckbox->setMenuListener(this);

    // scale all Bullets
    scaleMenuPoti = new coPotiMenuItem("Scale Bullets", 0, 10, 1);
    BulletMenu->add(scaleMenuPoti);
    scaleMenuPoti->setMenuListener(this);

    // add delete all button
    deleteAllButton = new coButtonMenuItem("Delete All");
    BulletMenu->add(deleteAllButton);
    deleteAllButton->setMenuListener(this);

    // Save
    SaveButton = new coButtonMenuItem("Save");
    BulletMenu->add(SaveButton);
    SaveButton->setMenuListener(this);

    // get the root node for COVISE objects
    mainNode = new osg::Group;
    mainNode->setName("BulletMainNode");
    cover->getObjectsRoot()->addChild((osg::Node *)mainNode.get());
    setScaleAll(scaleMenuPoti->getValue()); // set the Main node

    //create a pop up menu for Bullet interaction
    BulletHandle = new coPopupHandle("BulletControl");
    BulletFrame = new coFrame("UI/Frame");
    BulletPanel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    BulletLabel = new coLabel;
    BulletLabel->setString(BULLET);
    BulletLabel->setPos(x[0], y[0], z + 1);
    BulletLabel->setFontSize(labelSize);
    BulletHandle->addElement(BulletFrame);
    BulletDeleteButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Bullet/delete"), this);
    BulletPlayButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Bullet/play"), this);
    BulletRecordButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Bullet/record"), this);
    colorPoti = new coValuePoti("Color", this, "Volume/valuepoti-bg");
    BulletPlayButton->setSize(40);
    BulletRecordButton->setSize(40);
    BulletDeleteButton->setSize(40);
    BulletPlayButton->setPos(x[0], y[1], z);
    BulletPlayButton->setVisible(false);
    BulletRecordButton->setVisible(false);
    BulletRecordButton->setPos(x[0], y[2], z);
    BulletDeleteButton->setPos(x[0], y[3], z);
    colorPoti->setPos(x[1], y[2], z);
    colorPoti->setMin(0);
    colorPoti->setMax(1);
    BulletPanel->addElement(BulletLabel);
    BulletPanel->addElement(BulletPlayButton);
    BulletPanel->addElement(BulletRecordButton);
    BulletPanel->addElement(BulletDeleteButton);
    BulletPanel->addElement(colorPoti);
    BulletPanel->setScale(5);
    BulletPanel->resize();
    BulletFrame->addElement(BulletPanel);
    BulletHandle->setVisible(false);

    currentBullet = NULL;
    previousBullet = NULL;

    // add Bullet menu to the main menu
    cover->getMenu()->add(BulletMenuItem);

    moving = false;
    interactionA = new coNavInteraction(coInteraction::ButtonA, "BulletPlacement", coInteraction::Medium);
    interactionC = new coNavInteraction(coInteraction::ButtonC, "BulletPlacement", coInteraction::Medium);

    return true;
}

string Bullet::getMyHost()
{
    return myHost;
}

void Bullet::setMMString(char *dst, string src)
{
    assert(src.size() < 20);
    strcpy(dst, src.c_str());
}

/*
 * Set pre determined Bullets if structure exists
 *
 */
/*void Bullet::setBullets(string structure)
{
  BulletMessage mm;
  mm.id = maxid;
  setMMString(mm.host, getMyHost());
  setMMString(mm.filename, structure);
  mm.token = 6;
  cover->sendMessage(this,
    coVRPluginSupport::TO_SAME,
    0,
    sizeof(BulletMessage),
    &mm);
}*/

// this is called if the plugin is removed at runtime
Bullet::~Bullet()
{

    coVRFileManager::instance()->unregisterFileHandler(&handlers[0]);
    vector<BulletProbe *>::iterator it = bullet.begin();
    for (; it < bullet.end(); it++)
    {
        delete (*it);
    }
    cover->getObjectsRoot()->removeChild((osg::Node *)mainNode.get());
    removeMenuEntry();
    delete interactionA;
    delete interactionC;
}

// hide all Bullets (make transparent)
void Bullet::setVisible(bool vis)
{
    vector<BulletProbe *>::iterator it = bullet.begin();
    for (; it < bullet.end(); it++)
    {
        (*it)->setVisible(vis);
    }
}

BulletProbe::BulletProbe(int i, string host, osg::Group *node, float initscale)
{
    id = i;
    pos = new osg::MatrixTransform();
    _scaleVal = 1.0f;

    // create a billboard to attach the text to
    billText = new osg::Billboard;
    billText->setNormal(osg::Vec3(0.0f, 0.0f, 1.0f));
    billText->setAxis(osg::Vec3(0.0f, 0.0f, 1.0f));
    billText->setMode(osg::Billboard::AXIAL_ROT);

    // create label for the the Bullet
    osgText::Text *BulletLabel = new osgText::Text;
    BulletLabel->setDataVariance(Object::DYNAMIC);
    BulletLabel->setCharacterSize(basevalue * cover->getScale());

    // set host name
    hname = host;

    // convert int to string for the label
    std::ostringstream label;
    if (label << id)
    {
        BulletLabel->setText(label.str(), osgText::String::ENCODING_UTF8);
    }
    BulletLabel->setAxisAlignment(osgText::Text::XZ_PLANE);

    // Set the text to render with alignment anchor
    BulletLabel->setDrawMode(osgText::Text::TEXT);
    BulletLabel->setAlignment(osgText::Text::CENTER_BOTTOM);
    //BulletLabel->setPosition( osg::Vec3(0.0f, 0.0f,-(basevalue * cover->getScale())));
    BulletLabel->setColor(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    billText->addDrawable(BulletLabel);
    billText->setPosition(0, osg::Vec3(0.0f, 0.0f, -(basevalue * cover->getScale())));

    geo = new osg::Geode;
    scale = new osg::MatrixTransform;
    //scale->addChild(BulletGeode);
    mat = new osg::Material;

    // call set color (with the default color loaded from the config file)
    setColor(0.5); // set all cones to the same color by default
    //  setColor((id % 10) / 10.0f);    // use different colors for cones by default (up to 10, then repeat)

    // set rendering hints
    osg::TessellationHints *hints = new osg::TessellationHints();
    hints->setDetailRatio(0.2f);

    float size = basevalue * cover->getScale();
    osg::Vec3 center(0.0f, 0.0f, -0.75f * size);
    cone = new osg::Cone(center, size / 5.0f, size);
    osg::ShapeDrawable *conedraw = new osg::ShapeDrawable(cone);
    conedraw->setTessellationHints(hints);
    osg::StateSet *state = conedraw->getOrCreateStateSet();
    state->setAttributeAndModes(mat);
    conedraw->setStateSet(state);
    conedraw->setUseDisplayList(false);

    // set main scale
    osg::Matrix matrix = scale->getMatrix();
    matrix.makeScale(initscale, initscale, initscale);
    scale->setMatrix(matrix);

    // attach nodes to the mainNode
    geo->addDrawable(conedraw);

    //geo->addDrawable(BulletLabel);
    scale->addChild(geo);
    //scale->addChild(billText);
    pos->addChild(scale);
    node->addChild(pos);
    mySensor = new BulletSensor(this, geo);
}

void BulletProbe::setVisible(bool vis)
{
    if (vis)
    {
        geo->setNodeMask(~0);
        billText->setNodeMask(~0);
    }
    else
    {
        geo->setNodeMask(0);
        billText->setNodeMask(0);
    }
}

int BulletProbe::getID()
{
    return id;
}

string BulletProbe::getHost()
{
    return hname;
}

void BulletProbe::setBaseSize(float basescale)
{
    basevalue = basescale;
}

void BulletProbe::HSBtoRGB(float h, float s, float v, float *r, float *g, float *b)
{
    float f, p, q, t;
    int i;

    // Convert hue:
    if (h == 1.0f)
        h = 0.0f;
    h *= 360.0f;

    if (s == 0.0f) // grayscale value?
    {
        *r = v;
        *g = v;
        *b = v;
    }
    else
    {
        h /= 60.0;
        i = int(h);
        f = h - i;
        p = v * (1.0f - s);
        q = v * (1.0f - (s * f));
        t = v * (1.0f - (s * (1.0f - f)));
        switch (i)
        {
        case 0:
            *r = v;
            *g = t;
            *b = p;
            break;
        case 1:
            *r = q;
            *g = v;
            *b = p;
            break;
        case 2:
            *r = p;
            *g = v;
            *b = t;
            break;
        case 3:
            *r = p;
            *g = q;
            *b = v;
            break;
        case 4:
            *r = t;
            *g = p;
            *b = v;
            break;
        case 5:
            *r = v;
            *g = p;
            *b = q;
            break;
        }
    }
}

void BulletProbe::setColor(float hue)
{
    float r, g, b;
    _hue = hue;
    HSBtoRGB(_hue, 1, 1, &r, &g, &b);
    osg::Vec4 color(r, g, b, 1.0);
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, color);
}

float BulletProbe::getColor()
{
    return _hue;
}

void BulletProbe::setSelectedHost(const char *host)
{
    _selectedhname.clear();
    if (host != NULL)
    {
        _selectedhname.append(host);
    }
}

void BulletProbe::setAmbient(osg::Vec4 spec)
{
    mat->setAmbient(osg::Material::FRONT_AND_BACK, spec);
}

/////// NEED TO COMPLETE ///////
BulletProbe::~BulletProbe()
{
    pos->getParent(0)->removeChild(pos);
    delete mySensor;
}

void BulletProbe::setMat(osg::Matrix &mat)
{
    pos->setMatrix(mat);
}

void BulletProbe::setScale(float scaleset)
{
    osg::Matrix mat;
    mat.makeScale(scaleset, scaleset, scaleset);
    scale->setMatrix(mat);
    _scaleVal = scaleset;
}

float BulletProbe::getScale()
{
    return _scaleVal;
}

osg::Matrix BulletProbe::getPos()
{
    return pos->getMatrix();
}

// check if the Bullet is available for selection
bool BulletProbe::isSelectable(string shost)
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

// check if the Bullet matches
bool BulletProbe::matches(string shost)
{
    return (_selectedhname == shost);
}

/*float BulletProbe::getDist(osg::Vec3 &a)
{
   osg::Vec3 b,diff;
   osg::Matrix mat;
   pos->getMatrix(mat);
   mat.getRow(3,b);
   diff = a - b;
   return diff.dot(diff);
}*/

void Bullet::preFrame()
{
    sensorList.update();

    static osg::Matrix startPos;
    static osg::Matrix invStartHand;

    BulletMessage mm;

    if (!interactionA->isRunning() && !interactionC->isRunning())
    {
        //check if the object was previously moving
        if (currentBullet)
        {
            if (moving && !BulletHandle->isVisible())
            {
                // send message giving up control of the Bullet
                mm.id = currentBullet->getID();
                setMMString(mm.host, currentBullet->getHost());
                setMMString(mm.selecthost, getMyHost());
                mm.token = 5;
                cover->sendMessage(this,
                                   coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::Bullet1,
                                   sizeof(BulletMessage),
                                   &mm);
            }
        }
        moving = false;
    }
    if ((currentBullet) || (menuSelected))
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
    if ((currentBullet))
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
        }
    }

    if (interactionA->wasStarted())
    {
        if (currentBullet) // start moving the current Bullet
        {
            invStartHand.invert(cover->getPointerMat());
            osg::Matrix tmp;
            currentBullet->getMat(tmp);
            startPos = tmp * cover->getBaseMat();
            //check if another Bullet is being selected to move (releases lock on old Bullet)
            if (currentBullet != previousBullet)
                BulletHandle->setVisible(false);

            // send move Message place lock selected Bullet
            if (currentBullet->isSelectable(getMyHost()))
            {
                mm.id = currentBullet->getID();
                setMMString(mm.host, currentBullet->getHost());
                setMMString(mm.selecthost, getMyHost());
                mm.token = 0;
                currentBullet->getMat(mm.mat);
                cover->sendMessage(this,
                                   coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::Bullet0,
                                   sizeof(BulletMessage),
                                   &mm);
            }
            moving = true;
        }
        else if (menuSelected) // create a new Bullet
        {
            mm.id = getLowestUnusedBulletID();
            setMMString(mm.host, getMyHost());

            osg::Matrix trans;
            trans.makeTranslate(0, 800, 0);
            mm.mat = trans * cover->getPointerMat() * cover->getInvBaseMat();
            mm.token = 0;
            cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Bullet0,
                               sizeof(BulletMessage),
                               &mm);
            BulletMenuCheckbox->setState(false);
            menuSelected = 0;
        }
    }
    if (interactionC->wasStarted())
    {
        if (currentBullet)
        {
            // check if the Bullet is selectable
            if (currentBullet->isSelectable(getMyHost()))
            {
                // check if selecting another Bullet
                if (currentBullet != previousBullet)
                {
                    mm.id = currentBullet->getID();
                    setMMString(mm.host, currentBullet->getHost());
                    setMMString(mm.selecthost, getMyHost());
                    mm.token = 2;
                    cover->sendMessage(this,
                                       coVRPluginSupport::TO_SAME,
                                       PluginMessageTypes::Bullet1,
                                       sizeof(BulletMessage),
                                       &mm);
                }
                else // close menu
                {
                    BulletHandle->setVisible(false);
                }
            }
        }
    }
    if (interactionA->isRunning())
    {
        if (currentBullet)
        {
            if (!moving) // start moving the current Bullet
            {
                invStartHand.invert(cover->getPointerMat());
                osg::Matrix tmp;
                currentBullet->getMat(tmp);
                startPos = tmp * cover->getBaseMat();
                moving = true;
            }
            osg::Matrix dMat = invStartHand * cover->getPointerMat();
            osg::Matrix current;
            osg::Matrix tmp;
            tmp = startPos * dMat;
            current = tmp * cover->getInvBaseMat();

            // send move Message
            if (currentBullet->isSelectable(getMyHost()))
            {
                mm.id = currentBullet->getID();
                setMMString(mm.host, currentBullet->getHost());
                setMMString(mm.selecthost, getMyHost());
                mm.token = 0;
                //currentBullet->getMat(mm.mat);
                mm.mat = current;
                cover->sendMessage(this,
                                   coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::Bullet0,
                                   sizeof(BulletMessage),
                                   &mm);
            }
        }
    }

    // check if someone closes the Bullet panel
    if ((!BulletHandle->isVisible()) && (previousBullet != NULL))
    {
        // send message giving up control of the Bullet
        if (previousBullet->isSelectable(getMyHost()))
        {
            mm.id = previousBullet->getID();
            setMMString(mm.host, previousBullet->getHost());
            setMMString(mm.selecthost, getMyHost());
            mm.token = 5;
            cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Bullet1,
                               sizeof(BulletMessage),
                               &mm);
        }
    }
}

/** Use this when creating new Bullets to ensure that the ID list is always continuous.
*/
int Bullet::getLowestUnusedBulletID()
{
    int id = 0;

    while (isIDInUse(id))
    {
        ++id;
    }
    return id;
}

bool Bullet::isIDInUse(int id)
{
    vector<BulletProbe *>::iterator it = bullet.begin();
    for (; it < bullet.end(); it++)
    {
        if ((*it)->getID() == id)
            return true;
    }
    return false;
}

void BulletProbe::getMat(osg::Matrix &m)
{
    m = pos->getMatrix();
}

void
Bullet::setCurrentBullet(BulletProbe *m)
{
    if ((currentBullet == NULL) || (m == NULL))
        currentBullet = m;
}

void
Bullet::removeMenuEntry()
{
    delete BulletMenuItem;
    delete BulletMenu;
    delete BulletMenuCheckbox;
    delete hideMenuCheckbox;
    delete scaleMenuPoti;
    delete deleteAllButton;
    delete SaveButton;
    delete BulletHandle;
    delete BulletFrame;
    delete BulletPanel;
    delete BulletLabel;
    delete BulletDeleteButton;
    delete BulletPlayButton;
    delete BulletRecordButton;
    delete colorPoti;
}

void Bullet::setScaleAll(float scaleset)
{
    vector<BulletProbe *>::iterator it = bullet.begin();
    for (; it < bullet.end(); it++)
    {
        (*it)->setScale(scaleset);
    }
}

// adjust individual maker colors
void Bullet::potiValueChanged(float, float newvalue, coValuePoti *, int)
{
    if (previousBullet != NULL)
    {
        BulletMessage mm;
        setMMString(mm.host, previousBullet->getHost());
        setMMString(mm.selecthost, getMyHost());
        mm.id = previousBullet->getID();
        mm.token = 3;
        mm.color = newvalue;
        cover->sendMessage(this,
                           coVRPluginSupport::TO_SAME,
                           PluginMessageTypes::Bullet1,
                           sizeof(BulletMessage),
                           &mm);
    }
}

void Bullet::menuEvent(coMenuItem *item)
{
    if (item == BulletMenuCheckbox)
    {
        menuSelected = BulletMenuCheckbox->getState();
    }
    else if (item == hideMenuCheckbox)
    {
        setVisible(hideMenuCheckbox->getState());
    }
    else if (item == scaleMenuPoti)
    {
        setScaleAll(scaleMenuPoti->getValue());
    }
    else if (item == deleteAllButton)
    {
        deleteAllBullets();
    }
    else if (item == SaveButton)
    {
        save();
    }
}

/*
 * Delete all Bullets
 *
 */
void Bullet::deleteAllBullets()
{
    //check if plugin exists
    if (plugin == NULL)
    {
        fprintf(stderr, "Bullet constuctor never called\n");
        return;
    }

    BulletMessage mm;
    mm.token = 4;
    cover->sendMessage(this,
                       coVRPluginSupport::TO_SAME,
                       PluginMessageTypes::Bullet1,
                       sizeof(BulletMessage),
                       &mm);
}

// check for button events
void Bullet::buttonEvent(coButton *cobutton)
{
    BulletMessage mm;

    if (cobutton == BulletRecordButton)
    {
    }
    else if (cobutton == BulletPlayButton)
    {
    }
    else if (cobutton == BulletDeleteButton)
    {
        if (previousBullet != NULL)
        {
            mm.id = previousBullet->getID();
            setMMString(mm.host, previousBullet->getHost());
            setMMString(mm.selecthost, getMyHost());
            mm.token = 1;
            cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Bullet1,
                               sizeof(BulletMessage),
                               &mm);
            setCurrentBullet(NULL); //currentBullet=NULL;
        }
    }
}

void Bullet::message(int toWhom, int type, int len, const void *buf)
{

    (void)len;

    if (type != PluginMessageTypes::Bullet0 && type != PluginMessageTypes::Bullet1)
        return;

    BulletMessage *mm = (BulletMessage *)buf;
    switch (mm->token)
    {
    case 0: // MOVE/ADD
    {
        BulletProbe *curr = NULL;
        vector<BulletProbe *>::iterator it = bullet.begin();
        for (; it < bullet.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    // need to update Bullet
                    (*it)->setSelectedHost(mm->selecthost);
                    (*it)->setAmbient(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
                    curr = *it;
                    break;
                }
            }
        }

        if (curr == NULL)
        {
            curr = new BulletProbe(mm->id, mm->host, mainNode.get(), scaleMenuPoti->getValue());
            bullet.push_back(curr);
        }

        curr->setMat(mm->mat);
        break;
    }

    case 1: // Remove a Bullet
    {
        vector<BulletProbe *>::iterator it = bullet.begin();
        for (; it < bullet.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    if (getMyHost() == mm->selecthost)
                    {
                        previousBullet = NULL;
                        // close BulletHandle
                        BulletLabel->setString(BULLET);
                        BulletHandle->setVisible(false);
                    }
                    delete *it;
                    bullet.erase(it);
                    break;
                }
            }
        }
        break;
    }

    case 2: // Bullet selected
    {

        vector<BulletProbe *>::iterator it = bullet.begin();
        for (; it < bullet.end(); it++)
        {

            // check for newly selected Bullet
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                // check for match and selectable
                if ((*it)->isSelectable(mm->selecthost))
                {
                    // set selected hostname
                    (*it)->setSelectedHost(mm->selecthost);
                    colorPoti->setValue((*it)->getColor());
                    (*it)->setAmbient(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));

                    // make panel visible only for user of the Bullet
                    if (getMyHost() == mm->selecthost)
                    {
                        setCurrentBullet(*it); //currentBullet = *it;
                        // convert int to string for the label
                        std::ostringstream label;
                        if (label << ((*it)->getID() + 1))
                        {
                            BulletLabel->setString(BULLET + label.str());
                        }
                        previousBullet = *it;
                        BulletHandle->setVisible(true);
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

    case 3: // Change Bullet color
    {
        vector<BulletProbe *>::iterator it = bullet.begin();
        for (; it < bullet.end(); it++)
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

    case 4: // Deletes all Bullets
    {
        vector<BulletProbe *>::iterator it;
        int size = bullet.size();
        for (int i = 0; i < size; i++)
        {

            it = bullet.begin();
            /*cout << "TransForm: ";
        for(int i = 0;i < 16;i++) {
         cout << ((*it)->getPos().ptr())[i] << ",";
          }
        cout << endl <<"Color " << (*it)->getColor() << endl << endl;
        */
            delete *it;
            bullet.erase(it);
            // close BulletHandle
            BulletLabel->setString(BULLET);
            BulletHandle->setVisible(false);
        }
        previousBullet = NULL;
        setCurrentBullet(NULL);
        break;
    }

    case 5: // Release current lock on a Bullet
    {
        vector<BulletProbe *>::iterator it = bullet.begin();
        for (; it < bullet.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    // reset color of currently selected Bullet
                    (*it)->setSelectedHost(NULL);
                    (*it)->setAmbient(osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
                    //check if selectable should be reset
                    if ((getMyHost() == mm->selecthost) && (*it == previousBullet))
                    {
                        BulletLabel->setString(BULLET);
                        BulletHandle->setVisible(false);
                        previousBullet = NULL;
                    }
                    break;
                }
            }
        }
        break;
    }
    }
}

int Bullet::sloadBullet(const char *filename, osg::Group *loadParent, const char *)
{

    instance()->loadBullet(filename, loadParent);
    return 0;
}

int Bullet::loadBullet(const char *filename, osg::Group *loadParent)
{
	XMLCh *t1 = NULL;
    fileName = filename;
    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(filename);
    }
    catch (...)
    {
        cerr << "error parsing Bullet xml file" << endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    xercesc::DOMElement *rootElement = NULL;
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    if (rootElement)
    {
        xercesc::DOMNodeList *nodeList = rootElement->getChildNodes();
        for (size_t i = 0; i < nodeList->getLength(); ++i)
        {
            xercesc::DOMElement *node = dynamic_cast<xercesc::DOMElement *>(nodeList->item(i));
            if (!node)
                continue;
            char *sizes = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("size"))); xercesc::XMLString::release(&t1);
            char *hostName = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("hostName"))); xercesc::XMLString::release(&t1);
            char *hS = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("h"))); xercesc::XMLString::release(&t1);
            char *pS = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("p"))); xercesc::XMLString::release(&t1);
            char *rS = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("r"))); xercesc::XMLString::release(&t1);
            char *xS = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("x"))); xercesc::XMLString::release(&t1);
            char *yS = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("y"))); xercesc::XMLString::release(&t1);
            char *zS = xercesc::XMLString::transcode(node->getAttribute(t1 = xercesc::XMLString::transcode("z"))); xercesc::XMLString::release(&t1);
            float size;
            sscanf(sizes, "%f", &size);

            float h, p, r, x, y, z;
            sscanf(hS, "%f", &h);
            sscanf(pS, "%f", &p);
            sscanf(rS, "%f", &r);
            sscanf(xS, "%f", &x);
            sscanf(yS, "%f", &y);
            sscanf(zS, "%f", &z);
            BulletProbe *bp = new BulletProbe(i, hostName, mainNode.get(), size);
            osg::Matrix rotMat, transMat;
            MAKE_EULER_MAT(rotMat, h, p, r)
            transMat.makeTranslate(x, y, z);
            osg::Matrix rtMat = rotMat * transMat;
            bp->setMat(rtMat);
            bullet.push_back(bp);
			xercesc::XMLString::release(&sizes);
			xercesc::XMLString::release(&hostName);
			xercesc::XMLString::release(&hS);
			xercesc::XMLString::release(&pS);
			xercesc::XMLString::release(&rS);
			xercesc::XMLString::release(&xS);
			xercesc::XMLString::release(&yS);
			xercesc::XMLString::release(&zS);
        }
    }

    return 0;
}

void Bullet::save()
{
	XMLCh *t1 = NULL;
	XMLCh *t2 = NULL;
    xercesc::DOMDocument *document = impl->createDocument(0, t1 = xercesc::XMLString::transcode("BulletProbes"), 0); xercesc::XMLString::release(&t1);

    xercesc::DOMElement *rootElement = document->getDocumentElement();

    for (std::vector<BulletProbe *>::iterator bp = bullet.begin(); bp != bullet.end(); ++bp)
    {
        xercesc::DOMElement *bpElement = document->createElement(t1 = xercesc::XMLString::transcode("BulletProbe")); xercesc::XMLString::release(&t1);

        char number[100];
        sprintf(number, "%f", (*bp)->getScale());
        bpElement->setAttribute(t1 = xercesc::XMLString::transcode("size"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        bpElement->setAttribute(t1 = xercesc::XMLString::transcode("hostName"), t2 = xercesc::XMLString::transcode((*bp)->getHost().c_str())); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);

        float x, y, z, h, p, r;
        osg::Matrix m;
        (*bp)->getMat(m);
        coCoord coord;
        coord = m;
        h = coord.hpr[0];
        p = coord.hpr[1];
        r = coord.hpr[2];
        x = coord.xyz[0];
        y = coord.xyz[1];
        z = coord.xyz[2];

        sprintf(number, "%f", h);
        bpElement->setAttribute(t1 = xercesc::XMLString::transcode("h"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        sprintf(number, "%f", p);
        bpElement->setAttribute(t1 = xercesc::XMLString::transcode("p"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        sprintf(number, "%f", r);
        bpElement->setAttribute(t1 = xercesc::XMLString::transcode("r"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        sprintf(number, "%f", x);
        bpElement->setAttribute(t1 = xercesc::XMLString::transcode("x"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        sprintf(number, "%f", y);
        bpElement->setAttribute(t1 = xercesc::XMLString::transcode("y"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        sprintf(number, "%f", z);
        bpElement->setAttribute(t1 = xercesc::XMLString::transcode("z"), t2 = xercesc::XMLString::transcode(number)); xercesc::XMLString::release(&t1); xercesc::XMLString::release(&t2);
        rootElement->appendChild(bpElement);
    }

#if XERCES_VERSION_MAJOR < 3
    xercesc::DOMWriter *writer = impl->createDOMWriter();
    // set the format-pretty-print feature
    if (writer->canSetFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
        writer->setFeature(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    xercesc::XMLFormatTarget *xmlTarget = new xercesc::LocalFileFormatTarget(fileName.c_str());
    bool written = writer->writeNode(xmlTarget, *rootElement);
    if (!written)
        fprintf(stderr, "Bullet::save info: Could not open file for writing !\n");

    delete writer;
    delete xmlTarget;
#else

    xercesc::DOMLSSerializer *writer = ((xercesc::DOMImplementationLS *)impl)->createLSSerializer();

    //writer->setNewLine(xercesc::XMLString::transcode("\n\r") );
    // Make the output more human readable by inserting line feeds.
    if (writer->getDomConfig()->canSetParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true))
        writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTFormatPrettyPrint, true);
    writer->getDomConfig()->setParameter(xercesc::XMLUni::fgDOMWRTXercesPrettyPrint, false);

    //xercesc::DOMConfiguration* dc = writer->getDomConfig();
    //dc->setParameter(xercesc::XMLUni::fgDOMErrorHandler,errorHandler);
    //dc->setParameter(xercesc::XMLUni::fgDOMWRTDiscardDefaultContent,true);

    xercesc::DOMLSOutput *theOutput = ((xercesc::DOMImplementationLS *)impl)->createLSOutput();
    theOutput->setEncoding(t1 = xercesc::XMLString::transcode("utf8")); xercesc::XMLString::release(&t1);

    bool written = writer->writeToURI(rootElement, t1 = xercesc::XMLString::transcode(fileName.c_str())); xercesc::XMLString::release(&t1);
    if (!written)
        fprintf(stderr, "Bullet::save info: Could not open file for writing %s!\n", fileName.c_str());
    delete writer;

#endif
    delete document;
}

int Bullet::unloadBullet(const char *filename, const char *)
{
    (void)filename;
    return 0;
}

COVERPLUGIN(Bullet)
