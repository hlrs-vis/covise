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
#include <virvo/vvtoolshed.h>
#include <cover/VRVruiRenderInterface.h>

#include <PluginUtil/PluginMessageTypes.h>

#include "Marker.h"

using namespace osg;

Marker *Marker::plugin = NULL;
float Mark::basevalue = 10;
const string MARKER = "Marker: ";
vrml::Player *Marker::player = NULL;

MarkerSensor::MarkerSensor(Mark *m, osg::Node *n)
    : coPickSensor(n)
{
    myMark = m;
    setThreshold(50);
    //threshold = 50*50;
}

MarkerSensor::~MarkerSensor()
{
    if (active)
        disactivate();
}

void MarkerSensor::activate()
{
    Marker::plugin->setCurrentMarker(myMark);
    active = 1;
    //myMark->setIcon(1);
}

void MarkerSensor::disactivate()
{
    Marker::plugin->setCurrentMarker(NULL);
    active = 0;
    //myMark->setIcon(0);
}

Marker::Marker()
: coVRPlugin(COVER_PLUGIN_NAME)
{
}

bool Marker::init()
{

    if (Marker::plugin != NULL)
        return false;

    Marker::plugin = this;

    //  cubeToolTip = NULL;
    menuSelected = false;
    //selectedMarkerId = -1;

    // default button size
    float buttonSize[] = { 80, 15 };

    // marker menu positions
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

    // create new marker menu
    markerMenuItem = new coSubMenuItem("Marker");

    // create row menu to add items
    markerMenu = new coRowMenu("Marker Options");
    markerMenuItem->setMenu(markerMenu);

    // create a new marker
    markerMenuCheckbox = new coCheckboxMenuItem("NewMarker", false);
    markerMenu->add(markerMenuCheckbox);
    markerMenuCheckbox->setMenuListener(this);

    // hide all markers
    hideMenuCheckbox = new coCheckboxMenuItem("Display Markers", true);
    markerMenu->add(hideMenuCheckbox);
    hideMenuCheckbox->setMenuListener(this);

    // scale all markers
    scaleMenuPoti = new coPotiMenuItem("Scale Markers", 0, 10, 1);
    markerMenu->add(scaleMenuPoti);
    scaleMenuPoti->setMenuListener(this);

    // add delete all button
    deleteAllButton = new coButtonMenuItem("Delete All");
    markerMenu->add(deleteAllButton);
    deleteAllButton->setMenuListener(this);

    // get the root node for COVISE objects
    mainNode = new osg::Group;
    mainNode->setName("MarkerMainNode");
    cover->getObjectsRoot()->addChild((osg::Node *)mainNode.get());
    setScaleAll(scaleMenuPoti->getValue()); // set the Main node

    //create a pop up menu for marker interaction
    markerHandle = new coPopupHandle("MarkerControl");
    markerFrame = new coFrame("UI/Frame");
    markerPanel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    markerLabel = new coLabel;
    markerLabel->setString(MARKER);
    markerLabel->setPos(x[0], y[0], z + 1);
    markerLabel->setFontSize(labelSize);
    markerHandle->addElement(markerFrame);
    markerDeleteButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Marker/delete"), this);
    markerPlayButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Marker/play"), this);
    markerRecordButton = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Marker/record"), this);
    colorPoti = new coValuePoti("Color", this, "Volume/valuepoti-bg");
    markerPlayButton->setSize(40);
    markerRecordButton->setSize(40);
    markerDeleteButton->setSize(40);
    markerPlayButton->setPos(x[0], y[1], z);
    markerPlayButton->setVisible(false);
    markerRecordButton->setVisible(false);
    markerRecordButton->setPos(x[0], y[2], z);
    markerDeleteButton->setPos(x[0], y[3], z);
    colorPoti->setPos(x[1], y[2], z);
    colorPoti->setMin(0);
    colorPoti->setMax(1);
    markerPanel->addElement(markerLabel);
    markerPanel->addElement(markerPlayButton);
    markerPanel->addElement(markerRecordButton);
    markerPanel->addElement(markerDeleteButton);
    markerPanel->addElement(colorPoti);
    markerPanel->setScale(5);
    markerPanel->resize();
    markerFrame->addElement(markerPanel);
    markerHandle->setVisible(false);

    currentMarker = NULL;
    previousMarker = NULL;

    // add marker menu to the main menu
    cover->getMenu()->add(markerMenuItem);

    moving = false;
    interactionA = new coNavInteraction(coInteraction::ButtonA, "MarkerPlacement", coInteraction::Medium);
    interactionC = new coNavInteraction(coInteraction::ButtonC, "MarkerPlacement", coInteraction::Medium);

    return true;
}

string Marker::getMyHost()
{
    return myHost;
}

void Marker::setMMString(char *dst, string src)
{
    assert(src.size() < 20);
    strcpy(dst, src.c_str());
}

/*
 * Set pre determined markers if structure exists
 *
 */
/*void Marker::setMarkers(string structure)
{
  MarkerMessage mm;
  mm.id = maxid;
  setMMString(mm.host, getMyHost());
  setMMString(mm.filename, structure);
  mm.token = 6;
  cover->sendMessage(this,
    coVRPluginSupport::TO_SAME,
    0,
    sizeof(MarkerMessage),
    &mm);
}*/

// this is called if the plugin is removed at runtime
Marker::~Marker()
{
    vector<Mark *>::iterator it = marker.begin();
    for (; it < marker.end(); it++)
    {
        delete (*it);
    }
    cover->getObjectsRoot()->removeChild((osg::Node *)mainNode.get());
    removeMenuEntry();
    delete interactionA;
    delete interactionC;
}

// hide all markers (make transparent)
void Marker::setVisible(bool vis)
{
    vector<Mark *>::iterator it = marker.begin();
    for (; it < marker.end(); it++)
    {
        (*it)->setVisible(vis);
    }
}

Mark::Mark(int i, string host, osg::Group *node, float initscale)
{
    id = i;
    pos = new osg::MatrixTransform();
    _scaleVal = 1.0f;

    // create a billboard to attach the text to
    billText = new osg::Billboard;
    billText->setNormal(osg::Vec3(0.0f, 0.0f, 1.0f));
    billText->setAxis(osg::Vec3(0.0f, 0.0f, 1.0f));
    billText->setMode(osg::Billboard::AXIAL_ROT);

    // create label for the the marker
    osgText::Text *markerLabel = new osgText::Text;
    markerLabel->setCharacterSize(basevalue * cover->getScale());

    // set host name
    hname = host;

    // convert int to string for the label
    std::ostringstream label;
    if (label << id)
    {
        markerLabel->setText(label.str(), osgText::String::ENCODING_UTF8);
    }
    markerLabel->setAxisAlignment(osgText::Text::XZ_PLANE);

    // Set the text to render with alignment anchor
    markerLabel->setDrawMode(osgText::Text::TEXT);
    markerLabel->setAlignment(osgText::Text::CENTER_BOTTOM);
    //markerLabel->setPosition( osg::Vec3(0.0f, 0.0f,-(basevalue * cover->getScale())));
    markerLabel->setColor(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));

    billText->addDrawable(markerLabel);
    billText->setPosition(0, osg::Vec3(0.0f, 0.0f, -(basevalue * cover->getScale())));

    geo = new osg::Geode;
    scale = new osg::MatrixTransform;
    //scale->addChild(markerGeode);
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

    //geo->addDrawable(markerLabel);
    scale->addChild(geo);
    //scale->addChild(billText);
    pos->addChild(scale);
    node->addChild(pos);
    mySensor = new MarkerSensor(this, geo);
    Marker::plugin->sensorList.append(mySensor);
}

void Mark::setVisible(bool vis)
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

int Mark::getID()
{
    return id;
}

string Mark::getHost()
{
    return hname;
}

void Mark::setBaseSize(float basescale)
{
    basevalue = basescale;
}

void Mark::setColor(float hue)
{
    float r, g, b;
    _hue = hue;
    vvToolshed::HSBtoRGB(_hue, 1, 1, &r, &g, &b);
    osg::Vec4 color(r, g, b, 1.0);
    mat->setDiffuse(osg::Material::FRONT_AND_BACK, color);
}

float Mark::getColor()
{
    return _hue;
}

void Mark::setSelectedHost(const char *host)
{
    _selectedhname.clear();
    if (host != NULL)
    {
        _selectedhname.append(host);
    }
}

void Mark::setAmbient(osg::Vec4 spec)
{
    mat->setAmbient(osg::Material::FRONT_AND_BACK, spec);
}

/////// NEED TO COMPLETE ///////
Mark::~Mark()
{
    pos->getParent(0)->removeChild(pos);
    if (Marker::plugin->sensorList.find(mySensor))
        Marker::plugin->sensorList.remove();
    delete mySensor;
}

void Mark::setMat(osg::Matrix &mat)
{
    pos->setMatrix(mat);
}

void Mark::setScale(float scaleset)
{
    osg::Matrix mat;
    mat.makeScale(scaleset, scaleset, scaleset);
    scale->setMatrix(mat);
    _scaleVal = scaleset;
}

float Mark::getScale()
{
    return _scaleVal;
}

osg::Matrix Mark::getPos()
{
    return pos->getMatrix();
}

// check if the marker is available for selection
bool Mark::isSelectable(string shost)
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

// check if the marker matches
bool Mark::matches(string shost)
{
    return (_selectedhname == shost);
}

/*float Mark::getDist(osg::Vec3 &a)
{
   osg::Vec3 b,diff;
   osg::Matrix mat;
   pos->getMatrix(mat);
   mat.getRow(3,b);
   diff = a - b;
   return diff.dot(diff);
}*/

void Marker::preFrame()
{
    sensorList.update();

    static osg::Matrix startPos;
    static osg::Matrix invStartHand;

    MarkerMessage mm;

    if (!interactionA->isRunning() && !interactionC->isRunning())
    {
        //check if the object was previously moving
        if (currentMarker)
        {
            if (moving && !markerHandle->isVisible())
            {
                // send message giving up control of the marker
                mm.id = currentMarker->getID();
                setMMString(mm.host, currentMarker->getHost());
                setMMString(mm.selecthost, getMyHost());
                mm.token = 5;
                cover->sendMessage(this,
                                   coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::Marker1,
                                   sizeof(MarkerMessage),
                                   &mm);
            }
        }
        moving = false;
    }
    if ((currentMarker) || (menuSelected))
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
    if ((currentMarker))
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
        if (currentMarker) // start moving the current Marker
        {
            invStartHand.invert(cover->getPointerMat());
            osg::Matrix tmp;
            currentMarker->getMat(tmp);
            startPos = tmp * cover->getBaseMat();
            //check if another marker is being selected to move (releases lock on old marker)
            if (currentMarker != previousMarker)
                markerHandle->setVisible(false);

            // send move Message place lock selected marker
            if (currentMarker->isSelectable(getMyHost()))
            {
                mm.id = currentMarker->getID();
                setMMString(mm.host, currentMarker->getHost());
                setMMString(mm.selecthost, getMyHost());
                mm.token = 0;
                currentMarker->getMat(mm.mat);
                cover->sendMessage(this,
                                   coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::Marker0,
                                   sizeof(MarkerMessage),
                                   &mm);
            }
            moving = true;
        }
        else if (menuSelected) // create a new Marker
        {
            mm.id = getLowestUnusedMarkerID();
            setMMString(mm.host, getMyHost());

            osg::Matrix trans;
            trans.makeTranslate(0, 800, 0);
            mm.mat = trans * cover->getPointerMat() * cover->getInvBaseMat();
            mm.token = 0;
            cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Marker0,
                               sizeof(MarkerMessage),
                               &mm);
            markerMenuCheckbox->setState(false);
            menuSelected = 0;
        }
    }
    if (interactionC->wasStarted())
    {
        if (currentMarker)
        {
            // check if the marker is selectable
            if (currentMarker->isSelectable(getMyHost()))
            {
                // check if selecting another marker
                if (currentMarker != previousMarker)
                {
                    mm.id = currentMarker->getID();
                    setMMString(mm.host, currentMarker->getHost());
                    setMMString(mm.selecthost, getMyHost());
                    mm.token = 2;
                    cover->sendMessage(this,
                                       coVRPluginSupport::TO_SAME,
                                       PluginMessageTypes::Marker1,
                                       sizeof(MarkerMessage),
                                       &mm);
                }
                else // close menu
                {
                    markerHandle->setVisible(false);
                }
            }
        }
    }
    if (interactionA->isRunning())
    {
        if (currentMarker)
        {
            if (!moving) // start moving the current Marker
            {
                invStartHand.invert(cover->getPointerMat());
                osg::Matrix tmp;
                currentMarker->getMat(tmp);
                startPos = tmp * cover->getBaseMat();
                moving = true;
            }
            osg::Matrix dMat = invStartHand * cover->getPointerMat();
            osg::Matrix current;
            osg::Matrix tmp;
            tmp = startPos * dMat;
            current = tmp * cover->getInvBaseMat();

            // send move Message
            if (currentMarker->isSelectable(getMyHost()))
            {
                mm.id = currentMarker->getID();
                setMMString(mm.host, currentMarker->getHost());
                setMMString(mm.selecthost, getMyHost());
                mm.token = 0;
                //currentMarker->getMat(mm.mat);
                mm.mat = current;
                cover->sendMessage(this,
                                   coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::Marker0,
                                   sizeof(MarkerMessage),
                                   &mm);
            }
        }
    }

    // check if someone closes the marker panel
    if ((!markerHandle->isVisible()) && (previousMarker != NULL))
    {
        // send message giving up control of the marker
        if (previousMarker->isSelectable(getMyHost()))
        {
            mm.id = previousMarker->getID();
            setMMString(mm.host, previousMarker->getHost());
            setMMString(mm.selecthost, getMyHost());
            mm.token = 5;
            cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Marker1,
                               sizeof(MarkerMessage),
                               &mm);
        }
    }
}

/** Use this when creating new markers to ensure that the ID list is always continuous.
*/
int Marker::getLowestUnusedMarkerID()
{
    int id = 0;

    while (isIDInUse(id))
    {
        ++id;
    }
    return id;
}

bool Marker::isIDInUse(int id)
{
    vector<Mark *>::iterator it = marker.begin();
    for (; it < marker.end(); it++)
    {
        if ((*it)->getID() == id)
            return true;
    }
    return false;
}

void Mark::getMat(osg::Matrix &m)
{
    m = pos->getMatrix();
}

void
Marker::setCurrentMarker(Mark *m)
{
    if ((currentMarker == NULL) || (m == NULL))
        currentMarker = m;
}

void
Marker::removeMenuEntry()
{
    delete markerMenuItem;
    delete markerMenu;
    delete markerMenuCheckbox;
    delete hideMenuCheckbox;
    delete scaleMenuPoti;
    delete deleteAllButton;
    delete markerHandle;
    delete markerFrame;
    delete markerPanel;
    delete markerLabel;
    delete markerDeleteButton;
    delete markerPlayButton;
    delete markerRecordButton;
    delete colorPoti;
}

void Marker::setScaleAll(float scaleset)
{
    vector<Mark *>::iterator it = marker.begin();
    for (; it < marker.end(); it++)
    {
        (*it)->setScale(scaleset);
    }
}

// adjust individual maker colors
void Marker::potiValueChanged(float, float newvalue, coValuePoti *, int)
{
    if (previousMarker != NULL)
    {
        MarkerMessage mm;
        setMMString(mm.host, previousMarker->getHost());
        setMMString(mm.selecthost, getMyHost());
        mm.id = previousMarker->getID();
        mm.token = 3;
        mm.color = newvalue;
        cover->sendMessage(this,
                           coVRPluginSupport::TO_SAME,
                           PluginMessageTypes::Marker1,
                           sizeof(MarkerMessage),
                           &mm);
    }
}

void Marker::menuEvent(coMenuItem *item)
{
    if (item == markerMenuCheckbox)
    {
        menuSelected = markerMenuCheckbox->getState();
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
        deleteAllMarkers();
    }
}

/*
 * Delete all markers
 *
 */
void Marker::deleteAllMarkers()
{
    //check if plugin exists
    if (plugin == NULL)
    {
        fprintf(stderr, "Marker constuctor never called\n");
        return;
    }

    MarkerMessage mm;
    mm.token = 4;
    cover->sendMessage(this,
                       coVRPluginSupport::TO_SAME,
                       PluginMessageTypes::Marker1,
                       sizeof(MarkerMessage),
                       &mm);
}

// check for button events
void Marker::buttonEvent(coButton *cobutton)
{
    MarkerMessage mm;

    if (cobutton == markerRecordButton)
    {
    }
    else if (cobutton == markerPlayButton)
    {
    }
    else if (cobutton == markerDeleteButton)
    {
        if (previousMarker != NULL)
        {
            mm.id = previousMarker->getID();
            setMMString(mm.host, previousMarker->getHost());
            setMMString(mm.selecthost, getMyHost());
            mm.token = 1;
            cover->sendMessage(this,
                               coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::Marker1,
                               sizeof(MarkerMessage),
                               &mm);
            setCurrentMarker(NULL); //currentMarker=NULL;
        }
    }
}

void Marker::message(int toWhom, int type, int len, const void *buf)
{

    (void)len;

    if (type != PluginMessageTypes::Marker0 && type != PluginMessageTypes::Marker1)
        return;

    MarkerMessage *mm = (MarkerMessage *)buf;
    switch (mm->token)
    {
    case 0: // MOVE/ADD
    {
        Mark *curr = NULL;
        vector<Mark *>::iterator it = marker.begin();
        for (; it < marker.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    // need to update marker
                    (*it)->setSelectedHost(mm->selecthost);
                    (*it)->setAmbient(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));
                    curr = *it;
                    break;
                }
            }
        }

        if (curr == NULL)
        {
            curr = new Mark(mm->id, mm->host, mainNode.get(), scaleMenuPoti->getValue());
            marker.push_back(curr);
        }

        curr->setMat(mm->mat);
        break;
    }

    case 1: // Remove a marker
    {
        vector<Mark *>::iterator it = marker.begin();
        for (; it < marker.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    if (getMyHost() == mm->selecthost)
                    {
                        previousMarker = NULL;
                        // close markerHandle
                        markerLabel->setString(MARKER);
                        markerHandle->setVisible(false);
                    }
                    delete *it;
                    marker.erase(it);
                    break;
                }
            }
        }
        break;
    }

    case 2: // marker selected
    {

        vector<Mark *>::iterator it = marker.begin();
        for (; it < marker.end(); it++)
        {

            // check for newly selected marker
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                // check for match and selectable
                if ((*it)->isSelectable(mm->selecthost))
                {
                    // set selected hostname
                    (*it)->setSelectedHost(mm->selecthost);
                    colorPoti->setValue((*it)->getColor());
                    (*it)->setAmbient(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));

                    // make panel visible only for user of the marker
                    if (getMyHost() == mm->selecthost)
                    {
                        setCurrentMarker(*it); //currentMarker = *it;
                        // convert int to string for the label
                        std::ostringstream label;
                        if (label << ((*it)->getID() + 1))
                        {
                            markerLabel->setString(MARKER + label.str());
                        }
                        previousMarker = *it;
                        markerHandle->setVisible(true);
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

    case 3: // Change marker color
    {
        vector<Mark *>::iterator it = marker.begin();
        for (; it < marker.end(); it++)
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

    case 4: // Deletes all Markers
    {
        vector<Mark *>::iterator it;
        int size = marker.size();
        for (int i = 0; i < size; i++)
        {

            it = marker.begin();
            /*cout << "TransForm: ";
        for(int i = 0;i < 16;i++) {
         cout << ((*it)->getPos().ptr())[i] << ",";
          }
        cout << endl <<"Color " << (*it)->getColor() << endl << endl;
        */
            delete *it;
            marker.erase(it);
            // close markerHandle
            markerLabel->setString(MARKER);
            markerHandle->setVisible(false);
        }
        previousMarker = NULL;
        setCurrentMarker(NULL);
        break;
    }

    case 5: // Release current lock on a marker
    {
        vector<Mark *>::iterator it = marker.begin();
        for (; it < marker.end(); it++)
        {
            if (((*it)->getID() == mm->id) && ((*it)->getHost() == string(mm->host)))
            {
                if ((*it)->isSelectable(mm->selecthost))
                {
                    // reset color of currently selected marker
                    (*it)->setSelectedHost(NULL);
                    (*it)->setAmbient(osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
                    //check if selectable should be reset
                    if ((getMyHost() == mm->selecthost) && (*it == previousMarker))
                    {
                        markerLabel->setString(MARKER);
                        markerHandle->setVisible(false);
                        previousMarker = NULL;
                    }
                    break;
                }
            }
        }
        break;
    }
    }
}

COVERPLUGIN(Marker)
