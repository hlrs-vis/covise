/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <cover/coIntersection.h>
#include <cover/coVRMSController.h>
#include <cover/coVRCommunication.h>

#include <config/CoviseConfig.h>

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
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <cover/coBillboard.h>
#include <cover/VRVruiRenderInterface.h>

#include <PluginUtil/PluginMessageTypes.h>

#include "AnnotationPlugin.h"

using namespace osg;

AnnotationPlugin *AnnotationPlugin::plugin = NULL;
const string ANNOTATIONS = "AnnotationPlugin: ";

static void matrix2array(const osg::Matrix &m, osg::Matrix::value_type *a)
{
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x)
        {
            a[y * 4 + x] = m(x, y);
        }
}

static void array2matrix(osg::Matrix &m, const osg::Matrix::value_type *a)
{
    for (unsigned y = 0; y < 4; ++y)
        for (unsigned x = 0; x < 4; ++x)
        {
            m(x, y) = a[y * 4 + x];
        }
}


/*
 * Constructor
 *
 */
AnnotationPlugin::AnnotationPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, currentAnnotation(NULL)
{
}

bool AnnotationPlugin::init()
{

    if (plugin)
    {
        std::cerr << "AnnotationPlugin already exists!" << std::endl;
        return false;
    }

    plugin = this;

    // create the TabletUI User-Interface
    annotationTab
        = new coTUIAnnotationTab("Annotations", coVRTui::instance()->mainFolder->getID());
    annotationTab->setPos(0, 0);
    annotationTab->setEventListener(this);

    menuSelected = false; //Menu Item "New Annotation" not selected by default

    // default button size
    float buttonSize[] = { 80, 15 };

    // annotation menu positions
    float x[] = { 0, 60 };
    float y[] = { 60, 45, 30, 15 };
    float z = 1;
    float labelSize = 8;

    // create new annotation menu
    annotationsMenuItem = new coSubMenuItem("Annotations");

    // create row menu to add items
    annotationsMenu = new coRowMenu("Annotation Options");
    annotationsMenuItem->setMenu(annotationsMenu);

    // create a new annotation
    annotationsMenuCheckbox = new coCheckboxMenuItem("New Annotation", false);
    annotationsMenu->add(annotationsMenuCheckbox);
    annotationsMenuCheckbox->setMenuListener(this);

    
    // hide all annotations
    showMenuCheckbox = new coCheckboxMenuItem("Display Annotations", !coCoviseConfig::isOn("COVER.DisplayAnnotations", false));
    annotationsMenu->add(showMenuCheckbox);
    showMenuCheckbox->setMenuListener(this);

    // scale all annotations
    scaleMenuPoti = new coPotiMenuItem("Scale Annotations", 0.01, 10, 1);
    annotationsMenu->add(scaleMenuPoti);
    scaleMenuPoti->setMenuListener(this);

    // add delete all button
    deleteAllButton = new coButtonMenuItem("Delete All");
    annotationsMenu->add(deleteAllButton);
    deleteAllButton->setMenuListener(this);

    // add unlock all button
    unlockAllButton = new coButtonMenuItem("Unlock All");
    annotationsMenu->add(unlockAllButton);
    unlockAllButton->setMenuListener(this);

    //create a pop up menu for annotation interaction
    annotationHandle = new coPopupHandle("AnnotationControl");
    annotationFrame = new coFrame("UI/Frame");
    annotationPanel = new coPanel(new coFlatPanelGeometry(coUIElement::BLACK));
    annotationLabel = new coLabel;
    annotationLabel->setString(ANNOTATIONS);
    annotationLabel->setPos(x[0], y[0], z);
    annotationLabel->setFontSize(labelSize);
    annotationHandle->addElement(annotationFrame);
    annotationDeleteButton
        = new coPushButton(new coRectButtonGeometry(buttonSize[0], buttonSize[1], "Annotation/delete"), this);
    colorPoti = new coValuePoti("Color", this, "Volume/valuepoti-bg");
    annotationDeleteButton->setSize(40);
    annotationDeleteButton->setPos(x[0], y[1], z);
    colorPoti->setPos(x[1], y[2], z);
    colorPoti->setMin(0);
    colorPoti->setMax(1);
    annotationPanel->addElement(annotationLabel);
    annotationPanel->addElement(annotationDeleteButton);
    annotationPanel->addElement(colorPoti);
    annotationPanel->setScale(5);
    annotationPanel->resize();
    annotationFrame->addElement(annotationPanel);
    annotationHandle->setVisible(false);

    activeAnnotation = NULL;
    previousAnnotation = NULL;

    // add annotation menu to the main menu
    cover->getMenu()->add(annotationsMenuItem);

    moving = false;
    interactionA
        = new coNavInteraction(coInteraction::ButtonA, "AnnotationPlacement", coInteraction::Medium);
    interactionC
        = new coNavInteraction(coInteraction::ButtonC, "AnnotationPlacement", coInteraction::Medium);
    interactionI
        = new coNavInteraction(coInteraction::ButtonA, "Annotation Deselection", coInteraction::Low);
    //interactionI->setNotifyOnly(true);
    //coInteractionManager::the()->registerInteraction(interactionI);

    collabID = coVRCommunication::instance()->getID();

    return true;
}

/*
 *
 * Deconstructor. Called if the plugin is removed at runtime
 */
AnnotationPlugin::~AnnotationPlugin()
{
    deleteAllAnnotations();
    removeMenuEntry();
    delete interactionA;
    delete interactionC;
    delete interactionI;
}

/*
 *
 * Called before each frame
 */
void AnnotationPlugin::preFrame()
{
    sensorList.update();

    static osg::Matrix startPos;
    static osg::Matrix invStartHand;

    //update the position of all annotation's labels
    if (!annotations.empty())
    {
        vector<Annotation *>::iterator iter;
        for (iter = annotations.begin(); iter != annotations.end(); iter++)
        {
            (*iter)->updateLabelPosition();
            (*iter)->scaleArrowToConstantSize();
        }
    }

    AnnotationMessage am;

    //check if the user deselects the current annotation
    //inteactionI is only startet if user clicks into empty space
    if (interactionI->wasStarted())
    {
        // send message giving up control of all annotations
        am.token = ANNOTATION_MESSAGE_TOKEN_UNLOCKALL;
        cover->sendMessage(this, coVRPluginSupport::TO_SAME, PluginMessageTypes::AnnotationMessage,
                           sizeof(AnnotationMessage), &am);
        cover->sendMessage(this, "Revit",
            PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
    }
    else if (interactionI->wasStopped() && interactionI->isRegistered())
    {
        interactionI->cancelInteraction();

        //the inteaction will be registered again when the mouse moves over an annotation
        coInteractionManager::the()->unregisterInteraction(interactionI);
    }

    if (!interactionA->isRunning() && !interactionC->isRunning())
    {
        //check if the object was previously moving
        moving = false;
    }

    if ((currentAnnotation) || (menuSelected))
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

    if ((currentAnnotation))
    {
        if (!interactionC->isRegistered())
        {
            coInteractionManager::the()->registerInteraction(interactionC);
        }
        if (!interactionI->isRegistered())
        {
            //cout << "register Interaction I \n\n";
            coInteractionManager::the()->registerInteraction(interactionI);
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
        if (currentAnnotation) // start moving the current Annotation
        {
            previousAnnotation = activeAnnotation;
            activeAnnotation = currentAnnotation; // remenber which annotation has been chosen

            invStartHand.invert(cover->getPointerMat());
            osg::Matrix tmp;
            currentAnnotation->getMat(tmp);
            startPos = tmp * cover->getBaseMat();

            //check if another annotation is being selected to move (releases lock on old annotation)
            if (currentAnnotation != previousAnnotation)
            {
                annotationHandle->setVisible(false);
            }

            // send move Message and lock selected annotation
            //if(currentAnnotation->changesAllowed())
            {
                am.id = currentAnnotation->getID();
                am.token = ANNOTATION_MESSAGE_TOKEN_MOVEADD;
                osg::Matrix translation;
                currentAnnotation->getMat(translation);
                matrix2array(translation, am.translation());

                cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
                cover->sendMessage(this, "Revit",
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
            }
            moving = true;
        }

        else if (menuSelected) // create a new Annotation
        {
            if (cover->getIntersectedNode())
            {
                osg::Matrix trans;

                const osg::Vec3 &hp = cover->getIntersectionHitPointWorld();
                trans.makeTranslate(hp[0], hp[1], hp[2]);
                trans = trans * cover->getInvBaseMat();
                trans.makeTranslate(trans.getTrans()); // get rid of scale part

                osg::Vec3 from(0.0, 0.0, 1.0);
                const osg::Vec3 to = cover->getIntersectionHitPointWorldNormal()
                                     * -1;
                osg::Matrix orientation;
                orientation.makeRotate(from, to);

                am.token = ANNOTATION_MESSAGE_TOKEN_MOVEADD;
                am.id = getLowestUnusedAnnotationID();

                matrix2array(trans, am.translation());
                matrix2array(orientation, am.orientation());

                cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
                cover->sendMessage(this, "Revit",
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
            }
            //set VRUI Menu Item to false
            this->annotationsMenuCheckbox->setState(false);

            //set TabletUI Menu Item to false
            //this->tuiNewButton->setState(false);
            this->annotationTab->setNewButtonState(false);

            // set internal memory variable to false
            menuSelected = 0;
            coIntersection::instance()->isectAllNodes(false);
        }
    }

    if (interactionC->wasStarted())
    {
        if (currentAnnotation)
        {
            // TODO: check if the annotation is selectable
            //if(currentAnnotation->changesAllowed() )
            {
                // check if selecting another annotation
                if (currentAnnotation != previousAnnotation)
                {
                    am.id = currentAnnotation->getID();
                    am.token = ANNOTATION_MESSAGE_TOKEN_SELECT;
                    cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                       PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
                    cover->sendMessage(this, "Revit",
                        PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
                }
                else // close menu
                {
                    annotationHandle->setVisible(false);
                }
            }
        }
    }

    if (interactionA->isRunning())
    {
        //cout << "Interaction A is running" << endl;
        if (currentAnnotation)
        {
            if (!moving) // start moving the current annotation
            {
                invStartHand.invert(cover->getPointerMat());
                osg::Matrix tmp;
                currentAnnotation->getMat(tmp);
                startPos = tmp * cover->getBaseMat();
                moving = true;
            }
            osg::Matrix dMat = invStartHand * cover->getPointerMat();
            osg::Matrix current;
            osg::Matrix tmp;
            tmp = startPos * dMat;
            current = tmp * cover->getInvBaseMat();

            // TODO: check whether this is allowed
            // send move Message
            //if(currentAnnotation->changesAllowed())
            {
                am.id = currentAnnotation->getID();
                am.token = ANNOTATION_MESSAGE_TOKEN_MOVEADD;
                matrix2array(current, am.translation());
                cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
                cover->sendMessage(this, "Revit",
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
            }
        }
    }

    // check if someone closes the annotation panel
    /*if((!annotationHandle->isVisible()) && (previousAnnotation != NULL))
    {
    // send message giving up control of the annotation
    //if(previousAnnotation->changesAllowed())
    {
    am.id = previousAnnotation->getID();
    am.token = ANNOTATION_MESSAGE_TOKEN_UNLOCK;
    cover->sendMessage(this,
    coVRPluginSupport::TO_SAME,
    PluginMessageTypes::AnnotationMessage,
    sizeof(AnnotationMessage),
    &am);
    }
    }*/
}

/*
 * hide one annotation (make transparent)
 */
void AnnotationPlugin::setVisible(Annotation *annot, bool vis)
{
    annot->setVisible(vis);
}
/*
 *
 * hide all annotations (make transparent)
 */
void AnnotationPlugin::setAllVisible(bool vis)
{
    vector<Annotation *>::iterator it = annotations.begin();
    for (; it < annotations.end(); it++)
    {
        (*it)->setVisible(vis);
    }
}

/*
 *
 * Use this when creating new annotations to ensure that the ID list is always continuous.
 */
int AnnotationPlugin::getLowestUnusedAnnotationID()
{
    int id = 0;

    while (isIDInUse(id))
    {
        ++id;
    }
    return id;
}

int AnnotationPlugin::getCollabID()
{
    int id = coVRCommunication::instance()->getID();

    if (id != collabID)
    {
        // we've entered collaboration mode
        // we need to change annotation IDs to our new
        // id, so we don't loose control over them!
        refreshAnnotationOwner(collabID, id);
    }

    return collabID = id;
}

void AnnotationPlugin::refreshAnnotationOwner(int oldID, int newID)
{
    vector<Annotation *>::iterator it = annotations.begin();
    for (; it < annotations.end(); it++)
    {
        if ((*it)->sameOwnerID(oldID))
        {
            (*it)->setOwnerID(newID);
        }
    }
} //refreshAnnotationOwner

/*
 *
 * Check if the id is already used by another annotation
 */
bool AnnotationPlugin::isIDInUse(int id)
{
    vector<Annotation *>::iterator it = annotations.begin();
    for (; it < annotations.end(); it++)
    {
        if ((*it)->getID() == id)
            return true;
    }
    return false;
}

/*
 *
 * Sets the current annotation to a, unless neither the current annotation or a is NULL
 */
void AnnotationPlugin::setCurrentAnnotation(Annotation *a)
{
    if ((currentAnnotation == NULL) || (a == NULL))
        currentAnnotation = a;
}

/*
 *
 * Removes the plugins entry from the VRUI Menu
 */
void AnnotationPlugin::removeMenuEntry()
{
    delete annotationsMenuItem;
}

/*
 *
 * Adjust individual maker colors
 */
void AnnotationPlugin::potiValueChanged(float, float newvalue, coValuePoti *,
                                        int)
{
    if (previousAnnotation != NULL)
    {
        AnnotationMessage mm;
        mm.id = previousAnnotation->getID();
        mm.token = ANNOTATION_MESSAGE_TOKEN_COLOR;
        mm.color = newvalue;
        cover->sendMessage(this, coVRPluginSupport::TO_SAME, PluginMessageTypes::AnnotationMessage,
            sizeof(AnnotationMessage), &mm);
        cover->sendMessage(this, "Revit",
            PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &mm);
    }
}

/*
 *
 * Handle VRUI Menu Events
 */
void AnnotationPlugin::menuEvent(coMenuItem *item)
{
    if (item == annotationsMenuCheckbox)
    {
        fprintf(stderr, "AnnotationPlugin::menuEvent annotationsMenuCheckbox\n");
        //toggle if the next mouseclick will create an new annotation
        if (!menuSelected && annotationsMenuCheckbox->getState())
        {
            menuSelected = annotationsMenuCheckbox->getState();
            coIntersection::instance()->isectAllNodes(true);
            //update Tablet UI
            //tuiNewButton->setState(annotationsMenuCheckbox->getState());
        }
    }
    else if (item == showMenuCheckbox)
    {
        //toggle all Annotations (in)visible
        setAllVisible(showMenuCheckbox->getState());

        //update Tablet UI
        //tuiShowButton->setState(hideMenuCheckbox->getState());
    }
    else if (item == scaleMenuPoti)
    {
        //scale all annotations
        AnnotationMessage mm;
        mm.token = ANNOTATION_MESSAGE_TOKEN_SCALEALL;
        mm.color = scaleMenuPoti->getValue();
        cover->sendMessage(this, coVRPluginSupport::TO_SAME, PluginMessageTypes::AnnotationMessage,
            sizeof(AnnotationMessage), &mm);
        cover->sendMessage(this, "Revit",
            PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &mm);

        //update tabletUI
        //tuiSlider->setValue(scaleMenuPoti->getValue());
    }
    else if (item == deleteAllButton)
    {
        deleteAllAnnotations();
    }
    else if (item == unlockAllButton)
    {
        AnnotationMessage mm;
        mm.token = ANNOTATION_MESSAGE_TOKEN_FORCEUNLOCK;
        cover->sendMessage(this, coVRPluginSupport::TO_SAME, PluginMessageTypes::AnnotationMessage,
                           sizeof(AnnotationMessage), &mm);
        cover->sendMessage(this, "Revit",
            PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &mm);
    }
}

/*
 *
 * Handle VRUI button events
 */
void AnnotationPlugin::buttonEvent(coButton *cobutton)
{
    if (cobutton == annotationDeleteButton)
    {
        deleteAnnotation(previousAnnotation);
    }
}

/*
 *
 * Delete all annotations
 */
void AnnotationPlugin::deleteAllAnnotations()
{
    //check if plugin exists
    if (plugin == NULL)
    {
        fprintf(stderr, "Annotations constuctor never called\n");
        return;
    }

    AnnotationMessage mm;
    mm.token = ANNOTATION_MESSAGE_TOKEN_DELETEALL;
    cover->sendMessage(this, coVRPluginSupport::TO_SAME, PluginMessageTypes::AnnotationMessage,
                       sizeof(AnnotationMessage), &mm);
    cover->sendMessage(this, "Revit",
        PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &mm);
}

/*
 *
 * Delete an annotation
 */
void AnnotationPlugin::deleteAnnotation(Annotation *annot)
{
    AnnotationMessage mm;

    if (annot != NULL)
    {
        mm.id = annot->getID();
        mm.token = ANNOTATION_MESSAGE_TOKEN_REMOVE;
        cover->sendMessage(this, coVRPluginSupport::TO_SAME, PluginMessageTypes::AnnotationMessage,
                           sizeof(AnnotationMessage), &mm);
        cover->sendMessage(this, "Revit",
            PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &mm);
        setCurrentAnnotation(NULL); //currentAnnotation=NULL;
    }
}

/*
 *
 * Handle TabletUI Events from Pushbuttons etc.
 */
void AnnotationPlugin::tabletPressEvent(coTUIElement *tUIItem)
{
    if (tUIItem == annotationTab)
    {
        cout << "AnnotationPlugin::tabletPressEvent: AnnotationTab " << std::endl;
    }
}

/*
 *
 * Handle TabletUI Events from ToggleButtons, Sliders, etc.
 */
void AnnotationPlugin::tabletEvent(coTUIElement *tUIItem)
{
    (void)tUIItem;
}

/*
 * Handle TabletUI Data Events
 */
void AnnotationPlugin::tabletDataEvent(coTUIElement *tUIItem, TokenBuffer &tb)
{
    if (tUIItem == annotationTab)
    {
        int type;
        tb >> type;

        const char *text;
        int annotID;
        int state;
        float color;
        float scaleVal;

        //std::cout << "AnnotationPlugin::tabletDataEvent type = " << type << std::endl;

        switch (type)
        {
        case TABLET_ANNOTATION_SEND_TEXT:
        {
            tb >> text;
            //std::cout << "AnnotationPlugin::TABLET_ANNOTATION_SEND_TEXT text = " << text << std::endl;

            if (activeAnnotation != NULL)
            {
                TokenBuffer tb2;
                tb2 << activeAnnotation->getID();
		tb2 << activeAnnotation->getDocumentID();
                tb2 << activeAnnotation->getOwnerID();
                tb2 << text;
                cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::AnnotationTextMessage, tb2.getData().length(), tb2.getData().data());
                cover->sendMessage(this, "Revit",
                                   PluginMessageTypes::AnnotationTextMessage, tb2.getData().length(), tb2.getData().data());
            }
            break;
        }
        case TABLET_ANNOTATION_NEW:
        {
            tb >> state;
            //toggle if the next mouseclick creates a new annotation
            menuSelected = state;
            //update VRUI
            annotationsMenuCheckbox->setState(state);
            break;
        }
        case TABLET_ANNOTATION_DELETE:
        {
            //std::cout << "AnnotationPlugin::TABLET_ANNOTATION_DELETE" /*<< activeAnnotation->getID()*/ << std::endl;
            deleteAnnotation(activeAnnotation);
            break;
        }
        case TABLET_ANNOTATION_DELETE_ALL:
        {
            //std::cout << "AnnotationPlugin::TABLET_ANNOTATION_DELETE_ALL" << std::endl;
            deleteAllAnnotations();
            break;
        }
        case TABLET_ANNOTATION_SCALE:
        {
            tb >> annotID;
            tb >> scaleVal;

            if (activeAnnotation != NULL)
            {
                AnnotationMessage am;
                am.id = activeAnnotation->getID();
                am.token = ANNOTATION_MESSAGE_TOKEN_SCALE;
                am.color = scaleVal < 0.01 ? 0.01 : scaleVal;
                cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
                cover->sendMessage(this, "Revit",
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
            }
            //std::cout << "AnnotationPlugin::TABLET_ANNOTATION_SCALE ScaleVal = " << scaleVal << std::endl;
            break;
        }
        case TABLET_ANNOTATION_SCALE_ALL:
        {
            tb >> scaleVal;
            AnnotationMessage am;
            am.token = ANNOTATION_MESSAGE_TOKEN_SCALEALL;
            am.color = scaleVal < 0.01 ? 0.01 : scaleVal;
            cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
            cover->sendMessage(this, "Revit",
                PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
            //std::cout << "AnnotationPlugin::TABLET_ANNOTATION_SCALE_ALL scaleVal = " << scaleVal << std::endl;
            break;
        }

        case TABLET_ANNOTATION_SET_COLOR:
        {
            tb >> annotID;
            tb >> color;

            if (activeAnnotation != NULL)
            {
                AnnotationMessage mm;
                mm.id = activeAnnotation->getID();
                mm.token = ANNOTATION_MESSAGE_TOKEN_COLOR;
                mm.color = color;
                cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &mm);
                cover->sendMessage(this, "Revit",
                                   PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &mm);
            }
            //std::cout << "AnnotationPlugin::TABLET_ANNOTATION_SET_COLOR color = " << color << std::endl;
            break;
        }
        case TABLET_ANNOTATION_SET_ALL_COLORS:
        {
            tb >> color;

            AnnotationMessage mm;
            mm.token = ANNOTATION_MESSAGE_TOKEN_COLORALL;
            mm.color = color;
            cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &mm);
            cover->sendMessage(this, "Revit",
                PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &mm);
            //std::cout << "AnnotationPlugin::TABLET_ANNOTATION_SET_ALL_COLORS color = " << color << std::endl;
            break;
        }
        case TABLET_ANNOTATION_SHOW_OR_HIDE:
        {
            //std::cout << "AnnotationPlugin::TABLET_ANNOTATION_SHOW_OR_HIDE state = " << state << std::endl;
            tb >> annotID;
            tb >> state;

            AnnotationMessage am;
            am.token = ANNOTATION_MESSAGE_TOKEN_HIDE;
            am.state = state;
            am.id = annotID;

            cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
            cover->sendMessage(this, "Revit",
                PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);

            break;
        }
        case TABLET_ANNOTATION_SHOW_OR_HIDE_ALL:
        {
            //std::cout << "AnnotationPlugin::TABLET_ANNOTATION_DELETE_ALL state = " << state << std::endl;

            tb >> state;

            AnnotationMessage am;
            am.token = ANNOTATION_MESSAGE_TOKEN_HIDEALL;
            am.state = state;
            cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                               PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
            cover->sendMessage(this, "Revit",
                PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);

            break;
        }
        case TABLET_ANNOTATION_SET_SELECTION:
        {
            tb >> annotID;

            vector<Annotation *>::iterator iter;
            for (iter = annotations.begin(); iter < annotations.end(); iter++)
            {
                if ((*iter)->getID() == annotID)
                {
                    activeAnnotation = (*iter);

                    AnnotationMessage am;
                    am.id = annotID;
                    am.token = ANNOTATION_MESSAGE_TOKEN_MOVEADD;
                    osg::Matrix translation;
                    activeAnnotation->getMat(translation);
                    matrix2array(translation, am.translation());
                    cover->sendMessage(this, coVRPluginSupport::TO_SAME,
                                       PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
                    cover->sendMessage(this, "Revit",
                        PluginMessageTypes::AnnotationMessage, sizeof(AnnotationMessage), &am);
                }
            }
        }
        }
    }
} //AnnotationPlugin::tabletDataEvent

/*
 * Handle incoming messages
 *
 */
void AnnotationPlugin::message(int toWhom, int type, int len, const void *buf)
{
    if (type == PluginMessageTypes::AnnotationMessage) // An AnnotationMessage has been received
    {
        AnnotationMessage *mm = (AnnotationMessage *)buf;

        switch (mm->token)
        {
        case ANNOTATION_MESSAGE_TOKEN_MOVEADD: // MOVE/ADD
        {
            Annotation *curr = NULL;
            vector<Annotation *>::iterator it = annotations.begin();
            for (; it < annotations.end(); it++)
            {
                // unlock annotations that were locked by that host
                // with exception of the selected one
                if (!(*it)->sameID(mm->id) && (*it)->sameOwnerID(mm->sender))
                {
                    (*it)->setOwnerID(-1);
                    (*it)->setAmbientUnlocked();
                }

                // lock annotation with specific id
                // set ambient color to locked
                if ((*it)->sameID(mm->id))
                {
                    curr = *it;

                    if ((*it)->getOwnerID() == -1)
                    {
                        (*it)->setOwnerID(mm->sender);

                        if (mm->sender == getCollabID())
                            (*it)->setAmbientLocalLocked();
                        else
                            (*it)->setAmbientRemoteLocked();
                    }
                }
            } //foreach annotation

            if (curr == NULL) //Add a new Annotation
            {
                osg::Node *pick = NULL;

                /*
                if(cover->getIntersectedNode())
                {
                pick = cover->getIntersectedNode();
                }
                */
                osg::Matrix orientation;
                array2matrix(orientation, mm->orientation());

                curr
                    = new Annotation(mm->id, mm->sender, pick, scaleMenuPoti->getValue(), orientation);
                if (showMenuCheckbox->getState() == false)
                    curr->setVisible(false);
                curr->setOwnerID(mm->sender);
                previousAnnotation = activeAnnotation; //remember last selected annotation
                if (previousAnnotation)
                {
                    previousAnnotation->setAmbientUnlocked();
                }
                activeAnnotation = curr;
                annotations.push_back(curr);
                annotationTab->addAnnotation(mm->id);
            }

            // an annotation may only be manipulated by its current
            // owner, so sender ID and owner ID must equal
            if (curr->sameOwnerID(mm->sender))
            {

                osg::Matrix translation;
                array2matrix(translation, mm->translation());

                curr->setPos(translation);

                if (getCollabID() == mm->sender)
                    curr->setAmbientLocalLocked();
                else
                    curr->setAmbientRemoteLocked();

                annotationTab->setSelectedAnnotation(curr->getID());
            }

            break;
        } // case moveadd

        case ANNOTATION_MESSAGE_TOKEN_REMOVE: // Remove an annotation
        {
            vector<Annotation *>::iterator it = annotations.begin();
            for (; it < annotations.end(); ++it)
            {
                if ((*it)->sameID(mm->id))
                {
                    previousAnnotation = NULL;
                    activeAnnotation = NULL;

                    // close annotationHandle
                    annotationLabel->setString(ANNOTATIONS);
                    annotationHandle->setVisible(false);

                    delete *it;
                    annotations.erase(it);
                    annotationTab->deleteAnnotation(1, mm->id);
                    break;
                }
            }
            break;
        } // case remove

        case ANNOTATION_MESSAGE_TOKEN_SELECT: // annotation selected (right-clicked)
        {
            vector<Annotation *>::iterator it = annotations.begin();
            for (; it < annotations.end(); it++)
            {
                // check for newly selected annotation
                if ((*it)->sameID(mm->id))
                {
                    colorPoti->setValue((*it)->getColor());
                    (*it)->setAmbient(osg::Vec4(1.0f, 1.0f, 1.0f, 1.0f));

                    // make panel visible only for user of the annotation
                    if ((*it)->sameOwnerID(mm->sender))
                    {
                        setCurrentAnnotation(*it);

                        // convert int to string for the label
                        std::ostringstream label;
                        if (label << ((*it)->getID() + 1))
                        {
                            annotationLabel->setString(ANNOTATIONS + label.str());
                        }
                        previousAnnotation = *it;
                        annotationHandle->setVisible(true);
                    }
                }
                else
                {
                    (*it)->setAmbient(osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
                }
            }
            break;
        } // case select

        case ANNOTATION_MESSAGE_TOKEN_COLOR: // Change annotation color
        {
            vector<Annotation *>::iterator it = annotations.begin();
            for (; it < annotations.end(); it++)
            {
                if ((*it)->sameID(mm->id))
                {
                    (*it)->setColor(mm->color);
                    break;
                }
            }
            break;
        } // case color

        case ANNOTATION_MESSAGE_TOKEN_DELETEALL: // Deletes all Annotations
        {
            vector<Annotation *>::iterator it;
            int size = annotations.size();
            for (int i = 0; i < size; i++)
            {
                it = annotations.begin();
                delete *it;
                annotations.erase(it);

                // close annotationHandle
                annotationLabel->setString(ANNOTATIONS);
                annotationHandle->setVisible(false);
            }

            previousAnnotation = NULL;
            activeAnnotation = NULL;
            setCurrentAnnotation(NULL);
            annotationTab->deleteAnnotation(0, 0);
            break;
        } // case deleteall

        // Release current lock on a specific annotation
        // TODO: Possibly remove this, as unlock all
        // does what this is supposed to do
        case ANNOTATION_MESSAGE_TOKEN_UNLOCK:
        {
            vector<Annotation *>::iterator it = annotations.begin();
            for (; it < annotations.end(); it++)
            {
                if ((*it)->sameID(mm->id))
                {
                    (*it)->setOwnerID(-1);

                    (*it)->setAmbientUnlocked();

                    // check if selectable should be reset
                    // TODO: what is this supposed to do?
                    if (*it == previousAnnotation)
                    {
                        annotationLabel->setString(ANNOTATIONS);
                        annotationHandle->setVisible(false);
                        previousAnnotation = NULL;
                    }

                    break;
                } //if id
            } //foreach annotation
            break;
        } //case unlock

        case ANNOTATION_MESSAGE_TOKEN_SCALE: // scale an annotation
        {
            vector<Annotation *>::iterator it = annotations.begin();
            for (; it < annotations.end(); it++)
            {
                if ((*it)->sameID(mm->id))
                {
                    (*it)->setScale(mm->color);
                    break;
                }
            }
            break;
        } //case scale

        case ANNOTATION_MESSAGE_TOKEN_SCALEALL: //scale all Annotations
        {
            vector<Annotation *>::iterator it;
            for (it = annotations.begin(); it < annotations.end(); it++)
            {
                (*it)->setScale(mm->color);
            }
            break;
        } //case scaleall

        case ANNOTATION_MESSAGE_TOKEN_COLORALL: //change all annotation's colors
        {
            vector<Annotation *>::iterator it;
            for (it = annotations.begin(); it < annotations.end(); it++)
            {
                (*it)->setColor(mm->color);
            }
            break;
        } //case colorall

        // release lock on all annotations that are owned by sender
        case ANNOTATION_MESSAGE_TOKEN_UNLOCKALL:
        {
            vector<Annotation *>::iterator it;
            for (it = annotations.begin(); it < annotations.end(); it++)
            {
                if ((*it)->sameOwnerID(mm->sender))
                {
                    std::cout << "Unlock on: " << (*it)->getID() << std::endl;

                    // unlock it
                    (*it)->setOwnerID(-1);

                    //change its color
                    (*it)->setAmbientUnlocked();
                }
            }
            activeAnnotation = NULL;
            break;
        } //case unlockall

        case ANNOTATION_MESSAGE_TOKEN_FORCEUNLOCK:
        {
            vector<Annotation *>::iterator it;
            for (it = annotations.begin(); it < annotations.end(); it++)
            {
                // unlock it
                (*it)->setOwnerID(-1);

                //change its color to unlocked state!
                (*it)->setAmbientUnlocked();
            }
            activeAnnotation = NULL;
            break;
        }

        case ANNOTATION_MESSAGE_TOKEN_HIDE: //hide an annotation
        {
            vector<Annotation *>::iterator iter;
            for (iter = annotations.begin(); iter < annotations.end(); iter++)
            {
                if ((*iter)->sameID(mm->id))
                {
                    setVisible(*iter, mm->state);
                }
            }
            break;
        } //case hide

        case ANNOTATION_MESSAGE_TOKEN_HIDEALL: //hide all annotations
        {
            setAllVisible(mm->state);
            break;
        } //case hideall

        default:
            std::cerr
                << "Annotation: Error: Bogus Annotation message with Token "
                << (int)mm->token << std::endl;
        } //switch mm->token
    } //if type == ann_message

    else if (type == PluginMessageTypes::AnnotationTextMessage)
    {
        TokenBuffer tb(DataHandle((char *)buf, len, false));
        int id;
        tb >> id;
        int document;
        tb >> document;
        int owner;
        tb >> owner;
        const char *text;
        tb >> text;
        //std::cout << "Annotation Text Message: " << text << std::endl;

        vector<Annotation *>::iterator it = annotations.begin();
        for (; it < annotations.end(); it++)
        {
            if ((*it)->sameID(id))
            {
                (*it)->setText(text);
                break;
            }
        }
    } //if type == ann_text_msg
} //AnnotationPlugin::message

COVERPLUGIN(AnnotationPlugin)
