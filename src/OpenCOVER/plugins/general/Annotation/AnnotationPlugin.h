/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ANNOTATIONS_H
#define _ANNOTATIONS_H
#include <util/common.h>

#include "Annotation.h"
#include "AnnotationSensor.h"

#include <cover/coTabletUI.h>
#include <util/coTabletUIMessages.h>

#include <cover/coVRTui.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRCommunication.h>

#include <cover/coVRPlugin.h>

#include <OpenVRUI/coButton.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coValuePoti.h>

#include <PluginUtil/coSensor.h>

#include <net/message.h>

namespace vrui
{
class coButtonMenuItem;
class coCheckboxMenuItem;
class coTrackerButtonInteraction;
class coNavInteraction;
class coInteraction;
class coMouseButtonInteraction;
class coPotiMenuItem;
class coSubMenuItem;
class coRowMenu;
class coFrame;
class coPanel;
}

class Annotation;

using namespace covise;
using namespace vrui;
using namespace opencover;

/// annotation message token types
enum
{
    ANNOTATION_MESSAGE_TOKEN_MOVEADD = 0, // create or move an annotation
    ANNOTATION_MESSAGE_TOKEN_REMOVE = 1, // remove an annotation
    ANNOTATION_MESSAGE_TOKEN_SELECT = 2, // selected through right-click
    ANNOTATION_MESSAGE_TOKEN_COLOR = 3, // change color of annotation
    ANNOTATION_MESSAGE_TOKEN_DELETEALL = 4, // delete all annotations
    ANNOTATION_MESSAGE_TOKEN_UNLOCK = 5, // release a lock on an annotation
    ANNOTATION_MESSAGE_TOKEN_SCALE = 6, // change scale on a single annotation
    ANNOTATION_MESSAGE_TOKEN_SCALEALL = 7, // scale all annotations
    ANNOTATION_MESSAGE_TOKEN_COLORALL = 8, // change color on all annotations
    ANNOTATION_MESSAGE_TOKEN_UNLOCKALL = 9, // release lock on all annotations
    ANNOTATION_MESSAGE_TOKEN_HIDE = 10, // hide an annotation
    ANNOTATION_MESSAGE_TOKEN_HIDEALL = 11, // hide all annotations
    ANNOTATION_MESSAGE_TOKEN_FORCEUNLOCK = 12 // 'emergency button' unlock all annotations
};

/*
 * We need to set data alignment to 1, so there are no problems
 * with struct alignment on different computers
 */
#pragma pack(push, 1)
struct AnnotationMessage // message of type ANNOTATION_MESSAGE
{
    int token; ///< token type
    int id; ///< id of annotation
    int sender; ///< sender id
    float color; ///< hue value
    bool state; ///< DOCUMENT ME

    osg::Matrix::value_type _translation[16]; ///< used for passing translation
    osg::Matrix::value_type _orientation[16]; ///< used for passing orientation

    const osg::Matrix::value_type *orientation() const
    {
        return &_orientation[0];
    }
    osg::Matrix::value_type *orientation()
    {
        return &_orientation[0];
    }

    const osg::Matrix::value_type *translation() const
    {
        return &_translation[0];
    }
    osg::Matrix::value_type *translation()
    {
        return &_translation[0];
    }
    
/*
 * AnnotationMessage Constructor
 */
    AnnotationMessage()
    {
        token = 0;
        id = -1;
        sender = coVRCommunication::instance()->getID();
        color = 0.0f;
        state = false;
        memset(_translation, 0, sizeof(_translation));
        memset(_orientation, 0, sizeof(_translation));
    }
};
#pragma pack(pop)


class AnnotationPlugin : public coVRPlugin,
                         public coMenuListener,
                         public coButtonActor,
                         public coValuePotiActor,
                         public coTUIListener
{
    friend class Annotation;

private:
    coSensorList sensorList;
    Annotation *currentAnnotation; ///< The Annotation the mouse is currently over (no click necessary!)
    Annotation *previousAnnotation; ///< The last Annotation that has been rightclicked.
    Annotation *activeAnnotation; ///< The currently selected Annotation
    vector<Annotation *> annotations; ///< collection of annotations

    coTUIAnnotationTab *annotationTab; ///< The TabletUI Interface

    // The VR Menu Interface
    void createMenuEntry(); ///< create a VR menu item "Annotations"
    void removeMenuEntry(); ///< remove the VR menu item
    void menuEvent(coMenuItem *); ///< handles VR menu events

    coCheckboxMenuItem *annotationsMenuCheckbox;
    coSubMenuItem *annotationsMenuItem;
    coPotiMenuItem *scaleMenuPoti;
    coCheckboxMenuItem *hideMenuCheckbox;
    coButtonMenuItem *deleteAllButton;
    coButtonMenuItem *unlockAllButton;
    coRowMenu *annotationsMenu;

    // annotation gui
    coPopupHandle *annotationHandle;
    coFrame *annotationFrame;
    coPanel *annotationPanel;
    coLabel *annotationLabel;
    coButton *annotationDeleteButton;
    coValuePoti *colorPoti;

    // button interaction
    coNavInteraction *interactionA; ///< interaction for first button
    coNavInteraction *interactionC; ///< interaction for third button
    coNavInteraction *interactionI; ///< interaction for deselection of annotations

    bool moving; ///< is the selected Annotation being moved
    float scale; ///< DOCUMENT ME!
    int selectedAnnotationId; ///< DOCUMENT ME!
    int collabID; ///< save collaboration ID

    //Events from the TabletUI

    /// DOCUMENT ME!
    void tabletPressEvent(coTUIElement *);

    /// DOCUMENT ME!
    void tabletEvent(coTUIElement *);

    /// DOCUMENT ME!
    void tabletDataEvent(coTUIElement *tUIItem, TokenBuffer &tb);

    /// DOCUMENT ME!
    void buttonEvent(coButton *);

    /// change visibility of specific annotation instance
    void setVisible(Annotation *annot, bool vis);

    /// change visibility of all annotations
    void setAllVisible(bool);

    /// returns local hostname
    const string &getMyHost() const;

    /// returns whether the parameter is the local host
    bool isLocalhost(const string &h) const
    {
        return h == getMyHost();
    }

    /// deprecated: copies string, checking for length
    void setMMString(char *, string);

    /// returns lowest ID that is unused
    int getLowestUnusedAnnotationID();

    /// checks whether a specific ID is in use
    bool isIDInUse(int);

    /// returns local ID
    int getCollabID();

    /// update all owner IDs for existing annotations!
    void refreshAnnotationOwner(int oldID, int newID);

protected:
    void potiValueChanged(float oldvalue, float newvalue, coValuePoti *poti,
                          int context);

public:
    static AnnotationPlugin *plugin;

    AnnotationPlugin();
    virtual ~AnnotationPlugin();

    bool init();

    void preFrame(); ///< this will be called in PreFrame
    void message(int type, int len, const void *buf); ///< handle incoming messages
    void setCurrentAnnotation(Annotation *m); ///< change current annotation
    int menuSelected; ///< TRUE if menu item "New Annotation" was selected
    void deleteAllAnnotations(); ///< deletes all unlocked/local annotations
    void deleteAnnotation(Annotation *annot); ///< deletes a specific annotation if it is local/unlocked
};
#endif
