/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// CLASS    InvAnnoManager
//
// Description: Handels the creation, removal and text-input of  annotation
//              flags.
//
// Initial version: 19.11.2001  by RM
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef INVANNOMANAGER_H
#define INVANNOMANAGER_H

#include <Inventor/SoPickedPoint.h>

#include <Inventor/nodes/SoGroup.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/SbBox.h>

#include <string>
#include <vector>

#include "InvCoviseViewer.h"
#include "InvAnnotationFlag.h"

class InvAnnoManager
{
public:
    enum actionMode
    {
        MAKE,
        REMOVE,
        EDIT
    };

    /// The one and only access to this object.
    //
    /// @return        pointer to the one and only instance
    static InvAnnoManager *instance();

    /// Initialize the singleton. Prior to this call only activating and deactivating
    /// will set internal states. All other methods will have no effect.
    /// You MUST use this method after INVENTOR is initialized!!!

    /// @param   v   viewer the object belongs to
    void initialize(InvCoviseViewer *v);

    /// add and create an AnnoFlag
    void add();

    /// add an already existing AnnoFlag and show it
    //
    /// @param flag object to add
    void add(const InvAnnoFlag *af);

    /// get an message and do whatever has to be done
    //
    /// @param message as it comes in from the communication
    void update(const char *msg);

    /// static member to be used as pick filter callback or to be used
    /// inside an existing one
    //
    /// @param me pointer to the actual instance (MUST be this)
    //
    /// @param picked point
    static SoPath *pickFilterCB(void *me, const SoPickedPoint *pick);

    /// static member to be used as selection callback or to be used
    /// inside an existing one
    //
    /// @param me pointer to the actual instance (MUST be this)
    //
    /// @param selected path
    static void selectionCB(void *me, SoPath *p);

    /// static member to be used as de-selection callback or to be used
    /// inside an existing one
    //
    /// @param me pointer to the actual instance (MUST be this)
    //
    /// @param de-selected path
    static void deSelectionCB(void *me, SoPath *p);

    /// set a global scaling factor for all annotation flags managed by this obj.
    /// To be used in all callbacks which change the camera parameters. If chosen
    /// correctly all flags appear to have the same size independant of camera parameters.
    //
    /// @param   appropriate scale factor
    void reScale(const float &s);

    /// set an initial global scale factor dependig on the bounding box of an object
    //
    /// @param bounding box structure
    void setSize(const SbBox3f &bb);

    /// returns the root node of the scene graph containing all annotation flags
    //
    /// @return  root node
    SoGroup *getGroup() const
    {
        return flagGroup_;
    };

    /// returns the currently active flag
    //
    /// @return active flag
    InvAnnoFlag *getActiveFlag()
    {
        return activeFlag_;
    };

    /// static member to be used as event handeler or to be used
    /// inside an existing one
    //
    /// @param me pointer to the actual instance (MUST be this)
    //
    /// @param X-event structure
    static SbBool kbEventHandler(void *me, XAnyEvent *event);

    /// if the object is not activated all callbacks will simply return
    /// without doing anything
    //
    /// @param  an activation mode (InvAnnoManager::MAKE,InvAnnoManager::EDIT or InvAnnoManager::REMOVE)
    void activate(const int &mode = InvAnnoManager::MAKE);

    /// deactivate the manager
    void deactivate();

    /// check if active
    //
    /// return true if active false else
    bool isActive() const;

    /// check if keyboard input is activated only in this case the kbEventhandler will
    /// do something
    bool kbIsActive();

    /// activate text input
    void setKbActive();

    /// deactivate text input
    void setKbInactive();

    /// @return number of flags currently managed by *this
    int getNumFlags() const;

    /// show all flags
    void showAll();

    /// set true if the viewer obj has shown its data if set to false no flags will be
    /// shown
    //
    /// @param boolean to indicate if the viewer has data
    void hasDataObj(const bool &b);

    /// send all flags in serialized form to COVISE
    void sendParameterData();

    /// DESTRUCTOR
    ~InvAnnoManager();

private:
    /// default CONSTRUCTOR
    InvAnnoManager();

    // !! helper: exit if not initialized
    void initCheck();

    static InvAnnoManager *instance_;

    bool isInitialized_;
    bool isActive_;

    int numFlags_;
    SoGroup *flagGroup_;
    SoSeparator *flagSep_;
    InvCoviseViewer *viewer_;
    InvAnnoFlag *activeFlag_;
    vector<InvAnnoFlag *> flags_;
    SoPickedPoint *actPickedPoint_;
    bool kbActive_;
    int mode_;
    int trueNumFlags_;
    bool hasDataObj_;
    SoOrthographicCamera *camera_;
    SoScale *reScale_;
    float scale_;
};

// global variable to simplify the access to the one and only instance
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// This global definition is DANGEROUS
// because it is very likely that the singleton is created before INVENTOR
// is initialized. In order to preserve the convience of the singleton pattern
// make sure that ONLY allocation of INVENTOR nodes occurs in the constructor
// BUT all other actions of the sub-scene graph managed by this singleton occur
// after INVENTOR is initialized e.g. addChild. (INVENTOR itself uses global data
// which is hidden to the application programmer.)
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
static InvAnnoManager *Annotations = InvAnnoManager::instance();
#endif
