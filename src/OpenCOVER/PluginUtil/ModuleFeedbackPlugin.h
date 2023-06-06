/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MODULE_FEEDBACK_PLUGIN_H_
#define MODULE_FEEDBACK_PLUGIN_H_

#include <util/DLinkList.h>
#include <util/coTypes.h>
#include <cover/coVRPlugin.h>

namespace opencover
{
class ModuleFeedbackManager;
class coInteractor;
class RenderObject;
}

// ModuleFeedbackPlugin is a base class for plugins which have feedback to covise modules
// the derived plugin class, for example TracerPlugin, handles the feedback to all Tracer Modules
// the class ModuleFeedbackPlugin keeps a list of ModuleFeedbackManagers
// the class ModuleFeedbackManager handles the feedback to a module
namespace opencover
{

class PLUGIN_UTILEXPORT ModuleFeedbackPlugin : public opencover::coVRPlugin
{
public:
    ModuleFeedbackPlugin(const char *pluginName);
    virtual ~ModuleFeedbackPlugin();

    // if a ModuleFeedbackManager object is already present in
    // the list ComplexModuleList for the module from which
    // container and inter have originated, then that ModuleFeedbackManager
    // object is updated (this produces menu updates, for instance),
    // if no such object is available, a new ModuleFeedbackManager object is
    // created and appended to the list
    void add(const opencover::RenderObject *container, opencover::coInteractor *inter);
    void add(const opencover::RenderObject *container, const opencover::RenderObject *geomObj);

    // removes the ModuleFeedbackManager object from the list
    void remove(const char *objName);

    // calls preFrame for all module feedback manager objects in the list which belong to that plugin
    void preFrame();

    // set visibility of an object from Gui
    void handleGeoVisibleMsg(const char *objectName, bool hide);

    //set Transformation of an object
    void setMatrix(const char *objectName, float *row0, float *row1, float *row2, float *row3);

    // if the case name is set from Gui the item is moved from the Covise menu to a case menu
    void handleSetCaseMsg(const char *objectName, const char *casename);

    // if the name is set from Gui the item name and menu title has to be changed
    void handleSetNameMsg(const char *coviseObjectName, const char *newName);

    // change the dcs of a a node
    void addNodeToCase(const char *objectName, osg::Node *node);

    // handle adding interactors to colorbars
    void newInteractor(const RenderObject *container, coInteractor *i);

protected:
    // factory method that returns pointers to object of derived classes of
    // ModuleFeedbackManager
    virtual opencover::ModuleFeedbackManager *NewModuleFeedbackManager(const opencover::RenderObject *, opencover::coInteractor *, const opencover::RenderObject *, const char *) = 0;

    // unique global list of all interactions
    static covise::DLinkList<opencover::ModuleFeedbackManager *> _ComplexModuleList;

    // list of all interaction of the derived class plugin
    covise::DLinkList<opencover::ModuleFeedbackManager *> myInteractions_;

public:
    virtual void getSyncInteractors(coInteractor *inter_); // collect all interactors from other modules which should be syncronized with this one
    // the following methods will set parameters on all those intersctors
    std::list<coInteractor *> interactors;
    /// execute the Modules
    virtual void executeModule();

    /// set Boolean Parameter
    virtual void setBooleanParam(const char *name, int val);

    /// set float scalar parameter
    virtual void setScalarParam(const char *name, float val);

    /// set int scalar parameter
    virtual void setScalarParam(const char *name, int val);

    /// set float slider parameter
    virtual void setSliderParam(const char *name, float min, float max, float value);

    /// set int slider parameter
    virtual void setSliderParam(const char *name, int min, int max, int value);

    /// set float Vector Param
    virtual void setVectorParam(const char *name, int numElem, float *field);
    virtual void setVectorParam(const char *name, float u, float v, float w);

    /// set int vector parameter
    virtual void setVectorParam(const char *name, int numElem, int *field);
    virtual void setVectorParam(const char *name, int u, int v, int w);

    /// set string parameter
    virtual void setStringParam(const char *name, const char *val);

    /// set choice parameter, pos starts with 0
    virtual void setChoiceParam(const char *name, int pos);
    virtual void setChoiceParam(const char *name, int num, const char *const *list, int pos);
};
}
#endif
