/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MODULE_FEEDBACK_MANAGER_H_
#define MODULE_FEEDBACK_MANAGER_H_

#include <util/coTypes.h>

#include <string>
using namespace std;

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>

namespace vrui
{
class coCheckboxMenuItem;
class coButtonMenuItem;
class coSubMenuItem;
class coRowMenu;
}
class cp3dplane;

namespace opencover
{
class RenderObject;
class ColorBar;
class coInteractor;
}

#include <osg/Node>
#include <osg/Geode>
#include <osg/MatrixTransform>

// this class manages the interaction (menu and execution)
// of a module (but not the icon)

namespace opencover
{

class PLUGIN_UTILEXPORT ModuleFeedbackManager : public vrui::coMenuListener
{
public:
    ModuleFeedbackManager(opencover::RenderObject *, opencover::coInteractor *, const char *pluginName);
    ModuleFeedbackManager(RenderObject *, RenderObject *, const char *pluginName);
    virtual ~ModuleFeedbackManager();

    // returns true if inter comes from the same module: used in
    // ModuleFeedbackPlugin::add for adding or updating an entry in
    // ModuleFeedbackPlugin::_ComplexModuleList
    bool compare(coInteractor *inter);
    // returns true if name is an object name from the same module:
    // used in ModuleFeedbackPlugin::remove for deleting an entry
    // in ModuleFeedbackPlugin::_ComplexModuleList
    bool compare(const char *name);
    // returns true if this moduelFeedbackManager is for the same plugin
    bool comparePlugin(const char *pluginName);

    // called for menu update when a new object is received
    virtual void update(RenderObject *container, coInteractor *);
    virtual void update(RenderObject *container, RenderObject *);

    // empty implementation
    virtual void preFrame();

    // empty implementation of 3DTex functionality
    virtual void update3DTex(std::string, cp3dplane *, const char *cmName);

    string ModuleName(const char *) const;
    string ModuleName()
    {
        return moduleName_;
    };

    // menu event for general items
    virtual void menuEvent(vrui::coMenuItem *menuItem);

    // set checkbox and and hide geometry
    void setHideFromGui(bool);

    // set the case name and put item into this menu
    void setCaseFromGui(const char *casename);

    // set the a new name
    void setNameFromGui(const char *casename);

    // show/hide geometry
    virtual void hideGeometry(bool);

    //Transform geometry
    void setMatrix(float *row0, float *row1, float *row2, float *row3);

    //Transform geometry
    void addNodeToCase(osg::Node *node);

    //enable feed back to Colors module
    void addColorbarInteractor(coInteractor *i);

    opencover::coInteractor *getInteractor()
    {
        return inter_;
    };
    bool getSyncState();

protected:
    // helper for constructor
    void createMenu();
    void registerObjAtUi(string name);

    // helper for update
    void updateMenuNames();
    void updateColorBar(RenderObject *container);

    // gets either from attribute OBJECTNAME or from the module name
    // a suggestion for the menu name, the result is kept in _menuName
    //std::string suggestMenuName();
    string getMenuName() const;

    vrui::coSubMenuItem *menuItem_; // submenu entry in covise menu  "Tracer_1..."
    vrui::coRowMenu *menu_; // the menu for the interaction of the module managed by the instance of this class
    vrui::coCheckboxMenuItem *hideCheckbox_; // hide geometry
    vrui::coCheckboxMenuItem *syncCheckbox_; // sync interaction
    vrui::coButtonMenuItem *newButton_; // copy this module
    vrui::coButtonMenuItem *deleteButton_; // delete this module
    vrui::coCheckboxMenuItem *executeCheckbox_; // execute module button
    bool inExecute_; // switch ceckbox off when new object arrived
    vrui::coSubMenuItem *colorsButton_; // open colorbar
    opencover::ColorBar *colorBar_; // colorbar menu
    opencover::coInteractor *inter_; // the last interaction got from the module at issue
    string menuName_; // name associated to _menuItem: its updated when a new object is received

    // Overload this function if you want to do anything before an EXEC
    // call is sent to a module. Always call it before doing an EXEC call!
    virtual void preExecCB(opencover::coInteractor *){};

    // search the corresponding node in sg
    osg::Node *findMyNode();
    std::vector<osg::Geode *> findMyGeode();

    string geomObjectName_;
    string containerObjectName_;
    string attrObjectName_;
    string attrPartName_;

    string initialObjectName_; //we have to save it for the grmsg, because _inter is not always valid
    osg::ref_ptr<osg::MatrixTransform> geometryCaseDCS_;

private:
    // it is used in ModuleFeedbackManager::compare(const char *name)
    string moduleName_;

    string pName_;

    vrui::coMenu *coviseMenu_;
    vrui::coMenu *parentMenu_;
    vrui::coRowMenu *caseMenu_; // up to now this was the "Covise" menu
    vrui::coSubMenuItem *caseMenuItem_;
    string caseName_;

    string visItemName_; //"Tracer_1..."
    string visMenuName_; // "Tracer_1"
    void sendHideMsg(bool);
    std::vector<osg::Geode *> findRecMyGeode(osg::Node *node);
    osg::ref_ptr<osg::Node> myNode_;
    osg::ref_ptr<osg::Group> myNodesParent_;
};
}
#endif
