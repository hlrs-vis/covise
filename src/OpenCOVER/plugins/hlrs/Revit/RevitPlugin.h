/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Revit_PLUGIN_H
#define _Revit_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: Revit Plugin (connection to Autodesk Revit Architecture)    **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                  **
 **                                                                          **
 ** History:  								                                         **
 ** Mar-09  v1	    				       		                                   **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <net/covise_connect.h>
#include <OpenVRUI/coMenu.h>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Group>
#include <stack>
#include <map>
#include <cover/coTabletUI.h>
#include <OpenVRUI/sginterface/vruiActionUserData.h>

class RevitInfo : public vrui::vruiUserData
{
public:
    RevitInfo();
    ~RevitInfo();
    int ObjectID;
};

namespace vrui
{
class coCheckboxMenuItem;
class coSubMenuItem;
class coRowMenu;
class coCheckboxGroup;
class coButtonMenuItem;
}

class RevitPlugin;
class RevitParameter;

using namespace vrui;
using namespace opencover;
using covise::Message;
using covise::ServerConnection;

class RevitViewpointEntry : public coMenuListener
{
public:
    RevitViewpointEntry(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, RevitPlugin *plugin, std::string n);
    virtual ~RevitViewpointEntry();
    virtual void menuEvent(coMenuItem *button);
    void setMenuItem(coCheckboxMenuItem *aMenuItem);
    coTUIToggleButton *getTUIItem()
    {
        return tuiItem;
    };
    void activate();
    int entryNumber;

private:
    std::string name;
    RevitPlugin *myPlugin;
    osg::Vec3 eyePosition;
    osg::Vec3 viewDirection;
    osg::Vec3 upDirection;
    coCheckboxMenuItem *menuItem;
    coTUIToggleButton *tuiItem;
};

class ElementInfo
{
public:
    ElementInfo();
    virtual ~ElementInfo();
    std::list<osg::Node *> nodes;
    std::list<RevitParameter *> parameters;
    void addParameter(RevitParameter *p);
    int ID;
    std::string name;

private:
    coTUIFrame *frame;
    static int yPos;
};

class RevitParameter : public coTUIListener
{
public:
    RevitParameter(int i, std::string n, int st, int pt, int num, ElementInfo *ele)
        : ID(i)
        , name(n)
        , StorageType(st)
        , ParameterType(pt)
        , number(num)
        , element(ele){};
    virtual ~RevitParameter();
    ElementInfo *element;
    int ID;
    int number; // param number in Element;
    std::string name;
    int StorageType;
    int ParameterType;
    double d;
    int ElementReferenceID;
    int i;
    std::string s;
    void createTUI(coTUIFrame *frame, int pos);
    virtual void tabletEvent(coTUIElement *tUIItem);

    coTUILabel *tuiLabel;
    coTUIElement *tuiElement;

private:
};

class RevitPlugin : public coVRPlugin, public coMenuListener, public coTUIListener
{
public:
    // Summary:
    //     An enumerated type listing all of the internal parameter data storage types
    //     that Autodesk Revit supports.
    enum StorageType
    {
        // Summary:
        //     None represents an invalid storage type. This value should not be used.
        None = 0,
        //
        // Summary:
        //     The internal data is stored in the form of a signed 32 bit integer.
        Integer = 1,
        //
        // Summary:
        //     The data will be stored internally in the form of an 8 byte floating point
        //     number.
        Double = 2,
        //
        // Summary:
        //     The internal data will be stored in the form of a string of characters.
        String = 3,
        //
        // Summary:
        //     The data type represents an element and is stored as the id of the element.
        ElementId = 4,
    };
    enum MessageTypes
    {
        MSG_NewObject = 500,
        MSG_DeleteObject = 501,
        MSG_ClearAll = 502,
        MSG_UpdateObject = 503,
        MSG_NewGroup = 504,
        MSG_NewTransform = 505,
        MSG_EndGroup = 506,
        MSG_AddView = 507,
        MSG_DeleteElement = 508,
        MSG_NewParameter = 509,
        MSG_SetParameter = 510,
        MSG_NewMaterial = 511,
        MSG_NewPolyMesh = 512,
        MSG_NewInstance = 513,
        MSG_EndInstance = 514,
        MSG_SetTransform = 515
    };
    enum ObjectTypes
    {
        OBJ_TYPE_Mesh = 1,
        OBJ_TYPE_Curve,
        OBJ_TYPE_Instance,
        OBJ_TYPE_Solid,
        OBJ_TYPE_RenderElement,
        OBJ_TYPE_PolyMesh
    };
    RevitPlugin();
    ~RevitPlugin();
    static RevitPlugin *instance()
    {
        return plugin;
    };

    // this will be called in PreFrame
    void preFrame();

    void destroyMenu();
    void createMenu();
    virtual void menuEvent(coMenuItem *aButton);
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    int maxEntryNumber;
    coTUITab *revitTab;
    void sendMessage(Message &m);
    
    void message(int type, int len, const void *buf);

protected:
    static RevitPlugin *plugin;
    coSubMenuItem *REVITButton;
    coRowMenu *viewpointMenu;
    coCheckboxGroup *cbg;
    std::list<RevitViewpointEntry *> viewpointEntries;
    coButtonMenuItem *addVPButton;
    coButtonMenuItem *reloadButton;
    coTUIButton *saveViewpoint;
    coTUIButton *reload;

    ServerConnection *serverConn;
    ServerConnection *toRevit;
    void handleMessage(Message *m);

    void setDefaultMaterial(osg::StateSet *geoState);
    osg::ref_ptr<osg::Material> globalmtl;
    osg::ref_ptr<osg::MatrixTransform> revitGroup;
    std::stack<osg::Group *> currentGroup;
    std::map<int, ElementInfo *> ElementIDMap;

    Message *msg;
};
#endif
