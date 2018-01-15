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
#include <cover/coVRShader.h>
#include <net/covise_connect.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coLabelMenuItem.h>
#include <osg/Material>
#include <osg/StateSet>
#include <osg/Group>
#include <stack>
#include <map>
#include <cover/coTabletUI.h>
#include <OpenVRUI/sginterface/vruiActionUserData.h>
// for AnnotationMessage:
#include <../../general/Annotation/AnnotationPlugin.h>

#define REVIT_FEET_TO_M 0.304799999536704
#define REVIT_M_TO_FEET 3.2808399

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
    RevitViewpointEntry(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, RevitPlugin *plugin, std::string n, int id,coCheckboxMenuItem *me);
    virtual ~RevitViewpointEntry();
    virtual void menuEvent(coMenuItem *button);
    void setMenuItem(coCheckboxMenuItem *aMenuItem);
    coTUIToggleButton *getTUIItem()
    {
        return tuiItem;
    };
    
    void setValues(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, std::string n);
    void activate();
    void deactivate();
    
    void updateCamera();
    int entryNumber;
    int ID;
    bool isActive = false;

private:
    std::string name;
    RevitPlugin *myPlugin = nullptr;
    osg::Vec3 eyePosition;
    osg::Vec3 viewDirection;
    osg::Vec3 upDirection;
    coCheckboxMenuItem *menuItem = nullptr;
    coTUIToggleButton *tuiItem = nullptr;
    coCheckboxMenuItem *menuEntry = nullptr;
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
    coTUIFrame *frame = nullptr;
    static int yPos;
};
class AnnotationInfo
{
public:
    double x,y,z,h,p,r;
    std::string text;
    int ID;
};

class TextureInfo
{
public:
	TextureInfo(TokenBuffer &tb);
	double sx, sy, ox, oy, angle,amount;
	std::string texturePath;
	unsigned char r,g,b;
	int ID;
};

class MaterialInfo
{
public:
	MaterialInfo(TokenBuffer &tb);
	unsigned char r, g, b, a;
	TextureInfo *bumpTexture;
	TextureInfo *diffuseTexture;
	osg::StateSet *geoState;
	coVRShader *shader;
	int ID;
};


class DoorInfo
{
public:
	DoorInfo(int id, const char *Name, osg::MatrixTransform *tn, TokenBuffer &tb);
	std::string name;
	osg::MatrixTransform *transformNode;
	int ID;
	bool HandFlipped;
	bool FaceFlipped;
	bool isSliding;
	osg::Vec3 HandOrientation;
	osg::Vec3 FaceOrientation;
	osg::Vec3 Direction;
	osg::Vec3 Origin;
	double maxDistance;
	osg::Vec3 Center;
	float activationDistance2;
	bool entered;
	bool left;
	bool isActive;
	double startTime;
	double animationTime;
	void checkStart(osg::Vec3 &viewerPosition); 
	void translateDoor(float fraction);
	osg::BoundingBox boundingBox;
	bool update(osg::Vec3 &viewerPosition); // returns false if updates are done and it can be removed from the list
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
    int ID;
    std::string name;
    int StorageType;
    int ParameterType;
    int number; // param number in Element;
    ElementInfo *element = nullptr;
    double d;
    int ElementReferenceID;
    int i;
    std::string s;
    void createTUI(coTUIFrame *frame, int pos);
    virtual void tabletEvent(coTUIElement *tUIItem);

    coTUILabel *tuiLabel = nullptr;
    coTUIElement *tuiElement = nullptr;

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
        MSG_SetTransform = 515,
        MSG_UpdateView = 516,
        MSG_AvatarPosition = 517,
        MSG_RoomInfo = 518,
        MSG_NewAnnotation = 519,
        MSG_ChangeAnnotation = 520,
        MSG_ChangeAnnotationText = 521,
        MSG_NewAnnotationID = 522,
        MSG_Views = 523,
        MSG_SetView = 524,
		MSG_Resend = 525,
		MSG_NewDoorGroup = 526,
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
    virtual bool init();
    static RevitPlugin *instance()
    {
        return plugin;
    };

	bool update();
    // this will be called in PreFrame
	void preFrame();

	bool checkDoors();

    void destroyMenu();
    void createMenu();
    virtual void menuEvent(coMenuItem *aButton);
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    int maxEntryNumber;
    coTUITab *revitTab = nullptr;
    void sendMessage(Message &m);
    
    void message(int toWhom, int type, int len, const void *buf);
    void deactivateAllViewpoints();
    int getAnnotationID(int revitID);
    int getRevitAnnotationID(int ai);
    void createNewAnnotation(int id, AnnotationMessage *am);
    void changeAnnotation(int id, AnnotationMessage *am);
	std::list<DoorInfo *> doors;
	std::list<DoorInfo *> activeDoors;
protected:
    static RevitPlugin *plugin;
    coSubMenuItem *REVITButton = nullptr;
    coSubMenuItem *roomInfoButton = nullptr;
    coLabelMenuItem *label1 = nullptr;
    coRowMenu *viewpointMenu = nullptr;
    coRowMenu *roomInfoMenu = nullptr;
    coCheckboxGroup *cbg = nullptr;
    std::list<RevitViewpointEntry *> viewpointEntries;
    coButtonMenuItem *addCameraButton = nullptr;
    coButtonMenuItem *updateCameraButton = nullptr;
    coTUIButton *addCameraTUIButton = nullptr;
    coTUIButton *updateCameraTUIButton = nullptr;
    coTUIComboBox *viewsCombo = nullptr;

    ServerConnection *serverConn = nullptr;
    ServerConnection *toRevit = nullptr;
    void handleMessage(Message *m);

	MaterialInfo * getMaterial(int revitID);


    void setDefaultMaterial(osg::StateSet *geoState);
    osg::ref_ptr<osg::Material> globalmtl;
    osg::ref_ptr<osg::MatrixTransform> revitGroup;
    std::stack<osg::Group *> currentGroup;
    std::map<int, ElementInfo *> ElementIDMap;
    osg::Matrix invStartMoveMat;
    osg::Matrix lastMoveMat;
    bool MoveFinished;
    int MovedID;
    RevitInfo  *info = nullptr;
    std::vector<int> annotationIDs;
	std::map<int, MaterialInfo *> MaterialInfos;

	osg::Image *createNormalMap(osg::Image *heightMap, double pStrength);
	
    float scaleFactor;
	std::string textureDir;
    

    Message *msg = nullptr;
};
#endif
