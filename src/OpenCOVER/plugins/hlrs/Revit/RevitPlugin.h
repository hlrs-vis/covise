/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Revit_PLUGIN_H
#define _Revit_PLUGIN_H

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
#include <OpenVRUI/coCombinedButtonInteraction.h>
// for AnnotationMessage:
#include <../../general/Annotation/AnnotationPlugin.h>

#include <cover/MarkerTracking.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Label.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Group.h>
#include <cover/ui/EditField.h>
#include <cover/ui/SelectionList.h>
#include <cover/coVRLabel.h>

#include "IK/CRobot.h"
#include "IK/CAlgoFactory.h"
#include "Doors.h"
#include "Elevators.h"

#include <PluginUtil/coSensor.h>

constexpr double REVIT_FEET_TO_M{0.304799999536704};
constexpr double REVIT_M_TO_FEET{3.2808399};

struct RevitInfo : public vrui::vruiUserData
{
    RevitInfo();
    ~RevitInfo();
    int ObjectID;
    int DocumentID;
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

class RevitDesignOption;
class RevitDesignOptionSet;

struct RevitDesignOption
{
    RevitDesignOption(RevitDesignOptionSet *s);
    std::string name;
    RevitDesignOptionSet *set;
    int ID;
    bool visible;
};

struct RevitDesignOptionSet
{
    RevitDesignOptionSet();
    ~RevitDesignOptionSet();

    void createSelectionList();

    std::list<RevitDesignOption> designOptions;
    std::string name;
    ui::SelectionList* designoptionsCombo=nullptr;
    int ID;
    int DocumentID;
};

class RevitViewpointEntry
{
public:
    RevitViewpointEntry(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, RevitPlugin *plugin, std::string n, int id,int docID, osg::Group *myGroup);
    virtual ~RevitViewpointEntry();
    
    const std::string & getName()const { return name; };

    void setValues(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, std::string n);
    void activate();
    void setActive(bool);
    void deactivate();
    void updateCamera();

    int entryNumber;
    int ID;
    int documentID;
    bool isActive = false;

private:
    osg::Vec3 eyePosition;
    osg::Vec3 viewDirection;
    osg::Vec3 upDirection;
    std::string name;
    RevitPlugin *myPlugin = nullptr;
    osg::MatrixTransform *myTransform;
    ui::Button * menuEntry = nullptr;
};

struct RevitRoomInfo
{
    RevitRoomInfo(osg::Vec3 pos, std::string n, int id, int docID, double a);
    virtual ~RevitRoomInfo();

    void updateBillboard();
    const std::string& getName()const { return name; };

    std::string name;
    osg::Vec3 textPosition;
    coVRLabel* label;

    double area;
    int ID;
    int documentID;
    bool isActive = false;
};

class ElementInfo
{
public:
    ElementInfo();
    virtual ~ElementInfo();

    void addParameter(RevitParameter *p);

    std::string name;
    std::list<osg::Node *> nodes;
    std::list<RevitParameter *> parameters;
    int ID;
    int DocumentID;
    int createdPhase;
    int demolishedPhase;

private:
    ui::Group *group = nullptr;
    static int yPos;
};

struct AnnotationInfo
{
    std::string text;
    double x,y,z,h,p,r;
    int ID;
};

struct IKAxisInfo
{
    enum AxisType { Rot=0, Trans,Scale };

    IKAxisInfo();
    void initIK( unsigned int myID);

    osg::Matrix mat;
    osg::Matrix invMat;
    osg::Matrix sumMat;
    osg::Matrix invSumMat;
    osg::Vec3 origin;
    osg::Vec3 direction;
    double min;
    double max;
    osg::MatrixTransform* transform;
    osg::MatrixTransform* rotTransform;
    osg::MatrixTransform* scaleTransform;
    AxisType type= AxisType::Rot;

};

inline double getAngle(const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& rotAxis) // v1 and v2  need to be normalized
{
    osg::Vec3 tmp = v1 ^ v2;
    float sp = v1 * v2;
    if (sp > 1)
        sp = 1;
    if (sp < -1)
        sp = -1;
    if (tmp * rotAxis > 0)
        return (acos(sp));
    else
        return (-acos(sp));
}
class IKSensor;
struct IKInfo
{
    enum IKType { Rot = 0, RotTrans, Trans };
    IKInfo();
    ~IKInfo();
    void update();
    void addHandle(osg::Node *n);
    void intiIK();
    void updateIK(const osg::Vec3& targetPos, const osg::Vec3& targetDir);
    void updateGeometry();
    osg::Vec3 getPosition();
    osg::Vec3 getOrientation();

    std::vector<IKAxisInfo> axis;

    osg::Vec3 basePos;
    osg::Vec3 vA;
    osg::Vec3 vB;
    osg::Vec3 vC;

    IKSensor* iks=nullptr;
    CRobot* robot = nullptr;

    float rA, rB, rC;
    float initialAngleA, initialAngleB, initialAngleC;
    float initialLength;

    IKType type = IKType::Rot;
    int ID;
    int DocumentID;
};


class IKSensor : public coPickSensor
{
private:
    IKInfo* myIKI;
    RevitPlugin* revitPlugin;
    bool scheduleUnregister = false;;
public:
    IKSensor(RevitPlugin *r,IKInfo* a, osg::Node* n);
    ~IKSensor() override;
    vrui::coCombinedButtonInteraction* getInteraction() { return interaction; };

    void miss() override;
    int hit(vruiHit* hit) override;
    void update() override;

    // this method is called if intersection just started
    // and should be overloaded
    void activate() override;

    // should be overloaded, is called if intersection finishes
    void disactivate() override;
};

struct ARMarkerInfo
{
	ARMarkerInfo();
	void setValues(int ID,int docID, int MarkerID, std::string& name, double angle, double offset, osg::Matrix& mat, osg::Matrix& hostMat, int hostID, double size, std::string markerType);
	void update();

	osg::Matrix mat; // marker coordinates in mm in Revit coordinate system (object coordinates)
	osg::Matrix invMarker;
	osg::Matrix invHost;
	osg::Matrix MarkerToHost;
	osg::Matrix hostMat; // host transformation in feet in Revit coordinate system
	std::string name;
	std::string markerType;
	MarkerTrackingMarker* marker=nullptr;

	double offset;
	double angle;
	double size;
	double lastUpdate = 0.0;

	int ID;
    int DocumentID;
	int MarkerID;
	int hostID;
};

struct TextureInfo
{
    enum textureType  { diffuse,bump};

	TextureInfo(TokenBuffer &tb);
	std::string texturePath;
    osg::Image *image;

	double sx, sy, ox, oy, angle,amount;
    textureType type;
	int ID;
    bool requestTexture; // try to get texture from remote after the model has been transferred completely
	unsigned char r,g,b;
};

struct MaterialInfo
{
	MaterialInfo(TokenBuffer &tb);
    void updateTexture(TextureInfo::textureType type, osg::Image *image);
    osg::Image *createNormalMap(osg::Image *heightMap, double pStrength);

	TextureInfo *bumpTexture;
	TextureInfo *diffuseTexture;
	osg::ref_ptr<osg::StateSet> geoState;
	coVRShader *shader;

	int ID;
    int DocumentID;

	unsigned char r, g, b, a;
    bool isTransparent;
};

struct PhaseInfo
{
    ~PhaseInfo();
    std::string PhaseName;
    ui::Button* button=nullptr;
    int ID=0;
};


struct RevitParameter
{
    RevitParameter(int i, std::string n, int st, std::string pt, int num, ElementInfo *ele)
        : ID(i)
        , name(n)
        , StorageType(st)
        , ParameterType(pt)
        , number(num)
        , element(ele){};

    virtual ~RevitParameter();
    void createUI(ui::Group *group, int pos);

    // string: 64 bit  gcc + msvc 32 bytes; llvm 24 bytes stack
    std::string name;
    std::string ParameterType;
    std::string s;

    // ptr: 8 byte on 64 byte typically
    ElementInfo *element = nullptr;
    ui::Label *uiLabel = nullptr;
    ui::Element *uiElement = nullptr;

    // double: 8 byte
    double d;

    // int: 4 byte
    int ID;
    int number; // param number in Element;
    int StorageType;
    int ElementReferenceID;
    int i;
};

struct FamilyType
{
    FamilyType(TokenBuffer& tb);
    ~FamilyType();
    void createMenuEntry();
    void createFamilyLabel();

    std::string FamilyName;
    std::string Name;

    ui::Action* selectType = nullptr;
    ui::Label* FamilyLabel = nullptr;

    int ID;
    
};

struct ObjectParamater
{
    ObjectParamater(TokenBuffer& tb,int i);
    ~ObjectParamater();
    void createMenu();

    std::string Name;
    std::string Value;
    std::string ParameterType;

    ui::Label* Label = nullptr;

    double d;

    int num;
    int StorageType;
    int ElementReferenceID;
    int i;
};

struct ObjectInfo
{
    ObjectInfo(TokenBuffer& tb);
    ~ObjectInfo();

    std::vector<FamilyType*> types;
    std::vector<ObjectParamater*> parameters;

    std::string TypeName;
    std::string CategoryName;

    ui::Action* flipLR = nullptr;
    ui::Action* flipIO = nullptr;
    ui::Label* TypeNameLabel = nullptr;

    int TypeID;
    int flipInfo;
};

// connection to Autodesk Revit Architecture
class RevitPlugin : public coVRPlugin, public opencover::ui::Owner
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
		MSG_File = 527,
		MSG_Finished = 528,
		MSG_DocumentInfo = 529,
		MSG_NewPointCloud = 530,
		MSG_NewARMarker = 531,
        MSG_DesignOptionSets = 532,
        MSG_SelectDesignOption = 533,
        MSG_IKInfo = 534,
        MSG_Phases = 535,
        MSG_ViewPhase = 536,
        MSG_AddRoomInfo = 537,
        MSG_ObjectInfo = 538,
        MSG_Flip = 539,
        MSG_SelectType = 540,
        MSG_ElevatorPart = 541
    };

    enum ObjectTypes
    {
        OBJ_TYPE_Mesh = 1,
        OBJ_TYPE_Curve,
        OBJ_TYPE_Instance,
        OBJ_TYPE_Solid,
        OBJ_TYPE_RenderElement,
        OBJ_TYPE_PolyMesh,
		OBJ_TYPE_Inline
    };

    RevitPlugin();
    ~RevitPlugin() override;

    static RevitPlugin *instance()
    {
        return plugin;
    };

    // this will be called in PreFrame
	void preFrame() override;
    void key(int type, int keySym, int mod) override;
    void destroyMenu();
    void createMenu();
    void updateIK();
    void registerInteraction(IKInfo* i);
    void unregisterInteraction(IKInfo* i);
    void message(int toWhom, int type, int len, const void *buf) override;
    void deactivateAllViewpoints();
    void flip(int dir);
    void createNewAnnotation(int id, AnnotationMessage *am);
    void changeAnnotation(int id, AnnotationMessage *am);
    void setPhase(std::string phaseName);
    void setPhase(int phase);
    void deleteChildlessParent(osg::Group *parent);
    /// <summary>
    /// set visibility depending on current selected phase
    /// </summary>
    void setPhaseVisible(ElementInfo* ei);

    osg::Group* getCurrentGroup() { return currentGroup.top(); };

    int getAnnotationID(int revitID);
    int getRevitAnnotationID(int ai);

    bool isInteractionRunning();
    bool sendMessage(Message &m);
	bool update() override;
	bool checkDoors();
    bool init() override;

	std::list<DoorInfo *> doors;
    std::list<DoorInfo *> activeDoors;
    std::list<Elevator *> elevators;
	std::map<int, ARMarkerInfo*> ARMarkers;
    std::list<RevitDesignOptionSet*> designOptionSets;
    std::list<PhaseInfo*> phaseInfos;

    osg::Matrix startCompleteMat;
    osg::Matrix invStartCompleteMat;
    osg::Matrix startHand, invStartHand;
    osg::Matrix NorthRotMat;
    osg::Matrix RevitScale;
    osg::Matrix RevitGeoRefference;
    osg::Vec3 startPosition;
    osg::Vec3 startOrientation;

    ui::Menu *revitMenu = nullptr;
    ui::ButtonGroup* viewpointGroup = nullptr;
    ui::Menu* viewpointMenu = nullptr;
    ui::Menu* parameterMenu = nullptr;
    ui::Menu* phaseMenu = nullptr;
    ui::Menu* objectInfoMenu = nullptr;
    ui::Menu* highlightMenu = nullptr;
    ui::Menu* highlightSystemMenu = nullptr;
    ui::ButtonGroup* PhaseGroup = nullptr;
    ui::ButtonGroup* objectInfoGroup = nullptr;
    ui::Action* selectObject = nullptr;
    ui::Menu* typesMenu = nullptr;
    ui::ButtonGroup* TypesGroup=nullptr;
    ui::Menu* parametersMenu = nullptr;
    ui::ButtonGroup* ParametersGroup = nullptr;
    ui::Button* toggleRoomLabels = nullptr;

    vrui::coCombinedButtonInteraction *selectObjectInteraction = nullptr;

    IKInfo *currentIKI=nullptr;
    ObjectInfo* currentObjectInfo=nullptr;

    double TrueNorthAngle = 0.0;
    double ProjectHeight = 0.0;

    int maxEntryNumber;
    int currentPhase=0;
    int currentObjectID=-1;
    int currentDocumentID=-1;

protected:
    void handleMessage(Message *m);
	MaterialInfo * getMaterial(int revitID);
    osg::Image *readImage(std::string name);
    void setDefaultMaterial(osg::StateSet *geoState);
    void requestTexture(int matID, TextureInfo *texture);

    std::vector<RevitViewpointEntry*> viewpointEntries;
    std::vector<RevitRoomInfo*> roomInfos;
    std::vector<std::map<int, ElementInfo *>> ElementIDMap;
    std::vector<IKInfo*> ikInfos;
    std::vector<int> annotationIDs;
	std::map<int, MaterialInfo *> MaterialInfos;
	std::map<std::string, osg::ref_ptr<osg::Node>> inlineNodes;
    std::stack<osg::Group *> currentGroup;

    osg::Matrix invStartMoveMat;
    osg::Matrix lastMoveMat;

    std::string textureDir;
    std::string localTextureDir;
    std::string localTextureFile;
    std::string currentRevitFile;

    osg::ref_ptr<osg::Material> globalmtl;
    osg::ref_ptr<osg::MatrixTransform> revitGroup;
    std::unique_ptr<ServerConnection> toRevit = nullptr;

    static RevitPlugin *plugin;
    ui::Label *label1 = nullptr;
    ui::SelectionList* viewsCombo;
    ui::Menu *roomInfoMenu = nullptr;
    ui::Action *addCameraButton = nullptr;
    ui::Action *updateCameraButton = nullptr;

    ServerConnection *serverConn = nullptr;
    RevitInfo *info = nullptr;
    Message *msg = nullptr;

    float scaleFactor;

    int MovedID;
    int MovedDocumentID;

    bool MoveFinished;
    bool setViewpoint;
    bool firstDocument=true;
	bool ignoreDepthOnly = false;
};



#endif
