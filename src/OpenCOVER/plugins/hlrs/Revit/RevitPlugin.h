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

#include <cover/ARToolKit.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Label.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Group.h>
#include <cover/ui/EditField.h>
#include <cover/ui/SelectionList.h>

#include <ik/ik.h>

/*
#include "rbdl/rbdl_mathutils.h"
#include "rbdl/Model.h"
#include "rbdl/Kinematics.h"
#include "rbdl/Dynamics.h"
*/

#define REVIT_FEET_TO_M 0.304799999536704
#define REVIT_M_TO_FEET 3.2808399

class RevitInfo : public vrui::vruiUserData
{
public:
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

class RevitDesignOption
{
public:
    RevitDesignOption(RevitDesignOptionSet *s);
    RevitDesignOptionSet *set;
    std::string name;
    int ID;
    bool visible;
};

class RevitDesignOptionSet
{
public:
    RevitDesignOptionSet();
    ~RevitDesignOptionSet();
    std::string name;
    int ID;
    int DocumentID;
    ui::SelectionList* designoptionsCombo=nullptr;
    std::list<RevitDesignOption> designOptions;
    void createSelectionList();
};

class RevitViewpointEntry
{
public:
    RevitViewpointEntry(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, RevitPlugin *plugin, std::string n, int id,int docID);
    virtual ~RevitViewpointEntry();
    
    void setValues(osg::Vec3 pos, osg::Vec3 dir, osg::Vec3 up, std::string n);
    void activate();
    void setActive(bool);
    void deactivate();
    
    void updateCamera();
    int entryNumber;
    int ID;
    int documentID;
    bool isActive = false;
    const std::string & getName()const { return name; };

private:
    std::string name;
    RevitPlugin *myPlugin = nullptr;
    osg::Vec3 eyePosition;
    osg::Vec3 viewDirection;
    osg::Vec3 upDirection;
    ui::Button * menuEntry = nullptr;
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
    int DocumentID;
    std::string name;

private:
    ui::Group *group = nullptr;
    static int yPos;
};
class AnnotationInfo
{
public:
    double x,y,z,h,p,r;
    std::string text;
    int ID;
};

class IKAxisInfo
{
public:
    IKAxisInfo();
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

   /* RigidBodyDynamics::Body body;
    RigidBodyDynamics::Joint joint;
    unsigned int body_id;

    void initIK(RigidBodyDynamics::Model* model, unsigned int parent_id);*/
    IKInfo* ikinfo=nullptr;
    ik_node_t* node;
    void initIK(IKInfo* iki, ik_node_t* parent, unsigned int myID);
};

class IKInfo
{
public:
    IKInfo(); 
    void IKInfo::intiIK();
    void updateIK();
    void updateGeometry();
    int ID;
    int DocumentID;
    std::vector<IKAxisInfo> axis;

    ik_node_t* base;
    ik_effector_t* effector;
    ik_solver_t* solver;

  /*  RigidBodyDynamics::Model* model;


    RigidBodyDynamics::Math::VectorNd Q;
    RigidBodyDynamics::Math::VectorNd QDot;
    RigidBodyDynamics::Math::VectorNd QDDot;
    RigidBodyDynamics::Math::VectorNd Tau;

    std::vector<unsigned int> body_ids;
    std::vector<RigidBodyDynamics::Math::Vector3d> body_points;
    std::vector<RigidBodyDynamics::Math::Vector3d> target_pos;*/
};

class ARMarkerInfo
{
public:
	ARMarkerInfo();
	ARToolKitMarker* marker=nullptr;

	osg::Matrix mat; // marker coordinates in mm in Revit coordinate system (object coordinates)
	osg::Matrix invMarker;
	osg::Matrix invHost;
	osg::Matrix MarkerToHost;
	osg::Matrix hostMat; // host transformation in feet in Revit coordinate system
	std::string name;
	std::string markerType;
	int ID;
    int DocumentID;
	int MarkerID;
	int hostID;
	double offset;
	double angle;
	double size;
	double lastUpdate = 0.0;
	void setValues(int ID,int docID, int MarkerID, std::string& name, double angle, double offset, osg::Matrix& mat, osg::Matrix& hostMat, int hostID, double size, std::string markerType);
	void update();
};

class TextureInfo
{
public:
	TextureInfo(TokenBuffer &tb);
	double sx, sy, ox, oy, angle,amount;
	std::string texturePath;
	unsigned char r,g,b;
    bool requestTexture; // try to get texture from remote after the model has been transferred completely
    enum textureType  { diffuse,bump};
    textureType type;
	int ID;
    osg::Image *image;
};

class MaterialInfo
{
public:
	MaterialInfo(TokenBuffer &tb);
	unsigned char r, g, b, a;
	TextureInfo *bumpTexture;
	TextureInfo *diffuseTexture;
	osg::ref_ptr<osg::StateSet> geoState;
	coVRShader *shader;
	int ID;
    int DocumentID;
    void updateTexture(TextureInfo::textureType type, osg::Image *image);
    osg::Image *createNormalMap(osg::Image *heightMap, double pStrength);
};


class DoorInfo
{
public:

    enum SlidingDirection { dirLeft=-1, dirNone=0,dirRight=1 };

	DoorInfo(int id, const char *Name, osg::MatrixTransform *tn, TokenBuffer &tb);
	std::string name;
	osg::MatrixTransform *transformNode;
	int ID;
	bool HandFlipped;
	bool FaceFlipped;
    SlidingDirection isSliding;
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


class RevitParameter
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
    void createUI(ui::Group *group, int pos);

    ui::Label *uiLabel = nullptr;
    ui::Element *uiElement = nullptr;

private:
};





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
        MSG_IKInfo = 534
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
    void updateIK();

    int maxEntryNumber;
    ui::Menu *revitMenu = nullptr;
    ui::ButtonGroup* viewpointGroup = nullptr;
    ui::Menu* viewpointMenu = nullptr;
    ui::Menu* parameterMenu = nullptr;
    ui::EditField* xPos;
    ui::EditField* yPos;
    ui::EditField* zPos;
    bool sendMessage(Message &m);
    
    void message(int toWhom, int type, int len, const void *buf);
    void deactivateAllViewpoints();
    int getAnnotationID(int revitID);
    int getRevitAnnotationID(int ai);
    void createNewAnnotation(int id, AnnotationMessage *am);
    void changeAnnotation(int id, AnnotationMessage *am);
	std::list<DoorInfo *> doors;
	std::list<DoorInfo*> activeDoors;
	std::map<int, ARMarkerInfo*> ARMarkers;
    std::list<RevitDesignOptionSet*> designOptionSets;
protected:
    static RevitPlugin *plugin;
    ui::Label *label1 = nullptr;
    ui::SelectionList* viewsCombo;
    ui::Menu *roomInfoMenu = nullptr;
    std::vector<RevitViewpointEntry *> viewpointEntries;
    ui::Action *addCameraButton = nullptr;
    ui::Action *updateCameraButton = nullptr;

	bool ignoreDepthOnly = false;

    ServerConnection *serverConn = nullptr;
    ServerConnection *toRevit = nullptr;
    void handleMessage(Message *m);

	MaterialInfo * getMaterial(int revitID);
    osg::Image *readImage(std::string name);


    void setDefaultMaterial(osg::StateSet *geoState);
    osg::ref_ptr<osg::Material> globalmtl;
    osg::ref_ptr<osg::MatrixTransform> revitGroup;
    std::stack<osg::Group *> currentGroup;
    std::vector<std::map<int, ElementInfo *>> ElementIDMap;
    std::vector<IKInfo*> ikInfos;
    osg::Matrix invStartMoveMat;
    osg::Matrix lastMoveMat;
    bool MoveFinished;
    int MovedID;
    int MovedDocumentID;
    RevitInfo  *info = nullptr;
    std::vector<int> annotationIDs;
	std::map<int, MaterialInfo *> MaterialInfos;
	std::map<std::string, osg::ref_ptr<osg::Node>> inlineNodes;
    void requestTexture(int matID, TextureInfo *texture);

	
    float scaleFactor;
    std::string textureDir;
    std::string localTextureDir;
    std::string localTextureFile;
    std::string currentRevitFile;
    bool setViewpoint;
    

    Message *msg = nullptr;

};
#endif
