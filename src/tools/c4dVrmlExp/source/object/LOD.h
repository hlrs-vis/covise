#ifndef VRML_EXPLOD_H
#define VRML_EXPLOD_H

#include "../scenesaver/VRMLexp.h"
#include "olevelofdetail.h"

//Fuers Tag-Plugin
#include "lib_description.h"
#include "customgui_priority.h"

#define CONTAINER_FOR_DISTANCE_CONTAINER 1025476		//unique ID for the main-subcontaioner; vom Plugincafe da auf Top Level gespeichert
#define DISTANCE_CONTAINER 1026000 						//unique ID for storing the IDs of the sub-subcontainers; 
#define ID_LODOBJECT       1025475	 					//ID vom Plugincafe



///Eine von public ObjectData Abgeleitete Klasse. Zuständig für das Erstellen von LOD Objekten.
class LODObject : public ObjectData
{
	INSTANCEOF(LODObject,ObjectData)

public:
	//Object Plugin Methoden
	virtual Bool Draw(BaseObject *op, Int32 type, BaseDraw *bd, BaseDrawHelp *bh);
	virtual BaseObject* GetVirtualObjects(BaseObject *op, HierarchyHelp *hh);
	
	//Node Data Methoden
	virtual Bool Message(GeListNode *node, Int32 type, void *data);
	virtual Bool Init(GeListNode *node); 
	//Fuer den Parameter Generator Teil:
	virtual Bool GetDDescription(GeListNode *node, Description *description, DESCFLAGS_DESC &flags);
	virtual Bool GetDParameter(GeListNode *node, const DescID &id, GeData &t_data, DESCFLAGS_GET &flags);
	virtual Bool SetDParameter(GeListNode *node, const DescID &id, const GeData &t_data, DESCFLAGS_SET &flags);

	//Eigene Methoden
	Bool InitialiseList(GeListNode *node);						///Funktion um nach dem Laden eines Dokuments die Werte (Parameter) der LOD-Objekte von den BaseContainer in die lodKidsHelper Liste zu schreiben
	static Int32 ChildCnt(BaseObject *op);						///Zählt nur Kinderobjekte eines Objekts, keine KindesKinder

	//Für das Auschreiben in die Exportdatei: 
	static void WriteLODStart(class VRMLSAVE &vrml,class VRMLmgmt *dataMgmt);	   ///Schreibt den Anfang eines LOD-Knoten
	static void WriteEndLOD(VRMLSAVE &vrml);							   ///Schreibt das Ende eines LOD-Knoten

	static NodeData *Alloc(void) { return NewObjClear(LODObject); }

	//Public Attributes
	std::list<class LODmgmt*> lodKidsHelper;		///Liste mit LODmgmt Objekten, die Parameter des LOD-Objekts verwalten. (Distance, Center, usw)
	std::list<class LODmgmt*>::iterator it;			
	std::list<class LODmgmt*>::iterator ittemp;

	~LODObject(void);
};



///Zuständig für die Verwaltung der vom LOD-Objekt geforderten Parameter eines KindObjekts (Distance, name, usw). 
class LODmgmt
{
public :
	BaseObject* getBaseObject();
	void setSubContainer(BaseContainer *baseC);			///Subcontainer in dem die zum Kindobjekt passenden "Distance" -Wert gespeichert werden
	BaseContainer &getSubContainer();
	String getName();									///Name des KindObjekts
	Int32 getID();										///Liefert die ID des Objekts.Die ID wird benötigt um später die BaseContainer unter eindeutiger ID in den Haupt-BaseContainer zu schreiben

	LODmgmt(BaseObject *object, String opName, Int32 newID);		///Konstruktor
	~LODmgmt(void);												///Destruktor											
private:			
	BaseObject *mOp;
	BaseContainer mBc;
	String mName;
	Int32 mId;
};




#endif