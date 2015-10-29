#ifndef VRML_NAVIGATIONINFO_H
#define VRML_NAVIGATIONINFO_H

#include "VRMLexp.h"
#include "onavigationinfo.h"
#include "c4d_baseobject.h"

static Float myPow(Float r, Float s);


class NavigationInfoObject : public ObjectData
{
	INSTANCEOF(NavigationInfoObject,ObjectData)

public:
	virtual Bool Message(GeListNode *node, Int32 type, void *data);
	virtual Bool Init(GeListNode *node); 
	virtual BaseObject* GetVirtualObjects(BaseObject *op, HierarchyHelp *hh);

	static NodeData *Alloc(void) { return NewObjClear(NavigationInfoObject); }


	//Für das Auschreiben in die Exportdatei: 
	static void WriteNavigationInfo(class VRMLSAVE &vrml, class VRMLmgmt *dataMgmt); 

	~NavigationInfoObject(void);

};


#endif