//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Navigation Info Knoten 																										//
//																																//
//Fertiges Gerüst, allerdings wird bisher nur der Maßstab abgefragt und es findet noch keine Ausgabe in die Exportdatei statt.  //
//																																//
//																																//
//																																//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include "NavigationInfo.h"

Bool NavigationInfoObject::Init(GeListNode *node)
{	
	BaseContainer *data = ((BaseList2D*)node)->GetDataInstance(); if (!data) return FALSE;
	BaseObject* op = (BaseObject*)node;
	op->SetName("NavigationInfo-Object");
	BaseContainer wc = GetWorldContainer();
	data->SetInt32(IDC_TYPE, IDS_TYPE_WALK);
	data->SetBool(IDC_HEADLIGHT, TRUE);
	data->SetFloat(AVATAR_DISTANCE, 0);
	data->SetFloat(AVATAR_HEIGHT_TERRAIN, 0);
	data->SetFloat(AVATAR_HEIGHT_OBJECT, 0);
	data->SetFloat(IDC_VISIBILITY_LIMIT, 0);

	data->SetInt32(IDC_BASICUNITS, wc.GetInt32(WPREF_UNITS_BASIC));
	switch (data->GetInt32(IDC_BASICUNITS))
	{
	case IDS_PIXEL:
		data->SetFloat(SPEED, 1);
		break;
	case IDS_KILOMETER:
		data->SetFloat(SPEED, myPow(10, 3));
		break;
	case IDS_METER:
		data->SetFloat(SPEED, 1);
		break;
	case IDS_CENTIMETER:
		data->SetFloat(SPEED, myPow(10, -2));
		break;
	case IDS_MILLIMETER:
		data->SetFloat(SPEED,  myPow(10, -3));
		break;
	case IDS_MICROMETER:
		data->SetFloat(SPEED,  myPow(10, -6));
		break;
	case IDS_NANOMETER:
		data->SetFloat(SPEED, myPow(10, -9));
		break;
	case IDS_MILE:
		data->SetFloat(SPEED, 1.609344);
		break;
	case IDS_YARD:
		data->SetFloat(SPEED, 1.0936132983377077865266842);
		break;
	case IDS_FOOT:
		data->SetFloat(SPEED, 3.2808398950131233595800525);
		break;
	case IDS_INCH:
		data->SetFloat(SPEED, 39.3700787401574803149606299);
		break;
	}

	return TRUE;
}

static Float myPow(Float r, Float s)
{
	return Pow(r, s);
}

Bool NavigationInfoObject::Message(GeListNode *node, Int32 type, void *data)
{	
	//Hier kann noch eine Abfrage nach der entsprechenden Message implementiert werden, damits nicht jedes mal ausgeführt wird
	BaseContainer *objectContainer = ((BaseList2D*)node)->GetDataInstance(); if (!data) return FALSE;

	switch (objectContainer->GetInt32(IDC_BASICUNITS))
	{
	case IDS_PIXEL:
		objectContainer->SetFloat(SPEED, 1);
		break;
	case IDS_KILOMETER:
		objectContainer->SetFloat(SPEED, myPow(10, 3));
		break;
	case IDS_METER:
		objectContainer->SetFloat(SPEED, 1);
		break;
	case IDS_CENTIMETER:
		objectContainer->SetFloat(SPEED, myPow(10, -2));
		break;
	case IDS_MILLIMETER:
		objectContainer->SetFloat(SPEED,  myPow(10, -3));
		break;
	case IDS_MICROMETER:
		objectContainer->SetFloat(SPEED,  myPow(10, -6));  //Eigentlich sollte das SetLFloat sein, aber dafür gibts keine Funktion
		break;
	case IDS_NANOMETER:
		objectContainer->SetFloat(SPEED, myPow(10, -9));	  //Eigentlich sollte das SetLFloat sein, aber dafür gibts keine Funktion
		break;
	case IDS_MILE:
		objectContainer->SetFloat(SPEED, 1.609344);
		break;
	case IDS_YARD:
		objectContainer->SetFloat(SPEED, 1.0936132983377077865266842);
		break;
	case IDS_FOOT:
		objectContainer->SetFloat(SPEED, 3.2808398950131233595800525);
		break;
	case IDS_INCH:
		objectContainer->SetFloat(SPEED, 39.3700787401574803149606299);
		break;
	}

	return TRUE;
}

BaseObject* NavigationInfoObject::GetVirtualObjects(BaseObject *op, HierarchyHelp *hh)
{
	BaseObject *ret = NULL;

	Bool dirty = op->CheckCache(hh) || op->IsDirty(DIRTYFLAGS_DATA);
	if (!dirty) return op->GetCache(hh);

	ret = PolygonObject::Alloc(0,0);
	if (!ret) goto Error;

	BaseContainer *main = ret->GetDataInstance();
	BaseContainer *objectContainer = op->GetDataInstance();
	
	//Kopiere Daten in das neu erzeugte Polygonobjekt, dass C4D zurückgegeben wird
	ret->SetName(op->GetName());
	main->SetBool(I_M_A_NAVIGATIONINFO_OBJECT,TRUE);
	main->SetInt32(IDC_BASICUNITS, objectContainer->GetInt32(IDC_BASICUNITS));
	main->SetInt32(IDC_TYPE, objectContainer->GetInt32(IDC_TYPE));
	main->SetInt32(IDC_BASICUNITS, objectContainer->GetInt32(IDC_BASICUNITS));
	main->SetBool(IDC_HEADLIGHT, objectContainer->GetBool(IDC_HEADLIGHT));
	main->SetFloat(SPEED, objectContainer->GetFloat(SPEED));
	main->SetFloat(AVATAR_DISTANCE, objectContainer->GetFloat(AVATAR_DISTANCE));
	main->SetFloat(AVATAR_HEIGHT_TERRAIN, objectContainer->GetFloat(AVATAR_HEIGHT_TERRAIN));
	main->SetFloat(AVATAR_HEIGHT_OBJECT, objectContainer->GetFloat(AVATAR_HEIGHT_OBJECT));
	main->SetFloat(IDC_VISIBILITY_LIMIT, objectContainer->GetFloat(IDC_VISIBILITY_LIMIT));


	return ret;

Error:
	blDelete(ret);
	return NULL;
}


void NavigationInfoObject::WriteNavigationInfo(class VRMLSAVE &vrml,class VRMLmgmt *dataMgmt)
{
	BaseContainer *main = dataMgmt->getOp()->GetDataInstance();

	String headlight_S;
	Bool headlight = main->GetBool(IDC_HEADLIGHT);
	if (headlight) {headlight_S = String("TRUE");}
	else {headlight_S = String("FALSE");}
	
	String type_S;
	switch (main->GetInt32(IDC_TYPE))
	{
	case IDS_TYPE_ANY:
		type_S = String("ANY");
		break;
	case IDS_TYPE_WALK:
		type_S = String("WALK");
		break;
	case IDS_TYPE_EXAMINE:
		type_S = String("EXAMINE");
		break;
	case IDS_TYPE_FLY:
		type_S = String("FLY");
		break;
	case IDS_TYPE_NONE:
		type_S = String("NONE");
		break;
	}

	vrml.writeC4DString("NavigationInfo{ \n");
	vrml.increaseLevel();
	vrml.writeC4DString("avatarSize [ " +String::FloatToString(main->GetFloat(AVATAR_DISTANCE)) + ", " +String::FloatToString(main->GetFloat(AVATAR_HEIGHT_TERRAIN)) + ", " +String::FloatToString(main->GetFloat(AVATAR_HEIGHT_OBJECT)) +"]\n");
	vrml.writeC4DString("headlight " +headlight_S +"\n");
	vrml.writeC4DString("speed " +String::FloatToString(main->GetFloat(SPEED)) +"\n");
	vrml.writeC4DString("type [\"" +type_S +"\"]\n");
	vrml.writeC4DString("visibilityLimit " +String::FloatToString(main->GetFloat(IDC_VISIBILITY_LIMIT)) +"\n");
	vrml.decreaseLevel();
	vrml.writeC4DString("}\n");	
}


NavigationInfoObject::~NavigationInfoObject(){}



// be sure to use a unique ID obtained from www.plugincafe.com
//#define ID_LODOBJECT 1025475  //ID vom Plugincafe

Bool RegisterNavigationInfo(void)
{
	// decide by name if the plugin shall be registered - just for user convenience
	String name="VRMLExporter-NavigationInfo"; 
	if (!name.Content()) return TRUE;
	if(RegisterObjectPlugin(1025474,name,OBJECT_GENERATOR,NavigationInfoObject::Alloc,"onavigationinfo",AutoBitmap("navinfo.tif"),0)) 
	{return TRUE;}
	return FALSE;
}

