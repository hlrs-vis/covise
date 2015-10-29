//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LOD Object																													//
//																																//
//In diesem Plugin wird ein LOD-Object erzeugt. Vorlage dafür ist das Circle.cpp Bsp. aus der SDK. 								//
//																																//
//Die Abstände (Distanzwerte) eines KindObjekts zu diesem Objekt können im Attributemanager eingestellt werden. 				//
//Für jedes Kindobjekt gibt es einen BaseContainer in dem der Distanzwert (DISTANCE) gespeichert ist. 							//
//Die BaseContainer mit den Distanzwerten befinden sich alle im BaseContainer CONTAINER_FOR_DISTANCE_CONTAINER, der 			//
//wiederum dem ObjektContainer untergeordnet ist. 																				//
//																																//
//Im Attribute Manager werden die Distanzwerte so angezeigt wie sie im BaseContainer CONTAINER_FOR_DISTANCE_CONTAINER			//
//angeordnet sind. 																												//
//																																//
//Für eine Reihnefolgenkorrekte anordnung sorgt die Liste lodKidsHelper.  														//
//In ihr wird für jedes KindObjekt eine Instanz der Klasse LODMgmt geführt. Diese hat unter anderem auch einen entsprechenden 	//
//Distanz BaseContainer. Dieser BaseContainer wird dann in GetDDescription() dem BaseContainer CONTAINER_FOR_DISTANCE_CONTAINER	//
//(nach voherigem löschen aller darin befindlicher BaseContainer) in richtiger Reihenfolge untergeordnet. 						//
//																																//
//Beim Speichern und anschließendem Laden einer Szene geht allerdings die Liste verloren. Daher wird in diesem Fall 			//
//beim Laden die Methode InitialiseList() aufgerufen, die lodKidsHelper Liste wieder anlegt und mit den entsprechenden			//
//Distanz BaseContainern befüllt. 																								//
//																																//
//																																//
//Da beim Exportieren durch doc->Polygonize() der BaseContainer CONTAINER_FOR_DISTANCE_CONTAINER verloren geht 					//
//wird der LOD Tag eingeführt. Er wird bevor das LOD Objekt in GetVirtualObjects() an Cinema 4D übergeben wird 					//
//dem Rückgabeobjekt zugewiesen. Ihm wird dann der BaseContainer CONTAINER_FOR_DISTANCE_CONTAINER mit all seinen 				//
//Distanz-BaseContainer untergeordnet. Dieser bleibt dann auch nach doc->Polygonize() erhalten.									//
//																																//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


#include "LOD.h"


Bool LODObject::Init(GeListNode *node)
{	
	BaseContainer *data = ((BaseList2D*)node)->GetDataInstance(); if (!data) return FALSE;
	BaseContainer main;
	data->InsData(CONTAINER_FOR_DISTANCE_CONTAINER, main);        //insert the main-subcontainer
	data->SetBool(LOD_OBJECT_CIRCLES, TRUE);
	BaseObject* op = (BaseObject*)node;
	op->SetName("LOD-Object");
	return TRUE;
}

Bool LODObject::InitialiseList(GeListNode *node)
{
	//Diese Funktion ist da um die Sub-BaseContainer vom main BaseContainer in die lodKidsHelper-Liste zu schreiben, 
	//wenn das Dokument geladen wird. Nach dem Laden sind die eingestellten Parameter (DISTANCE)
	//noch in den ganzen main SubContainer vorhanden, aber halt nicht in der Liste. 

	BaseContainer *bc = ((BaseList2D*)node)->GetDataInstance(); if (!bc) return FALSE;            //laedt den HauptContainer
	BaseContainer *main  = bc->GetContainerInstance(CONTAINER_FOR_DISTANCE_CONTAINER); if (!main) return FALSE;    //laedt den Haupt-SubContainer

	BrowseContainer brw(main);  
	GeData *data = NULL;
	Int32 id=0;

	ittemp = lodKidsHelper.end();
	--ittemp;

	while (brw.GetNext(&id,&data))
	{
		if (data->GetType() != DA_CONTAINER) continue;
		BaseContainer *object = data->GetContainer(); if (!object) continue;    //get the sub-subcontainer 
		(*ittemp)->getSubContainer().SetFloat(DISTANCE,object->GetFloat(DISTANCE));
		if(ittemp != lodKidsHelper.begin())	--ittemp;
	}
	return true;
}


Bool LODObject::Message(GeListNode *node, Int32 type, void *data)
{
	Bool changed = FALSE;
	BaseObject		*obj   = (BaseObject*)node;

	//if (type==MSG_MENUPREPARE)
	//{
	//	BaseDocument *doc = (BaseDocument*)data;
	//	((BaseObject*)node)->GetDataInstance()->SetInt32(PRIM_PLANE,doc->GetSplinePlane());
	//}	

	Int32 kidsCnt = ChildCnt(obj);  //Wieviele Kinder hat der LOD Knoten? 
	((BaseObject*)node)->GetDataInstance()->SetInt32(LOD_KIDS,kidsCnt);

	Int32 lodKidCnt = lodKidsHelper.size();
	if (lodKidCnt != kidsCnt) //Wenn sich die Anzahl geaendert hat mache:   Wenn ein Kind dazukam: kidsCnt>lodKidCnt  ;wenn eins entfernt wurde lodKidCnt>kidsCnt  :Das nacher Nutzen um Container und REAL wieder zu entfernen 
	{
		changed = TRUE;
		DescriptionCommand *dc = (DescriptionCommand*)data;

		if(changed)	
		{
			//Wenn sich die Kinder veraendert haben: 
			//-->Hat sich erhoeht: dem Container des neuen Objects wird ein  object.SetFloat hinzugefuegt und dann in den Hauptcontainer sortiert
			//-->Hat sich veringert: der Container des fehlenden Objects wird aus dem Hauptcontainer geloescht
			//-->Hat sich veraendert ... 

			BaseList2D *op = (BaseList2D*)node;
			BaseDocument *doc = op->GetDocument(); if (!doc) return FALSE;
			BaseContainer *bc = ((BaseList2D*)node)->GetDataInstance(); if (!bc) return FALSE;			//hole den Haupt BaseContainer
			BaseContainer *main  = bc->GetContainerInstance(CONTAINER_FOR_DISTANCE_CONTAINER); if (!main) return FALSE;
			doc->StartUndo();

			BaseObject *workingObject = obj;
			workingObject=workingObject->GetDown();	//Geh zum ersten Kind

			//KindesKinder werden nicht beruecksichtigt. 
			if (kidsCnt>lodKidCnt)	//Wenn ein Kindsobjekt hinzugefuegt wurde
			{
				doc->AddUndo(UNDOTYPE_CHANGE_SMALL, node);

				it = lodKidsHelper.begin();  
				while (workingObject) //Keine KindesKinder 
				{
					if ( it ==lodKidsHelper.end() || workingObject != (*it)->getBaseObject())	//wenn das Object nicht an der Position in der Liste ist schreibe es rein
					{
						//Hier fuer jedes neue KinderObjekt eine neue ID vergeben. Dazu Liste durchlaufen und die erste ID nehmen die nicht vergeben ist. 
						Int32 newId = 0;
						for ( Int32 j=0; j<kidsCnt; j++) 
						{
							Bool found = FALSE;
							for(ittemp = lodKidsHelper.begin(); ittemp!=lodKidsHelper.end(); ++ittemp)
							{
								if ((j+DISTANCE_CONTAINER) != (*ittemp)->getID())
								{
									found = FALSE;
								}
								else
								{ 
									found = TRUE;
									break; 
								}
							}			
							if (!found||lodKidsHelper.empty())  //wenn die j in der Liste nicht gefunden wurde ist dies die neue ID
							{
								newId = j + DISTANCE_CONTAINER;
								break;
							}
						}
						LODmgmt *lodData = new LODmgmt(workingObject,workingObject->GetName(), newId);  //WO loeschen? 
						lodData->getSubContainer().SetFloat(DISTANCE, 0.0);
						lodKidsHelper.insert(it,lodData);
						workingObject=workingObject->GetNext();		//Geh zum  naechsten Objekt
					}
					else
					{
						workingObject=workingObject->GetNext();		//Geh zum  naechsten Objekt
						++it;										// Geh zum naechsten ListenObjekt
					}
				}//while (workingObject)
				doc->EndUndo();   //start und endUndo nacher auch fuer den kind entfernt fall
			}//if (kidsCnt>lodKidCnt)

			//Ein KindObjekt wurde entfernt
			else 
			{
				doc->AddUndo(UNDOTYPE_CHANGE_SMALL, node);
				for (it = lodKidsHelper.begin();it != lodKidsHelper.end(); ++it)
				{
					if ( workingObject != (*it)->getBaseObject())	//wenn das Object nicht an der Position in der Liste ist loesche es raus
					{
						//Den BaseContainer muss ich wahrscheinlich nicht erst Flushen versuchs aber mal, da manchmal geloeschte Werte wieder auftauchen
						(*it)->getSubContainer().FlushAll();   //vielleicht noch delete anstatt Flush?
						//Test Ende
						lodKidsHelper.erase(it);
						break;
					}
					if(workingObject)
						workingObject=workingObject->GetNext();
				}
				doc->EndUndo();   //start und endUndo nacher auch fuer den kind entfernt fall
			}//end else
			changed= FALSE;
		} 
		lodKidCnt = kidsCnt; //setzte lodKidCnt = kidsCnt, damit beim naechsten mal der Kinderzahl unterschied zum jetzigen zustand erkannt wird	
	} //Wenn sich die Anzahl der Kinder geaendert hat

	//Wenn eine Szene geladen wird muessen die BaseContainer von main in die Liste geschrieben werden. 
	//Messages bei LOAD die sich von SAVE unterscheiden:
	//#define MSG_GETCUSTOMICON	1001090	
	//und 13. 
	//	Was nicht kommt aber wahrscheinlich angebracht waere: #define MSG_DOCUMENTINFO_TYPE_LOAD		1001

	if (type==13)
	{
		InitialiseList(node);  //Das soll nur ausgefuehrt werden, wenn eine Szene geladen wird...
	}
	return     SUPER::Message(node,type,data);
}



Bool LODObject::GetDDescription(GeListNode *node, Description *description, DESCFLAGS_DESC &flags)
{

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//	Beschreibung
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	it = lodKidsHelper.end();
	if (lodKidsHelper.begin() != lodKidsHelper.end()){--it;} // Zeig auf das letzte Objekt
	ittemp = lodKidsHelper.begin();

	if (!description->LoadDescription(node->GetType())) return FALSE;

	// important to check for speedup c4d!
	const DescID *singleid = description->GetSingleDescID();

	BaseContainer *bc = ((BaseList2D*)node)->GetDataInstance(); if (!bc) return FALSE;            //laedt den HauptContainer
	BaseContainer *main  = bc->GetContainerInstance(CONTAINER_FOR_DISTANCE_CONTAINER); if (!main) return FALSE;    //laedt den Haupt-SubContainer
	BaseDocument *doc = ((BaseList2D*)node)->GetDocument();

	Bool initbc2 = FALSE;
	Bool initbc3 = FALSE;
	BaseContainer  bc2, bc3;

	//loesch alle Subcontainer im main BaseContainer			
	for ( unsigned int i = 0; i < lodKidsHelper.size(); i++)
	{
		main->RemoveData(i+DISTANCE_CONTAINER);
	}

	//Durchlauf die lodKidsHelperListe und schreib fuer jedes Objekt den BaseContainer zur passenden ID in den main Container
	if (!lodKidsHelper.empty())
	{
		ittemp = lodKidsHelper.end();
		--ittemp;

		while((*ittemp)->getID() >= DISTANCE_CONTAINER)
		{
			main->InsData((*ittemp)->getID(), (*ittemp)->getSubContainer()); 
			if (ittemp == lodKidsHelper.begin())break;
			--ittemp;
		}
	}

	BrowseContainer brw(main);  
	GeData *data = NULL;
	Int32 id=0;

	for (ittemp = lodKidsHelper.begin(); ittemp!= lodKidsHelper.end();++ittemp)
	{
		BrowseContainer brw(main);
		while (brw.GetNext(&id,&data))
		{
			if(id == (*ittemp)->getID())
			{
				break;
			}
		}

		if (data->GetType() != DA_CONTAINER) continue;
		BaseContainer *object = data->GetContainer(); if (!object) continue;    //get the sub-subcontainer 

		//read out the sub-subcontainer and accordingly add the elements to the description
		//DISTANCE and SEPARATOR are defined in the header file
		DescID cid = DescID(DescLevel(SEPARATOR,DTYPE_SUBCONTAINER,0), DescLevel(id,DTYPE_SEPARATOR,0));  //DescLevel(Int32 t_id, Int32 t_datatype, Int32 t_creator); //DescLevel Represents a level within a DescID.
		//DescLevel An ID class for description elements. Contains a stack of DescLevel objects. The description ID is used to exactly identify a parameter value.
		if (!singleid || cid.IsPartOf(*singleid,NULL)) // important to check for speedup c4d!
		{
			if (!initbc3)
			{
				initbc3 = TRUE;
				bc3 = GetCustomDataTypeDefault(DTYPE_SEPARATOR);
				bc3.SetInt32(DESC_CUSTOMGUI, CUSTOMGUI_SEPARATOR);
				bc3.SetBool(DESC_SEPARATORLINE, TRUE);
				bc3.SetInt32(DESC_ANIMATE, DESC_ANIMATE_OFF);
				bc3.SetBool(DESC_REMOVEABLE, FALSE);
			}
			bc3.SetString(DESC_NAME, "");
			bc3.SetString(DESC_SHORT_NAME, "");
			if (!description->SetParameter(cid,bc3,DescLevel(ID_OBJECTPROPERTIES))) return FALSE;
		}
		cid = DescID(DescLevel(DISTANCE,DTYPE_SUBCONTAINER,0), DescLevel(id,DTYPE_REAL,0));

		if (!singleid || cid.IsPartOf(*singleid,NULL)) // important to check for speedup c4d!
		{
			if (!initbc2)		//Nur fuer das Neue KindObjekt soll ein neues Feld hinzugefuegt werden
			{
				initbc2 = TRUE;
				bc2 = GetCustomDataTypeDefault(DTYPE_REAL);
				bc2.SetInt32(DESC_CUSTOMGUI, CUSTOMGUI_REALSLIDER);
				bc2.SetFloat(DESC_MIN,0.0);
				bc2.SetFloat(DESC_MAX,10000.0);  //Maximum mal auf 10.000 meter
				bc2.SetFloat(DESC_STEP,1.0);
				bc2.SetInt32(DESC_UNIT,DESC_UNIT_METER);
				bc2.SetInt32(DESC_ANIMATE, DESC_ANIMATE_ON);
				bc2.SetBool(DESC_REMOVEABLE, FALSE);
			}
			bc2.SetString(DESC_NAME, (*ittemp)->getName());
			if (!description->SetParameter(cid,bc2,DescLevel(ID_OBJECTPROPERTIES))) return FALSE;
		}
	}//for
	flags |= DESCFLAGS_DESC_LOADED;	//Set if elements have been added to the description, either by loading or manual addition.
	return SUPER::GetDDescription(node,description,flags);
}


BaseObject* LODObject::GetVirtualObjects(BaseObject *op, HierarchyHelp *hh)
{
	BaseObject *ret = NULL;

	Bool dirty = op->CheckCache(hh) || op->IsDirty(DIRTYFLAGS_DATA);
	if (!dirty) return op->GetCache(hh);

	ret = PolygonObject::Alloc(0,0);
	if (!ret) goto Error;

	//Kopiere alle wichtigen Werte in das PolygonObjekt das C4D zurückgegeben wird: 
	ret->SetName(op->GetName());
	BaseContainer *bc = ret->GetDataInstance();
	bc->SetBool(I_M_A_LOD_OBJECT, TRUE);
	bc->SetFloat(LOD_CENTER_X, op->GetDataInstance()->GetFloat(LOD_CENTER_X));
	bc->SetFloat(LOD_CENTER_Y, op->GetDataInstance()->GetFloat(LOD_CENTER_Y));
	bc->SetFloat(LOD_CENTER_Z, op->GetDataInstance()->GetFloat(LOD_CENTER_Z));
	BaseContainer *main = op->GetDataInstance()->GetContainerInstance(CONTAINER_FOR_DISTANCE_CONTAINER);
	ret->GetDataInstance()->SetContainer(CONTAINER_FOR_DISTANCE_CONTAINER, *main);

	return ret;

Error:
	blDelete(ret);
	return NULL;
}


static BaseContainer *GetPose(GeListNode *node, Int32 id)	//bekomm die Adresse des Sub-Containers
{
	BaseContainer *bc = ((BaseObject*)node)->GetDataInstance(); if (!bc) return NULL;
	BaseContainer *main  = bc->GetContainerInstance(CONTAINER_FOR_DISTANCE_CONTAINER); if (!main) return NULL;

	if (main->GetType(id)!=DA_CONTAINER) return NULL;
	return main->GetContainerInstance(id);
}

Bool LODObject::GetDParameter(GeListNode* node, const DescID& id, GeData& t_data, DESCFLAGS_GET& flags)
{    
	BaseContainer *pose=NULL;

	switch (id[0].id)
	{
	case DISTANCE:
		{
			pose = GetPose(node,id[1].id); if (!pose) return FALSE;
			t_data = pose->GetFloat(DISTANCE); flags |= DESCFLAGS_GET_PARAM_GET;
		}
		break;
	}

	return SUPER::GetDParameter(node,id,t_data,flags);

}


Bool LODObject::SetDParameter(GeListNode *node, const DescID &id, const GeData &t_data, DESCFLAGS_SET &flags)
{
	BaseContainer *pose=NULL;

	switch (id[0].id)
	{
	case DISTANCE:
		{
			pose = GetPose(node,id[1].id); if (!pose) return FALSE;
			pose->SetFloat(DISTANCE,t_data.GetFloat()); flags |= DESCFLAGS_SET_PARAM_SET;  //Schreib in den SUBCONTAINER den Wert in DISTANCE

			//Versuch den BaseContainer (mit dem eingestellten Wert) in die lodKidsHelper Liste zu schreiben
			for (it = lodKidsHelper.begin(); it != lodKidsHelper.end();++it)
			{
				if ((*it)->getID() == id[1].id)
				{
					(*it)->setSubContainer(pose);
					break;
				}
			}
		}
		break;
	}
	return SUPER::SetDParameter(node,id,t_data,flags);
}


Bool LODObject::Draw(BaseObject *op, Int32 type, BaseDraw *bd, BaseDrawHelp *bh)
{
	//Hier drin werden die Distance Kreise gezeichnet
	BaseContainer *bc = op->GetDataInstance(); if (!bc) return FALSE;

	if(bc->GetBool(LOD_OBJECT_CIRCLES))
	{

		BaseContainer *main  = bc->GetContainerInstance(CONTAINER_FOR_DISTANCE_CONTAINER); if (!main) return FALSE;
		Int32 i = 0;
		Int32 kidsCnt = ChildCnt(op);
		Float *distances = new Float[kidsCnt];
		
		BrowseContainer brw(main);
		GeData *data = NULL;
		Int32 id;
		while (brw.GetNext(&id,&data))    //browse through the main-subcontainer
		{
			if (data->GetType() != DA_CONTAINER) continue;
			BaseContainer *object = data->GetContainer(); if (!object) continue;    //get the sub-subcontainer
			distances[i] = object->GetFloat(DISTANCE);
			i++;
		}
		for(Int32 i = 0; i<kidsCnt;i++)
		{
			Matrix ml = op->GetMl();
			ml.v1.x = distances[i];
			ml.v2.y = distances[i];
			//GePrint("ml.off.x " +String::IntToString(ml.off.x));
			//GePrint("ml.off.y " +String::IntToString(ml.off.y));
			//GePrint("ml.off.z " +String::IntToString(ml.off.z));
			//GePrint("ml.v1.x " +String::IntToString(ml.v1.x));
			//GePrint("ml.v1.y " +String::IntToString(ml.v1.y));
			//GePrint("ml.v1.z " +String::IntToString(ml.v1.z));
			//GePrint("ml.v2.x " +String::IntToString(ml.v2.x));
			//GePrint("ml.v2.y " +String::IntToString(ml.v2.y));
			//GePrint("ml.v2.z " +String::IntToString(ml.v2.z));
			//GePrint("ml.v3.x " +String::IntToString(ml.v3.x));
			//GePrint("ml.v3.y " +String::IntToString(ml.v3.y));
			//GePrint("ml.v3.z " +String::IntToString(ml.v3.z));
			bd->DrawCircle(ml);
			ml.v2.y = 0;
			ml.v2.z =  distances[i];
			bd->DrawCircle(ml);
		}
		delete[] distances;
	}
	//return SUPER::Draw(op, type, bd, bh);
	return TRUE;
}


Int32 LODObject::ChildCnt(BaseObject *op)
{	
	Int32 kidsCnt=0;
	op = op->GetDown();
	while (op)
	{
		op=op->GetNext();
		kidsCnt++;
	}
	return kidsCnt;
}


LODObject::~LODObject(){}



LODmgmt::LODmgmt(BaseObject *object, String opName, Int32 newID)
{
	mOp = object;
	mName = opName;
	mId = newID;
}
LODmgmt::~LODmgmt(){}
BaseObject* LODmgmt::getBaseObject(){return mOp;}
BaseContainer &LODmgmt::getSubContainer(){return mBc;}
String LODmgmt::getName(){return mName;}
Int32 LODmgmt::getID(){return mId;}
void LODmgmt::setSubContainer(BaseContainer *baseC){mBc = *baseC;}  



void LODObject::WriteLODStart(VRMLSAVE &vrml, VRMLmgmt *dataMgmt)
{
	//////////////////////////////////////////////////////////////////////////////////////////
	// Schreibt den Header des LOD -Objekts raus:											//
	//	DEF Name LOD{																		//
	//	center X Y Z																		//
	//	range [ x , x , ... ]																//
	//////////////////////////////////////////////////////////////////////////////////////////
	
	//WriteTransform(vrml,dataMgmt);

	// Ermitteln der Distanz- und Centerwerte: 

	//Anzahl der Kinder des LOD-Objekts: 
	BaseObject *op = dataMgmt->getOp();
	String knotennameC4D = dataMgmt->getKnotenName();
	//Int32 kidsCnt = op->GetDataInstance()->GetInt32(LOD_KIDS);
	Int32 kidsCnt = ChildCnt(op);

	//Hohl die DistanzWerte und schreib sie in das DistanzWerte Array
	//Int32 *DistanzWerte = new Int32[kidsCnt]; 
	//Float *CenterWerte = new Float[2];
	Int32 *distanzWerte = NULL; 
	Float *centerWerte = NULL;
	////distanzWerte = (Int32*) GeAlloc(sizeof(Int32)*kidsCnt);
	////centerWerte = (Float*) GeAlloc(sizeof(Float)*2);
	distanzWerte = NewMem(Int32,sizeof(Int32)*kidsCnt);
	centerWerte = NewMem(Float,sizeof(Float)*2);

	BaseContainer *bc = op->GetDataInstance();
	centerWerte[2]=bc->GetFloat(LOD_CENTER_X);
	centerWerte[1]=bc->GetFloat(LOD_CENTER_Y);
	centerWerte[0]=bc->GetFloat(LOD_CENTER_Z);


	BaseContainer *main  = bc->GetContainerInstance(CONTAINER_FOR_DISTANCE_CONTAINER); 
	BrowseContainer brw(main);
	GeData *data = NULL;
	Int32 id;
	Int32 i = (kidsCnt-1);
	while (brw.GetNext(&id,&data))   
	{
		if (data->GetType() != DA_CONTAINER) continue;
		BaseContainer *object = data->GetContainer(); if (!object) continue;  
		distanzWerte[i] = object->GetFloat(DISTANCE);
		i--;
	}			


	//Ausschreiben des Headers: 
	vrml.writeC4DString("DEF "+knotennameC4D +" LOD{\n");
	vrml.writeC4DString("center ");
		for (Int32 i=0;i<3;i++)
		{
			vrml.noIndent();
			vrml.writeC4DString(String::FloatToString(centerWerte[i]) +" ");
		}
	vrml.writeC4DString("\n"); 
			
	vrml.writeC4DString("range [ ");
		for (Int32 i = 0; i < (kidsCnt-1) ; i++)	//Letztes Element der Range muß nicht ausgegeben werden. 
		{
			vrml.noIndent();
			vrml.writeC4DString(String::IntToString(distanzWerte[i]));
			if (i != (kidsCnt-2))
			{
				vrml.noIndent();
				vrml.writeC4DString(" ,");
			}
		}
	vrml.noIndent();
	vrml.writeC4DString(" ]\n");
	vrml.writeC4DString("level [");

	//Eigentlich sollten die beiden Arrays wieder freigegeben werden, das führt hier aber zu Fehlern 
	//bei Szenen mit meheren LOD - Objekten + Kinderobjekten
	//GeFree(distanzWerte);
	//GeFree(centerWerte);
	//delete [] centerWerte; 
	//delete [] distanzWerte;
}
void LODObject::WriteEndLOD(VRMLSAVE &vrml)
{
	vrml.writeC4DString("]\n");
	vrml.writeC4DString("}\n");
	//WriteTransformEnd(vrml);
}



// be sure to use a unique ID obtained from www.plugincafe.com
//#define ID_LODOBJECT 1025475  //ID vom Plugincafe

Bool RegisterLOD(void)
{
	// decide by name if the plugin shall be registered - just for user convenience
	String name="VRMLExporter-LOD"; 
	if (!name.Content()) return TRUE;
	if(RegisterObjectPlugin(ID_LODOBJECT,name,OBJECT_GENERATOR,LODObject::Alloc,"Olevelofdetail",AutoBitmap("lod.tif"),0))  //spaeter noch auf Olod umstellen
	{return TRUE;}
	return FALSE;
}


