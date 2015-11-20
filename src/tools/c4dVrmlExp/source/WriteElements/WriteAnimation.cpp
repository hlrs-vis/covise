#include "WriteAnimation.h"


//Aufgebaut wie die WriteObject()
void WriteAnimation(VRMLSAVE &vrml, BaseDocument *doc, BaseObject *op, Bool parentEditorMode, Bool parentRenderMode, Bool timeSensorWriten)
{
	while (op)
	{
		Bool currentEditorMode, currentRenderMode;
		Matrix mg;
		currentEditorMode = parentEditorMode;	
		currentRenderMode = parentRenderMode;

		//Kreeire eindeutigen Objektname indem an ein Objekt eine aufsteigende Nummer gehaengt wird
		String knotennameC4D = VRMLName(op->GetName());
		Int32 count = 0; 
		String knotennametmp = knotennameC4D;
		while(vrml.objnameHashSet.find(StringC4DtoStringSTD(knotennametmp)) != vrml.objnameHashSet.end())
		{
			knotennametmp = String(knotennameC4D +"-" +String::IntToString(count));
			count++;
		}
		knotennameC4D = knotennametmp;
		vrml.objnameHashSet.insert(StringC4DtoStringSTD(knotennameC4D));
		vrml.writeC4DString("\n");

		//Dialog option, wann exportiert werden soll und wann nicht. Bei Editor und/oder Render deaktivert?
		//Wenn aktiviert und Editor Modus an oder Wenn aktiviert und Editor Modus Undef aber parentEditorModus TRUE
		Bool WriteThisObject = FALSE;

		Int32 visibleInEditor = op->GetEditorMode();	
		Int32 visibleInRender = op->GetRenderMode();

		if(vrml.getStartDialog()->getDLGvisibilityEditor()){visibleInEditor = OBJECT_ON;}
		if(vrml.getStartDialog()->getDLGvisibilityRender()){visibleInRender = OBJECT_ON;}

		if (visibleInEditor == OBJECT_ON && visibleInRender == OBJECT_ON) WriteThisObject = TRUE;
		if (visibleInEditor == OBJECT_ON && visibleInRender == OBJECT_UNDEF && parentRenderMode) WriteThisObject = TRUE;
		if (visibleInEditor == OBJECT_UNDEF && parentEditorMode && visibleInRender == OBJECT_ON)  WriteThisObject = TRUE;
		if (visibleInEditor == OBJECT_UNDEF && parentEditorMode && visibleInRender == OBJECT_UNDEF && parentRenderMode)  WriteThisObject = TRUE;
		//	OBJECT_ON    = 0,	OBJECT_OFF   = 1,	OBJECT_UNDEF = 2,

		if(WriteThisObject)
		{
			if(op->GetFirstCTrack())
			{	
				if(!timeSensorWriten) //Einmal einen globalen TimeSensor rausschreiben
				{
					timeSensorWriten = WriteTimeSensor(vrml);
				}
				WritePositionInterpolator(vrml,op,knotennameC4D);
				WriteOrientationInterpolator(vrml,doc,op,knotennameC4D);
 				//WriteCoordinateInterpolator(vrml,op,knotennameC4D);
			}

			currentEditorMode = TRUE;
			currentRenderMode = TRUE;
		} //if visibleInEditor
		else{
			currentEditorMode = FALSE;
			currentRenderMode = FALSE;
		}
		WriteAnimation(vrml,doc,op->GetDown(),currentEditorMode,currentRenderMode, timeSensorWriten);
		op=op->GetNext();
	}
}


void WritePositionInterpolator(VRMLSAVE &vrml,BaseObject *op, String knotennameC4D)
{	
	Float temp=0;
	Float key=0;
	Float unterteilung = vrml.getStartDialog()->getDLGnoOfKeyFrames(); //Anzahl der Keys, Einstellung vom Dialog
	BaseTime min = vrml.getDoc()->GetMinTime();   
	BaseTime max = vrml.getDoc()->GetMaxTime();
	BaseTime dauer = (max - min);
	Float dauerInSec = dauer.Get();

	vrml.writeC4DString("DEF " +knotennameC4D +"-POS-INTERP PositionInterpolator { \n");
	vrml.writeC4DString("key[ \n");
	vrml.increaseLevel();

	for (Int32 i = 0; i < unterteilung+1; i++)
	{
		key=((i*(dauerInSec/unterteilung)/dauerInSec));
		vrml.writeC4DString(String::FloatToString(key) +",");
		if(i%7 == 0 && i!= 0)vrml.writeC4DString("\n");
	}
	vrml.decreaseLevel();
	vrml.writeC4DString("] \n");
	vrml.writeC4DString("keyValue [ \n");
	vrml.increaseLevel();

	Float* posX = new Float[unterteilung+1];
	Float* posY = new Float[unterteilung+1];
	Float* posZ = new Float[unterteilung+1];

	CTrack *ctrack = op->GetFirstCTrack();
	Float zeitabschnitt = (dauerInSec/unterteilung);
	Int32 fps = vrml.getDoc()->GetFps();
	while (ctrack)
	{ 
		temp=0;
		if (ctrack->GetName() == "Position . X")
		{
			for (Int32 i = 0; i < unterteilung+1; i++)
			{
				temp+= zeitabschnitt;
#ifdef C4D_R16
				posX[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, 0),vrml.getDoc()->GetFps());
#else
				posX[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, 0));
#endif
			}
		}
		if (ctrack->GetName() == "Position . Y")
		{
			for (Int32 i = 0; i < unterteilung+1; i++)
			{
				temp+= zeitabschnitt;
#ifdef C4D_R16
				posY[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, 0),vrml.getDoc()->GetFps());
#else
				posY[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, 0));
#endif
			}
		}
		if (ctrack->GetName() == "Position . Z")
		{
			for (Int32 i = 0; i < unterteilung+1; i++)
			{
				temp+= zeitabschnitt;
#ifdef C4D_R16
				posZ[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, 0),vrml.getDoc()->GetFps());
#else
				posZ[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, 0));
#endif
			}
		}
		ctrack = ctrack->GetNext();
	}
	Int32 pointPerRow = 7;
	for (Int32 i = 0; i < unterteilung+1; i++)
	{
		vrml.writeC4DString(String::FloatToString(posX[i]) +" " +String::FloatToString(posY[i]) +" " +String::FloatToString(posZ[i]) +",");
		if(i%pointPerRow==0 && i!=0)vrml.writeC4DString("\n");
	}
	vrml.decreaseLevel();
	vrml.writeC4DString("] } \n");

	delete[] posX;
	delete[] posY;
	delete[] posZ;
	vrml.writeC4DString("ROUTE Timer.fraction_changed TO "+knotennameC4D +"-POS-INTERP.set_fraction\n");
	vrml.writeC4DString("ROUTE "+knotennameC4D +"-POS-INTERP.value_changed TO "+knotennameC4D +".set_translation\n\n");
}



void WriteOrientationInterpolator(VRMLSAVE &vrml, BaseDocument *doc, BaseObject *op,String knotennameC4D)
{
	//Versuchen mit doc->SetTime(const BaseTime& t) die Zeit im Dokument nacheinander auf die n unterteilungswerte zu stellen 
	//und mit GetValue bzw. const Vector &pos = op->GetPos();const Vector &gr = op->GetScale();const Vector &rot = op->GetRot();
	//arbeiten. 

	//Alternativer Ansatz waere wie bei WritePositionInterpolator() alles ueber den CTrack zu bekommen. 
	//Im Ctrack gibts: 
	//Postition . X
	//Postition . Y
	//Postition . Z
	//Rotation . H
	//Rotation . B
	//Rotation . P
	//Size . X
	//Size . Y
	//Size . Z

	Float temp=0;
	Float key=0;
	Int32 pointPerRow = 7;

	Float unterteilung = vrml.getStartDialog()->getDLGnoOfKeyFrames(); //Anzahl der Keys, Einstellung vom Dialog
	BaseTime min = doc->GetMinTime();   
	BaseTime max = doc->GetMaxTime();
	BaseTime dauer = (max - min);
	Float dauerInSec = dauer.Get();
	Float zeitabschnitt = (dauerInSec/unterteilung);

	vrml.writeC4DString("DEF " +knotennameC4D +"-ROT-INTERP OrientationInterpolator {\n");  //So ists in der Vorlage
	vrml.writeC4DString("key [ \n");
	vrml.increaseLevel();

	for (Int32 i = 0; i < unterteilung+1; i++)
	{
		key=((i*(dauerInSec/unterteilung)/dauerInSec));
		vrml.writeC4DString(String::FloatToString(key) +",");
		if(i%7 == 0)vrml.writeC4DString("\n");
	}
	vrml.decreaseLevel();
	vrml.writeC4DString("\n] \n");
	vrml.writeC4DString("keyValue [ \n");
	vrml.increaseLevel();
	for (Int32 i = 0; i < unterteilung+1; i++)
	{
		temp+= zeitabschnitt; 
		doc->SetTime(BaseTime(temp, 0));  
//		doc->AnimateDocument(NULL, FALSE, TRUE);

		//Eulerwinkel 
		const Vector &rot = op->GetRelRot();   //Get the HPB rotation of the object relative to any parent object. Return Vector -The HPB rotation  (Eulerwinkel) (Muss auf Achse und Winkel umgerechnet werden --google)

		//Umrechnung von Eulerwinkel auf Achse Winke
		Float c1,c2,c3,s1,s2,s3,c1c2,s1s2,x,y,z,angle, w, norm;

		// http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToAngle/index.htm
		// Assuming the angles are in radians.
		c1 = cos(rot.x/2); //heading um y-Achse
		s1 = sin(rot.x/2);
		c2 = cos(-rot.z/2); //attitude (Pitch)  um x-Achse
		s2 = sin(-rot.z/2);
		c3 = cos(rot.y/2);  //bank(Roll) um z-Achse
		s3 = sin(rot.y/2);

		c1c2 = c1*c2;
		s1s2 = s1*s2;
		w =c1c2*c3 - s1s2*s3;
		x =c1c2*s3 + s1s2*c3;
		y =s1*c2*c3 - c1*s2*s3;   
		z =c1*s2*c3 - s1*c2*s3;
		angle = 2 * acos(w);
		norm = x*x+y*y+z*z;
		if (norm < 0.001) { // when all euler angles are zero angle =0 so
			// we can set axis to anything to avoid divide by zero
			x=1;
			y=z=0;
		} else {
			norm = sqrt(norm);
			x /= norm;
			y /= norm;
			z /= norm;
		}
		vrml.writeC4DString(String::FloatToString(z) +" " +String::FloatToString(-y) +" " +String::FloatToString(-x) +" " +String::FloatToString(angle) +", ");
		if(i%pointPerRow==0 && i!=0)vrml.writeC4DString("\n");
	}
	vrml.decreaseLevel();
	vrml.writeC4DString("] }\n\n");
	vrml.writeC4DString("ROUTE Timer.fraction_changed TO "+knotennameC4D +"-ROT-INTERP.set_fraction\n");
	vrml.writeC4DString("ROUTE "+knotennameC4D +"-ROT-INTERP.value_changed TO "+knotennameC4D +".set_rotation\n\n");
}




//void WriteOrientationInterpolator(VRMLSAVE &vrml, BaseDocument *doc, BaseObject *op,String knotennameC4D)
//{	
//	Float temp=0;
//	Float key=0;
//	Int32 unterteilung = 76; //Anzahl der Keys (Wert auf den des internen VRMLexporters gestellt)
//	BaseTime min = vrml.getDoc()->GetMinTime();   
//	BaseTime max = vrml.getDoc()->GetMaxTime();
//	BaseTime dauer = (max - min);
//	Float dauerInSec = dauer.Get();
//
//	vrml.writeC4DString("DEF " +knotennameC4D +"-ROT-INTERP OrientationInterpolator {\n");
//	vrml.writeC4DString("key[ \n");
//	vrml.increaseLevel();
//
//	for (Int32 i = 0; i < unterteilung+1; i++)
//	{
//		key=((i*(dauerInSec/unterteilung)/dauerInSec));
//		vrml.writeC4DString(String::FloatToString(key) +",");
//		if(i%7 == 0 && i!= 0)vrml.writeC4DString("\n");
//	}
//	vrml.decreaseLevel();
//	vrml.writeC4DString("] \n");
//	vrml.writeC4DString("keyValue [ \n");
//	vrml.increaseLevel();
//
//	Float* rotH = new Float[unterteilung+1];
//	Float* rotP = new Float[unterteilung+1];
//	Float* rotB = new Float[unterteilung+1];
//
//
//	CTrack *ctrack = CTrack::Alloc(op,DescID(DescLevel(ID_BASEOBJECT_POSITION,DTYPE_VECTOR,0),DescLevel(VECTOR_X,DTYPE_REAL,0)));  //postition Track
//	ctrack = op->GetFirstCTrack();
//
//	Float zeitabschnitt = (dauerInSec/unterteilung);
//	Int32 fps = vrml.getDoc()->GetFps();
//	while (ctrack)
//	{ 
//		temp=0;
//		if (ctrack->GetName() == "Rotation . H")
//		{
//			for (Int32 i = 0; i < unterteilung+1; i++)
//			{
//				temp+= zeitabschnitt;
//				rotH[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, NOREDUCE),vrml.getDoc()->GetFps());
//				ctrack->TrackInformation(vrml.getDoc(),
//				GePrint("rot h bei i [" +String::IntToString(i) +"] :" +String::IntToString(rotH[i]));
//			}
//			
//		}
//		if (ctrack->GetName() == "Rotation . P")
//		{
//			for (Int32 i = 0; i < unterteilung+1; i++)
//			{
//				temp+= zeitabschnitt;
//				rotP[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, NOREDUCE),vrml.getDoc()->GetFps());
//				GePrint("rot p bei i [" +String::IntToString(i) +"] :" +String::IntToString(rotH[i]));
//			}
//		}
//		if (ctrack->GetName() == "Rotation . B")
//		{
//			for (Int32 i = 0; i < unterteilung+1; i++)
//			{
//				temp+= zeitabschnitt;
//				rotB[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, NOREDUCE),vrml.getDoc()->GetFps());
//				GePrint("rot b bei i [" +String::IntToString(i) +"] :" +String::IntToString(rotH[i]));
//			}
//		}
//		ctrack = ctrack->GetNext();
//	}
//	Int32 pointPerRow = 7;
//	for (Int32 i = 0; i < unterteilung+1; i++)
//	{
//
//		//Eulerwinkel 
//		const Vector &rot = Vector(rotH[i],rotP[i],rotB[i]);   //Get the HPB rotation of the object relative to any parent object. Return Vector -The HPB rotation  (Eulerwinkel) (Muss auf Achse und Winkel umgerechnet werden --google)
//
//		//Umrechnung von Eulerwinkel auf Achse Winke
//		Float c1,c2,c3,s1,s2,s3,c1c2,s1s2,x,y,z,angle, w, norm;
//
//		// http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToAngle/index.htm
//		// Assuming the angles are in radians.
//		c1 = cos(rot.x/2); //heading um y-Achse
//		s1 = sin(rot.x/2);
//		c2 = cos(-rot.z/2); //attitude (Pitch)  um x-Achse
//		s2 = sin(-rot.z/2);
//		c3 = cos(rot.y/2);  //bank(Roll) um z-Achse
//		s3 = sin(rot.y/2);
//
//		c1c2 = c1*c2;
//		s1s2 = s1*s2;
//		w =c1c2*c3 - s1s2*s3;
//		x =c1c2*s3 + s1s2*c3;
//		y =s1*c2*c3 - c1*s2*s3;   
//		z =c1*s2*c3 - s1*c2*s3;
//		angle = 2 * acos(w);
//		norm = x*x+y*y+z*z;
//		if (norm < 0.001) { // when all euler angles are zero angle =0 so
//			// we can set axis to anything to avoid divide by zero
//			x=1;
//			y=z=0;
//		} else {
//			norm = sqrt(norm);
//			x /= norm;
//			y /= norm;
//			z /= norm;
//		}
//		vrml.writeC4DString(String::FloatToString(z) +" " +String::FloatToString(-y) +" " +String::FloatToString(-x) +" " +String::FloatToString(angle) +", ");
//		if(i%pointPerRow==0 && i!=0)vrml.writeC4DString("\n");
//	}
//	vrml.decreaseLevel();
//	vrml.writeC4DString("] } \n");
//
//	delete[] rotH;
//	delete[] rotP;
//	delete[] rotB;
//	vrml.writeC4DString("ROUTE Timer.fraction_changed TO "+knotennameC4D +"-POS-INTERP.set_fraction\n");
//	vrml.writeC4DString("ROUTE "+knotennameC4D +"-POS-INTERP.value_changed TO "+knotennameC4D +".set_translation\n\n");
//}


//void WriteCoordinateInterpolator(VRMLSAVE &vrml,BaseObject *op, String knotennameC4D)
//{	//Vielleicht noch auf CTrack umstellen. S. unten
//	//Versuchen mit doc->SetTime(const BaseTime& t) die Zeit im Dokument nacheinander auf die n unterteilungswerte zu stellen 
//	//und mit GetValue bzw. const Vector &pos = op->GetPos();const Vector &gr = op->GetScale();const Vector &rot = op->GetRot();
//	//arbeiten. 
//
//	Float temp=0;
//	Float key=0;
//	Int32 unterteilung = 76; //Anzahl der Keys (Wert auf den des internen VRMLexporters gestellt)
//	BaseTime min = vrml.getDoc()->GetMinTime();   
//	BaseTime max = vrml.getDoc()->GetMaxTime();
//	BaseTime dauer = (max - min);
//	Float dauerInSec = dauer.Get();
//
//	vrml.writeC4DString("DEF " +knotennameC4D +"-COORD-INTERP CoordinateInterpolator { \n");
//	vrml.writeC4DString("key[ \n");
//	vrml.increaseLevel();
//
//	for (Int32 i = 0; i < unterteilung+1; i++)
//	{
//		key=((i*(dauerInSec/unterteilung)/dauerInSec));
//		vrml.writeC4DString(String::FloatToString(key) +",");
//		if(i%7 == 0 && i!= 0)vrml.writeC4DString("\n");
//	}
//	vrml.decreaseLevel();
//	vrml.writeC4DString("] \n");
//	vrml.writeC4DString("keyValue [ \n");
//	vrml.increaseLevel();
//
//	//Besser so wie unten machen
//	//for (Int32 i = 0; i < unterteilung+1; i++)
//	//{
//	//	temp+= zeitabschnitt; //temp kann spaeter in BaseTime(...) gleich reingeschrieben werden.
//	//	vrml.getDoc()->SetTime(BaseTime(temp, NOREDUCE));  
//	//	vrml.getDoc()->AnimateDocument(NULL, FALSE, TRUE);
//	//	//myPrintf("%f %f %f, \n", op->GetPos().x, op->GetPos().y,op->GetPos().z);
//	//	GePrint("ComeOn");
//	//	GePrint(String::FloatToString(op->GetPos().x) +String::FloatToString(op->GetPos().y) +String::FloatToString(op->GetPos().z));
//	//	vrml.writeC4DString(String::FloatToString(op->GetPos().z) +" " +String::FloatToString(op->GetPos().y) +" " +String::FloatToString(op->GetPos().x) +", \n");
//	//}
//	//vrml.decreaseLevel();
//	//vrml.writeC4DString("] } \n");
//
//
//	//Alternativ kann man die Values auch ueber die CTrack bekommen:
//	//Diese CTracks sind bei mir vorhanden: 
//	//Postition . X
//	//Postition . Y
//	//Postition . Z
//	//Rotation . H
//	//Rotation . B
//	//Rotation . P
//	//Size . X
//	//Size . Y
//	//Size . Z
//
//	Float* sizeX = new Float[unterteilung+1];
//	Float* sizeY = new Float[unterteilung+1];
//	Float* sizeZ = new Float[unterteilung+1];
//
//
//	CTrack *ctrack = CTrack::Alloc(op,DescID(DescLevel(ID_BASEOBJECT_POSITION,DTYPE_VECTOR,0),DescLevel(VECTOR_X,DTYPE_REAL,0)));  //postition Track
//	ctrack = op->GetFirstCTrack();
//
//	Float zeitabschnitt = (dauerInSec/unterteilung);
//	Int32 fps = vrml.getDoc()->GetFps();
//	while (ctrack)
//	{ 
//		temp=0;
//		if (ctrack->GetName() == "Size . X")
//		{
//			for (Int32 i = 0; i < unterteilung+1; i++)
//			{
//				temp+= zeitabschnitt;
//				sizeX[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, NOREDUCE),vrml.getDoc()->GetFps());
//			}
//		}
//		if (ctrack->GetName() == "Size . Y")
//		{
//			for (Int32 i = 0; i < unterteilung+1; i++)
//			{
//				temp+= zeitabschnitt;
//				sizeY[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, NOREDUCE),vrml.getDoc()->GetFps());
//			}
//		}
//		if (ctrack->GetName() == "Size . Z")
//		{
//			for (Int32 i = 0; i < unterteilung+1; i++)
//			{
//				temp+= zeitabschnitt;
//				sizeZ[i] = ctrack->GetValue(vrml.getDoc(),BaseTime(temp, NOREDUCE),vrml.getDoc()->GetFps());
//			}
//		}
//		ctrack = ctrack->GetNext();
//	}
//	Int32 pointPerRow = 7;
//	for (Int32 i = 0; i < unterteilung+1; i++)
//	{
//		vrml.writeC4DString(String::FloatToString(sizeX[i]) +" " +String::FloatToString(sizeY[i]) +" " +String::FloatToString(sizeZ[i]) +",");
//		if(i%pointPerRow==0 && i!=0)vrml.writeC4DString("\n");
//	}
//	vrml.decreaseLevel();
//	vrml.writeC4DString("] } \n");
//
//	delete[] sizeX;
//	delete[] sizeY;
//	delete[] sizeZ;
//	vrml.writeC4DString("ROUTE Timer.fraction_changed TO "+knotennameC4D +"-COORD-INTERP.set_fraction\n");
//	vrml.writeC4DString("ROUTE "+knotennameC4D +"-COORD-INTERP.value_changed TO "+knotennameC4D +".set_point\n\n");
//}


Bool WriteTimeSensor(VRMLSAVE &vrml)
{
	BaseTime min = vrml.getDoc()->GetMinTime();   
	BaseTime max = vrml.getDoc()->GetMaxTime();
	BaseTime dauer = (max - min);
	Float dauerInSec = dauer.Get();
	//Bisher nur statischer Timesensor. (-->Dauer jetzt dynamisch)
	vrml.writeC4DString("\n");
	vrml.writeC4DString("DEF Timer TimeSensor {\n");
	vrml.writeC4DString("      startTime 0\n");
	vrml.writeC4DString("      stopTime 0\n");
	vrml.writeC4DString("      cycleInterval "+String::FloatToString(+dauerInSec) +"\n");
	vrml.writeC4DString("      loop TRUE\n");
	vrml.writeC4DString("}\n\n");
	return TRUE;
}

