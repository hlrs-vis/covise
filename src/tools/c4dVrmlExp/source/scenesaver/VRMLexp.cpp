//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Florian Methner    																											//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//(c) 2010			Version alpha_0.3																							//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//VRML Exporter für Cinema4D 																									//
//Das Plugin wird in RegisterVRML() registriert. Die Klasse VRMLSaverData ist vom SceneSaverPlugin-Hook abgeleitet. Ihre Methode//
//Save() wird von Cinema 4D aufgerufen. Save() organisert den Exportvorgang. Sie ruft den Startdialog auf. Ausserdem wird in 	//
//ihr eine Instanz der Klasse VRMLSAVE erstellt, welche neben wichtiger Variablen, die Exportdatei verwaltet und für 			//
//deren Formatierung zuständig ist. 																							//
//																																//
//Weiterhin wird in Save() wird die Methode WriteObjects aufgerufen, die Rekursive durch alle Objekte einer Szene durchgeht 	//
//und diese herausgeschreibt.																									//
//Korrekt herausgeschrieben werden bisher alle Geometrischen Objekten (WriteGeometry() und LOD-Objekte (WriteLODStart()). 		//
//Sollen weitere Elemente wie z.B. Camera, Sky, ProximitySensor usw. herrausgeschrieben werden müssen diese an der passenden 	//
//Stellen in WriteObjects() implementiert werden. 																				//
//																																//
//Enthält eine Szene Animationen werden diese in WriteAnimation() exportiert, die in auch in Save() aufgerufen wird. 			//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//change log: normalen angepasst
//			  Apperance Knoten, anstatt nur Materialknoten wird für Kinderobjekte wiederverwendet
//			  Mehrere Texturen können einem Objekt zugewiesen werden (Multitexturing?)

#include "VRMLexp.h"
using namespace std;


FILEERROR VRMLSaverData::Save(BaseSceneSaver *node, const Filename &name, BaseDocument *doc, SCENEFILTER flags)
{
	if (!(flags&SCENEFILTER_OBJECTS)) return FILEERROR_NONE;

	StartDialog		startDialog;
	if(startDialog.Open(-1, -1, 200, 100))
	{
		VRMLSAVE      vrml = VRMLSAVE(name, doc, &startDialog);

		if (!vrml.getDoc() || !vrml.getFile()) return FILEERROR_OUTOFMEMORY;
		if (!vrml.getFile()->Open(name,FILEOPEN_WRITE,FILEDIALOG_NONE,BYTEORDER_INTEL)) return vrml.getFile()->GetError();	//oeffne die Datei

		vrml.objnameHashSet.clear();		
		vrml.texNameHashSet.clear();

		//Kopieren Texturen wenn dies im Startdialog so ausgewaehlt
		if(vrml.getStartDialog()->getDLGexportTextures())
		{
			GetCopyPaths(vrml);
		}

		WriteHeader(name,vrml);
		WriteObjects(vrml, vrml.getDoc()->GetFirstObject(),Matrix(),TRUE, TRUE, "");	

		vrml.objnameHashSet.clear();	
		if(vrml.getStartDialog()->getDLGwriteAnimation())
		{
			WriteAnimation(vrml,doc, doc->GetFirstObject(),TRUE, TRUE, FALSE);	//Das an dieser Stelle um Animationen am Ende rauszuschreiben //doc wird mituebergeben, da WriteOrientationInterpolator() mit vrml.getDoc() nicht funktioniert		
		}
		return vrml.getFile()->GetError();
	} 
//	return FALSE;
	return FILEERROR_NONE;
}

void WriteHeader(const Filename &name, VRMLSAVE &vrml)
{
	//Write Header
	time_t ltime;
	time( &ltime );
	CHAR * time = ctime(&ltime);
	// strip the CR
	time[strlen(time)-1] = '\0';

	vrml.writeC4DString("#VRML V2.0 utf8\n\n");	
	vrml.writeC4DString("## Produced by Florian's VRML Plugin, Version alpha_0.2 \n");	
	vrml.writeC4DString("# CINEMA4D File: " +vrml.getDoc()->GetDocumentName().GetString() +" ");
	vrml.writeC4DString("Date: " +String(time) +"\n");

	CHAR		  header[80];
	ClearMem(header,sizeof(header));
	name.GetFileString().GetCString(header,78,STRINGENCODING_7BIT);
	vrml.writeC4DString("# ");
	vrml.getFile()->WriteBytes(header,name.GetFileString().GetCStringLen(STRINGENCODING_UTF8));
	vrml.writeC4DString("\n\n");
	//Header Ende
}


static void WriteObjects(VRMLSAVE &vrml, BaseObject *op, Matrix up, Bool parentEditorMode, Bool parentRenderMode, String parentApperance)
{	
	while (op)
	{	
		Bool dontWriteMyGeometry = FALSE;		//Variable damit die Geometrien von SonderObjekte (NavigationInfo, ...) nicht herausgeschrieben werden. 
		Bool dontWriteTransform = FALSE;		//Für manche Objekte muß kein Transformknoten geschrieben werden (NavigationInfo,...)

		//Fuer den Undef Editormodus immer schauen ob das aktuelle objekt an,aus oder undef ist und dann den Editormodus dementsprechend beim rekursiven aufruf uebergeben
		Bool currentEditorMode, currentRenderMode;
		Matrix mg;
		currentEditorMode = parentEditorMode;	//currentModes erforderlich um zu wissen was im OBJECT_UNDEF Fall zu tun ist
		currentRenderMode = parentRenderMode;

		//Hier die Abfrage ob das aktuelle Objekt eine Instanz eines anderen ist.Muss noch vollstaending implementiert werden!
		//if (opOrg->GetType() == Oinstance) 
		//{
		//	BaseContainer *daten = opOrg->GetDataInstance();	
		//	//Wer ist mein Reference Object. 
		//	//Vermerke, dass Nr.y eine Instanz von Nr.x ist
		//}

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

		//Hilfsklasse VRMLmgmt mit haeufigen Werten und damit ToPoly usw. nicht staendig ausgefuehrt werden muss
		VRMLmgmt *dataMgmt;   
		dataMgmt = new VRMLmgmt(op, knotennameC4D, parentApperance);

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
			//////////////////////////////////////////////////////////////////////////////
			///Hier alle Elemente Implementieren die Ausgegeben werden sollen:			//
			//////////////////////////////////////////////////////////////////////////////
			
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//Hier kommen Objekte hin, für die kein Write Transform geschrieben werden soll:							//
			//dontWriteTransform muss auf TRUE setzen,																	//
			//soll für das Objekt auch keine Geometrie rausgeschrieben werden muss dontWriteMyGeometry auf TRUE setzten	//
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////


			//NavigationInfoObject
			if(op->GetDataInstance()->GetBool(I_M_A_NAVIGATIONINFO_OBJECT))
			{
				NavigationInfoObject::WriteNavigationInfo(vrml, dataMgmt);
				dontWriteTransform = TRUE;
				dontWriteMyGeometry = TRUE;
			}


			//Transformknoten
			if (!dontWriteTransform)
			{
				WriteTransform(vrml,dataMgmt);
			}

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////
			//Hier kommen Objekte hin, für die ein Transformknoten geschrieben werden soll:								//
			//soll für das Objekt keine Geometrie rausgeschrieben werden muss dontWriteMyGeometry auf TRUE setzten		//
			//////////////////////////////////////////////////////////////////////////////////////////////////////////////

			//LOD - Objekt
			if(op->GetDataInstance()->GetBool(I_M_A_LOD_OBJECT))
			{
				LODObject::WriteLODStart(vrml,dataMgmt);
				dontWriteMyGeometry = TRUE;
			}

			//Alles weitere, z.B. Camera: 
			if (dataMgmt->getType()== Ocamera)
			{
				WriteCamera(vrml,dataMgmt);
			}

			// Schreibe die geometrie der Objektes
			if (dataMgmt->getType()==Opolygon && !dontWriteMyGeometry)		
			{
				WriteGeometricObjects(vrml,up,dataMgmt);
			}


			currentEditorMode = TRUE;
			currentRenderMode = TRUE;
		} //if visibleInEditor
		else{
			currentEditorMode = FALSE;
			currentRenderMode = FALSE;
		}
		
		WriteObjects(vrml, op->GetDown(), mg, currentEditorMode, currentRenderMode, dataMgmt->getParentApperance());
		
		if(WriteThisObject)
		{
			//////////////////////////////////////////////////////////
			///Hier Gruppenknoten schließen						    //
			//////////////////////////////////////////////////////////

			//LOD - Objekt
			if(op->GetDataInstance()->GetBool(I_M_A_LOD_OBJECT))
			{
				LODObject::WriteEndLOD(vrml);
			}
			// Geometrische Objekte (Polygon Objekte)
			if (!dontWriteTransform)
			{
				WriteTransformEnd(vrml);
			}
		}
		op=op->GetNext();
		delete dataMgmt;
	} // while end
}



void WriteTransform(VRMLSAVE &vrml, VRMLmgmt *dataMgmt)
{
	vrml.writeC4DString("DEF "+dataMgmt->getKnotenName() +" Transform { \n");
	vrml.increaseLevel();
	const Vector &pos = dataMgmt->getOp()->GetRelPos();  //Get the position vector for this object. These will be local coordinates within its parent object.
	vrml.writeC4DString("translation "+String::FloatToString(pos.z) +" " +String::FloatToString(pos.y) +" " +String::FloatToString(pos.x) +"\n");
	const Vector &gr = dataMgmt->getOp()->GetRelScale();
	vrml.writeC4DString("scale "+String::FloatToString(gr.x) +" " +String::FloatToString(gr.y) +" " +String::FloatToString(gr.z) +"\n");

	//Umrechnung von Eulerwinkel auf Achse Winkel
	Matrix ml1 = dataMgmt->getOp()->GetMln();
	Float angle;
	Vector axis;
	// Convert Euler matrix to angle/axis
	MyMatrixToRotAxis(ml1,&axis,&angle);
	//MatrixToRotAxis(ml1,&axis,&angle);

	//angle = angle + (pi/2);
	vrml.writeC4DString("rotation "+String::FloatToString(axis.z) +" " +String::FloatToString(axis.y) +" " +String::FloatToString(axis.x) +" " +String::FloatToString(angle) +"\n");
	vrml.increaseLevel();
	vrml.writeC4DString("children [\n");
	vrml.increaseLevel();
}


void MyMatrixToRotAxis(const Matrix &mm, Vector *v, Float *w)
{
    Matrix m = mm;


    // MatrixVectoren MUESSEN normiert sein!!!
    m.v1=!m.v1;
    m.v2=!m.v2;
    m.v3=!m.v3;

    // Winkel berechnen
    *w = ACos((m.v1.x+m.v2.y+m.v3.z-1.0)/2.0);

    // NOTE: subtraction order reversed
    // v->x= m.v2.z-m.v3.y;
    // v->y= m.v3.x-m.v1.z;
    // v->z= m.v1.y-m.v2.x;
    v->x= m.v3.y-m.v2.z;
    v->y= m.v1.z-m.v3.x;
    v->z= m.v2.x-m.v1.y;
    *v = !(*v);

    if (*v==0.0)
    {
        *v = Vector(0.0,1.0,0.0);
        *w = 0.0;
    }
}



void WriteTransformEnd(VRMLSAVE &vrml)
{
	vrml.decreaseLevel();
	vrml.writeC4DString("]\n"); 
	vrml.decreaseLevel();
	vrml.decreaseLevel();
	vrml.writeC4DString("}\n");
}



//
Bool RegisterVRML(void)
{
	String name = "VRML Exporter";
	Int32 id = 1025451;		// ID vom Plugincafe
	if (!RegisterSceneSaverPlugin(id,name,0,VRMLSaverData::Alloc,"fvrmlexport","wrl")) return FALSE;
	return TRUE;
}



//Behlfsmethoden:
//Ermittelt die Bitmap -Namen und -Pfade um diese dann in CopyTextures() zu kopieren.
void GetCopyPaths(VRMLSAVE &vrml)
{
	// Alle verwendeten Texturen werden in /maps/ ordner des Zielordners kopiert
	BaseContainer ctr = vrml.getDoc()->GetAllTextures(NULL); 
	BrowseContainer bc(&ctr); 
	Int32 id; 
	GeData *dat; 

	while(bc.GetNext(&id, &dat)) 
	{ 
		Filename fn = dat->GetFilename();
		Filename bitmapDirectory = fn.GetDirectory();
		Int32 bitmapDirectoryLength = bitmapDirectory.GetString().GetLength();
		String bitmapFileName = fn.GetFileString();
		Int32 bitmapFileNameLength = bitmapFileName.GetLength();
		String bitmapDirectoryString = bitmapDirectory.GetString();
		//bitmapDirectoryString.Delete((bitmapDirectoryLength - bitmapFileNameLength), bitmapDirectoryLength);
		String out = vrml.getFolderName();
		Int32 outLength = out.GetLength();
		out.Delete(outLength-vrml.getFileName().GetLength(),outLength);
		out+=+"\\maps\\";

		//Falls der Pfad nicht geladen wurde schau in den SearchPaths
		//und anschliessend im /tex/ Pfad des Dokumentenpfads nach
		if (bitmapDirectoryString.GetLength() == 0)
		{
			Filename documentPath = vrml.getDoc()->GetDocumentPath();
			Filename temp = NULL ;

			if(IsInSearchPath(bitmapFileName, documentPath))
			{
				for (int i = 0; i<9; i++)
				{
					Filename searchPath = GetGlobalTexturePath(i);

					if(GenerateTexturePath(documentPath,bitmapFileName,searchPath.GetString(),&temp))
					{
						bitmapDirectoryString = temp.GetString();
						bitmapDirectoryString.Delete(temp.GetString().GetLength() - bitmapFileNameLength, temp.GetString().GetLength());
						break;
					}
				}
			}
			else 
			{	//nicht im Searchpfad, also schau in \tex\ des Dokumentenpfads
				bitmapDirectory = documentPath.GetString() +"\\tex\\";
				if(!GenerateTexturePath(documentPath,bitmapFileName,bitmapDirectory,&temp))
				{
					vrml.writeC4DString("#Couldn't find/copy Textur: "+bitmapFileName);
				}
			}
		}
		Bool check = CopyTextures(vrml, bitmapDirectoryString, bitmapFileName, out);
	} //END while
	//vrml.getStartDialog()->setDLGTextureCopy(FALSE);
} //END if(vrml.getDLGTextureCopy())


//Kopiert die Textur in den /maps/ Folder. 
//Vom 3DS Max exporter übernommen und angepasst. 
Bool CopyTextures(VRMLSAVE &vrml, String bitmapDirectory, String bitmapFileName,  String destDirectory) 
//void CopyTextures(TSTR bitmapFile, TSTR &fileName, TSTR &url)
{
	String bitmapFileString = bitmapDirectory + bitmapFileName;
	String fileName = bitmapFileName;

	//Evtl. noch zu implementieren: 
	// noch keine Statusbar implementiert, daher: 
	Bool mEnableProgressBar = FALSE;
	// optionaler URL Prefix noch nicht implementiert: 
	String url = ""; 

	if (bitmapFileString.Content() == NULL) return FALSE;
	Int32 l = bitmapFileString.GetLength()-1;
	if (l < 0)return FALSE;

	String path = bitmapDirectory;

	if(vrml.getStartDialog()->getDLGexportTextures())
	{
		// check, if destination is older than source, then overwrite, otherwise ask.
		// TODO
		String progressText;
		String destPath = destDirectory;
		String wrlFileName = VRMLName(bitmapFileName);
		String slash = "/";
		String sourceFile = bitmapFileString;
		String space = " ";
		struct _finddata_t fileinfo;
		//try to find destdir
		String destDirString = destPath;
		String backSlash = "\\";

		//optional noch einfügbar
		//if (mUsePrefix && mUrlPrefix.Length() > 0)
		//{
		//   destDir = destDir + backSlash + mUrlPrefix;
		//   if(mUrlPrefix[mUrlPrefix.Length() - 1] == '/')
		//      destDir.remove(mUrlPrefix.Length() - 1);
		//}

		String destFileString = destDirString + backSlash + fileName;

		//Konvertierung der C4D Strings in CHAR*
		CHAR* destFile = StringC4DtoCstr(destFileString);
		CHAR* bitmapFile = StringC4DtoCstr(bitmapFileString);
		CHAR* destDir = StringC4DtoCstr(destDirString);


		intptr_t res = _findfirst(destDir,&fileinfo);
		if(res == -1)
		{
			// destdir does not exist so create it
			//command = mkdir + destDir;
			//system(command);
			_mkdir(destDir);
		}
		DeleteMem(destDir);

		//Dialogeinstellungen, noch implementieren:
		Bool mReplaceAll = FALSE;
		Bool mSkipAll = FALSE;
		Bool mReplace = FALSE;
		Int32 exportOption = vrml.getStartDialog()->getDLGexportOption() ;
		switch (exportOption)
		{
		case IDC_REPLACEALL:	mReplaceAll = TRUE;
			break;
		case IDC_SKIPEXISTING:	mSkipAll = TRUE;
			break;
		case IDC_ASKREPLACE:	mReplace = TRUE;
			break;
		}

		Bool copyToDest = FALSE;
		Bool copyFromDest = FALSE;
		Int32 fd = _open(destFile,O_RDONLY);
		if(fd>0)
		{
			Int32 fdS = _open(bitmapFile,O_RDONLY);
			if(fdS>0)
			{
				struct _stat dBuf,sBuf;
				_fstat(fd,&dBuf);
				_fstat(fdS,&sBuf);
				if(dBuf.st_mtime < sBuf.st_mtime)
				{
					if(!mReplaceAll && !mSkipAll)
					{
						ConfirmTextureCopyDLG confirmdialog;
						confirmdialog.setFilename(String(bitmapFile));
						if(confirmdialog.Open()){mReplace = TRUE;}
						else{mReplace = FALSE;}
					}
					if(mReplaceAll)
					{
						copyToDest = TRUE;
					}
					if(mReplace)
					{
						copyToDest = TRUE;
					}
				}
				_close(fdS);

			}
			else
			{
				copyFromDest = TRUE;
			}
			_close(fd);
		}
		else
		{
			Int32 fdS = _open(bitmapFile,O_RDONLY);
			if(fdS>0)
			{
				copyToDest = TRUE;
				_close(fdS);
			}

		}
		if(copyToDest)
		{
			if (mEnableProgressBar) //Progressbar ist noch nicht implementiert
			{
				/*    progressText = TSTR("copying ") + bitmapFile + TSTR(" to ") + destFile; 
				SendMessage(hWndPDlg, 666, 0,	(LPARAM) (char *)progressText);*/
			}
			int lenA = lstrlenA(bitmapFile);
			int lenW = ::MultiByteToWideChar(CP_ACP, 0, bitmapFile, -1, NULL, 0);
			int lenAd = lstrlenA(destFile);
			int lenWd = ::MultiByteToWideChar(CP_ACP, 0, destFile, -1, NULL, 0);
			if (lenW>0&&lenWd>0)
			{
				wchar_t *wBitmapFile = new wchar_t[lenW];
				::MultiByteToWideChar(CP_ACP, 0, bitmapFile, -1, wBitmapFile, lenW);
				wchar_t *wDestFile = new wchar_t[lenW];
				::MultiByteToWideChar(CP_ACP, 0, destFile, -1, wDestFile, lenWd);
				CopyFile(wBitmapFile, wDestFile, FALSE);
			}
		}
		else if(copyFromDest)
		{
			if (mEnableProgressBar) 
			{
				/*  progressText = TSTR("copying ") + destFile + TSTR(" to ") + bitmapFile; 
				SendMessage(hWndPDlg, 666, 0,	(LPARAM) (char *)progressText);*/
			}
			int lenA = lstrlenA(bitmapFile);
			int lenW = ::MultiByteToWideChar(CP_ACP, 0, bitmapFile, -1, NULL, 0);
			int lenAd = lstrlenA(destFile);
			int lenWd = ::MultiByteToWideChar(CP_ACP, 0, destFile, -1, NULL, 0);
			if (lenW>0 && lenWd>0)
			{
				wchar_t *wBitmapFile = new wchar_t[lenW];
				::MultiByteToWideChar(CP_ACP, 0, bitmapFile, -1, wBitmapFile, lenW);
				wchar_t *wDestFile = new wchar_t[lenW];
				::MultiByteToWideChar(CP_ACP, 0, destFile, -1, wDestFile, lenWd);
				CopyFile(wDestFile, wBitmapFile, FALSE);
			}
		}
		DeleteMem(destDir);
		DeleteMem(bitmapFile);
	}
	//url = PrefixUrl(fileName);

	return TRUE;
}


// Translates name (if necessary) to VRML compliant name.
// Returns name in static buffer, so calling a second time trashes
// the previous contents.
#define CTL_CHARS      31
String VRMLName(String nameIN)
{
	char* name = new char[nameIN.GetCStringLen() + 1];
	nameIN.GetCString(name, nameIN.GetCStringLen()+1,STRINGENCODING_XBIT);	//Hier vielleicht noch ne andere Formatierung

	char * cPtr;

	cPtr = name;
	if (*cPtr >= '0' && *cPtr <= '9')
	{
		char* name2 = new char[nameIN.GetCStringLen() + 1];
		name2[0] = '_';
		strcpy(name2 + 1, name);
		cPtr = name2;
		delete[] name;
		name = name2;

	}
	while(*cPtr) {
		if( *cPtr <= CTL_CHARS ||
			*cPtr == ' ' ||
			*cPtr == '\''||
			*cPtr == '"' ||
			*cPtr == '\\'||
			*cPtr == '{' ||
			*cPtr == '}' ||
			*cPtr == ',' ||            
			*cPtr == '.' ||
			*cPtr == '[' ||
			*cPtr == ']' ||
			*cPtr == '.' ||
			*cPtr == '#' ||
			*cPtr >= 127
			) *cPtr = '_';
		cPtr++;
	}
	String n(name);
	delete[] name;
	return n;
}


std::string StringC4DtoStringSTD(String s1)
{
	CHAR *charline = NULL;
	Int32 strlength = s1.GetCStringLen(STRINGENCODING_XBIT);
	charline = NewMem(Char,strlength+1);
	if(!charline) return FALSE;
	strlength = s1.GetCString(charline, strlength+1, STRINGENCODING_7BITHEX);

	string StringSTD = string(charline);
	DeleteMem(charline);
	return StringSTD;
}

CHAR* StringC4DtoCstr(String s1)
{
	CHAR *charline = NULL;
	Int32 strlength = s1.GetCStringLen(STRINGENCODING_XBIT);
	charline = NewMem(CHAR,strlength+1);
	if(!charline) return FALSE;
	strlength = s1.GetCString(charline, strlength+1, STRINGENCODING_7BITHEX);
	return charline; 
}




//Klassenimplementierung: 

//VRMLSAVE: 
VRMLSAVE::VRMLSAVE(const Filename &name, BaseDocument *doc,StartDialog *startDialog)
{
	mFile  = BaseFile::Alloc();
	mDoc   = doc->Polygonize();
	mFilename = name.GetFileString(); 
	mFoldername = name.GetString();
	mStartDialog = startDialog;
	mLevel = 0;	
	mIndent = TRUE;	
}

VRMLSAVE::~VRMLSAVE(void)
{
	BaseFile::Free(mFile);
	BaseDocument::Free(mDoc);
}

BaseFile* VRMLSAVE::getFile(){return mFile;}
BaseDocument* VRMLSAVE::getDoc(){return mDoc;}
void VRMLSAVE::increaseLevel(){mLevel++;}
void VRMLSAVE::decreaseLevel(){mLevel--;}
String VRMLSAVE::getFileName(){return mFilename;}
String VRMLSAVE::getFolderName(){return mFoldername;}
void VRMLSAVE::noIndent(){mIndent=FALSE;}
StartDialog* VRMLSAVE::getStartDialog(){return mStartDialog;}


//Schreibt die Cinema4D Unicode Strings (String) in die Exportdatei
Bool VRMLSAVE::writeC4DString(const String line)
{	
	BaseFile* file = mFile;
	if(!file) return FALSE;

	CHAR *charline = NULL;
	Int32 strlength = line.GetCStringLen(STRINGENCODING_7BITHEX);
	charline = NewMem(Char,strlength+1);
	if(!charline) return FALSE;
	strlength = line.GetCString(charline, strlength+1, STRINGENCODING_7BITHEX);

	if(mIndent)
	{
		//Einschub, ausser es wurde ausdruecklich keiner gewollt
		for(Int32 i=0; i<mLevel; i++)	
		{
			file->WriteChar(' ');
			file->WriteChar(' ');
		}
	}
	else {mIndent = TRUE;}

	Int32 i;
	for(i=0; i<strlength; i++)
	{
		if(!file->WriteChar(charline[i])) return FALSE;
	}
	DeleteMem(charline);
	return TRUE;
}



//VRMLmgmt: 
VRMLmgmt::VRMLmgmt(BaseObject *op, String knotennameC4D, String parentApperance)
{
	PolygonObject *toPoly = ToPoly(op);		//Cast des BaseObjects auf ein PolygonObject
	mPadr = toPoly->GetPointR();
	mVadr = toPoly->GetPolygonR();
	mPcnt = toPoly->GetPointCount();
	mVcnt = toPoly->GetPolygonCount();
	mType = op->GetType();
	mKnotenname = knotennameC4D;
	mOpj = op;
	mParentApperance = parentApperance;
}

VRMLmgmt::~VRMLmgmt() 
{
	BaseObject::Free(mOpj);
}

const Vector* VRMLmgmt::getPadr(){return mPadr;}
const CPolygon* VRMLmgmt::getVadr(){return mVadr;}
Int32 VRMLmgmt::getPcnt(){return mPcnt;}
Int32 VRMLmgmt::getVcnt(){return mVcnt;}
Int32 VRMLmgmt::getType(){return mType;}
BaseObject* VRMLmgmt::getOp(){return mOpj;}
String VRMLmgmt::getKnotenName(){return mKnotenname;}
String VRMLmgmt::getParentApperance(){return mParentApperance;}
void VRMLmgmt::setParentApperance(String parentApperance){mParentApperance = parentApperance;}
