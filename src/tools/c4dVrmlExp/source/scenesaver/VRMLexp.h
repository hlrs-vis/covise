#ifndef VRML_EXP_H
#define VRML_EXP_H

//Includes: 

#include <windows.h>	//Fuer Copy(), vielleicht noch anders, Plattformunabhaenging, loesen
#include <time.h>		//Fuer die Uhrzeit im Header

//Fuer eindeutige Namen im Hashset, std:: string ansonsten eigentlich nicht noetig.
#include <hash_set>
#include <string>

//Fuer VRMLNames()
#include <Tchar.h>
#include <io.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <direct.h>

//Allgemein
#include <stdio.h>
#include <string.h>
#include "c4d.h"
#include "c4d_symbols.h"

//Fuers LOD - Objekt
#include "../object/LOD.h"
//Fuers NavigationInfo - Objekt
#include "../object/NavigationInfo.h"

//Fuer den Dialog
#include "../dialog/StartDialog.h"
#include "../dialog/ConfirmTextureCopy.h"

//Fuer die WriteElements Methoden (Hier später alle zusätzlichen WriteElement Methode inkludieren)
#include "../WriteElements/WriteGeometry.h"
#include "../WriteElements/WriteAnimation.h"
#include "../WriteElements/WriteCamera.h"



//Bekanntmachung der Methoden die nicht in einer Klasse stehn.

//Für den Exporter: 

//Header: 
void WriteHeader(const Filename &name, VRMLSAVE &vrml);				///Schreibt den VRML-Header

//Rausschreiben der Objekte
 void WriteObjects(VRMLSAVE &vrml, BaseObject *op, Matrix up, Bool parentEditorMode, Bool parentRenderMode, String parentApperance);		///Schreibt die Objekte und ruft die dazu benötigten Funktionen auf
 void WriteTransform(VRMLSAVE &vrml, class VRMLmgmt *dataMgmt);		///Schreibt den Transform-Knoten 
 void WriteTransformEnd(VRMLSAVE &vrml);							///Schließt den Transform-Knoten wieder

 //Registrierung: 
Bool RegisterVRML(void);											///Registiert das Plugin

//Behelfsmethoden: 
void GetCopyPaths(VRMLSAVE &vrml);									///Ermittelt die Bitmap -Namen und -Pfade um diese dann in CopyTextures() zu kopieren.
Bool CopyTextures(VRMLSAVE &vrml,String bitmapDirectory, String bitmapFileName,  String destDirectory) ;	///kopiert Texturen ins /map/ Verzeichnis
String VRMLName(String nameIN);										///Liefert den VRML-Kompatiblen (bereinigten) String eines C4D Strings zurück
std::string StringC4DtoStringSTD(String s1);						///Wandelt Cinema4D String in std::string um
CHAR* StringC4DtoCstr(String s1);									///Wandelt Cinema4D String in Cstr um
void MyMatrixToRotAxis(const Matrix &mm, Vector *v, Float *w);		///Berechnet die Rotationsachse und den Winkel aus der Matrix (Wird für rotation benötigt). Ähnlich wie MatrixToRotAxis 


//Klassendeklaration: 

///SceneSaverPlugin Hook
class VRMLSaverData : public SceneSaverData		
{
public:
	virtual FILEERROR Save(BaseSceneSaver *node, const Filename &name, BaseDocument *doc, SCENEFILTER filterflags);
	static NodeData *Alloc(void) { return NewObj(VRMLSaverData); }
};



///Klasse zur Verwaltung der Ausgabedatei und ihrer Formatierung sowie weiterer wichtiger Variablen.
class VRMLSAVE		
{	
public:
	//Accsessor Methoden um auf wichtige Variablen zurückzugreifen: 
	BaseDocument* getDoc();							///Gibt das Poligonized Dokument zurueck
	String getFileName();							///Liefert den Dateinamen des C4D Dokuments zurueck
	String getFolderName();							///Liefert den Pfad des C4D Dokuments zuueck
	BaseFile* getFile();							///Gibt die Export-Datei zurueck
		
	///Zugriff auf Dialog Parameter: 
	class StartDialog* getStartDialog();			///Gibt Zeiger auf Instanz des Dialog zurück, von diesem Objekt können dann die Optionen abgefragt werden.
	
	//Methoden um in die Export-Datei zu schreiben und diese zu Formatieren: 
	Bool writeC4DString(const String line);			///Schreibt C4D String in die ExportDatei 
	void increaseLevel();							///Erhoeht das Level des Zeilen Einschubs um eins
	void decreaseLevel();							///Verringert das Level des Zeilen Einschubs um eins
	void noIndent();								///Zeileneinschub soll nicht erfolgen

	//Hash-Sets (vielleicht noch Private machen und mit Accessormethoden versehen) 
	stdext::hash_set<std::string> texNameHashSet;	///Hash_Set in dem die Texturname gespeichert werden
	stdext::hash_set<std::string> objnameHashSet;	///Hash_Set in dem die Objektnamen gespeichert werden

	//Konstruktor /Destruktor 
	VRMLSAVE(const Filename &name, BaseDocument *doc, StartDialog* startDialog);
	~VRMLSAVE(void);

private:
	BaseFile		*mFile;
	BaseDocument	*mDoc, *mOrgDoc;				///Poligonized Document, orginal Document; 
	Int32			mLevel;
	String			mFilename, mFoldername;

	Bool		mIndent;							///Should we Indent (den Level anwenden?)
	Int32		mCntTuvw;
	Int32		mTextTagCnt;

	//Start - Dialog:
	StartDialog *mStartDialog; 
};


/// Hilfsklasse die oft verwendete Parameter zwischenspeichert und sorgt dass diese nicht jedes Mal neu berechnet werden muessen. 
class VRMLmgmt 
{
public:
	const Vector* getPadr();			///Liefert die Startadresse des "read-only" Punkte Array 
	const CPolygon* getVadr();			///Liefert die Startadresse des "read-only" polygonen Array
	Int32 getPcnt();						///Liefert Anzahl der Punkte im Objekt
	Int32 getVcnt();						///Liefert Anzahl der Polygone im Objekt
	Int32 getType();						///Liefert den ObjektTyp zurück (z.B. Ocube, Opolygon, ...) 
	String getKnotenName();				///Gibt Name des aktuellen Objekts zurück
	BaseObject*	getOp();				///Gibt den Pointer auf das aktuelle Objekt zurück
	String getParentApperance();		///Gibt den String des Eltern-Apperance Knoten zurück, damit dieser wiederverwendet werden kann
	void setParentApperance(String parentApperance);		///Setzt den Eltern-Apperance Knoten

	VRMLmgmt(BaseObject *opj, String knotennameC4D, String parentApperance);
	~VRMLmgmt(void);

private:
	BaseObject		*mOpj;
	const Vector	*mPadr;			
	const CPolygon	*mVadr;			
	Int32			mPcnt,mVcnt,mType;
	String			mKnotenname;
	String			mParentApperance;			//Verwaltet den Apperance Knoten, damit Kinderobjekte diesen wiederverwenden können	

};



#endif