#ifndef VRML_DIALOG_H
#define VRML_DIALOG_H
#include "../scenesaver/VRMLexp.h"
#include "c4d_symbols.h"

///Zustaendig fuer den StartDialog. Abgeleitet von ModalDialog
class StartDialog : public GeModalDialog
{
public:
	///Functions to override
	virtual Bool CreateLayout(void);	///siehe SDK-Docu		
	virtual Bool InitValues(void);		///siehe SDK-Docu	
	virtual Bool AskClose(void);		///siehe SDK-Docu	

	//AccessFunktionen für die StartDialogeinstellungen 
	Bool getDLGexportTextures();	///"Sollen Texturen kopiert werden?" Ja - TRUE; Nein - FALSE; 
	Int32 getDLGexportOption();		///Sollen vorhandene Texturen überschrieben (IDC_REPLACEALL) oder übersprungen (IDC_SKIPEXISTING) werden, oder soll eine extra Nachfrage erscheinen (IDC_ASKREPLACE)
	Bool getDLGObjectColorWhite();	///"Soll die Objektfarbe für texturierte Objekte auf weiß gesetzt werden?"
	Bool getDLGexportNormals();
	Int32 getDLGdataFormat();		///"In welchem Format sollen die Daten exportiert werden?" OpenCoverVRML Mod: IDC_OPENCOVERVRMLMOD; Standard VRML 2.0: IDC_NORMALVRML
	Bool getDLGvisibilityEditor();	///
	Bool getDLGvisibilityRender();	///
	Float getDLGambientIntensity();
	Bool getDLGwriteAnimation();
	Float getDLGnoOfKeyFrames();


private:
	Bool mExportTextures;
	Int32 mExportOption;
	Bool mObjectColorWhite;
	Int32 mDataFormat;
	Bool mVisiblityEditor;
	Bool mVisiblityRender;
	Float mAmbientIntensity;
	Bool mWriteAnimation;
	Float mNoOfKeyFrames;
	Bool mExportNormals;
		
};
#endif
