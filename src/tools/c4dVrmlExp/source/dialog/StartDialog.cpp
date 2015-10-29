#include "StartDialog.h"


Bool StartDialog::CreateLayout()
{
	if(!LoadDialogResource(IDD_VRMLEXP_DLG,NULL,0))return FALSE;
	return TRUE;
}

Bool StartDialog::InitValues()
{
	SetBool(IDC_CHECK_COPYTEXTURES,TRUE);
	SetInt32(IDC_COMBO_TEXTUREEXPORT,IDC_REPLACEALL);
	SetBool(IDC_CHECK_SETTEXTUREDOBJECTCOLORWHITE, TRUE);
	SetInt32(IDC_COMBO_DATAFORMAT, IDC_OPENCOVERVRMLMOD);
	SetBool(IDC_CHECK_VISIBILITYEDITOR, FALSE);
	SetBool(IDC_CHECK_VISIBILITYRENDER, FALSE);
	SetFloat(IDC_EDIT_IDC_CHECK_AMBIENTINTENSITY, 0.2,0.0,1.0,0.1);
	SetBool(IDC_CHECK_WRITEANIMATION, TRUE);
	SetFloat(IDC_EDIT1_NOOFKEYFRAMES, 76);
	SetBool(IDC_CHECK_EXPORTNORMALS, TRUE); 
	return TRUE;
}

Bool StartDialog::AskClose()
{			
	//Eigentlich haette das auch in Command gehoert, aber das wird bei mir nicht aufgerufen
	//Werte von Check und Combobox werden erfragt und in die entsprechenden private variablen geschrieben
	GetBool(IDC_CHECK_COPYTEXTURES,mExportTextures);
	GetInt32(IDC_COMBO_TEXTUREEXPORT,mExportOption);
	GetBool(IDC_CHECK_SETTEXTUREDOBJECTCOLORWHITE,mObjectColorWhite);
	GetInt32(IDC_COMBO_DATAFORMAT,mDataFormat);
	GetBool(IDC_CHECK_VISIBILITYEDITOR,mVisiblityEditor);
	GetBool(IDC_CHECK_VISIBILITYRENDER,mVisiblityRender);
	GetFloat(IDC_EDIT_IDC_CHECK_AMBIENTINTENSITY, mAmbientIntensity);
	GetBool(IDC_CHECK_WRITEANIMATION, mWriteAnimation);
	GetFloat(IDC_EDIT1_NOOFKEYFRAMES, mNoOfKeyFrames);
	GetBool(IDC_CHECK_EXPORTNORMALS, mExportNormals);

	//hier checken, ob alle zwingenden Optionen eingestellt wurden, 
	//so lange TRUE zurueckgeben (Dialog wird nicht geschlossen)
	return FALSE;
}


Bool StartDialog::getDLGexportTextures(){return mExportTextures;}
Int32 StartDialog::getDLGexportOption(){return mExportOption;} 
Bool StartDialog::getDLGObjectColorWhite(){return mObjectColorWhite;}
Int32 StartDialog::getDLGdataFormat(){return mDataFormat;}
Bool StartDialog::getDLGvisibilityEditor(){return mVisiblityEditor;}	
Bool StartDialog::getDLGvisibilityRender(){return mVisiblityRender;}	
Float StartDialog::getDLGambientIntensity(){return mAmbientIntensity;}
Bool StartDialog::getDLGwriteAnimation(){return mWriteAnimation;}
Float StartDialog::getDLGnoOfKeyFrames(){return mNoOfKeyFrames;}
Bool StartDialog::getDLGexportNormals(){return mExportNormals;}


