#include "ConfirmTextureCopy.h"

Bool ConfirmTextureCopyDLG::CreateLayout()
{
	SetTitle("Dataconflict!");
	AddStaticText(100011, 0, 0, 0, "Do you want to replace the older file: " +mFilename +" with the new one?", 0);
    AddDlgGroup(DLG_OK | DLG_CANCEL);
	return TRUE;
}

void ConfirmTextureCopyDLG::setFilename(String filename){mFilename = filename;}
