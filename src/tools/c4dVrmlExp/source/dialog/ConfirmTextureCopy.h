#ifndef VRML_DIALOG_CONFIRM_H
#define VRML_DIALOG_CONFIRM_H
#include "VRMLexp.h"
#include "c4d_symbols.h"

///Zustaendig fuer den StartDialog. Abgeleitet von ModalDialog
class ConfirmTextureCopyDLG : public GeModalDialog
{
public:
	///Functions to override
	virtual Bool CreateLayout(void);	///siehe SDK-Docu		

	void setFilename(String filename);

private:
	String mFilename;

		
};
#endif
