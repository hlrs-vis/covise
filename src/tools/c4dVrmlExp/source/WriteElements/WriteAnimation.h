//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WriteAnimation: 																												//
//																																//
//Die Methode WriteAnimation() organisiert den Export von Annimationen . In ihr werden alle weiteren dafür 						//
//benötigten Methoden aufgerufen. 																								//
//																																//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#ifndef VRML_EXP_WRITEANI_H
#define VRML_EXP_WRITEANI_H

#include "../scenesaver/VRMLexp.h"


void WriteAnimation(VRMLSAVE &vrml, BaseDocument *doc, BaseObject *op, Bool parentEditorMode, Bool parentRenderMode, Bool timeSensorWritten);	//Schreibt die Animation raus und ruft die dazugehörigen Methoden auf

//Methoden für WriteAnimation
Bool WriteTimeSensor(VRMLSAVE &vrml);	///Schreibt den TimeSensor
void WritePositionInterpolator(VRMLSAVE &vrml,BaseObject *op, String knotenname);	///Schreibt den Animations Positions Interpolator
void WriteOrientationInterpolator(VRMLSAVE &vrml, BaseDocument *doc, BaseObject *op,String knotenname);		///Schreibt den Animations Orientation Interpolator
//void WriteCoordinateInterpolator(VRMLSAVE &vrml,BaseObject *op, String knotenname);


#endif
