#ifndef VRML_EXP_WRITEGEOM_H
#define VRML_EXP_WRITEGEOM_H

#include "../scenesaver/VRMLexp.h"


 void WriteGeometricObjects(class VRMLSAVE &vrml, Matrix up,class VRMLmgmt *dataMgmt);  //Exportiert geometrische Objekten 

//Methoden für WriteGeometricObjects
 void WriteShape(VRMLSAVE &vrml, String knotennameC4D, VRMLmgmt *dataMgmt);					///Schreibt den Shape-Knoten
 void WriteCoordinate(VRMLSAVE &vrml,VRMLmgmt *dataMgmt);									///Schreibt die Polygon Coordinaten
 void WriteCoordinateIndex(VRMLSAVE &vrml,VRMLmgmt* dataMgmt);								///Schreibt den Polygon Coordinaten Index
 void WriteNormals(VRMLSAVE &vrml, VRMLmgmt *dataMgmt );									///Schreibt die Normalen
 void WriteNormalsIndex(VRMLSAVE &vrml,VRMLmgmt* dataMgmt );								///Schreibt den Normalen Index
 void WriteUVCoords(VRMLSAVE &vrml,VRMLmgmt *dataMgmt );									///Schreibt die UV-Coordinaten
 void WriteUVIndex(VRMLSAVE &vrml, VRMLmgmt *dataMgmt );									///Schreibt den UV-Coordinaten Index
 void CloseIndexedFaceSetAndShape(VRMLSAVE &vrml);											///Schließt den IndexedFaceSet- und Shape- Knoten 


#endif

