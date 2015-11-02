//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// WriteGeometry: 																												//
//																																//
//Die Methode WriteGeometricObjects() organisiert den Export geometrischer Objekte. In ihr werden weitere dafür 				//
//benötigte Methoden aufgerufen. 																								//
//																																//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



#include "WriteGeometry.h"


void WriteGeometricObjects(VRMLSAVE &vrml, Matrix up, VRMLmgmt *dataMgmt)
{					
	WriteShape(vrml, dataMgmt->getKnotenName(), dataMgmt);	
	WriteCoordinate(vrml,dataMgmt);
	if(vrml.getStartDialog()->getDLGexportNormals()){WriteNormals(vrml,dataMgmt);}
	WriteCoordinateIndex(vrml,dataMgmt);

	BaseTag *tag = dataMgmt->getOp()->GetFirstTag();
	while (tag)	
	{
		if (tag->GetType()==Tuvw)
		{
			WriteUVCoords(vrml,dataMgmt);
			WriteUVIndex(vrml,dataMgmt);
		}
		tag=tag->GetNext();
	}	
	if(vrml.getStartDialog()->getDLGexportNormals()){WriteNormalsIndex(vrml,dataMgmt);}
	CloseIndexedFaceSetAndShape(vrml);
}




 void WriteCoordinate(VRMLSAVE &vrml,VRMLmgmt *dataMgmt)
{	
	vrml.increaseLevel();
	vrml.writeC4DString("geometry DEF "+dataMgmt->getKnotenName() +"-GEOMETRY IndexedFaceSet {\n");
	vrml.increaseLevel();
	vrml.writeC4DString("solid TRUE\n");

	//Schreibt die Polygon Koordinaten, ein Wuerfel wird so ausgegeben: 
	//			-100 -100 -100, #0 links unten hinten
	//			-100 100 -100,  #1 links oben hinten
	//          -100 -100 100, 	#2 links unten vorn
	//          -100 100 100,   #3 links oben vorn
	//          100 -100 100,   #4 rechts unten vorn
	//          100 100 100, 	#5 rechts oben vorn
	//          100 -100 -100,  #6 rechts oben hinten
	//          100 100 -100,   #7 rechts unten hinten

	Int32 pointInRow = 3; //Wieviele Punkte sollen in einer Reihe ausgegeben werden	
	Int32 i;
	vrml.writeC4DString("coord DEF "+dataMgmt->getKnotenName() +"-COORD Coordinate { \n"); 
	vrml.increaseLevel();
	vrml.writeC4DString("point ["); 
	vrml.noIndent();
	for (i=0; i<(dataMgmt->getPcnt()); i++)
	{	
		//Ortsvektoren werden ausgegeben //a,b,c jeweils Ortsvektor fuer eine Ecke des Polygon
		if(i==0)vrml.noIndent(); //Kein Einschub
		vrml.writeC4DString(String::FloatToString(dataMgmt->getPadr()[i].z) +" " +String::FloatToString(dataMgmt->getPadr()[i].y) +" " +String::FloatToString(dataMgmt->getPadr()[i].x) +", ");
		//vrml.writeC4DString(String::FloatToString(dataMgmt->getPadr()[i].x) +" " +String::FloatToString(dataMgmt->getPadr()[i].y) +" " +String::FloatToString(dataMgmt->getPadr()[i].z) +", ");
		vrml.noIndent(); //Kein Einschub
		if ((i%pointInRow == 0) && i!=0){vrml.writeC4DString("\n");}
	}
	vrml.noIndent();
	vrml.writeC4DString("] \n");
	vrml.decreaseLevel();
	vrml.writeC4DString("}\n");  
}		

void WriteCoordinateIndex(VRMLSAVE &vrml,VRMLmgmt* dataMgmt)
{		
	//Gibt die Indeces fuer die Koordinaten aus
	//Fuer diese Punkte: 
	//			-100 -100 -100, #0 links unten hinten
	//			-100 100 -100,  #1 links oben hinten
	//          -100 -100 100, 	#2 links unten vorn
	//          -100 100 100,   #3 links oben vorn
	//          100 -100 100,   #4 rechts unten vorn
	//          100 100 100, 	#5 rechts oben vorn
	//          100 -100 -100,  #6 rechts unten hinten
	//          100 100 -100,   #7 rechts oben hinten
	//Wird folgendes ausgegeben: 
	/*		0, 1, 3, 2, -1,		#linkes Quadrat
	2, 3, 5, 4, -1,		#vorderes Quadrat
	4, 5, 7, 6, -1,		#rechtes Quadrat
	6, 7, 1, 0, -1,		#hinteres Quadrat
	1, 7, 5, 3, -1,		#oberes Quadrat
	6, 0, 2, 4, -1,		#unteres Quadat

	Das ist richtig, 
	*/

	Int32    i;
	vrml.writeC4DString("coordIndex  [ \n");
	for (i=0; i<dataMgmt->getVcnt(); i++)
	{			 
		//a,b,c,d Vertices Liste wird ausgegeben
		if(dataMgmt->getVadr()[i].c==dataMgmt->getVadr()[i].d)
		{
			//vrml.writeC4DString(String::IntToString(dataMgmt->getVadr()[i].a) + ", " + String::IntToString(dataMgmt->getVadr()[i].b) + ", " + String::IntToString(dataMgmt->getVadr()[i].c) + ", -1,\n");
			vrml.writeC4DString(String::IntToString(dataMgmt->getVadr()[i].c) + ", " + String::IntToString(dataMgmt->getVadr()[i].b) + ", " + String::IntToString(dataMgmt->getVadr()[i].a) + ", -1,\n");
		}
		else
		{
			//vrml.writeC4DString(String::IntToString(dataMgmt->getVadr()[i].a) + ", " + String::IntToString(dataMgmt->getVadr()[i].b) + ", " + String::IntToString(dataMgmt->getVadr()[i].c) + ", " + String::IntToString(dataMgmt->getVadr()[i].d) + ", -1,\n");
			vrml.writeC4DString(String::IntToString(dataMgmt->getVadr()[i].d) + ", " + String::IntToString(dataMgmt->getVadr()[i].c) + ", " + String::IntToString(dataMgmt->getVadr()[i].b) + ", " + String::IntToString(dataMgmt->getVadr()[i].a) + ", -1,\n");
		}
	}
	vrml.writeC4DString("]\n");  
}




 void WriteNormals(VRMLSAVE &vrml, VRMLmgmt *dataMgmt )
{
	//Falls kein Phong Tag vorliegt, soll einfach einer erstellt werden und nach rausschreiben der Normalen wieder gelöscht werden
	Bool hasPhongTag = FALSE;
	BaseTag *tag = dataMgmt->getOp()->GetFirstTag(); 
	while (tag)
	{
		if( tag->GetType()==Tphong)   //Abfrage ob das Objekt PhongNormals hat, wenn nicht dann weder die Normalen noch die Indeces rausschreiben
		{ 
			hasPhongTag = TRUE;
		}
	tag= tag->GetNext();
	}
	if (!hasPhongTag)
	{
		dataMgmt->getOp()->MakeTag(Tphong);
	}

	Int32 pointInRow = 3; //Wieviele Punkte sollen in einer Reihe ausgegeben werden
	tag = dataMgmt->getOp()->GetFirstTag();

	while (tag)
	{	
		if( tag->GetType()==Tphong)   //Abfrage ob das Objekt PhongNormals hat, wenn nicht dann weder die Normalen noch die Indeces rausschreiben
		{   
			Vector32* phongNormal = ToPoly(dataMgmt->getOp())->CreatePhongNormals();  //creates normals for each of a polygon's points
			Int32 punktnummer = 0;
			vrml.writeC4DString("normalPerVertex TRUE \n");
			vrml.writeC4DString("normal Normal {\n");
			vrml.increaseLevel();
			vrml.writeC4DString("vector[ ");
			vrml.increaseLevel();
			for (Int32 i=0; i<dataMgmt->getVcnt(); i++)
			{			 
				if((dataMgmt->getVadr())[i].c==(dataMgmt->getVadr())[i].d)
				{
					for (Int32 j=0; j<3; j++)
					{
						if(i==0)vrml.noIndent(); //Kein Einschub
						vrml.writeC4DString(String::FloatToString(phongNormal[punktnummer].z) + " " + String::FloatToString(phongNormal[punktnummer].y) + " " + String::FloatToString(phongNormal[punktnummer].x) + ", ");
						//vrml.writeC4DString(String::FloatToString(phongNormal[punktnummer].x) + " " + String::FloatToString(phongNormal[punktnummer].y) + " " + String::FloatToString(phongNormal[punktnummer].z) + ", ");
						vrml.noIndent(); //Kein Einschub
						punktnummer++;
					}
					punktnummer++;  //Da immer 4 Punkte geben werden muss hier auch noch hochgezaehlt werden (4. Pkt. wird uebersprungen)
					vrml.writeC4DString("\n");
				}
				else
				{
					for (Int32 j=0; j<4; j++)
					{
						if(i==0)vrml.noIndent(); //Kein Einschub
						//vrml.writeC4DString(String::FloatToString(phongNormal[punktnummer].x) + " " + String::FloatToString(phongNormal[punktnummer].y) + " " + String::FloatToString(phongNormal[punktnummer].z) + ", ");
						vrml.writeC4DString(String::FloatToString(phongNormal[punktnummer].z) + " " + String::FloatToString(phongNormal[punktnummer].y) + " " + String::FloatToString(phongNormal[punktnummer].x) + ", ");
						vrml.noIndent(); //Kein Einschub
						punktnummer++;
					}
					vrml.writeC4DString("\n");
				}
			}
			vrml.writeC4DString("]\n");
			vrml.decreaseLevel();
			vrml.decreaseLevel();
			vrml.writeC4DString("}\n"); 
			DeleteMem(phongNormal);
		}//end if( tag->GetType()==Tphong)
		tag = tag->GetNext();	
	} //end while
	
	//Lösche Phong Tag, falls dieser extra erstellt wurde
	if (!hasPhongTag)
	{
		dataMgmt->getOp()->KillTag(Tphong);
	}
}


 void WriteUVCoords(VRMLSAVE &vrml,VRMLmgmt *dataMgmt )
{
	Vector texturCoord;
	Int32 cnt = 0;		 //Nummer des aktuellen TuvwTag. 
	UVWTag			*uvwTag;

	Int32 tuvCnt = 0;
	Int32 texturCnt = 0;

	//Partikel Tag auslesen
	BaseTag			*tag = dataMgmt->getOp()->GetFirstTag(); 
	while (tag)
	{
		if( tag->GetType()==Tuvw)   
		{	
			tuvCnt++;
			cnt++;
			uvwTag = static_cast<UVWTag*>(tag);
			Int32    i;  

			//Wenn mehere Tuvw Tags da sind muessen jedes mal neue TextCoord ausgegeben werden. Ist nur ein Tag da kann man Use nehmen. 
			vrml.writeC4DString("texCoord DEF " +dataMgmt->getKnotenName() +"-TEXCOORD" +String::IntToString(cnt) +" TextureCoordinate { \n"); 
			vrml.increaseLevel();
			vrml.writeC4DString("point [ \n");

			UVWStruct uvwKoord;
			Int32 dataCount = uvwTag->GetDataCount();
			ConstUVWHandle data = uvwTag->GetDataAddressR();
			

			for (i=0; i<dataMgmt->getVcnt(); i++)
			{			 
				//
				//uvwTag->Get(,i,uvwKoord);  //Int32 i, The index of the polygon to get the coordinates for.
				
		//UVWStruct uvwKoord = uvwTag->GetSlow(i);

				UVWTag::Get(data, i, uvwKoord);
				if(dataMgmt->getVadr()[i].c==dataMgmt->getVadr()[i].d)
				{
					/*//Vektoren spiegeln an der x -Achse
					Vector a = Vector(uvwKoord.a.x,uvwKoord.a.y,uvwKoord.a.z);
					Vector b = Vector(uvwKoord.b.x,uvwKoord.b.y,uvwKoord.b.z);
					Vector c = Vector(uvwKoord.c.x,uvwKoord.c.y,uvwKoord.c.z);
					Vector d = Vector(uvwKoord.d.x,uvwKoord.d.y,uvwKoord.d.z);
					Matrix mirrorX = Matrix(dataMgmt->getOp()->GetRelPos(), Vector(1,0,0),Vector(0,-1,0), Vector(0,0,-1));
					//gespiegelte Vektoren
					a = mirrorX * a;
					b = mirrorX * b;
					c = mirrorX * c;
					d = mirrorX * d;
					
					vrml.writeC4DString(String::FloatToString(a.x) +" " +String::FloatToString(a.y) +", " +String::FloatToString(b.x) +" " +String::FloatToString(b.y) +", " +String::FloatToString(c.x) +" " +String::FloatToString(c.y) +",\n");  //So gehts im Textur-Testfall mit Transformknoten, nicht aber für Speciosa.
					*/
					vrml.writeC4DString(String::FloatToString(uvwKoord.a.x) + " " + String::FloatToString(1 - uvwKoord.a.y) + ", " + String::FloatToString(uvwKoord.b.x) + " " + String::FloatToString(1 - uvwKoord.b.y) + ", " + String::FloatToString(uvwKoord.c.x) + " " + String::FloatToString(1 - uvwKoord.c.y) + ",\n");  //So gehts im Textur-Testfall mit Transformknoten, nicht aber für Speciosa.					
				}
				else
				{
					/*//Vektoren an der x-Achse Spiegeln
					Vector a = Vector(uvwKoord.a.x,uvwKoord.a.y,uvwKoord.a.z);
					Vector b = Vector(uvwKoord.b.x,uvwKoord.b.y,uvwKoord.b.z);
					Vector c = Vector(uvwKoord.c.x,uvwKoord.c.y,uvwKoord.c.z);
					Vector d = Vector(uvwKoord.d.x,uvwKoord.d.y,uvwKoord.d.z);
					Matrix mirrorX = Matrix(dataMgmt->getOp()->GetRelPos(), Vector(1,0,0),Vector(0,-1,0), Vector(0,0,-1));
					
					//gespiegelte Vektoren
					a = mirrorX * a;
					b = mirrorX * b;
					c = mirrorX * c;
					d = mirrorX * d;
						
					vrml.writeC4DString(String::FloatToString(a.x) +" " +String::FloatToString(a.y) +", " +String::FloatToString(b.x) +" " +String::FloatToString(b.y) +", " +String::FloatToString(c.x) +" " +String::FloatToString(c.y) +", " +String::FloatToString(d.x) +" " +String::FloatToString(d.y) +",\n");
					*/
					vrml.writeC4DString(String::FloatToString(uvwKoord.a.x) + " " + String::FloatToString(1 - uvwKoord.a.y) + ", " + String::FloatToString(uvwKoord.b.x) + " " + String::FloatToString(1 - uvwKoord.b.y) + ", " + String::FloatToString(uvwKoord.c.x) + " " + String::FloatToString(1 - uvwKoord.c.y) + ", " + String::FloatToString(uvwKoord.d.x) + " " + String::FloatToString(1 - uvwKoord.d.y) + ",\n");
				}
			}
			vrml.writeC4DString("] \n"); 
			vrml.decreaseLevel();
			vrml.writeC4DString("}\n"); 
		} 

		if( tag->GetType()==Ttexture)
		{
			texturCnt++;
		}
		tag = tag->GetNext();	
	} //end while
	cnt=0;


	//Wenn nur 1 Tuvw Tag vorhanden ist, aber mehrere Texturen sollen fuer alle Texturen die UV Koordinaten des Tags verwendet werden
	if (tuvCnt==1)
	{
		for (Int32 j=2; j<=texturCnt;j++)
		{
			vrml.writeC4DString("texCoord" +String::IntToString(j) +" USE " +dataMgmt->getKnotenName() +"-TEXCOORD1\n");
		}
	}
}


  void WriteUVIndex(VRMLSAVE &vrml, VRMLmgmt *dataMgmt )
{
	//Für jeden TextureTag soll ein texCoordIndex rausgeschrieben werden. 
	Int32 texturCnt = 0;
	BaseTag *tag = dataMgmt->getOp()->GetFirstTag();  
	while (tag)
	{
		if( tag->GetType()==Ttexture)
		{
			texturCnt++;
		}
		tag = tag->GetNext();
	}
	Int32 j=0; 
	
	do
	{
		if (j==0)
		{
			vrml.writeC4DString("texCoordIndex [ \n");  
		}
		else 
		{	
			vrml.writeC4DString("texCoordIndex" +String::IntToString(j+1) +" [ \n");  
		}
		Int32 punktcnt = 0;
		for (Int32 i=0; i<(dataMgmt->getVcnt()); i++)
		{	
			if((dataMgmt->getVadr())[i].c==(dataMgmt->getVadr())[i].d)
			{
				//vrml.writeC4DString(" " + String::IntToString(punktcnt) + ", " + String::IntToString(punktcnt + 1) + ", " + String::IntToString(punktcnt + 2) + ", -1,\n ");
				vrml.writeC4DString(" " + String::IntToString(punktcnt+2) + ", " + String::IntToString(punktcnt + 1) + ", " + String::IntToString(punktcnt) + ", -1,\n ");
				punktcnt+=3;
			}
			else
			{
				//vrml.writeC4DString(" " + String::IntToString(punktcnt) + ", " + String::IntToString(punktcnt + 1) + ", " + String::IntToString(punktcnt + 2) + ", " + String::IntToString(punktcnt + 3) + ", -1,\n ");
				vrml.writeC4DString(" " + String::IntToString(punktcnt+3) + ", " + String::IntToString(punktcnt + 2) + ", " + String::IntToString(punktcnt + 1) + ", " + String::IntToString(punktcnt) + ", -1,\n ");
				punktcnt+=4;
			}
		}
		vrml.writeC4DString("] \n");
		j++;
	}
	while(j<texturCnt);
}


 void WriteNormalsIndex(VRMLSAVE &vrml,VRMLmgmt* dataMgmt )
{
	//Falls kein Phong Tag vorliegt, soll einfach einer erstellt werden und nach rausschreiben der Normalen wieder gelöscht werden
	Bool hasPhongTag = FALSE;
	BaseTag *tag = dataMgmt->getOp()->GetFirstTag(); 
	while (tag)
	{
		if( tag->GetType()==Tphong)   //Abfrage ob das Objekt PhongNormals hat, wenn nicht dann weder die Normalen noch die Indeces rausschreiben
		{ 
			hasPhongTag = TRUE;
		}
	tag= tag->GetNext();
	}
	if (!hasPhongTag)
	{
		dataMgmt->getOp()->MakeTag(Tphong);
	}

	tag = dataMgmt->getOp()->GetFirstTag();
	while (tag)
	{	
		if( tag->GetType()==Tphong)
		{   
			vrml.writeC4DString("normalIndex [ \n");
			Int32 punktCnt = 0;
			for (Int32 i=0; i<(dataMgmt->getVcnt()); i++)
			{			 
				if((dataMgmt->getVadr())[i].c==(dataMgmt->getVadr())[i].d)
				{
					//vrml.writeC4DString(String::IntToString(punktCnt + 0) + ", " + String::IntToString(punktCnt + 1) + ", " + String::IntToString(punktCnt + 2) + ", -1,\n");
					vrml.writeC4DString(String::IntToString(punktCnt + 2) + ", " + String::IntToString(punktCnt + 1) + ", " + String::IntToString(punktCnt + 0) + ", -1,\n");
					punktCnt+=3;
				}
				else
				{
					//vrml.writeC4DString(String::IntToString(punktCnt + 0) + ", " + String::IntToString(punktCnt + 1) + ", " + String::IntToString(punktCnt + 2) + ", " + String::IntToString(punktCnt + 3) + ", -1,\n");
					vrml.writeC4DString(String::IntToString(punktCnt + 3) + ", " + String::IntToString(punktCnt + 2) + ", " + String::IntToString(punktCnt + 1) + ", " + String::IntToString(punktCnt + 0) + ", -1,\n");
					punktCnt+=4;
				}
			}
			vrml.writeC4DString("]\n"); 
		}//end if( tag->GetType()==Tphong)
		tag = tag->GetNext();	
	} //end while

	//Lösche Phong Tag, falls dieser extra erstellt wurde
	if (!hasPhongTag)
	{
		dataMgmt->getOp()->KillTag(Tphong);
	}
}



 void WriteShape(VRMLSAVE &vrml, String knotennameC4D, VRMLmgmt *dataMgmt)
{
	vrml.writeC4DString("DEF "+knotennameC4D +"-SHAPE Shape {\n");
	vrml.increaseLevel();

	//INIT
	BaseChannel* colorChannel = NULL;
	BaseChannel	*kanal = NULL;

	//Color Channel
	Vector ColorColor;
	Float ColorBrightness = 0;
	String ColorTexName = "";
	//Transparency Channel
	Vector TransparencyColor;
	Float TransparencyBrightnes= 0;
	//Specular Channel
	Float SpecularWidth = 0;
	Float SpecularHeight = 0;

	//Sonstiges: 
	Float shininess= 0;
	Float transparency= 0;
	String texturName;
	Int32 materialCnt = 0;




	BaseTag *tag = dataMgmt->getOp()->GetFirstTag();  
	while (tag)
	{
		if (tag->GetType()==Ttexture)
		{
			TextureTag	*texTag=(TextureTag*)tag; //Cast in TextureTag
			BaseMaterial* texMaterial = texTag->GetMaterial();
			BaseContainer *bc = texMaterial->GetDataInstance();

			if(texMaterial)
			{
				texturName ="";
				for (Int32 i = 0; i<MAX_MATERIALCHANNELS; i++)
				{
					Material	*material = NULL;
					if(texMaterial->GetType()==Mmaterial) //Downcast nur moeglich wenn Standard Material.Types hier: file:///C:/Users/flow/Desktop/Studienarbeit/Cinema4D%20Ressources/von%20der%20Maxxon%20Seite/C++%20SDK%20documentation%20for%20R10.5/pages/c4d_basematerial/class_BaseMaterial54.html
					{
						material = static_cast<Material*>(texMaterial);
					}
					TransparencyColor = Vector(1,1,1);
					TransparencyBrightnes = 1;
					
					if (material)
					{
						if (material->GetChannelState(i))  //Wenn der Kanal nicht ausgewaehlt ist, dann wird er ueberspringen
						{
							switch(i)
							{
								//Bisher nur die Material Kanäle Color, Transparency und Specular implementiert
							case CHANNEL_COLOR: //The color channel of the material.
								ColorColor = bc->GetVector(MATERIAL_COLOR_COLOR);				
								ColorBrightness = bc->GetFloat(MATERIAL_COLOR_BRIGHTNESS);	
								kanal = texMaterial->GetChannel(i); 
								texturName = kanal->GetData().GetString(BASECHANNEL_TEXTURE);

								//Sonstige Parameter: 
								//MATERIAL_COLOR_SHADER
								//MATERIAL_COLOR_TEXTUREMIXING
								//MATERIAL_COLOR_TEXTURESTRENGTH
								break;

							case CHANNEL_TRANSPARENCY: //The transparency channel of the material.
								TransparencyColor = bc->GetVector(MATERIAL_TRANSPARENCY_COLOR);
								TransparencyBrightnes = bc->GetFloat(MATERIAL_TRANSPARENCY_BRIGHTNESS);
								//Sonstige Parameter: 
								//MATERIAL_TRANSPARENCY_REFRACTION
								//MATERIAL_TRANSPARENCY_FRESNEL
								//MATERIAL_TRANSPARENCY_ADDITIVE
								//MATERIAL_TRANSPARENCY_SHADER
								//MATERIAL_TRANSPARENCY_DISPERSION
								break;

							case CHANNEL_SPECULAR: //The specular channel of the material.
								SpecularWidth = bc->GetFloat(MATERIAL_SPECULAR_WIDTH, 0);
								SpecularHeight = bc->GetFloat(MATERIAL_SPECULAR_HEIGHT, 0);
								//MATERIAL_SPECULAR_MODE
								//MATERIAL_SPECULAR_WIDTH
								//MATERIAL_SPECULAR_HEIGHT
								//MATERIAL_SPECULAR_FALLOFF
								//MATERIAL_SPECULAR_INNERWIDTH
								break;
								// Für spätere implementierung der andern Kanäle
								/*
								case CHANNEL_LUMINANCE: //The luminance channel of the material.
								break;
								case CHANNEL_REFLECTION: //The reflection channel of the material.
								break;
								case CHANNEL_ENVIRONMENT: //The environment channel of the material.
								break;
								case CHANNEL_FOG: //The fog channel of the material.
								break;
								case CHANNEL_BUMP: //The bump channel of the material.
								break;
								case CHANNEL_ALPHA: //	The alpha channel of the material.
								break;
								case CHANNEL_SPECULARCOLOR: //	The specular color channel of the material.
								break;
								case CHANNEL_GLOW: //The glow channel of the material.
								break;
								*/
							} //End Switch
						}//End if GetChannelState()
					}//End if(material)
				}//End for()



				//Berechnungen: 

				//diffuseColor:
					//diffuseColor ist bisher einfach die Farbe aus dem Color Channel, dass kann noch variiert werden.
					Vector diffuseColor = ColorColor;

					//Optional (nach Dialogeinstellung) die diffuseColor eines Objekts auf Weiß stellen, wenn eine textur draufgemappt ist. 
					if(texturName!="")
					{
						if(vrml.getStartDialog()->getDLGObjectColorWhite())
						{
							diffuseColor = Vector(1,1,1);
						}
					}
				//Shininess: 
					shininess = (SpecularWidth * 0.95f + 0.05f)/10;
				//specularColor:
					// Scheint als ob der C4D interne VRML Exporter das so macht. Kann noch angepasst werden. 
					Float specularColor = shininess; 
				//Transparency: 
					//Luminenzgleichung
					//alle auf eins
					Float rot = TransparencyColor.x;
					Float green = TransparencyColor.y;
					Float blau = TransparencyColor.z;
					Float luminenz = (0.3*rot + 0.59*green + 0.11*blau);
					transparency = 1 - (luminenz * TransparencyBrightnes);

				//ambientIntensity:
					//Wird im Dialog eingestellt. Standartwert ist 0.2
					Float ambientIntensity = vrml.getStartDialog()->getDLGambientIntensity();

				//Nur vom ersten Material soll die Apperance ausgegeben werden, die Kinderknoten sollen den Apperance Knoten mit USE wiederverwenden
				if (materialCnt == 0)		//materialCnt kann ich eigentlich auch noch auf bool umstellen
				{
					dataMgmt->setParentApperance(VRMLName(texMaterial->GetName()));
					vrml.writeC4DString("appearance  DEF " +VRMLName(texMaterial->GetName()) +" Appearance {\n");
					vrml.increaseLevel();
					vrml.writeC4DString("material Material { \n");
					vrml.increaseLevel();
					vrml.writeC4DString("ambientIntensity "+String::FloatToString(ambientIntensity) +" \n");
					vrml.writeC4DString("diffuseColor " +String::FloatToString(diffuseColor.x) +" " +String::FloatToString(diffuseColor.y) +" " +String::FloatToString(diffuseColor.z) +"\n");
					vrml.writeC4DString("specularColor "+String::FloatToString(specularColor) +" " +String::FloatToString(specularColor) +" " +String::FloatToString(specularColor) +"\n");  
					vrml.writeC4DString("shininess "+String::FloatToString(shininess) +"\n");
					vrml.writeC4DString("transparency "+String::FloatToString(transparency) +"\n");
					vrml.decreaseLevel();
					vrml.writeC4DString("} \n");
				}

				//Textur von jedem Material wird nun ausgegeben, wird eine Textur aber in einem anderen Channel (Transparency usw.) oder ueber Layer eingestellt wird dies noch nicht beachtet
				//Hier das Hashset fuer die Texturen rein. Kommt eine Textur schon mal vor soll USE benutzte werden	
				
				if (texturName != "")
				{
					if (vrml.texNameHashSet.find(StringC4DtoStringSTD(texturName)) != vrml.texNameHashSet.end())
					{
						//Dann USE Verwenden
						vrml.writeC4DString("texture USE TEXTURE_" +VRMLName(texturName) +"\n"); 
					}
					else 
					{
						//DEF verwenden und die Textur in das HashSet eintragen
						vrml.writeC4DString("texture DEF TEXTURE_" +VRMLName(texturName) +" ImageTexture{ url \"maps/" +texturName); 
						vrml.noIndent();
						vrml.writeC4DString("\" }\n");
						vrml.texNameHashSet.insert(StringC4DtoStringSTD(texturName));
					}
				}
				materialCnt++;	
			}//End if(texMaterial)
		}//End (tag->GetType()==Ttexture)
		tag=tag->GetNext();
	}// while (tag)

	if (materialCnt != 0)
	{
		vrml.decreaseLevel();
		vrml.writeC4DString("}\n");  //END WRITE APPERANCE
	}

	if (materialCnt == 0) //gibt es keinen Texture Tag: 
	{
		//Falls das Eltern Objekt einen Appearance Knoten hat, benutze diesen
		if(dataMgmt->getParentApperance()!="")
		{
			vrml.writeC4DString("appearance  USE " +dataMgmt->getParentApperance() +"\n");
		}

		//Falls nicht, schreib einfach einen Appearance Knoten mit einem Standard Materialknoten
		else
		{
			vrml.writeC4DString("appearance Appearance {\n");
			vrml.increaseLevel();
			vrml.writeC4DString("material Material { }\n");
			vrml.decreaseLevel();
			vrml.decreaseLevel();
			vrml.writeC4DString("}\n");  //END WRITE APPERANCE
		}
	}
	//vrml.decreaseLevel();
	//vrml.writeC4DString("}\n");  //END WRITE APPERANCE
}


void CloseIndexedFaceSetAndShape(VRMLSAVE &vrml)
{
	vrml.decreaseLevel(); 
	vrml.writeC4DString("}\n");  //Schliesst IndexedFaceSet 
	vrml.decreaseLevel();
	vrml.decreaseLevel();
	vrml.writeC4DString("}\n"); //Schliesst Shape wieder
}


//Winkel umrechnung selbst implementiert. Unnötig
//void WriteTransform(VRMLSAVE &vrml, VRMLmgmt *dataMgmt)
//{
//	vrml.writeC4DString("DEF "+dataMgmt->getKnotenName() +" Transform { \n");
//	vrml.increaseLevel();
//	const Vector &pos = dataMgmt->getOp()->GetPos();  //Get the position vector for this object. These will be local coordinates within its parent object.
//	vrml.writeC4DString("translation "+String::FloatToString(pos.z) +" " +String::FloatToString(pos.y) +" " +String::FloatToString(pos.x) +"\n");
//	const Vector &gr = dataMgmt->getOp()->GetScale();
//	vrml.writeC4DString("scale "+String::FloatToString(gr.x) +" " +String::FloatToString(gr.y) +" " +String::FloatToString(gr.z) +"\n");
//	//Eulerwinkel 
//	const Vector &rot = dataMgmt->getOp()->GetRot();   //Get the HPB rotation of the object relative to any parent object. Return Vector -The HPB rotation  (Eulerwinkel) (Muss auf Achse und Winkel umgerechnet werden --google)
//	//Umrechnung von Eulerwinkel auf Achse Winke
//	Float c1,c2,c3,s1,s2,s3,c1c2,s1s2,x,y,z,angle, w, norm;
//	// http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToAngle/index.htm
//	// Assuming the angles are in radians.
//	c1 = cos(rot.x/2); //heading um y-Achse
//	s1 = sin(rot.x/2);
//	c2 = cos(-rot.z/2); //attitude (Pitch)  um x-Achse
//	s2 = sin(-rot.z/2);
//	c3 = cos(rot.y/2);  //bank(Roll) um z-Achse
//	s3 = sin(rot.y/2);
//
//	c1c2 = c1*c2;
//	s1s2 = s1*s2;
//	w =c1c2*c3 - s1s2*s3;
//	x =c1c2*s3 + s1s2*c3;
//	y =s1*c2*c3 - c1*s2*s3;   
//	z =c1*s2*c3 - s1*c2*s3;
//	angle = 2 * acos(w);
//	norm = x*x+y*y+z*z;
//	if (norm < 0.001) { // when all euler angles are zero angle =0 so
//		// we can set axis to anything to avoid divide by zero
//		x=1;
//		y=z=0;
//	} else {
//		norm = sqrt(norm);
//		x /= norm;
//		y /= norm;
//		z /= norm;
//	}
//	vrml.writeC4DString("rotation "+String::FloatToString(-z) +" " +String::FloatToString(y) +" " +String::FloatToString(x) +" " +String::FloatToString(angle) +"\n");
//	vrml.increaseLevel();
//	vrml.writeC4DString("children [\n");
//	vrml.increaseLevel();
//}
