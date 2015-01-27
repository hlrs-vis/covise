/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                     (C) 2001 VirCinity **
 **                                                                        **
 ** Description:   COVISE ReadPlot3D FreeCut module                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                       **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 ** Date: 26.09.97                                                         **
 **                                                                        **
 ** changed to newAPI + further FreeCut:                                 **
 ** 30.10.2001        Sven Kufer                                           **
 **                   VirCinity IT-Consulting GmbH                         **
 **                   Nobelstrasse 15                                      **
 **                   70569 Stuttgart                                      **
 **                                                                        **
\**************************************************************************/

#include "FreeCut.h"
#include <iostream>
#include <vector>
#include <cstddef>
#include <stdlib.h>
#include <cmath>
#include <do/coDoPolygons.h>
#include <do/coDoOctTreeP.h>
#include <do/coDoOctTree.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <util/coVector.h>
#include <util/coWristWatch.h>
#include <algorithm>

coWristWatch ww_;

FreeCut::FreeCut(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Calculate Plot3D-Solution data")
{
    p_grid = addInputPort("grid", "UnstructuredGrid", "Input mesh");
    p_data = addInputPort("data", "Float|Vec3", "Input data");
    p_surface = addInputPort("surface", "Polygons", "Input surface");
    p_surfaceOut = addOutputPort("surfaceOut", "Polygons", "cuttingsurface");
    p_dataOut = addOutputPort("dataOut", "Float|Vec3", "interpolated data");
}

//bool FreeCut::myfunction (int i,int j) { return (i<j); }
bool myfunction(std::vector<float> i, std::vector<float> j) { return (i[3] < j[3]); }
bool mySortFunction2(std::vector<int> i, std::vector<int> j)
{
    if ((i[2] < j[2]) || ((i[2] == j[2]) && (i[3] < j[3])))
    {
        return (true);
    }
    else
    {
        return (false);
    }
}
bool mySortFunction3(std::vector<int> i, std::vector<int> j) { return (i[2] < j[2]); }

int FreeCut::compute(const char *) //Modul ist executed
{
    ww_.reset();

    float toleranz = 1e-4f;
    int out_numPoints, out_numCorners, out_numPolygons;
    int numConn;
    int *inElemList, *inConnList, *inTypeList;
    float *x_coord, *y_coord, *z_coord;
    int numElements, numPolygons;
    int *inPolygonList, *inCornerList;
    //int *outPolygonList, *outCornerList;
    numVertices = 0;
    vector<int> outPolygonList(0);
    int *polygonList_out;
    int *cornerList_out;
    vector<int> outCornerList(0);
    int PolygonNummer = 0;
    float *x_polyStart, *y_polyStart, *z_polyStart;
    float *out_x_Start, *out_y_Start, *out_z_Start;
    float *outData;
    int outDataLength;
    int numCoordPoly;
    const coDoOctTreeP *octtree;
    const coDoOctTree *gridOcttree;
    bool gotScalarData = false;
    float *VecData1, *VecData2, *VecData3;
    bool gotVectorData = false;
    vector<float> outScalarData;
    vector<float> outVecData1;
    vector<float> outVecData2;
    vector<float> outVecData3;
    vector<float> ScalarData_temp;
    float *inScalarData;
    bool einmalig = false;
    //Hole das Grid-Objekt vom Port ab und ueberpruefe ob es
    //sich um ein unstrukturiertes Gitter handelt!
    const coDistributedObject *GridDo = p_grid->getCurrentObject();
    if (GridDo != NULL) //Wenn GridDo ungleich Null, dann wurden Daten abgeholt
    {
        if (const coDoUnstructuredGrid *unsGrd = dynamic_cast<const coDoUnstructuredGrid *>(GridDo))
        {
            //Wenn es sich um ein unstrukturiertes Gitter handelt werden
            //die Daten in den shared-Memorie uebertragen
            char *OctTreeGrid = new char[strlen(unsGrd->getName()) + 8];
            strcpy(OctTreeGrid, unsGrd->getName());
            strcat(OctTreeGrid, "_octree");
            int numCoord; //, hasTypes;
            unsGrd->getGridSize(&numElements, &numConn, &numCoord); //erstellt die Netzdimension
            unsGrd->getTypeList(&inTypeList);
            unsGrd->getAddresses(&inElemList, &inConnList, &x_coord, &y_coord, &z_coord);
            gridOcttree = unsGrd->GetOctTree(NULL, OctTreeGrid);
        }
        else //Wenn es sich um kein unstrukturiertes Gitter handelt
        {
            sendError("Illegal object type at port %s : %s",
                      p_grid->getName(), GridDo->getType());
            return FAIL;
        }
    }
    else //Es konnten keine Daten abgeholt werden
    {
        sendError("no data object at port %s",
                  p_grid->getName());
        return FAIL;
    }

    //Hole Data-Objekt am Port ab

    const coDistributedObject *DataDo = p_data->getCurrentObject();
    if (DataDo != NULL) //Daten wurden abgeholt
    {
        if (const coDoFloat *ustsdt = dynamic_cast<const coDoFloat *>(DataDo)) //ueberpruefe den Typ
        {
            sendInfo("Daten wurden als Skalardaten erkannt!");
            //int num_values;
            inScalarData = ustsdt->getAddress();
            gotScalarData = true;
        }
        else
        {
            if (const coDoVec3 *ustvdt = dynamic_cast<const coDoVec3 *>(DataDo)) //ueberpruefe Daten
            {
                //auf Vektortyp
                ustvdt->getAddresses(&VecData1, &VecData2, &VecData3);
                gotVectorData = true;
            }
            else
            {
                sendError("Illegal object type at port %s : %s",
                          p_data->getName(), DataDo->getType());
            }
        }
    }
    else //Es konnten keine Daten abgeholt werden
    {
        sendError("no data object at port %s",
                  p_data->getName());
        return FAIL;
    }
    //Hole Surface-Objekt am Port ab

    const coDistributedObject *SurfaceDo = p_surface->getCurrentObject();
    if (SurfaceDo != NULL) //Daten wurden abgeholt
    {
        if (const coDoPolygons *polygn = dynamic_cast<const coDoPolygons *>(SurfaceDo)) //ueberpruefe den Typ
        {
            fprintf(stderr, "Polygone wurden erkannt!\n");
            char *OctTreePolygons = new char[strlen(polygn->getName()) + 8];
            strcpy(OctTreePolygons, polygn->getName());
            strcat(OctTreePolygons, "_octree");
            //int num_corners;
            numPolygons = polygn->getNumPolygons();
            numVertices = polygn->getNumVertices();
            numCoordPoly = polygn->getNumPoints();
            polygn->getAddresses(&x_polyStart, &y_polyStart, &z_polyStart, &inCornerList, &inPolygonList);
            octtree = polygn->GetOctTree(NULL, OctTreePolygons); //Erstelle Octree
        }
        else
        {
            sendError("Illegal object type at port %s : %s",
                      p_surface->getName(), SurfaceDo->getType());
            return FAIL;
        }
    }
    else //Es konnten keine Daten abgeholt werden
    {
        sendError("no data object at port %s",
                  p_surface->getName());
        return FAIL;
    }

    //KANTENLISTEN FUER DIE UNSTRUKTURIERTEN GITTERTYPEN

    //(Bsp.: Prsima-Kante 0: von prism_start[0] = 1 bis prism_end[0] = 2)
    //Prisma
    int prism_start[9] = { 0, 1, 2, 3, 4, 5, 0, 1, 2 };
    int prism_end[9] = { 1, 2, 0, 4, 5, 3, 3, 4, 5 };
    //Tetraeder
    int tetra_start[6] = { 0, 1, 2, 0, 1, 2 };
    int tetra_end[6] = { 1, 2, 0, 3, 3, 3 };
    //Pyramide
    int pyr_start[8] = { 0, 1, 2, 3, 0, 1, 2, 3 };
    int pyr_end[8] = { 1, 2, 3, 0, 4, 4, 4, 4 };
    //Hexaeder
    int hex_start[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
    int hex_end[12] = { 1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7 };
    //Quadrat
    int quad_start[4] = { 0, 1, 2, 3 };
    int quad_end[4] = { 1, 2, 3, 0 };
    //Dreieck
    int tri_start[3] = { 0, 1, 2 };
    int tri_end[3] = { 1, 2, 0 };
    //Gerade
    int bar_start[1] = { 0 };
    int bar_end[1] = { 1 };

    vector<float> XCutVek(0), XCutVek_temp(0);
    vector<float> YCutVek(0), YCutVek_temp(0);
    vector<float> ZCutVek(0), ZCutVek_temp(0);
    coVector point1;
    coVector point2;
    std::vector<std::vector<float> > myCutVeks;
    std::vector<std::vector<float> > myCutVeks_temp;
    std::vector<float> CutVek_temp;

    /*const char* TypName[] = {"LEER\n", "BAR\n", "TRIANGLE\n", "QUAD\n", "TETRAHEDRON\n", "PYRAMID\n",
	"PRISM\n", "HEXAHEDRON\n", "POINT\n", "POLYHEDRON\n"};*/
    int KantenArrayLaenge = 0;

    //int *octreePolygonList;

    std::vector<int> OctreeGridElemList;
    ///Octree der Gitterelemente einbinden und Elementeauswahl auf wesentliche beschraenken
    //lade Bbox des PolygonenOctree
    std::vector<float> PolygonBbox;
    const int *cellList1 = octtree->getBbox(PolygonBbox);
    //for( int v = 0; v < 6; v++){
    //	fprintf(stderr,"Bbox[%d] = %f\n",v,PolygonBbox[v]);
    //}
    //finde heraus, welche Zellen des Gridelemente-Octree innerhalb oder auf der Polygonen-Bbox liegen
    //Fall 1 - Zellen liegen innerhalb der Polygonen-Bbox
    //Fall 2 - Zellen liegen auf der Polygonen-Bbox (teilweise auserhalb)
    //const int *CellList0 = gridOcttree->extended_search(,,OctreeGridElemList);
    ///Ende Octree der Gitterelemente
    OctreeGridElemList.clear();
    const int *cellList2 = gridOcttree->area_search(PolygonBbox, OctreeGridElemList);
    //fprintf(stderr,"OctreeGridElemList.size() = %d\n",(int) OctreeGridElemList.size());
    sort(OctreeGridElemList.begin(), OctreeGridElemList.end());

    /*for(int h = 0; h < 20; h++){
		fprintf(stderr,"OctreeGridElemList[%d] = %d\n",h,OctreeGridElemList[h]);
	}*/
    Covise::sendInfo("Gitterzellensuche: %6.3f s", ww_.elapsed());
    ///HIER GEHTS LOS! /////////////////////////////////////////////////////////////////////////////////
    int schleifendurchlaeufe = 0;
    int i = 0;
    int status = 10000;
    std::vector<std::vector<int> > KantenVektor;

    //for(int i=0; i<numElements; i++)//Kopf ohne Grid-Octree
    for (int x = 0; x < (int)OctreeGridElemList.size(); x++)
    {
        //int x = 0;///nur fuer version ohne gitterzellen-octree
        if (x == status)
        {
            fprintf(stderr, "Fortschritt: %d / %d\n", x, (int)OctreeGridElemList.size());
            status = status + 10000;
        }
        if ((x == 0) || (i != OctreeGridElemList[x]))
        {
            schleifendurchlaeufe++;
            //Einleitung fuer GridOctree
            i = OctreeGridElemList[x];
            //Pruefe welcher Element-Typ vorliegt

            char currType = inTypeList[i];
            int max = 0;

            if (currType == TYPE_PRISM)
            {
                max = 9;
            }
            if (currType == TYPE_TETRAHEDER)
            {
                max = 6;
            }
            if (currType == TYPE_PYRAMID)
            {
                max = 8;
            }
            if (currType == TYPE_HEXAEDER)
            {
                max = 12;
            }
            if (currType == TYPE_HEXAGON)
            {
                max = 12;
            }
            if (currType == TYPE_QUAD)
            {
                max = 4;
            }
            if (currType == TYPE_TRIANGLE)
            {
                max = 3;
            }
            if (currType == TYPE_BAR)
            {
                max = 1;
            }

            vector<int> PolyHedStart, PolyHedEnd;
            if (currType == TYPE_POLYHEDRON) // Fall POLYHEDRON - Typ 11
            {
                int AnzPunkte;
                int Start_u;

                if (i == (numElements - 1))
                {
                    AnzPunkte = numConn - inElemList[i];
                }
                else
                {
                    AnzPunkte = inElemList[i + 1] - inElemList[i];
                }

                Start_u = 0;
                int z = 0;
                std::vector<int> StartEndeKnoten;
                KantenVektor.clear();
                PolyHedStart.clear();
                PolyHedEnd.clear();
                for (int u = 0; u < (AnzPunkte - 1); u++)
                {
                    StartEndeKnoten.clear();
                    z = u + 1;
                    if ((inConnList[inElemList[i] + u]) < (inConnList[inElemList[i] + z]))
                    {
                        StartEndeKnoten.push_back(u);
                        StartEndeKnoten.push_back(z);
                        StartEndeKnoten.push_back(inConnList[inElemList[i] + u]);
                        StartEndeKnoten.push_back(inConnList[inElemList[i] + z]);
                        KantenVektor.push_back(StartEndeKnoten);
                        PolyHedStart.push_back(u);
                        PolyHedEnd.push_back(z);
                    }
                    if ((inConnList[inElemList[i] + u]) > (inConnList[inElemList[i] + z]))
                    {
                        StartEndeKnoten.push_back(z);
                        StartEndeKnoten.push_back(u);
                        StartEndeKnoten.push_back(inConnList[inElemList[i] + z]);
                        StartEndeKnoten.push_back(inConnList[inElemList[i] + u]);
                        KantenVektor.push_back(StartEndeKnoten);
                        PolyHedStart.push_back(z);
                        PolyHedEnd.push_back(u);
                    }
                    if (inConnList[inElemList[i] + Start_u] == inConnList[inElemList[i] + z])
                    {
                        if ((u + 2) < (AnzPunkte - 1))
                        {
                            Start_u = u + 2;
                            u++;
                        }
                        else
                        {
                            u = AnzPunkte;
                        }
                    }
                }
                sort(KantenVektor.begin(), KantenVektor.end(), mySortFunction2);
                //sort( KantenVektor.begin(),KantenVektor.end(),mySortFunction3 );

                std::vector<int> temp;
                /*if( einmalig == false )
				{
					fprintf(stderr,"Anzahl Kanten = %d\n",(int) KantenVektor.size());
					for( int b = 0; b < (int) KantenVektor.size(); b++)
					{
						temp = KantenVektor[b];
						fprintf(stderr,"KantenVektor[%d] = %d / %d \n",b,temp[2],temp[3]);
						temp.clear();
					}
					
				}*/
                max = ((int)KantenVektor.size() / 2);
            }
            if (max == 0)
            {
                sendError("Element-Typ wurde nicht erkannt! Element-Typ = %d", inTypeList[i]);
            }

            //int *Ende= new int[max];
            //int *Start= new int[max];
            std::vector<int> Start(max);
            std::vector<int> Ende(max);

            if (currType == TYPE_PRISM) //Typ Prisma - Typ 6
            {
                for (int a = 0; a < max; a++)
                {
                    Start[a] = prism_start[a];
                    Ende[a] = prism_end[a];
                }
            }
            if (currType == TYPE_TETRAHEDER) //Typ Tetraeder - Typ 4
            {
                for (int a = 0; a < max; a++)
                {
                    Start[a] = tetra_start[a];
                    Ende[a] = tetra_end[a];
                }
            }
            if (currType == TYPE_PYRAMID) //Typ Pyramide - Typ 5
            {
                for (int a = 0; a < max; a++)
                {
                    Start[a] = pyr_start[a];
                    Ende[a] = pyr_end[a];
                }
            }
            if (currType == TYPE_HEXAEDER) //Typ Hexaeder - Typ 7
            {
                for (int a = 0; a < max; a++)
                {
                    Start[a] = hex_start[a];
                    Ende[a] = hex_end[a];
                }
            }
            if (currType == TYPE_HEXAGON) //ebenfalls Typ 7
            {
                for (int a = 0; a < max; a++)
                {
                    Start[a] = hex_start[a];
                    Ende[a] = hex_end[a];
                }
            }
            if (currType == TYPE_QUAD) //Typ Viereck - Typ 3
            {
                for (int a = 0; a < max; a++)
                {
                    Start[a] = quad_start[a];
                    Ende[a] = quad_end[a];
                }
            }
            if (currType == TYPE_TRIANGLE) //Typ Dreieck - Typ 2
            {
                for (int a = 0; a < max; a++)
                {
                    Start[a] = tri_start[a];
                    Ende[a] = tri_end[a];
                }
            }
            if (currType == TYPE_BAR) //Typ Gerade - Typ 1
            {
                for (int a = 0; a < max; a++)
                {
                    Start[a] = bar_start[a];
                    Ende[a] = bar_end[a];
                }
            }
            if (currType == TYPE_POLYHEDRON) //Typ POLYHEDRON - Typ 8 ???
            {
                int laufindex = 0;
                std::vector<int> temp;
                for (int a = 0; a < (int)PolyHedStart.size(); a = a + 2)
                {
                    temp = KantenVektor[a];
                    Start[laufindex] = temp[0];
                    Ende[laufindex] = temp[1];
                    laufindex++;
                }
                PolyHedStart.clear();
                PolyHedEnd.clear();

                /*if( einmalig == false )
				{
					fprintf(stderr,"Anzahl Kanten = %d\n",max);
					for( int b = 0; b < max; b++)
					{
						fprintf(stderr,"KantenArray[%d] = %d / %d \n",b,Start[b],Ende[b]);
					}
					einmalig = true;
				}*/
            }

            //Deklaration
            coVector PolyStuetz; //Stuetzvektor der Gerade aus Polygondaten
            coVector PolyRicht1;
            coVector PolyRicht2; //Richtungsvektor der Gerade aus Poly..
            vector<float> GridKoord(3);
            coVector GridStuetz; //Stuetzvektor der Gerade aus Gitterdaten
            coVector GridRicht; //Richtungsvektor der Gerade aus Gitter..
            coVector NormVek; //Normalenvektor der Ebene
            float PolyLambda1, PolyLambda2; //, PolyLambda2Nenner, PolyLambda2Zaehler, PolyLambda1Nenner, PolyLambda1Zaehler;
            //float GridLambda, GridLambdaNenner;//, GridLambdaZaehler;
            //float testvar;
            float lambda; //, lambdaNenner, lambdaZaehler;
            //float *Punktsuche;
            std::vector<int> OctreePolygonList;

            //Schleife ueber alle Kanten
            for (int g = 0; g < max; g++) //Schleife ueber alle Kanten
            {

                GridStuetz[0] = x_coord[inConnList[inElemList[i] + Start[g]]];
                GridStuetz[1] = y_coord[inConnList[inElemList[i] + Start[g]]];
                GridStuetz[2] = z_coord[inConnList[inElemList[i] + Start[g]]];

                GridKoord[0] = x_coord[inConnList[inElemList[i] + Ende[g]]];
                GridKoord[1] = y_coord[inConnList[inElemList[i] + Ende[g]]];
                GridKoord[2] = z_coord[inConnList[inElemList[i] + Ende[g]]];

                GridRicht[0] = x_coord[inConnList[inElemList[i] + Ende[g]]] - GridStuetz[0];
                GridRicht[1] = y_coord[inConnList[inElemList[i] + Ende[g]]] - GridStuetz[1];
                GridRicht[2] = z_coord[inConnList[inElemList[i] + Ende[g]]] - GridStuetz[2];

                point1[0] = GridStuetz[0];
                point1[1] = GridStuetz[1];
                point1[2] = GridStuetz[2];
                point2[0] = GridKoord[0];
                point2[1] = GridKoord[1];
                point2[2] = GridKoord[2];

                //Suche aus Octree alle Polygone, die für einen Schnitt in die engere Auswahl kommen
                OctreePolygonList.clear();
                const int *cellList0 = octtree->extended_search(point1, point2, OctreePolygonList);
                //fprintf(stderr,"Anzahl Polygone = %d / %d\n",(int) OctreePolygonList.size(), numPolygons);
                //Schneide alle Suchtreffer mit der aktuell gewaehlten Kante des Gitters
                //for(int o = 0; o < numPolygons; o++) //Schleifenkopf ohne Octree
                for (int o = 0; o < OctreePolygonList.size(); o++)
                {
                    int PolygonNummer = OctreePolygonList[o];
                    //int PolygonNummer = o;
                    //Polygone koennen mehr als 3 Ecken haben, daher muss genaue Anzahl ermittelt werden
                    int AnzPolyPkte;
                    if (PolygonNummer == (numPolygons - 1))
                    {
                        AnzPolyPkte = numVertices - inPolygonList[PolygonNummer];
                    }
                    else
                    {
                        AnzPolyPkte = inPolygonList[PolygonNummer + 1] - inPolygonList[PolygonNummer];
                    }

                    //Teile Polygon in Dreiecks-Ebenen auf (z.b.4Ecken -> 2 Dreiecke)
                    lambda = 0;
                    PolyLambda1 = 0;
                    PolyLambda2 = 0;
                    bool weg1 = false;
                    bool weg2 = false;
                    bool weg3 = false;
                    bool weg4 = false;
                    bool weg5 = false;
                    bool weg6 = false;

                    for (int s = 2; s < AnzPolyPkte; s++)
                    {
                        //Stuetzvektorbedatung Gerade aus Polygon
                        PolyStuetz[0] = x_polyStart[inCornerList[inPolygonList[PolygonNummer] + s]];
                        PolyStuetz[1] = y_polyStart[inCornerList[inPolygonList[PolygonNummer] + s]];
                        PolyStuetz[2] = z_polyStart[inCornerList[inPolygonList[PolygonNummer] + s]];
                        //Richtungsvektorbedatung
                        PolyRicht1[0] = x_polyStart[inCornerList[inPolygonList[PolygonNummer] + (s - 1)]] - PolyStuetz[0];
                        PolyRicht1[1] = y_polyStart[inCornerList[inPolygonList[PolygonNummer] + (s - 1)]] - PolyStuetz[1];
                        PolyRicht1[2] = z_polyStart[inCornerList[inPolygonList[PolygonNummer] + (s - 1)]] - PolyStuetz[2];

                        PolyRicht2[0] = x_polyStart[inCornerList[inPolygonList[PolygonNummer]]] - PolyStuetz[0];
                        PolyRicht2[1] = y_polyStart[inCornerList[inPolygonList[PolygonNummer]]] - PolyStuetz[1];
                        PolyRicht2[2] = z_polyStart[inCornerList[inPolygonList[PolygonNummer]]] - PolyStuetz[2];

                        /// GAUSS ELIMINATION
                        /// Berechnung der Lambdas!

                        const int n = 3;
                        float A[n][n]; //Matrix des LGS
                        //float L[n][n];	//Untere linke Dreiecksmatrix
                        //float R[n][n];	//Obere rechte Dreiecksmatrix
                        float y[n]; //Hilfsvektor
                        float b[n]; //Vektor der Eingangsgroessen des LGS ... rechte Seite
                        //double x, sum;

                        bool lambdaausGl1 = true;
                        bool lambdaausGl2 = true;
                        bool lambdaausGl3 = true;
                        if ((GridRicht[0] > -0.0000001) && (GridRicht[0] < 0.0000001)) //Wenn GridRicht in x-Ri Null, kann lambda nicht aus 1.Gl bestimmt werden!
                        {
                            lambdaausGl1 = false;
                        }
                        if ((GridRicht[1] > -0.0000001) && (GridRicht[1] < 0.0000001)) // Wenn GridRicht iny-Ri Null, kann lambda nicht aus 2.Gl bestimmt werden!
                        {
                            lambdaausGl2 = false;
                        }
                        if ((GridRicht[2] > -0.0000001) && (GridRicht[2] < 0.0000001)) //Wenn GridRicht...
                        {
                            lambdaausGl3 = false;
                        }

                        bool lambda1ausGl1 = true;
                        bool lambda1ausGl2 = true;
                        bool lambda1ausGl3 = true;
                        if ((PolyRicht1[0] > -0.0000001) && (PolyRicht1[0] < 0.0000001)) //Wenn PolyRicht1 in x-Ri Null, kann lambda nicht aus 1.Gl bestimmt werden!
                        {
                            lambda1ausGl1 = false;
                        }
                        if ((PolyRicht1[1] > -0.0000001) && (PolyRicht1[1] < 0.0000001)) // Wenn PolyRicht1 in y-Ri Null, kann lambda nicht aus 2.Gl bestimmt werden!
                        {
                            lambda1ausGl2 = false;
                        }
                        if ((PolyRicht1[2] > -0.0000001) && (PolyRicht1[2] < 0.0000001)) //Wenn PolyRicht1...
                        {
                            lambda1ausGl3 = false;
                        }

                        bool lambda2ausGl1 = true;
                        bool lambda2ausGl2 = true;
                        bool lambda2ausGl3 = true;
                        if ((PolyRicht2[0] > -0.0000001) && (PolyRicht2[0] < 0.0000001)) //Wenn PolyRicht2 in x-Ri Null, kann lambda nicht aus 1.Gl bestimmt werden!
                        {
                            lambda2ausGl1 = false;
                        }
                        if ((PolyRicht2[1] > -0.0000001) && (PolyRicht2[1] < 0.0000001)) // Wenn PolyRicht2 in y-Ri Null, kann lambda nicht aus 2.Gl bestimmt werden!
                        {
                            lambda2ausGl2 = false;
                        }
                        if ((PolyRicht2[2] > -0.0000001) && (PolyRicht2[2] < 0.0000001)) //Wenn PolyRicht2...
                        {
                            lambda2ausGl3 = false;
                        }

                        //3 Ausgangsvarianten zur Bestimmung der Lambdas. Aus Gleichung 1, 2 oder 3.
                        if ((lambdaausGl1 == true) && (((lambda1ausGl2 == true) && (lambda2ausGl3 == true)) || ((lambda1ausGl3 == true) && (lambda2ausGl2 == true)))) //Variante1
                        {
                            A[0][0] = GridRicht[0];
                            A[0][1] = -PolyRicht1[0];
                            A[0][2] = -PolyRicht2[0];
                            b[0] = PolyStuetz[0] - GridStuetz[0];

                            if ((lambda1ausGl2 == true) && (lambda2ausGl3 == true))
                            {
                                A[1][0] = GridRicht[1];
                                A[1][1] = -PolyRicht1[1];
                                A[1][2] = -PolyRicht2[1];
                                b[1] = PolyStuetz[1] - GridStuetz[1];
                                A[2][0] = GridRicht[2];
                                A[2][1] = -PolyRicht1[2];
                                A[2][2] = -PolyRicht2[2];
                                b[2] = PolyStuetz[2] - GridStuetz[2];
                                weg1 = true;
                            }
                            else
                            {
                                if ((lambda1ausGl3 == true) && (lambda2ausGl2 == true))
                                {
                                    A[1][0] = GridRicht[2];
                                    A[1][1] = -PolyRicht1[2];
                                    A[1][2] = -PolyRicht2[2];
                                    b[1] = PolyStuetz[2] - GridStuetz[2];
                                    A[2][0] = GridRicht[1];
                                    A[2][1] = -PolyRicht1[1];
                                    A[2][2] = -PolyRicht2[1];
                                    b[2] = PolyStuetz[1] - GridStuetz[1];
                                    weg2 = true;
                                }
                                else
                                {
                                    sendInfo("This is impossible!");
                                }
                            }
                        }
                        else
                        {
                            if ((lambdaausGl2 == true) && (((lambda1ausGl1 == true) && (lambda2ausGl3 == true)) || ((lambda1ausGl3 == true) && (lambda2ausGl1 == true)))) //Variante2
                            {
                                A[0][0] = GridRicht[1];
                                A[0][1] = -PolyRicht1[1];
                                A[0][2] = -PolyRicht2[1];
                                b[0] = PolyStuetz[1] - GridStuetz[1];

                                if ((lambda1ausGl1 == true) && (lambda2ausGl3 == true))
                                {
                                    A[1][0] = GridRicht[0];
                                    A[1][1] = -PolyRicht1[0];
                                    A[1][2] = -PolyRicht2[0];
                                    b[1] = PolyStuetz[0] - GridStuetz[0];
                                    A[2][0] = GridRicht[2];
                                    A[2][1] = -PolyRicht1[2];
                                    A[2][2] = -PolyRicht2[2];
                                    b[2] = PolyStuetz[2] - GridStuetz[2];
                                    weg3 = true;
                                }
                                else
                                {
                                    if ((lambda1ausGl3 == true) && (lambda2ausGl1 == true))
                                    {
                                        A[1][0] = GridRicht[2];
                                        A[1][1] = -PolyRicht1[2];
                                        A[1][2] = -PolyRicht2[2];
                                        b[1] = PolyStuetz[2] - GridStuetz[2];
                                        A[2][0] = GridRicht[0];
                                        A[2][1] = -PolyRicht1[0];
                                        A[2][2] = -PolyRicht2[0];
                                        b[2] = PolyStuetz[0] - GridStuetz[0];
                                        weg4 = true;
                                    }
                                    else
                                    {
                                        sendInfo("This is impossible!");
                                    }
                                }
                            }
                            else
                            {
                                if ((lambdaausGl3 == true) && (((lambda1ausGl1 == true) && (lambda2ausGl2 == true)) || ((lambda1ausGl2 == true) && (lambda2ausGl1 == true)))) //Variante3
                                {
                                    A[0][0] = GridRicht[2];
                                    A[0][1] = -PolyRicht1[2];
                                    A[0][2] = -PolyRicht2[2];
                                    b[0] = PolyStuetz[2] - GridStuetz[2];

                                    if ((lambda1ausGl1 == true) && (lambda2ausGl2 == true))
                                    {
                                        A[1][0] = GridRicht[0];
                                        A[1][1] = -PolyRicht1[0];
                                        A[1][2] = -PolyRicht2[0];
                                        b[1] = PolyStuetz[0] - GridStuetz[0];
                                        A[2][0] = GridRicht[1];
                                        A[2][1] = -PolyRicht1[1];
                                        A[2][2] = -PolyRicht2[1];
                                        b[2] = PolyStuetz[1] - GridStuetz[1];
                                        weg5 = true;
                                    }
                                    else
                                    {
                                        if ((lambda1ausGl2 == true) && (lambda2ausGl1 == true))
                                        {
                                            A[1][0] = GridRicht[1];
                                            A[1][1] = -PolyRicht1[1];
                                            A[1][2] = -PolyRicht2[1];
                                            b[1] = PolyStuetz[1] - GridStuetz[1];
                                            A[2][0] = GridRicht[0];
                                            A[2][1] = -PolyRicht1[0];
                                            A[2][2] = -PolyRicht2[0];
                                            b[2] = PolyStuetz[0] - GridStuetz[0];
                                            weg6 = true;
                                        }
                                        else
                                        {
                                            sendInfo("This is impossible!");
                                        }
                                    }
                                }
                                else
                                {
                                    //sendInfo("Unbekannter LGS-Fall oder LGS ist nicht berechenbar!");
                                }
                            }
                        }
                        if ((weg1 == true) || (weg2 == true) || (weg3 == true) || (weg4 == true) || (weg5 == true) || (weg6 == true))
                        {
                            //Vorwärtselimination! Zerlegung von A in R und L mit lösen des Hilfsvektors y
                            for (int u = 0; u < n; u++) // Matrixeintraege uebertragen bzw. Nullen
                            {
                                for (int j = u; j < n; j++)
                                { //Bestimmen von R
                                    for (int k = 0; k < u; k++)
                                    {
                                        A[u][j] = A[u][j] - A[u][k] * A[k][j];
                                    }
                                }
                                for (int j = u + 1; j < n; j++)
                                { //Bestimmen von L
                                    for (int k = 0; k < u; k++)
                                    {
                                        A[j][u] = A[j][u] - A[j][k] * A[k][u];
                                    }
                                    A[j][u] = A[j][u] / A[u][u];
                                }
                            }
                            //Loesen der Vorwaertselimination
                            for (int a = 0; a < n; a++)
                            {
                                y[a] = b[a];
                                for (int k = 0; k < a; k++)
                                {
                                    y[a] = y[a] - A[a][k] * y[k];
                                }
                            }
                            //Rueckwaertseinsetzen
                            for (int a = n - 1; a > -1; a--)
                            {
                                b[a] = y[a] / A[a][a];
                                for (int k = a + 1; k < n; k++)
                                {
                                    b[a] = b[a] - A[a][k] * b[k] / A[a][a];
                                }
                            }

                            /// ENDE GAUSS ELIMINATION
                            //Bedingung, dass Schnittpunkt im Dreieck liegt:
                            //0 <= Lambda1 <= 1 und 0 <= Lambda2 <= 1 und Lambda1 + Lambda2 <=1
                            float testvariable;
                            lambda = b[0];
                            PolyLambda1 = b[1];
                            PolyLambda2 = b[2];
                            testvariable = PolyLambda1 + PolyLambda2;

                            if ((PolyLambda1 > (-toleranz)) && (PolyLambda1 < (1 + toleranz)) && (PolyLambda2 < (1 + toleranz)) && (PolyLambda2 > (-toleranz)) && (testvariable < (1 + toleranz)) && (testvariable > (-toleranz)) && (lambda > (-toleranz)) && (lambda < (1 + toleranz)))
                            {
                                //fprintf(stderr,"Kante %d, Element %d, Polygon %d\n",g,i,PolygonNummer);

                                float XCut = GridStuetz[0] + lambda * (GridRicht[0]);
                                float YCut = GridStuetz[1] + lambda * (GridRicht[1]);
                                float ZCut = GridStuetz[2] + lambda * (GridRicht[2]);

                                XCutVek_temp.push_back(XCut);
                                YCutVek_temp.push_back(YCut);
                                ZCutVek_temp.push_back(ZCut);

                                ///Alternative (Vektor enthaelt x,y,z Koordinaten, Winkel und Skalardaten)
                                CutVek_temp.push_back(XCut);
                                CutVek_temp.push_back(YCut);
                                CutVek_temp.push_back(ZCut);
                                CutVek_temp.push_back(0); //Platzhalter für Winkel
                                ///
                                if (gotScalarData == true)
                                {
                                    int pos1, pos2; //Sind die momentan verwendeten Knoten des Gitters
                                    float wert1, wert2, interpWert;
                                    pos1 = inConnList[inElemList[i] + Start[g]];
                                    pos2 = inConnList[inElemList[i] + Ende[g]];
                                    wert1 = inScalarData[pos1];
                                    wert2 = inScalarData[pos2];
                                    interpWert = wert1 + lambda * (wert2 - wert1);
                                    ScalarData_temp.push_back(interpWert);
                                    CutVek_temp.push_back(interpWert);
                                }
                                if (gotVectorData == true)
                                {
                                    int pos1, pos2;
                                    float interpWert1, interpWert2, interpWert3;
                                    pos1 = inConnList[inElemList[i] + Start[g]];
                                    pos2 = inConnList[inElemList[i] + Ende[g]];
                                    interpWert1 = VecData1[pos1] + lambda * (VecData1[pos2] - VecData1[pos1]);
                                    interpWert2 = VecData2[pos1] + lambda * (VecData2[pos2] - VecData2[pos1]);
                                    interpWert3 = VecData3[pos1] + lambda * (VecData3[pos2] - VecData3[pos1]);

                                    CutVek_temp.push_back(interpWert1);
                                    CutVek_temp.push_back(interpWert2);
                                    CutVek_temp.push_back(interpWert3);
                                }
                                myCutVeks_temp.push_back(CutVek_temp);
                                CutVek_temp.clear();
                            }
                            else
                            {
                                //fprintf(stderr,"Kein Schittpunkt berechnet!\n");
                            }
                        }
                    }
                }
            }
            ///Vortsetzung Alternative
            if ((int)myCutVeks_temp.size() > 2)
            {
                //fprintf(stderr,"Anz Schnittpunkte = %d\n",(int) myCutVeks_temp.size());
                //Geometrischen Mittelpunkt ermitteln
                float Xmittel, Ymittel, Zmittel;
                Xmittel = 0;
                Ymittel = 0;
                Zmittel = 0;
                for (int w = 0; w < (int)myCutVeks_temp.size(); w++)
                {
                    std::vector<float> temp = myCutVeks_temp[w];
                    Xmittel = Xmittel + temp[0];
                    Ymittel = Ymittel + temp[1];
                    Zmittel = Zmittel + temp[2];
                }
                Xmittel = Xmittel / ((int)myCutVeks_temp.size());
                Ymittel = Ymittel / ((int)myCutVeks_temp.size());
                Zmittel = Zmittel / ((int)myCutVeks_temp.size());
                //fprintf(stderr,"Xmittel = %f\n",Xmittel);
                //fprintf(stderr,"Ymittel = %f\n",Ymittel);
                //fprintf(stderr,"Zmittel = %f\n",Zmittel);

                //Referenz/Startvektor bestimmen
                coVector Referenzvektor;
                std::vector<float> ersterSchnittpunkt = myCutVeks_temp[0];
                Referenzvektor[0] = ersterSchnittpunkt[0] - Xmittel;
                Referenzvektor[1] = ersterSchnittpunkt[1] - Ymittel;
                Referenzvektor[2] = ersterSchnittpunkt[2] - Zmittel;
                //Normalenvektor bestimmen
                coVector Referenzvektor2;
                coVector Normalenvektor;
                std::vector<float> zweiterSchnittpunkt;
                for (int l = 1; l < (int)myCutVeks_temp.size(); l++)
                {
                    zweiterSchnittpunkt = myCutVeks_temp[l];
                    if ((zweiterSchnittpunkt[0] != ersterSchnittpunkt[0]) || (zweiterSchnittpunkt[1] != ersterSchnittpunkt[1]) || (zweiterSchnittpunkt[2] != ersterSchnittpunkt[2]))
                    {
                        Referenzvektor2[0] = zweiterSchnittpunkt[0] - Xmittel;
                        Referenzvektor2[1] = zweiterSchnittpunkt[1] - Ymittel;
                        Referenzvektor2[2] = zweiterSchnittpunkt[2] - Zmittel;
                        Normalenvektor = Referenzvektor.cross(Referenzvektor2);
                        l = (int)myCutVeks_temp.size();
                    }
                }
                coVector Kreuzprodukt;
                //Restliche Vektoren und Winkel/Bogenmase zum Refernzvektor bestimmen
                float Betrag, aBetrag, bBetrag;
                float Skalarprodukt;
                float cosinusX;
                float X;
                const float PI = M_PI;
                float Volumen;
                coVector Vektor_temp;
                //vector<float> winkelArray(0);
                //winkelArray.push_back(0);
                ersterSchnittpunkt[3] = 0; //Winkel des ersten Schnittpunktes laut Def = 0
                myCutVeks.push_back(ersterSchnittpunkt); //enthaelt auser Schnittpunkt auch Winkeldaten
                aBetrag = sqrt(Referenzvektor[0] * Referenzvektor[0] + Referenzvektor[1] * Referenzvektor[1] + Referenzvektor[2] * Referenzvektor[2]);
                for (int t = 1; t < (int)myCutVeks_temp.size(); t++)
                {
                    std::vector<float> temp = myCutVeks_temp[t];
                    Vektor_temp[0] = temp[0] - Xmittel;
                    Vektor_temp[1] = temp[1] - Ymittel;
                    Vektor_temp[2] = temp[2] - Zmittel;
                    X = 0;
                    bBetrag = sqrt(Vektor_temp[0] * Vektor_temp[0] + Vektor_temp[1] * Vektor_temp[1] + Vektor_temp[2] * Vektor_temp[2]);
                    Betrag = aBetrag * bBetrag;

                    Skalarprodukt = Referenzvektor.dot(Vektor_temp);
                    if ((Betrag > 0))
                    {
                        cosinusX = Skalarprodukt / Betrag;

                        if ((cosinusX < 1) && (cosinusX > -1))
                        {
                            X = acos(cosinusX) * 180 / PI; //Winkel in Grad
                        }
                        else
                        {
                            if (cosinusX >= 1)
                            {
                                X = 0;
                            }
                            if (cosinusX <= -1)
                            {
                                X = 180;
                            }
                        }
                        //Spatprodukt um Drehsinn festzustellen
                        Kreuzprodukt = Referenzvektor.cross(Vektor_temp);
                        Volumen = Kreuzprodukt.dot(Normalenvektor);

                        if ((Volumen <= 0) && (X > 0))
                        {
                            X = 360 - X;
                        }
                    }

                    temp[3] = X;
                    myCutVeks.push_back(temp);
                }
                //Punkte in Bezug auf ihren Winkel zum Referenzvektor sortieren
                sort(myCutVeks.begin(), myCutVeks.end(), myfunction);

                //fprintf(stderr,"myCutVeks.size = %d\n",(int) myCutVeks.size());
                std::vector<float> temp3 = myCutVeks[0];
                //fprintf(stderr,"temp3[1] = %f, temp3[2] = %f\n",temp3[1],temp3[2]);
                std::vector<float> temp1;
                std::vector<float> temp2;

                for (int a = 2; a < (int)myCutVeks.size(); a++)
                {
                    temp1 = myCutVeks[a];
                    temp2 = myCutVeks[a - 1];

                    //fprintf(stderr,"a = %d\n",a);
                    if ((temp3[3] == temp1[3]) || (temp3[3] == temp2[3]) || (temp1[3] == temp2[3]))
                    {
                        //springe zum naechsten Schnittpunkt!
                    }
                    else
                    {

                        outPolygonList.push_back((int)outCornerList.size());

                        outCornerList.push_back((int)XCutVek.size());
                        XCutVek.push_back(temp1[0]);
                        YCutVek.push_back(temp1[1]);
                        ZCutVek.push_back(temp1[2]);
                        if (gotScalarData == true)
                        {
                            outScalarData.push_back(temp1[4]);
                        }
                        if (gotVectorData == true)
                        {
                            outVecData1.push_back(temp1[4]);
                            outVecData2.push_back(temp1[5]);
                            outVecData3.push_back(temp1[6]);
                        }

                        outCornerList.push_back((int)XCutVek.size());
                        XCutVek.push_back(temp2[0]);
                        YCutVek.push_back(temp2[1]);
                        ZCutVek.push_back(temp2[2]);
                        if (gotScalarData == true)
                        {
                            outScalarData.push_back(temp2[4]);
                        }
                        if (gotVectorData == true)
                        {
                            outVecData1.push_back(temp2[4]);
                            outVecData2.push_back(temp2[5]);
                            outVecData3.push_back(temp2[6]);
                        }

                        outCornerList.push_back((int)XCutVek.size());
                        XCutVek.push_back(temp3[0]);
                        YCutVek.push_back(temp3[1]);
                        ZCutVek.push_back(temp3[2]);
                        if (gotScalarData == true)
                        {
                            outScalarData.push_back(temp3[4]);
                        }
                        if (gotVectorData == true)
                        {
                            outVecData1.push_back(temp3[4]);
                            outVecData2.push_back(temp3[5]);
                            outVecData3.push_back(temp3[6]);
                        }
                    }
                    temp1.clear();
                    temp2.clear();
                }

                //fprintf(stderr,"Anz Polygone = %d\n",(int) outPolygonList.size());
            }

            ///
            ///Generieren des Ausgangsvektors durch Triangulation der Schnittpunkte

            else
            {
                //fprintf(stderr,"Weniger als 3 Schnittpunkte!\n");
            }
            //fprintf(stderr,"Anz Polygone: %d\n",(int) outPolygonList.size());
            //fprintf(stderr,"Anz Ecken: %d\n",(int) outCornerList.size());
            //fprintf(stderr,"Anz Koordinaten: %d\n",(int) XCutVek.size());

            XCutVek_temp.clear();
            YCutVek_temp.clear();
            ZCutVek_temp.clear();
            ScalarData_temp.clear();
            CutVek_temp.clear();
            myCutVeks_temp.clear();
            myCutVeks.clear();

            //delete[] Ende;
            //delete[] Start;
            Start.clear();
            Ende.clear();
            /// Ende Triangulation
        }
        //Covise::sendInfo("Dauer fuer 1 Element: %6.5f s", ww_.elapsed());
    }
    fprintf(stderr, "Schleifendurchlaeufe = %d\n", schleifendurchlaeufe);
    // generiere Output
    if ((int)outPolygonList.size() > 0)
    {
        out_x_Start = &XCutVek[0];
        out_y_Start = &YCutVek[0];
        out_z_Start = &ZCutVek[0];
        polygonList_out = &outPolygonList[0];
        cornerList_out = &outCornerList[0];
        out_numPoints = (int)XCutVek.size();
        out_numCorners = (int)outCornerList.size();
        out_numPolygons = (int)outPolygonList.size();

        coDoPolygons *surface_out = new coDoPolygons(p_surfaceOut->getObjName(), out_numPoints, out_x_Start, out_y_Start, out_z_Start, out_numCorners, cornerList_out, out_numPolygons, polygonList_out);
        p_surfaceOut->setCurrentObject(surface_out);
    }
    else
    {
        sendError("Keine Schnittpunkte berechnet!\n");
    }
    if (((int)outScalarData.size() > 0) && (gotScalarData == true))
    {
        outData = &outScalarData[0];
        outDataLength = (int)outScalarData.size();

        coDoFloat *data_out = new coDoFloat(p_dataOut->getObjName(), outDataLength, outData);
        p_dataOut->setCurrentObject(data_out);
    }
    else if (gotScalarData == true)
    {
        sendError("Skalardaten wurden nicht interpoliert!\n");
    }
    if (((int)outVecData1.size() > 0) && (gotVectorData == true))
    {
        float *data1, *data2, *data3;
        int outVecDataLength;
        data1 = &outVecData1[0];
        data2 = &outVecData2[0];
        data3 = &outVecData3[0];
        outVecDataLength = (int)outVecData1.size();

        coDoVec3 *data_out = new coDoVec3(p_dataOut->getObjName(), outVecDataLength, data1, data2, data3);
        p_dataOut->setCurrentObject(data_out);
    }
    else if (gotVectorData == true)
    {
        sendError("Vectordaten wurden nicht interpoliert!\n");
    }
    sendInfo("done"); //done
    ///

    ///
    Covise::sendInfo("complete run: %6.3f s", ww_.elapsed());

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Tools, FreeCut)
