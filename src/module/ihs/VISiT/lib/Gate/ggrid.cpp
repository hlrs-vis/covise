#include <stdio.h>
#include <string.h>
#ifndef _WIN32
#include <strings.h>
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#ifndef WIN32
#include <iostream>
#endif
#include <stdlib.h>

#include <Gate/include/ggrid.h>
#include <General/include/elements.h>
#include <General/include/profile.h>
#include <Gate/include/gate.h>
#include <Gate/include/ga2cov.h>
#include <General/include/vector.h>
#include <General/include/fatal.h>
#include <General/include/common.h>
#include <General/include/v.h>
#include <Gate/include/stf.h>
#include <Gate/include/leit.h>                    // functions used by CreateGGrid

#ifndef  ILIST_H_INCLUDED
#include <General/include/ilist.h>
#endif

#ifndef  FLIST_H_INCLUDED
#include <General/include/flist.h>
#endif

#define grad2bog M_PI/180.
#define bog2grad 180./M_PI

static void GGridInit(struct ggrid *gg)
{
   gg->epsilon   = 0.003f;
   gg->k         = 0.001f;
   gg->T         = 0.0f;
   gg->bc_inval  = 100;
   gg->bc_outval = 110;
   gg->bc_periodic_left = 120;
   gg->bc_periodic_right = 130;
}


void FreeStructGGrid(struct ggrid *gg)
{
   FreePointStruct(gg->p);
   FreeElementStruct(gg->e);

   FreeIlistStruct(gg->bcin);
   FreeIlistStruct(gg->bcinpol);
   FreeIlistStruct(gg->bcout);
   FreeIlistStruct(gg->bcoutpol);
   FreeIlistStruct(gg->bcwall);
   FreeIlistStruct(gg->bcwallpol);
   FreeIlistStruct(gg->bcperiodic);
   FreeIlistStruct(gg->bcperiodicpol);

   free(gg);
}


static struct ggrid *AllocGGrid(void)
{
   struct ggrid *gg;

   if ((gg = (struct ggrid *)calloc(1, sizeof(struct ggrid))) != NULL)
   {
      gg->p = AllocPointStruct();
      gg->e = AllocElementStruct();

      gg->bcin = AllocIlistStruct(100);
      gg->bcinpol = AllocIlistStruct(25);
      gg->bcinvol = AllocIlistStruct(25);

      gg->bcout = AllocIlistStruct(100);
      gg->bcoutpol = AllocIlistStruct(25);
      gg->bcoutvol = AllocIlistStruct(25);

      gg->bcwall = AllocIlistStruct(500);
      gg->bcwallpol = AllocIlistStruct(125);
      gg->bcwallvol = AllocIlistStruct(125);

      gg->bcperiodic = AllocIlistStruct(500);
      gg->bcperiodicpol = AllocIlistStruct(125);
      gg->bcperiodicvol = AllocIlistStruct(125);
      gg->bcperiodicval = AllocIlistStruct(125);

      gg->bcpressval = AllocFlistStruct(100);

   }
   return gg;
}


struct ggrid *CreateGGrid(struct gate *ga)
{
   struct ggrid *gg;

   if ((gg = AllocGGrid()) != NULL)
   {
      GGridInit(gg);
   }

   printf("\n================= leit-start =================\n\n");

   // ==================================================================
   // Deklaration der benoetigten Variablen
   // ==================================================================

   int i,j;
   //int temp;
   int i_temp;
   //FILE *stream;
   //char datei_steuer[200];
   //char buf[200];
   double bvs_temp;
   int schnitt;
   double delta;
   const int ANZ_GEBIETE = 15;
   int seed;                                      //Geamtknotenanzahl

   //Variablen fuer Berechnungen Gebiet 9
   double x10ru;                                  //Phi des Pkt Gebiet 15 ganz aussen an Grenze zu 10
   double x10lu;                                  //zweiter Eckpunkt 15 - 10
   double x11lu;
   double x11ru;

   //Variablen Einlauf
   double delta_x;
   double delta_y;
   double phi_start;
   double phi_ende;
   double phi;

   //allg. Variablen
   int *randpunkt;
   int *knot_nr;
   int pos;
   double *knotx;
   double *knoty;
   int anz_doppelt;
   int *ersatz_nr;
   double deltax;
   double deltay;
   double TOL;

   //Elementvariablen
   int *el_liste;
   int anz_elemente;
   int k;
   int nr;

   //Leitschaufel
   double *bgl_ls_n, *bgl_ls_k, *phi_ls;
   int anz_ls, anz_punkte_ls;
   //double dreh_x, dreh_y;
   double *ls_x, *ls_y, *ls_r, *ls_phi;
   //double *ls_xa, *ls_ya;
   double *ls_bgl_k;
   double *ls_bgl_n;
   double *ls_bgl_sort_n;                         //fr MESHSEED ls_bgl und ls_phi umsortieren
   double *ls_bgl_sort_k;
   double *ls_phi_sort;
   double ls_beta2;                               //Austrittswinkel
   double ls_hikara;                              //Hinterkantenradius
   double *r_aus;                                 //->Feld der Radien am Ausgang zw. r_max_nabe u. r_max_kranz

   //Nabe
   //double R0_n;
   double *nabex, *nabey, *nabez, *naber;
   int anz_punkte_nabe;
   double *n_z, *n_bgl, *n_r;
   double r_max_nabe;                             //Radius am Ausgang des Berechnungsgebietes

   //Kranz
   //double R0_k;
   double *kranzx, *kranzy, *kranzz, *kranzr;
   int anz_punkte_kranz;
   double *k_z, *k_bgl, *k_r;
   double bgl_max_kranz;
   double z;                                      //Z-Koordinate am Ausgang des Berechnungsgebiets
   double r_max_kranz;                            //Radius am Ausgang des Berechnungsgebietes

   //Druckrandbedingungen
   double *vu2;
   double A2;                                     //Querschnittsflaeche Ausgang
   double ls_hoehe;
   double A1;                                     //Querschnitt Ausgang Leitschaufel;
   double vm2;
   double *p2;
   double rho;
   double vu1;
   double vm1;
   double v1;
   double *v2;

   //===============================================================
   // Netzparameter, read stf
   //===============================================================

   // hub and shroud contour
   anz_punkte_nabe  = ga->phub->nump;
   anz_punkte_kranz = ga->pshroud->nump;
   n_z = new double[anz_punkte_nabe];
   k_z = new double[anz_punkte_kranz];
   n_r = new double[anz_punkte_nabe];
   k_r = new double[anz_punkte_kranz];

   // profile
   /*    if (ga->geofromfile==1)
         anz_punkte_ls = ga->bp->num + 1;    // this was a bit different here ...
       else */
   anz_punkte_ls = ga->bp->num;

   // ls_.. enthaelt zuerst Saugseite (vom Staupunkt aus), dann die Druckseite,
   // dann die ersten beiden Knoten der Skelettlinie
   ls_x   = new double [2*(anz_punkte_ls) + 2];
   ls_y   = new double [2*(anz_punkte_ls) + 2];
   ls_r   = new double [2*(anz_punkte_ls) + 2];
   ls_phi = new double [2*(anz_punkte_ls) + 2];

   struct stf *stf_para;
   if ((stf_para = (struct stf *)calloc(1, sizeof(struct stf))) == NULL)
      printf("not enough memory for (struct stf *)");

   if ( parameter_from_covise(stf_para, ga, &anz_punkte_ls,
      n_r, n_z,
      k_r, k_z,
      ls_x, ls_y,
      ls_r, ls_phi) )
   {
      printf("Netzgenerator hat Parameter von Covise erhalten\n");
   }
   anz_ls = stf_para->anz_ls;

#ifdef DEBUG
   // some debug prints

   printf("anz_punkte_ls: %d\n", anz_punkte_ls);
   printf("anz_punkte_nabe =%d\n", anz_punkte_nabe);
   printf("anz_punkte_kranz=%d\n", anz_punkte_kranz);
#endif

   /*
      // Covise-Version: kein Steuerfile, sondern Parameteruebergabe

      strcpy(datei_steuer,"/mnt/raid/pr/rus00390/mbb/covise/src/application/ihs/VISiT/stf/gate_grid.stf");

      if ( EINLESEN_STEUERFILE(datei_steuer, stf_para) )
      {
         printf("Steuerfile %s eingelesen!\n", datei_steuer);
      }
   */

   // ----------------------------------------------------------------------
   // ANZAHL DER MESHSEEDS FESTSETZEN
   // ----------------------------------------------------------------------

   int ase[ANZ_GEBIETE+1][2];                     //+1, damit Gebiet-Nummer bei 1 anfaengt
   //Ende Netzparameter

   ase[1][0]=stf_para->anz_knot1;
   ase[1][1]=stf_para->anz_grenz;

   ase[2][0]=stf_para->anz_knot2;
   ase[2][1]=stf_para->anz_grenz;                 //muss gleich ase[1][1] sein!

   ase[3][0]=stf_para->anz_knot3;
   ase[3][1]=stf_para->anz_grenz;                 //muss gleich ase[1][1] sein!

   ase[4][0]=stf_para->anz_knot4;
   ase[4][1]=stf_para->anz_grenz;                 //muss gleich ase[1][1] sein!

   ase[5][0]=stf_para->anz_knot3;                 //muss gleich ase[3][0] sein!
   ase[5][1]=stf_para->anz_einlauf;

   ase[6][0]=stf_para->anz_knot2;                 //muss gleich ase[2][0] sein!
   ase[6][1]=stf_para->anz_einlauf;               //muss gleich ase[5][1] sein!

   ase[7][0]=stf_para->anz_einlauf;               //muss gleich ase[5][1] sein!
   ase[7][1]=stf_para->anz_mitte;

   ase[8][0]=stf_para->anz_knot1;                 //muss gleich ase[1][0] sein!
   ase[8][1]=stf_para->anz_mitte;                 //muss gleich ase[7][1] sein!

   ase[9][0]=stf_para->anz_knot9;                 //muss gleich  ase[4][0] - ase[1][0] + 1  sein
   ase[9][1]=stf_para->anz_mitte;                 //muss gleich ase[7][1] sein!

   ase[10][0]=stf_para->anz_knot9;                //muss gleich  ase[4][0] - ase[1][0] + 1  sein
   ase[10][1]=stf_para->anz_grenz;                //muss gleich ase[1][1] sein!

   ase[11][0]=stf_para->anz_mitte;                //muss gleich ase[7][1] sein!
   ase[11][1]=stf_para->anz_grenz;                //musss gleich ase[1][1] sein!

   ase[12][0]=(int)((stf_para->anz_mitte+1)/2);
   ase[12][1]=(int)((stf_para->anz_mitte+1)/2);
   ase[13][0]=(int)((stf_para->anz_mitte+1)/2);
   ase[13][1]=(int)((stf_para->anz_mitte+1)/2);
   ase[14][0]=(int)((stf_para->anz_mitte+1)/2);
   ase[14][1]=(int)((stf_para->anz_mitte+1)/2);

   ase[15][0]=stf_para->anz_breit;
   ase[15][1]=stf_para->anz_15;

   int anz_dreieck= 3 * (int)((stf_para->anz_mitte+1)/2) * (int)((stf_para->anz_mitte+1)/2);

   // Gesamtanzahl der Knoten berechnen!
   seed=0;
   for (j=1;j<12;j++)
   {
      seed += ase[j][0]*ase[j][1];                // Vierecksgebiete
   }
   seed += anz_dreieck;                           // Dreiecksgebiet hinzufgen!
   seed += ase[15][0]*ase[15][1];                 // Nachlauf hinzufgen!

   double *fixr, *fiyr;                           // Zeiger auf Gesamtdatenfeld

   double *fixr1, *fiyr1;                         // Zeiger auf Teildatenfelder
   double *fixr2, *fiyr2;
   double *fixr3, *fiyr3;
   double *fixr4, *fiyr4;
   double *fixr5, *fiyr5;
   double *fixr6, *fiyr6;
   double *fixr7, *fiyr7;
   double *fixr8, *fiyr8;
   double *fixr9, *fiyr9;
   double *fixr10, *fiyr10;
   double *fixr11, *fiyr11;
   double *fixr12, *fiyr12;
   double *fixr15, *fiyr15;

   fixr = new double[2*seed];
   fiyr = new double [2*seed];

   // Teilgebietszeiger setzen!

   i_temp=0;
   fixr1 = fixr;
   fiyr1 = fiyr;

   i_temp += ase[1][0]*ase[1][1];
   fixr2 = fixr+i_temp;
   fiyr2 = fiyr+i_temp;

   i_temp += ase[2][0]*ase[2][1];
   fixr3 = fixr+i_temp;
   fiyr3 = fiyr+i_temp;

   i_temp += ase[3][0]*ase[3][1];
   fixr4 = fixr+i_temp;
   fiyr4 = fiyr+i_temp;

   i_temp += ase[4][0]*ase[4][1];
   fixr5 = fixr+i_temp;
   fiyr5 = fiyr+i_temp;

   i_temp += ase[5][0]*ase[5][1];
   fixr6 = fixr+i_temp;
   fiyr6 = fiyr+i_temp;

   i_temp += ase[6][0]*ase[6][1];
   fixr7 = fixr+i_temp;
   fiyr7 = fiyr+i_temp;

   i_temp += ase[7][0]*ase[7][1];
   fixr8 = fixr+i_temp;
   fiyr8 = fiyr+i_temp;

   i_temp += ase[8][0]*ase[8][1];
   fixr9 = fixr+i_temp;
   fiyr9 = fiyr+i_temp;

   i_temp += ase[9][0]*ase[9][1];
   fixr10 = fixr+i_temp;
   fiyr10 = fiyr+i_temp;

   i_temp += ase[10][0]*ase[10][1];
   fixr11 = fixr+i_temp;
   fiyr11 = fiyr+i_temp;

   i_temp += ase[11][0]*ase[11][1];
   fixr12 = fixr+i_temp;
   fiyr12 = fiyr+i_temp;

   i_temp += anz_dreieck;
   fixr15 = fixr+i_temp;
   fiyr15 = fiyr+i_temp;

   //Randbedingungsvariablen
   //Dimensionen = anzahl der Elemente, nicht der Knoten!
   int anz_wrb_ls = (ase[1][0]+ase[2][0]+ase[3][0]+ase[4][0]-4);
   int anz_elmark = (2*(ase[3][1]+ase[5][1]-2)
      +(ase[5][0]+ase[7][1]+ase[6][0]-3)
      +(ase[11][0]-1)
      +(ase[10][0]-1)
      +2*(ase[15][1]-1)
      +(ase[15][0]-1));
   int anz_kmark = (ase[5][0]+ase[7][1]+ase[6][0]-2);
   int anz_elmark_einlauf = (ase[5][0]+ase[7][1]+ase[6][0]-3);
   int anz_elmark_eli = ase[3][1]+ase[5][1]-2;
   int anz_elmark_ere = ase[2][1]+ase[6][1]-2;
   int anz_elmark_11li = ase[11][0]-1;
   int anz_elmark_10re = ase[10][0]-1;
   int anz_elmark_15li = ase[15][1]-1;
   int anz_elmark_15re = ase[15][1]-1;
   int anz_elmark_auslauf = ase[15][0]-1;

   //int *elmark; // never used
   int *kmark;
   int *elmark_einlauf;
   int *elmark_eli;
   int *elmark_ere;
   int *elmark_11li;
   int *elmark_10re;
   int *elmark_15li;
   int *elmark_15re;
   int *elmark_auslauf;
   int *wrb_ls;

   //===========================================================
   // Endgltige Randpunkte entlang Leitschaufel definieren
   //===========================================================

   double *fixb11, *fiyb11;
   double *fixb12, *fiyb12;

   double *fixb21, *fiyb21;
   double *fixb22, *fiyb22;

   double *fixb31, *fiyb31;
   double *fixb32, *fiyb32;

   double *fixb41, *fiyb41;
   double *fixb42, *fiyb42;

   double *fixb52, *fiyb52;

   double *fixb62, *fiyb62;

   double *fixb92, *fiyb92;

   double *fixb102, *fiyb102;

   double *fixb111, *fiyb111;
   double *fixb112, *fiyb112;

   double *fixb12_2, *fiyb12_2;
   double *fixb12_3, *fiyb12_3;

   double *fixb15_1, *fiyb15_1;
   double *fixb15_2, *fiyb15_2;

   fixb11 = new double [ase[1][0]];
   fiyb11 = new double [ase[1][0]];
   fixb12 = new double [ase[1][0]];
   fiyb12 = new double [ase[1][0]];

   fixb21 = new double [ase[2][0]];
   fiyb21 = new double [ase[2][0]];
   fixb22 = new double [ase[2][0]];
   fiyb22 = new double [ase[2][0]];

   fixb31 = new double [ase[3][0]];
   fiyb31 = new double [ase[3][0]];
   fixb32 = new double [ase[3][0]];
   fiyb32 = new double [ase[3][0]];

   fixb41 = new double [ase[4][0]];
   fiyb41 = new double [ase[4][0]];
   fixb42 = new double [ase[4][0]];
   fiyb42 = new double [ase[4][0]];

   fixb52 = new double [ase[5][0]];
   fiyb52 = new double [ase[5][0]];

   fixb62 = new double [ase[6][0]];
   fiyb62 = new double [ase[6][0]];

   fixb92 = new double [ase[9][0]];
   fiyb92 = new double [ase[9][0]];

   fixb102 = new double [ase[10][0]];
   fiyb102 = new double [ase[10][0]];

   fixb111 = new double [ase[11][0]];
   fiyb111 = new double [ase[11][0]];
   fixb112 = new double [ase[11][0]];
   fiyb112 = new double [ase[11][0]];

   fixb12_2 = new double [ase[11][0]];
   fiyb12_2 = new double [ase[11][0]];
   fixb12_3 = new double [ase[11][0]];
   fiyb12_3 = new double [ase[11][0]];

   fixb15_1 = new double [ase[15][0]];
   fiyb15_1 = new double [ase[15][0]];
   fixb15_2 = new double [ase[15][0]];
   fiyb15_2 = new double [ase[15][0]];

   double fixplo, fiyplo;                         //Punkte am Eintritt
   double fixpro, fiypro;
   double fixpmlo, fiypmlo;
   double fixpmro, fiypmro;

#ifdef DEBUG
   printf("\nEndgueltige Randpunkte entlang Leitschaufel definiert!\n");
#endif

   // ==================================================================
   // Dateieingabe
   // ==================================================================
   /*
      strcpy(datei_steuer, stf_para->nabekranz_pfad);

      if( (stream = fopen( &datei_steuer[0], "r" )) == NULL )
      {
         printf( "Kann '%s' nicht lesen!\n", datei_steuer);
      }
      else   					//Ab mit der Datei in die Arrays!
      {
         fgets(buf,300,stream);

   sscanf( buf, "%d %d", &anz_punkte_nabe, &anz_punkte_kranz);

   n_z = new double[anz_punkte_nabe];
   k_z = new double[anz_punkte_kranz];

   n_r = new double[anz_punkte_nabe];
   k_r = new double[anz_punkte_kranz];

   for(i = 0; i < anz_punkte_nabe; i++)
   {
   fgets(buf,300,stream);          //liest nur eine Zeile
   sscanf( buf, "%lf %lf", &n_r[i], &n_z[i]);
   }

   fgets(buf,300,stream);

   for(i = 0; i < anz_punkte_kranz; i++)
   {
   fgets(buf,300,stream);
   sscanf(buf, "%lf %lf", &k_r[i], &k_z[i]);
   }

   printf("Meridiankontur %s eingelesen!\n", datei_steuer);
   fclose(stream);

   }
   */

   // ==================================================================
   // Leitschaufeldatei oeffnen und x-y-r-phi Koordinaten auslesen, Felder anlegen
   // ==================================================================
   /*
      strcpy(datei_steuer, stf_para->leitxyz_pfad);
      if( (stream = fopen( &datei_steuer[0], "r" )) == NULL )
      {
         printf( "Kann '%s' nicht lesen!\n", datei_steuer);
      }

      else					//Ab mit der Datei in die Arrays!
      {
         fgets(buf,300,stream);
         sscanf( buf, "%d", &anz_ls);

   fgets(buf,300,stream);
   sscanf( buf, "%d", &anz_punkte_ls);

   fgets(buf,300,stream);
   sscanf( buf, "%lf %lf", &dreh_x, &dreh_y);

   // ls_.. enthaelt zuerst Saugseite (vom Staupunkt aus), dann die Druckseite,
   // dann die ersten beiden Knoten der Skelettlinie
   ls_x   = new double [2*anz_punkte_ls + 2];
   ls_y   = new double [2*anz_punkte_ls + 2];
   ls_xa  = new double [2*anz_punkte_ls + 2];
   ls_ya  = new double [2*anz_punkte_ls + 2];
   ls_r   = new double [2*anz_punkte_ls + 2];
   ls_phi = new double [2*anz_punkte_ls + 2];

   for (i = 0; i < 2*anz_punkte_ls + 2; i++)
   {
   fgets(buf,300,stream);
   sscanf(buf,"%d%lf%lf", &temp, &ls_xa[i], &ls_ya[i]);
   }

   printf("Leitschaufelprofil %s eingelesen!\n", datei_steuer);
   fclose(stream);
   }

   double gamma;
   double r;

   //Leitschaufeldrehung
   for(i = 0; i < 2*anz_punkte_ls + 2; i++)
   {

   ls_x[i] = ( ls_xa[i] - dreh_x );
   ls_y[i] = ( ls_ya[i] - dreh_y );

   gamma = bog2grad * atan2 ( ls_y[i] , ls_x[i] );
   r = pow ( (pow(ls_x[i],2) + pow ( ls_y[i] , 2 ) ) , 0.5 );

   ls_x[i] = r * cos ( grad2bog * (gamma+stf_para->deltagamma) ) + dreh_x;
   ls_y[i] = r * sin ( grad2bog * (gamma+stf_para->deltagamma) ) + dreh_y;

   ls_r[i] = pow ( (pow(ls_x[i],2) + pow(ls_y[i],2) ) , 0.5);
   gamma = atan2 ( ls_y[i], ls_x[i] );
   ls_phi[i] = -gamma;
   }
   */

#ifdef DEBUG
   char datei_steuer[200];
   FILE *stream;

   strcpy(datei_steuer,"test.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }

   else
   {
      for(i=0; i<2*anz_punkte_ls+2; i++)
      {
         fprintf(stream, "%d %lf %lf %lf %lf\n", i, ls_x[i], ls_y[i], ls_r[i], ls_phi[i]);
      }
      /*
      for(i=0; i<2*anz_punkte_ls; i++)
      {
         fprintf(stream, "%d %lf %lf\n", i, ls_xa[i], ls_ya[i]);
      }
      fprintf(stream, "%d %lf %lf\n", i+1, dreh_x, dreh_y);
      */

      fclose(stream);
   }
#endif

   i=2*anz_punkte_ls;

   double hyp = abstand(ls_x[i], ls_y[i], ls_x[i+1],ls_y[i+1]);
   double dy = ls_y[i] - ls_y[i+1];               // kann auch < 0 sein! Dann Drall falschrum.

   ls_beta2 = bog2grad * asin(dy / hyp);          //Austrittswinkel

   ls_hikara = ls_r[anz_punkte_ls-1];             //Hinterkantenradius

   // Vorzeichen des Versatzes ueberpruefen! Ist OK!
   delta = 2.* M_PI/anz_ls;                       //Versatz Leitschaufeln in Phi-Richtung

   // ==========================================================
   // Bogenlaengen der Punkte auf Nabe und Kranz berechnen
   // ==========================================================

   n_bgl = new double[anz_punkte_nabe];
   k_bgl = new double[anz_punkte_kranz];

   n_bgl[0]=0.;                                   //Nullpunkt der Bogenlaenge
   k_bgl[0]=0.;

   // R0_n = n_r[0];
   // R0_k = k_r[0];

   //printf("\nBogenlaengen Nabe: \n\n");

   for (i = 1; i < anz_punkte_nabe; i++)
   {
      //(old) n_bgl[i] =  (n_bgl[i-1]) + ( abstand( n_r[i], n_z[i], n_r[i-1], n_z[i-1]) * R0_n / k_r[i] );
      n_bgl[i] =  (n_bgl[i-1]) + abstand( n_r[i], n_z[i], n_r[i-1], n_z[i-1]) ;
   }

   //printf("\nBogenlaengen Kranz: \n\n");

   for (i=1; i<anz_punkte_kranz; i++)
   {
      //(old) k_bgl[i] =  (k_bgl[i-1]) + ( abstand( k_r[i], k_z[i], k_r[i-1], k_z[i-1]) * R0_k / k_r[i] );
      k_bgl[i] =  (k_bgl[i-1]) + abstand( k_r[i], k_z[i], k_r[i-1], k_z[i-1]);

   }

#ifdef DEBUG
   strcpy(datei_steuer,"leit_test.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }

   else
   {
      fprintf(stream,"Bogenlaenge Nabe, Radius Nabe, z Nabe:\n\n");

      for (i=0; i<anz_punkte_nabe; i++)
      {
         fprintf(stream,"%d %lf %lf %lf\n",i, n_bgl[i], n_r[i], n_z[i]);
      }

      fprintf(stream,"Bogenlaenge Kranz, Radius Kranz, z Kranz:\n\n");

      for (i=0; i<anz_punkte_kranz; i++)
      {
         fprintf(stream,"%d %lf %lf %lf\n",i, k_bgl[i], k_r[i], k_z[i]);
      }

      fprintf(stream,"Radius Leitschaufel:\n\n");

      for (i=0; i<2*anz_punkte_ls; i++)
      {
         fprintf(stream,"%d %lf\n",i, ls_r[i]);
      }

      fclose(stream);
   }

   printf("Bogenlaengen der Punkte auf Nabe und Kranz berechnet!\n");
#endif

   // ==========================================================
   // Berechnen der Bogenlaengen der Punkte der Leitschaufel
   // ==========================================================

   ls_bgl_n = new double[2*anz_punkte_ls];
   ls_bgl_k = new double[2*anz_punkte_ls];

   for (i=0; i<2*anz_punkte_ls; i++)              //Nabe: erst Druckseite
   {
      j=0;
      while (n_r[j] > ls_r[i])
      {
         j++;                                     // jetzt ist n_r[j] < ls_r[i]
      }

      ls_bgl_n[i] = n_bgl[j-1] + (n_bgl[j] - n_bgl[j-1]) * (n_r[j-1] - ls_r[i]) / (n_r[j-1] - n_r[j]);
      // Interpolation!
   }

   for (i=0; i<2*anz_punkte_ls; i++)              //Kranz
   {
      j=0;
      while (k_r[j] > ls_r[i])
      {
         j++;
      }
      ls_bgl_k[i] = k_bgl[j-1] + (k_bgl[j] - k_bgl[j-1]) * (k_r[j-1] - ls_r[i]) / (k_r[j-1] - k_r[j]);
   }

#ifdef DEBUG
   printf("Bogenlaengen der Leitschaufelpunkte berechnet!\n");
#endif

   // =====================================================
   // calculate bgl_15_n and bgl_15_k from %-parameters
   // 0 % = trailing edge, 100 % = outlet
   // =====================================================

   // calculate bgl_max_kranz
   if ( ga->radial == 0 )
   {
      i = 0;

      while (stf_para->bgl_max_nabe > n_bgl[i])
      {
         i++;
      }
      z = n_z[i-1]+(stf_para->bgl_max_nabe-n_bgl[i-1])/(n_bgl[i]-n_bgl[i-1])*(n_z[i]-n_z[i-1]);
      r_max_nabe = n_r[i-1]+(stf_para->bgl_max_nabe-n_bgl[i-1])/(n_bgl[i]-n_bgl[i-1])*(n_r[i]-n_r[i-1]);
      i=0;
      while (z < k_z[i])
      {
         i++;
      }
      bgl_max_kranz = k_bgl[i-1] + (z-k_z[i-1]) / (k_z[i]-k_z[i-1]) * (k_bgl[i]-k_bgl[i-1]);
      r_max_kranz=k_r[i-1]+(z-k_z[i-1])/(k_z[i]-k_z[i-1])*(k_r[i]-k_r[i-1]);
   }
   if ( ga->radial == 1 )
   {
      // so far only implemented if geofromfile!
      for ( i = 1; i < ga->pshroud->nump; i++)
      {
         bgl_max_kranz += pow ( pow(k_z[i]-k_z[i-1] , 2) + pow(k_r[i]-k_r[i-1] , 2) , 0.5 );
      }
   }

   stf_para->bgl_15_n*=0.01;                      // [%]!
   stf_para->bgl_15_k*=0.01;                      // [%]!

   stf_para->bgl_15_n = ls_bgl_n[anz_punkte_ls-1] * ( 1 - stf_para->bgl_15_n) + stf_para->bgl_max_nabe * stf_para->bgl_15_n;
   stf_para->bgl_15_k = ls_bgl_k[anz_punkte_ls-1] * ( 1 - stf_para->bgl_15_k) + bgl_max_kranz * stf_para->bgl_15_k;

   // printf("bgl_15_n	 =%6.2lf\n", stf_para->bgl_15_n);
   // printf("bgl_15_k	 =%6.2lf\n", stf_para->bgl_15_k);

   // =====================================================
   // Ausgabe der Bogenlaengen der Leitschaufelpunkte
   // (Bildschirm und Datei) leit_bgl.dat
   // =====================================================
#ifdef DEBUG
   strcpy(datei_steuer,"leit_bgl.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {

      fprintf(stream, "#%d\t", anz_punkte_ls);
      fprintf(stream, "#(Anzahl der Punkte der Leitschaufel)\n");
      fprintf(stream, "#%d\t", anz_ls);
      fprintf(stream, "#(Anzahl der Leitschaufeln)\n");

      //printf("Koordinaten der Leitschaufelpunkte transformiert\n");
      //printf("\nDruckseite + Saugseite (bgl, phi)\n");

      fprintf(stream, "#Punkte Nabe (Phi Bogenlaenge)\n\n");

      for (i=0; i<2*anz_punkte_ls; i++)           //Druckseite
      {
         //printf("%2d %8.4lf %8.4lf\n", i, ls_bgl[i], ls_phi[i]);
         fprintf(stream, "%2d %8.4lf %8.4lf \n", i, (ls_phi[i]), ls_bgl_n[i]);
      }

      fprintf(stream, "\n\n#Punkte Kranz (Phi Bogenlaenge)\n\n");

      for (i=0; i<2*anz_punkte_ls; i++)           //Druckseite
      {
         fprintf(stream, "%2d %8.4lf %8.4lf\n", i, (ls_phi[i]), ls_bgl_k[i]);
      }

      fclose (stream);
   }

   printf("leit_bgl erstellt!\n");
#endif

   //================================================================
   // fuer MESHSEED ls_bgl und ls_phi umsortieren
   //================================================================

   ls_bgl_sort_n   = new double[2*anz_punkte_ls-1];
   ls_bgl_sort_k   = new double[2*anz_punkte_ls-1];
   ls_phi_sort   = new double[2*anz_punkte_ls-1];

   j=0;

   for (i=anz_punkte_ls; i<2*anz_punkte_ls; i++)
   {
      ls_bgl_sort_n[j]=ls_bgl_n[i];
      ls_bgl_sort_k[j]=ls_bgl_k[i];
      ls_phi_sort[j]=ls_phi[i];
      j++;
   }

   for (i=1; i<anz_punkte_ls; i++)
   {
      ls_bgl_sort_n[j]=ls_bgl_n[i];
      ls_bgl_sort_k[j]=ls_bgl_k[i];
      ls_phi_sort[j]=ls_phi[i];
      j++;
   }

#ifdef DEBUG
   printf("ls_bgl und ls_phi umsortiert!\n");
#endif

   //========================================================================
   // Auf Spline durch die Leitschaufelpunkte PG Zwischenpunkte generieren
   //========================================================================

   bgl_ls_n   = new double[stf_para->PG];
   bgl_ls_k   = new double[stf_para->PG];
   phi_ls   = new double[stf_para->PG];

   MESHSEED (  ls_phi_sort, ls_bgl_sort_n,        //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      0, (anz_punkte_ls-1),                       //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., 0.,                                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1./0.2,                                     //Verh. L1/L2 bei M=0 ignoriert
      (2 * anz_punkte_ls-1),                      //=anz_naca (Anzahl der Stuetzpunkte)
      stf_para->PG3+1,                            //Anzahl der gewuenschten Kurvenpunkte
      phi_ls, bgl_ls_n);                          //Ausgabe

   MESHSEED (  ls_phi_sort, ls_bgl_sort_n,        //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      anz_punkte_ls-1, (2*anz_punkte_ls-2),       //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., 0.,                                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      0.2,                                        //Verh. L1/L2 bei M=0 ignoriert
      (2*anz_punkte_ls-1),                        //=anz_naca (Anzahl der Stuetzpunkte)
      stf_para->PG - stf_para->PG3,               //Anzahl der gewuenschten Kurvenpunkte
                                                  //Ausgabe
      &phi_ls[stf_para->PG3], &bgl_ls_n[stf_para->PG3]);

   MESHSEED (  ls_phi_sort, ls_bgl_sort_k,        //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      0, (anz_punkte_ls-1),                       //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., 0.,                                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1./0.2,                                     //Verh. L1/L2 bei M=0 ignoriert
      (2*anz_punkte_ls-1),                        //=anz_naca (Anzahl der Stuetzpunkte)
      stf_para->PG3+1,                            //Anzahl der gewuenschten Kurvenpunkte
      phi_ls, bgl_ls_k);                          //Ausgabe

   MESHSEED (  ls_phi_sort, ls_bgl_sort_k,        //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      anz_punkte_ls-1, (2*anz_punkte_ls-2),       //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., 0.,                                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      0.2,                                        //Verh. L1/L2 bei M=0 ignoriert
      (2*anz_punkte_ls-1),                        //=anz_naca (Anzahl der Stuetzpunkte)
      stf_para->PG - stf_para->PG3,               //Anzahl der gewuenschten Kurvenpunkte
                                                  //Ausgabe
      &phi_ls[stf_para->PG3], &bgl_ls_k[stf_para->PG3]);

#ifdef DEBUG
   strcpy(datei_steuer,"leit_spline.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {

      fprintf(stream, "#(Anzahl der Splinepunkte der Leitschaufel): ");
      fprintf(stream, "%d\n", stf_para->PG);

      fprintf(stream, "#Nabe\n");
      for (i = 0; i < stf_para->PG; i++)
      {
         fprintf(stream, "%d %lf \t %lf \n", i, phi_ls[i], bgl_ls_n[i]);
      }

      fprintf(stream, "#Kranz\n");
      for (i=0; i<stf_para->PG; i++)
      {
         fprintf(stream, "%d %lf \t %lf \n", i, phi_ls[i], bgl_ls_k[i]);
      }

      fclose(stream);
   }
#endif

   delete[] ls_bgl_n;
   delete[] ls_bgl_k;
   delete[] ls_phi;

#ifdef DEBUG
   printf("Zwischenpunkte auf Leitschaufel erstellt!\n");
#endif

   //===============================================================
   // vordersten Punkt bestimmen, PG-Parameter setzen
   //===============================================================

   int PGvorne = 0;

   for (i = 1; i < stf_para->PG; i++)
   {
      if ( bgl_ls_n[i] < bgl_ls_n[PGvorne] ) { PGvorne = i; }
   }

   stf_para->PG3 = PGvorne;
   stf_para->PG2 = PGvorne - (int) ( 0.01 * stf_para->PG2 * ( PGvorne - 0 ) );
   stf_para->PG4 = PGvorne + (int) ( 0.01 * stf_para->PG4 * ( stf_para->PG5 - PGvorne ) );

   //================================================================
   // Gebiet 1
   //================================================================

   MESHSEED    (phi_ls,bgl_ls_n,                  //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG1, stf_para->PG2,               //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., 0.,                                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      stf_para->lvs1,                             //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[1][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb11, fiyb11 );                           //Ausgabe

   MESHSEED    (phi_ls,bgl_ls_n,                  //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG1, stf_para->PG2,               //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., stf_para->grenz,                    //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      stf_para->lvs1,                             //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[1][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb12, fiyb12 );                           //Ausgabe

   for (i=0; i<ase[1][0]; i++)
   {
      GERADE (&fixb11[i],                         //Startpunkte
         &fiyb11[i],
         &fixb12[i],                              //Endpunkte
         &fiyb12[i],
         ase[1][1], 1, 1./stf_para->bvs1,         //Anzahl Punkte quer, Verdichtung
         &fixr1[i*ase[1][1]],                     //Hierhin speichern
         &fiyr1[i*ase[1][1]]);
   }
   //Gebiet 1 fertig!

   //================================================================
   // Gebiet 2
   //================================================================

   MESHSEED    (phi_ls, bgl_ls_n,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG2, stf_para->PG3,               //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., 0.,                                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      stf_para->lvs2,                             //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[2][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb21, fiyb21 );                           //Ausgabe

   MESHSEED    (phi_ls, bgl_ls_n,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG2, stf_para->PG3,               //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., stf_para->grenz,                    //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      stf_para->lvs2,                             //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[2][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb22, fiyb22 );                           //Ausgabe

   for (i=0; i<ase[2][0]; i++)
   {
      GERADE (&fixb21[i],                         //Startpunkte
         &fiyb21[i],
         &fixb22[i],                              //Endpunkte
         &fiyb22[i],
         ase[2][1], 1, 1/stf_para->bvs2,          //Anzahl Punkte quer, Verdichtung
         &fixr2[i*ase[2][1]],                     //Hierhin speichern
         &fiyr2[i*ase[2][1]]);
   }
   //Gebiet 2 fertig!

   //================================================================
   // Gebiet 3 (Anzahl ist anzahl_knot3)
   //================================================================

   MESHSEED (  phi_ls,  bgl_ls_n,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG3, stf_para->PG4,               //(PA, PE) Start- und Endpunkt (Nr)
      delta, 0., 0.,                              //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1/stf_para->lvs3,                           //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[3][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb31, fiyb31 );                           //Ausgabe

   MESHSEED (  phi_ls,  bgl_ls_n,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG3, stf_para->PG4,               //(PA, PE) Start- und Endpunkt (Nr)
      delta, 0., stf_para->grenz,                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1/stf_para->lvs3,                           //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[3][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb32, fiyb32 );                           //Ausgabe

   for (i=0; i<ase[3][0]; i++)
   {
      GERADE (&fixb31[i],                         //Startpunkte
         &fiyb31[i],
         &fixb32[i],                              //Endpunkte
         &fiyb32[i],
         ase[3][1], 1, 1./stf_para->bvs3,         //Anzahl Punkte quer, Verdichtung
         &fixr3[i*ase[3][1]],                     //Hierhin speichern
         &fiyr3[i*ase[3][1]]);
   }
   //Gebiet 3 fertig!

   //================================================================
   // Gebiet 4 (Anzahl ist anzahl_knot4)
   //================================================================

   MESHSEED (  phi_ls, bgl_ls_n,                  //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG4, stf_para->PG5,               //(PA, PE) Start- und Endpunkt (Nr)
      delta, 0., 0.,                              //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1./stf_para->lvs4,                          //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[4][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb41, fiyb41 );                           //Ausgabe

   MESHSEED (  phi_ls,  bgl_ls_n,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG4, stf_para->PG5,               //(PA, PE) Start- und Endpunkt (Nr)
      delta, 0., stf_para->grenz,                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1./stf_para->lvs4,                          //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[4][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb42, fiyb42 );                           //Ausgabe

   for (i=0; i<ase[4][0]; i++)
   {
      GERADE (&fixb41[i],                         //Startpunkte
         &fiyb41[i],
         &fixb42[i],                              //Endpunkte
         &fiyb42[i],
         ase[4][1], 1, 1./stf_para->bvs4,         //Anzahl Punkte quer, Verdichtung
         &fixr4[i*ase[4][1]],                     //Hierhin speichern
         &fiyr4[i*ase[4][1]]);
   }
   //Gebiet 4 fertig!

   //================================================================
   //Gebiet 8 (Anzahl in Leitschaufelrichtung ist anz_knot1
   //================================================================

   for (i=0; i<ase[8][0]; i++)
   {
      GERADE (&fixb12[i],                         //Startpunkte
         &fiyb12[i],
         &fixb42[ase[8][0]-1-i],                  //Endpunkte
         &fiyb42[ase[8][0]-1-i],
         ase[8][1], 2, 1./stf_para->bvs8,         //Anzahl Punkte quer, Verdichtung
         &fixr8[i*ase[8][1]],                     //Hierhin speichern
         &fiyr8[i*ase[8][1]]);
   }

   //Gebiet 8 fertig

   //================================================================
   //Gebiet 9
   //================================================================

   //Berechnung Schnittpunkt mit bgl_15_n (Gerade)
   //hier Eintrittsverschiebung!

   // change from (1 + versch_austr)
                                                  //Phi des Pkt Gebiet 15 ganz aussen an Grenze zu 10
   x10ru = fixb11[0] + (delta / 2) * (stf_para->versch_austr);
                                                  //zweiter Eckpunkt 15 - 10
   x10lu = x10ru + delta / (2*ase[10][1] + ase[9][1] - 3)*(ase[10][1]-1);

   x11lu = x10ru + delta;
   x11ru = x11lu - delta / (2*ase[11][1] + ase[9][1] - 3)*(ase[11][1]-1);

   // rechte Seite Gebiet 9 als Gerade von 10ru bis fixb12[0]

   GERADE(fixb12,                                 // Startpunkt
      fiyb12,
      &x10lu,                                     // Endpunkt
      &stf_para->bgl_15_n,
      ase[9][0], 1, stf_para->lvs5,               // Anz. Punkte quer
      fixb92,                                     // Hierhin speichern!
      fiyb92);

   for (i=0;i<ase[9][0];i++)
   {
      double bvs_temp = stf_para->bvs8 + (1.-stf_para->bvs8) * ((double)i) / ((double)ase[9][0]-1.);

      GERADE( fixb42 +i +(ase[1][0]-1),           // Startpunkte
         fiyb42 +i +(ase[1][0]-1),
         fixb92 +i,                               // Endpunkte
         fiyb92 +i,
         ase[9][1], 2, 1./bvs_temp,               // Anz. Punkte quer
         fixr9 +i*ase[9][1],                      // Hierhin speichern!
         fiyr9 +i*ase[9][1]);

   }
   // Gebiet 9 fertig!

   //================================================================
   //Gebiet10
   //================================================================

   GERADE(fixb11,                                 // Startpunkt
      fiyb11,
      &x10ru,                                     // Endpunkt
      &stf_para->bgl_15_n,
      ase[10][0], 1, 1./stf_para->lvs5,           // Anz. Punkte laengs
      fixb102,                                    // Hierhin speichern!
      fiyb102);

   for (i=0;i<ase[10][0];i++)
   {
      bvs_temp = stf_para->bvs1 + (1.-stf_para->bvs1) * ((double)i) / ((double)(ase[10][0]-1.));

      GERADE( fixb92 +i,                          // Startpunkte
         fiyb92 +i,
         fixb102 +i,                              // Endpunkte
         fiyb102 +i,
         ase[10][1], 1, bvs_temp,                 //bvs_temp,	Anz. Punkte quer
         fixr10 +i*ase[10][1],                    // Hierhin speichern!
         fiyr10 +i*ase[10][1]);

   }

   //Gebiet 10 fertig

   //================================================================
   //Gebiet 11
   //================================================================

   GERADE(&x11lu,                                 // Startpunkt linke Seite
      &stf_para->bgl_15_n,
      &fixb41[ase[4][0]-1],                       // Endpunkt
      &fiyb41[ase[4][0]-1],
      ase[11][0], 1, stf_para->lvs5,              // Anz. Punkte laengs
      fixb111,                                    // Hierhin speichern!
      fiyb111);

   GERADE(&x11ru,                                 // Startpunkt rechte Seite
      &stf_para->bgl_15_n,
      &fixb42[ase[4][0]-1],                       // Endpunkt
      &fiyb42[ase[4][0]-1],
      ase[11][0], 0, 1,                           // Anz. Punkte laengs rechts
      fixb112,                                    // Hierhin speichern!
      fiyb112);

   for (i=0; i<ase[11][0]; i++)
   {
      bvs_temp = stf_para->bvs4 + (1.-stf_para->bvs4) * ((double)(ase[11][0]-1-i)) / ((double)(ase[11][0]-1.));

      GERADE( fixb111 +i,                         // Startpunkte
         fiyb111 +i,
         fixb112 +i,                              // Endpunkte
         fiyb112 +i,
         ase[11][1], 1, 1./bvs_temp,              // Anz. Punkte quer
         fixr11 +i*ase[11][1],                    // Hierhin speichern!
         fiyr11 +i*ase[11][1]);
   }

   //============================================================
   //Dreieck-Gebiet 12-14
   //============================================================

   GERADE((fixb42 +ase[4][0]-1),                  // Startpunkte
      (fiyb42 +ase[4][0]-1),
      &x10lu,                                     // Endpunkte
      &stf_para->bgl_15_n,
      ase[9][1], 0, 1.,                           // Anz. Punkte quer
      fixb12_2,                                   // Hierhin speichern!
      fiyb12_2);

   GERADE(&x10lu,                                 // Startpunkte
      &stf_para->bgl_15_n,
      &x11ru,                                     // Endpunkte
      &stf_para->bgl_15_n,
      ase[9][1], 0, 1.,                           // Anz. Punkte quer
      fixb12_3,                                   // Hierhin speichern!
      fiyb12_3);

   DREIECK(fixb112,
      fiyb112,
      fixb12_2,
      fiyb12_2,
      fixb12_3,
      fiyb12_3,
      ase[9][1],
      fixr12,
      fiyr12);

   //Dreieck Austritt fertig!

   //================================================================
   //Gebiet 15 (Austritt)
   //================================================================

   // add outlet expansion to bgl_max_nabe
   // stf_para->bgl_max_nabe += stf_para->bgl_aus;

   GERADE(&x11lu,                                 // Startpunkte
      &stf_para->bgl_15_n,
      &x10ru,                                     // Endpunkte
      &stf_para->bgl_15_n,
      ase[15][0], 0, 1.,                          // Anz. Punkte quer
      fixb15_1,                                   // Hierhin speichern!
      fiyb15_1);

   GERADE(&x11lu,                                 // Startpunkte
      &stf_para->bgl_max_nabe,
      &x10ru,                                     // Endpunkte
      &stf_para->bgl_max_nabe,
      ase[15][0], 0, 1.,                          // Anz. Punkte quer
      fixb15_2,                                   // Hierhin speichern!
      fiyb15_2);

   for (i=0; i<ase[15][0]; i++)
   {
      GERADE( fixb15_1 +i,                        // Startpunkte
         fiyb15_1 +i,
         fixb15_2 +i,                             // Endpunkte
         fiyb15_2 +i,
         ase[15][1], 1, 1./stf_para->lvs6,        // Anz. Punkte quer
         fixr15 +i*ase[15][1],                    // Hierhin speichern!
         fiyr15 +i*ase[15][1]);
   }

   //Gebiet 15 fertig!

   //================================================================
   // Eintritt
   //================================================================

   schnitt=0;

   RECHNE_EINLAUF(
      schnitt,
      fixb32,
      &stf_para->bgl_start, &delta,
      &fixplo, &fiyplo,
      &fixpro, &fiypro);

   RECHNE_MITTELPUNKTE(&fixplo,
      &fixpro,
      &stf_para->bgl_start,
      &ase[3][0],
      &ase[7][1],
      &ase[2][0],
      &fixpmlo, &fiypmlo,
      &fixpmro, &fiypmro);

   //================================================================
   //Gebiet 5
   //================================================================

   GERADE(&fixplo,                                // Startpunkt
      &fiyplo,
      &fixpmlo,                                   // Endpunkt
      &fiypmlo,
      ase[3][0], 0, 1,                            // Anz. Punkte quer
      fixb52,                                     // Hierhin speichern!
      fiyb52);

   delta_x = *(fixb32) - *(fixb31);
   delta_y = *(fiyb32) - *(fiyb31);

   //phi_start = bog2grad * atan2(delta_y, delta_x);
   phi_start = -90;

   delta_x = *(fixb32 +ase[3][1]-1) - *(fixb31 +ase[3][1]-1);
   delta_y = *(fiyb32 +ase[3][1]-1) - *(fiyb31 +ase[3][1]-1);
   phi_ende = bog2grad * atan2(delta_y, delta_x);

   for (i=0;i<ase[3][0];i++)
   {

      phi = phi_start + 0.3 * (phi_ende-phi_start) * (double)i / (double)(ase[3][0]-1);

      // printf("phi= %5.2lf\n", phi);

      BEZIER( fixb32 +i,                          // Startpunkte
         fiyb32 +i,
         phi,
         fixb52 +i,                               // Endpunkte
         fiyb52 +i,
         90.,
         2,&ase[5][1], 1, &stf_para->lvs7,        // Anz. Punkte quer
         fixr5 +i*ase[5][1],                      // Hierhin speichern!
         fiyr5 +i*ase[5][1]);
   }

   //================================================================
   //Gebiet 6
   //================================================================

   GERADE(&fixpmro,                               // Startpunkt
      &fiypmro,
      &fixpro,                                    // Endpunkt
      &fiypro,
      ase[6][0], 0, 1,                            // Anz. Punkte quer
      fixb62,                                     // Hierhin speichern!
      fiyb62);

   delta_x = *(fixb22) - *(fixb21);
   delta_y = *(fiyb22) - *(fiyb21);

   phi_ende = bog2grad * atan2(delta_y, delta_x);

   phi_start = -90;

   for (i=0;i<ase[2][0];i++)
   {
      // 70% of the average of both angles
      phi = i * ( phi_start - (phi_ende + phi_start) * 0.7 ) / ( ase[2][0] - 1 ) + ( phi_ende + phi_start ) * 0.7;

      BEZIER( fixb22 +i,                          // Startpunkte
         fiyb22 +i,
         phi,
         fixb62 +i,                               // Endpunkte
         fiyb62 +i,
         90.,
         2,&ase[6][1], 1, &stf_para->lvs7,        // Anz. Punkte quer
         fixr6 +i*ase[6][1],                      // Hierhin speichern!
         fiyr6 +i*ase[6][1]);
   }

   //================================================================
   //Gebiet 7
   //================================================================

   for (i=0; i<stf_para->anz_einlauf; i++)
   {
      bvs_temp = stf_para->bvs8 + (1.-stf_para->bvs8) * ((double)i) / ((double)(ase[7][0]-1.));

      GERADE(fixr5 +i +(ase[3][0]-1)*ase[7][0],   // Startpunkt
         fiyr5 +i +(ase[3][0]-1)*ase[7][0],
         fixr6 +i,                                // Endpunkt
         fiyr6 +i,
         ase[7][1], 2, 1./bvs_temp,               // Anz. Punkte quer
         fixr7 +i*ase[7][1],                      // Hierhin speichern!
         fiyr7 +i*ase[7][1]);
   }

   //================================================================
   //Ausgabe Gebiete
   //================================================================

#ifdef DEBUG
   strcpy(datei_steuer,"leit_block_nabe.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
      for (i=0; i<ase[1][0]*ase[1][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 1 Nabe)\n",i, fixr1[i], fiyr1[i]);
      }

      for (i=0; i<ase[2][0]*ase[2][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 2 Nabe)\n",i, fixr2[i], fiyr2[i]);
      }

      for (i=0; i<ase[3][0]*ase[3][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 3 Nabe)\n",i, fixr3[i], fiyr3[i]);
      }

      for (i=0; i<ase[4][0]*ase[4][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 4 Nabe)\n",i, fixr4[i], fiyr4[i]);
      }

      for (i=0; i<ase[8][0]*ase[8][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 8 Nabe)\n",i, fixr8[i], fiyr8[i]);
      }

      for (i=0; i<ase[9][0]*ase[9][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 9 Nabe)\n",i, fixr9[i], fiyr9[i]);
      }

      for (i=0; i<ase[10][0]*ase[10][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 10 Nabe)\n",i, fixr10[i], fiyr10[i]);
      }

      for (i=0; i<ase[11][0]*ase[11][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 11 Nabe)\n",i, fixr11[i], fiyr11[i]);
      }

      for (i=0; i<anz_dreieck; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 12-14 Nabe)\n",i, fixr12[i], fiyr12[i]);
      }

      for (i=0; i<ase[15][0]*ase[15][1] ; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 15 Nabe)\n",i, fixr15[i], fiyr15[i]);
      }

      for (i=0; i<ase[5][0]*ase[5][1] ; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 5 Nabe)\n",i, fixr5[i], fiyr5[i]);
      }

      for (i=0; i<ase[6][0]*ase[6][1] ; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 6 Nabe)\n",i, fixr6[i], fiyr6[i]);
      }

      for (i=0; i<ase[7][0]*ase[7][1] ; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 7 Nabe)\n",i, fixr7[i], fiyr7[i]);
      }

      fclose(stream);

   }
#endif

#ifdef DEBUG
   printf("Nabe-Netz fertig!\n");
#endif

   //=================================================================
   // Randpunkte markieren
   //=================================================================

   //Alle Gebietsrandpunkte werden in der Randpunktliste mit -1 versehen

   randpunkt = new int [seed];                    //Randpunkt ja oder nein?

   knot_nr = new int [seed];

   for (i=0; i<seed; i++)
   {
      randpunkt[i] = 0;
      knot_nr[i] = i;
   }

   pos=0;

   for (i=1; i<ANZ_GEBIETE+1; i++)
   {
      for (j=0; j < ase[i][1]-1; j++)
      {
         randpunkt[pos+j]=1;
      }

      while (j < (ase[i][0]*ase[i][1]-ase[i][1]))
      {
         *(randpunkt+pos+j)=1;
         j++;
         *(randpunkt+pos+j)=1;
         j=j+ase[i][1]-1;
      }

      for (j=ase[i][0]*ase[i][1]-1; j>ase[i][0]*ase[i][1]-ase[i][1]; j--)
      {
         *(randpunkt+pos+j)=1;
      }
      pos = pos + ase[i][0] * ase[i][1];
   }

   knotx = new double [seed];                     //eigentlich zu lang, aber anz_doppelt
   knoty = new double [seed];                     //ist leider noch unbekannt

   ersatz_nr = new int[seed];                     //gibt an, zu welcher Knoten_Nr. die
   //doppelten werden!

   deltax = fixb31[1]-fixb31[0];                  // Staupunkt: hier gewoehnlich dichtestes Netz
   deltay = fiyb31[1]-fiyb31[0];
   TOL = 0.1 *pow ( (pow(deltax,2) + pow(deltay,2)), 0.5 );

#ifdef DEBUG
   printf("Randpunkte markiert!\n");
#endif

   //===========================================================
   // Doppelte Punkte entfernen
   //===========================================================

   DOPPELTE_WEG ( fixr, fiyr,                     //Koordinaten Gesamtfeld
      seed,                                       //Anzahl Knoten Gesamt
      TOL,                                        //kleinster Punktabstand
      knot_nr,                                    //Fortlaufende Nummer, -1 wenn doppelt
      randpunkt,                                  //=1, wenn Randpunkt, sonst =0
      knotx, knoty,                               //Koordinaten bereinigtes Feld
      &anz_doppelt,                               //Anzahl der doppelten Knoten
      ersatz_nr);                                 //Knoten ersetzt durch (neue Nummer)
   //diese Knotennummer ist bereits
   //die Nummer im neuen System!

#ifdef DEBUG
   printf("Doppelte Punkte entfernt!\n");
#endif

   // =================================
   // Ausgabe Feld ohne doppelte Knoten
   // =================================

#ifdef DEBUG
   strcpy(datei_steuer,"leit_block_nabe2.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
      for (i=0; i<seed-anz_doppelt; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf\n",i, knotx[i], knoty[i]);
      }
   }
#endif

   //=============================================================
   //2D-Elemente generieren
   //=============================================================
   //Bildet aus den Knoten kleine viereckige Elemente

   //Zuerst Anzahl der Elemente bestimmen

   anz_elemente=seed;

   for (i=1; i<ANZ_GEBIETE+1; i++)
   {
      anz_elemente=anz_elemente-(ase[i][0])-(ase[i][1])+1;
   }

   el_liste = new int [4*anz_elemente];

   pos=0;
   nr=0;

   for (k=1; k<ANZ_GEBIETE+1; k++)
   {
      for (j=0; j<(ase[k][0]-1); j++)
      {
         for(i=0; i<(ase[k][1]-1); i++)
         {
            el_liste[4*nr]   = ersatz_nr[pos];
            el_liste[4*nr+1] = ersatz_nr[pos+1];
            el_liste[4*nr+2] = ersatz_nr[pos+1 + ase[k][1]];
            el_liste[4*nr+3] = ersatz_nr[pos+ ase[k][1]];

            nr++;                                 //1 Element erzeugt (Uhrzeigersinn)
            pos++;                                //1 Punkte erschlagen
         }
         pos++;                                   //ein Punkt ausgelassen
      }
      pos += ase[k][1];                           //1 Reihe ausgelassen
   }

   //Ausgabe der 2D-Elemente

#ifdef DEBUG
   strcpy(datei_steuer,"leit_2D_elemente.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }

   fprintf(stream,"Seedalt =%5d\n",seed);
   fprintf(stream,"pos =%5d\n",pos);

   fprintf(stream,"Anzahl Elemente =%5d\n",anz_elemente);

   for (i=0; i<anz_elemente; i++)
   {
      fprintf(stream,"%5d: %5d%5d%5d%5d", i, el_liste[4*i], el_liste[4*i+1],
         el_liste[4*i+2], el_liste[4*i+3]);
      fprintf(stream,"\n");

   }

   fclose(stream);
#endif

   //===========================================================
   // Rcktrafo
   //===========================================================

   nabex = new double[seed-anz_doppelt];
   nabey = new double[seed-anz_doppelt];
   nabez = new double[seed-anz_doppelt];
   naber = new double[seed-anz_doppelt];

   //Uebergang von bgl / phi - System ins x,y,z - System

   j=0;

   for (i=0; i<seed-anz_doppelt; i++)
   {
      if ( knoty[i] <= 0.0 )                      // es kann auch negative Bogenlaengen geben!
      {                                           // hier extrapolieren
         naber[i] = n_r[0] + ( (n_bgl[0]-knoty[i]) / (n_bgl[1]-n_bgl[0]) ) * (n_r[0]-n_r[1]);
         nabez[i] = n_z[0] + ( (n_bgl[0]-knoty[i]) / (n_bgl[1]-n_bgl[0]) ) * (n_z[0]-n_z[1]);
         nabex[i] = naber[i] * cos( knotx[i] );
         nabey[i] = naber[i] * sin( knotx[i] );
      }
      else
      {
         j = 0;
         while ( (n_bgl[j]) < (knoty[i]) )        //hier interpolieren
         {
            j++;
         }

         naber[i] = n_r[j] + ( (knoty[i]-n_bgl[j]) / (n_bgl[j-1]-n_bgl[j]) ) * (n_r[j-1]-n_r[j]);
         nabez[i] = n_z[j] + ( (knoty[i]-n_bgl[j]) / (n_bgl[j-1]-n_bgl[j]) ) * (n_z[j-1]-n_z[j]);
         nabex[i] = naber[i] * cos(knotx[i]);
         nabey[i] = naber[i] * sin(knotx[i]);
      }
   }

#ifdef DEBUG
   strcpy(datei_steuer,"netz_nabe.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }

   fprintf(stream,"#Nr., X Y Z Koordinaten\n");

   for (i=0; i<seed-anz_doppelt; i++)
   {
      fprintf(stream, "%4d %lf %lf %lf\n", i, nabex[i], nabey[i], nabez[i]);
   }

   fclose(stream);
#endif

#ifdef DEBUG
   printf("Nabe-Punkte ruecktransformiert!\n");
#endif

   //=================================================================
   //
   //  UND JETZT DIE GANZE GESCHICHTE NOCHMAL FUER DEN KRANZ!
   //
   //=================================================================
   /*
      i=0;

      while (stf_para->bgl_max_nabe > n_bgl[i])
      {
         i++;
      }

      z = n_z[i-1]+(stf_para->bgl_max_nabe-n_bgl[i-1])/(n_bgl[i]-n_bgl[i-1])*(n_z[i]-n_z[i-1]);
      r_max_nabe = n_r[i-1]+(stf_para->bgl_max_nabe-n_bgl[i-1])/(n_bgl[i]-n_bgl[i-1])*(n_r[i]-n_r[i-1]);
      //printf("r_max_nabe=%lf\n",r_max_nabe);

   i=0;

   while (z < k_z[i])
   {
   i++;
   }

   bgl_max_kranz = k_bgl[i-1] + (z-k_z[i-1]) / (k_z[i]-k_z[i-1]) * (k_bgl[i]-k_bgl[i-1]);
   printf("bgl_max_kranz=%6.2lf\n", bgl_max_kranz);
   printf("z_out        =%6.2lf\n", z);
   r_max_kranz=k_r[i-1]+(z-k_z[i-1])/(k_z[i]-k_z[i-1])*(k_r[i]-k_r[i-1]);
   //printf("r_max_kranz=%lf\n",r_max_kranz);
   */
   //================================================================
   // Gebiet 1
   //================================================================

   MESHSEED    (phi_ls, bgl_ls_k,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG1, stf_para->PG2,               //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., 0.,                                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      stf_para->lvs1,                             //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[1][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb11, fiyb11 );                           //Ausgabe

   MESHSEED    (phi_ls,bgl_ls_k,                  //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG1, stf_para->PG2,               //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., stf_para->grenz,                    //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      stf_para->lvs1,                             //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[1][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb12, fiyb12 );                           //Ausgabe

   for (i=0; i<ase[1][0]; i++)
   {
      GERADE (&fixb11[i],                         //Startpunkte
         &fiyb11[i],
         &fixb12[i],                              //Endpunkte
         &fiyb12[i],
         ase[1][1], 1, 1./stf_para->bvs1,         //Anzahl Punkte quer, Verdichtung
         &fixr1[i*ase[1][1]],                     //Hierhin speichern
         &fiyr1[i*ase[1][1]]);
   }
   //Gebiet 1 fertig!

   //================================================================
   // Gebiet 2
   //================================================================

   MESHSEED    (phi_ls, bgl_ls_k,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG2, stf_para->PG3,               //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., 0.,                                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      stf_para->lvs2,                             //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[2][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb21, fiyb21 );                           //Ausgabe

   MESHSEED    (phi_ls, bgl_ls_k,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG2, stf_para->PG3,               //(PA, PE) Start- und Endpunkt (Nr)
      0., 0., stf_para->grenz,                    //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      stf_para->lvs2,                             //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[2][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb22, fiyb22 );                           //Ausgabe

   for (i=0; i<ase[2][0]; i++)
   {
      GERADE (&fixb21[i],                         //Startpunkte
         &fiyb21[i],
         &fixb22[i],                              //Endpunkte
         &fiyb22[i],
         ase[2][1], 1, 1/stf_para->bvs2,          //Anzahl Punkte quer, Verdichtung
         &fixr2[i*ase[2][1]],                     //Hierhin speichern
         &fiyr2[i*ase[2][1]]);
   }
   //Gebiet 2 fertig!

   //================================================================
   // Gebiet 3 (Anzahl ist anzahl_knot3)
   //================================================================

   MESHSEED (  phi_ls,  bgl_ls_k,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG3, stf_para->PG4,               //(PA, PE) Start- und Endpunkt (Nr)
      delta, 0., 0.,                              //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1./stf_para->lvs3,                          //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[3][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb31, fiyb31 );                           //Ausgabe

   MESHSEED (  phi_ls,  bgl_ls_k,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG3, stf_para->PG4,               //(PA, PE) Start- und Endpunkt (Nr)
      delta, 0., stf_para->grenz,                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1./stf_para->lvs3,                          //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[3][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb32, fiyb32 );                           //Ausgabe

   for (i=0; i<ase[3][0]; i++)
   {
      GERADE (&fixb31[i],                         //Startpunkte
         &fiyb31[i],
         &fixb32[i],                              //Endpunkte
         &fiyb32[i],
         ase[3][1], 1, 1./stf_para->bvs3,         //Anzahl Punkte quer, Verdichtung
         &fixr3[i*ase[3][1]],                     //Hierhin speichern
         &fiyr3[i*ase[3][1]]);
   }
   //Gebiet 3 fertig!

   //================================================================
   // Gebiet 4 (Anzahl ist anzahl_knot4)
   //================================================================

   MESHSEED (  phi_ls, bgl_ls_k,                  //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG4, stf_para->PG5,               //(PA, PE) Start- und Endpunkt (Nr)
      delta, 0., 0.,                              //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1./stf_para->lvs4,                          //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[4][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb41, fiyb41 );                           //Ausgabe

   MESHSEED (  phi_ls,  bgl_ls_k,                 //Stuetzpunkt x, y
      0, 0., 0.,                                  //M0, Ableitungen (bei M0=0 ignoriert)
      stf_para->PG4, stf_para->PG5,               //(PA, PE) Start- und Endpunkt (Nr)
      delta, 0., stf_para->grenz,                 //(Verschiebungen)
      1,                                          //(m=0 aequidistant)
      1./stf_para->lvs4,                          //Verh. L1/L2 bei M=0 ignoriert
      stf_para->PG,                               //(Anzahl der Stuetzpunkte)
      ase[4][0],                                  //Anzahl der gewuenschten Kurvenpunkte
      fixb42, fiyb42 );                           //Ausgabe

   for (i=0; i<ase[4][0]; i++)
   {
      GERADE (&fixb41[i],                         //Startpunkte
         &fiyb41[i],
         &fixb42[i],                              //Endpunkte
         &fiyb42[i],
         ase[4][1], 1, 1./stf_para->bvs4,         //Anzahl Punkte quer, Verdichtung
         &fixr4[i*ase[4][1]],                     //Hierhin speichern
         &fiyr4[i*ase[4][1]]);
   }
   //Gebiet 4 fertig!

   //================================================================
   //Gebiet 8 (Anzahl in Leitschaufelrichtung ist anz_knot1
   //================================================================

   for (i=0; i<ase[8][0]; i++)
   {
      GERADE (&fixb12[i],                         //Startpunkte
         &fiyb12[i],
         &fixb42[ase[8][0]-1-i],                  //Endpunkte
         &fiyb42[ase[8][0]-1-i],
         ase[8][1], 2, 1./stf_para->bvs8,         //Anzahl Punkte quer, Verdichtung
         &fixr8[i*ase[8][1]],                     //Hierhin speichern
         &fiyr8[i*ase[8][1]]);
   }

   //Gebiet 8 fertig

   //================================================================
   //Gebiet 9
   //================================================================

   // rechte Seite Gebiet 9 als Gerade von 10ru bis fixb12[0]

   GERADE(fixb12,                                 // Startpunkt
      fiyb12,
      &x10lu,                                     // Endpunkt
      &stf_para->bgl_15_k,
      ase[9][0], 1, stf_para->lvs5,               // Anz. Punkte quer
      fixb92,                                     // Hierhin speichern!
      fiyb92);

   for (i=0;i<ase[9][0];i++)
   {
      bvs_temp = stf_para->bvs8 + (1.-stf_para->bvs8) * ((double)i) / ((double)ase[9][0]-1.);

      GERADE( fixb42 +i +(ase[1][0]-1),           // Startpunkte
         fiyb42 +i +(ase[1][0]-1),
         fixb92 +i,                               // Endpunkte
         fiyb92 +i,

         ase[9][1], 2, 1./bvs_temp,               // Anz. Punkte quer
         fixr9 +i*ase[9][1],                      // Hierhin speichern!
         fiyr9 +i*ase[9][1]);
   }

   // Gebiet 9 fertig!

   //================================================================
   //Gebiet10
   //================================================================

   GERADE(fixb11,                                 // Startpunkt
      fiyb11,
      &x10ru,                                     // Endpunkt
      &stf_para->bgl_15_k,
      ase[10][0], 1, 1./stf_para->lvs5,           // Anz. Punkte laengs
      fixb102,                                    // Hierhin speichern!
      fiyb102);

   for (i=0;i<ase[10][0];i++)
   {
      bvs_temp = stf_para->bvs1 + (1.-stf_para->bvs1) * ((double)i) / ((double)(ase[10][0]-1.));

      GERADE( fixb92 +i,                          // Startpunkte
         fiyb92 +i,
         fixb102 +i,                              // Endpunkte
         fiyb102 +i,
         ase[10][1], 1, bvs_temp,                 //bvs_temp,	Anz. Punkte quer
         fixr10 +i*ase[10][1],                    // Hierhin speichern!
         fiyr10 +i*ase[10][1]);
   }

   //Gebiet 10 fertig

   //================================================================
   //Gebiet 11
   //================================================================

   GERADE(&x11lu,                                 // Startpunkt linke Seite
      &stf_para->bgl_15_k,
      &fixb41[ase[4][0]-1],                       // Endpunkt
      &fiyb41[ase[4][0]-1],
      ase[11][0], 1, stf_para->lvs5,              // Anz. Punkte laengs
      fixb111,                                    // Hierhin speichern!
      fiyb111);

   GERADE(&x11ru,                                 // Startpunkt rechte Seite
      &stf_para->bgl_15_k,
      &fixb42[ase[4][0]-1],                       // Endpunkt
      &fiyb42[ase[4][0]-1],
      ase[11][0], 0, 1,                           // Anz. Punkte laengs rechts
      fixb112,                                    // Hierhin speichern!
      fiyb112);

   for (i=0; i<ase[11][0]; i++)
   {
      bvs_temp = stf_para->bvs4 + (1.-stf_para->bvs4) * ((double)(ase[11][0]-1-i)) / ((double)(ase[11][0]-1.));

      GERADE( fixb111 +i,                         // Startpunkte
         fiyb111 +i,
         fixb112 +i,                              // Endpunkte
         fiyb112 +i,
         ase[11][1], 1, 1./bvs_temp,              // Anz. Punkte quer
         fixr11 +i*ase[11][1],                    // Hierhin speichern!
         fiyr11 +i*ase[11][1]);
   }

   //Dreieck - Gebiet 12-14

   GERADE((fixb42 +ase[4][0]-1),                  // Startpunkte
      (fiyb42 +ase[4][0]-1),
      &x10lu,                                     // Endpunkte
      &stf_para->bgl_15_k,
      ase[9][1], 0, 1.,                           // Anz. Punkte quer
      fixb12_2,                                   // Hierhin speichern!
      fiyb12_2);

   GERADE(&x10lu,                                 // Startpunkte
      &stf_para->bgl_15_k,
      &x11ru,                                     // Endpunkte
      &stf_para->bgl_15_k,
      ase[9][1], 0, 1.,                           // Anz. Punkte quer
      fixb12_3,                                   // Hierhin speichern!
      fiyb12_3);

   DREIECK(fixb112,
      fiyb112,
      fixb12_2,
      fiyb12_2,
      fixb12_3,
      fiyb12_3,
      ase[9][1],
      fixr12,
      fiyr12);

   //Dreieck Austritt fertig!

   //================================================================
   //Gebiet 15 (Austritt)
   //================================================================

   // add outlet expansion to bgl_max_kranz
   // bgl_max_kranz += stf_para->bgl_aus;

   GERADE(&x11lu,                                 // Startpunkte
      &stf_para->bgl_15_k,
      &x10ru,                                     // Endpunkte
      &stf_para->bgl_15_k,
      ase[15][0], 0, 1.,                          // Anz. Punkte quer
      fixb15_1,                                   // Hierhin speichern!
      fiyb15_1);

   GERADE(&x11lu,                                 // Startpunkte
      &bgl_max_kranz,
      &x10ru,                                     // Endpunkte
      &bgl_max_kranz,
      ase[15][0], 0, 1.,                          // Anz. Punkte quer
      fixb15_2,                                   // Hierhin speichern!
      fiyb15_2);

   for (i=0; i<ase[15][0]; i++)
   {
      GERADE( fixb15_1 +i,                        // Startpunkte
         fiyb15_1 +i,
         fixb15_2 +i,                             // Endpunkte
         fiyb15_2 +i,
         ase[15][1], 1, 1./stf_para->lvs6,        // Anz. Punkte quer
         fixr15 +i*ase[15][1],                    // Hierhin speichern!
         fiyr15 +i*ase[15][1]);
   }

   //Gebiet 15 fertig!

   //================================================================
   // Eintritt
   //================================================================

   schnitt=0;

   RECHNE_EINLAUF(
      schnitt,
      fixb32,
      &stf_para->bgl_start, &delta,
      &fixplo, &fiyplo,
      &fixpro, &fiypro);

   RECHNE_MITTELPUNKTE(&fixplo,
      &fixpro,
      &stf_para->bgl_start,
      &ase[3][0],                                 //??? noch richtig nach ase
      &ase[7][1],
      &ase[2][0],
      &fixpmlo, &fiypmlo,
      &fixpmro, &fiypmro);

   //================================================================
   //Gebiet 5
   //================================================================

   GERADE(&fixplo,                                // Startpunkt
      &fiyplo,
      &fixpmlo,                                   // Endpunkt
      &fiypmlo,
      ase[3][0], 0, 1.,                           // Anz. Punkte quer
      fixb52,                                     // Hierhin speichern!
      fiyb52);

   delta_x = *(fixb32) - *(fixb31);
   delta_y = *(fiyb32) - *(fiyb31);

   //phi_start = bog2grad * atan2(delta_y, delta_x);
   phi_start = -90;

   delta_x = *(fixb32 +ase[3][1]-1) - *(fixb31 +ase[3][1]-1);
   delta_y = *(fiyb32 +ase[3][1]-1) - *(fiyb31 +ase[3][1]-1);
   phi_ende = bog2grad * atan2(delta_y, delta_x);

   for (i=0;i<ase[3][0];i++)
   {
      // 70% eg 30%
      phi = phi_start + 0.3 * (phi_ende - phi_start) * (double)i / (double)(ase[3][0]-1);

      BEZIER( fixb32 +i,                          // Startpunkte
         fiyb32 +i,
         phi,
         fixb52 +i,                               // Endpunkte
         fiyb52 +i,
         90.,
         2,&ase[5][1], 1, &stf_para->lvs7,        // Anz. Punkte quer
         fixr5 +i*ase[5][1],                      // Hierhin speichern!
         fiyr5 +i*ase[5][1]);
   }

   //================================================================
   //Gebiet 6
   //================================================================

   GERADE(&fixpmro,                               // Startpunkt
      &fiypmro,
      &fixpro,                                    // Endpunkt
      &fiypro,
      ase[6][0], 0, 1,                            // Anz. Punkte quer
      fixb62,                                     // Hierhin speichern!
      fiyb62);

   phi_ende = phi_start;

   delta_x = *(fixb22) - *(fixb21);
   delta_y = *(fiyb22) - *(fiyb21);

   phi_start = -90;
   phi_ende = bog2grad * atan2(delta_y, delta_x);

   for (i=0;i<ase[2][0];i++)
   {

      // 70% of the average of both angles
      phi = i * ( phi_start - (phi_ende + phi_start) * 0.7 ) / ( ase[2][0] - 1 ) + ( phi_ende + phi_start ) * 0.7;

      BEZIER( fixb22 +i,                          // Startpunkte
         fiyb22 +i,
         phi,
         fixb62 +i,                               // Endpunkte
         fiyb62 +i,
         90.,
         2,&ase[6][1], 1, &stf_para->lvs7,        // Anz. Punkte quer
         fixr6 +i*ase[6][1],                      // Hierhin speichern!
         fiyr6 +i*ase[6][1]);
   }

   //================================================================
   //Gebiet 7
   //================================================================

   for (i=0; i<stf_para->anz_einlauf; i++)
   {
      bvs_temp = stf_para->bvs8 + (1.-stf_para->bvs8) * ((double)i) / ((double)(ase[7][0]-1.));

      GERADE(fixr5 +i +(ase[3][0]-1)*ase[7][0],   // Startpunkt
         fiyr5 +i +(ase[3][0]-1)*ase[7][0],
         fixr6 +i,                                // Endpunkt
         fiyr6 +i,
         ase[7][1], 2, 1./bvs_temp,               // Anz. Punkte quer
         fixr7 +i*ase[7][1],                      // Hierhin speichern!
         fiyr7 +i*ase[7][1]);
   }

   //================================================================
   //Ausgabe Gebiete
   //================================================================

#ifdef DEBUG
   strcpy(datei_steuer,"leit_block_kranz.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
      for (i=0; i<ase[1][0]*ase[1][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 1 Kranz)\n",i, fixr1[i], fiyr1[i]);
      }

      for (i=0; i<ase[2][0]*ase[2][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 2 Kranz)\n",i, fixr2[i], fiyr2[i]);
      }

      for (i=0; i<ase[3][0]*ase[3][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 3 Kranz)\n",i, fixr3[i], fiyr3[i]);
      }

      for (i=0; i<ase[4][0]*ase[4][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 4 Kranz)\n",i, fixr4[i], fiyr4[i]);
      }

      for (i=0; i<ase[8][0]*ase[8][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 8 Kranz)\n",i, fixr8[i], fiyr8[i]);
      }

      for (i=0; i<ase[9][0]*ase[9][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 9 Kranz)\n",i, fixr9[i], fiyr9[i]);
      }

      for (i=0; i<ase[10][0]*ase[10][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 10 Kranz)\n",i, fixr10[i], fiyr10[i]);
      }

      for (i=0; i<ase[11][0]*ase[11][1]; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 11 Kranz)\n",i, fixr11[i], fiyr11[i]);
      }

      for (i=0; i<anz_dreieck; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 12-14 Kranz)\n",i, fixr12[i], fiyr12[i]);
      }

      for (i=0; i<ase[15][0]*ase[15][1] ; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 15 Kranz)\n",i, fixr15[i], fiyr15[i]);
      }

      for (i=0; i<ase[5][0]*ase[5][1] ; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 5 Kranz)\n",i, fixr5[i], fiyr5[i]);
      }

      for (i=0; i<ase[6][0]*ase[6][1] ; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 6 Kranz)\n",i, fixr6[i], fiyr6[i]);
      }

      for (i=0; i<ase[7][0]*ase[7][1] ; i++)
      {
         fprintf(stream, "%4d %8.4lf %8.4lf (Gebiet 7 Kranz)\n",i, fixr7[i], fiyr7[i]);
      }

      fclose(stream);

   }
#endif

   //=========================================================================
   // Kranzfeld bereinigen
   //=========================================================================

   j=0;

   for (i=0; i<seed; i++)
   {
      if (knot_nr[i]==-1) continue;
      *(knotx+j)=*(fixr+i);                       //nur nicht doppelte Punkte in neux
      *(knoty+j)=*(fiyr+i);                       //Koordinaten im neuen Feld
      j++;
   }

   //===========================================================
   // Rcktrafo
   //===========================================================

   kranzx = new double[seed-anz_doppelt];
   kranzy = new double[seed-anz_doppelt];
   kranzz = new double[seed-anz_doppelt];
   kranzr = new double[seed-anz_doppelt];

   //Uebergang von bgl / phi - System ins x,y,z - System

   for (i=0; i<seed-anz_doppelt; i++)
   {
      if (knoty[i] <= 0)                          //es gibt auch negative Bogenlaengen!
      {                                           //hier extrapolieren
         kranzr[i] = k_r[0]+( (k_bgl[0]-knoty[i]) / (k_bgl[1]-k_bgl[0]) ) * (k_r[0]-k_r[1]);
         kranzz[i] = k_z[0]+( (k_bgl[0]-knoty[i]) / (k_bgl[1]-k_bgl[0]) ) * (k_z[0]-k_z[1]);
         kranzx[i] = kranzr[i]*cos(knotx[i]);
         kranzy[i] = kranzr[i]*sin(knotx[i]);
      }
      else
      {
         j=0;
         while((j < anz_punkte_kranz) && ((k_bgl[j]) < (knoty[i])))          //hier interpolieren
         {
            j++;
         }
         if(j < anz_punkte_kranz)
         {
            kranzr[i] = k_r[j] + ( (knoty[i]-k_bgl[j]) / (k_bgl[j-1]-k_bgl[j]) ) * (k_r[j-1]-k_r[j]);
            kranzz[i] = k_z[j] + ( (knoty[i]-k_bgl[j]) / (k_bgl[j-1]-k_bgl[j]) ) * (k_z[j-1]-k_z[j]);
         }
         else
         {
            kranzr[i] = k_r[j];
            kranzz[i] = k_z[j];
         }
         kranzx[i] = kranzr[i]*cos(knotx[i]);
         kranzy[i] = kranzr[i]*sin(knotx[i]);
      }
   }

#ifdef DEBUG
   strcpy(datei_steuer,"leit_test.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
      fprintf(stream,"#i knotx knoty \n");

      for (i=0; i<seed-anz_doppelt; i++)
      {
         fprintf(stream,"%5d %8.4lf %8.4lf\n",i, knotx[i], knoty[i]);
      }

      fclose(stream);
   }
#endif

#ifdef DEBUG
   strcpy(datei_steuer,"netz_kranz.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
      fprintf(stream,"#Nr., X Y Z Koordinaten\n");

      for (i=0; i<seed-anz_doppelt; i++)
      {
         fprintf(stream, "%4d %lf %lf %lf\n", i, kranzx[i], kranzy[i], kranzz[i]);
      }

      fclose(stream);
   }
#endif

   //int seedalt = seed;				//alte Laenge mit doppelten
   seed = seed - anz_doppelt;                     //neue Laenge ohne doppelte

#ifdef DEBUG
   printf("Kranznetz erstellt!\n");
#endif

   //==================================================================
   // Erstellen der 3D-Knoten
   //==================================================================

   double *NETZX, *NETZY, *NETZZ;

   NETZX = new double[stf_para->anz_schnitte*seed];
   NETZY = new double[stf_para->anz_schnitte*seed];
   NETZZ = new double[stf_para->anz_schnitte*seed];

   for (i=0; i<seed; i++)
   {
      GERADE3D(nabex+i,                           //Startpunkt
         nabey+i,
         nabez+i,
         kranzx+i,                                //Endpunkt
         kranzy+i,
         kranzz+i,
                                                  // Anz. Punkte quer
         stf_para->anz_schnitte, seed, 2, 1./stf_para->verd_radial,
         NETZX+i,                                 // Hierhin speichern!
         NETZY+i,
         NETZZ+i);
   }

#ifdef DEBUG
   strcpy(datei_steuer,"netz_3D.dat");
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
      for (i=0*seed; i<stf_para->anz_schnitte*seed; i++)
      {
         fprintf(stream,"%6d %lf %lf %lf\n",i, NETZX[i], NETZY[i], NETZZ[i]);

      }

      fclose(stream);
   }
#endif

   printf("\nNetzknoten generiert!\n");

   // -------------------------------------------------------
   // AUSGABE GEO_FILE
   // -------------------------------------------------------

   printf("\nAnzahl 2D-Elemente: %5d\n", anz_elemente);
   printf("Anzahl Schnitte:    %5d\n", stf_para->anz_schnitte);
   printf("Anzahl Knoten:      %5d\n", (seed*stf_para->anz_schnitte));

   if (ga->gr->savegrid==1)
   {
      AUSGABE_3D_GEO(stf_para->geo_pfad, NETZX, NETZY ,NETZZ,
         stf_para->anz_schnitte,
         seed,
         el_liste,                                //Knotennummern der Elemente Nabe 2D
         anz_elemente);                           //Anzahl 2D-Elemente

      printf("\nGEO-File geschrieben nach %s!\n", stf_para->geo_pfad);
   }

   // ============================================================
   // Fill Geometry and Connectivity list for Covise arrays
   // ============================================================

   // fill Point list gg->p
   for (i = 0; i < stf_para->anz_schnitte * seed; i++)
   {
      AddPoint(gg->p, (float) NETZX[i], (float) NETZY[i], (float) NETZZ[i]);
   }

   // fill connectivity list gg->elem
   int elem[8];
   for (j = 0; j < stf_para->anz_schnitte-1; j++)
   {
      for (i=0; i < anz_elemente; i++)
      {
         elem[0] = el_liste[4*i]+j*seed;
         elem[1] = el_liste[4*i+1]+j*seed;
         elem[2] = el_liste[4*i+2]+j*seed;
         elem[3] = el_liste[4*i+3]+j*seed;
         elem[4] = el_liste[4*i]+(j+1)*seed;
         elem[5] = el_liste[4*i+1]+(j+1)*seed;
         elem[6] = el_liste[4*i+2]+(j+1)*seed;
         elem[7] = el_liste[4*i+3]+(j+1)*seed;
         AddElement(gg->e, elem);
      }
   }

   // ---------------------------------------------------------------
   // ERZEUGE 3D_RANDBEDINGUNGEN
   // ---------------------------------------------------------------
   /*
   strcpy(datei_steuer, "check_elmark.dat");

   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
      fprintf(stream, "VOR RECHNE_RB\n\n");
      fprintf(stream, "anz_kmark		   : %d\n", anz_kmark);
   fprintf(stream, "anz_elmark_einlauf: %d\n", anz_elmark_einlauf);
   fprintf(stream, "anz_elmark_eli	   : %d\n", anz_elmark_eli);
   fprintf(stream, "anz_elmark_ere	   : %d\n", anz_elmark_ere);
   fprintf(stream, "anz_elmark_11li   : %d\n", anz_elmark_11li);
   fprintf(stream, "anz_elmark_10re   : %d\n", anz_elmark_10re);
   fprintf(stream, "anz_elmark_15li   : %d\n", anz_elmark_15li);
   fprintf(stream, "anz_elmark_15re   : %d\n", anz_elmark_15re);
   fprintf(stream, "anz_elmark_auslauf: %d\n", anz_elmark_auslauf);
   fprintf(stream, "anz_wrb_ls        : %d\n\n", anz_wrb_ls);

   fclose(stream);
   }
   */
   //hier Dimensionen Anzahl der Knoten
   //elmark = new int[anz_elmark*4]; never used
   kmark = new int[anz_kmark];
   elmark_einlauf = new int[anz_elmark_einlauf*4];
   elmark_eli = new int[anz_elmark_eli*4];
   elmark_ere = new int[anz_elmark_ere*4];
   elmark_11li = new int[anz_elmark_11li*4];
   elmark_10re = new int[anz_elmark_10re*4];
   elmark_15li = new int[anz_elmark_15li*4];
   elmark_15re = new int[anz_elmark_15re*4];
   elmark_auslauf = new int[anz_elmark_auslauf*4];
   wrb_ls = new int[anz_wrb_ls*4];

   RECHNE_RB(
      stf_para->anz_schnitte,
      anz_elemente,
      ersatz_nr,
      seed,
      ase,
      anz_kmark,
      kmark,
      anz_wrb_ls,
      wrb_ls,
      anz_elmark,
      anz_elmark_einlauf,
      anz_elmark_eli,
      anz_elmark_ere,
      anz_elmark_11li,
      anz_elmark_10re,
      anz_elmark_15li,
      anz_elmark_15re,
      anz_elmark_auslauf,
      elmark_einlauf,
      elmark_eli,
      elmark_ere,
      elmark_11li,
      elmark_10re,
      elmark_15li,
      elmark_15re,
      elmark_auslauf);

   printf("\n\nAnzahl Randbedingungen und Markierungen je Schnitt (Gesamt)\n");
   printf("***********************************************************\n\n");

   printf("Wandrandbedingungen Leitschaufel:	%5d (%5d)\n", anz_wrb_ls,(anz_wrb_ls*(stf_para->anz_schnitte-1)));
   printf("Wandrandbedingungen Nabe und Kranz:	%5d (%5d)\n", anz_elemente, (anz_elemente*2));

   printf("Elementmarkierungen gesamt:    		%5d (%5d)\n", anz_elmark, (anz_elmark*(stf_para->anz_schnitte-1)));
   printf("Elementmarkierungen Einlauf links:	%5d (%5d)\n", anz_elmark_eli, (anz_elmark_eli*(stf_para->anz_schnitte-1)));
   printf("Elementmarkierungen Einlauf oben:	%5d (%5d)\n", anz_elmark_einlauf, (anz_elmark_einlauf*(stf_para->anz_schnitte-1)));
   printf("Elementmarkierungen Einlauf rechts:	%5d (%5d)\n", anz_elmark_ere, (anz_elmark_ere*(stf_para->anz_schnitte-1)));
   printf("Elementmarkierungen Geb 10 rechts:	%5d (%5d)\n", anz_elmark_10re, (anz_elmark_10re*(stf_para->anz_schnitte-1)));
   printf("Elementmarkierungen Auslauf rechts:	%5d (%5d)\n", anz_elmark_15re, (anz_elmark_15re*(stf_para->anz_schnitte-1)));
   printf("Elementmarkierungen Auslauf:		%5d (%5d)\n", anz_elmark_auslauf, (anz_elmark_auslauf*(stf_para->anz_schnitte-1)));
   printf("Elementmarkierungen Auslauf links:	%5d (%5d)\n", anz_elmark_15li, (anz_elmark_15li*(stf_para->anz_schnitte-1)));
   printf("Elementmarkierungen Geb 11 links:	%5d (%5d)\n", anz_elmark_11li, (anz_elmark_11li*(stf_para->anz_schnitte-1)));

   printf("Knotenmarkierungen:         		%5d (%5d)\n", anz_kmark, anz_kmark*stf_para->anz_schnitte);

   //-------------------------------------------------------------
   //Bestimmung der 2D-Elementnummern, bei denen bestimmte Gebiete
   //anfangen bzw. aufhren um spaeter Elementmarkierungen
   //best. 3D-Elementen zuordnen zu knnen
   //-------------------------------------------------------------

   pos=0;
   pos+=(ase[1][0]-1)*(ase[1][1]-1);
   pos+=(ase[2][0]-1)*(ase[2][1]-1);
   int start3 = pos;
   pos+=(ase[3][0]-1)*(ase[3][1]-1);
   pos+=(ase[4][0]-1)*(ase[4][1]-1);
   int start5 = pos;
   pos+=(ase[5][0]-1)*(ase[5][1]-1);
   pos+=(ase[6][0]-1)*(ase[6][1]-1);
   int end6 = pos;
   pos+=(ase[7][0]-1)*(ase[7][1]-1);
   int end7 = pos-1;
   pos+=(ase[8][0]-1)*(ase[8][1]-1);
   pos+=(ase[9][0]-1)*(ase[9][1]-1);
   int start10 = pos;
   pos+=(ase[10][0]-1)*(ase[10][1]-1);
   int start11 = pos;
   pos+=(ase[11][0]-1)*(ase[11][1]-1);
   pos+=(ase[12][0]-1)*(ase[12][1]-1);
   pos+=(ase[13][0]-1)*(ase[13][1]-1);
   pos+=(ase[14][0]-1)*(ase[13][1]-1);
   int start15 = pos;

   /*	
      printf("Start5=%6d\n", start5);
      printf("Start6=%6d\n", start6);
      printf("Ende7=%6d\n", end7);
      printf("Start10=%6d\n", start10);
      printf("Start11=%6d\n", start11);
      printf("Start15=%6d\n", start15);
   */

   // -------------------------------------------------------
   // Druckrandbedingungen
   // -------------------------------------------------------

   r_aus = new double[stf_para->anz_schnitte+1];
   vu2 = new double[stf_para->anz_schnitte];
   v2 = new double[stf_para->anz_schnitte];
   p2 = new double[stf_para->anz_schnitte];

   rho = 1000.;

   RECHNE_EINTEILUNG_L12(
      &stf_para->anz_schnitte,
      &r_max_nabe, &r_max_kranz,
      2,
      1./stf_para->verd_radial,
      r_aus);

   r_aus[stf_para->anz_schnitte]=r_max_kranz;
   //(Druck liegt auf dem Element, siehe naechste for-Schleife!)

   if ( ga->radial == 0 )                         // axial runner
   {
      A2 = M_PI * (pow(r_max_kranz,2) - (pow(r_max_nabe,2)));
   }
   else                                           // radial runner
   {
      A2 = M_PI * (pow(n_z[ga->phub->nump-1],2) - (pow(k_z[ga->pshroud->nump-1],2)));
   }

   vm2 = stf_para->Q / A2;

   ls_hoehe = n_z[0]-k_z[0];
   A1 = 2 * M_PI * ls_hikara * ls_hoehe;

   vm1=stf_para->Q / A1;

   vu1=vm1/(tan(grad2bog*ls_beta2));

   v1=pow((pow(vm1,2)+pow(vu1,2)),0.5);

   for (i=0; i<stf_para->anz_schnitte; i++)
   {
      vu2[i] = vu1 * ls_hikara / ((r_aus[i+1]+r_aus[i])/2);
      v2[i] = pow ( ( pow(vu2[i],2) + pow(vm2,2) ) , 0.5);
      //		printf("%lf\n",vu2[i]);
      //		printf("%lf\n",r_aus[i]);
      p2[i] = 0.5 * (pow(v1,2)-pow(v2[i],2)) * rho;
      //		printf("%lf\n",p2[i]);
   }

#ifdef DEBUG
   double d = p2[0];

   printf("Druckverlauf am Austritt von der Nabe zum Kranz:\n");
   for (i = 0; i < stf_para->anz_schnitte; i++)
   {
      p2[i] -= d;
      p2[i] = -p2[i];
      printf("Schnitt %d: %lf\n", i, p2[i]);
   }
#endif

   //printf("r_max_nabe=%lf\n", r_max_nabe);
   //printf("r_max_kranz=%lf\n", r_max_kranz);

   printf("\n\nGroessen zur Berechnung der Druckrandbedingungen\n");
   printf("************************************************\n\n");
   printf("Q=%lf\n",stf_para->Q);
   printf("rho=%lf\n",rho);
   printf("Austrittswinkel beta2: %lf\n", ls_beta2);
   printf("Hinterkantenradius: %lf\n", ls_hikara);
   printf("A1 (Flaeche Hinterkante Leitschaufel) =%lf [m2]\n",A1);
   printf("vu1=%lf [m/s]\n",vu1);
   printf("vm1=%lf [m/s]\n",vm1);
   printf("v1 =%lf [m/s]\n",v1);
   printf("A2 (Flaeche Austritt)                 =%lf [m2]\n",A2);
   printf("vm2=%lf\n",vm2);
   printf("vu2[Nabe]=%lf\n",vu2[0]);
   printf("v2[Nabe]=%lf\n",v2[0]);

#ifdef DEBUG
   strcpy(datei_steuer,"druck.dat");

   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
      for (i=0; i<stf_para->anz_schnitte; i++)
      {
         fprintf(stream, "%lf %lf \n", r_aus[i], p2[i]);
      }

      fclose(stream);
   }
#endif

   // -------------------------------------------------------
   // ERSTELLEN DER 3D-RB-ARRAYS FUER COVISE / FENFLOSS
   // -------------------------------------------------------

   RECHNE_3D_RB(gg,
      stf_para->anz_schnitte,
      seed,

      anz_wrb_ls,
      wrb_ls,
      anz_elemente,
      el_liste,
      stf_para->anz_grenz,

      anz_elmark_einlauf,
      anz_elmark_eli,
      anz_elmark_ere,
      anz_elmark_11li,
      anz_elmark_10re,
      anz_elmark_15li,
      anz_elmark_15re,
      anz_elmark_auslauf,

      elmark_einlauf,
      elmark_eli,
      elmark_ere,
      elmark_11li,
      elmark_10re,
      elmark_15li,
      elmark_15re,
      elmark_auslauf,
      anz_kmark,
      kmark,
      ase,
      start3,
      start5,
      end6,
      end7,
      start10,
      start11,
      start15,
      p2);

   if (ga->gr->savegrid==1)
   {

      // -------------------------------------------------------
      // AUSGABE RB_FILE
      // -------------------------------------------------------

      AUSGABE_3D_RB(
         stf_para->rb_pfad,
         stf_para->anz_schnitte,
         seed,

         anz_wrb_ls,
         wrb_ls,
         anz_elemente,
         el_liste,
         stf_para->anz_grenz,

         anz_elmark_einlauf,
         anz_elmark_eli,
         anz_elmark_ere,
         anz_elmark_11li,
         anz_elmark_10re,
         anz_elmark_15li,
         anz_elmark_15re,
         anz_elmark_auslauf,

         elmark_einlauf,
         elmark_eli,
         elmark_ere,
         elmark_11li,
         elmark_10re,
         elmark_15li,
         elmark_15re,
         elmark_auslauf,
         anz_kmark,
         kmark,
         ase,
         start3,
         start5,
         end6,
         end7,
         start10,
         start11,
         start15,
         p2);

      printf("\nRB-File geschrieben nach %s!\n", stf_para->rb_pfad);
   }

   // Speicher freigeben
   free(stf_para);
   delete [] fixr;
   delete [] fiyr;
   delete [] fixb11;
   delete [] fiyb11;
   delete [] fixb12;
   delete [] fiyb12;
   delete [] fixb21;
   delete [] fiyb21;
   delete [] fixb22;
   delete [] fiyb22;
   delete [] fixb31;
   delete [] fiyb31;
   delete [] fixb32;
   delete [] fiyb32;
   delete [] fixb41;
   delete [] fiyb41;
   delete [] fixb42;
   delete [] fiyb42;
   delete [] fixb52;
   delete [] fiyb52;
   delete [] fixb62;
   delete [] fiyb62;
   delete [] fixb92;
   delete [] fiyb92;
   delete [] fixb102;
   delete [] fiyb102;
   delete [] fixb111;
   delete [] fiyb111;
   delete [] fixb112;
   delete [] fiyb112;
   delete [] fixb12_2;
   delete [] fiyb12_2;
   delete [] fixb12_3;
   delete [] fiyb12_3;
   delete [] fixb15_1;
   delete [] fiyb15_1;
   delete [] fixb15_2;
   delete [] fiyb15_2;
   delete [] n_z;
   delete [] n_r;
   delete [] k_r;
   delete [] k_z;
   delete [] ls_x;
   delete [] ls_y;
   //	delete [] ls_xa;
   //	delete [] ls_ya;
   delete [] ls_r;
   delete [] n_bgl;
   delete [] k_bgl;
   delete [] ls_bgl_sort_n;
   delete [] ls_bgl_sort_k;
   delete [] ls_phi_sort;
   delete [] bgl_ls_n;
   delete [] bgl_ls_k;
   delete [] phi_ls;
   delete [] randpunkt;
   delete [] knot_nr;
   delete [] knotx;
   delete [] knoty;
   delete [] ersatz_nr;
   delete [] NETZX;
   delete [] NETZY;
   delete [] NETZZ;
   delete [] kranzx;
   delete [] kranzy;
   delete [] kranzz;
   delete [] kranzr;
   delete [] nabex;
   delete [] nabey;
   delete [] nabez;
   delete [] naber;
   delete [] el_liste;
   delete [] kmark;
   delete [] elmark_einlauf;
   delete [] elmark_eli;
   delete [] elmark_ere;
   delete [] elmark_11li;
   delete [] elmark_10re;
   delete [] elmark_15li;
   delete [] elmark_15re;
   delete [] elmark_auslauf;
   delete [] wrb_ls;
   delete [] r_aus;
   delete [] vu2;
   delete [] v2;
   delete [] p2;

   printf("\n================= leit-ende =================\n\n");

   return gg;
}


int   parameter_from_covise(struct stf *stf_para, struct gate *ga,
int *anz_punkte_ls,
double *n_r, double *n_z,
double *k_r, double *k_z,
double *ls_x, double *ls_y,
double *ls_r, double *ls_phi)
{
   int i, j;
   int pos;

   // set output paths
   stf_para->geo_pfad = (char *) "./gate.geo";
   stf_para->rb_pfad = (char *) "./gate.rb";

   // contour
   for(i = 0; i < ga->phub->nump; i++)
   {
      n_r[i] = ga->phub->y[i];
      n_z[i] = ga->phub->z[i];
   }
   for(i = 0; i < ga->pshroud->nump; i++)
   {
      k_r[i] = ga->pshroud->y[i];
      k_z[i] = ga->pshroud->z[i];
   }
   // outlet expansion
   if (ga->radial==0)
   {
      n_z[ga->phub->nump-1] -= ga->gr->len_expand_out;
      k_z[ga->pshroud->nump-1] -= ga->gr->len_expand_out;
   }

   if (ga->geofromfile==1)
   {
      ga->out_z = ga->phub->z[ga->phub->nump-1];
   }

   ga->out_z -= ga->gr->len_expand_out;

   // profile

   pos = 0;
   for (i = 0; i < *anz_punkte_ls; i++)
   {
      ls_x[pos] = ga->ss->x[i];
      ls_y[pos] = ga->ss->y[i];
      ls_r[pos] = pow ( pow(ls_x[pos],2) + pow(ls_y[pos],2) , 0.5 );
      ls_phi[pos] = asin ( ls_y[pos] / ls_r[pos] );
      pos++;
   }
   for (i = *anz_punkte_ls-1; i >= 0; i--)
   {
      ls_x[pos] = ga->ps->x[i];
      ls_y[pos] = ga->ps->y[i];
      ls_r[pos] = pow ( pow(ls_x[pos],2) + pow(ls_y[pos],2) , 0.5 );
      ls_phi[pos] = asin ( ls_y[pos] / ls_r[pos] );
      pos++;
   }

   for (i = *anz_punkte_ls-1; i > *anz_punkte_ls-3; i--)
   {
      ls_x[pos] = ga->cl->x[i];
      ls_y[pos] = ga->cl->y[i];
      ls_r[pos] = pow ( pow(ls_x[pos],2) + pow(ls_y[pos],2) , 0.5 );
      ls_phi[pos] = asin ( ls_y[pos] / ls_r[pos] );
      pos++;
   }

   stf_para->anz_ls = ga->nob;
   stf_para->Q = ga->Q;
   stf_para->PG2 = ga->gr->edge_ps;
   stf_para->PG3 = 100;                           // so far not adjustable!
   stf_para->PG4 = ga->gr->edge_ss;
   stf_para->grenz = ga->gr->bound_layer;
   stf_para->anz_schnitte = ga->gr->n_rad;
   stf_para->anz_grenz = ga->gr->n_bound;
   stf_para->anz_15 = ga->gr->n_out;
   stf_para->anz_einlauf = ga->gr->n_in;
   stf_para->anz_knot1 = ga->gr->n_blade_ps_back;
   stf_para->anz_knot2 = ga->gr->n_blade_ps_front;
   stf_para->anz_knot3 = ga->gr->n_blade_ss_front;
   stf_para->anz_knot4 = ga->gr->n_blade_ss_back;

   // calculate in % between trailing edge and outlet
                                                  // [%]
   stf_para->bgl_15_n = (float) ga->gr->len_start_out_hub;
                                                  // [%]
   stf_para->bgl_15_k = (float) ga->gr->len_start_out_shroud;

   // calculation of bgl_max_nabe

   stf_para->bgl_max_nabe=0.;

   // axial gate:
   // calculate bgl_max_nabe from z_out!
   // TODO: error, if z_out is not within the range given by the meridian contour point list (possible, if geofromfile == 1)
   if ( ga->radial == 0 )
   {
      i = 0;
      while (n_z[i] > ga->out_z)
      {
         i++;
      }
      for (j = 1; j < i; j++)
      {
         stf_para->bgl_max_nabe += pow ( pow(n_z[j]-n_z[j-1] , 2) + pow(n_r[j]-n_r[j-1] , 2) , 0.5 );
      }
      stf_para->bgl_max_nabe += ( ga->out_z - n_z[i-1] ) / ( n_z[i] - n_z[i-1] ) * pow ( pow(n_z[i]-n_z[i-1] , 2) + pow(n_r[i]-n_r[i-1] , 2) , 0.5 );
   }
   // radial gate
   // so far only if geofromfile (TODO: change parameters to enable design of radial gate)
   if ( ga->radial == 1 )
   {
      for ( i = 1; i < ga->phub->nump; i++)
      {
         stf_para->bgl_max_nabe += pow ( pow(n_z[i]-n_z[i-1] , 2) + pow(n_r[i]-n_r[i-1] , 2) , 0.5 );
      }
   }

   stf_para->bgl_start = - ga->gr->len_expand_in;
   stf_para->bgl_aus = ga->gr->len_expand_out;
   stf_para->lvs1 = ga->gr->comp_ps_back;
   stf_para->lvs2 = ga->gr->comp_ps_front;
   stf_para->lvs3 = ga->gr->comp_ss_front;
   stf_para->lvs4 = ga->gr->comp_ss_back;
   stf_para->lvs5 = ga->gr->comp_trail;
   stf_para->lvs6 = ga->gr->comp_out;
   stf_para->lvs7 = ga->gr->comp_in;
   stf_para->bvs1 = ga->gr->comp_bound;
   stf_para->bvs2 = ga->gr->comp_bound;
   stf_para->bvs3 = ga->gr->comp_bound;
   stf_para->bvs4 = ga->gr->comp_bound;
   stf_para->bvs8 = ga->gr->comp_middle;
   stf_para->verd_radial = ga->gr->comp_rad;
   //stf_para->versch_eintr = ga->gr->shift_in;	// not necessary any more
   stf_para->versch_austr = ga->gr->shift_out;

   // calculate dependent parameters here
   stf_para->anz_mitte = stf_para->anz_knot4 - stf_para->anz_knot1 + 1;
                                                  //Gesamtanzahl Breite Austritt
   stf_para->anz_breit = 2 * (stf_para->anz_grenz) + (stf_para->anz_mitte) - 2;
   stf_para->anz_knot9 = stf_para->anz_knot4 - stf_para->anz_knot1 + 1;
   stf_para->PG = 201;                            //Anzahl der Randpunkte Leitschaufel nach erstem Meshseed
   stf_para->PG1 = 0;
   stf_para->PG5 = stf_para->PG - 1;

   // change direction of lvs7
   stf_para->lvs7 = 1. / stf_para->lvs7;

   return 1;
}


int WriteGGrid(struct ggrid *gg, const char *fn)
{
   int i;
   int res;
   char buf[256];
   FILE *fp;

   res = 0;
   sprintf(buf, "%s.geo", fn);
   if ((fp = fopen(buf, "w")) != NULL)
   {
      fputs("## Geomtriedaten (automatisch erzeugt)\n\n\n\n\n\n\n\n\n\n", fp);
      fprintf(fp, "%d %d 0 0 %d %d %d %d\n", gg->p->nump, gg->e->nume,
         gg->p->nump, gg->e->nume,
         gg->p->nump, gg->e->nume);
      for (i = 0; i < gg->p->nump; i++)
      {
         fprintf(fp, "%5d %12.6f %12.6f %12.6f 0\n", i+1, gg->p->x[i], gg->p->y[i], gg->p->z[i]);
      }
      for (i = 0; i < gg->e->nume; i++)
      {
         fprintf(fp, "%6d %6d %6d %6d %6d %6d %6d %6d %6d 0\n",i+1,
            gg->e->e[i][0]+1, gg->e->e[i][1]+1, gg->e->e[i][2]+1, gg->e->e[i][3]+1,
            gg->e->e[i][4]+1, gg->e->e[i][5]+1, gg->e->e[i][6]+1, gg->e->e[i][7]+1);
      }
      fclose(fp);
      res++;
   }
   return res;
}


/*
#ifdef DEBUG
void DumpGGrid(struct ggrid *gg)
{
   //SI(gg->num_o);
   WriteGGrid(gg, "ggrid");
   WriteGBoundaryConditions(gg, "ggrid");
}
#endif
*/
