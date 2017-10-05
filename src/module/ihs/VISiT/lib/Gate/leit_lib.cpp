
#include <string.h>
#ifndef _WIN32
#include <strings.h>
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#include <Gate/include/basis.h>
#include <Gate/include/myspline.h>
#include <Gate/include/kubsplin.h>
#include <Gate/include/vmblock.h>
#include <Gate/include/spliwert.h>
#include <Gate/include/bezier.h>

#include "General/include/ilist.h"
#include "General/include/flist.h"
#include "Gate/include/ggrid.h"

#define grad2bog M_PI/180.
#define bog2grad 180./M_PI

// =========================================================
// Berechnen des Abstandes zwischen zwei Punkten
// =========================================================

double abstand(double x1, double y1, double x2, double y2)
{
   double l = sqrt  ( (x1 - x2) * (x1 - x2)  + (y1 - y2) * (y1 - y2) );
   return(l);
}


// ----------------------------------------------------------------------
// EINT_L12
// ----------------------------------------------------------------------

void RECHNE_EINTEILUNG_L12(
int *NL,
double *TBEG,  double *TEND,
int   mo,
double L12,
double *TAB)

// Teilt ein Intervall in einer beliebigen Anzahl und Einteilung auf

// EINGABE

// NL......Anzahl der gewuenschten Punkte
// TBEG....Anfang der Ausgabe von Punkten
// TEND....Ende der Ausgabe von Punkten
// mo......Modus: M=0 Aequidistante Einteilung (L12 ignoriert)
//                M=1 Einteilung im Verhaeltnis L1/L2 (eine Richtung)
//                M=2 Einteilung im Verhaeltnis L1/L2 (beide Richt.)
// L12.....Verhaeltnis L1/L2
//

// AUSGABE

// TAB.....Einteilung (TBEG..TEND)

{
   int i;
   double exponent;
   double *laenge;
   double *sw;
   double start;
   int anzahl;
   laenge = (double *)malloc((*NL)*sizeof(double));
   sw    = (double *)malloc((*NL)*sizeof(double));

   sw[0]=1.;
   sw[(*NL)-2]=1.;
   laenge[0]=0.;

   // Lineare Verteilung

   if (mo==0)
   {
      for (i=0;i<(*NL)-1;i++)    sw[i]=1;
   }

   // Exponentielle Verteilung, einseitig

   if (mo==1)
   {
      exponent = pow((1./L12),(1./((double)(*NL)-2)));
      for (i=1;i<(*NL)-1;i++)    sw[i]=sw[i-1] * exponent;
   }

   // Exponentielle Verteilung, zweiseitig

   if (mo==2)
   {
      exponent = pow((1./L12),(2./((double)(*NL)-3)));
      for (i=1;i<((*NL)-1)/2;i++)
      {
         sw[i]    =sw[i-1]  * exponent;
         sw[(*NL)-i-2]  =sw[(*NL)-i-1] * exponent;
      }
   }

   // Hyperbolische Verteilung, einseitig

   if (mo==3)
   {
      anzahl = (*NL)-1;
      start = ((double)((anzahl)-1))/((1./L12)-1.);
      for (i=0;i<(*NL)-1;i++)
      {
         sw[i]    = start + (double) (i);
      }
   }

   // Hyperbolische Verteilung, zweiseitig

   if (mo==4)
   {
      anzahl = (int)((*NL)-1)/2;
      start = ((double)((anzahl)-1))/((1./L12)-1.);
      for (i=0;i<((*NL)-1)/2;i++)
      {
         sw[i]    = start + (double) (i);
         sw[(*NL)-i-2]  = sw[i];
      }
   }

   if (mo==5)
   {
      anzahl = (int)((*NL)-1)/2;
      start = ((double)((anzahl)-1))/((1./L12)-1.);
      for (i=0;i<((*NL)-1)/2;i++)
      {
         sw[i]    = start + (double) (i);
         sw[(*NL)-i-2]  = sw[i];
      }
   }

   for (i=1;i<(*NL);i++)      laenge[i]=laenge[i-1]+sw[i-1];

   for (i=0;i<(*NL);i++)
   {
      *TAB=*TBEG+(*TEND-*TBEG)*laenge[i]/laenge[*NL-1];
      TAB++;
   }

   free(laenge);
   free(sw);
}


// ----------------------------------------------------------------------
// MESHSEED
// ----------------------------------------------------------------------

void MESHSEED(    double *XK, double *YK,
int MO,
double AA,  double AE,
int PA,     int PE,
double VX,  double VY,  double VT,
int M,
double L12,
int anz_naca,
int NL,
double *XS, double *YS)

// Berechnet die Punkte eines kubischen Splines

// EINGABE
// XK...Stuetzpunkt x-Koordinate
// YK...Stuetzpunkt y-Koordinate
// MO...Modus: 0..Anfangs/Endsteigung wird ueber nachfolgende/
//                vorhergehenden Knoten berechnet
//             1..Angabe der Steigungen ueber AA,AE
// AA...1. Ableitung am 1. Stuetzpunkt (wird bei MO=0 ignoriert)
// AE...1. Ableitung am letzten Stuetzpunkt (wird bei MO=0 ignoriert)
// PA...Nummer des Stuetzpunktes ab dem Punkte gesetzt werden sollen
// PE...Nummer des Stuetzpnuktes bis zu dem Punkte gesetzt werden
//      sollen
// VX...Betrag der Verschiebung in x-Richtung
// VY...Betrag der Verschiebung in y-Richtung
// VT...Betrag der Verschiebung in Richtung des Normalenvektors
// M....Modus: M=0 Aequidistante Einteilung (L12 ignoriert)
//             M=1 Einteilung im Verhaeltnis L1/L2 (eine Richtung)
//             M=2 Einteilung im Verhaeltnis L1/L2 (beide Richtungen)
// L12..Verhaeltnis L1/L2
// anz_naca..Anzahl der Stuetzpunkte
// NL...Anzahl der gewuenschten Kurvenpunkte

// AUSGABE
// XS...Punkte x-Koordinate
// YS...Punkte y-Koordinate

{
   int i;
   double *ALPHA, *BETA;
   double *T;
   double *X, *Y;
   double *BX, *CX, *DX;
   double *KBY, *CY, *DY;
   double *TSOLL;
   double *XTAB, *YTAB;
   double *PHI;
   double xabl, yabl;

   ALPHA = (double *)malloc(2*sizeof(double));
   BETA  = (double *)malloc(2*sizeof(double));

   X = (double *)malloc((anz_naca)*sizeof(double));
   Y = (double *)malloc((anz_naca)*sizeof(double));
   T = (double *)malloc((anz_naca)*sizeof(double));

   BX = (double *)malloc((anz_naca)*sizeof(double));
   CX = (double *)malloc((anz_naca)*sizeof(double));
   DX = (double *)malloc((anz_naca)*sizeof(double));

   KBY = (double *)malloc((anz_naca)*sizeof(double));
   CY = (double *)malloc((anz_naca)*sizeof(double));
   DY = (double *)malloc((anz_naca)*sizeof(double));

   TSOLL = (double *)malloc(NL*sizeof(double));

   XTAB = (double *)malloc(NL*sizeof(double));
   YTAB = (double *)malloc(NL*sizeof(double));
   PHI = (double *)malloc(NL*sizeof(double));

   ALPHA[1]=0;
   BETA [1]=0;

   if (MO==0)
   {
      ALPHA[0] = (*(YK+1) - *YK) / (*(XK+1) - *XK);
      BETA[0] = (*(YK+(anz_naca)-1) - *(YK+(anz_naca)-2))
         /(*(XK+(anz_naca)-1) - *(XK+(anz_naca)-2));
   }
   else
   {
      ALPHA[0]=AA;
      BETA [0]=AE;
   }

   for (i=0;i<anz_naca;i++)
   {
      X[i]=*XK;
      Y[i]=*YK;
      XK++;
      YK++;
   }

   // Splinekoeffizienten bestimmen
   parspl (anz_naca,X,Y,4,ALPHA,BETA,0,T,BX,CX,DX,KBY,CY,DY);

   // Punkteverteilung bestimmen
   RECHNE_EINTEILUNG_L12(&NL, &T[PA], &T[PE], M, L12, TSOLL);

   // Splineauswertung
   for (i=0;i<NL;i++)
   {
      pspwert (anz_naca-1,TSOLL[i],T,X,BX,CX,DX,Y,KBY,CY,DY,&XTAB[i],&YTAB[i],&xabl,&yabl);
      PHI[i]=atan2(yabl,xabl);
   }

   // Verschiebung in Normalenrichtung aus den Ableitungen
   // berechnen.
   if (VT!=0.)
   {
      for (i=0;i<NL;i++)
      {
         XTAB[i]=XTAB[i]+VT*cos(PHI[i]+M_PI/2.);
         YTAB[i]=YTAB[i]+VT*sin(PHI[i]+M_PI/2.);
      }
   }

   for (i=0;i<NL;i++)
   {
      XS[i]=XTAB[i]+VX;
      YS[i]=YTAB[i]+VY;
   }

   free(BX);
   free(CX);
   free(DX);
   free(KBY);
   free(CY);
   free(DY);
   free(T);
   free(XTAB);
   free(YTAB);
   free(X);
   free(Y);
   free(ALPHA);
   free(BETA);
   free(TSOLL);
   free(PHI);

}


// ----------------------------------------------------------------------
// GERADE
// ----------------------------------------------------------------------

void GERADE(      double *x1, double *y1,
double *x2, double *y2,
int seed,
int m,
double L12,
double *x,  double *y)

// Berechnet die Punkte einer Gerade

// EINGABE
// x1,y1...Koordinaten Startpunkt
// x2,y2...Koordinaten Endpunkt
// seed....Anzahl der Punkte
// m.......Modus: m=0 Aequidistante Einteilung (L12 ignoriert)
//                m=1 Einteilung im Verhaeltnis L1/L2 (eine Seite)
//                m=2 Einteilung im Verhaeltnis L1/L2 (beide Seiten)
// l12.....Verhaeltnis L1/L2

// AUSGABE
// x,y Koordinaten Geradenpunkte

{
   int i;
   double start=0.;
   double ende=1.;
   double *TAB;

   TAB = (double *)malloc(seed*sizeof(double));

   RECHNE_EINTEILUNG_L12(&seed, &start, &ende, m, L12, TAB);

   for (i=0;i<seed;i++)
   {
      *(x+i) = *x1 + ((*x2)-(*x1)) * (*(TAB+i));
      *(y+i) = *y1 + ((*y2)-(*y1)) * (*(TAB+i));
   }

   free(TAB);
}


// ------------------------------------------------------------------
// DREIECK
// ------------------------------------------------------------------

void DREIECK(double *fixbord1,   double *fiybord1,
double *fixbord2, double *fiybord2,
double *fixbord3, double *fiybord3,
int ase,
double *fixr12,
double *fiyr12)
{

   int i;
   int ase_temp;
   double halbx1, halby1;
   double halbx2, halby2;
   double halbx3, halby3;
   double mittelx, mittely;

   double *seitenhalbx1, *seitenhalby1;
   double *seitenhalbx2, *seitenhalby2;
   double *seitenhalbx3, *seitenhalby3;

   ase_temp = (int)(ase+1)/2;

   seitenhalbx1 = new double[ase_temp];
   seitenhalby1 = new double[ase_temp];

   seitenhalbx2 = new double[ase_temp];
   seitenhalby2 = new double[ase_temp];

   seitenhalbx3 = new double[ase_temp];
   seitenhalby3 = new double[ase_temp];

   // Schwerpunkt

   mittelx = ((*fixbord1)+(*fixbord2)+(*fixbord3))/3.;
   mittely = ((*fiybord1)+(*fiybord2)+(*fiybord3))/3.;

   // Endpunkte der Seitenhalbierenden

   halbx1= *(fixbord1 +(ase_temp-1));
   halby1= *(fiybord1 +(ase_temp-1));

   halbx2= *(fixbord2 +(ase_temp-1));
   halby2= *(fiybord2 +(ase_temp-1));

   halbx3= *(fixbord3 +(ase_temp-1));
   halby3= *(fiybord3 +(ase_temp-1));

   // Berechnung des kürzeren Teils der Seitenhalbierenden

   GERADE(&halbx1,   &halby1,                     // Startpunkt
      &mittelx,   &mittely,                       // Endpunkt
      ase_temp, 0, 0.,                            // Anz. Punkte quer
      seitenhalbx1,                               // Hierhin speichern!
      seitenhalby1);

   GERADE(&halbx2,   &halby2,                     // Startpunkt
      &mittelx,   &mittely,                       // Endpunkt
      ase_temp, 0, 0.,                            // Anz. Punkte quer
      seitenhalbx2,                               // Hierhin speichern!
      seitenhalby2);

   GERADE(&halbx3,   &halby3,                     // Startpunkt
      &mittelx,   &mittely,                       // Endpunkt
      ase_temp, 0, 0.,                            // Anz. Punkte quer
      seitenhalbx3,                               // Hierhin speichern!
      seitenhalby3);

   // Erstes Dreiecksgebiet

   for (i=0;i<ase_temp;i++)
   {
      GERADE(  seitenhalbx3+i,                    // Startpunkte
         seitenhalby3+i,
         fixbord1+i,                              // Endpunkte
         fiybord1+i,
         ase_temp, 0, 0.,                         // Anz. Punkte quer
         fixr12 +i*ase_temp,                      // Hierhin speichern!
         fiyr12 +i*ase_temp);
   }

   // Zweites Dreiecksgebiet

   for (i=0;i<ase_temp;i++)
   {
      GERADE( seitenhalbx1+i,                     // Startpunkte
         seitenhalby1+i,
         fixbord2+i,                              // Endpunkte
         fiybord2+i,
         ase_temp, 0, 0.,                         // Anz. Punkte quer
         fixr12 +i*ase_temp + ase_temp*ase_temp,  // Hierhin speichern!
         fiyr12 +i*ase_temp + ase_temp*ase_temp);
   }

   // Drittes Dreiecksgebiet

   for (i=0;i<ase_temp;i++)
   {
      GERADE(fixbord2+ase-1 -i,                   // Startpunkte
         fiybord2+ase-1 -i,
         seitenhalbx3+i,                          // Endpunkte
         seitenhalby3+i,
         ase_temp, 0, 0.,                         // Anz. Punkte quer
         fixr12 +i*ase_temp + 2*ase_temp*ase_temp,// Hierhin speichern!
         fiyr12 +i*ase_temp + 2*ase_temp*ase_temp);
   }

   delete[] seitenhalbx1;
   delete[] seitenhalby1;

   delete[] seitenhalbx2;
   delete[] seitenhalby2;

   delete[] seitenhalbx3;
   delete[] seitenhalby3;

}


// ----------------------------------------------------------------------
// BEZIER
// ----------------------------------------------------------------------

void BEZIER(double *x1,
double *y1,
double phi_start,
double *x2,
double *y2,
double phi_ende,
int modus,
int *anz_pkt,
int m,
double *L12,
double *x,  double *y)

// Berechnet die Punkte einer Gerade

// EINGABE
// x1,y1...Koordinaten Startpunkt
// x2,y2...Koordinaten Endpunkt
// seed....Anzahl der Punkte
// m.......Modus: m=0 Aequidistante Einteilung (L12 ignoriert)
//                m=1 Einteilung im Verhaeltnis L1/L2 (eine Seite)
//                m=2 Einteilung im Verhaeltnis L1/L2 (beide Seiten)
// l12.....Verhaeltnis L1/L2

// AUSGABE
// x,y Koordinaten Geradenpunkte

{
   int i;
   int anz_interpol=50;
   double start=0.;
   double ende=1.;

   //double bez_start=0.5;
   //double bez_ende=0.5;
   double *laenge, *laenge_plus;
   void *vmblock;
   double **b, **d;
   double xs, ys;
   double z;
   double fak_y=0.2;
   double *TAB;
   double abstand;

   vmblock = vminit();                            /* Speicherblock initialisieren */
   b = (REAL **)vmalloc(vmblock, MATRIX, 3*4, 4);
   d = (REAL **)vmalloc(vmblock, MATRIX, 4, 4);

   TAB = (double *)malloc((*anz_pkt)*sizeof(double));
   laenge      = (double *)malloc(anz_interpol*sizeof(double));
   laenge_plus = (double *)malloc(anz_interpol*sizeof(double));

   d[0][0] = *x1;
   d[0][1] = *y1;
   d[0][2] = 0;

   d[3][0] = *x2;
   d[3][1] = *y2;
   d[3][2] = 0;

   if (modus==0)
   {

      // Der Einlauf des Rechennetzes erfolgt schiefwinklig
      ys = (*y1) + fak_y*((*y2)-(*y1));
      xs = *x1 + fabs(((ys-(*y1)) / tan(grad2bog*phi_start)));

      d[1][0] = *x1 + fak_y*(xs-(*x1));
      d[1][1] = *y1 + fak_y*(ys-(*y1));

      d[2][0] = xs + (1-fak_y)*((*x2)-xs);
      d[2][1] = ys + (1-fak_y)*((*y2)-ys);
   }

   if (modus==1)
   {
      /*// Der Einlauf des Rechennetzes erfolgt schiefwinklig
      ys = (*y1) + fabs(((*x2)-(*x1))*tan(grad2bog*phi));
      xs = *x2;

      d[1][0] = *x1 + fak_y*(xs-(*x1));
      d[1][1] = *y1 + fak_y*(ys-(*y1));

      d[2][0] = xs + (1-fak_y)*((*x2)-xs);
      d[2][1] = ys + (1-fak_y)*((*y2)-ys);*/

      // Vertikaler Einlauf
      abstand=pow(pow((*x1)-(*x2),2.)+pow((*y1)-(*y2),2.),0.5);

      d[1][0] = *x1 + fak_y*abstand * cos(grad2bog*phi_start);
      d[1][1] = *y1 + fak_y*abstand * sin(grad2bog*phi_start);

      d[2][0] = *x2;
      d[2][1] = *y2-fak_y*abstand;
   }

   if (modus==2)
   {
      // Gewichtspunkte gemaess Winkel
      abstand=pow(pow((*x1)-(*x2),2.)+pow((*y1)-(*y2),2.),0.5);

      d[1][0] = *x1 + fak_y*abstand * cos(grad2bog*phi_start);
      d[1][1] = *y1 + fak_y*abstand * sin(grad2bog*phi_start);

      d[2][0] = *x2 + fak_y*abstand * cos(grad2bog*phi_ende);
      d[2][1] = *y2 + fak_y*abstand * sin(grad2bog*phi_ende);
   }

   // Berechnung der Bezierpunkte (Definition der Kurve)

   kubbez(  b,                                    // Gewichtspunkte
      d,                                          // Bezierpunkte
      *anz_pkt,                                   // Genauigkeit
      laenge,                                     // Bogenlaengen der Stuetzpunkte
      laenge_plus,                                // Bogenlaengen dimensionslos
      3,                                          // Anzahl der Splinestuecke
      2);                                         // 2,3 fuer ebene, Raumkurve

   RECHNE_EINTEILUNG_L12(anz_pkt, &start, &ende, m, *L12, TAB);

   // Berechnung von Punkten auf der Kurve in Abhängigkeit
   // der Bogenlänge s! (nicht des Parameters t!)

   for (i=0;i<(*anz_pkt);i++)
   {
      valbez( 1,                                  // Bogenlänge s verwenden!
         TAB[i],                                  // gewünschte Bogenlaenge
         3,                                       // Anzahl der Splinestuecke
         2,                                       // ebene Kurve
         b,                                       // Bezierpunkte
         *anz_pkt,                                // Anzahl der interpol. Punkte
         laenge_plus,                             // dimensionslose Bogenlänge von t
         (x +i),                                  // Hierhin speichern!
         (y +i),
         &z);
   }

   free(TAB);
   free(laenge);
   free(laenge_plus);
}


// ----------------------------------------------------------------------
// RECHNE_EINLAUF
// ----------------------------------------------------------------------

void RECHNE_EINLAUF(int schnitt,
double *fixb32,
double *yo, double *delta,
double *fixplo, double *fiyplo,
double *fixpro, double *fiypro)

// Berechnet díe oberen Randpunkte

// EINGABE
// fixb31..... X-Koordinate des Staupunkts
// fiyb31..... Y-Koordinate des Staupunkts
// fixb32..... X-Koordinate des Staupunkts um bgs verschoben
// fiyb32..... Y-Koordinate des Staupunkts um bgs verschoben
// yo......... obere Grenze

// AUSGABE
// fixplo..... X-Koordinate des oberen Randpunktes links
// fiyplo..... Y-Koordinate des oberen Randpunktes links
// fixpro..... X-Koordinate des oberen Randpunktes rechts
// fiypro..... Y-Koordinate des oberen Randpunktes rechts

{

   static double xplo;
   static double xpro;

   *fiypro = *yo;
   *fiyplo = *yo;

   if (schnitt==0)
   {
      // jetzt radial ab vorderstem LS-Punkt!
      xplo = *fixb32;

      xpro = xplo - *delta;
   }

   *fixplo = xplo;
   *fixpro = xpro;

}


// ------------------------------------------------------------------
// RECHNE_MITTELPUNKTE
// ------------------------------------------------------------------

void RECHNE_MITTELPUNKTE(
double *fixplo,   double *fixpro,
double *yo,
int *dat1,  int *dat2, int *dat3,
double *fixpmlo,  double *fiypmlo,
double *fixpmro,  double *fiypmro)

// Berechnet díe oberen Randpunkte

// EINGABE
// fixplo..... X-Koordinate des Randpunkts links
// fixpro..... X-Koordinate des Randpunkts rechts
// yo......... obere Grenze
// dat1....... Anzahl der Punkte des ersten Gebiets (5)
// dat2....... Anzahl der Punkte des zeiten Gebiets (7)
// dat3....... Anzahl der Punkte des dritten Gebiets (6)

// AUSGABE
// fixpmlo..... X-Koordinate des oberen Mittelpunktes links
// fiypmlo..... Y-Koordinate des oberen Mittelpunktes links
// fixpmro..... X-Koordinate des oberen Mittelpunktes rechts
// fiypmro..... Y-Koordinate des oberen Mittelpunktes rechts

{
   double gesamt;

   gesamt = (double)((*dat1-1) + (*dat2-1) + (*dat3-1));

   *fiypmro = *yo;
   *fiypmlo = *yo;

   *fixpmlo = *fixplo + (*fixpro - *fixplo) * (double)(*dat1-1) / gesamt;
   *fixpmro = *fixpro - (*fixpro - *fixplo) * (double)(*dat3-1) / gesamt;

}


//=================================================================
// doppelte Punkte rausschmeißen
//=================================================================

void DOPPELTE_WEG (double *fixr, double *fiyr,
int seed, double TOL,
int *knot_nr, int *randpunkt,
double *neux, double *neuy,
int *anz_doppelt, int *ersatz_nr)

// Entfernt (unter Kenntnis ob Randpunkt) alle doppelten Punkte

// EINGABE

// fixr.........X-Wert Gesamtfeld
// fiyr.........Y-Wert Gesamtfeld
// seed.........Anzahl der Knoten
// TOL..........Toleranz (kleinster Punktabstand)
// knot_nr......Fortlaufend, =-1 wenn doppelt
// randpunkt....Feld Randpunkt ja / nein

// AUSGABE

// neux.........X-Wert Feld ohne doppelte
// neuy.........Y-Wert Feld ohne doppelte
// anz_doppelt..Anzahl der doppelten Punkte
// ersatz_nr....Gibt die neue Nummer der doppelten Punkte an

{
   *anz_doppelt=0;
   int i,j;
   int z=0;

   int *versch;
   versch = new int[seed];

   for (j=0; j<seed-1; j++)                       //hier hoellisch aufpassen!!!
   {
      if (randpunkt[j]==0) continue;

      for (i=j+1; i<seed; i++)
      {                                           //nur Randpunkte koennen doppelt sein
         if (randpunkt[i]==0) continue;           //kein Randpunkt

         if (knot_nr[i]==-1) continue;            //schon rausgeschmissen

         double dx = fixr[i] - fixr[j];
         double dy = fiyr[i] - fiyr[j];
         if ( fabs(dx) > TOL ) continue;          //nicht doppelt
         if ( fabs(dy) > TOL ) continue;          //nicht doppelt

         *(knot_nr+i)=-1;                         //doppelter Punkt!
         (*anz_doppelt)++;
         ersatz_nr[i]=j;                          //Ersatznummer im alten System!
      }
   }

   j=0;

   for (i=0; i<seed; i++)
   {
      if (knot_nr[i]==-1) continue;
      *(neux+j)=*(fixr+i);                        //nur nicht doppelte Punkte in neux
      *(neuy+j)=*(fiyr+i);                        //Koordinaten im neuen Feld
      j++;
   }

   for (j=0; j<seed; j++)                         //Anzahl der vorhergehenden "Rausschmeißer"
   {                                              //(ist Verschiebung des Feldes)
      z = 0;
      for (i=0; i<j; i++)
      {
         if (knot_nr[i]==-1) z++;
      }
      versch[j]=z;
   }

   for(i=0; i<seed; i++)                          //alle Knoten sind dann der doppelten bereinigt
   {
      if (knot_nr[i]!=-1) ersatz_nr[i]=knot_nr[i]-versch[i];
      else ersatz_nr[i]= ersatz_nr[i]-versch[ersatz_nr[i]];
   }

   delete[] versch;

}


// ----------------------------------------------------------------------
// 3D-GERADE
// ----------------------------------------------------------------------

void GERADE3D(double *x1, double *y1, double *z1,
double *x2, double *y2, double *z2,
int anz,
int seed,
int m,
double L12,
double *x, double *y, double*z)

// Berechnet die Punkte einer Gerade

// EINGABE
// x1,y1,z1...Koordinaten Startpunkt
// x2,y2,z2...Koordinaten Endpunkt
// anz........Anzahl der Punkte Gerade
//	seed.......Gesamtfeldgröße 2D! Reihenfolge!!
// m..........Modus: m=0 Aequidistante Einteilung (L12 ignoriert)
//                m=1 Einteilung im Verhaeltnis L1/L2 (eine Seite)
//                m=2 Einteilung im Verhaeltnis L1/L2 (beide Seiten)
// l12........Verhaeltnis L1/L2

// AUSGABE
// x,y,z......Koordinaten Geradenpunkte

{
   int i;
   double start=0.;
   double ende=1.;
   double *TAB;

   TAB = (double *)malloc(seed*sizeof(double));

   RECHNE_EINTEILUNG_L12(&anz, &start, &ende, m, L12, TAB);

   for (i=0;i<anz;i++)
   {
      *(x+seed*i) = *x1 + ((*x2)-(*x1)) * (*(TAB+i));
      *(y+seed*i) = *y1 + ((*y2)-(*y1)) * (*(TAB+i));
      *(z+seed*i) = *z1 + ((*z2)-(*z1)) * (*(TAB+i));

   }

   free(TAB);
}


//===============================================================
// Erstellen des GEO-Files
//===============================================================

void AUSGABE_3D_GEO(char *geo_pfad, double *NETZX, double *NETZY , double *NETZZ,
int anz_schnitte,
int seed,
int *elliste,                                     //Knotennummern der Elemente Nabe 2D
int anz_elemente)                                 //Anzahl 2D-Elemente

//EINGABE

//NETZX, NETZY, NETZZ.........Koordinaten der 3D-Knoten
//anz_schnitte................Anzahl der "Ebenen" inkl Nabe und Kranz
//seed........................Anzahl der Knoten 2D
//elliste....................jeweils 4 Eckpunkte der 2D-Elemente
//anz_elemente................Anzahl der 2D-Elemente

//Ausgabe

//GEOFILE Leit.GEO

{

   int anz_knot3D = anz_schnitte * seed;
   int anz_elemente3D = anz_elemente * (anz_schnitte-1);

   int i,j;

   FILE *stream;
   char datei_steuer[200];

   strcpy(datei_steuer, geo_pfad);
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
      for (i=0; i<10; i++)
      {
         fprintf(stream,"C\n");
      }

      fprintf(stream,"%6d %6d %d %d %d %d %6d %6d\n", anz_knot3D, anz_elemente3D, 0, 0, 0, 0, anz_knot3D, anz_elemente3D);

      for (i=0; i<anz_knot3D; i++)
      {
         fprintf(stream,"%7d %12.8lf %12.8lf %12.8lf\n", i+1, NETZX[i], NETZY[i], NETZZ[i]);
      }

      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_elemente; i++)
         {
            fprintf(stream,"%7d %7d %7d %7d %7d %7d %7d %7d %7d\n", j*anz_elemente+i+1, elliste[4*i]+j*seed+1, elliste[4*i+1]+j*seed+1, elliste[4*i+2]+j*seed+1, elliste[4*i+3]+j*seed+1,
               elliste[4*i]+(j+1)*seed+1, elliste[4*i+1]+(j+1)*seed+1, elliste[4*i+2]+(j+1)*seed+1, elliste[4*i+3]+(j+1)*seed+1);

         }
      }

      fclose(stream);

   }

}


void RECHNE_RB(int anz_schnitte, int anz_elemente, int *ersatz_nr,
int seed, int ase[16][2], int anz_kmark, int *kmark, int anz_wrb_ls,
int *wrb_ls, int anz_elmark, int anz_elmark_einlauf,
int anz_elmark_eli, int anz_elmark_ere, int anz_elmark_11li,
int anz_elmark_10re, int anz_elmark_15li, int anz_elmark_15re,
int anz_elmark_auslauf, int *elmark_einlauf, int *elmark_eli,
int *elmark_ere, int *elmark_11li, int *elmark_10re, int *elmark_15li,
int *elmark_15re, int *elmark_auslauf)

// EINGABE
// anz_schnitte ........ Anzahl der Schnitte zwischen Nabe und Kranz
// anz_elemente ........ Anzahl der (2D) Elemente
// ersatz_nr ........... Zuordnung Knotennummer doppelt - bereinigt
// seed ................ Anzahl der 2D - Knoten (bereinigt)
// ase ................. Anzahl der Knoten am Rand der Gebiete
// anz_kmark ........... Anzahl der Knotenmarkierungen
// anz_wrb_ls .......... Anzahl der Wandelemente an der Leitschaufel
// anz_elmark .......... Gesamte Anzahl der Elementmarkierungen
// anz_elmark_einlauf .. Anzahl mark. Elemente in Einlauf
// anz_elmark_eli ...... Anzahl mark. Elemente Einlauf links
// anz_elmark_ere ...... Anzahl mark. Elemente Einlauf rechts
// anz_elmark_11li ..... Anzahl mark. Elemente Gebiet 11 links
// anz_elmark_10re ..... Anzahl mark. Elemente Gebiet 10 rechts
// anz_elmark_15li ..... Anzahl mark. Elemente Gebiet 15 links
// anz_elmark_15re ..... Anzahl mark. Elemente Gebiet 15 re
// anz_elmark_auslauf .. Anzahl mark. Elemente Auslauf

// AUSGABE
// kmark ............... Knotenmarkierungen
// wrb_ls .............. Knoten der Wandelemente an der Leitschaufel
// elmark_einlauf ...... Knoten der mark. Elemente Einlauf
// elmark_eli .......... Knoten der mark. Elemente Einlauf links
// elmark_ere .......... Knoten der mark. Elemente Einlauf rechts
// elmark_11li ......... Knoten der mark. Elemente Gebiet 11 links
// elmark_10re ......... Knoten der mark. Elemente Gebiet 10 rechts
// elmark_15li ......... Knoten der mark. Elemente Gebiet 15 links
// elmark_15re ......... Knoten der mark. Elemente Gebiet 15 re
// elmark_auslauf ...... Knoten der mark. Elemente Auslauf

{
   (void) anz_schnitte;
   (void) anz_elemente;
   int i=0;
   int pos=0;

   //alle Zaehler auf Null setzen!
   anz_kmark=0;
   anz_wrb_ls=0;
   anz_elmark=0;
   anz_elmark_einlauf=0;
   anz_elmark_eli=0;
   anz_elmark_ere=0;
   anz_elmark_11li=0;
   anz_elmark_10re=0;
   anz_elmark_15li=0;
   anz_elmark_15re=0;
   anz_elmark_auslauf=0;

   //Gebiet 1 versorgen: wrb_ls
   for(i=pos; i<ase[1][0]*ase[1][1]-2*ase[1][1]+1; i+=ase[1][1])
   {
      wrb_ls[4*anz_wrb_ls]=ersatz_nr[i];
      wrb_ls[4*anz_wrb_ls+1]=ersatz_nr[i+ase[1][1]];
      wrb_ls[4*anz_wrb_ls+2]=ersatz_nr[i+ase[1][1]]+seed;
      wrb_ls[4*anz_wrb_ls+3]=ersatz_nr[i]+seed;
      anz_wrb_ls++;
   }
   pos+=ase[1][0]*ase[1][1];

   //Gebiet  2 versorgen: wrb_ls
   for(i=pos; i<pos+ase[2][0]*ase[2][1]-2*ase[2][1]+1; i+=ase[2][1])
   {
      wrb_ls[4*anz_wrb_ls]=ersatz_nr[i];
      wrb_ls[4*anz_wrb_ls+1]=ersatz_nr[i+ase[2][1]];
      wrb_ls[4*anz_wrb_ls+2]=ersatz_nr[i+ase[2][1]]+seed;
      wrb_ls[4*anz_wrb_ls+3]=ersatz_nr[i]+seed;
      anz_wrb_ls++;
   }
   //Gebiet  2 versorgen: elmark_ere
   for(i = pos + (ase[2][0]-1) * ase[2][1]; i < pos + ase[2][1]*ase[2][0]-1; i++)
      //for(i = pos; i < pos+ase[2][0]*ase[2][1]-1; i++)		// alt, falsch??
   {
      elmark_ere[4*anz_elmark_ere]=ersatz_nr[i];
      elmark_ere[4*anz_elmark_ere+1]=ersatz_nr[i+1];
      elmark_ere[4*anz_elmark_ere+2]=ersatz_nr[i+1]+seed;
      elmark_ere[4*anz_elmark_ere+3]=ersatz_nr[i]+seed;
      anz_elmark_ere++;
   }
   pos+=ase[2][0]*ase[2][1];

   //Gebiet  3 versorgen: elmark_eli
   for(i=pos; i<pos+ase[3][1]-1; i++)
   {
      elmark_eli[4*anz_elmark_eli]=ersatz_nr[i];
      elmark_eli[4*anz_elmark_eli+1]=ersatz_nr[i+1];
      elmark_eli[4*anz_elmark_eli+2]=ersatz_nr[i+1]+seed;
      elmark_eli[4*anz_elmark_eli+3]=ersatz_nr[i]+seed;
      anz_elmark_eli++;
   }
   //Gebiet  3 versorgen: wrb_ls
   for(i=pos; i<pos+ase[3][0]*ase[3][1]-2*ase[2][1]+1; i+=ase[3][1])
   {
      wrb_ls[4*anz_wrb_ls]=ersatz_nr[i];
      wrb_ls[4*anz_wrb_ls+1]=ersatz_nr[i+ase[3][1]];
      wrb_ls[4*anz_wrb_ls+2]=ersatz_nr[i+ase[3][1]]+seed;
      wrb_ls[4*anz_wrb_ls+3]=ersatz_nr[i]+seed;
      anz_wrb_ls++;
   }
   pos+=ase[3][0]*ase[3][1];

   //Gebiet  4 versorgen: wrb_ls
   for(i=pos; i<pos+ase[4][0]*ase[4][1]-2*ase[4][1]+1; i+=ase[4][1])
   {
      wrb_ls[4*anz_wrb_ls]=ersatz_nr[i];
      wrb_ls[4*anz_wrb_ls+1]=ersatz_nr[i+ase[4][1]];
      wrb_ls[4*anz_wrb_ls+2]=ersatz_nr[i+ase[4][1]]+seed;
      wrb_ls[4*anz_wrb_ls+3]=ersatz_nr[i]+seed;
      anz_wrb_ls++;
   }
   pos+=ase[4][0]*ase[4][1];

   //Gebiet  5 versorgen: elmark_eli
   for(i=pos; i<pos+ase[5][1]-1; i++)
   {
      elmark_eli[4*anz_elmark_eli]=ersatz_nr[i];
      elmark_eli[4*anz_elmark_eli+1]=ersatz_nr[i+1];
      elmark_eli[4*anz_elmark_eli+2]=ersatz_nr[i+1]+seed;
      elmark_eli[4*anz_elmark_eli+3]=ersatz_nr[i]+seed;
      anz_elmark_eli++;
   }
   //Gebiet 5 versorgen: kmark
   for(i=pos+ase[5][1]-1; i<pos+(ase[5][0]-1)*ase[5][1]; i+=ase[5][1])
   {
      kmark[anz_kmark]=ersatz_nr[i];
      anz_kmark++;
   }
   //Gebiet  5 versorgen:elmark_einlauf
   for(i=pos+ase[5][1]-1; i<pos+(ase[5][0]-1)*ase[5][1]; i+=ase[5][1])
   {
      elmark_einlauf[4*anz_elmark_einlauf]=ersatz_nr[i];
      elmark_einlauf[4*anz_elmark_einlauf+1]=ersatz_nr[i+ase[5][1]];
      elmark_einlauf[4*anz_elmark_einlauf+2]=ersatz_nr[i+ase[5][1]]+seed;
      elmark_einlauf[4*anz_elmark_einlauf+3]=ersatz_nr[i]+seed;
      anz_elmark_einlauf++;
   }
   pos+=ase[5][0]*ase[5][1];

   //Gebiet  6 versorgen: elmark_einlauf
   for(i=pos+ase[6][1]-1; i<pos+(ase[6][0]-1)*ase[6][1]; i+=ase[6][1])
   {
      elmark_einlauf[4*anz_elmark_einlauf]=ersatz_nr[i];
      elmark_einlauf[4*anz_elmark_einlauf+1]=ersatz_nr[i+ase[6][1]];
      elmark_einlauf[4*anz_elmark_einlauf+2]=ersatz_nr[i+ase[6][1]]+seed;
      elmark_einlauf[4*anz_elmark_einlauf+3]=ersatz_nr[i]+seed;
      anz_elmark_einlauf++;
   }
   //Gebiet 6 versorgen: kmark
   for(i=pos+ase[6][1]-1; i<pos+ase[6][0]*ase[6][1]; i+=ase[6][1])
   {
      kmark[anz_kmark]=ersatz_nr[i];
      anz_kmark++;
   }
   //Gebiet  6 versorgen: elmark_ere
   pos+=ase[6][0]*ase[6][1];
   for(i=pos-1; i>pos-ase[6][1]; i--)
   {
      elmark_ere[4*anz_elmark_ere]=ersatz_nr[i];
      elmark_ere[4*anz_elmark_ere+1]=ersatz_nr[i-1];
      elmark_ere[4*anz_elmark_ere+2]=ersatz_nr[i-1]+seed;
      elmark_ere[4*anz_elmark_ere+3]=ersatz_nr[i]+seed;
      anz_elmark_ere++;
   }

   //Gebiet  7 versorgen: elmark_einlauf
   for(i=pos+(ase[7][0]-1)*ase[7][1]; i<pos+ase[7][0]*ase[7][1]-1; i++)
   {
      elmark_einlauf[4*anz_elmark_einlauf]=ersatz_nr[i];
      elmark_einlauf[4*anz_elmark_einlauf+1]=ersatz_nr[i+1];
      elmark_einlauf[4*anz_elmark_einlauf+2]=ersatz_nr[i+1]+seed;
      elmark_einlauf[4*anz_elmark_einlauf+3]=ersatz_nr[i]+seed;
      anz_elmark_einlauf++;
   }

   //Gebiet 7 versorgen: kmark
   for(i=pos+(ase[7][0]-1)*ase[7][1]; i<pos+ase[7][0]*ase[7][1]-1; i++)
   {
      kmark[anz_kmark]=ersatz_nr[i];
      anz_kmark++;
   }
   pos+=ase[7][0]*ase[7][1];
   pos+=ase[8][0]*ase[8][1];
   pos+=ase[9][0]*ase[9][1];

   //Gebiet 10 versorgen: elmark_10re
   for(i=pos+ase[10][1]-1; i<pos+(ase[10][0]-1)*ase[10][1]; i+=ase[10][1])
   {
      elmark_10re[4*anz_elmark_10re]=ersatz_nr[i];
      elmark_10re[4*anz_elmark_10re+1]=ersatz_nr[i+ase[10][1]];
      elmark_10re[4*anz_elmark_10re+2]=ersatz_nr[i+ase[10][1]]+seed;
      elmark_10re[4*anz_elmark_10re+3]=ersatz_nr[i]+seed;
      anz_elmark_10re++;
   }
   pos+=ase[10][0]*ase[10][1];

   //Gebiet 11 versorgen: elmark_11li
   for(i=pos; i<pos+(ase[11][0]-1)*ase[11][1]; i+=ase[11][1])
   {
      elmark_11li[4*anz_elmark_11li]=ersatz_nr[i];
      elmark_11li[4*anz_elmark_11li+1]=ersatz_nr[i+ase[11][1]];
      elmark_11li[4*anz_elmark_11li+2]=ersatz_nr[i+ase[11][1]]+seed;
      elmark_11li[4*anz_elmark_11li+3]=ersatz_nr[i]+seed;
      anz_elmark_11li++;
   }
   pos+=ase[11][0]*ase[11][1];
   pos+=ase[12][0]*ase[12][1];
   pos+=ase[13][0]*ase[13][1];
   pos+=ase[14][0]*ase[14][1];

   //Gebiet 15 versorgen: elmark_15li
   for(i=pos; i<pos+ase[15][1]-1; i++)
   {
      elmark_15li[4*anz_elmark_15li]=ersatz_nr[i];
      elmark_15li[4*anz_elmark_15li+1]=ersatz_nr[i+1];
      elmark_15li[4*anz_elmark_15li+2]=ersatz_nr[i+1]+seed;
      elmark_15li[4*anz_elmark_15li+3]=ersatz_nr[i]+seed;
      anz_elmark_15li++;
   }

   //Gebiet 15 versorgen: elmark_auslauf
   for(i=pos+ase[15][1]-1; i<pos+(ase[15][0])*ase[15][1]-1; i+=ase[15][1])
   {
      elmark_auslauf[4*anz_elmark_auslauf]=ersatz_nr[i];
      elmark_auslauf[4*anz_elmark_auslauf+1]=ersatz_nr[i+ase[15][1]];
      elmark_auslauf[4*anz_elmark_auslauf+2]=ersatz_nr[i+ase[15][1]]+seed;
      elmark_auslauf[4*anz_elmark_auslauf+3]=ersatz_nr[i]+seed;
      anz_elmark_auslauf++;
   }

   //Gebiet 15 versorgen: elmark_15re
   for(i=pos+(ase[15][1])*(ase[15][0]-1); i<pos+ase[15][0]*ase[15][1]-1; i++)
   {
      //printf("%d\n", ersatz_nr[i]);
      elmark_15re[4*anz_elmark_15re]=ersatz_nr[i];
      elmark_15re[4*anz_elmark_15re+1]=ersatz_nr[i+1];
      elmark_15re[4*anz_elmark_15re+2]=ersatz_nr[i+1]+seed;
      elmark_15re[4*anz_elmark_15re+3]=ersatz_nr[i]+seed;
      anz_elmark_15re++;
   }
   pos+=ase[15][0]*ase[15][1];

   /*
   FILE *stream;
   char datei_steuer[200];
   strcpy(datei_steuer, "check_elmark.dat");

   if( (stream = fopen( &datei_steuer[0], "a" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
   }
   else
   {
   fprintf(stream, "IN RECHNE_RB\n\n");
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

}


// -------------------------------------------------------
// AUSGABE RB_FILE
// -------------------------------------------------------

void AUSGABE_3D_RB(char *rb_pfad, int anz_schnitte, int seed, int anz_wrb_ls, int *wrb_ls,
int anz_elemente, int *el_liste, int anz_grenz, int anz_elmark_einlauf,
int anz_elmark_eli,  int anz_elmark_ere,  int anz_elmark_11li,
int anz_elmark_10re, int anz_elmark_15li, int anz_elmark_15re,
int anz_elmark_auslauf, int *elmark_einlauf, int *elmark_eli,
int *elmark_ere, int *elmark_11li, int *elmark_10re,
int *elmark_15li, int *elmark_15re, int *elmark_auslauf,
int anz_kmark, int *kmark, int ase[16][2], int start3,
int start5, int end6, int end7, int start10, int start11, int start15, double *p2)

// EINGABE
// siehe oben

// AUSGABE
// RB-FILE leit.RB

{
   FILE *stream;
   char datei_steuer[200];

   int anz_wandrb, anz_elmark, anz_druckrb, i, j;

   strcpy(datei_steuer, rb_pfad);
   if( (stream = fopen( &datei_steuer[0], "w" )) == NULL )
   {
      printf( "Kann  in '%s' nicht schreiben!\n", datei_steuer);
   }

   else
   {
      for (i=0; i<10; i++)
      {
         fprintf(stream,"C\n");                   //zehn Kommentarzeilen
      }

      anz_wandrb = 2*anz_elemente + (anz_schnitte-1)*anz_wrb_ls;
      anz_elmark = (anz_schnitte-1)*(anz_wrb_ls+anz_elmark_einlauf+anz_elmark_eli+anz_elmark_ere+anz_elmark_11li
         +anz_elmark_10re+anz_elmark_15li+anz_elmark_15re+anz_elmark_auslauf);
      anz_druckrb = anz_elmark_auslauf * (anz_schnitte-1);

      fprintf(stream,"%6d %6d %d %d %d %d %6d %6d\n", 0, anz_wandrb, anz_druckrb, 0,
         0, 0, anz_elmark, (anz_schnitte*anz_kmark));

      //Wände: Nabe: OK
      j=0;
      {
         for (i=0; i<anz_elemente; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %d %d %d %6d\n", el_liste[4*i]+j*seed+1, el_liste[4*i+1]+j*seed+1,
               el_liste[4*i+2]+j*seed+1, el_liste[4*i+3]+j*seed+1, 0, 0, 0, i+1);
         }
      }
      //Wände: Kranz: OK
      j=anz_schnitte-1;
      {
         for (i=0; i<anz_elemente; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %d %d %d %6d\n", el_liste[4*i]+j*seed+1, el_liste[4*i+1]+j*seed+1,
               el_liste[4*i+2]+j*seed+1, el_liste[4*i+3]+j*seed+1, 0, 0, 0, anz_elemente*(anz_schnitte-2)+i+1);
         }
      }

      //Wände: Leitschaufel: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_wrb_ls; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %d %d %d %6d\n", wrb_ls[4*i]+j*seed+1, wrb_ls[4*i+1]+j*seed+1,
               wrb_ls[4*i+2]+j*seed+1, wrb_ls[4*i+3]+j*seed+1, 0, 0, 0, j*anz_elemente+(anz_grenz-1)*i+1);
         }
      }

      //Druckrandbedingungen, aus Elementmarkierungen Auslauf abgeleitet Markierung 77
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_elmark_auslauf; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %10.2lf %6d\n", elmark_auslauf[4*i]+j*seed+1, elmark_auslauf[4*i+1]+j*seed+1,
               elmark_auslauf[4*i+2]+j*seed+1, elmark_auslauf[4*i+3]+j*seed+1, 77,
               p2[j],
               j*anz_elemente+start15+ase[15][1]-1+(ase[15][1]-1)*i);
         }
      }

      //Elementmarkierungen Leitschaufel: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_wrb_ls; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", wrb_ls[4*i]+j*seed+1, wrb_ls[4*i+1]+j*seed+1,
               wrb_ls[4*i+2]+j*seed+1, wrb_ls[4*i+3]+j*seed+1,
               j*anz_elemente+(anz_grenz-1)*i+1, 10);
         }
      }

      //Elementmarkierungen Einlauf Gebiet 5+6: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<ase[5][0]+ase[6][0]-2; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_einlauf[4*i]+j*seed+1, elmark_einlauf[4*i+1]+j*seed+1,
               elmark_einlauf[4*i+2]+j*seed+1, elmark_einlauf[4*i+3]+j*seed+1,
               j*anz_elemente+start5+(ase[5][1]-1)*(i+1), 100);
         }
      }

      //Elementmarkierungen Einlauf Gebiet 7: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=ase[5][0]+ase[6][0]-2; i<anz_elmark_einlauf; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_einlauf[4*i]+j*seed+1,
               elmark_einlauf[4*i+1]+j*seed+1,  elmark_einlauf[4*i+2]+j*seed+1,
               elmark_einlauf[4*i+3]+j*seed+1,
            //in dieser Zeile war ein Fehler drin, der jetzt hoffentlich weg ist!
               j*anz_elemente + end7 - ase[7][1] - ase[5][0] - ase[6][0] +i +5, 100);
         }
      }

      //Elementmarkierungen eli Gebiet 3: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<ase[3][1]-1; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_eli[4*i]+j*seed+1, elmark_eli[4*i+1]+j*seed+1,
               elmark_eli[4*i+2]+j*seed+1, elmark_eli[4*i+3]+j*seed+1,
               j*anz_elemente+start3+i+1, 150);
         }
      }

      //Elementmarkierungen eli Gebiet 5: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=ase[3][1]-1; i<anz_elmark_eli; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_eli[4*i]+j*seed+1, elmark_eli[4*i+1]+j*seed+1,
               elmark_eli[4*i+2]+j*seed+1, elmark_eli[4*i+3]+j*seed+1,
               j*anz_elemente+start5+i-ase[3][1]+2, 150);
         }
      }

      //Elementmarkierungen ere Gebiet 2: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<ase[2][1]-1; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_ere[4*i]+j*seed+1, elmark_ere[4*i+1]+j*seed+1,
               elmark_ere[4*i+2]+j*seed+1, elmark_ere[4*i+3]+j*seed+1,
               j*anz_elemente+start3-ase[2][1]+i+2, 160);
         }
      }

      //Elementmarkierungen ere Gebiet 6: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=ase[2][1]-1; i<anz_elmark_ere; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_ere[4*i]+j*seed+1, elmark_ere[4*i+1]+j*seed+1,
               elmark_ere[4*i+2]+j*seed+1, elmark_ere[4*i+3]+j*seed+1,
               j*anz_elemente+end6-i+ase[2][1]-1, 160);
         }
      }

      //Elementmarkierungen 11li: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_elmark_11li; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_11li[4*i]+j*seed+1, elmark_11li[4*i+1]+j*seed+1,
               elmark_11li[4*i+2]+j*seed+1, elmark_11li[4*i+3]+j*seed+1,
               j*anz_elemente+start11+(ase[11][1]-1)*i+1, 150);
         }
      }

      //Elementmarkierungen 10re: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_elmark_10re; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_10re[4*i]+j*seed+1, elmark_10re[4*i+1]+j*seed+1,
               elmark_10re[4*i+2]+j*seed+1, elmark_10re[4*i+3]+j*seed+1,
               j*anz_elemente+start10+ase[10][1]-2+(ase[10][1]-1)*i+1, 160);
         }
      }

      //Elementmarkierungen 15li: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_elmark_15li; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_15li[4*i]+j*seed+1, elmark_15li[4*i+1]+j*seed+1,
               elmark_15li[4*i+2]+j*seed+1, elmark_15li[4*i+3]+j*seed+1,
               j*anz_elemente+start15+i+1, 150);
         }
      }

      //Elementmarkierungen Auslauf: OK
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_elmark_auslauf; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_auslauf[4*i]+j*seed+1, elmark_auslauf[4*i+1]+j*seed+1,
               elmark_auslauf[4*i+2]+j*seed+1, elmark_auslauf[4*i+3]+j*seed+1,
               j*anz_elemente+start15+ase[15][1]-1+(ase[15][1]-1)*i, 110);
         }
      }

      //Elementmarkierungen 15re
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_elmark_15re; i++)
         {
            fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_15re[4*i]+j*seed+1, elmark_15re[4*i+1]+j*seed+1,
               elmark_15re[4*i+2]+j*seed+1, elmark_15re[4*i+3]+j*seed+1,
               j*anz_elemente+start15+(ase[15][1]-1)*(ase[15][0]-2)+i+1, 160);
         }
      }

      //Knotenmarkierungen
      for (j=0; j<anz_schnitte; j++)
      {
         for (i=0; i<anz_kmark; i++)
         {
            fprintf(stream,"%6d %6d\n", kmark[i]+j*seed+1, 7);
         }
      }

      fclose(stream);
   }

}


// -------------------------------------------------------
// ERSTELLEN DER 3D-RB-ARRAYS FUER COVISE / FENFLOSS
// (created from AUSGABE_3D_RB)
// -------------------------------------------------------

void RECHNE_3D_RB(struct ggrid *gg, int anz_schnitte, int seed, int anz_wrb_ls, int *wrb_ls,
int anz_elemente, int *el_liste, int anz_grenz, int anz_elmark_einlauf,
int anz_elmark_eli,  int anz_elmark_ere,  int anz_elmark_11li,
int anz_elmark_10re, int anz_elmark_15li, int anz_elmark_15re,
int anz_elmark_auslauf, int *elmark_einlauf, int *elmark_eli,
int *elmark_ere, int *elmark_11li, int *elmark_10re,
int *elmark_15li, int *elmark_15re, int *elmark_auslauf,
int anz_kmark, int *kmark, int ase[16][2], int start3,
int start5, int end6, int end7, int start10, int start11, int start15, double *p2)

// EINGABE
// siehe oben

// AUSGABE
// Datenstruktur fuer Covise / Fenfloss mit den Randbedingungen

{
   (void) anz_kmark;
   (void) kmark;
   //int anz_wandrb, anz_elmark, anz_druckrb
   int i, j;
   int n_per, n_in;
   /*		*** not needed for covise-version ***
      int anz_wandrb, anz_elmark, anz_druckrb;
      anz_wandrb = 2*anz_elemente + (anz_schnitte-1)*anz_wrb_ls;
      anz_elmark = (anz_schnitte-1)*(anz_wrb_ls+anz_elmark_einlauf+anz_elmark_eli+anz_elmark_ere+anz_elmark_11li
                              +anz_elmark_10re+anz_elmark_15li+anz_elmark_15re+anz_elmark_auslauf);
      anz_druckrb = anz_elmark_auslauf * (anz_schnitte-1);
   */

   /* old header line
      fprintf(stream,"%6d %6d %d %d %d %d %6d %6d\n", 0, anz_wandrb, anz_druckrb, 0,
      0, 0, anz_elmark, (anz_schnitte*anz_kmark));
   */

   // *******************************************
   // all values in C-notation!!!!!

   //Wände: Nabe: OK
   j=0;
   {
      for (i=0; i<anz_elemente; i++)
      {
         // polygon list
         Add2Ilist(gg->bcwallpol, 4*i);

         // 0-3: corner list, 4: referring 3D-element
         Add2Ilist(gg->bcwall, el_liste[4*i]+j*seed);
         Add2Ilist(gg->bcwall, el_liste[4*i+1]+j*seed);
         Add2Ilist(gg->bcwall, el_liste[4*i+2]+j*seed);
         Add2Ilist(gg->bcwall, el_liste[4*i+3]+j*seed);

         Add2Ilist(gg->bcwallvol, i+1      -1);

         //fprintf(stream,"%6d %6d %6d %6d %d %d %d %6d\n", el_liste[4*i]+j*seed+1, el_liste[4*i+1]+j*seed+1,
         //el_liste[4*i+2]+j*seed+1, el_liste[4*i+3]+j*seed+1, 0, 0, 0, i+1);
      }
   }

   //Wände: Kranz: OK
   j=anz_schnitte-1;
   {
      for (i=0; i<anz_elemente; i++)
      {
         // polygon list
         Add2Ilist(gg->bcwallpol, 4*anz_elemente + 4*i);

         // 0-3: corner list, 4: referring 3D-element
         Add2Ilist(gg->bcwall, el_liste[4*i]+j*seed);
         Add2Ilist(gg->bcwall, el_liste[4*i+1]+j*seed);
         Add2Ilist(gg->bcwall, el_liste[4*i+2]+j*seed);
         Add2Ilist(gg->bcwall, el_liste[4*i+3]+j*seed);

         Add2Ilist(gg->bcwallvol, anz_elemente*(anz_schnitte-2)+i+1      -1);

         //fprintf(stream,"%6d %6d %6d %6d %d %d %d %6d\n", el_liste[4*i]+j*seed+1, el_liste[4*i+1]+j*seed+1,
         //el_liste[4*i+2]+j*seed+1, el_liste[4*i+3]+j*seed+1, 0, 0, 0, anz_elemente*(anz_schnitte-2)+i+1);
      }
   }

   //Wände: Leitschaufel: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=0; i<anz_wrb_ls; i++)
      {
         // polygon list
         Add2Ilist(gg->bcwallpol, 8*anz_elemente + 4*j*anz_wrb_ls + 4*i);

         // corner list
         Add2Ilist(gg->bcwall, j*seed + wrb_ls[4*i+0]);
         Add2Ilist(gg->bcwall, j*seed + wrb_ls[4*i+1]);
         Add2Ilist(gg->bcwall, j*seed + wrb_ls[4*i+2]);
         Add2Ilist(gg->bcwall, j*seed + wrb_ls[4*i+3]);

         Add2Ilist(gg->bcwallvol, j*anz_elemente+(anz_grenz-1)*i+1      -1);

         //fprintf(stream,"%6d %6d %6d %6d %d %d %d %6d\n", wrb_ls[4*i]+j*seed+1, wrb_ls[4*i+1]+j*seed+1,
         //wrb_ls[4*i+2]+j*seed+1, wrb_ls[4*i+3]+j*seed+1, 0, 0, 0, j*anz_elemente+(anz_grenz-1)*i+1);
      }
   }

   /*
      //Elementmarkierungen Leitschaufel: not implemented so far!
      for (j=0; j<anz_schnitte-1; j++)
      {
         for (i=0; i<anz_wrb_ls; i++)
         {
            // polygon list
            Add2Ilist(gg->xxxpol, 4*i);

            // corner list
            Add2Ilist(gg->xxx, wrb_ls[4*i]+j*seed);
   Add2Ilist(gg->xxx, wrb_ls[4*i+1]+j*seed);
   Add2Ilist(gg->xxx, wrb_ls[4*i+2]+j*seed);
   Add2Ilist(gg->xxx, wrb_ls[4*i+3]+j*seed);

   Add2Ilist(gg->xxxvol, j*anz_elemente+(anz_grenz-1)*i+1      -1);

   //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", wrb_ls[4*i]+j*seed+1, wrb_ls[4*i+1]+j*seed+1,
   //wrb_ls[4*i+2]+j*seed+1, wrb_ls[4*i+3]+j*seed+1,
   //j*anz_elemente+(anz_grenz-1)*i+1, 10);
   }
   }
   */

   n_in = 0;

   //Elementmarkierungen Einlauf Gebiet 5+6: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=0; i<ase[5][0]+ase[6][0]-2; i++)
      {
         // polygon list
         Add2Ilist(gg->bcinpol, 4*n_in);
         n_in++;

         // corner list
         Add2Ilist(gg->bcin, elmark_einlauf[4*i]+j*seed);
         Add2Ilist(gg->bcin, elmark_einlauf[4*i+1]+j*seed);
         Add2Ilist(gg->bcin, elmark_einlauf[4*i+2]+j*seed);
         Add2Ilist(gg->bcin, elmark_einlauf[4*i+3]+j*seed);

         Add2Ilist(gg->bcinvol, j*anz_elemente+start5+(ase[5][1]-1)*(i+1)      -1);

         // fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_einlauf[4*i]+j*seed+1, elmark_einlauf[4*i+1]+j*seed+1,
         // elmark_einlauf[4*i+2]+j*seed+1, elmark_einlauf[4*i+3]+j*seed+1,
         // j*anz_elemente+start5+(ase[5][1]-1)*(i+1), 100);
      }
   }

   //Elementmarkierungen Einlauf Gebiet 7: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=ase[5][0]+ase[6][0]-2; i<anz_elmark_einlauf; i++)
      {
         // polygon list
         Add2Ilist(gg->bcinpol, 4*n_in);
         n_in++;

         // corner list
         Add2Ilist(gg->bcin, elmark_einlauf[4*i]+j*seed);
         Add2Ilist(gg->bcin, elmark_einlauf[4*i+1]+j*seed);
         Add2Ilist(gg->bcin, elmark_einlauf[4*i+2]+j*seed);
         Add2Ilist(gg->bcin, elmark_einlauf[4*i+3]+j*seed);

         Add2Ilist(gg->bcinvol, j*anz_elemente + end7 - ase[7][1] - ase[5][0] - ase[6][0] +i +5      -1);

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_einlauf[4*i]+j*seed+1,
         //elmark_einlauf[4*i+1]+j*seed+1,	elmark_einlauf[4*i+2]+j*seed+1,
         //elmark_einlauf[4*i+3]+j*seed+1,
         //j*anz_elemente + end7 - ase[7][1] - ase[5][0] - ase[6][0] +i +5, 100);
      }
   }

   n_per = 0;
   //Elementmarkierungen eli Gebiet 3: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=0; i<ase[3][1]-1; i++)
      {
         // polygon list
         Add2Ilist(gg->bcperiodicpol, 4*n_per);
         n_per++;

         // corner list
         Add2Ilist(gg->bcperiodic, elmark_eli[4*i]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_eli[4*i+1]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_eli[4*i+2]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_eli[4*i+3]+j*seed);

         Add2Ilist(gg->bcperiodicvol, j*anz_elemente+start3+i+1      -1);

         Add2Ilist(gg->bcperiodicval, gg->bc_periodic_left);

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_eli[4*i]+j*seed+1, elmark_eli[4*i+1]+j*seed+1,
         //elmark_eli[4*i+2]+j*seed+1, elmark_eli[4*i+3]+j*seed+1,
         //j*anz_elemente+start3+i+1, 150);
      }
   }

   //Elementmarkierungen eli Gebiet 5: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=ase[3][1]-1; i<anz_elmark_eli; i++)
      {
         // polygon list
         Add2Ilist(gg->bcperiodicpol, 4*n_per);
         n_per++;

         // corner list
         Add2Ilist(gg->bcperiodic, elmark_eli[4*i]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_eli[4*i+1]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_eli[4*i+2]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_eli[4*i+3]+j*seed);

         Add2Ilist(gg->bcperiodicvol, j*anz_elemente+start5+i-ase[3][1]+2      -1);

         Add2Ilist(gg->bcperiodicval, gg->bc_periodic_left);

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_eli[4*i]+j*seed+1, elmark_eli[4*i+1]+j*seed+1,
         //elmark_eli[4*i+2]+j*seed+1, elmark_eli[4*i+3]+j*seed+1,
         //j*anz_elemente+start5+i-ase[3][1]+2, 150);
      }
   }

   //Elementmarkierungen ere Gebiet 2: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=0; i<ase[2][1]-1; i++)
      {
         // polygon list
         Add2Ilist(gg->bcperiodicpol, 4*n_per);
         n_per++;

         // corner list
         Add2Ilist(gg->bcperiodic, elmark_ere[4*i]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_ere[4*i+1]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_ere[4*i+2]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_ere[4*i+3]+j*seed);

         Add2Ilist(gg->bcperiodicvol, j*anz_elemente+start3-ase[2][1]+i+2      -1);

         Add2Ilist(gg->bcperiodicval, gg->bc_periodic_right);

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_ere[4*i]+j*seed+1, elmark_ere[4*i+1]+j*seed+1,
         //elmark_ere[4*i+2]+j*seed+1, elmark_ere[4*i+3]+j*seed+1,
         //j*anz_elemente+start3-ase[2][1]+i+2, 160);
      }
   }

   //Elementmarkierungen ere Gebiet 6: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=ase[2][1]-1; i<anz_elmark_ere; i++)
      {
         // polygon list
         Add2Ilist(gg->bcperiodicpol, 4*n_per);
         n_per++;

         // corner list
         Add2Ilist(gg->bcperiodic, elmark_ere[4*i]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_ere[4*i+1]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_ere[4*i+2]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_ere[4*i+3]+j*seed);

         Add2Ilist(gg->bcperiodicvol, j*anz_elemente+end6-i+ase[2][1]-1      -1);

         Add2Ilist(gg->bcperiodicval, gg->bc_periodic_right);

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_ere[4*i]+j*seed+1, elmark_ere[4*i+1]+j*seed+1,
         //elmark_ere[4*i+2]+j*seed+1, elmark_ere[4*i+3]+j*seed+1,
         //j*anz_elemente+end6-i+ase[2][1]-1, 160);
      }
   }

   //Elementmarkierungen 11li: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=0; i<anz_elmark_11li; i++)
      {
         // polygon list
         Add2Ilist(gg->bcperiodicpol, 4*n_per);
         n_per++;

         // corner list
         Add2Ilist(gg->bcperiodic, elmark_11li[4*i]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_11li[4*i+1]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_11li[4*i+2]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_11li[4*i+3]+j*seed);

         Add2Ilist(gg->bcperiodicvol, j*anz_elemente+start11+(ase[11][1]-1)*i+1      -1);

         Add2Ilist(gg->bcperiodicval, gg->bc_periodic_left);

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_11li[4*i]+j*seed+1, elmark_11li[4*i+1]+j*seed+1,
         //elmark_11li[4*i+2]+j*seed+1, elmark_11li[4*i+3]+j*seed+1,
         //j*anz_elemente+start11+(ase[11][1]-1)*i+1, 150);
      }
   }

   //Elementmarkierungen 10re: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=0; i<anz_elmark_10re; i++)
      {
         // polygon list
         Add2Ilist(gg->bcperiodicpol, 4*n_per);
         n_per++;

         // corner list
         Add2Ilist(gg->bcperiodic, elmark_10re[4*i]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_10re[4*i+1]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_10re[4*i+2]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_10re[4*i+3]+j*seed);

         Add2Ilist(gg->bcperiodicvol, j*anz_elemente+start10+ase[10][1]-2+(ase[10][1]-1)*i+1      -1);

         Add2Ilist(gg->bcperiodicval, gg->bc_periodic_right);

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_10re[4*i]+j*seed+1, elmark_10re[4*i+1]+j*seed+1,
         //elmark_10re[4*i+2]+j*seed+1, elmark_10re[4*i+3]+j*seed+1,
         //j*anz_elemente+start10+ase[10][1]-2+(ase[10][1]-1)*i+1, 160);
      }
   }

   //Elementmarkierungen 15li: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=0; i<anz_elmark_15li; i++)
      {
         // polygon list
         Add2Ilist(gg->bcperiodicpol, 4*n_per);
         n_per++;

         // corner list
         Add2Ilist(gg->bcperiodic, elmark_15li[4*i]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_15li[4*i+1]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_15li[4*i+2]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_15li[4*i+3]+j*seed);

         Add2Ilist(gg->bcperiodicvol, j*anz_elemente+start15+i+1      -1);

         Add2Ilist(gg->bcperiodicval, gg->bc_periodic_left);

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_15li[4*i]+j*seed+1, elmark_15li[4*i+1]+j*seed+1,
         //elmark_15li[4*i+2]+j*seed+1, elmark_15li[4*i+3]+j*seed+1,
         //j*anz_elemente+start15+i+1, 150);
      }
   }

   //Elementmarkierungen Auslauf: OK
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=0; i<anz_elmark_auslauf; i++)
      {
         // polygon list
         Add2Ilist(gg->bcoutpol, 4*j*anz_elmark_auslauf + 4*i);

         // corner list
         Add2Ilist(gg->bcout, elmark_auslauf[4*i]+j*seed);
         Add2Ilist(gg->bcout, elmark_auslauf[4*i+1]+j*seed);
         Add2Ilist(gg->bcout, elmark_auslauf[4*i+2]+j*seed);
         Add2Ilist(gg->bcout, elmark_auslauf[4*i+3]+j*seed);

         Add2Ilist(gg->bcoutvol, j*anz_elemente+start15+ase[15][1]-1+(ase[15][1]-1)*i      -1);

         // pressure boundary condition
         Add2Flist(gg->bcpressval, float(p2[j]));

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_auslauf[4*i]+j*seed+1, elmark_auslauf[4*i+1]+j*seed+1,
         //elmark_auslauf[4*i+2]+j*seed+1, elmark_auslauf[4*i+3]+j*seed+1,
         //j*anz_elemente+start15+ase[15][1]-1+(ase[15][1]-1)*i, 110);
      }
   }

   //Elementmarkierungen 15re
   for (j=0; j<anz_schnitte-1; j++)
   {
      for (i=0; i<anz_elmark_15re; i++)
      {
         // polygon list
         Add2Ilist(gg->bcperiodicpol, 4*n_per);
         n_per++;

         // corner list
         Add2Ilist(gg->bcperiodic, elmark_15re[4*i]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_15re[4*i+1]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_15re[4*i+2]+j*seed);
         Add2Ilist(gg->bcperiodic, elmark_15re[4*i+3]+j*seed);

         Add2Ilist(gg->bcperiodicvol, j*anz_elemente+start15+(ase[15][1]-1)*(ase[15][0]-2)+i+1      -1);

         Add2Ilist(gg->bcperiodicval, gg->bc_periodic_right);

         //fprintf(stream,"%6d %6d %6d %6d %6d %6d\n", elmark_15re[4*i]+j*seed+1, elmark_15re[4*i+1]+j*seed+1,
         //elmark_15re[4*i+2]+j*seed+1, elmark_15re[4*i+3]+j*seed+1,
         //j*anz_elemente+start15+(ase[15][1]-1)*(ase[15][0]-2)+i+1, 160);
      }
   }

   /*
      //Knotenmarkierungen
      for (j=0; j<anz_schnitte; j++)
      {
         for (i=0; i<anz_kmark; i++)
         {
            fprintf(stream,"%6d %6d\n", kmark[i]+j*seed, 7);
         }
      }
   */

}
