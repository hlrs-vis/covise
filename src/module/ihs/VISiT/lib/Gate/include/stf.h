struct stf
{

   // Pfade
   char *leitxyz_pfad;                            // Pfad der Datei mit den Lietschaufelprofilen
   char *nabekranz_pfad;                          // Pfad der Datei mit der Meridiankontur
   char *geo_pfad;                                // Pfad des Geo-Files (Ausgabe)
   char *rb_pfad;                                 // Pfad des RB-Files (Ausgabe)

   double Q;                                      // Durchfluss in m3/s

   int anz_ls;                                    // Anzahl der Leitschaufeln

   // Grenzpunkte Profil
   int PG2;                                       // Grenzpunkt auf Leitschaufelprofil Gebiet1 - Gebiet 2 (insgesamt 200 Punkte)
   int PG3;                                       // Grenzpunkt auf Leitschaufelprofil Gebiet2 - Gebiet 3 (insgesamt 200 Punkte)
   int PG4;                                       // Grenzpunkt auf Leitschaufelprofil Gebiet3 - Gebiet 4 (insgesamt 200 Punkte)

   double grenz;                                  // Grenzschichtdicke

   double deltagamma;                             // Drehwinkel der Leitschaufel

   // Knotenanzahl
   int anz_schnitte;                              // Schnitte zwischen Nabe und Kranz
   int anz_grenz;                                 // Grenzschicht
   int anz_15;                                    // Austritt
   int anz_einlauf;                               // Einlauf
   int anz_knot1;                                 // Gebiet 1, entlang Schaufel
   int anz_knot2;                                 // Gebiet 2, entlang Schaufel
   int anz_knot3;                                 // Gebiet 3, entlang Schaufel
   int anz_knot4;                                 // Gebiet 4, entlang Schaufel

   // Bogenlaengen
   // die Bogenlaenge am Staupunkt wird als 0 definiert
   // besser waere gewesen: Bogenlaenge am Drehpunkt = 0!!
   double bgl_15_n;                               // Bogenlaenge Gebiet 15 oben Nabe
   double bgl_15_k;                               // Bogenlaenge Gebiet 15 oben Kranz
   double bgl_max_nabe;                           // Bogenlaenge Austritt (Nabe)
   double bgl_start;                              // Bogenlaenge Eintritt (Vorlauf)
   double bgl_aus;                                // Bogenlaenge Austritt (Nachlauf)

   // Verdichtungen laengs
   double lvs1;                                   // Gebiet 1
   double lvs2;                                   // Gebiet 2
   double lvs3;                                   // Gebiet 3
   double lvs4;                                   // Gebiet 4
   double lvs5;                                   // Gebiet 5
   double lvs6;                                   // Gebiet 9 rechts, 10 rechts, 11 links
   double lvs7;                                   // Gebiete 5-7 Eintritt

   // Verdichtungen quer
   double bvs1;                                   // Gebiet 1
   double bvs2;                                   // Gebiet 2
   double bvs3;                                   // Gebiet 3
   double bvs4;                                   // Gebiet 4
   double bvs8;                                   // Gebiete 8 und 9

   // Verdichtung radial
   double verd_radial;                            // Verschiebung zwischen Nabe und Kranz

   // Verschiebungen
   double versch_eintr;                           // Verschiebung am Eintritt (<1!)
   double versch_austr;                           // Verschiebung am Austritt (kann <0 oder >0 sein)

   //abhaengige Netzparameter
   int anz_mitte;
   int anz_breit;
   int anz_knot9;
   int PG;
   int PG1;
   int PG5;

};

// in stf.c

int EINLESEN_STEUERFILE(char *datei_steuer, struct stf *stf_para);

void DumpSTF(struct stf *stf_para);

int read_string(char *buf, char* &str, const char *separator);
int read_int(char *buf, int *zahl, const char *separator);
int read_double(char *buf, double *zahl, const char *separator);
