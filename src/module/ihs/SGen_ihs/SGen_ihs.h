#ifndef _READIHS_H
#define _READIHS_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Ihs data         	                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  17.03.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>

#ifdef _MSC_VER
#define io_ IO
#define fona_ FONA
#define umle_ UMLE
#define maxw_ MAXW
#define wust_ WUST
#define prof_ PROF
#define dick_ DICK
#define kont_ KONT
#define zwis_ ZWIS
#define absol_ ABSOL
#define anet_ ANET
#define axel_ AXEL
#define akno_ AKNO
#define leits_ LEITS
#define leitg_ LEITG
#define fokr_ FOKR
#define wink_ WINK
#define xyz_ XYZ
#define lauf_ LAUF
#define abwik_ ABWIK
#define geogen_ GEOGEN
#define axnet_ AXNET
#define netzoeffnen_ NETZOEFFNEN
#define eingabe_xyz_ EINGABE_XYZ
#define schreibe_randbedingung_ SCHREIBE_RANDBEDINGUNG
#define rechne_kugel_ RECHNE_KUGEL
#define ainf_ AINF
#define ver2_ VER2
#define verz_ VERZ
#define lesn_ LESN
#endif
#ifndef WIN32
#include <unistd.h>
#endif
/*
#ifdef CO_gcc3
#define schreibe_randbedingung_ schreibe_randbedingung__
#define rechne_kugel_ rechne_kugel__
#define eingabe_xyz_ eingabe_xyz__
#endif
*/
extern "C"
{
   void geogen_();
   void axnet_();
   void netzoeffnen_();
   void eingabe_xyz_();
   void schreibe_randbedingung_();
   void rechne_kugel_();
   /* --- Variablendefinitionen fuer die Uebergabe des FORTRAN-Programms --- */
   extern struct netz
   {
      float x_ds[11][19];
      float y_ds[11][19];
      float z_ds[11][19];
      float x_ss[11][19];
      float y_ss[11][19];
      float z_ss[11][19];
      float x_sl[11][19];
      float y_sl[11][19];
      float z_sl[11][19];
   }xyz_;

   extern struct abwik
   {
      float x_sl_ab[11][19];
      float y_sl_ab[11][19];
      float x_ds_ab[11][19];
      float y_ds_ab[11][19];
      float x_ss_ab[11][19];
      float y_ss_ab[11][19];
   }abwik_;

   extern struct nenn
   {
      float Q;
      float H;
      float N;
   }absol_;

   extern struct eina
   {
      int lese;
      int schreibe;
      int fehler;
      int WRSPEZIFISCH;
      int WRBETRIEBSPUNKT;
      int WRMEKO;
      int WROUTPUT;
      int WRPROFILE_NEU;
      int WRZWISCHEN_FILE;
      int WRLA_WINKEL;
      int WRSTROE_WINKEL;
      int WRGE_DREIECKE;
      int WRZWISCHEN_PROF;
      int WRSCHAUFEL_XYZ;
      int WRBATREKHY;
      int WRPROTOTYP;
      int success;
   }io_;

   extern struct zwis
   {
      float d2;
      float di_da;
      long int   nlschaufel;
   }zwis_;

   extern struct leitg
   {
      int   Nleit;
      float b0;
      float d0;
      float l0;
      float leit_dr;
   }leitg_;

   extern struct nab
   {
      int anznabe;
      float r_nabe[30];
      float z_nabe[30];
      float d2_kugel;
   }fona_;

   extern struct kra
   {
      int anzkranz;
      float r_kranz[30];
      float z_kranz[30];
   }fokr_;

   extern struct wust
   {
      float db2i;
      float db2a;
      float db1i;
      float db1a;
   }wust_;

   extern struct maxw
   {
      float mwi;
      float mwa;
   }maxw_;

   extern struct kont
   {
      float umschlingung;
      float max1;
      float max2;
      float pe1i;
      float pe1a;
      float pe2i;
      float pe2a;
   }kont_;

   extern struct lauf
   {
      float d_1[11];
      float d_2[11];
   }lauf_;

   extern struct wink
   {
      float beta1_sch[11];
      float beta2_sch[11];
   }wink_;

   extern struct verz
   {
      char datei_prototyp     [200];
      char datei_zentral_file [200];
      char datei_daten        [200];
      char datei_dicke        [200];
      char datei_profil       [200];
      char datei_steuerfile   [200];
      char datei_profil_neu   [200];
      char datei_output       [200];
      char datei_zwischen_file[200];
      char datei_meko         [200];
      char datei_spezifisch   [200];
      char datei_La_Winkel    [200];
      char datei_Stroe_Winkel [200];
      char datei_Ge_Dreiecke  [200];
      char datei_schaufel_xyz [200];
      char datei_zwischen_prof[200];
      char datei_Batrekhy_prof[200];
      char datei_randbedingung[200];
      char datei_leitprofil   [200];
   }verz_;

   extern struct ver2
   {
      char datei_ax_stf       [200];
      char datei_kart3d_geo   [200];
      char datei_kart3d_rb    [200];
   }ver2_;

   extern struct lesn{ int lesenok;}
   lesn_;

   extern struct dick
   {
      float dicke_i;
      float dicke_a;
      float d_strich_a;
      float hk_dicke_i;
      float hk_dicke_a;
   }dick_;

   extern struct prof
   {
      float versch_i;
      float versch_a;
   }prof_;

   extern struct akno
   {
      int an_kno;
      int bi;
      int ixkn;
      int iykn;
      int izkn;
   }akno_;

   extern struct anet
   {
      float f[1500000];
      int   e[3000000];
   }anet_;

   extern struct ainf
   {
      int lese_xyz_dat;
      int netz_speichern;
   }ainf_;

   extern struct awrb
   {
      int an3_wr;
      int iwrb1;
      int iwrb2;
      int iwrb3;
      int iwrb4;
   }awrb_;
   extern struct akbi
   {
      int an3_kb;
      int ikbi1;
      int ikbi2;
      int ikbi3;
      int ikbi4;
   }akbi_;
   extern struct axel
   {
      int a_3Del;
      int iel1;
      int iel2;
      int iel3;
      int iel4;
      int iel5;
      int iel6;
      int iel7;
      int iel8;
   }axel_;
   extern struct aplo
   {
      int ixr;
      int iyr;
      int seed;
   }aplo_;
   extern struct leits
   {
      float x_leit_prof[38];
      float y_leit_prof[38];
   }leits_;
   extern struct umle
   {
      float uml_r;
      float uml_z;
   }umle_;

}


class Application
{

   private:

      //  member functions
      void compute(void *callbackData);
      void quit(void *callbackData);
      void parameter(void *callbackData);

      //  Static callback stubs
      static void computeCallback(void *userData, void *callbackData);
      static void quitCallback(void *userData, void *callbackData);
      static void parameterCallback(bool inMapLoading, void *userData, void *callbackData);

      float minH, maxH,minQ,maxQ,minN,maxN,mind2,maxd2;
      float mindi_da,maxdi_da,mindb2i,maxdb2i;
      float mindb1i,maxdb1i,mindb2a,maxdb2a,mindb1a,maxdb1a,minmwi,maxmwi;
      float minmwa,maxmwa;
      float minumschlingung, maxumschlingung;
      float minmax1, maxmax1;
      float minmax2, maxmax2;
      float minpe1i, maxpe1i;
      float minpe1a, maxpe1a;
      float minpe2i, maxpe2i;
      float minpe2a, maxpe2a;
      float mindicke_a, maxdicke_a;
      float mindicke_i, maxdicke_i;
      float mind_strich_a, maxd_strich_a;
      float minversch_i,maxversch_i;
      float minversch_a,maxversch_a;
      float minlp,maxlp,lp;
      float minhk_dicke_a,maxhk_dicke_a;
      float minhk_dicke_i,maxhk_dicke_i;
      float minuml_r, maxuml_r;
      float minuml_z, maxuml_z;

      float nlschaufel_alt, q_alt, h_alt, n_alt, d2_alt, di_da_alt;
      float umschlingung_alt, max2_alt, pe2i_alt, pe2a_alt;
      float max1_alt, pe1i_alt, pe1a_alt;
      float dicke_i_alt, dicke_a_alt, d_strich_a_alt;
      float versch_i_alt, versch_a_alt;
      float db2i_alt, db2a_alt, db1i_alt, db1a_alt, mwi_alt, mwa_alt;
      float mind2_kugel,maxd2_kugel,d2_kugel_alt;
      float sp;
      float hk_dicke_i_alt, hk_dicke_a_alt;
      float uml_r_alt, uml_z_alt;

      long int minn,maxn,minkranxwin,maxkranxwin;
      long int minschaufelzahl, maxschaufelzahl,kranzwin;

      int rotate,drehsinn,drehsinn_alt;
      int zeige_laufrad, zeige_nabe, zeige_kranz, zeige_netz, schaufelzahl_alt;
      long int schaufelzahl;
      int zeige_leit;
      int netz_aktuell;
      int netzstart,netzlesen;
      int xyz_lesen,proto_lesen;
      int neue_geometrie;

   public:

      Application(int argc, char *argv[])

      {
         char *file;

         Covise::set_module_description("IHS Schaufelgenerator");

         /* --- Ausgabe-Geometrie --- */
         Covise::add_port(OUTPUT_PORT,"geom","Polygons","Schaufelgeometrie");
         Covise::add_port(OUTPUT_PORT,"nabe","Polygons","Nabe");
         Covise::add_port(OUTPUT_PORT,"kranz","Polygons","Kranz");
         Covise::add_port(OUTPUT_PORT,"leit","Polygons","Leitschaufeln");
         Covise::add_port(OUTPUT_PORT,"netz","UnstructuredGrid","Netz");
         Covise::add_port(OUTPUT_PORT,"eintritt","Lines","RB_Eintritt");

         Covise::add_port(OUTPUT_PORT,"2dplotkonform","Set_Vec2","Plotdata");
         Covise::add_port(OUTPUT_PORT,"2dplotschnitt","Set_Vec2","Plotdata_schnitt");
         Covise::add_port(OUTPUT_PORT,"2dplotnetznabe","Set_Vec2","Plotdata_netz_nabe");
         Covise::add_port(OUTPUT_PORT,"2dplotkanalnabe","Set_Vec2","Plotdata_kanalnabe");
         Covise::add_port(OUTPUT_PORT,"2dplotkanalkranz","Set_Vec2","Plotdata_kanalkranz");

         Covise::add_port(PARIN,"Datenverzeichnis","Browser","Datenverzeichnis");
#ifdef _WIN32
         std::string dp = coCoviseConfig::getEntry("value","IHS.DataPath",getenv("USERPROFILE"));
#else
         std::string dp = coCoviseConfig::getEntry("value","IHS.DataPath",getenv("HOME"));
#endif
         file = new char[dp.length()+100];
         strcpy(file,dp.c_str());
         strcat(file," *.dat");
         Covise::set_port_default("Datenverzeichnis",file);

         /* --- Eingabe-Parameter --- */

         Covise::add_port(PARIN,"Hauptmenue","Choice","Auswahl Hauptmenue");
         Covise::set_port_default("Hauptmenue","2 Darstellung Nenndaten Schaufelkontur Schaufeldicken Profilverschiebung Winkeluebertreibung Woelbung Nabe Kranz Netz Datei");

         Covise::add_port(PARIN,"Laufrad"    ,"Boolean","Laufrad, an/aus");
         Covise::set_port_default("Laufrad"  ,"TRUE");

         Covise::add_port(PARIN,"Nabe"    ,"Boolean","Nabe, an/aus");
         Covise::set_port_default("Nabe"  ,"TRUE");

         Covise::add_port(PARIN,"Kranz"    ,"Boolean","Kranz, an/aus");
         Covise::set_port_default("Kranz"  ,"TRUE");

         Covise::add_port(PARIN,"Netz"    ,"Boolean","Netz, an/aus");
         Covise::set_port_default("Netz"  ,"TRUE");

         Covise::add_port(PARIN,"Rotieren"    ,"Boolean","Rotieren, an/aus");
         Covise::set_port_default("Rotieren"  ,"FALSE");

         Covise::add_port(PARIN,"Drehsinn"    ,"Boolean","Drehsinn, an/aus");
         Covise::set_port_default("Drehsinn"  ,"FALSE");

         Covise::add_port(PARIN,"Schaufelzahl"    ,"IntSlider","value for schaufelzahl");
         Covise::set_port_default("Schaufelzahl"  ,"1 10 3");
         minschaufelzahl= 1;
         maxschaufelzahl=10;
         schaufelzahl= 3;

         Covise::add_port(PARIN,"Kranzwinkel"        ,"FloatSlider","value for Winkel");
         Covise::set_port_default("Kranzwinkel"      ,"10 360 180");
         minkranxwin= 10;
         maxkranxwin=360;
         kranzwin= 180;

         Covise::add_port(PARIN,"Lastpunkt"        ,"FloatSlider","value for Lastpunkt");
         Covise::set_port_default("Lastpunkt"      ,"0.10 1.50 1.00");
         minlp=0.10f;
         maxlp=1.50f;
         lp= 1.00f;

         Covise::add_port(PARIN,"Q"       ,"FloatSlider","value for Q");
         Covise::set_port_default("Q"     ,"1.0 10.0 4.7");
         minQ=1.0;
         maxQ=10.0;
         absol_.Q=4.7f;

         Covise::add_port(PARIN,"H"          ,"FloatSlider","value for H");
         Covise::set_port_default("H"        ,"1.0 10.0 2.7");
         minH=1.0;
         maxH=10.0;
         absol_.H=2.7f;

         Covise::add_port(PARIN,"n"           ,"FloatSlider","value for n");
         Covise::set_port_default("n"         ,"150.0 500.0 220.0");
         minN=150.0;
         maxN=500.0;
         absol_.N=220.0;

         Covise::add_port(PARIN,"D2"  ,"FloatSlider","value for D2");
         Covise::set_port_default("D2","1.0 2.0 1.28");
         mind2=1.0;
         maxd2=2.0;
         zwis_.d2=1.28f;

         Covise::add_port(PARIN,"D1/D2"              ,"FloatSlider","value for D2izuD2a");
         Covise::set_port_default("D1/D2"            ,"0.20 1.00 0.4375");
         mindi_da= 0.2f;
         maxdi_da= 1.0f;
         zwis_.di_da=0.4375f;

         Covise::add_port(PARIN,"z","IntSlider","Number of blades");
         Covise::set_port_default("z","1 20 3");

         Covise::add_port(PARIN,"Umschlingung"              ,"FloatSlider","value for Umschlingung");
         Covise::set_port_default("Umschlingung"            ,"0.5 1.2 0.9");
         minumschlingung= 0.5f;
         maxumschlingung= 1.2f;
         kont_.umschlingung=0.9f;

         Covise::add_port(PARIN,"max1"              ,"FloatSlider","value for max1");
         Covise::set_port_default("max1"            ,"0.0 1.0 0.35");
         minmax1= 0.0f;
         maxmax1= 1.0f;
         kont_.max1=0.35f;

         Covise::add_port(PARIN,"max2"              ,"FloatSlider","value for max2");
         Covise::set_port_default("max2"            ,"0.0 1.0 0.4");
         minmax1= 0.0f;
         maxmax1= 1.0f;
         kont_.max2=0.4f;

         Covise::add_port(PARIN,"pe1i"              ,"FloatSlider","value for pe1i");
         Covise::set_port_default("pe1i"            ,"0.0 0.4 0.15");
         minpe1i= 0.0f;
         maxpe1i= 0.4f;
         kont_.pe1i=0.15f;

         Covise::add_port(PARIN,"pe1a"              ,"FloatSlider","value for pe1a");
         Covise::set_port_default("pe1a"            ,"0.0 0.4 0.1");
         minpe1a= 0.0f;
         maxpe1a= 0.4f;
         kont_.pe1a=0.1f;

         Covise::add_port(PARIN,"pe2i"              ,"FloatSlider","value for pe2i");
         Covise::set_port_default("pe2i"            ,"0.0 0.4 0.08");
         minpe2i= 0.0f;
         maxpe2i= 0.4f;
         kont_.pe2i=0.08f;

         Covise::add_port(PARIN,"pe2a"              ,"FloatSlider","value for pe2a");
         Covise::set_port_default("pe2a"            ,"0.0 0.4 0.1");
         minpe2a= 0.0f;
         maxpe2a= 0.4f;
         kont_.pe2a=0.1f;

         Covise::add_port(PARIN,"dicke_i"              ,"FloatSlider","value for dicke_i");
         Covise::set_port_default("dicke_i"            ,"0.0 40.0 12.0");
         mindicke_i= 0.0f;
         maxdicke_i= 40.0f;
         dick_.dicke_i= 20.0f;

         Covise::add_port(PARIN,"dicke_a"              ,"FloatSlider","value for dicke_a");
         Covise::set_port_default("dicke_a"            ,"0.0 40.0 8.0");
         mindicke_a= 0.0f;
         maxdicke_a= 40.0f;
         dick_.dicke_a= 8.0f;

         Covise::add_port(PARIN,"d_strich_a"              ,"FloatSlider","value for d_strich_a");
         Covise::set_port_default("d_strich_a"            ,"0.0 1.0 0.0");
         mind_strich_a= 0.0f;
         maxd_strich_a= 1.0f;
         dick_.d_strich_a= 0.0f;

         Covise::add_port(PARIN,"hk_dicke_i"              ,"FloatSlider","value for hk_dicke_i");
         Covise::set_port_default("hk_dicke_i"            ,"0.0 5.0 1.0");
         minhk_dicke_i= 0.0f;
         maxhk_dicke_i= 5.0f;
         dick_.hk_dicke_i= 1.0f;

         Covise::add_port(PARIN,"hk_dicke_a"              ,"FloatSlider","value for hk_dicke_a");
         Covise::set_port_default("hk_dicke_a"            ,"0.0 5.0 1.0");
         minhk_dicke_a= 0.0f;
         maxhk_dicke_a= 5.0f;
         dick_.hk_dicke_a= 1.0f;

         Covise::add_port(PARIN,"versch_i"              ,"FloatSlider","value for versch_i");
         Covise::set_port_default("versch_i"            ,"1.0 1.5 1.1");
         minversch_i= 1.0f;
         maxversch_i= 1.5f;
         prof_.versch_i= 1.1f;

         Covise::add_port(PARIN,"versch_a"              ,"FloatSlider","value for versch_a");
         Covise::set_port_default("versch_a"            ,"1.0 1.5 1.3");
         minversch_a= 1.0f;
         maxversch_a= 1.5f;
         prof_.versch_a= 1.3f;

         Covise::add_port(PARIN,"db2i"    ,"FloatSlider","value for db2i");
         Covise::set_port_default("db2i"  ,"0.0 5.0 3.0");
         mindb2i=0.0f;
         maxdb2i=5.0f;
         wust_.db2i=3.0f;

         Covise::add_port(PARIN,"db2a"    ,"FloatSlider","value for db2a");
         Covise::set_port_default("db2a"  ,"0.0 5.0 1.0");
         mindb2a=0.0;
         maxdb2a=5.0;
         wust_.db2a=1.0;

         Covise::add_port(PARIN,"db1i"    ,"FloatSlider","value for db1i");
         Covise::set_port_default("db1i"  ,"0.0 8.0 5.0");
         mindb1i=0.0f;
         maxdb1i=8.0f;
         wust_.db1i=5.0f;

         Covise::add_port(PARIN,"db1a"    ,"FloatSlider","value for db1a");
         Covise::set_port_default("db1a"  ,"0.0 8.0 4.0");
         mindb1a=0.0f;
         maxdb1a=8.0f;
         wust_.db1a=4.0f;

         Covise::add_port(PARIN,"mwi"    ,"FloatSlider","value for mwi");
         Covise::set_port_default("mwi"  ,"0.30 0.48 0.40");
         minmwi=0.3f;
         maxmwi=0.48f;
         maxw_.mwi=0.4f;

         Covise::add_port(PARIN,"mwa"    ,"FloatSlider","value for mwa");
         Covise::set_port_default("mwa"  ,"0.30 0.48 0.40");
         mindb2i=0.3f;
         maxdb2i=0.48f;
         wust_.db2i=0.4f;

         Covise::add_port(PARIN,"Kugelradius"    ,"FloatSlider","value for Kugelradius");
         Covise::set_port_default("Kugelradius"  ,"1.00 1.20 1.07");
         mind2_kugel=1.00f;
         maxd2_kugel=1.20f;
         fona_.d2_kugel=1.07f;

         Covise::add_port(PARIN,"uml_r"    ,"FloatSlider","value for uml_r");
         Covise::set_port_default("uml_r"  ,"0.00 0.10 0.05");
         minuml_r=0.00f;
         maxuml_r=0.10f;
         umle_.uml_r=0.05f;

         Covise::add_port(PARIN,"uml_z"    ,"FloatSlider","value for uml_z");
         Covise::set_port_default("uml_z"  ,"0.00 0.20 0.1");
         minuml_z=0.00f;
         maxuml_z=0.20f;
         umle_.uml_z=0.1f;

         Covise::add_port(PARIN,"oeffnen prototyp.dat"    ,"Boolean","value for oeffnen prototyp.dat");
         Covise::set_port_default("oeffnen prototyp.dat"  ,"FALSE");

         Covise::add_port(PARIN,"oeffnen schaufel_xyz.dat"    ,"Boolean","value for oeffnen schaufel_xyz.dat");
         Covise::set_port_default("oeffnen schaufel_xyz.dat"  ,"FALSE");

         Covise::add_port(PARIN,"Stroe-Winkel.dat"    ,"Boolean","value for Dateiausgabe");
         Covise::set_port_default("Stroe-Winkel.dat"  ,"TRUE");

         Covise::add_port(PARIN,"La-Winkel.dat"    ,"Boolean","value for Dateiausgabe");
         Covise::set_port_default("La-Winkel.dat"  ,"TRUE");

         Covise::add_port(PARIN,"schaufel_xyz.dat"    ,"Boolean","value for Dateiausgabe");
         Covise::set_port_default("schaufel_xyz.dat"  ,"TRUE");

         Covise::add_port(PARIN,"prototyp.dat"   ,"Boolean","value for prototyp.dat");
         Covise::set_port_default("prototyp.dat"  ,"TRUE");

         Covise::add_port(PARIN,"Dateiausgabe"   ,"Boolean","value for Dateiausgabe");
         Covise::set_port_default("Dateiausgabe"  ,"FALSE");

         Covise::add_port(PARIN,"Netz oeffnen"    ,"Boolean","value for Netz oeffnen");
         Covise::set_port_default("Netz oeffnen"  ,"FALSE");

         Covise::add_port(PARIN,"Netz speichern"   ,"Boolean","value for Netz speichern");
         Covise::set_port_default("Netz speichern"  ,"FALSE");

         Covise::add_port(PARIN,"Netzgenerierung"    ,"Boolean","value for Netzgenerierung");
         Covise::set_port_default("Netzgenerierung"  ,"FALSE");

         Covise::init(argc,argv);

         char buf[500];
         sprintf(buf,"C%s\n%s\n%s\n",Covise::get_module(),Covise::get_instance(),Covise::get_host());
         Covise::set_feedback_info(buf);
         Covise::set_quit_callback(Application::quitCallback,this);
         Covise::set_start_callback(Application::computeCallback,this);
         Covise::set_param_callback(Application::parameterCallback,this);

         Covise::show_param("Hauptmenue");
         Covise::show_param("Q");
         Covise::show_param("H");
         Covise::show_param("n");
         Covise::show_param("D2");
         Covise::show_param("D1/D2");
         Covise::show_param("z");

         neue_geometrie=1;
         netz_aktuell=0;

      }

      void run() { Covise::main_loop(); }

      ~Application() {}

};
#endif                                            // _READIHS_H
