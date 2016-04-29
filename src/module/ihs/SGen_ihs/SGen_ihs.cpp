/**************************************************************************\ 
 **                                                           (C)1998 IHS  **
 **                                                                        **
 ** Description: Interaktiver Schaufelgenerator     	                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **     Alexander Kaps                                                     **
 **     Universitaet Stuttgart                                             **
 **     Pfaffenwaldring 10                                                 **
 **     70550 Stuttgart                                                    **
 **                                                                        **
 ** Date:  26.08.98 (Start)                                                **
\**************************************************************************/
#include <config/CoviseConfig.h>
#include <appl/ApplInterface.h>
#include "SGen_ihs.h"

#include <do/coDoPolygons.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoSet.h>
#include <do/coDoData.h>

int main(int argc, char *argv[])
{

   io_.lese=1;

   Application *application = new Application(argc,argv);
   application->run();

   return(1);

}


//
// static stub callback functions calling the real class
// member functions
//

void Application::quitCallback(void *userData, void *callbackData)
{
   Application *thisApp = (Application *)userData;
   thisApp->quit(callbackData);
}


void Application::computeCallback(void *userData, void *callbackData)
{
   Application *thisApp = (Application *)userData;
   thisApp->compute(callbackData);
}


void Application::parameterCallback(bool /*inMapLoading*/, void *userData, void *callbackData)
{
   Application *thisApp = (Application *)userData;
   thisApp->parameter(callbackData);
}


//
//
//..........................................................................
//
void Application::quit(void *)
{
   //
   // ...... delete your data here .....
   //

}


void Application::compute(void *)
{
   //
   // ...... do work here ........
   //

   // read input parameters and data object name

   int num_slices = 11;
   int num_pts    = 19;
   int num_kreis  = 40;
   int num_leit = 5;
   int num_leit_pts = 38;
   int i,j,n,b,zl,anznabe;
   int num_blades;
   int num_kranz;
   int n_coord, aussen;

   float PI=3.14159f;
   float enderad;

   char buf[500000];
   char verzeichnis[500];

   int *plkranz, *vlkranz;
   int *pl_leit, *vl_leit;
   int *el_netz, *vl_netz, *tl;

   float *x_cokranz, *y_cokranz, *z_cokranz;
   float *x_co_netz, *y_co_netz, *z_co_netz;
   float *x_leit,*y_leit,*z_leit;

   char *grid_path;
   static char verz_alt[2000];

   char *Geom_obj_name;
   char *Nabe_obj_name;
   char *Kranz_obj_name;
   char *Leit_obj_name;
   char *Mesh;

   coDoPolygons         *nabe_strip;
   coDoPolygons         *kranz_strip;
   coDoPolygons         *leit_strip;
   coDoUnstructuredGrid *mesh;

   if (lesn_.lesenok==0) io_.lese=1;

   Covise::get_browser_param("Datenverzeichnis", &grid_path);

   strcpy(verzeichnis,grid_path);
   if (verzeichnis[strlen(verzeichnis)-1]=='t')
   {
      for(i=strlen(verzeichnis);(verzeichnis[i]!='/')&&(verzeichnis[i]!='\\')&&(i!=0);i--)
         {}
   }
   else
   {
      for(i=strlen(verzeichnis);(i!=0)&&!((verzeichnis[i]=='/')&&(verzeichnis[i+1]=='/'));i--)
         ;                
   }

   verzeichnis[++i]='\0';

   printf("verz : %s\n",verzeichnis);

   if (strcmp(verzeichnis,verz_alt)!=0)
   {
      io_.lese=1;
      neue_geometrie=1;
   }

   strcpy(verz_alt,verzeichnis);

   /* --- Eingabe-Dateien --- */
   strcpy(verz_.datei_prototyp      ,verzeichnis);
   strcpy(verz_.datei_zentral_file  ,verzeichnis);
   strcpy(verz_.datei_daten        ,verzeichnis);
   strcpy(verz_.datei_dicke        ,verzeichnis);
   strcpy(verz_.datei_profil       ,verzeichnis);
   strcpy(verz_.datei_steuerfile    ,verzeichnis);
   strcpy(verz_.datei_leitprofil    ,verzeichnis);
   strcpy(ver2_.datei_ax_stf        ,verzeichnis);

   /* --- Ausgabe-Dateien --- */
   strcpy(verz_.datei_profil_neu    ,verzeichnis);
   strcpy(verz_.datei_output       ,verzeichnis);
   strcpy(verz_.datei_zwischen_file ,verzeichnis);
   strcpy(verz_.datei_meko        ,verzeichnis);
   strcpy(verz_.datei_spezifisch    ,verzeichnis);
   strcpy(verz_.datei_La_Winkel     ,verzeichnis);
   strcpy(verz_.datei_Stroe_Winkel  ,verzeichnis);
   strcpy(verz_.datei_Ge_Dreiecke   ,verzeichnis);
   strcpy(verz_.datei_schaufel_xyz  ,verzeichnis);
   strcpy(verz_.datei_zwischen_prof ,verzeichnis);
   strcpy(verz_.datei_Batrekhy_prof ,verzeichnis);
   strcpy(verz_.datei_randbedingung ,verzeichnis);
   strcpy(ver2_.datei_kart3d_geo    ,verzeichnis);
   strcpy(ver2_.datei_kart3d_rb     ,verzeichnis);

   /* --- Eingabe-Dateien --- */
   strcat(verz_.datei_prototyp      ,"prototyp.dat");
   strcat(verz_.datei_zentral_file  ,"zentral_file.dat");
   strcat(verz_.datei_daten        ,"daten.dat");
   strcat(verz_.datei_dicke        ,"dicke.dat");
   strcat(verz_.datei_profil       ,"n0018.dat");
   strcat(verz_.datei_steuerfile    ,"steuerfile.dat");
   strcat(verz_.datei_leitprofil    ,"leitprofil.dat");
   strcat(ver2_.datei_ax_stf        ,"ax_stf.dat");

   /* --- Ausgabe-Dateien --- */
   strcat(verz_.datei_profil_neu    ,"ausgabe/profil_neu.dat");
   strcat(verz_.datei_output       ,"ausgabe/output.dat");
   strcat(verz_.datei_zwischen_file ,"ausgabe/zwischen_file.dat");
   strcat(verz_.datei_meko      ,"ausgabe/meko.dat");
   strcat(verz_.datei_spezifisch    ,"ausgabe/spezifisch.dat");
   strcat(verz_.datei_La_Winkel     ,"ausgabe/La-Winkel.dat");
   strcat(verz_.datei_Stroe_Winkel  ,"ausgabe/Stroe-Winkel.dat");
   strcat(verz_.datei_Ge_Dreiecke   ,"ausgabe/Ge-Dreiecke.dat");
   strcat(verz_.datei_schaufel_xyz  ,"ausgabe/schaufel_xyz.dat");
   strcat(verz_.datei_zwischen_prof ,"ausgabe/zwischen_prof.dat");
   strcat(verz_.datei_Batrekhy_prof ,"ausgabe/Batrekhy_prof.dat");
   strcat(verz_.datei_randbedingung ,"ausgabe/RB.dat");
   strcat(ver2_.datei_kart3d_geo    ,"ausgabe/KART3D.GEO");
   strcat(ver2_.datei_kart3d_rb     ,"ausgabe/KART3D.RB");
   //}

   /* --- Welche Eingabe wurde getaetigt ? --- */
   nlschaufel_alt=(float)zwis_.nlschaufel;
   drehsinn_alt=drehsinn;
   q_alt=absol_.Q;
   h_alt=absol_.H;
   n_alt=absol_.N;
   d2_alt=zwis_.d2;
   di_da_alt=zwis_.di_da;
   umschlingung_alt=kont_.umschlingung;
   max2_alt=kont_.max2;
   pe2i_alt=kont_.pe2i;
   pe2a_alt=kont_.pe2a;
   max1_alt=kont_.max1;
   pe1i_alt=kont_.pe1i;
   pe1a_alt=kont_.pe1a;
   dicke_i_alt=dick_.dicke_i;
   dicke_a_alt=dick_.dicke_a;
   d_strich_a_alt=dick_.d_strich_a;
   hk_dicke_i_alt=dick_.hk_dicke_i;
   hk_dicke_a_alt=dick_.hk_dicke_a;
   versch_i_alt=prof_.versch_i;
   versch_a_alt=prof_.versch_a;
   db2i_alt=wust_.db2i;
   db2a_alt=wust_.db2a;
   db1i_alt=wust_.db1i;
   db1a_alt=wust_.db1a;
   mwi_alt=maxw_.mwi;
   mwa_alt=maxw_.mwa;
   d2_kugel_alt=fona_.d2_kugel;

   Covise::get_slider_param  ("z"        , &minn, &maxn, &zwis_.nlschaufel);
   Covise::get_slider_param  ("Q"           , &minQ,&maxQ,&absol_.Q);
   Covise::get_slider_param  ("H"        , &minH,&maxH,&absol_.H);
   Covise::get_slider_param  ("n"        , &minN,&maxN,&absol_.N);
   Covise::get_slider_param  ("D2"       , &mind2,&maxd2,&zwis_.d2);
   Covise::get_slider_param  ("D1/D2"               , &mindi_da,&maxdi_da,&zwis_.di_da);

   Covise::get_slider_param  ("Umschlingung"      , &minumschlingung,&maxumschlingung,&kont_.umschlingung);
   Covise::get_slider_param  ("max2"        , &minmax2,&maxmax2,&kont_.max2);
   Covise::get_slider_param  ("pe2i"        , &minpe2i,&maxpe2i,&kont_.pe2i);
   Covise::get_slider_param  ("pe2a"        , &minpe2a,&maxpe2a,&kont_.pe2a);
   Covise::get_slider_param  ("max1"        , &minmax1,&maxmax1,&kont_.max1);
   Covise::get_slider_param  ("pe1i"        , &minpe1i,&maxpe1i,&kont_.pe1i);
   Covise::get_slider_param  ("pe1a"        , &minpe1a,&maxpe1a,&kont_.pe1a);

   Covise::get_slider_param  ("dicke_i"     , &mindicke_i, &maxdicke_i, &dick_.dicke_i);
   Covise::get_slider_param  ("dicke_a"     , &mindicke_a, &maxdicke_a, &dick_.dicke_a);
   Covise::get_slider_param  ("d_strich_a"     , &mind_strich_a, &maxd_strich_a, &dick_.d_strich_a);
   Covise::get_slider_param  ("hk_dicke_i"     , &minhk_dicke_i, &maxhk_dicke_i, &dick_.hk_dicke_i);
   Covise::get_slider_param  ("hk_dicke_a"     , &minhk_dicke_a, &maxhk_dicke_a, &dick_.hk_dicke_a);

   Covise::get_slider_param  ("versch_i"    , &minversch_i, &maxversch_i, &prof_.versch_i);
   Covise::get_slider_param  ("versch_a"    , &minversch_a, &maxversch_a, &prof_.versch_a);

   Covise::get_slider_param  ("db2i"        , &mindb2i,&maxdb2i,&wust_.db2i);
   Covise::get_slider_param  ("db2a"        , &mindb2a,&maxdb2a,&wust_.db2a);
   Covise::get_slider_param  ("db1i"        , &mindb1i,&maxdb1i,&wust_.db1i);
   Covise::get_slider_param  ("db1a"        , &mindb1a,&maxdb1a,&wust_.db1a);

   Covise::get_slider_param  ("mwi"         , &minmwi,&maxmwi,&maxw_.mwi);
   Covise::get_slider_param  ("mwa"         , &minmwa,&maxmwa,&maxw_.mwa);

   Covise::get_slider_param  ("Kugelradius"       , &mind2_kugel,&maxd2_kugel,&fona_.d2_kugel);

   Covise::get_boolean_param ("oeffnen prototyp.dat"     , &proto_lesen);
   Covise::get_boolean_param ("oeffnen schaufel_xyz.dat" , &xyz_lesen);
   Covise::get_boolean_param ("Stroe-Winkel.dat"    , &io_.WRSTROE_WINKEL);
   Covise::get_boolean_param ("La-Winkel.dat"       , &io_.WRLA_WINKEL);
   Covise::get_boolean_param ("schaufel_xyz.dat"    , &io_.WRSCHAUFEL_XYZ);
   Covise::get_boolean_param ("prototyp.dat"             , &io_.WRPROTOTYP);
   Covise::get_boolean_param ("Dateiausgabe"        , &io_.schreibe);

   Covise::get_boolean_param ("Netzgenerierung"   , &netzstart);
   Covise::get_boolean_param ("Netz speichern"    , &ainf_.netz_speichern);
   Covise::get_boolean_param ("Netz oeffnen"      , &netzlesen);

   Covise::get_boolean_param ("Laufrad"        , &zeige_laufrad);
   Covise::get_boolean_param ("Nabe"        , &zeige_nabe);
   Covise::get_boolean_param ("Kranz"          , &zeige_kranz);
   Covise::get_boolean_param ("Netz"           , &zeige_netz);
   Covise::get_boolean_param ("Rotieren"    , &rotate);
   Covise::get_boolean_param ("Drehsinn"    , &drehsinn);
   Covise::get_slider_param  ("Schaufelzahl"      , &minschaufelzahl, &maxschaufelzahl, &schaufelzahl);
   Covise::get_slider_param  ("Kranzwinkel"    , &minkranxwin, &maxkranxwin, &kranzwin);
   Covise::get_slider_param  ("Lastpunkt"      , &minlp, &maxlp, &lp);

   Covise::get_slider_param  ("uml_r"       ,&minuml_r, &maxuml_r, &umle_.uml_r);
   Covise::get_slider_param  ("uml_z"       ,&minuml_z, &maxuml_z, &umle_.uml_z);

   if (nlschaufel_alt!=zwis_.nlschaufel) neue_geometrie=1;
   if (q_alt!=absol_.Q)              neue_geometrie=1;
   if (h_alt!=absol_.H)              neue_geometrie=1;
   if (n_alt!=absol_.N)                 neue_geometrie=1;
   if (d2_alt!=zwis_.d2)               neue_geometrie=1;
   if (di_da_alt!=zwis_.di_da)        neue_geometrie=1;
   if (umschlingung_alt!=kont_.umschlingung) neue_geometrie=1;
   if (max2_alt!=kont_.max2)             neue_geometrie=1;
   if (pe2i_alt!=kont_.pe2i)             neue_geometrie=1;
   if (pe2a_alt!=kont_.pe2a)             neue_geometrie=1;
   if (max1_alt!=kont_.max1)             neue_geometrie=1;
   if (pe1i_alt!=kont_.pe1i)             neue_geometrie=1;
   if (pe1a_alt!=kont_.pe1a)             neue_geometrie=1;
   if (dicke_i_alt!=dick_.dicke_i)       neue_geometrie=1;
   if (dicke_a_alt!=dick_.dicke_a)       neue_geometrie=1;
   if (d_strich_a_alt!=dick_.d_strich_a)     neue_geometrie=1;
   if (hk_dicke_i_alt!=dick_.hk_dicke_i)     neue_geometrie=1;
   if (hk_dicke_a_alt!=dick_.hk_dicke_a)     neue_geometrie=1;
   if (versch_i_alt!=prof_.versch_i)        neue_geometrie=1;
   if (versch_a_alt!=prof_.versch_a)        neue_geometrie=1;
   if (db2i_alt!=wust_.db2i)      neue_geometrie=1;
   if (db2a_alt!=wust_.db2a)      neue_geometrie=1;
   if (db1i_alt!=wust_.db1i)      neue_geometrie=1;
   if (db1a_alt!=wust_.db1a)      neue_geometrie=1;
   if (mwi_alt!=maxw_.mwi)            neue_geometrie=1;
   if (mwa_alt!=maxw_.mwa)            neue_geometrie=1;
   if (drehsinn!=drehsinn_alt)       neue_geometrie=1;
   if (io_.schreibe==1)            neue_geometrie=1;
   if (uml_r_alt!=umle_.uml_r)       neue_geometrie=1;
   if (uml_z_alt!=umle_.uml_z)       neue_geometrie=1;

   if (d2_kugel_alt!=fona_.d2_kugel) rechne_kugel_();

   if (proto_lesen==1)
   {
      neue_geometrie=1;
      io_.lese=1;
      proto_lesen=0;
      Covise::update_boolean_param ("oeffnen prototyp.dat" , proto_lesen);
   }

   if (neue_geometrie==1) netz_aktuell=0;

   Geom_obj_name = Covise::get_object_name("geom");
   if (Geom_obj_name==NULL)
   {
      Covise::sendError("Can't get name for 'geometry' object");
      return;
   }
   Nabe_obj_name = Covise::get_object_name("nabe");
   if (Nabe_obj_name==NULL)
   {
      Covise::sendError("Can't get name for 'nabe' object");
      return;
   }
   Kranz_obj_name = Covise::get_object_name("kranz");
   if (Kranz_obj_name==NULL)
   {
      Covise::sendError("Can't get name for 'kranz' object");
      return;
   }
   Leit_obj_name = Covise::get_object_name("leit");
   if (Leit_obj_name==NULL)
   {
      Covise::sendError("Can't get name for 'leit' object");
      return;
   }
   Mesh = Covise::get_object_name("netz");
   if (Mesh==NULL)
   {
      Covise::sendError("Can't get name for 'netz_flaechen' object");
      return;
   }

   /* --- Programmstart --- */

   if (neue_geometrie==1) geogen_();

   if (xyz_lesen==1)
   {
      xyz_lesen=0;
      Covise::update_boolean_param ("oeffnen schaufel_xyz.dat"      , xyz_lesen);

      eingabe_xyz_();

      if (io_.success==1)
      {
         Covise::sendWarning("schaufel_xyz.dat eingelesen");
      }
      else 
      {
         Covise::sendWarning("schaufel_xyz.dat nicht gefunden.");
      }
   }

   neue_geometrie=0;

   if (io_.schreibe==1)
   {
      io_.schreibe=0;
      strcpy(buf,"Dateien ausgegeben in ");
      strcat(buf,verzeichnis);
      strcat(buf,"ausgabe/");
      Covise::update_boolean_param ("Dateiausgabe"      , io_.schreibe);
      Covise::sendWarning("Dateien ausgegeben in %sausgabe/",verzeichnis);
   }

   if (lesn_.lesenok==1)
   {
      if (io_.fehler==1)
      {
         Covise::sendWarning("Keine Konvergenz!");
         io_.fehler=0;
      }

      if (io_.lese==1)
      {

         if (zwis_.nlschaufel>maxn) maxn=zwis_.nlschaufel;
         if (absol_.Q>maxQ) maxQ=absol_.Q;
         if (absol_.Q<minQ) minQ=absol_.Q;
         if (absol_.H>maxH) maxH=absol_.H;
         if (absol_.H<minH) minH=absol_.H;
         if (absol_.N>maxN) maxN=absol_.N;
         if (absol_.N<minN) minN=absol_.N;
         if (zwis_.d2>maxd2) maxd2=zwis_.d2;
         if (zwis_.d2<mind2) mind2=zwis_.d2;

         Covise::update_slider_param("z"    ,minn,maxn,zwis_.nlschaufel);
         Covise::update_slider_param("Q"    ,minQ,maxQ,absol_.Q);
         Covise::update_slider_param("H"    ,minH,maxH,absol_.H);
         Covise::update_slider_param("n"    ,minN,maxN,absol_.N);
         Covise::update_slider_param("D2"   ,mind2,maxd2,zwis_.d2);
         Covise::update_slider_param("D1/D2"    ,0.4f,0.80f,zwis_.di_da);
         Covise::update_slider_param("db2i"    ,0.0f,5.0f,wust_.db2i);
         Covise::update_slider_param("db2a"    ,0.0f,5.0f,wust_.db2a);
         Covise::update_slider_param("db1i"    ,-20.0f,8.0f,wust_.db1i);
         Covise::update_slider_param("db1a"    ,-20.0f,8.0f,wust_.db1a);
         Covise::update_slider_param("mwi"     ,0.3f,0.48f,maxw_.mwi);
         Covise::update_slider_param("mwa"     ,0.3f,0.48f,maxw_.mwa);
         Covise::update_slider_param("Kugelradius" ,1.0f,1.2f,fona_.d2_kugel);
         Covise::update_slider_param("Umschlingung",0.5f,1.5f,kont_.umschlingung);
         Covise::update_slider_param("max2"     ,0.0,1.0f,kont_.max2);
         Covise::update_slider_param("pe2i"     ,0.0,0.4f,kont_.pe2i);
         Covise::update_slider_param("pe2a"     ,0.0,0.4f,kont_.pe2a);
         Covise::update_slider_param("max1"     ,0.0,1.0f,kont_.max1);
         Covise::update_slider_param("pe1i"     ,0.0,0.4f,kont_.pe1i);
         Covise::update_slider_param("pe1a"     ,0.0,0.4f,kont_.pe1a);
         Covise::update_slider_param("dicke_i"     ,mindicke_i, maxdicke_i, dick_.dicke_i);
         Covise::update_slider_param("dicke_a"     ,mindicke_a, maxdicke_a, dick_.dicke_a);
         Covise::update_slider_param("d_strich_a"  ,mind_strich_a, maxd_strich_a, dick_.d_strich_a);
         Covise::update_slider_param("hk_dicke_i"  ,minhk_dicke_i, maxhk_dicke_i, dick_.hk_dicke_i);
         Covise::update_slider_param("hk_dicke_a"  ,minhk_dicke_a, maxhk_dicke_a, dick_.hk_dicke_a);
         Covise::update_slider_param("versch_i"    ,minversch_i, maxversch_i, prof_.versch_i);
         Covise::update_slider_param("versch_a"    ,minversch_a, maxversch_a, prof_.versch_a);
         Covise::update_slider_param("uml_r"       ,minuml_r, maxuml_r, umle_.uml_r);
         Covise::update_slider_param("uml_z"       ,minuml_z, maxuml_z, umle_.uml_z);
      }

      if(zwis_.nlschaufel!=nlschaufel_alt)
      {
         if(schaufelzahl>zwis_.nlschaufel)
         {
            schaufelzahl=zwis_.nlschaufel;
            maxschaufelzahl=zwis_.nlschaufel;
         }
         if(schaufelzahl_alt==nlschaufel_alt)
         {
            schaufelzahl=zwis_.nlschaufel;
            maxschaufelzahl=zwis_.nlschaufel;
         }
         Covise::update_slider_param("Schaufelzahl",minschaufelzahl,maxschaufelzahl,schaufelzahl);
      }

      num_blades=schaufelzahl;

      schaufelzahl_alt=schaufelzahl;
      nlschaufel_alt=(float)zwis_.nlschaufel;

      if(num_blades<1)
         num_blades=1;
      if(drehsinn==0)
         {sp=1.;}
         else
            {sp=-1.;};

      io_.lese=0;

      /* --- 2D-Plot Konforme Abbildung anfertigen --- */
      char *PlotData;
      float *xpl,*ypl;
      char plbuf[1000];
      char ticks[200];
      char world[400];

      int temp;
      int temp2;

      temp=(int)-(1.1*abwik_.y_ds_ab[num_slices-1][num_pts-1]+M_PI/zwis_.nlschaufel*kont_.umschlingung*lauf_.d_2[num_slices-1]/2);
      temp2=(int)-(1.1*abwik_.x_ss_ab[0][0]);
      if (temp2>temp) temp=temp2;
      temp=temp*100;

      for(i=0;temp>20;i++)
      {
         temp=temp/10;
      }
      temp++;

      for(;i>0;i--)
      {
         temp=temp*10;
      }
      //temp=1000;
      temp=temp/100;
      float xwmin=-1*(float)temp;
      float ywmin=-1*(float)temp;
      float xwmax=(float)temp;
      float ywmax=(float)temp;

      sprintf(world, "WORLD %f,%f,%f,%f\n", xwmin, ywmin, xwmax, ywmax);
      strcpy(plbuf, world);

      sprintf(ticks,"XAXIS TICK MAJOR %f\n", xwmax /2);
      strcat(plbuf, ticks);
      sprintf(ticks,"XAXIS TICK MINOR %f\n", xwmax/4);
      strcat(plbuf, ticks);
      sprintf(ticks,"YAXIS TICK MAJOR %f\n", ywmax/2);
      strcat(plbuf, ticks);
      sprintf(ticks,"YAXIS TICK MINOR %f\n", ywmax/4);
      strcat(plbuf, ticks);

      // --- Legende ---

      //strcpy(plbuf,"AUTOSCALE\n");
      strcat(plbuf,"TITLE \"Konforme Abbildung\"\n");
      strcat(plbuf,"XAXIS LABEL \"");
      strcat(plbuf,"Umfang");
      strcat(plbuf,"\"\n");
      strcat(plbuf,"YAXIS LABEL \"");
      strcat(plbuf,"Z-Achse");
      strcat(plbuf,"\"\n");

      PlotData = Covise::get_object_name("2dplotkonform");

      if (PlotData!=NULL)
      {
         coDoSet *plot_set = new coDoSet(PlotData,SET_CREATE);

         plot_set -> addAttribute("COMMANDS",plbuf);

         for(i=0;i<num_slices;i++)
         {
            sprintf(plbuf,"%s_%d",PlotData,i);

            coDoVec2 *plot_elem = new
               coDoVec2(plbuf,2*num_pts);

            if (plot_elem->objectOk())
            {
               plot_elem -> getAddresses(&xpl,&ypl);
               for(int j=0;j<num_pts;j++)
               {
                  xpl[j]=(float)(abwik_.y_ds_ab[i][j]+M_PI/zwis_.nlschaufel*kont_.umschlingung*lauf_.d_2[i]/2)*sp;
                  ypl[j]=-abwik_.x_ds_ab[i][j];
                  xpl[j+num_pts]=(float)(abwik_.y_ss_ab[i][num_pts-j-1]+M_PI/zwis_.nlschaufel*kont_.umschlingung*lauf_.d_2[i]/2)*sp;
                  ypl[j+num_pts]=-abwik_.x_ss_ab[i][num_pts-j-1];
               }
               plot_set -> addElement(plot_elem);
               delete plot_elem;
            }
         }
         delete plot_set;
      }

      /* --- 2D-Plot Kanal Nabe anfertigen --- */

      float teilung;
      teilung=(float)(M_PI*lauf_.d_2[1]/zwis_.nlschaufel);

      temp=(int)-(1.1*abwik_.y_ds_ab[num_slices-1][num_pts-1]+M_PI/zwis_.nlschaufel*kont_.umschlingung*lauf_.d_2[num_slices-1]/2);
      temp2=(int)(-(1.1*abwik_.x_ss_ab[0][0])+teilung);
      if (temp2>temp) temp=temp2;
      temp=temp*100;

      for(i=0;temp>20;i++)
      {
         temp=temp/10;
      }
      temp++;

      for(;i>0;i--)
      {
         temp=temp*10;
      }
      //temp=1000;
      temp=temp/100;
      /*float xwmin=-1*(float)temp;
      float ywmin=-1*(float)temp;
      float xwmax=(float)temp;
      float ywmax=(float)temp;*/

      sprintf(world, "WORLD %f,%f,%f,%f\n", xwmin, ywmin, xwmax, ywmax);
      strcpy(plbuf, world);

      sprintf(ticks,"XAXIS TICK MAJOR %f\n", xwmax /2);
      strcat(plbuf, ticks);
      sprintf(ticks,"XAXIS TICK MINOR %f\n", xwmax/4);
      strcat(plbuf, ticks);
      sprintf(ticks,"YAXIS TICK MAJOR %f\n", ywmax/2);
      strcat(plbuf, ticks);
      sprintf(ticks,"YAXIS TICK MINOR %f\n", ywmax/4);
      strcat(plbuf, ticks);

      // --- Legende ---

      //strcpy(plbuf,"AUTOSCALE\n");
      strcat(plbuf,"TITLE \"Stroemungskanal Nabe\"\n");
      strcat(plbuf,"XAXIS LABEL \"");
      strcat(plbuf,"Umfang");
      strcat(plbuf,"\"\n");
      strcat(plbuf,"YAXIS LABEL \"");
      strcat(plbuf,"Z-Achse");
      strcat(plbuf,"\"\n");

      PlotData = Covise::get_object_name("2dplotkanalnabe");

      if (PlotData!=NULL)
      {
         coDoSet *plot_set = new coDoSet(PlotData,SET_CREATE);

         plot_set -> addAttribute("COMMANDS",plbuf);

         for(i=0;i<2;i++)
         {
            sprintf(plbuf,"%s %d",PlotData,i);

            coDoVec2 *plot_elem = new
               coDoVec2(plbuf,2*num_pts);

            if (plot_elem->objectOk())
            {
               plot_elem -> getAddresses(&xpl,&ypl);
               for(j=0;j<num_pts;j++)
               {
                  xpl[j]=(float)(abwik_.y_ds_ab[1][j]+M_PI/zwis_.nlschaufel*kont_.umschlingung*lauf_.d_2[1]/2+teilung*(i-0.5))*sp;
                  ypl[j]=-abwik_.x_ds_ab[1][j];
                  xpl[j+num_pts]=(float)(abwik_.y_ss_ab[1][num_pts-j-1]+M_PI/zwis_.nlschaufel*kont_.umschlingung*lauf_.d_2[1]/2+teilung*(i-0.5))*sp;
                  ypl[j+num_pts]=-abwik_.x_ss_ab[1][num_pts-j-1];
               }
               plot_set -> addElement(plot_elem);
               delete plot_elem;
            }

         }

         delete plot_set;
      }

      /* --- 2D-Plot Kanal Kranz anfertigen --- */

      teilung=(float)(M_PI*lauf_.d_2[num_slices-1]/zwis_.nlschaufel);

      temp=(int)-(1.1*abwik_.y_ds_ab[num_slices-1][num_pts-1]+M_PI/zwis_.nlschaufel*kont_.umschlingung*lauf_.d_2[num_slices-1]/2);
      temp2=(int)(-(1.1*abwik_.x_ss_ab[0][0])+teilung);
      if (temp2>temp) temp=temp2;
      temp=temp*100;

      for(i=0;temp>20;i++)
      {
         temp=temp/10;
      }
      temp++;

      for(;i>0;i--)
      {
         temp=temp*10;
      }
      //temp=1000;
      temp=temp/100;
      /*float xwmin=-1*(float)temp;
      float ywmin=-1*(float)temp;
      float xwmax=(float)temp;
      float ywmax=(float)temp;*/

      sprintf(world, "WORLD %f,%f,%f,%f\n", xwmin, ywmin, xwmax, ywmax);
      strcpy(plbuf, world);

      sprintf(ticks,"XAXIS TICK MAJOR %f\n", xwmax /2);
      strcat(plbuf, ticks);
      sprintf(ticks,"XAXIS TICK MINOR %f\n", xwmax/4);
      strcat(plbuf, ticks);
      sprintf(ticks,"YAXIS TICK MAJOR %f\n", ywmax/2);
      strcat(plbuf, ticks);
      sprintf(ticks,"YAXIS TICK MINOR %f\n", ywmax/4);
      strcat(plbuf, ticks);

      // --- Legende ---

      //strcpy(plbuf,"AUTOSCALE\n");
      strcat(plbuf,"TITLE \"Stroemungskanal Kranz\"\n");
      strcat(plbuf,"XAXIS LABEL \"");
      strcat(plbuf,"Umfang");
      strcat(plbuf,"\"\n");
      strcat(plbuf,"YAXIS LABEL \"");
      strcat(plbuf,"Z-Achse");
      strcat(plbuf,"\"\n");

      PlotData = Covise::get_object_name("2dplotkanalkranz");

      if (PlotData!=NULL)
      {
         coDoSet *plot_set = new coDoSet(PlotData,SET_CREATE);

         plot_set -> addAttribute("COMMANDS",plbuf);

         for(i=0;i<2;i++)
         {
            sprintf(plbuf,"%s %d",PlotData,i);

            coDoVec2 *plot_elem = new
               coDoVec2(plbuf,2*num_pts);

            if (plot_elem->objectOk())
            {
               plot_elem -> getAddresses(&xpl,&ypl);
               for(j=0;j<num_pts;j++)
               {
                  xpl[j]=(float)(abwik_.y_ds_ab[num_slices-1][j]+M_PI/zwis_.nlschaufel*kont_.umschlingung*lauf_.d_2[num_slices-1]/2+teilung*(i-0.5))*sp;
                  ypl[j]=-abwik_.x_ds_ab[num_slices-1][j];
                  xpl[j+num_pts]=(float)(abwik_.y_ss_ab[num_slices-1][num_pts-j-1]+M_PI/zwis_.nlschaufel*kont_.umschlingung*lauf_.d_2[num_slices-1]/2+teilung*(i-0.5))*sp;
                  ypl[j+num_pts]=-abwik_.x_ss_ab[num_slices-1][num_pts-j-1];
               }
               plot_set -> addElement(plot_elem);
               delete plot_elem;
            }

         }

         delete plot_set;
      }

      /* --- 2D-Plot Schnitt anfertigen --- */
      char *PlotData_Schnitt;
      float *xpl_schnitt,*ypl_schnitt;
      float xpl_sch[50],ypl_sch[50];

      char plbuf_schnitt[3000];

      // --- Legende ---

      strcpy(plbuf_schnitt,"AUTOSCALE\n");
      strcat(plbuf_schnitt,"TITLE \"Laufrad\"\n");
      strcat(plbuf_schnitt,"XAXIS LABEL \"");
      strcat(plbuf_schnitt,"X-Achse");
      strcat(plbuf_schnitt,"\"\n");
      strcat(plbuf_schnitt,"YAXIS LABEL \"");
      strcat(plbuf_schnitt,"Y-Achse");
      strcat(plbuf_schnitt,"\"\n");

      PlotData_Schnitt = Covise::get_object_name("2dplotschnitt");

      if (PlotData_Schnitt!=NULL)
      {
         coDoSet *plot_set_schnitt = new coDoSet(PlotData_Schnitt,SET_CREATE);

         plot_set_schnitt -> addAttribute("COMMANDS",plbuf_schnitt);

         for(b=0;b<num_blades;b++)
         {
            /* --- Kreisausschnitte plotten (Profile) ---*/
            for(i=0;i<num_slices;i++)
            {
               sprintf(plbuf_schnitt,"%s %d",PlotData_Schnitt,i+b*(num_slices+num_pts));
               coDoVec2 *plot_elem_schnitt = new
                  coDoVec2(plbuf_schnitt,2*num_pts);
               if (plot_elem_schnitt->objectOk())
               {
                  plot_elem_schnitt -> getAddresses(&xpl_schnitt,&ypl_schnitt);
                  for(j=0;j<num_pts;j++)
                  {
                     xpl_sch[j]        =xyz_.x_ds[i][j];
                     ypl_sch[j]        =xyz_.y_ds[i][j];
                     xpl_sch[j+num_pts]=xyz_.x_ss[i][num_pts-j-1];
                     ypl_sch[j+num_pts]=xyz_.y_ss[i][num_pts-j-1];

                     xpl_schnitt[j]        =(float)((xpl_sch[j]*cos(b*(2.0*M_PI/zwis_.nlschaufel))-ypl_sch[j]*sin(b*(2.0*M_PI/zwis_.nlschaufel)))*sp);
                     ypl_schnitt[j]        =(float)(xpl_sch[j]*sin(b*(2.0*M_PI/zwis_.nlschaufel))+ypl_sch[j]*cos(b*(2.0*M_PI/zwis_.nlschaufel)));
                     xpl_schnitt[j+num_pts]=(float)((xpl_sch[j+num_pts]*cos(b*(2.0*M_PI/zwis_.nlschaufel))-ypl_sch[j+num_pts]*sin(b*(2.0*M_PI/zwis_.nlschaufel)))*sp);
                     ypl_schnitt[j+num_pts]=(float)(xpl_sch[j+num_pts]*sin(b*(2.0*M_PI/zwis_.nlschaufel))+ypl_sch[j+num_pts]*cos(b*(2.0*M_PI/zwis_.nlschaufel)));
                  }

                  plot_set_schnitt -> addElement(plot_elem_schnitt);
                  delete plot_elem_schnitt;
               }
            }

            /* --- Radiale Linien plotten ---*/
            for(i=0;i<num_pts;i++)
            {
               sprintf(plbuf_schnitt,"%s %d",PlotData_Schnitt,i+num_slices+b*(num_slices+num_pts));
               coDoVec2 *plot_elem_schnitt = new
                  coDoVec2(plbuf_schnitt,2*num_slices);
               if (plot_elem_schnitt->objectOk())
               {
                  plot_elem_schnitt -> getAddresses(&xpl_schnitt,&ypl_schnitt);
                  for(j=0;j<num_slices;j++)
                  {
                     xpl_sch[j]           =xyz_.x_ds[j][i];
                     ypl_sch[j]           =xyz_.y_ds[j][i];
                     xpl_sch[j+num_slices]=xyz_.x_ss[num_slices-j-1][i];
                     ypl_sch[j+num_slices]=xyz_.y_ss[num_slices-j-1][i];

                     xpl_schnitt[j]           =(float)((xpl_sch[j]*cos(b*(2.0*M_PI/zwis_.nlschaufel))-ypl_sch[j]*sin(b*(2.0*M_PI/zwis_.nlschaufel)))*sp);
                     ypl_schnitt[j]           =(float)(xpl_sch[j]*sin(b*(2.0*M_PI/zwis_.nlschaufel))+ypl_sch[j]*cos(b*(2.0*M_PI/zwis_.nlschaufel)));
                     xpl_schnitt[j+num_slices]=(float)((xpl_sch[j+num_slices]*cos(b*(2.0*M_PI/zwis_.nlschaufel))-ypl_sch[j+num_slices]*sin(b*(2.0*M_PI/zwis_.nlschaufel)))*sp);
                     ypl_schnitt[j+num_slices]=(float)(xpl_sch[j+num_slices]*sin(b*(2.0*M_PI/zwis_.nlschaufel))+ypl_sch[j+num_slices]*cos(b*(2.0*M_PI/zwis_.nlschaufel)));
                  }
                  plot_set_schnitt -> addElement(plot_elem_schnitt);
                  delete plot_elem_schnitt;
               }
            }
         }

         delete plot_set_schnitt;
      }

      if (zeige_laufrad==1)
      {

         /* --- Speicherung der Geometriedaten --- */

         n_coord=num_pts*num_slices;

         int numCoord=num_pts*num_slices*2*num_blades;
         int numPoly=((num_slices-1)*2*(num_pts-1))*num_blades + (num_slices-1)*num_blades;
         int numVert=4*numPoly;
         coDoPolygons *schaufeln = new coDoPolygons(Geom_obj_name, numCoord +2*num_pts*num_blades, numVert+4*(num_pts-1)*num_blades,numPoly+(num_pts-1)*num_blades);

         float *x_coord=NULL, *y_coord=NULL, *z_coord=NULL;
         int  *vl=NULL,*pl=NULL;
                
         if (schaufeln->objectOk())
         {
            schaufeln->getAddresses(&x_coord,&y_coord,&z_coord,&vl,&pl);
            
            memset (vl,0,numVert*sizeof(int));
            
            zl=0;
            for(i=0;i<num_slices;i++)
            {
               for(j=0;j<num_pts;j++)
               {
                  x_coord[zl]=xyz_.x_ds[i][j]*sp;
                  y_coord[zl]=xyz_.y_ds[i][j];
                  z_coord[zl]=xyz_.z_ds[i][j];
                  x_coord[zl+n_coord]=xyz_.x_ss[i][j]*sp;
                  y_coord[zl+n_coord]=xyz_.y_ss[i][j];
                  z_coord[zl+n_coord]=xyz_.z_ss[i][j];
                  zl++;
               }
            }

            /* --- Definfition der Schaufelkoordinaten fuer COVISE --- */
            for(b=1;b<num_blades;b++)
            {
               for(i=0;i<n_coord*2;i++)
               {
                  x_coord[i+b*(n_coord*2)]=(float)(x_coord[i]*cos(b*(2.0*M_PI/zwis_.nlschaufel))-y_coord[i]*sin(b*(2.0*M_PI/zwis_.nlschaufel)));
                  y_coord[i+b*(n_coord*2)]=(float)(x_coord[i]*sin(b*(2.0*M_PI/zwis_.nlschaufel))+y_coord[i]*cos(b*(2.0*M_PI/zwis_.nlschaufel)));
                  z_coord[i+b*(n_coord*2)]=z_coord[i];
               }
            }

            /* --- Definfition der Polygonliste fuer COVISE --- */
            for(i=0;i<numPoly;i++)
            {
               pl[i]=i*4;
            }

            /* --- Definfition der Vertexliste fuer COVISE --- */
            for(b=0;b<num_blades;b++)
            {
               for(i=0;i<(num_slices-1);i++)
               {
                  for(n=0;n<(num_pts-1);n++)
                  {
                     int index = n+i*(num_pts-1)+b*((num_slices-1)*2*(num_pts-1));
                     vl[pl[index]+0]=i*num_pts+n+b*(n_coord*2);
                     vl[pl[index]+1]=i*num_pts+n+b*(n_coord*2)+1;
                     vl[pl[index]+2]=(i+1)*num_pts+n+b*(n_coord*2)+1;
                     vl[pl[index]+3]=(i+1)*num_pts+n+b*(n_coord*2);
                     index = n+i*(num_pts-1)+b*((num_slices-1)*2*(num_pts-1))+((num_slices-1)*(num_pts-1));
                     vl[pl[index]+0]=i*num_pts+n+b*(n_coord*2)+n_coord;
                     vl[pl[index]+1]=i*num_pts+n+b*(n_coord*2)+n_coord+1;
                     vl[pl[index]+2]=(i+1)*num_pts+n+b*(n_coord*2)+n_coord+1;
                     vl[pl[index]+3]=(i+1)*num_pts+n+b*(n_coord*2)+n_coord;
                  }
               }
            }

            int base = pl[(num_pts-2)+(num_slices-2)*(num_pts-1)+b*((num_slices-1)*2*(num_pts-1))+((num_slices-1)*(num_pts-1))]+4;

            for(b=0;b<num_blades;b++)
            {
               for(i=0;i<(num_slices-1);i++)
               {
                  vl[base +4*i +4*b*(num_slices-1) +0] = i    *num_pts +num_pts-1   +b*(n_coord*2);
                  vl[base +4*i +4*b*(num_slices-1) +1] = (i+1)*num_pts +num_pts-1   +b*(n_coord*2);
                  vl[base +4*i +4*b*(num_slices-1) +2] = (i+1)*num_pts +num_pts-1 +n_coord +b*(n_coord*2);
                  vl[base +4*i +4*b*(num_slices-1) +3] = i    *num_pts +num_pts-1 +n_coord +b*(n_coord*2);
               }
            }

            schaufeln->addAttribute("vertexOrder","2");
            if(rotate)
            {
               schaufeln->addAttribute("ROTATE_POINT","0 0 0");
               schaufeln->addAttribute("ROTATE_VECTOR","0 0 1");
               sprintf(buf,"%f",(absol_.N/100.0));
               schaufeln->addAttribute("ROTATE_SPEED",buf);
            }
            schaufeln->addAttribute("MATERIAL","metal metal.30");
         }

         char b2[100];

         // ... MENUE-VR-Slider fuer Parameter Q ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\nQ\n%f\n%f\n%f\nNenndaten\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minQ,maxQ,absol_.Q);
         schaufeln->addAttribute("SLIDER0",buf);

         // ... MENUE-VR-Slider fuer Parameter mwa ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\nH\n%f\n%f\n%f\nNenndaten\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minH,maxH,absol_.H);
         schaufeln->addAttribute("SLIDER1",buf);

         // ... MENUE-VR-Slider fuer Parameter mwa ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\nn\n%f\n%f\n%f\nNenndaten\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minN,maxN,absol_.N);
         schaufeln->addAttribute("SLIDER2",buf);

         // ... MENUE-VR-Slider fuer Parameter mwa ...

         sprintf(buf,"M%s\n%s\n%s\nint\nz\n%ld\n%ld\n%ld\nNenndaten\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minn,maxn,zwis_.nlschaufel);
         schaufeln->addAttribute("SLIDER3",buf);

         // ... MENUE-VR-Slider fuer Parameter mwa ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\ndb2i\n%f\n%f\n%f\nWinkeluebertreibung\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,mindb2i,maxdb2i,wust_.db2i);
         schaufeln->addAttribute("SLIDER4",buf);

         // ... MENUE-VR-Slider fuer Parameter mwa ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\ndb2a\n%f\n%f\n%f\nWinkeluebertreibung\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,mindb2a,maxdb2a,wust_.db2a);
         schaufeln->addAttribute("SLIDER5",buf);

         // ... MENUE-VR-Slider fuer Parameter mwa ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\ndb1i\n%f\n%f\n%f\nWinkeluebertreibung\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,mindb1i,maxdb1i,wust_.db1i);
         schaufeln->addAttribute("SLIDER6",buf);

         // ... MENUE-VR-Slider fuer Parameter mwa ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\ndb1a\n%f\n%f\n%f\nWinkeluebertreibung\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,mindb1a,maxdb1a,wust_.db1a);
         schaufeln->addAttribute("SLIDER7",buf);

         // ... MENUE-VR-Slider fuer Parameter mwa ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\nUmschlingung\n%f\n%f\n%f\nSchaufelkontur\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minumschlingung,maxumschlingung,kont_.umschlingung);
         schaufeln->addAttribute("SLIDER8",buf);

         // ... MENUE-VR-Slider fuer Parameter max2 ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\nmax2\n%f\n%f\n%f\nSchaufelkontur\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minmax2,maxmax2,kont_.max2);
         schaufeln->addAttribute("SLIDER9",buf);

         // ... MENUE-VR-Slider fuer Parameter pe2i ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\npe2i\n%f\n%f\n%f\nSchaufelkontur\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minpe2i,maxpe2i,kont_.pe2i);
         schaufeln->addAttribute("SLIDER10",buf);

         // ... MENUE-VR-Slider fuer Parameter pe2a ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\npe2a\n%f\n%f\n%f\nSchaufelkontur\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minpe2a,maxpe2a,kont_.pe2a);
         schaufeln->addAttribute("SLIDER11",buf);

         // ... MENUE-VR-Slider fuer Parameter max1 ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\nmax1\n%f\n%f\n%f\nSchaufelkontur\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minmax1,maxmax1,kont_.max1);
         schaufeln->addAttribute("SLIDER12",buf);

         // ... MENUE-VR-Slider fuer Parameter pe1i ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\npe1i\n%f\n%f\n%f\nSchaufelkontur\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minpe1i,maxpe1i,kont_.pe1i);
         schaufeln->addAttribute("SLIDER13",buf);

         // ... MENUE-VR-Slider fuer Parameter pe1a ...

         sprintf(buf,"M%s\n%s\n%s\nfloat\npe1a\n%f\n%f\n%f\nSchaufelkontur\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minpe1a,maxpe1a,kont_.pe1a);
         schaufeln->addAttribute("SLIDER14",buf);

         // ... VR-Slider Intuitive Design ...

         // ... VR-Slider fuer Parameter mwi ...

         /*		sprintf(buf,"V%s\n%s\n%s\nfloat\nmwi\n%f\n%f\n%f\n"
                        ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
                        ,minmwi,maxmwi,maxw_.mwi);

                        sprintf(b2,"%d\n",2);
                        strcat(buf,b2);
                        sprintf(b2,"%d\n",num_pts);
                        strcat(buf,b2);
                        for(i=0;i<num_pts;i++)
                        {
                        sprintf(b2,"%f\n%f\n%f\n",xyz_.x_sl[0][i],xyz_.y_ss[0][i],xyz_.z_sl[0][i]);
                        strcat(buf,b2);
                        }
                        schaufeln->addAttribute("SLIDER15",buf);*/

         // ... VR-Slider fuer Parameter mwa ...

         /*		sprintf(buf,"V%s\n%s\n%s\nfloat\nmwa\n%f\n%f\n%f\n"
                        ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
                        ,minmwa,maxmwa,maxw_.mwa);

                        sprintf(b2,"%d\n",2);
                        strcat(buf,b2);
                        sprintf(b2,"%d\n",num_pts);
                        strcat(buf,b2);
                        for(i=0;i<num_pts;i++)
                        {
                        sprintf(b2,"%f\n%f\n%f\n",xyz_.x_sl[num_slices-1][i],xyz_.y_sl[num_slices-1][i],xyz_.z_sl[num_slices-1][i]);
                        strcat(buf,b2);
                        }
                        schaufeln->addAttribute("SLIDER16",buf);*/

         // ... VR-Slider fuer Parameter Umschlingung ...

         aussen=num_slices-1;
         sprintf(buf,"V%s\n%s\n%s\nfloat\nUmschlingung\n%f\n%f\n%f\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minumschlingung,maxumschlingung,kont_.umschlingung);

         sprintf(b2,"%d\n",2);
         strcat(buf,b2);
         sprintf(b2,"%d\n",20);
         strcat(buf,b2);
         float h0, hmin, hmax, hx, winkelmin, winkelmax,radius;
         h0=xyz_.z_ds[aussen][num_pts-1];
         hx=xyz_.z_ds[aussen][0];
         hmax=hx-((hx-h0)/kont_.umschlingung)*maxumschlingung;
         hmin=hx-((hx-h0)/kont_.umschlingung)*minumschlingung;

         winkelmin=(float)(2*M_PI/zwis_.nlschaufel)*minumschlingung;
         winkelmax=(float)(2*M_PI/zwis_.nlschaufel)*maxumschlingung;
         radius=zwis_.d2*530;
         for(i=0;i<20;i++)
         {
            sprintf(b2,"%f\n%f\n%f\n",radius*sin(winkelmin+((winkelmax-winkelmin)/19)*i),radius*cos(winkelmin+((winkelmax-winkelmin)/19)*i),hmin+((hmax-hmin)/19)*i);
            strcat(buf,b2);
         }
         schaufeln->addAttribute("SLIDER15",buf);

         // ... VR-Slider fuer Parameter max2 ...

         float winkel;
         sprintf(buf,"V%s\n%s\n%s\nfloat\nmax2\n%f\n%f\n%f\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minmax2,maxmax2,kont_.max2);
         sprintf(b2,"%d\n",2);
         strcat(buf,b2);
         sprintf(b2,"%d\n",num_slices);
         strcat(buf,b2);
         winkel=(float)(2*M_PI/zwis_.nlschaufel)*kont_.umschlingung;
         for(i=0;i<num_slices;i++)
         {
            sprintf(b2,"%f\n%f\n%f\n",lauf_.d_2[i]/2*sin(winkel),lauf_.d_2[i]/2*cos(winkel),xyz_.z_ss[i][num_pts-1]);
            strcat(buf,b2);
         }
         schaufeln->addAttribute("SLIDER16",buf);

         // ... VR-Slider fuer Parameter max1 ...

         sprintf(buf,"V%s\n%s\n%s\nfloat\nmax1\n%f\n%f\n%f\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minmax1,maxmax1,kont_.max1);
         sprintf(b2,"%d\n",2);
         strcat(buf,b2);
         sprintf(b2,"%d\n",num_slices);
         strcat(buf,b2);
         winkel=0;
         for(i=0;i<num_slices;i++)
         {
            sprintf(b2,"%f\n%f\n%f\n",lauf_.d_2[i]/2*sin(winkel),lauf_.d_2[i]/2*cos(winkel),xyz_.z_ss[i][0]);
            strcat(buf,b2);
         }
         schaufeln->addAttribute("SLIDER17",buf);

         // ... VR-Slider fuer Parameter pe2i ...

         int x60=13;
         sprintf(buf,"V%s\n%s\n%s\nfloat\npe2i\n%f\n%f\n%f\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minpe2i,maxpe2i,kont_.pe2i);
         sprintf(b2,"%d\n",2);
         strcat(buf,b2);
         sprintf(b2,"%d\n",num_pts-x60+10);
         strcat(buf,b2);

         // extrapolierte Punkte
         for(i=10;i>0;i--)
         {
            winkel=(float)(2*M_PI/zwis_.nlschaufel)*kont_.umschlingung*(1-kont_.pe2i*(10-i)/10);

            sprintf(b2,"%f\n%f\n%f\n",lauf_.d_2[0]/2*sin(winkel),lauf_.d_2[0]/2*cos(winkel)
                  ,xyz_.z_sl[0][num_pts-1]-(2*M_PI/zwis_.nlschaufel*kont_.umschlingung*kont_.pe1i)*lauf_.d_2[0]/2*tan(wink_.beta2_sch[0]*M_PI/180)*i/10);
            strcat(buf,b2);
         }

         // Punkte auf der Skelettlinie
         for(i=num_pts-1;i>x60-1;i--)
         {
            sprintf(b2,"%f\n%f\n%f\n",xyz_.x_sl[0][i],xyz_.y_sl[0][i],xyz_.z_sl[0][i]);
            strcat(buf,b2);
         }

         schaufeln->addAttribute("SLIDER18",buf);

         // ... VR-Slider fuer Parameter pe2a ...

         sprintf(buf,"V%s\n%s\n%s\nfloat\npe2a\n%f\n%f\n%f\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minpe2a,maxpe2a,kont_.pe2a);
         sprintf(b2,"%d\n",2);
         strcat(buf,b2);
         sprintf(b2,"%d\n",num_pts-x60+10);
         strcat(buf,b2);

         // extrapolierte Punkte
         for(i=10;i>0;i--)
         {
            winkel=(float)(2*M_PI/zwis_.nlschaufel)*kont_.umschlingung*(1-kont_.pe2a*(10-i)/10);

            sprintf(b2,"%f\n%f\n%f\n",
                  lauf_.d_2[num_slices-1]/2*sin(winkel),
                  lauf_.d_2[num_slices-1]/2*cos(winkel),
                  xyz_.z_sl[num_slices-1][num_pts-1]-(2*M_PI/zwis_.nlschaufel*kont_.umschlingung*kont_.pe1a)*lauf_.d_2[num_slices-1]/2*tan(wink_.beta2_sch[num_slices-1]*M_PI/180)*i/10);
            strcat(buf,b2);
         }

         // Punkte auf der Skelettlinie
         for(i=num_pts-1;i>x60-1;i--)
         {
            sprintf(b2,"%f\n%f\n%f\n",xyz_.x_sl[num_slices-1][i],xyz_.y_sl[num_slices-1][i],xyz_.z_sl[num_slices-1][i]);
            strcat(buf,b2);
         }

         schaufeln->addAttribute("SLIDER19",buf);

         // ... VR-Slider fuer Parameter pe1i ...

         int x40;
         x40=11;
         // Bestimmung des Maximalwerts auf der Skelettlinie
         winkel=(float)(2*M_PI/zwis_.nlschaufel*kont_.umschlingung*kont_.pe1i);
         i=0;
         while(atan(xyz_.x_sl[0][i]/xyz_.y_sl[0][i])<winkel)
         {
            i++;
         };
         x40=i;

         // Slideraufruf
         sprintf(buf,"V%s\n%s\n%s\nfloat\npe1i\n%f\n%f\n%f\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minpe1i,maxpe1i,kont_.pe1i);
         sprintf(b2,"%d\n",2);
         strcat(buf,b2);
         sprintf(b2,"%d\n",x40+11);
         strcat(buf,b2);

         // extrapolierte Punkte
         for(i=0;i<10;i++)
         {
            winkel=(float)(2*M_PI/zwis_.nlschaufel*kont_.umschlingung*kont_.pe1i*i/10);

            sprintf(b2,"%f\n%f\n%f\n",lauf_.d_2[0]/2*sin(winkel),lauf_.d_2[0]/2*cos(winkel)
                  ,xyz_.z_sl[0][0]+(2*M_PI/zwis_.nlschaufel*kont_.umschlingung*kont_.pe1i)*lauf_.d_2[0]/2*tan(wink_.beta1_sch[0]*M_PI/180)*(10-i)/10);
            strcat(buf,b2);
         }

         // Punkte auf der Skelettlinie
         for(i=0;i<x40+1;i++)
         {
            sprintf(b2,"%f\n%f\n%f\n",xyz_.x_sl[0][i],xyz_.y_sl[0][i],xyz_.z_sl[0][i]);
            strcat(buf,b2);
         }

         schaufeln->addAttribute("SLIDER20",buf);

         // ... VR-Slider fuer Parameter pe1a ...

         // Bestimmung des Maximalwerts auf der Skelettlinie
         winkel=(float)(2*M_PI/zwis_.nlschaufel*kont_.umschlingung*kont_.pe1a);
         i=0;
         while(atan(xyz_.x_sl[num_slices-1][i]/xyz_.y_sl[num_slices-1][i])<winkel)
         {
            i++;
         }
         x40=i;
         x40=11;
         // Slideraufruf
         sprintf(buf,"V%s\n%s\n%s\nfloat\npe1i\n%f\n%f\n%f\n"
               ,Covise::get_module(),Covise::get_instance(),Covise::get_host()
               ,minpe1a,maxpe1a,kont_.pe1a);
         sprintf(b2,"%d\n",2);
         strcat(buf,b2);
         sprintf(b2,"%d\n",x40+11);
         strcat(buf,b2);

         // extrapolierte Punkte
         for(i=0;i<10;i++)
         {
            winkel=(float)(2*M_PI/zwis_.nlschaufel*kont_.umschlingung*kont_.pe1a*i/10);

            sprintf(b2,"%f\n%f\n%f\n",
                  lauf_.d_2[num_slices-1]/2*sin(winkel),
                  lauf_.d_2[num_slices-1]/2*cos(winkel),
                  xyz_.z_sl[num_slices-1][0]+(2*M_PI/zwis_.nlschaufel*kont_.umschlingung*kont_.pe1a)*lauf_.d_2[num_slices-1]/2*tan(wink_.beta1_sch[num_slices-1]*M_PI/180)*(10-i)/10);
            strcat(buf,b2);
         }

         // Punkte auf der Skelettlinie
         for(i=0;i<x40+1;i++)
         {
            sprintf(b2,"%f\n%f\n%f\n",xyz_.x_sl[num_slices-1][i],xyz_.y_sl[num_slices-1][i],xyz_.z_sl[num_slices-1][i]);
            strcat(buf,b2);
         }

         schaufeln->addAttribute("SLIDER20",buf);

         /// Schaufelkranz

         zl=numCoord;

         if(schaufeln->objectOk())
         {
            for(j=0;j<num_pts;j++)
            {
               x_coord[zl]=xyz_.x_ds[aussen][j]*sp;
               y_coord[zl]=xyz_.y_ds[aussen][j];
               z_coord[zl]=xyz_.z_ds[aussen][j];
               x_coord[zl+num_pts]=xyz_.x_ss[aussen][j]*sp;
               y_coord[zl+num_pts]=xyz_.y_ss[aussen][j];
               z_coord[zl+num_pts]=xyz_.z_ss[aussen][j];
               zl++;
            }

            /* --- Definfition der Schaufelkoordinaten fuer COVISE --- */
            for(b=1;b<num_blades;b++)
            {
               for(i=0;i<num_pts*2;i++)
               {
                  x_coord[numCoord+i+b*(num_pts*2)]=(float)(x_coord[numCoord+i]*cos(b*(2.0*M_PI/zwis_.nlschaufel))-y_coord[numCoord+i]*sin(b*(2.0*M_PI/zwis_.nlschaufel)));
                  y_coord[numCoord+i+b*(num_pts*2)]=(float)(x_coord[numCoord+i]*sin(b*(2.0*M_PI/zwis_.nlschaufel))+y_coord[numCoord+i]*cos(b*(2.0*M_PI/zwis_.nlschaufel)));
                  z_coord[numCoord+i+b*(num_pts*2)]=z_coord[numCoord+i];
               }
            }
         }

         /* --- Definfition der Polygonliste fuer COVISE --- */
         for(i=0;i<(num_pts-1)*num_blades;i++)
         {
            pl[numPoly+i]=numVert+i*4;
         }

         /* --- Definfition der Vertexliste fuer COVISE --- */
         for(int i=0; i<num_blades; i++)
         {
            for (int j=0; j<num_pts-1;j++)
            {
               vl[numVert+i*(num_pts-1)*4+j*4]=numCoord+j+i*num_pts*2;
               vl[numVert+i*(num_pts-1)*4+j*4+1]=numCoord+num_pts+j+i*num_pts*2;
               vl[numVert+i*(num_pts-1)*4+j*4+2]=numCoord+num_pts+j+1+i*num_pts*2;
               vl[numVert+i*(num_pts-1)*4+j*4+3]=numCoord+j+1+i*num_pts*2;
            }
         }

         delete schaufeln;

      }

      if (zeige_nabe==1)
      {
         anznabe=fona_.anznabe;
         enderad=fona_.r_nabe[anznabe-1]*zwis_.di_da;
         int numCoordNabe= (num_kreis+1)*anznabe;
         int numPolNabe= (num_kreis)*(anznabe-1);
         int numVertNabe= 4*numPolNabe;
         nabe_strip = new coDoPolygons(Nabe_obj_name, numCoordNabe+(num_kreis+2), numVertNabe+3*num_kreis ,numPolNabe+num_kreis);

         int *plnabe, *vlnabe;
         float *x_conabe, *y_conabe, *z_conabe;
         if (nabe_strip->objectOk())
         {
            //Covise::sendWarning("Polygone OK!");
            nabe_strip->getAddresses(&x_conabe,&y_conabe,&z_conabe,&vlnabe,&plnabe);

            zl=0;
            for(i=0;i<anznabe;i++)
            {
               for(j=0;j<(num_kreis+1);j++)
               {
                  x_conabe[zl]=1000*zwis_.di_da*zwis_.d2*fona_.r_nabe[i]*(float)cos(j*2.0*PI/num_kreis);
                  y_conabe[zl]=1000*zwis_.di_da*zwis_.d2*fona_.r_nabe[i]*(float)sin(j*2.0*PI/num_kreis);
                  z_conabe[zl]=1000*zwis_.di_da*zwis_.d2*fona_.z_nabe[i];
                  zl++;
               }
            }

            for(i=0;i<numPolNabe;i++)
            {
               plnabe[i]=i*4;
            }

            for(int i=0; i<(anznabe-1); i++)
            {
               for (int j=0; j<(num_kreis);j++)
               {
                  vlnabe[i*4*num_kreis+j*4]   = j+i*(num_kreis+1);
                  vlnabe[i*4*num_kreis+j*4+1] = num_kreis+j+1+i*(num_kreis+1);
                  vlnabe[i*4*num_kreis+j*4+2] = num_kreis+j+2+i*(num_kreis+1);
                  vlnabe[i*4*num_kreis+j*4+3] = j+1+i*(num_kreis+1);
               }
            }
            // Naben Ende

            zl=numCoordNabe;
            for(j=0;j<(num_kreis+1);j++)
            {
               x_conabe[zl]=1000*zwis_.d2*enderad*(float)cos(j*2.0*PI/num_kreis);
               y_conabe[zl]=1000*zwis_.d2*enderad*(float)sin(j*2.0*PI/num_kreis);
               z_conabe[zl]=1000*zwis_.d2*zwis_.di_da*fona_.z_nabe[anznabe-1];
               zl++;
            }
            x_conabe[zl]=0;
            y_conabe[zl]=0;
            z_conabe[zl]=1000*zwis_.d2*zwis_.di_da*fona_.z_nabe[anznabe-1];

            for(int i=0;i<(num_kreis);i++)
            {
               plnabe[numPolNabe+i]=numVertNabe+i*3;
            }

            for (j=0; j<(num_kreis);j++)
            {
               vlnabe[plnabe[numPolNabe+j]]   = numCoordNabe+j;
               vlnabe[plnabe[numPolNabe+j]+1] = zl;
               vlnabe[plnabe[numPolNabe+j]+2] = numCoordNabe+j+1;
            }
         }

         nabe_strip->addAttribute("vertexOrder","2");
         nabe_strip->addAttribute("MATERIAL","metal metal.30");
         delete nabe_strip;

      }

      if (zeige_kranz==1)
      {
         num_kranz=(int)(num_kreis*kranzwin/360);
         int anzahl=(int)((fokr_.anzkranz-1)/2)-1;

         int anfang=4*(num_kranz)*(fokr_.anzkranz-1);
         kranz_strip = new coDoPolygons(Kranz_obj_name, (num_kranz+1)*fokr_.anzkranz, 4*((num_kranz)*(fokr_.anzkranz-1)+2*anzahl) ,(num_kranz)*(fokr_.anzkranz-1)+2*anzahl);

         if (kranz_strip->objectOk())
         {
            //Covise::sendWarning("Polygone OK!");
            kranz_strip->getAddresses(&x_cokranz,&y_cokranz,&z_cokranz,&vlkranz,&plkranz);

            zl=0;
            for(i=0;i<fokr_.anzkranz;i++)
            {
               for(j=0;j<(num_kranz+1);j++)
               {
                  x_cokranz[zl]=1000*zwis_.d2*fokr_.r_kranz[i]*(float)cos((j*kranzwin/num_kranz+(-kranzwin/2-90))/180.0*PI);
                  y_cokranz[zl]=1000*zwis_.d2*fokr_.r_kranz[i]*(float)sin((j*kranzwin/num_kranz+(-kranzwin/2-90))/180.0*PI);
                  z_cokranz[zl]=1000*zwis_.d2*fokr_.z_kranz[i];
                  zl++;
               }
            }

            for(i=0;i<((num_kranz)*(fokr_.anzkranz-1)+2*anzahl);i++)
            {
               plkranz[i]=i*4;
            }

            for(i=0; i<(fokr_.anzkranz-1); i++)
            {
               for (j=0; j<(num_kranz);j++)
               {
                  vlkranz[i*4*num_kranz+j*4]   = j+i*(num_kranz+1);
                  vlkranz[i*4*num_kranz+j*4+1] = num_kranz+j+1+i*(num_kranz+1);
                  vlkranz[i*4*num_kranz+j*4+2] = num_kranz+j+2+i*(num_kranz+1);
                  vlkranz[i*4*num_kranz+j*4+3] = j+1+i*(num_kranz+1);
               }
            }
            for (j=0; j<(anzahl);j++)
            {
               vlkranz[anfang+j*4]   = j*(num_kranz+1);
               vlkranz[anfang+j*4+1] = (fokr_.anzkranz-j-2)*(num_kranz+1);
               vlkranz[anfang+j*4+2] = (fokr_.anzkranz-j-3)*(num_kranz+1);
               vlkranz[anfang+j*4+3] = (j+1)*(num_kranz+1);
            }
            anfang=anfang+4*anzahl;
            for (j=0; j<(anzahl);j++)
            {
               vlkranz[anfang+j*4]   = j*(num_kranz+1)+num_kranz;
               vlkranz[anfang+j*4+1] = (fokr_.anzkranz-j-2)*(num_kranz+1)+num_kranz;
               vlkranz[anfang+j*4+2] = (fokr_.anzkranz-j-3)*(num_kranz+1)+num_kranz;
               vlkranz[anfang+j*4+3] = (j+1)*(num_kranz+1)+num_kranz;
            }

         }
         kranz_strip->addAttribute("vertexOrder","2");
         kranz_strip->addAttribute("MATERIAL","metal metal.30");
         delete kranz_strip;
      }

      float x_temp,y_temp;
      float x_eintr,y_eintr;
      float phi_temp;
      float alpha_0;
      float g = 9.81f;

      alpha_0=atan(absol_.N*absol_.Q*lp/60/g/absol_.H/leitg_.b0);
      num_leit=leitg_.Nleit;
      //num_leit=16;
      num_leit_pts=38;
      int anzahl= (int)(num_leit_pts/2)-1;
      int anfang_oben  = 4*num_leit*(num_leit_pts-1);
      int anfang_unten = anfang_oben+4*anzahl*num_leit;
      zeige_leit=1;
      if (zeige_leit==1)
      {

         leit_strip = new coDoPolygons(Leit_obj_name, num_leit*(2*num_leit_pts), 4*((num_leit)*(num_leit_pts-1+2*anzahl)) ,(num_leit)*(num_leit_pts-1+2*anzahl));
         if (leit_strip->objectOk())
         {

            leit_strip->getAddresses(&x_leit,&y_leit,&z_leit,&vl_leit,&pl_leit);

            zl=0;

            for(i=0;i<num_leit;i++)
            {
               for(j=0;j<2;j++)
               {
                  for(int k=0;k<(num_leit_pts);k++)
                  {
                     phi_temp=2.0f*i/num_leit*PI;
                     x_eintr=1000*leitg_.l0*leitg_.leit_dr*sin(phi_temp+alpha_0);
                     y_eintr=-1000*leitg_.l0*leitg_.leit_dr*cos(phi_temp+alpha_0);
                     x_temp=(float)(x_eintr-leitg_.l0*(leits_.x_leit_prof[k]*10.0*sin(phi_temp+alpha_0)-leits_.y_leit_prof[k]*10.0*cos(phi_temp+alpha_0)));
                     y_temp=(float)(y_eintr+leitg_.l0*(leits_.y_leit_prof[k]*10.0*sin(phi_temp+alpha_0)+leits_.x_leit_prof[k]*10.0*cos(phi_temp+alpha_0)));

                     x_leit[zl] =(500*leitg_.d0*cos(phi_temp)+x_temp)*sp;
                     y_leit[zl] =500*leitg_.d0*sin(phi_temp)+y_temp;
                     z_leit[zl] =1000*leitg_.b0*j;
                     zl++;
                  }
               }
            }

            for(i=0;i<((num_leit)*(num_leit_pts-1))+(num_leit*(num_leit_pts-2));i++)
            {
               pl_leit[i]=i*4;
            }

            for(i=0; i<num_leit; i++)
            {
               for (j=0; j<(num_leit_pts-1);j++)
               {
                  vl_leit[i*4*(num_leit_pts-1)+j*4]   = j                + i*(2*num_leit_pts);
                  vl_leit[i*4*(num_leit_pts-1)+j*4+1] = j+num_leit_pts   + i*(2*num_leit_pts);
                  vl_leit[i*4*(num_leit_pts-1)+j*4+2] = j+num_leit_pts+1 + i*(2*num_leit_pts);
                  vl_leit[i*4*(num_leit_pts-1)+j*4+3] = j+1              + i*(2*num_leit_pts);
               }
            }

            for(int i=0; i<num_leit; i++)
            {

               // Oberseite Leitschaufeln
               for (j=0; j<(anzahl);j++)
               {
                  vl_leit[anfang_oben+i*4*(anzahl)+j*4]   = j                  + i*(2*num_leit_pts);
                  vl_leit[anfang_oben+i*4*(anzahl)+j*4+1] = num_leit_pts-j-1   + i*(2*num_leit_pts);
                  vl_leit[anfang_oben+i*4*(anzahl)+j*4+2] = num_leit_pts-j-2   + i*(2*num_leit_pts);
                  vl_leit[anfang_oben+i*4*(anzahl)+j*4+3] = j+1                + i*(2*num_leit_pts);
               }

               // Unterseite Leitschaufeln
               for (int j=0; j<(anzahl);j++)
               {
                  vl_leit[anfang_unten+i*4*(anzahl)+j*4]   = j                  + i*(2*num_leit_pts) + num_leit_pts;
                  vl_leit[anfang_unten+i*4*(anzahl)+j*4+1] = num_leit_pts-j-1   + i*(2*num_leit_pts) + num_leit_pts;
                  vl_leit[anfang_unten+i*4*(anzahl)+j*4+2] = num_leit_pts-j-2   + i*(2*num_leit_pts) + num_leit_pts;
                  vl_leit[anfang_unten+i*4*(anzahl)+j*4+3] = j+1                + i*(2*num_leit_pts) + num_leit_pts;
               }
            }
         }
         leit_strip->addAttribute("vertexOrder","2");
         leit_strip->addAttribute("MATERIAL","metal metal.30");
         delete leit_strip;
      }

      if (netzstart==1)
      {
         netzstart=0;
         Covise::update_boolean_param ("Netzgenerierung" , netzstart);
         Covise::sendWarning("Netzgenerierung gestartet!");
         ainf_.lese_xyz_dat=0;
         netz_aktuell=1;

         /* --- Netzgenerator starten ! --- */
         //schreibe_randbedingung_();
         axnet_();

         Covise::sendWarning("Netzgenerierung abgeschlossen!");
      }

      if (netzlesen==1)
      {
         netzlesen=0;
         Covise::update_boolean_param ("Netz oeffnen" , netzlesen);
         strcpy(buf,"Netz wird geoeffnet!");
         Covise::sendWarning("Netz wird geoeffnet!");
         netz_aktuell=1;

         /* --- Netz oeffnen ! --- */
         netzoeffnen_();

         Covise::sendWarning("Netz geoeffnet!");

      }

      if ((zeige_netz==1)&&(netz_aktuell==1))
      {

         mesh = new coDoUnstructuredGrid(Mesh, axel_.a_3Del, (axel_.a_3Del*8), (akno_.an_kno*akno_.bi), 1);
         if (mesh->objectOk())
         {
            mesh->getAddresses(&el_netz,&vl_netz,&x_co_netz,&y_co_netz,&z_co_netz);
            mesh->getTypeList(&tl);

            for(i=0;i<axel_.a_3Del;i++)
            {
               tl[i]=7;
            }
            for(i=0;i<(akno_.an_kno*akno_.bi);i++)
            {
               x_co_netz[i]=anet_.f[akno_.ixkn-1+i];
               y_co_netz[i]=anet_.f[akno_.iykn-1+i];
               z_co_netz[i]=anet_.f[akno_.izkn-1+i];
            }
            for(i=0;i<axel_.a_3Del;i++)
            {
               el_netz[i]=i*8;
            }
            for(i=0; i<(axel_.a_3Del); i++)
            {
               vl_netz[8*i]   = (anet_.e[axel_.iel1-1+i])-1;
               vl_netz[8*i+1] = (anet_.e[axel_.iel2-1+i])-1;
               vl_netz[8*i+2] = (anet_.e[axel_.iel3-1+i])-1;
               vl_netz[8*i+3] = (anet_.e[axel_.iel4-1+i])-1;
               vl_netz[8*i+4] = (anet_.e[axel_.iel5-1+i])-1;
               vl_netz[8*i+5] = (anet_.e[axel_.iel6-1+i])-1;
               vl_netz[8*i+6] = (anet_.e[axel_.iel7-1+i])-1;
               vl_netz[8*i+7] = (anet_.e[axel_.iel8-1+i])-1;
            }
            delete mesh;

            /* --- 2D-Plot Schnitt anfertigen --- */
            /*char *PlotData_netz_nabe;
            float *xplnetz,*yplnetz;

            coDoSet *netz_plot_nabe_set;
            char plbuf_netz_nabe[3000];

            // --- Legende ---

            strcpy(plbuf_netz_nabe,"AUTOSCALE\n");
            strcat(plbuf_netz_nabe,"TITLE \"Rechennetz an der Nabe\"\n");
            strcat(plbuf_netz_nabe,"XAXIS LABEL \"");
            strcat(plbuf_netz_nabe,"Umfang");
            strcat(plbuf_netz_nabe,"\"\n");
            strcat(plbuf_netz_nabe,"YAXIS LABEL \"");
            strcat(plbuf_netz_nabe,"Z-Achse");
            strcat(plbuf_netz_nabe,"\"\n");

            PlotData_netz_nabe = Covise::get_object_name("2dplotnetznabe");

            if (PlotData_netz_nabe!=NULL)
            {
            netz_plot_nabe_set = new coDoSet(PlotData_netz_nabe,SET_CREATE);
            }

            netz_plot_nabe_set -> addAttribute("COMMANDS",plbuf_netz_nabe);

            for(i=0;i<20;i++)//aplo_.seed;i++)
            {
            sprintf(plbuf_netz_nabe,"%s %d",Plotdata_netz_nabe,i);

            coDoVec2 *plot_elem_netz_nabe = new
            coDoVec2(plbuf_netz_nabe,10);

            if (plot_elem_netz_nabe->objectOk())
            {
            plot_elem_netz_nabe -> getAddresses(&xplnetz,&yplnetz);
            for(j=0;j<4;i++)
            {
            xplnetz[j]  =i*j;//anet_.f[aplo_.ixr+i*4+j-1]-1;
            yplnetz[j]  =i*j;//anet_.f[aplo_.iyr+i*4+j-1]-1;
            xplnetz[j+5]=i*j;//anet_.f[aplo_.ixr+i*4+j]-1;
            yplnetz[j+5]=i*j;//anet_.f[aplo_.iyr+i*4+j]-1;
            }
            xplnetz[4]  =xplnetz[0];
            yplnetz[4]  =yplnetz[0];
            xplnetz[9]  =xplnetz[5];
            yplnetz[9]  =yplnetz[5];

            netz_plot_nabe_set -> addElement(plot_elem_netz_nabe);
            delete plot_elem_netz_nabe;
            }

            }

            delete netz_plot_nabe_set;*/

         }

      }

   }

}


void Application::parameter(void *)
{

   const char                *pname;
   //char buf[600];
   int i;
   pname = Covise::get_reply_param_name();

   Covise::hide_param("Lastpunkt");
   Covise::hide_param("Kranzwinkel");
   Covise::hide_param("Schaufelzahl");
   Covise::hide_param("Drehsinn");
   Covise::hide_param("Rotieren");
   Covise::hide_param("Netz");
   Covise::hide_param("Kranz");
   Covise::hide_param("Nabe");
   Covise::hide_param("Laufrad");

   Covise::hide_param("z");
   Covise::hide_param("D1/D2");
   Covise::hide_param("D2");
   Covise::hide_param("n");
   Covise::hide_param("H");
   Covise::hide_param("Q");

   Covise::hide_param("Umschlingung");
   Covise::hide_param("max2");
   Covise::hide_param("pe2i");
   Covise::hide_param("pe2a");
   Covise::hide_param("max1");
   Covise::hide_param("pe1i");
   Covise::hide_param("pe1a");

   Covise::hide_param("hk_dicke_a");
   Covise::hide_param("hk_dicke_i");
   Covise::hide_param("d_strich_a");
   Covise::hide_param("dicke_a");
   Covise::hide_param("dicke_i");

   Covise::hide_param("versch_a");
   Covise::hide_param("versch_i");

   Covise::hide_param("mwa");
   Covise::hide_param("mwi");

   Covise::hide_param("Kugelradius");

   Covise::hide_param("uml_r");
   Covise::hide_param("uml_z");

   Covise::hide_param("db1a");
   Covise::hide_param("db1i");
   Covise::hide_param("db2a");
   Covise::hide_param("db2i");

   Covise::hide_param("Dateiausgabe");
   Covise::hide_param("prototyp.dat");
   Covise::hide_param("schaufel_xyz.dat");
   Covise::hide_param("La-Winkel.dat");
   Covise::hide_param("Stroe-Winkel.dat");
   Covise::hide_param("oeffnen schaufel_xyz.dat");
   Covise::hide_param("oeffnen prototyp.dat");

   Covise::hide_param("Netzgenerierung");
   Covise::hide_param("Netz speichern");
   Covise::hide_param("Netz oeffnen");

   if (strcmp("Hauptmenue", pname) == 0)
   {
      Covise::get_reply_choice(&i);
      switch(i)
      {
         case 1:

            Covise::show_param("Lastpunkt");
            Covise::show_param("Kranzwinkel");
            Covise::show_param("Schaufelzahl");
            Covise::show_param("Drehsinn");
            Covise::show_param("Rotieren");
            Covise::show_param("Netz");
            Covise::show_param("Kranz");
            Covise::show_param("Nabe");
            Covise::show_param("Laufrad");
            break;

         case 2:
            Covise::show_param("Q");
            Covise::show_param("H");
            Covise::show_param("n");
            Covise::show_param("D2");
            Covise::show_param("D1/D2");
            Covise::show_param("z");
            break;

         case 3:
            Covise::show_param("Umschlingung");
            Covise::show_param("max2");
            Covise::show_param("pe2i");
            Covise::show_param("pe2a");
            Covise::show_param("max1");
            Covise::show_param("pe1i");
            Covise::show_param("pe1a");
            break;

         case 4:
            Covise::show_param("dicke_i");
            Covise::show_param("dicke_a");
            Covise::show_param("d_strich_a");
            Covise::show_param("hk_dicke_i");
            Covise::show_param("hk_dicke_a");
            break;

         case 5:
            Covise::show_param("versch_i");
            Covise::show_param("versch_a");
            break;

         case 6:
            Covise::show_param("db2i");
            Covise::show_param("db2a");
            Covise::show_param("db1i");
            Covise::show_param("db1a");
            break;

         case 7:
            Covise::show_param("mwa");
            Covise::show_param("mwi");
            break;

         case 8:
            Covise::show_param("Kugelradius");
            break;

         case 9:
            Covise::show_param("uml_r");
            Covise::show_param("uml_z");
            break;

         case 10:
            Covise::show_param("Netz oeffnen");
            Covise::show_param("Netz speichern");
            Covise::show_param("Netzgenerierung");
            break;

         case 11:
            Covise::show_param("Dateiausgabe");
            Covise::show_param("prototyp.dat");
            Covise::show_param("schaufel_xyz.dat");
            Covise::show_param("La-Winkel.dat");
            Covise::show_param("Stroe-Winkel.dat");
            Covise::show_param("oeffnen schaufel_xyz.dat");
            Covise::show_param("oeffnen prototyp.dat");
            break;

      }

   }

}
