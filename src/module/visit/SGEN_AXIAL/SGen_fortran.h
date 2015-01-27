/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

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
#include <unistd.h>

extern "C" {
void geogen_();
void axnet_();
void netzoeffnen_();
void eingabe_xyz_();
void schreibe_randbedingung_();
void rechne_kugel_();
}

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
} xyz_;

extern struct abwik
{
    float x_sl_ab[11][19];
    float y_sl_ab[11][19];
    float x_ds_ab[11][19];
    float y_ds_ab[11][19];
    float x_ss_ab[11][19];
    float y_ss_ab[11][19];
} abwik_;

extern struct nenn
{
    float Q;
    float H;
    float N;
} absol_;

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
} io_;

extern struct zwis
{
    float d2;
    float di_da;
    int nlschaufel;
} zwis_;

extern struct leitg
{
    int Nleit;
    float b0;
    float d0;
    float l0;
    float leit_dr;
} leitg_;

extern struct nab
{
    int anznabe;
    float r_nabe[30];
    float z_nabe[30];
    float d2_kugel;
} fona_;

extern struct kra
{
    int anzkranz;
    float r_kranz[30];
    float z_kranz[30];
} fokr_;

extern struct wust
{
    float db2i;
    float db2a;
    float db1i;
    float db1a;
} wust_;

extern struct maxw
{
    float mwi;
    float mwa;
} maxw_;

extern struct kont
{
    float umschlingung;
    float max1;
    float max2;
    float pe1i;
    float pe1a;
    float pe2i;
    float pe2a;
} kont_;

extern struct lauf
{
    float d_1[11];
    float d_2[11];
} lauf_;

extern struct wink
{
    float beta1_sch[11];
    float beta2_sch[11];
} wink_;

extern struct verz
{
    char datei_prototyp[200];
    char datei_zentral_file[200];
    char datei_daten[200];
    char datei_dicke[200];
    char datei_profil[200];
    char datei_steuerfile[200];
    char datei_profil_neu[200];
    char datei_output[200];
    char datei_zwischen_file[200];
    char datei_meko[200];
    char datei_spezifisch[200];
    char datei_La_Winkel[200];
    char datei_Stroe_Winkel[200];
    char datei_Ge_Dreiecke[200];
    char datei_schaufel_xyz[200];
    char datei_zwischen_prof[200];
    char datei_Batrekhy_prof[200];
    char datei_randbedingung[200];
    char datei_leitprofil[200];
} verz_;

extern struct ver2
{
    char datei_ax_stf[200];
    char datei_kart3d_geo[200];
    char datei_kart3d_rb[200];
} ver2_;

extern struct lesn
{
    int lesenok;
} lesn_;

extern struct dick
{
    float dicke_i;
    float dicke_a;
    float d_strich_a;
} dick_;

extern struct prof
{
    float versch_i;
    float versch_a;
} prof_;

extern struct akno
{
    int an_kno;
    int bi;
    int ixkn;
    int iykn;
    int izkn;
} akno_;

extern struct anet
{
    float f[1500000];
    int e[3000000];
} anet_;

extern struct ainf
{
    int lese_xyz_dat;
    int netz_speichern;
} ainf_;

extern struct awrb
{
    int an3_wr;
    int iwrb1;
    int iwrb2;
    int iwrb3;
    int iwrb4;
} awrb_;
extern struct akbi
{
    int an3_kb;
    int ikbi1;
    int ikbi2;
    int ikbi3;
    int ikbi4;
} akbi_;
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
} axel_;
extern struct aplo
{
    int ixr;
    int iyr;
    int seed;
} aplo_;
extern struct leits
{
    float x_leit_prof[38];
    float y_leit_prof[38];
} leits_;

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
    static void parameterCallback(void *userData, void *callbackData);

    float minH, maxH, minQ, maxQ, minN, maxN, mind2, maxd2;
    float mindi_da, maxdi_da, mindb2i, maxdb2i;
    float mindb1i, maxdb1i, mindb2a, maxdb2a, mindb1a, maxdb1a, minmwi, maxmwi;
    float minmwa, maxmwa;
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
    float minversch_i, maxversch_i;
    float minversch_a, maxversch_a;
    float minlp, maxlp, lp;

    float nlschaufel_alt, q_alt, h_alt, n_alt, d2_alt, di_da_alt;
    float umschlingung_alt, max2_alt, pe2i_alt, pe2a_alt;
    float max1_alt, pe1i_alt, pe1a_alt;
    float dicke_i_alt, dicke_a_alt, d_strich_a_alt;
    float versch_i_alt, versch_a_alt;
    float db2i_alt, db2a_alt, db1i_alt, db1a_alt, mwi_alt, mwa_alt;
    float mind2_kugel, maxd2_kugel, d2_kugel_alt;
    float sp;

    int minn, maxn, minkranxwin, maxkranxwin;
    int minschaufelzahl, maxschaufelzahl;

    int rotate, drehsinn, drehsinn_alt, kranzwin;
    int zeige_laufrad, zeige_nabe, zeige_kranz, zeige_netz, schaufelzahl, schaufelzahl_alt;
    int zeige_leit;
    int netz_aktuell;
    int netzstart, netzlesen;
    int xyz_lesen, proto_lesen;
    int neue_geometrie;

public:
    Application(int argc, char *argv[])

    {

        Covise::set_module_description("IHS Schaufelgenerator");

        /* --- Ausgabe-Geometrie --- */
        Covise::add_port(OUTPUT_PORT, "geom", "coDoPolygons", "Schaufelgeometrie");
        Covise::add_port(OUTPUT_PORT, "nabe", "coDoPolygons", "Nabe");
        Covise::add_port(OUTPUT_PORT, "kranz", "coDoPolygons", "Kranz");
        Covise::add_port(OUTPUT_PORT, "leit", "coDoPolygons", "Leitschaufeln");
        Covise::add_port(OUTPUT_PORT, "netz", "coDoUnstructuredGrid", "Netz");
        Covise::add_port(OUTPUT_PORT, "eintritt", "coDoLines", "RB_Eintritt");

        Covise::add_port(OUTPUT_PORT, "2dplotkonform", "Set_Vec2", "Plotdata");
        Covise::add_port(OUTPUT_PORT, "2dplotschnitt", "Set_Vec2", "Plotdata_schnitt");
        Covise::add_port(OUTPUT_PORT, "2dplotnetznabe", "Set_Vec2", "Plotdata_netz_nabe");
        Covise::add_port(OUTPUT_PORT, "2dplotkanalnabe", "Set_Vec2", "Plotdata_kanalnabe");
        Covise::add_port(OUTPUT_PORT, "2dplotkanalkranz", "Set_Vec2", "Plotdata_kanalkranz");

        Covise::add_port(PARIN, "Datenverzeichnis", "Browser", "Datenverzeichnis");
        Covise::set_port_default("Datenverzeichnis", "/mnt/fs2/studenten/kaps/KIEB_daten/ *.dat");

        /* --- Eingabe-Parameter --- */

        Covise::add_port(PARIN, "Hauptmenue", "Choice", "Auswahl Hauptmenue");
        Covise::set_port_default("Hauptmenue", "2 Darstellung Nenndaten Schaufelkontur Schaufeldicken Profilverschiebung Winkeluebertreibung Woelbung Nabe Kranz Netz Datei");
        Covise::set_port_immediate("Hauptmenue", 1);

        Covise::add_port(PARIN, "Laufrad", "Boolean", "Laufrad, an/aus");
        Covise::set_port_default("Laufrad", "TRUE");

        Covise::add_port(PARIN, "Nabe", "Boolean", "Nabe, an/aus");
        Covise::set_port_default("Nabe", "TRUE");

        Covise::add_port(PARIN, "Kranz", "Boolean", "Kranz, an/aus");
        Covise::set_port_default("Kranz", "TRUE");

        Covise::add_port(PARIN, "Netz", "Boolean", "Netz, an/aus");
        Covise::set_port_default("Netz", "TRUE");

        Covise::add_port(PARIN, "Rotieren", "Boolean", "Rotieren, an/aus");
        Covise::set_port_default("Rotieren", "FALSE");

        Covise::add_port(PARIN, "Drehsinn", "Boolean", "Drehsinn, an/aus");
        Covise::set_port_default("Drehsinn", "FALSE");

        Covise::add_port(PARIN, "Schaufelzahl", "Slider", "value for schaufelzahl");
        Covise::set_port_default("Schaufelzahl", "1 10 3");
        minschaufelzahl = 1;
        maxschaufelzahl = 10;
        schaufelzahl = 3;

        Covise::add_port(PARIN, "Kranzwinkel", "Slider", "value for Winkel");
        Covise::set_port_default("Kranzwinkel", "10 360 180");
        minkranxwin = 10;
        maxkranxwin = 360;
        kranzwin = 180;

        Covise::add_port(PARIN, "Lastpunkt", "Slider", "value for Lastpunkt");
        Covise::set_port_default("Lastpunkt", "0.10 1.50 1.00");
        minlp = 0.10;
        maxlp = 1.50;
        lp = 1.00;

        Covise::add_port(PARIN, "Q", "Slider", "value for Q");
        Covise::set_port_default("Q", "1.0 10.0 4.7");
        minQ = 1.0;
        maxQ = 10.0;
        absol_.Q = 4.7;

        Covise::add_port(PARIN, "H", "Slider", "value for H");
        Covise::set_port_default("H", "1.0 10.0 2.7");
        minH = 1.0;
        maxH = 10.0;
        absol_.H = 2.7;

        Covise::add_port(PARIN, "n", "Slider", "value for n");
        Covise::set_port_default("n", "150.0 500.0 220.0");
        minN = 150.0;
        maxN = 500.0;
        absol_.N = 220.0;

        Covise::add_port(PARIN, "D2", "Slider", "value for D2");
        Covise::set_port_default("D2", "1.0 2.0 1.28");
        mind2 = 1.0;
        maxd2 = 2.0;
        zwis_.d2 = 1.28;

        Covise::add_port(PARIN, "D1/D2", "Slider", "value for D2izuD2a");
        Covise::set_port_default("D1/D2", "0.20 1.00 0.4375");
        mindi_da = 0.2;
        maxdi_da = 1.0;
        zwis_.di_da = 0.4375;

        Covise::add_port(PARIN, "z", "Slider", "Number of blades");
        Covise::set_port_default("z", "1 20 3");

        Covise::add_port(PARIN, "Umschlingung", "Slider", "value for Umschlingung");
        Covise::set_port_default("Umschlingung", "0.5 1.2 0.9");
        minumschlingung = 0.5;
        maxumschlingung = 1.2;
        kont_.umschlingung = 0.9;

        Covise::add_port(PARIN, "max1", "Slider", "value for max1");
        Covise::set_port_default("max1", "0.0 1.0 0.35");
        minmax1 = 0.0;
        maxmax1 = 1.0;
        kont_.max1 = 0.35;

        Covise::add_port(PARIN, "max2", "Slider", "value for max2");
        Covise::set_port_default("max2", "0.0 1.0 0.4");
        minmax1 = 0.0;
        maxmax1 = 1.0;
        kont_.max2 = 0.4;

        Covise::add_port(PARIN, "pe1i", "Slider", "value for pe1i");
        Covise::set_port_default("pe1i", "0.0 0.4 0.15");
        minpe1i = 0.0;
        maxpe1i = 0.4;
        kont_.pe1i = 0.15;

        Covise::add_port(PARIN, "pe1a", "Slider", "value for pe1a");
        Covise::set_port_default("pe1a", "0.0 0.4 0.1");
        minpe1a = 0.0;
        maxpe1a = 0.4;
        kont_.pe1a = 0.1;

        Covise::add_port(PARIN, "pe2i", "Slider", "value for pe2i");
        Covise::set_port_default("pe2i", "0.0 0.4 0.08");
        minpe2i = 0.0;
        maxpe2i = 0.4;
        kont_.pe2i = 0.08;

        Covise::add_port(PARIN, "pe2a", "Slider", "value for pe2a");
        Covise::set_port_default("pe2a", "0.0 0.4 0.1");
        minpe2a = 0.0;
        maxpe2a = 0.4;
        kont_.pe2a = 0.1;

        Covise::add_port(PARIN, "dicke_i", "Slider", "value for dicke_i");
        Covise::set_port_default("dicke_i", "0.0 40.0 12.0");
        mindicke_i = 0.0;
        maxdicke_i = 40.0;
        dick_.dicke_i = 20.0;

        Covise::add_port(PARIN, "dicke_a", "Slider", "value for dicke_a");
        Covise::set_port_default("dicke_a", "0.0 40.0 8.0");
        mindicke_a = 0.0;
        maxdicke_a = 40.0;
        dick_.dicke_a = 8.0;

        Covise::add_port(PARIN, "d_strich_a", "Slider", "value for d_strich_a");
        Covise::set_port_default("d_strich_a", "0.0 1.0 0.0");
        mind_strich_a = 0.0;
        maxd_strich_a = 1.0;
        dick_.d_strich_a = 0.0;

        Covise::add_port(PARIN, "versch_i", "Slider", "value for versch_i");
        Covise::set_port_default("versch_i", "1.0 1.5 1.1");
        minversch_i = 1.0;
        maxversch_i = 1.5;
        prof_.versch_i = 1.1;

        Covise::add_port(PARIN, "versch_a", "Slider", "value for versch_a");
        Covise::set_port_default("versch_a", "1.0 1.5 1.3");
        minversch_a = 1.0;
        maxversch_a = 1.5;
        prof_.versch_a = 1.3;

        Covise::add_port(PARIN, "db2i", "Slider", "value for db2i");
        Covise::set_port_default("db2i", "0.0 5.0 3.0");
        mindb2i = 0.0;
        maxdb2i = 5.0;
        wust_.db2i = 3.0;

        Covise::add_port(PARIN, "db2a", "Slider", "value for db2a");
        Covise::set_port_default("db2a", "0.0 5.0 1.0");
        mindb2a = 0.0;
        maxdb2a = 5.0;
        wust_.db2a = 1.0;

        Covise::add_port(PARIN, "db1i", "Slider", "value for db1i");
        Covise::set_port_default("db1i", "0.0 8.0 5.0");
        mindb1i = 0.0;
        maxdb1i = 8.0;
        wust_.db1i = 5.0;

        Covise::add_port(PARIN, "db1a", "Slider", "value for db1a");
        Covise::set_port_default("db1a", "0.0 8.0 4.0");
        mindb1a = 0.0;
        maxdb1a = 8.0;
        wust_.db1a = 4.0;

        Covise::add_port(PARIN, "mwi", "Slider", "value for mwi");
        Covise::set_port_default("mwi", "0.30 0.48 0.40");
        minmwi = 0.3;
        maxmwi = 0.48;
        maxw_.mwi = 0.4;

        Covise::add_port(PARIN, "mwa", "Slider", "value for mwa");
        Covise::set_port_default("mwa", "0.30 0.48 0.40");
        mindb2i = 0.3;
        maxdb2i = 0.48;
        wust_.db2i = 0.4;

        Covise::add_port(PARIN, "Kugelradius", "Slider", "value for Kugelradius");
        Covise::set_port_default("Kugelradius", "1.00 1.20 1.07");
        mind2_kugel = 1.00;
        maxd2_kugel = 1.20;
        fona_.d2_kugel = 1.07;

        Covise::add_port(PARIN, "oeffnen prototyp.dat", "Boolean", "value for oeffnen prototyp.dat");
        Covise::set_port_default("oeffnen prototyp.dat", "FALSE");

        Covise::add_port(PARIN, "oeffnen schaufel_xyz.dat", "Boolean", "value for oeffnen schaufel_xyz.dat");
        Covise::set_port_default("oeffnen schaufel_xyz.dat", "FALSE");

        Covise::add_port(PARIN, "Stroe-Winkel.dat", "Boolean", "value for Dateiausgabe");
        Covise::set_port_default("Stroe-Winkel.dat", "TRUE");

        Covise::add_port(PARIN, "La-Winkel.dat", "Boolean", "value for Dateiausgabe");
        Covise::set_port_default("La-Winkel.dat", "TRUE");

        Covise::add_port(PARIN, "schaufel_xyz.dat", "Boolean", "value for Dateiausgabe");
        Covise::set_port_default("schaufel_xyz.dat", "TRUE");

        Covise::add_port(PARIN, "prototyp.dat", "Boolean", "value for prototyp.dat");
        Covise::set_port_default("prototyp.dat", "TRUE");

        Covise::add_port(PARIN, "Dateiausgabe", "Boolean", "value for Dateiausgabe");
        Covise::set_port_default("Dateiausgabe", "FALSE");

        Covise::add_port(PARIN, "Netz oeffnen", "Boolean", "value for Netz oeffnen");
        Covise::set_port_default("Netz oeffnen", "FALSE");

        Covise::add_port(PARIN, "Netz speichern", "Boolean", "value for Netz speichern");
        Covise::set_port_default("Netz speichern", "FALSE");

        Covise::add_port(PARIN, "Netzgenerierung", "Boolean", "value for Netzgenerierung");
        Covise::set_port_default("Netzgenerierung", "FALSE");

        Covise::init(argc, argv);
        char buf[500];
        sprintf(buf, "C%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
        Covise::set_feedback_info(buf);
        Covise::set_quit_callback(Application::quitCallback, this);
        Covise::set_start_callback(Application::computeCallback, this);
        Covise::set_param_callback(Application::parameterCallback, this);

        showParam("Hauptmenue", 2);
        showParam("Q", minQ, maxQ, absol_.Q);
        showParam("H", minH, maxH, absol_.H);
        showParam("n", minN, maxN, absol_.N);
        showParam("D2", mind2, maxd2, zwis_.d2);
        showParam("D1/D2", mindi_da, maxdi_da, zwis_.di_da);
        //showParam("Kranzwinkel",minkranxwin,maxkranxwin,kranzwin);
        showParam("z", minn, maxn, zwis_.nlschaufel);

        neue_geometrie = 1;
        netz_aktuell = 0;
    }
    void showParam(char *paramname, int min, int max, int value, int displaytype = 0);
    void showParam(char *paramname, float min, float max, float value, int displaytype = 0);
    void showParam(char *paramname, int val, int displaytype = 2);

    void hideParam(char *paramname, int min, int max, int value, int displaytype = 0);
    void hideParam(char *paramname, float min, float max, float value, int displaytype = 0);
    void hideParam(char *paramname, int value, int displaytype = 2);

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _READIHS_H
