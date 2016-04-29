#include <stdio.h>
#include <string.h>
#ifndef WIN32
#include <iostream>
#endif
#include <Gate/include/stf.h>

int EINLESEN_STEUERFILE(char *datei_steuer, struct stf *stf_para)
{
   FILE *stream;
   char buf[200];

   const char *separator = "->";

   int i;

   if( (stream = fopen( &datei_steuer[0], "r" )) == NULL )
   {
      printf( "Kann '%s' nicht lesen!\n", datei_steuer);
      return 0;
   }
   else
   {
      for (i = 0; i < 8; i++)
      {
         fgets(buf,200,stream);
      }
      // leit_xyz_pfad
      fgets(buf,200,stream); read_string(buf, stf_para->leitxyz_pfad, separator);

      // Zeile nabekranz_pfad
      fgets(buf,200,stream); read_string(buf, stf_para->nabekranz_pfad, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // GEO-Pfad
      fgets(buf,200,stream); read_string(buf, stf_para->geo_pfad, separator);
      // RB-Pfad
      fgets(buf,200,stream); read_string(buf, stf_para->rb_pfad, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // Durchfluss
      fgets(buf,200,stream); read_double(buf, &stf_para->Q, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // Grenzpunkte auf Leitschaufelprofil
      fgets(buf,200,stream); read_int(buf, &stf_para->PG2, separator);
      fgets(buf,200,stream); read_int(buf, &stf_para->PG3, separator);
      fgets(buf,200,stream); read_int(buf, &stf_para->PG4, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // Grenzschichtdicke
      fgets(buf,200,stream); read_double(buf, &stf_para->grenz, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // deltagamma
      fgets(buf,200,stream); read_double(buf, &stf_para->deltagamma, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // Knotenanzahlen
      fgets(buf,200,stream); read_int(buf, &stf_para->anz_schnitte, separator);
      fgets(buf,200,stream); read_int(buf, &stf_para->anz_grenz, separator);
      fgets(buf,200,stream); read_int(buf, &stf_para->anz_15, separator);
      fgets(buf,200,stream); read_int(buf, &stf_para->anz_einlauf, separator);
      fgets(buf,200,stream); read_int(buf, &stf_para->anz_knot1, separator);
      fgets(buf,200,stream); read_int(buf, &stf_para->anz_knot2, separator);
      fgets(buf,200,stream); read_int(buf, &stf_para->anz_knot3, separator);
      fgets(buf,200,stream); read_int(buf, &stf_para->anz_knot4, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // Bogenlaengen
      fgets(buf,200,stream); read_double(buf, &stf_para->bgl_15_n, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->bgl_15_k, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->bgl_max_nabe, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->bgl_start, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // Verdichtungen laengs
      fgets(buf,200,stream); read_double(buf, &stf_para->lvs1, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->lvs2, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->lvs3, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->lvs4, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->lvs5, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->lvs6, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->lvs7, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // Verdichtungen quer
      fgets(buf,200,stream); read_double(buf, &stf_para->bvs1, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->bvs2, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->bvs3, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->bvs4, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->bvs8, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // Verdichtung radial
      fgets(buf,200,stream); read_double(buf, &stf_para->verd_radial, separator);
      for (i = 0; i < 4; i++)
      {
         fgets(buf,200,stream);
      }
      // Verschiebungen
      fgets(buf,200,stream); read_double(buf, &stf_para->versch_eintr, separator);
      fgets(buf,200,stream); read_double(buf, &stf_para->versch_austr, separator);

      fclose(stream);

      // abhaengige Netzparameter berechnen
      stf_para->anz_mitte = stf_para->anz_knot4 - stf_para->anz_knot1 + 1;
                                                  //Gesamtanzahl Breite Austritt
      stf_para->anz_breit = 2 * (stf_para->anz_grenz) + (stf_para->anz_mitte) - 2;
      stf_para->anz_knot9 = stf_para->anz_knot4 - stf_para->anz_knot1 + 1;
      stf_para->PG = 201;                         //Anzahl der Randpunkte Leitschaufel nach erstem Meshseed
      stf_para->PG1 = 0;
      stf_para->PG5 = stf_para->PG - 1;

      // Hier muss die Verdichtung andersherum sein
      stf_para->lvs7 = 1. / stf_para->lvs7;

      return(1);
   }

}


void DumpSTF(struct stf *stf_para)
{
   (void) stf_para;
}


int read_string(char *buf, char* &str, const char *separator)
// buf        ... Zeile / Quelle
// str        ... zu lesender String (Ergebnis)
// separator  ... Zeichenfolge, nach der str steht

{
   int pos, i;
   char buffer2[100];
   buf = strstr(buf, separator);                  // mit separator
   buf += sizeof(char) * strlen(separator);       // ohne separator
   pos=0;
   for(i = 0; i < strlen(buf); i++)               // Leertasten, Tabs weg
   {
      if (!(  ( buf[i]==' ' ) || (  buf[i]=='\t'  ) || (  buf[i]=='\n'  ) ))
      {
         buffer2[pos] = buf[i];
         pos++;
      }
   }
   buffer2[pos]='\0';

   str = new char[strlen(buffer2)+1];
   strcpy(str, buffer2);

   return(1);
}


int read_int(char *buf, int *izahl, const char *separator)
{
   buf = strstr(buf, separator);                  // mit separator
   buf += sizeof(char) * strlen(separator);       // ohne separator

   sscanf(buf, "%d ", izahl);

   return(1);
}


int read_double(char *buf, double *dzahl, const char *separator)
{
   buf = strstr(buf, separator);                  // mit separator
   buf += sizeof(char) * strlen(separator);       // ohne separator

   sscanf(buf, "%lf", dzahl);

   return(1);
}
