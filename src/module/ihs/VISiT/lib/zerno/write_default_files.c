/* WARNING: This file is automatically created
 *          DONT EDIT manually !! */

#include <stdio.h>
#include <errno.h>


static void write_file_1()
{
	char *filename = "zerno.stf.default";
	FILE *fp;

	if ((fp = fopen(filename, "w")) == NULL)
	{
		fprintf(stderr, "Couldn't open file: %s (errno=%d)\n", filename,
																errno);
		exit(1);
	}
	fputs("***********************************************************************\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("**************** STEUERFILE FUER ZERLEGUNG *******+********************\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("-> Dimension                               :2\n", fp);
	fputs("**---------------------------------------------------------------------\n", fp);
	fputs("-> Zerlegung in wieviel Gebiete ?          :4\n", fp);
	fputs("**---------------------------------------------------------------------\n", fp);
	fputs("-> Geometrie-File :shop_2.geo\n", fp);
	fputs("-> Randdbed.-File :/mnt/fs2/ihs/mai/SHOP/shop_2.rbe_lami\n", fp);
	fputs("-> Ergebnis -File :\n", fp);
        fputs("-> Gemitt.Erg-File:\n", fp);
	fputs("**---------------------------------------------------------------------\n", fp);
	fputs("-> Geometrie-Pfad :GEO_04/\n", fp);
	fputs("-> Randbed  -Pfad :GEO_04_NEU/\n", fp);
	fputs("-> Ergebnis -Pfad :GEO_04/\n", fp);
	fputs("-> Gemitt.Erg-Pfad:\n", fp);
	fputs("**---------------------------------------------------------------------\n", fp);
	fputs("-> Liegt unter dem Geometrie-Pfad  eine Partition vor  ( ja / ne ) :ja\n", fp);
	fputs("-> Zerlegung der Randbedingungen mit dieser Partition  ( ja / ne ) :ja\n", fp);
	fputs("-> Zerlegung der Ergebnisse      mit dieser Partition  ( ja / ne ) :ne\n", fp);
	fputs("-> Zerlegung der gemittelte Erg. mit dieser Partition  ( ja / ne ) :ne\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("-> Zerlegung mit METIS_PartGraphKway       ( ja / ne ) :ja\n", fp);
	fputs("-> Zerlegung mit METIS_PartGraphVKway      ( ja / ne ) :ja\n", fp);
	fputs("-> Zerlegung mit METIS_PartGraphRecursive  ( ja / ne ) :ja\n", fp);
	fputs("**---------------------------------------------------------------------\n", fp);
	fputs("-> Zerlegung auf reduziertem Graph         ( ja / ne ) :ja\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("-> Ausgabe der partitionierten Daten    ( ja / ne ):ja\n", fp);
	fputs("**---------------------------------------------------------------------\n", fp);
	fputs("-> Ausgabe der Oberflaechengeometrie    ( ja / ne ):ne\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("-> doppelte Ueberlappung                ( ja / ne ):ne\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("\n", fp);
	fputs("***********************************************************************\n", fp);
	fputs("**************** STEUERFILE ENDE **************************************\n", fp);
	fputs("***********************************************************************\n", fp);
	fclose(fp);
}

void write_default_files()
{
	write_file_1();
}
