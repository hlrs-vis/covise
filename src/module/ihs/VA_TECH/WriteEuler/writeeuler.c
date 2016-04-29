#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <include/geo.h>
#include <include/log.h>
#include <AxialRunner/include/axial.h>



static void WriteEulerHead(FILE *fp, struct axial *ar);
static void WriteEulerMeridianContour(FILE *fp, struct axial *ar);
static void WriteEulerBladeContour(FILE *fp, struct axial *ar);
static void WriteEulerFoot(FILE *fp);
static void WriteOneCrop(FILE *fp, struct be *be, int ind, float scale);
static void OneEdgeLine(FILE *fp, struct Point *p, int i);


int WriteEuler(char *fn, struct axial *ar)
{
	FILE *fp;

	if ((fp = fopen(fn , "w")) == NULL)
		return errno;

	WriteEulerHead(fp, ar);
	WriteEulerMeridianContour(fp, ar);
	WriteEulerBladeContour(fp, ar);
	WriteEulerFoot(fp);
	
	fclose(fp);

	return 0;
}

static void WriteEulerMeridianContour(FILE *fp, struct axial *ar)
{
	int i;
	float x, y;

	fprintf(fp, " Meridiankontur : Nabe und Kranz vom Eintritt zum Austritt\n");
	fprintf(fp, " ----------------\n");
	fprintf(fp, " Anzahl Punkte pro Kontur\n");
	fprintf(fp, "    %d\n", ar->me[0]->ml->p->nump);

		// Nabe
	fprintf(fp, "\n Nabe   R          Z\n\n");
	for (i = 0; i < ar->me[0]->ml->p->nump; i++) {
		x = ar->me[0]->ml->p->x[i];
		y = ar->me[0]->ml->p->y[i];
		fprintf(fp, "%11.3f%11.3f\n", sqrt(x*x + y*y), ar->me[0]->ml->p->z[i]);
	}

		// Kranz
	fprintf(fp, "\n Kranz  R          Z\n\n");
	for (i = 0; i < ar->me[ar->be_num-1]->ml->p->nump; i++) {
		x = ar->me[ar->be_num-1]->ml->p->x[i];
		y = ar->me[ar->be_num-1]->ml->p->y[i];
		fprintf(fp, "%11.3f%11.3f\n", sqrt(x*x + y*y), ar->me[ar->be_num-1]->ml->p->z[i]);
	}
	fprintf(fp, "\n");
}

static void OneEdgeLine(FILE *fp, struct Point *p, int i)
{
	float r;

	r = sqrt(p->x[i]*p->x[i] + p->y[i]*p->y[i]);
	// ACHTUNG: Der Euler braucht den x-Wert immer negativ !!
	//          das wird aber nur hier an der Ausgabe gedreht
	fprintf(fp, "%11.3f %11.3f %11.3f %11.3f\n", p->x[i]*-1.0,
							p->y[i], p->z[i], r);
}

static void WriteEulerBladeContour(FILE *fp, struct axial *ar)
{
	int i;
	int np;
	float scale;

	fprintf(fp, " Eintritts- und Austrittskanten : von Nabe zu Kranz\n");
	fprintf(fp, " --------------------------------\n");
	fprintf(fp, " Anzahl Punkte pro Kante\n");
	fprintf(fp, "%5d\n", ar->be_num);

	
	fprintf(fp, "\n %s     X          Y          Z          R\n\n", "EK");
	for (i = 0; i < ar->be_num; i++)
		OneEdgeLine(fp, ar->be[i]->cl_cart, 0);
	fprintf(fp, "\n");

	fprintf(fp, "\n %s     X          Y          Z          R\n\n", "AKD");
	for (i = 0; i < ar->be_num; i++)
		OneEdgeLine(fp, ar->be[i]->ps_cart, ar->be[i]->ps_cart->nump-1);
	fprintf(fp, "\n");

	fprintf(fp, "\n %s     X          Y          Z          R\n\n", "AKS");
	for (i = 0; i < ar->be_num; i++)
		OneEdgeLine(fp, ar->be[i]->ss_cart, ar->be[i]->ss_cart->nump-1);
	fprintf(fp, "\n");

	fprintf(fp, " Schaufeldaten : Profile von Nabe zu Kranz\n");
	fprintf(fp, " --------------  Punkte  von AK-Druck- bis AK-Saugseite (Turbine)\n");
	fprintf(fp, " Anzahl Profile          Anzahl Punkte pro Profil\n");
	np = ar->be[0]->ps_cart->nump + ar->be[0]->ss_cart->nump - 1;
	fprintf(fp, " %6d                  %6d\n\n", ar->be_num, np);

	for (i = 0; i < ar->be_num; i++) {
		scale = (i == ar->be_num-1 ? 1.1 : 1.0);
		WriteOneCrop(fp, ar->be[i], i, scale);
	}
}

static void WriteOneCrop(FILE *fp, struct be *be, int ind, float scale)
{
	int i;
	float r;
	float v[3];

	fprintf(fp, "\n Profil  %3d\n", ind+1);
	fprintf(fp, "        X          Y          Z          R\n\n");

	for ( i = be->ps_cart->nump-1; i >= 0; i--) {
		// ACHTUNG: Der Euler braucht den x-Wert immer negativ !!
		//          das wird aber nur hier an der Ausgabe gedreht
		v[0] = be->ps_cart->x[i] * scale * -1.0;
		v[1] = be->ps_cart->y[i] * scale;
		v[2] = be->ps_cart->z[i];
		r = sqrt(v[0]*v[0] + v[1]*v[1]);
		fprintf(fp, "%13.3f%13.3f%13.3f%13.3f\n", v[0], v[1], v[2], r);
	}
	for ( i = 1; i < be->ss_cart->nump; i++) {
		// ACHTUNG: Der Euler braucht den x-Wert immer negativ !!
		//          das wird aber nur hier an der Ausgabe gedreht
		v[0] = be->ss_cart->x[i] * scale * -1.0;
		v[1] = be->ss_cart->y[i] * scale;
		v[2] = be->ss_cart->z[i];
		r = sqrt(v[0]*v[0] + v[1]*v[1]);
		fprintf(fp, "%13.3f%13.3f%13.3f%13.3f\n", v[0], v[1], v[2], r);
	}
}


static void WriteEulerHead(FILE *fp, struct axial *ar)
{
	time_t secs;
	struct tm *tm;

	time(&secs);
	tm = localtime(&secs);
	fprintf(fp, "                     Laufradgeometrie\n");
	fprintf(fp, "                     ****************\n");
	fprintf(fp, "\n");
	fprintf(fp, " Stichwort   : CKT-K4              \n");
	fprintf(fp, " Laufrad-Nr  : K614  \n");
	fprintf(fp, " Variante    : A \n");
	fprintf(fp, " Filename    : K614-GEO-A.DAT      \n");
	fprintf(fp, " Bpkt-File   : \n");
	fprintf(fp, " Meko-File   :\n");
	fprintf(fp, " Leitapparat :\n");
	fprintf(fp, " Reserve     :\n");
	fprintf(fp, " Reserve     :\n");
	fprintf(fp, "\n");
	fprintf(fp, " Datum       : %d-%d-%d %d:%d:%d\n", tm->tm_mday, tm->tm_mon,
							tm->tm_year, tm->tm_hour, tm->tm_min, tm->tm_sec);
	fprintf(fp, " Abteilung   : autogenerated by IHS-AxialRunner-module\n");
	fprintf(fp, "\n");
	fprintf(fp, " Z2       Bo        D1        D2\n");
	fprintf(fp, "  %d    %11.4f    %11.4f      %11.4f\n", ar->nob, ar->h_inl_ext, ar->diam[1], 0.0);
	fprintf(fp, "\n");
}

static void WriteEulerFoot(FILE *fp)
{
	fprintf(fp, "\n Ende des Geometrie-File\n");
}


#ifdef	MAIN
int main(int argc, char **argv)
{
	struct geometry *geo;
	char *fn = "test.euler.zentraldatei";
	char *in = "./axial.cfg";

	SetDebugLevel(0);
	if (getenv(ENV_IHS_DEBUGLEVEL))
		SetDebugLevel(atoi(getenv(ENV_IHS_DEBUGLEVEL)));
	SetDebugPath("./debug/", getenv(ENV_IHS_DEBPATH));
	dopen("debug/debug.rei");
	if ((geo = ReadGeometry(in)) != NULL) {
		dprintf(0, "ReadGeometry(%s) ok\n", in);
		CreateGeometry4Covise(geo);

		WriteEuler(fn, geo->ar);
	}
	else
		dprintf(0, "Datei :%s (errno=%d)\n", in, errno);

	return 0;
}
#endif
