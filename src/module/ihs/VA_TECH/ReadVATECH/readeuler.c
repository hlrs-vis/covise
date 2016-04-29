#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <include/CreateFileNameParam.h>
#include <include/IOChecks.h>
#include <include/v.h>
#include <include/log.h>
#include "EuGri.h"


static float *ReadEulerDataBlock(FILE *fp, struct EuGri *s);
static int IgnoreLines(FILE *fp, int n);
static int ReadAndAllocDimensions(FILE *fp, struct EuGri *s);
static int ReadGridLine(FILE *fp, struct EuGri *s);
static int ReadResultHeader(FILE *fp, struct EuGri *s);
static void RotateGrid(struct EuGri *s, float angle);
static void CalcRelVelocity(struct EuGri *s, float omega);


struct EuGri *ReadEuler(char *grid, char *res, float omega)
{
	struct EuGri *s = NULL;

	if (grid && *grid) {
		s = ReadEulerGrid(grid, 0.0);
		if (s && res && *res) {
			s = ReadEulerResults(res, s, omega);
		}
	}

	return s;
}

void FreeStructEuler(struct EuGri *eu)
{
	if (eu) {
		if (eu->x)	free(eu->x);
		if (eu->y)	free(eu->y);
		if (eu->z)	free(eu->z);
		if (eu->p)	free(eu->p);
		if (eu->u)	free(eu->u);
		if (eu->v)	free(eu->v);
		if (eu->w)	free(eu->w);
		if (eu->ur)	free(eu->ur);
		if (eu->vr)	free(eu->vr);
		if (eu->wr)	free(eu->wr);
		free(eu);
	}
}

void NormEulerGrid(struct EuGri *s, float norm)
{
	int i;

	for ( i = 0; i < s-> num; i++) {
		s->x[i] *= norm;
		s->y[i] *= norm;
		s->z[i] *= norm;
	}
	s->norm = norm;
}

static float *ReadEulerDataBlock(FILE *fp, struct EuGri *s)
{
	int nread = -999;
	float p[5];
	int nl, nr, count;
	int i, j, k;
	int ind;
	float *ptmp;
	float *d;
	int num;
	char buf[1024+1];

	dprintf(1, "ReadEulerDataBlock()\n");
	// Das Gitter geht von 1..i, 1..j, 1..k; die Werte werden aber immer
	// von 0..i, 0..j, 0..k geliefert; also alles Zwischenspeichern und
	// dann umkopieren ...
	num = (s->i + 1) * (s->j + 1) * (s->k + 1);

	// Zuerst lesen wir den Druck ...
	nl    = num/5;	// Anzahl der zu lesenden Zeilen
	count = 0;
	ptmp = (float *)calloc(num, sizeof(float));
		// Hier alle voll gefuellten Zeilen ...
	for (i = 0; i < nl; i++) {
		if (buf != fgets(buf, sizeof(buf), fp)) {
			dprintf(0, "ERROR: ReadEulerDataBlock(): fgets failed ! i=%d, (count=%d)\n", i, count);
			exit(1);
		}
		dprintf(4, "i=%4d: buf=%s", i, buf);
		if ((nread = sscanf(buf, " %f %f %f %f %f", p, p+1, p+2, p+3, p+4)) == 5) {
			ptmp[count++] = p[0];
			ptmp[count++] = p[1];
			ptmp[count++] = p[2];
			ptmp[count++] = p[3];
			ptmp[count++] = p[4];
		}
		else {
			dprintf(0, "ERROR: ReadEulerDataBlock(): i=%d, nread=%d (count=%d)\n", i, nread, count);
			exit(1);
		}
	}
		// ... dann ev. noch die letzte "Rest"-Zeile
	if (( nr = num%5 ) != 0) {
		if (buf != fgets(buf, sizeof(buf), fp)) {
			dprintf(0, "ERROR: ReadEulerDataBlock(): fgets failed ! nr=%d, (count=%d)\n", nr, count);
			exit(1);
		}
		dprintf(4, "nr=%d: buf=%s", nr, buf);
		if ( nr != sscanf(buf, "%f %f %f %f %f", p, p+1, p+2, p+3, p+4)) {
			ptmp[count++] = p[0];
			if (nr > 1) {
				ptmp[count++] = p[1];
				if (nr > 2) {
					ptmp[count++] = p[2];
					if (nr > 3) {
						ptmp[count++] = p[3];
					}
				}
			}
		}
	}
	dprintf(2, "ReadEulerDataBlock(): Vor Umkopieren ...\n");
	dprintf(3, "  count=%d, num=%d, nr=%d, nread=%d\n", count, num, nr, nread);
	dprintf(4, "  ptmp[0] = %f\t", ptmp[0]);
	dprintf(4, "  ptmp[1] = %f\t", ptmp[1]);
	dprintf(4, "  ptmp[2] = %f\t", ptmp[2]);
	dprintf(4, "  ptmp[3] = %f\t", ptmp[3]);
	dprintf(4, "  ptmp[4] = %f\n", ptmp[4]);
	dprintf(4, "  ptmp[5] = %f\t", ptmp[5]);
	dprintf(4, "  ptmp[6] = %f\n", ptmp[6]);
	dprintf(4, "  ptmp[%d] = %f\t", count-8, ptmp[count-8]);
	dprintf(4, "  ptmp[%d] = %f\t", count-7, ptmp[count-7]);
	dprintf(4, "  ptmp[%d] = %f\t", count-6, ptmp[count-6]);
	dprintf(4, "  ptmp[%d] = %f\t", count-5, ptmp[count-5]);
	dprintf(4, "  ptmp[%d] = %f\n", count-4, ptmp[count-4]);
	dprintf(4, "  ptmp[%d] = %f\t", count-3, ptmp[count-3]);
	dprintf(4, "  ptmp[%d] = %f\t", count-2, ptmp[count-2]);
	dprintf(4, "  ptmp[%d] = %f\n", count-1, ptmp[count-1]);
	dprintf(4, "  nl = %d, count = %d, nr = %d, num = %d\n", nl, count, nr, num);
	count = 0;
	d = (float *)calloc(s->num, sizeof(float));
	for (k = 0; k <= s->k; k++) {
		for (j = 0; j <= s->j; j++) {
			for (i = 0; i <= s->i; i++) {
				ind = (i)*s->j*s->k + (j)*s->k + k ;
				if (i != s->i && j != s->j  && k != s->k )
					d[ind] = ptmp[count];
				count++;
				dprintf(5, "%5d, %5d, %3d, %3d, %3d %f\n", ind, count, i, j, k, ptmp[count-1]);
			}
		}
	}
	
	free(ptmp);
	dprintf(1, "Leaving ReadEulerDataBlock()\n");

	return d;
}

struct EuGri *ReadEulerResults(char *fn, struct EuGri *s, float omega)
{
	FILE *fp;

	dprintf(1, "ReadEulerResults(): fn=%s\n", fn);
	if (!IsRegularFile(fn) || (fp = fopen(fn, "r")) == NULL)
		return s;
	IgnoreLines(fp,  6);		// Kommentarzeilen
	ReadResultHeader(fp, s);	// Kopf lesen und vergleichen
	IgnoreLines(fp,  1);		// ??
	IgnoreLines(fp,  2);		// ??

	s->p = ReadEulerDataBlock(fp, s);
	s->u = ReadEulerDataBlock(fp, s);
	s->v = ReadEulerDataBlock(fp, s);
	s->w = ReadEulerDataBlock(fp, s);
	CalcRelVelocity(s, omega);
	
	fclose(fp);
	dprintf(1, "Leaving ReadEulerResults()\n");

	return s;
}

struct EuGri *ReadEulerGrid(char *fn, float alpha)
{
	FILE *fp;
	struct EuGri *s;

	dprintf(1, "Entering ReadEulerGrid(%s)\n", fn);
		// file readable ??
	if (!IsRegularFile(fn) || (fp = fopen(fn, "r")) == NULL)
		return NULL;
	if ((s = (struct EuGri *)calloc(1, sizeof(struct EuGri))) == NULL)
		return NULL;
	s->norm=1.0;
	
	// Im Moment: header ueberlesen ...
	IgnoreLines(fp,  6);		// Kommentarzeilen
	IgnoreLines(fp,  1);		// Name des Projekts
	IgnoreLines(fp,  1);		// ????
	IgnoreLines(fp,  1);		// 154.000     0.100E-02
	ReadAndAllocDimensions(fp, s);		// Dimension
	IgnoreLines(fp, 37);		// ??
	while (ReadGridLine(fp, s))
		;
	fclose(fp);

	RotateGrid(s, alpha);
	dprintf(1, "Leaving ReadEulerGrid(%s)\n", fn);

	return s;
}

struct EuGri *MultiRotateGrid(struct EuGri *s, int nob)
{
  // i ... meridian, j ... b2b, k ... hub to shroud
  int i, j, k, n, ixt,ixs, jj, nmax;
  float alpha, dalpha, roma[2][2];

  struct EuGri *t;

  dprintf(1, "Entering MultiRotateGrid(s,%d)\n",nob);
  // alloc memory for multi rotated grid and init i,j,k
  if ((t = (struct EuGri *)calloc(1, sizeof(struct EuGri))) == NULL)
    return NULL;
  t->norm  = 1.0;
  nmax = nob;
  t->i = s->i;
  t->j = s->j*nmax;
  t->k = s->k;
  t->num = t->i*t->j*t->k;
  dprintf(3, "MultiRotateGrid(): i=%d, j=%d, k=%d\n", t->i, t->j, t->k);
  t->x = (float *)calloc(t->num, sizeof(float));
  t->y = (float *)calloc(t->num, sizeof(float));
  t->z = (float *)calloc(t->num, sizeof(float));
  t->u = (float *)calloc(t->num, sizeof(float));
  t->v = (float *)calloc(t->num, sizeof(float));
  t->w = (float *)calloc(t->num, sizeof(float));
  
  // rotate grid nob-times --> new grid t.
  dalpha = 2*M_PI/nob;
  dprintf(3, "MultiRotateGrid(): dalpha = %f\n",dalpha*180/M_PI);
  for(i = 0; i < t->i; i++) {
    for(n = 0, jj = 0; n < nmax; n++) {
      alpha = n*dalpha;
      roma[0][0] =  cos(alpha); roma[0][1] = sin(alpha);
      roma[1][0] = -sin(alpha); roma[1][1] = cos(alpha);
      for(j = 0; j < s->j; j++, jj++) {
	ixt = i*t->j*s->k + jj*s->k;
	ixs = i*s->j*s->k +  j*s->k;
	for(k = 0; k < s->k; k++, ixt++, ixs++) {
	  dprintf(3, "MultiRotateGrid(): k=%3d, n=%3d, j=%3d, i=%3d, jj=%4d",
		  k, n, j, i, jj);
	  dprintf(3, ", ixt=%8d, ixs=%8d\n",ixt,ixs);
	  t->x[ixt] = s->x[ixs]*roma[0][0] + s->y[ixs]*roma[0][1];
	  t->y[ixt] = s->x[ixs]*roma[1][0] + s->y[ixs]*roma[1][1];
	  t->z[ixt] = s->z[ixs];
	  if(s->u && s->v && s->w) {
	    t->u[ixt] = s->u[ixs]*roma[0][0] + s->v[ixs]*roma[0][1];
	    t->v[ixt] = s->u[ixs]*roma[1][0] + s->v[ixs]*roma[1][1];
	    t->w[ixt] = s->w[ixs];
	  }
	}
      } // j
    }
  } // end k
  if(s->u && s->v && s->w)
    CalcRelVelocity(t, s->omega);

  return t;
}

static int IgnoreLines(FILE *fp, int n)
{
	char x[1000];
	int i;

	for (i = 0; i < n; i++) {
		if (x != fgets(x, sizeof(x)-1, fp))
			break;
	}

	return i-1;
}

static int ReadAndAllocDimensions(FILE *fp, struct EuGri *s)
{
	int i, j, k, x1, x2;

	if (fscanf(fp, "%d %d %d %d %d", &i, &j, &k, &x1, &x2) != 5)
		return 0;
	s->i = i;
	s->j = j;
	s->k = k;
	s->num = i*j*k;
	dprintf(3, "ReadDimensions(): i=%d, j=%d, k=%d\n", i, j, k);
	s->x = (float *)calloc(s->num, sizeof(float));
	s->y = (float *)calloc(s->num, sizeof(float));
	s->z = (float *)calloc(s->num, sizeof(float));
	s->p = NULL;	// memory allocation later ...
	s->u = NULL;	// memory allocation later ...
	s->v = NULL;	// memory allocation later ...
	s->w = NULL;	// memory allocation later ...
	s->ur = NULL;	// memory allocation later ...
	s->vr = NULL;	// memory allocation later ...
	s->wr = NULL;	// memory allocation later ...

	return 1;
}

static int ReadGridLine(FILE *fp, struct EuGri *s)
{
	int i, j, k, ind;
	float x, y, z;

	if (fscanf(fp, "%d %d %d %f %f %f", &i, &j, &k, &x, &y, &z) != 6)
		return 0;
	ind = (i-1)*s->j*s->k + (j-1)*s->k + k-1;

	if (ind < 0 || ind > s->num) {
		dprintf(0, "ERROR: ind=%d (i=%d, j=%d, k=%d) > num=%d !!\n", ind, i, j, k, s->num);
		exit(1);
	}
	s->x[ind] = x;
	s->y[ind] = y;
	s->z[ind] = z;
	dprintf(5, "ReadGridLine(): ind=%5d, i=%3d, j=%3d, k=%3d, x=%f, y=%f, z=%f\n", ind, i, j, k, x, y, z);
	return 1;	
}

static int ReadResultHeader(FILE *fp, struct EuGri *s)
{
	int dummy, i, j, k;

	if ( 4 == fscanf(fp, "%d %d %d %d", &dummy, &i, &j, &k)) {
		if (i != s->i || j != s->j || k != s->k) {
			dprintf(0, "ERROR: ReadResultHeader(): s->i = %d, i = %d, s->j = %d, j = %d, s->k = %d, k = %d\n",
					s->i, i, s->j, j, s->k, k);
			exit(1);
		}
	}
	else {
		dprintf(0, "ReadResultHeader(): Wrong number of header-numbers\n");
		exit(1);
	}
	return 1;
}

static void RotateGrid(struct EuGri *s, float angle)
{
	int i;
	float roma[2][2];
	float x[2];

	dprintf(1, "Entering RotateGrid(angle=%f)\n", angle);
	roma[0][0] =  cos(angle);
	roma[0][1] = -sin(angle);
	roma[1][0] =  sin(angle);
	roma[1][1] =  cos(angle);

	for (i=0; i<s->num; i++) {
	  x[0] = s->x[i]; x[1] = s->y[i];
		s->x[i] = x[0] * roma[0][0] + x[1] * roma[0][1];
		s->y[i] = x[0] * roma[1][0] + x[1] * roma[1][1];
		dprintf(6,"RotateGrid(,%f): x = [%f, %f], x,y = [%f, %f]\n",
			angle*180/M_PI,x[0],x[1],s->x[i],s->y[i]);
	}
	dprintf(1, "Leaving RotateGrid\n");
}

static void CalcRelVelocity(struct EuGri *s, float omega)
{
	int i;

	dprintf(1, "Entering CalcRelVelocity(omega=%f)\n", omega);
	s->omega = omega;

	if (s->ur)	free(s->ur);
	if (s->vr)	free(s->vr);
	if (s->wr)	free(s->wr);
	s->ur = (float *)calloc(s->num, sizeof(float));
	s->vr = (float *)calloc(s->num, sizeof(float));
	s->wr = (float *)calloc(s->num, sizeof(float));

	for ( i = 0; i < s->num; i++) {
				// bei VATECH werden alle Angaben in mm gemacht
		s->ur[i] = s->u[i] + s->y[i]/1000*omega;
		s->vr[i] = s->v[i] - s->x[i]/1000*omega;
		s->wr[i] = s->w[i];
	}
	dprintf(1, "leaving CalcRelVelocity(omega=%f)\n", omega);
}

#ifdef	MAIN

int main(int argc, char **argv)
{
	ReadEuler("testdata/k614__a.netz", "testdata/k614__a_20.euler");

	return 0;
}
#endif
