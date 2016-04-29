#ifndef  ISO_INCLUDED
#define  ISO_INCLUDED

struct Isocurve
{
   int num;
   int max;
   float *offset;
   float *val1;
   float *val2;
   float isoval;
   int measured;
   int calc;
};

struct Isofield
{
   int num;
   int max;
   struct Isocurve **ic;
};

struct Isofield *AllocIsofieldStruct(void);
int AddIsocurve(struct Isofield *isof, float isoval);
int AddIsotupel(struct Isocurve *ic, float offset, float *val1, float *val2);
void FreeIsocurveStruct(struct Isocurve *ic);
void FreeIsofieldStruct(struct Isofield *isof);
int CalcIsocurve(float isoval, struct Isofield *isof);
void CalcIsotupel(struct Isocurve *ic_min, struct Isocurve *ic_max, float off, float isoval, float *rv1, float *rv2);
void SortIsofield(struct Isofield *isof);
int isocmp(const void *ae, const void *be);
struct Isofield * IsoRead(char *fn);
int IsoGetIndex(struct Isofield *iso, float val);

struct Isofield *ReadMeasured(char *fn);
int ReadCalc(char *fn, float **calc);

#ifdef DEBUG
void DumpIsocurve(struct Isocurve *ic, FILE *fp);
#endif                                            // DEBUG
#endif                                            // ISO_INCLUDED
