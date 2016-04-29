#ifndef PARAMETER_H_INCLUDED
#define PARAMETER_H_INCLUDED

#include "profile.h"

struct parameter
{
   int num;                                       // number of stations
   int max;                                       // number allocated
   int portion;                                   // portion to (re)allocate
   float *loc;                                    // station coords
   float *val;                                    // station values
};

struct parafield
{
   int num;                                       // number of parameter sets
   int max;                                       // number allocated
   int portion;                                   // portion to (re)allocate
   float *loc;                                    // parameter set against coord1
   struct parameter **para;                       // parameter set against coord2
};

extern  struct parameter *AllocParameterStruct(int portion);
extern  int AddParameter(struct parameter *para, float loc, float val);
extern  int ReadParameterSet(struct parameter *para, const char *sec, const char *fn);
extern  float InterpolateParameterSet(struct parameter *para, float loc, int extrapol);
extern  int Parameter2Profile(struct parameter *para, struct profile *prof);
extern  int Parameter2Radians(struct parameter *para);
extern  void FreeParameterStruct(struct parameter *para);

extern  struct parafield *AllocParameterField(int portion);
extern  int ReadParameterField(struct parafield *paraf, char *sec, char *subsec, const char *fn);
extern  struct parameter *InterpolateParameterField(struct parafield *paraf, float loc, int extrapol);
extern  void FreeParameterField(struct parafield *paraf);

extern  void DumpParameterSet(struct parameter *para);
extern  void DumpParameterField(struct parafield *paraf);
#endif                                            // PARAMETER_H_INCLUDED
