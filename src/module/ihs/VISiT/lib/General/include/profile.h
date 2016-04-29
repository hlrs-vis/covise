#ifndef PROFILE_H_INCLUDED
#define PROFILE_H_INCLUDED

struct profile
{
   int      num;                                  // number of stations
   int      max;                                  // max. number allocated
   int      t_sec;                                // index of max. thickness station
   int      naca;                                 // naca style data[1] or relative to max. thickness[0]
   float *c;                                      // rel. chord station
   float    *t;                                   // rel. profile thickness at c[]
};

extern  struct profile *AllocBladeProfile(void);
extern  void FreeBladeProfile(struct profile *bp);
extern  int AddProfilePoint(struct profile *p, float c, float t);
extern  int ReadBladeProfile(struct profile *bp, const char *sec, const char *fn);
extern  int AssignBladeProfile(struct profile *bp1, struct profile *bp2);
extern  int ShiftBladeProfile(struct profile *bp, float shift);
extern  void NormBladeProfile(struct profile *bp);
extern  void DumpBladeProfile(struct profile *bp);
#endif                                            // PROFILE_H_INCLUDED
