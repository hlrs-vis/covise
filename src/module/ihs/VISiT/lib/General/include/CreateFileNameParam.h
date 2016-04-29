#ifndef  CreateFileNameParam_INCLUDED
#define  CreateFileNameParam_INCLUDED

#define  CFNP_NORM   0
#define  CFNP_ENV_ONLY  1

extern  char *CreateFileNameParam(const char *def, const char *env, const char *ext, int mode);
#endif                                            // CreateFileNameParam_INCLUDED
