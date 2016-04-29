#ifndef  IHS_READCFG
#define  IHS_READCFG

char * IHS_GetCFGValue(const char *file, const char *section, const char *key);
char ** dump_cfg(void);
#endif                                            /* IHS_READCFG	*/
