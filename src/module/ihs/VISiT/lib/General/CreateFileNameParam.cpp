#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/CreateFileNameParam.h"
#include "include/log.h"

#define  MAXPATH  255

char *CreateFileNameParam(const char *def, const char *env, const char *file, int mode)
{
	char buf[MAXPATH+1];
	char *p;

	*buf = '\0';
	dprintf(2,"CreateFileNameParam(): def=%s, env=%s, file=%s, mode=%d\n",
			def,env,file,mode);
	// Environment overrules default
	p = ((getenv(env) && *getenv(env)) ? getenv(env) : (char *)(def));
	if (p && *p) {
		if (file && *file) {
			if (strlen(p)+strlen(file)+1 <= MAXPATH)
				sprintf(buf, "%s/%s", p, file);
		}
		else {
			if (strlen(p) <= MAXPATH)
				strcpy(buf, p);
		}
	}
	else {
		if (mode == CFNP_NORM) {
			if (file && *file) {
				if (strlen(file)+2 <= MAXPATH)
					sprintf(buf, "./%s", file);
			}
			else {
				strcpy(buf, "./");
			}
		}
	}

	return (*buf ? strdup(buf) : NULL);
}
