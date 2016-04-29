#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifndef WIN32
#include <unistd.h>
#endif

#include "include/IOChecks.h"
#include "include/log.h"

int IsRegularFile(char *fn)
{
   struct stat s;

   if (stat(fn, &s))
   {
      dprintf(0, (char *)"ERROR IsRegularFile(%s): errno=%d(%s)\n", fn, errno, strerror(errno));
      return 0;
   }
#ifdef WIN32
   dprintf(1, "fn=%s: S_ISREG=%d\n", fn, (s.st_mode & _S_IFREG));
   return (s.st_mode & _S_IFREG);
#else
   dprintf(1, (char *)"fn=%s: S_ISREG=%d\n", fn, S_ISREG(s.st_mode));
   return S_ISREG(s.st_mode);
#endif
}
