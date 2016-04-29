#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#ifndef WIN32
#include <sys/utsname.h>
#endif
#include "include/log.h"

#define SRCLEN  256
static FILE *_D = NULL;
static int _debuglevel = 0;
static int _debugstderr = 0;
static char _debugpath[SRCLEN+1];

/* debuglevels (suggestion)
 * 0   ... allways
 * 1   ... function calls
 * 2   ... parts of functions
 * 3   ... parts of functions
 * >=4 ... value debugging
 */

/* compile unconditionally, as DEBUG might be defined for individual files */
static char __src[SRCLEN+1];
static int __srcline;

int dSetSrc(const char *src, int srcline)
{
   if (*src)
      strncpy(__src, src, SRCLEN);
   __srcline = srcline;

   return 1;                                      // immer 1, wegen dem Makro
}

int dopen(char *fn)
{
   if (fn && *fn)
      _D = fopen(fn, "w");
   if (_D)
   {
#ifndef _WIN32
     struct tm *tm;
      time_t secs;
      struct utsname uts;

      time(&secs);
      tm = localtime(&secs);
      uname(&uts);

      fprintf(_D, "Session from %02d.%02d.%4d, %02d:%02d:%02d h\n  on %s (%s, %s, %s, %s)\n",
         tm->tm_mday, tm->tm_mon, tm->tm_year+1900,
         tm->tm_hour, tm->tm_min, tm->tm_sec,
         uts.nodename,
         uts.machine,
         uts.sysname,
         uts.release,
         uts.version);
#else
     /* fprintf(_D, "Session from %02d.%02d.%4d, %02d:%02d:%02d h\n",
         tm->tm_mday, tm->tm_mon, tm->tm_year+1900,
         tm->tm_hour, tm->tm_min, tm->tm_sec
         );*/
#endif 
   }
   return (_D ? 1 : 0);
}


void dclose(void)
{
   if (_D)
      fclose(_D);
}


int __dprintf(int dlevel, const char *fmt, ...)
{
   va_list ap;

   if (dlevel <= _debuglevel)
   {
      if (_debugstderr || dlevel == 0)
      {
         va_start(ap, fmt);
         vfprintf(stderr, fmt, ap);
         va_end(ap);
         fflush(stderr);
      }
      if (_D)
      {
#ifdef  DEBUG
         if (*__src)
            fprintf(_D, "%-25.25s(%4d): ", __src, __srcline);
#endif
         va_start(ap, fmt);
         vfprintf(_D, fmt, ap);
         va_end(ap);
         fflush(_D);
      }
   }
   return 1;                                      // wegen Makro !!!!
}


void SetDebugLevel(int dlevel)
{
   _debuglevel = dlevel;
   if (getenv(ENV_IHS_DEBUGSTDERR) && !strcmp(getenv(ENV_IHS_DEBUGSTDERR),"ON"))
      _debugstderr = 1;
   __dprintf(0, "SetDebugLevel()=%d\n", dlevel);
   __dprintf(0, "** log.c: BK Version-key: %K%\n");
   __dprintf(0, "** log.c: BK Checkout   : %D%, %T% h\n");
   __dprintf(0, "** log.c: BK LastPatch  : %E%, %U% h\n");
}


void SetDebugPath(const char *dft, const char *pre)
{
   memset(_debugpath, 0, sizeof(_debugpath));
   if (pre && * pre)
      strcpy(_debugpath, pre);
   else if (dft && *dft)
      strcpy(_debugpath, dft);
}


char *DebugFilename(const char *fn)
{
   static char buf[256+1];

   memset(buf, 0, sizeof(buf));
   if (_debuglevel > 0 && fn && *fn)
   {
      strcpy(buf, _debugpath);
      if (buf[strlen(buf)-1] != '/')
         strcat(buf, "/");
      strcat(buf, fn);
      dprintf(4, "DebugFilename(%s)=%s\n", fn, buf);
   }

   return (*buf ? buf : NULL);
}
