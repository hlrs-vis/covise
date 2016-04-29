#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "include/cfg.h"
#include "include/log.h"
#ifdef   DEMO
#define  EXTERN
#endif

#define  NONE_ERROR  0
#define  OPEN_ERROR  1
#define  RAM_ERROR   2

#define  CFG_CUT  '='
#define  COMMENT  '#'

static int splitline(char *buf, char *key, char *value);
static int WhiteSpace(char c);
static int IHS_AllocCFG(void);
static int IHS_ReadCFG(const char *fn);
static int FindCFGIndex(const char *fn);
static int _FindCFGIndex(const char *fn);
static void DumpIHSCfg(void);
#ifdef   DEMO
static void print_cfg(void);
#endif

struct ihs_icfg
{
   char *section;
   char *key;
   char *value;
};

struct ihs_cfg
{
   int num;
   char *filename;
   struct ihs_icfg **icfg;
};

#define  NUM_CFG  50
static struct ihs_cfg *cfg[NUM_CFG];

char *IHS_GetCFGValue(const char *file, const char *section, const char *key)
{
   int i;
   int icfg = -1;
   char *ms = NULL;
   char *mk = NULL;
   char *p;
   char *res = NULL;

   if ((icfg = FindCFGIndex(file)) >= 0)
   {
      if (section && *section)
      {
         ms = strdup(section);
         p = ms;
         while (*p++)
            *(p-1) = tolower(*(p-1));
      }
      if (key && *key)
      {
         mk = strdup(key);
         p = mk;
         while (*p++)
            *(p-1) = tolower(*(p-1));
      }

      if (cfg+icfg && cfg[icfg]->num)
      {
         for (i = 0; i< cfg[icfg]->num; i++)
         {
            if ((((ms && *ms && !strcmp(cfg[icfg]->icfg[i]->section , ms)))
               || (!ms && !cfg[icfg]->icfg[i]->section))
               && (mk && *mk && !strcmp(cfg[icfg]->icfg[i]->key, mk)))
               res = strdup(cfg[icfg]->icfg[i]->value);
         }
      }
      if (ms)  free(ms);
      if (mk)     free(mk);
   }
   else
      dprintf(0 , (char *)"Couldn't process file: %s\n", file);

   return res;
}


static int FindCFGIndex(const char *fn)
{
   int ind, error;

   if ((ind = _FindCFGIndex(fn)) < 0)
   {
      if ((error = IHS_ReadCFG(fn)) != NONE_ERROR)
         dprintf(0, (char *)"File open error: %s (ihserror = %d)\n", fn, error);
      ind = _FindCFGIndex(fn);
   }
   return ind;
}


static int _FindCFGIndex(const char *fn)
{
   int i;

   for (i = 0; i < NUM_CFG; i++)
   {
      if (cfg[i] && !strcmp(fn, cfg[i]->filename))
      {
         return i;
      }
   }
   return -1;
}


static int IHS_ReadCFG(const char *fn)
{
   FILE *fp;
   char buf[200];
   int error;
   char key[150];
   char value[150];
   int line = 0;
   char *section = NULL;
   char *p;
   int cfgi;
   struct ihs_cfg *c;

   error = NONE_ERROR;
   if ((fp = fopen(fn, "r")) == NULL)
   {
      error = OPEN_ERROR;
   }
   else
   {
      if ((cfgi = IHS_AllocCFG()) < 0 )
         return RAM_ERROR;
      c = cfg[cfgi];
      c->filename = strdup(fn);
      while (buf == fgets(buf, sizeof(buf), fp))
      {
         line++;
         if (splitline(buf, key, value))
         {
            if (*value && *key)
            {
               int ind = c->num++;

               p = key;
               while (*p++)
                  *(p-1) = tolower(*(p-1));
               c->icfg = (struct ihs_icfg **)realloc(c->icfg, c->num*sizeof(struct ihs_icfg *));
               c->icfg[ind] = (struct ihs_icfg *)calloc(1, sizeof(struct ihs_icfg));
               if (c->icfg)
               {
                  if (section && *section)
                     c->icfg[ind]->section = strdup(section);
                  c->icfg[ind]->key = strdup(key);
                  c->icfg[ind]->value = strdup(value);
               }
            }
            else if (*key)
            {

               p = key;
               while (*p++)
                  *(p-1) = tolower(*(p-1));
               if (section)   free(section);
               section = (char *)strdup(key);
            }
         }
         else
            dprintf(0, "Line %d NOT ok!\n", line);
      }
   }
   if (section)   free(section);
   DumpIHSCfg();
   return error;
}


static int splitline(char *buf, char *key, char *value)
{
   char *mbuf = NULL;
   char *s;
   char *p = NULL;
   char *d;
   int len;

   if ((d = strrchr(buf, '\n')) != NULL)
      *d = '\0';
   dprintf(10, "Entering splitline(buf=%s)\n", buf);
   *key = *value = '\0';

   if ((len = strlen(buf)+1) == 1)
      return 1;
   if ((p = (char *)calloc(len, sizeof(char))) == NULL)
      return 0;
   if ((mbuf = (char *)calloc(len, sizeof(char))) == NULL)
   {
      free(p);
      return 0;
   }
   /* first we destroy all comments	*/
   if (*buf != COMMENT)
   {
      d = mbuf;
      s = buf;
      *d++ = *buf;
      while (*++s)
      {
         if (*s == '\\' && *(s+1) == COMMENT)
         {
            *d++ = '#';
            s++;
         }
         else if (*s == COMMENT)
         {
            *d = '\0';
            break;
         }
         else if (*s == '\n')
            ;
         else
            *d++ = *s;
      }

      if (*mbuf)
      {
         /* ok, first we kick all leading white space ...	*/
         s = mbuf;
         d = p;

         while (*s && WhiteSpace(*s))
            s++;
         /* we do a little copy ...	*/
         while (*s && *s != CFG_CUT)
            *d++ = *s++;
         /* all white space before = should be killed ...	*/
         if (*p)
         {
            while (WhiteSpace(*--d))
               *d = '\0';
            strcpy(key, p);
         }

         if (!s)
            dprintf(0, "ERROR: no value: buf=%s, d=%s, key=%s", buf, d, key);
         if (s && *s)                             /* Section || key = value ??	*/
         {
            memset(p, 0, len);

            while (WhiteSpace(*++s))
               ;
            strcpy(p, s);
            s = p+strlen(p)-1;
            while (WhiteSpace(*s))
               *s-- = '\0';
            strcpy(value, p);
         }
      }
   }
   if (p)      free(p);
   if (mbuf)   free(mbuf);
   dprintf(10, "Leaving splitline(buf=%s, key=%s,value=%s)\n", buf, key, value);
   return 1;
}


static int WhiteSpace(char c)
{
   return (c == ' ' || c == '\t');
}


static int IHS_AllocCFG(void)
{
   int i;

   for (i = 0; i < NUM_CFG; i++)
   {
      if (!cfg[i])
      {
         cfg[i] = (struct ihs_cfg *) calloc(1, sizeof(struct ihs_cfg));
         cfg[i]->icfg = (struct ihs_icfg **)calloc(1,sizeof(struct ihs_icfg*));
         cfg[i]->num = 0;

         return i;
      }
   }
   return -1;
}


#define  MAX_ENTRIES 2000
char **dump_cfg(void)
{
   int i, j;
   struct ihs_cfg *c;
   int numbuf;
   char **buf;
   char tmp[200];

   buf = (char **)calloc(MAX_ENTRIES+1, sizeof(char *));
   for (j = 0, numbuf = 0; j < NUM_CFG; j++)
   {
      if ((c = cfg[j]) != NULL)
      {
         dprintf(2, "Dump of ihs_cfg:\nFile: %s\tnum : %d\n", c->filename, c->num);
         for (i = 0; i< c->num; i++)
         {
            sprintf(tmp, "%s.%s=%s",
               c->icfg[i]->section, c->icfg[i]->key,
               c->icfg[i]->value);
            buf[numbuf++] = strdup(tmp);
            if (numbuf == MAX_ENTRIES)
            {
               dprintf(0, "Space (%s, %d)", __FILE__, __LINE__);
               exit(1);
            }
         }
      }
   }
   return buf;
}


#ifdef   DEMO

int main(int argc, char **argv)
{
#ifdef   IHS_DEBUG
   ihs_debug = NULL;
#endif
   printf("Starting programm ...\n");
   IHS_GetCFGValue("a.cfg", NULL, NULL);
   IHS_GetCFGValue("b.cfg", NULL, NULL);
   IHS_GetCFGValue("c.cfg", NULL, NULL);
   DumpIHSCfg();
   return 1;
}
#endif

static void DumpIHSCfg(void)
{
   int i, j;
   struct ihs_cfg *c;
   char sec[200];
   char key[200];
   char val[200];

   for (j = 0; j < NUM_CFG; j++)
   {
      if ((c = cfg[j]) != NULL)
      {
         dprintf(6, "Dump of ihs_cfg: file: %s  num : %d\n", c->filename, c->num);
         for (i = 0; i< c->num; i++)
         {
            sprintf(sec, "Section: <%s>", c->icfg[i]->section);
            sprintf(key, "Key: <%s>", c->icfg[i]->key);
            sprintf(val, "Value: <%s>", c->icfg[i]->value);
            dprintf(6, "  %-40s %-40s %-40s\n", sec, key, val);
         }
      }
   }
}
