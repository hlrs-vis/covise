#include "RadialRunner.h"

char *RadialRunner::IndexedParameterName(const char *name, int index)
{
   char buf[255];

   sprintf(buf, "%s__%d_", name, index + 1);
   return strdup(buf);
}


int RadialRunner::CheckUserFloatSliderValue(coFloatSliderParam *f, float old,
float min, float max, float *dest)
{
   int changed = 0;

   if (old != f->getValue())
   {
      if (f->getValue() > max || f->getValue() < min)
      {
         coModule::sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", f->getName(), min, max);
         *dest = old;
         f->setValue(old);
      }
      else
      {
         *dest = f->getValue();
         changed = 1;
      }
   }
   return changed;
}


int RadialRunner::CheckUserFloatValue(coFloatParam *f, float old,
float min, float max, float *dest)
{
   int changed = 0;

   if (old != f->getValue())
   {
      if (f->getValue() > max || f->getValue() < min)
      {
         coModule::sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", f->getName(), min, max);
         *dest = old;
         f->setValue(old);
      }
      else
      {
         *dest = f->getValue();
         changed = 1;
      }
   }
   return changed;
}


int RadialRunner::CheckUserFloatVectorValue(coFloatVectorParam *v, float *old,
float min, float max, float *dest,int c)
{
   int changed = 0;
   int i;
   for(i = 0; i < c; i++)
   {
      if (old[i] != v->getValue(i))
      {
         if (v->getValue(i) > max || v->getValue(i) < min)
         {
            sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", v->getName(), min, max);
            dest[i] = old[i];
            v->setValue(i,old[i]);
         }
         else
         {
            dest[i] = v->getValue(i);
            changed = i+1;
         }
      }
   }
   return changed;
}


int RadialRunner::CheckUserFloatVectorValue2(coFloatVectorParam *v, float old,
float min, float max, float *dest,int c)
{
   int changed = 0;

   if (old != v->getValue(c))
   {
      if (v->getValue(c) > max || v->getValue(c) < min)
      {
         coModule::sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", v->getName(), min, max);
         *dest = old;
         v->setValue(c,old);
      }
      else
      {
         *dest = v->getValue(c);
         changed = c+1;
      }
   }
   return changed;
}


int RadialRunner::CheckUserIntValue( coIntScalarParam *i, int old, int min, int max, int *dest)
{
   int changed = 0;

   if (old != i->getValue())
   {
      if (i->getValue() > max || i->getValue() < min)
      {
         coModule::sendError("The input value for %s is out of range: min=%d,max=%d (--> reset to old value)", i->getName(), min, max);
         *dest = old;
         i->setValue(old);
      }
      else
      {
         *dest = i->getValue();
         changed = 1;
      }
   }
   return changed;
}


int RadialRunner::CheckUserChoiceValue( coChoiceParam *c, int max)
{
   if(c->getValue() >= max)
   {
      sendError("Selection %d is out of valid range (%d)! Set to max. value!",
         c->getValue(), max);
      // moves param. to end of menu if modified ???!!!
      c->setValue(max);
      return 0;
   }
   else
      return 1;
}


// !!! Different from AxialRunner::CheckUserChoiceValue !!!
int RadialRunner::CheckUserChoiceValue2( coChoiceParam *i, int old, int min, int max, int *dest)
{
   int changed = 0;

   if (old != i->getValue())
   {
      if (i->getValue() > max || i->getValue() < min)
      {
         coModule::sendError("The input value for %s is out of range: min=%d,max=%d (--> reset to old value)", i->getName(), min, max);
         *dest = old;
         i->setValue(old);
      }
      else
      {
         *dest = i->getValue();
         changed = 1;
      }
   }
   return changed;
}


int RadialRunner::CheckDiscretization(coFloatVectorParam *v, int *dis, float *bias, int *type)
{
   int changed = 0;
   float dum, dis_min;

   if( (v->getValue(2)) == (float)(2)) dis_min = (float)(4);
   else dis_min = (float)(3);

   // check discretization
   dum = (float)(*dis);
   changed = CheckUserFloatVectorValue2(v, dum,
      dis_min,100,&dum,0);
   (*dis) = (int)dum;
   // check type
   dum = (float)(*type);
   changed = CheckUserFloatVectorValue2(v, dum,
      0,2,&dum,2);
   (*type) = (int)dum;
   // set bias without check
   (*bias) = v->getValue(1);

   return changed;
}


int RadialRunner::SplitPortname(const char *portname, char *name, int *index)
{
   int offs = strlen(portname);

   *name = '\0';
   *index = -1;
   if (portname[--offs] == '_')
   {
      while (strchr("0123456789", portname[--offs]))
         ;
      if (portname[offs] == '_' && portname[offs-1] == '_')
      {
         strncpy(name, portname, offs-1);
         name[offs-1] = '\0';
         *index = atoi(portname+offs+1) - 1;
         return 1;
      }
   }
   else
      strcpy(name, portname);
   return 0;
}
