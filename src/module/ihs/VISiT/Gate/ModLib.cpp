#include "Gate.h"

#ifdef GRAD_SLIDERS
int Gate::CheckUserRadSliderValue(coFloatSliderParam *f, float old,
float min, float max, float *dest)
{
   int changed = 0;

   if (old != RAD(f->getValue()))
   {
      if (f->getValue() > max || f->getValue() < min)
      {
         char errtext[255];

         sprintf(errtext, "The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", f->getName(), min, max);
         coModule::sendError(errtext);
         *dest = old;
         f->setValue(GRAD(old));
      }
      else
      {
         *dest = RAD(f->getValue());
         changed = 1;
      }
   }
   return changed;
}
#endif                                            // GRAD_SLIDERS

int Gate::CheckUserFloatSliderValue(coFloatSliderParam *f, float old,
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


int Gate::CheckUserFloatValue(coFloatParam *f, float old,
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


int Gate::CheckUserFloatVectorValue(coFloatVectorParam *f, int index, float old,
float min, float max, float *dest)
{
   int changed = 0;

   if (old != f->getValue(index))
   {
      if (f->getValue(index) > max || f->getValue(index) < min)
      {
         coModule::sendError("The input value for %s[%d] is out of range: min=%f,max=%f (--> reset to old value)", f->getName(), index, min, max);
         *dest = old;
         f->setValue(index, old);
      }
      else
      {
         *dest = f->getValue(index);
         changed = 1;
      }
   }
   return changed;
}


int Gate::CheckUserIntValue( coIntScalarParam *i, int old, int min, int max, int *dest)
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


int Gate::SplitPortname(const char *portname, char *name, int *index)
{
   int offs = strlen(portname);

   *name = '\0';
   *index = -1;
   if (portname[--offs] == ']')
   {
      while (strchr("0123456789", portname[--offs]))
         ;
      if (portname[offs] == '[' && portname[offs-1] == ' ')
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
