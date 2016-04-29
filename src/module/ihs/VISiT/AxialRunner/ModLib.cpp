#include "AxialRunner.h"
#include "../lib/General/include/log.h"

int CheckUserFloatVectorValue2(coFloatVectorParam *v, float old,
float min, float max, float *dest,int c);

char *AxialRunner::IndexedParameterName(const char *name, int index)
{
   char buf[255];

   sprintf(buf, "%s__%d_", name, index + 1);;
   return strdup(buf);
}


int AxialRunner::SetFloatDoubleVector(coFloatVectorParam *v, float val0, float val1)
{
   float vv[2];
   vv[0] = val0; vv[1] = val1;
   v->setValue(2,vv);
   return 1;
}


#ifdef GRAD_SLIDERS
int AxialRunner::CheckUserRadSliderValue(coFloatSliderParam *f, float old,
float min, float max, float *dest)
{
   int changed = 0;

   if (old != RAD(f->getValue()))
   {
      if (f->getValue() > max || f->getValue() < min)
      {
         sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", f->getName(), min, max);
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

int AxialRunner::CheckUserFloatSliderValue(coFloatSliderParam *f, float old,
float min, float max, float *dest)
{
   int changed = 0;

   if (old != f->getValue())
   {
      if (f->getValue() > max || f->getValue() < min)
      {
         sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", f->getName(), min, max);
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


int AxialRunner::CheckUserFloatSliderAbsValue(coFloatSliderParam *f, float old, float scale,
float min, float max, float *dest)
{
   int changed = 0;

   float abs_old;

   abs_old = scale*old;
   if (abs_old != f->getValue())
   {
      if (f->getValue() > max || f->getValue() < min)
      {
         sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", f->getName(), min, max);
         *dest = old;
         f->setValue(abs_old);
      }
      else
      {
         *dest = f->getValue()/scale;
         changed = 1;
      }
   }
   return changed;
}


int AxialRunner::CheckUserFloatAbsValue(coFloatParam *f, float old, float scale,
float min, float max, float *dest)
{
   int changed = 0;

   float abs_old;

   abs_old = scale*old;
   if ((abs_old) != f->getValue())
   {
      if (f->getValue() > max || f->getValue() < min)
      {
         sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", f->getName(), min, max);
         *dest = old;
         f->setValue(abs_old);
      }
      else
      {
         *dest = f->getValue()/scale;
         changed = 1;
      }
   }
   return changed;
}


int AxialRunner::CheckUserFloatValue(coFloatParam *f, float old,
float min, float max, float *dest)
{
   int changed = 0;

   if (old != f->getValue())
   {
      if (f->getValue() > max || f->getValue() < min)
      {
         sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)", f->getName(), min, max);
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


int AxialRunner::CheckUserBooleanValue(coBooleanParam *b, int old,
int min, int max, int *dest)
{
   int changed = 0;

   if (old != (int)(b->getValue()))
   {
      /* hmm was sol das? if (b->getValue() > max || b->getValue() < min)
      {
         char errtext[255];

         sprintf(errtext, "The input value for %s is out of range: min=%d,max=%d (--> reset to old value)", b->getName(), min, max);
         sendError(errtext);
         *dest = old;
         b->setValue(old);
      }
      else */
      {
         *dest = (int)(b->getValue());
         changed = 1;
      }
   }
   return changed;
}


int AxialRunner::CheckUserFloatVectorValue(coFloatVectorParam *v, float *old, float scale,
float min, float max, float *dest,int c)
{
   int changed = 0;
   int i;

   float abs_old;

   dprintf(4,"AxialRunner::CheckUserFloatVectorValue()\n");
   dprintf(4,"v(IN)  : %f, %f\n", v->getValue(0), v->getValue(1));
   dprintf(4,"old(IN): %f, %f\n", old[0], old[1]);
   for(i = 0; i < c; i++)
   {
      abs_old = scale*old[i];
      if (abs_old != v->getValue(i))
      {
         if (v->getValue(i) > max || v->getValue(i) < min)
         {
            sendError("The input value for %s is out of range: min=%f,max=%f (--> reset to old value)",
                      v->getName(), min, max);
            dest[i] = old[i];
            v->setValue(i,abs_old);
         }
         else
         {
            dest[i] = v->getValue(i)/scale;
            changed = 1;
         }
      }
   }
   return changed;
}


int AxialRunner::CheckUserFloatVectorValue2(coFloatVectorParam *v, float old,
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


// !!! Different from RadialRunner::CheckUserChoiceValue2 !!!
int AxialRunner::CheckUserChoiceValue( coChoiceParam *c, int max)
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


int AxialRunner::CheckUserIntValue( coIntScalarParam *i, int old, int min, int max, int *dest)
{
   int changed = 0;

   if (old != i->getValue())
   {
      if (i->getValue() > max || i->getValue() < min)
      {
         sendError("The input value for %s is out of range: min=%d,max=%d (--> reset to old value)", i->getName(), min, max);
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


int AxialRunner::CheckDiscretization(coFloatVectorParam *v, int *dis, float *bias, int *type)
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


int AxialRunner::SplitPortname(const char *portname, char *name, int *index)
{
   int offs = strlen(portname);
   int ret = 0;

   *name = '\0';
   *index = -1;
   dprintf(1, "Entering AxialRunner::SplitPortname(%s, %s, %d)\n",
      portname, name, *index);
   if (portname[--offs] == '_')
   {
      while (strchr("0123456789", portname[--offs]))
         ;
      if (portname[offs] == '_' && portname[offs-1] == '_')
      {
         strncpy(name, portname, offs-1);
         name[offs-1] = '\0';
         *index = atoi(portname+offs+1) - 1;
         ret = 1;
      }
   }
   else
      strcpy(name, portname);
   dprintf(1, "Leaving AxialRunner::SplitPortname(%s, %s, %d) with ret = %d\n",
      portname, name, *index, ret);
   return ret;
}
