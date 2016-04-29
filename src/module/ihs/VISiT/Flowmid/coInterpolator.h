#include "api/coModule.h"
#define SUCCESS 1
#define ERROR 0

class coInterpolator
{
   public:                                        //virtueller Konstruktor? Evtl. Abstrakte Basis-Klasse, private-Mehrfach-Vererbung
      virtual int getScalarValue(double, double, double, double *) = 0;
      virtual int getFieldValue(double, double, double, double *) = 0;
      virtual void setTargetArea(coDoPolygons *, coDoIntArr *) = 0;
      virtual void writeInfo(char *) = 0;
      virtual string getType() = 0;               //evtl. const char * statt string, fuer Vergleich dann string str(const char *);
      virtual ~coInterpolator() {};
};
