#include <math.h>
#include <string.h>
#include <stdio.h>

#include "ELIST.H"
#include "numeric.h"


/*----------------------------------------------------------------*/
/* elist::uncompiled_value: Wertet den Ausdruck numerisch aus, in */
/*                          dem der Funktionsbaum rekursiv        */
/*                          abgearbeitet wird                     */
/*----------------------------------------------------------------*/


double HlExprList::evalF()
{
	double h;
	double *valptr;
	int i;

	switch( typeOfHead() ) {

		case NUMBER:
			return getDoubleVal();

		case IDENT:
			return getsym()->getVal();

		case PLUS:
			for( i=0,h=0; i<Length(); i++ )
				h += arg(i)->evalF();
			return h;

		case TIMES:
			for( i=0,h=1; i<Length(); i++ )
				h *= arg(i)->evalF();
			return h;

		case POWER:
			h = Second()->evalF();
			return pow(First()->evalF(), h);

		case LESS:
			return First()->evalF() < Second()->evalF();

		case GREATER:
			return First()->evalF() > Second()->evalF();

		case LESSEQUAL:
			return First()->evalF() <= Second()->evalF();

		case GREATEREQUAL:
			return First()->evalF() >= Second()->evalF();

		case EQUAL:
			return First()->evalF() == Second()->evalF();

		case SQRT:
			return sqrt( First()->evalF() );

		case SIN:
			return sin( First()->evalF() );

		case COS:
			return cos( First()->evalF() );

		case TAN:
			return tan( First()->evalF() );

		case ASIN:
			h = asin( First()->evalF() );
			return h;

		case ACOS:
			h = acos( First()->evalF() );
			return h;

		case ATAN:
			return atan( First()->evalF() );

		case SINH:
			return sinh( First()->evalF() );

		case COSH:
			return cosh( First()->evalF() );

		case TANH:
			return tanh( First()->evalF() );

		case ASINH:
			h = First()->evalF();
			h = log(h+sqrt(h*h + 1));
			return h;

		case ACOSH:
			h = First()->evalF();
			h = log(h+sqrt(h*h - 1));
			return h;

		case ATANH:
			h = First()->evalF();
			h = log(sqrt((1+h)/(1-h)));
			return h;

		case EXP:
			return exp( First()->evalF() );

		case LOG:
			h = log( First()->evalF() );
			return h;

		case FRESNELSIN:
			return HlFresnelSinus( First()->evalF() );

		case FRESNELCOS:
			return HlFresnelCosinus( First()->evalF() );

		case ERF:
			return HlErf( First()->evalF() );

		case SI:
			return HlSi( First()->evalF() );

		case SINC:
			return HlSinc( First()->evalF() );

		case ABS:
			return HlBetrag( First()->evalF(), int(Second()->evalF()) );

		case SIGN:
			return HlSignum( First()->evalF(), int(Second()->evalF()) );

		case ROMBERG:
		case DEFINT:
			valptr = Second()->getsym()->getValuePtr();
			return First()->Romberg( valptr, 0, Third()->evalF(), 10 );

		case IFF:
			return First()->evalF() ? Second()->evalF() : Third()->evalF();

		default:
			break;
	}
	return 10000000.0;
}



/*-------------------------------------------------------------------------*/
/* elist::assign: Zuweisungsoperator fuer Variable                         */
/*                Einer Variablen kann 3 Arten von Werten enthalten:       */
/*                1. Ungebunden  -> Symboltabelle.Expressionpointer = NULL */
/*                2. Zahl        -> Symboltabelle.Expressionpointer = NULL */
/*                                  Wertfeld auf Zahlenwert                */
/*                3. Ausdruck    -> Symboltabelle.Expressionpointer = expr */
/*-------------------------------------------------------------------------*/

HlExprList* HlExprList::assign(HlExprList *e)
{

	if( getsym() == e->getsym() )
		getsym()->setExprPtr( NULL );

	else

		if( e->NumberQ() ) {
			getsym()->setVal( e->getDoubleVal() );
			getsym()->setExprPtr( e );
		}

		else
			getsym()->setExprPtr( N(e) );

	return this;
}

/*----------------------------------------------------------------------*/
/* elist::eval: Wertet einen Ausdruck rekursiv aus.                     */
/*----------------------------------------------------------------------*/

HlExprList* HlExprList::eval()
{

	HlExprList *q;

	for( int i=0; i<Length(); i++ )
		if( !(Is(ASSIGN) && i==0) && !Is(SELF) && !Is(FREE) )
			setarg(i, arg(i)->eval());


	switch( typeOfHead() ) {

		case SELF:
			return First();

		case IDENT:
			q = getsym()->getExprPtr();

			if( "%" == getsym()->getLexem() ) {
				return q->eval();
			}

			if( q ){
				if( !q->NumberQ() ) {
					return q->eval();
				}
			}

			return this;


		case ASSIGN:
			return First()->assign(Second());

		case KOMM:
			return First()->Order();

		case FREE:
			First()->getsym()->setExprPtr( NULL );
			return First();

		case ASSO:
			return First()->Flat();

		case DIST:
			if( Length() == 2 )
				return Distribute( First()->typeOfHead(), Second() );
			else if( Length() > 2 )
				return Distribute( First()->typeOfHead(), Second(), arg(2) );

		case DIFF:
			return this->diff();

		case SIMPLIFY:
			return First()->simplify();

		case VAL:
			return N( First()->evalF() );

		case COMP:
			return N( (double)First()->compare(Second()) );

      default:
         break;
	}
	return this;
}


double HlExprList::Romberg(double *var, double a, double b, int N)
{
	int i;
	if(N>14) N=14;

	double T[15];

	unsigned int w = 1;
	for (i=0; i <= N; i++)
	{
		double temp = 0.;
		double h = (b-a)/w;
		double h2 = h/2.0;

		*var = a;
		double f0 = evalF();
		for (unsigned int v = 1; v <= w; v++)
		{
			*var = a+v*h;
			double f1 = evalF();
			temp += h2 * ( f0 + f1 );
			f0 = f1;
		}
		w *= 2;
		T[i] = temp;
	}

	double nenner = 4.0;
	for (int k = 1; k <= N; k++)
	{ 
		double ti1 = T[i-1];
		for (i = k; i <= N; i++) 
		{
			double ti = T[i];
			T[i] = ti + ( ti - ti1 ) / (nenner-1);
			ti1 = ti; 
		}
		nenner *= 4.0;
	}

	return T[N];
}



double HlExprList::RiemannSumme(double *var, double a, double b, int N)
{
	if(N>100) N=100;

	double h = (b-a)/double(N);
	double sum = 0;

	for (int i = 0; i < N; i++)
	{
		*var = a+i*h;
		double f = evalF();
		sum += f;
	}

	return sum * h;
}

