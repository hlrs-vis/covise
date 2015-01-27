#include <string>

#include "parser.h"

HlSymbolTable* SYMTABPTR = NULL;

/*-------------------------------------------------------------*/
/* Memberfunktionen der Klasse parser                          */
/*-------------------------------------------------------------*/


HlParser::HlParser()
{
}


/*-------------------------------------------------------------*/


HlParser::~HlParser()
{
}


/*-------------------------------------------------------------*/


void HlParser::fillSymTab()
{
	static const double PI = 3.14159265358979323846;

	mSymTab->insert( "", EOS, "EOS" );

	mSymTab->insert( "(", RIGHTB );
	mSymTab->insert( ")", LEFTB );
	mSymTab->insert( ",", KOMMA );

	mSymTab->insert( "-", MINUS,  "Minus" );
	mSymTab->insert( "/", DIV,    "Div" );
	mSymTab->insert( "umin", UMIN );

	mSymTab->insertInfix( "<"  ,LESS,         "Less",    false,  false,  2 );
	mSymTab->insertInfix( ">"  ,GREATER,      "Greater", false,  false,  2 );
	mSymTab->insertInfix( "<=" ,LESSEQUAL,    "le",      false,  false,  2 );
	mSymTab->insertInfix( ">=" ,GREATEREQUAL, "ge",      false,  false,  2 );
	mSymTab->insertInfix( "==" ,EQUAL,        "eq",      false,  false,  2 );
	
	mSymTab->insertInfix( "*" ,TIMES,   "Times",  true,  true,  NARGS );
	mSymTab->insertInfix( "+", PLUS,    "Plus",   true,  true,  NARGS );
	mSymTab->insertInfix( "%", POWER,   "Power",  false, false, 2     );
	mSymTab->insertInfix( "^", POWER,   "Power",  false, false, 2     );
	mSymTab->insertInfix( "=", ASSIGN,  "Assign", false, false, 2     );
	mSymTab->insertInfix( "~", OP,      "~",      false, false, 2     );

	mSymTab->insertFunction( "sqrt",  SQRT,  1 );
	mSymTab->insertFunction( "sin",   SIN,   1 );
	mSymTab->insertFunction( "cos",   COS,   1 );
	mSymTab->insertFunction( "tan",   TAN,   1 );
	mSymTab->insertFunction( "asin",  ASIN,  1 );
	mSymTab->insertFunction( "acos",  ACOS,  1 );
	mSymTab->insertFunction( "atan",  ATAN,  1 );
	mSymTab->insertFunction( "exp",   EXP,   1 );
	mSymTab->insertFunction( "log",   LOG,   1 );
	mSymTab->insertFunction( "sinh",  SINH,  1 );
	mSymTab->insertFunction( "cosh",  COSH,  1 );
	mSymTab->insertFunction( "tanh",  TANH,  1 );
	mSymTab->insertFunction( "asinh", ASINH, 1 );
	mSymTab->insertFunction( "acosh", ACOSH, 1 );
	mSymTab->insertFunction( "atanh", ATANH, 1 );
	mSymTab->insertFunction( "abs",   ABS,   NARGS );
	mSymTab->insertFunction( "sign",  SIGN,  NARGS );
	mSymTab->insertFunction( "fresnelsin", FRESNELSIN, 1 );
	mSymTab->insertFunction( "fresnelcos", FRESNELCOS, 1 );
	mSymTab->insertFunction( "erf", ERF, 1 );
	mSymTab->insertFunction( "sinc", SINC, 1 );
	mSymTab->insertFunction( "si", SI, 1 );
	mSymTab->insertFunction( "romberg", ROMBERG, 3 );
	mSymTab->insertFunction( "defint",  DEFINT, 3 );
	mSymTab->insertFunction( "iff",  IFF, 3 );

	mSymTab->insertFunction( "simp",   SIMPLIFY, 1 );
	mSymTab->insertFunction( "komm",   KOMM,     1 );
	mSymTab->insertFunction( "asso",   ASSO,     1 );
	mSymTab->insertFunction( "val",    VAL,      1 );
	mSymTab->insertFunction( "post",   POST,     1 );
	mSymTab->insertFunction( "free",   FREE,     1 );
	mSymTab->insertFunction( "diffop", DIFFOP,   1 );

	mSymTab->insertFunction( "diff",   DIFF,     2 );
	mSymTab->insertFunction( "comp",   COMP,     2 );
	mSymTab->insertFunction( "dist",   DIST, NARGS );

	mSymTab->insert( "pi", IDENT, "Pi" );
	mSymTab->lookup("pi")->setVal(PI);
	mSymTab->insert( "@", NUMBER );

	mSymTab->insert( "'", SELF, "SELF" );
	mSymTab->lookup(SELF)->setProp( PREFIX );
	mSymTab->lookup(SELF)->setArgn(  1 );
}


/*-------------------------------------------------------------*/


HlBaseList *HlParser::parseString(const string& s)
{
	HlBaseList *e = NULL;
	mError->setNoError();

	try
	{
		mLexer.setInput(s);
		mLookahead = mLexer.nextToken();

		e = assignment();
		match(EOS);
	}

	catch( HlError* Err )
	{
		delete e;
		mError->setState(*Err);

		delete Err;
		return NULL;
	}

	return e;
}


/*-------------------------------------------------------------*/


HlBaseList *HlParser::assignment()
{
	HlBaseList *e = NULL;
	HlBaseList *q = NULL;

	e = q = comparisation();

	try
	{
		if(mLookahead->tokenIn(ASSIGN)) 
		{
			if(e->Is(IDENT) ) 
			{
				match(mLookahead);
				q = new HlBaseList(ASSIGN);
				q->apparg(e);
				q->apparg(comparisation());
			}
			else 
				throw new HlError(1, "HlParser::assignment: Rechte Seite kein IDENT" ); ;
		}
	}

	catch( HlError *err )
	{
		delete q;
		throw( err );
	}

	return q;
}


/*-------------------------------------------------------------*/

HlBaseList *HlParser::comparisation()
{
	HlBaseList *q = NULL;

	q = expression();

	try
	{
		if(mLookahead->tokenIn(LESS,GREATER,LESSEQUAL,GREATEREQUAL,EQUAL))
		{
			HlBaseList *e = q;
			q = new HlBaseList(mLookahead->getToken());
			q->apparg(e);
			match(mLookahead);
			q->apparg(expression());
		}
	}

	catch( HlError *err )
	{
		delete q;
		throw( err );
	}

	return q;
}


/*-------------------------------------------------------------*/

HlBaseList *HlParser::expression()
{
	HlBaseList *q = NULL;

	q = term();

	try
	{
		if(mLookahead->tokenIn(PLUS,MINUS))
		{
			HlBaseList *e = q;
			q = new HlBaseList(PLUS);
			q->apparg(e);

			while(mLookahead->tokenIn(PLUS,MINUS)) 
			{
				if(mLookahead->tokenIn(PLUS)) 
				{
					match(mLookahead);
					q->apparg(term());
				}

				else 
				{
					match( mLookahead );
					HlBaseList *h = q->apparg(TIMES);
					h->apparg(-1.0);
					h->apparg(term());
				}
			}
		}
	}

	catch( HlError *err )
	{
		delete q;
		throw( err );
	}

	return q;
}


/*-------------------------------------------------------------*/


HlBaseList *HlParser::term()
{
	HlBaseList *q = NULL;

	q = power();

	try
	{
		if(mLookahead->tokenIn(TIMES,DIV))
		{
			HlBaseList *e = q;
			q = new HlBaseList(TIMES);
			q->apparg(e);
			while(mLookahead->tokenIn(TIMES,DIV))
			{
				if(mLookahead->tokenIn(TIMES)) 
				{
					match(mLookahead);
					q->apparg(power());
				}
				else 
				{
					match(mLookahead);
					HlBaseList *h = q->apparg(POWER);
					h->apparg(power());
					h->apparg(-1.0);
				}
			}
		}
	}

	catch( HlError *err )
	{
		delete q;
		throw( err );
	}

	return q;
}


/*-------------------------------------------------------------*/


HlBaseList *HlParser::power()
{
	HlBaseList *e = NULL;
	HlBaseList *q = NULL;

	q = factor();

	try
	{
		while(mLookahead->tokenIn(POWER)) 
		{
			e = q;
			q = new HlBaseList(mLookahead);
			q->apparg(e);
			match(mLookahead);
			q->apparg(factor());
		}

	}

	catch( HlError *err )
	{
		delete q;
		throw( err );
	}
	
	return q;
}

 
/*-------------------------------------------------------------*/


HlBaseList *HlParser::factor()
{
	HlBaseList *e = NULL;
	HlBaseList *h = NULL;

	if(NULL == mLookahead)
		throw(  new HlError( 1, string("HlParser::faktor: NULL Pointer fuer lookahead")) );

	if(mLookahead->getProp(PREFIX)) 
	{
		int n = mLookahead->getArgn();
		int b = mLookahead->getProp(BRACKET);

		e = new HlBaseList(mLookahead);
		match(mLookahead);
		if(b) match(RIGHTB);

		if( n != NARGS ) 
			for(int i=0; i<n; i++) 
			{
				if(b)
					e->apparg(comparisation());
				else
					e->apparg(factor());
				if(i < n-1) match(KOMMA);
			}

		else {
			if(e->Is(SIGN) || e->Is(ABS)) {
				h = comparisation();
				if(mLookahead->tokenIn(KOMMA)) {
					match(KOMMA);
					e->apparg(comparisation());
					e->apparg(h);
				}
				else {
					e->apparg(h);
					e->apparg(0.0);
				}
			}
			else
				for(;;) 
				{
					e->apparg(comparisation());
					if(!mLookahead->tokenIn(KOMMA)) break;
					match(KOMMA);
				}
		}

		if(b) match(LEFTB);
		return e;
	}

	else
	{
		switch(mLookahead->getToken()) 
		{

			case NUMBER:
				e = new HlBaseList(mLookahead);
				match(NUMBER);
				return e;

			case IDENT:
				e = new HlBaseList(mLookahead);
				match(IDENT);
				return e;

			case RIGHTB:
				match(RIGHTB);
				e = comparisation();
				match(LEFTB);
				return e;

			case UMIN:
				match(UMIN);
				if( mLookahead->tokenIn(NUMBER) ) 
				{
					e = new HlBaseList( -mLookahead->getVal() );
					match( mLookahead );
				}
				else 
				{
					e = new HlBaseList(TIMES);
					e->apparg(-1.0);
					e->apparg(factor());
				}
				return e;

			default:
				throw(  new HlError( 1, string("HlParser::factor: Unerwartetes Token: ") + mLookahead->getLexemName() ));
		}
	}

	// Dass der Compiler nicht meckert !
	return new HlBaseList(0.0);
}


/*-------------------------------------------------------------*/


void HlParser::match(types t)
{
	static string errorstring;
	static types prevtoken;

	if( !mLookahead->tokenIn(t) ) 
	{
		errorstring  = "HlParser::match: Unerwartetes Token nach '";
		errorstring += mSymTab->lookup(prevtoken)->getLexem();
		errorstring += "' :  '";
		errorstring += mLookahead->getLexem();
		errorstring += "' statt '";
		errorstring += mSymTab->lookup(t)->getLexem();
		errorstring += "'";
		throw( new HlError( 1, errorstring) );
	}

	prevtoken = mLookahead->getToken();

	if((mLookahead = mLexer.nextToken()) == NULL)
		throw(  new HlError( 1, "HlParser::match: Nullpointer lookahead") );

}


/*-------------------------------------------------------------*/


void HlParser::match(HlSymEntry *t)
{
	HlParser::match(t->getToken());
}

