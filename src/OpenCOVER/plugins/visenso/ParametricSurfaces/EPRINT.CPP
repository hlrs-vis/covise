#include <stdio.h>
#include <string.h>
#include <math.h>

#include "ELIST.H"


static string sbuffer;


void appbuf( const string& s )
{
	sbuffer += s;
}


int  HlExprList::need_bracket(HlExprList *e)
{
	if(e->NumberQ()) {
		if( Is(POWER) ) return (e->getDoubleVal() < 0) ? true : false;
			return false;
	}

	if(e->Is(IDENT))  return false;

	return( e->typeOfHead() < typeOfHead() ) ? true : false;
}



const string& HlExprList::fktstr()
{
	sbuffer = "";

	if( !ok() ) 
		return sbuffer;

	if( Is(POST) )
		First()->postfix();
	else
		infix();

	size_t j;
	while ((j = sbuffer.find( "+-" )) != string::npos)
	{
		sbuffer.replace( j, 2, "-" );
	}

	return sbuffer;
}



void HlExprList::postfix()
{
	char h[100];
	size_t i, n;

	if( NumberQ() ) 
	{
#ifdef WIN32
		sprintf_s( h, "%f", getDoubleVal() );
#else
        sprintf( h, "%f", getDoubleVal() );
#endif
		for( i=strlen(h)-1; h[i]=='0'; i-- )
			h[i] = '\0';
		if( h[i] == '.' ) h[i] = '\0';
			appbuf(h);
	}

	else if( Is(IDENT) ) 
	{
		appbuf( getsym()->getLexemName() );
	}

	else 
	{
		appbuf( getsym()->getLexemName() );
		appbuf( "(" );
		for( i=0, n=Length(); i<n; i++ ) 
		{
			if( i > 0 )
				appbuf( "," );
			arg(i)->postfix();
		}
		appbuf( ")" );
	}
}



void HlExprList::infix()
{
	char h[100];
	size_t i,start;

	switch( typeOfHead() ) {

		case NUMBER:
#ifdef WIN32
			sprintf_s( h, "%f", getDoubleVal() );
#else
            sprintf( h, "%f", getDoubleVal() );
#endif 
			for( i=strlen(h)-1; h[i]=='0'; i-- )
				h[i] = '\0';
			if( h[i] == '.' ) h[i] = '\0';
			appbuf(h);
			break;

		case IDENT:
			appbuf( getsym()->getLexemName() );
			break;

		case OP:
			First()->infix();
			appbuf("(");
			Second()->infix();
			appbuf(")");
			break;

		case TIMES:
			start = 0;

			if( First()->Is(-1.0) ) 
			{
				appbuf("-");
				start = 1;
			}

			for( i=start; i<Length(); i++ ) {

				if( arg(i)->Is(POWER) ) {
					if( arg(i)->Second()->Is(-1.0) ) 
					{
						if( i == start ) 
						{
							appbuf("1");
						}
						appbuf("/");
						if(need_bracket(arg(i)->First()) ||	arg(i)->First()->Is(TIMES))
						{
							appbuf( "(" );
							arg(i)->First()->infix();
							appbuf( ")" );
						}
						else 
						{
							arg(i)->First()->infix();
						}
						continue;
						}
					}

					if( i > start )
						appbuf( getsym()->getLexem() );

					if( need_bracket(arg(i)) ) 
					{
						appbuf( "(" );
						arg(i)->infix();
						appbuf( ")" );
					}
					else 
					{
						arg(i)->infix();
					}

				}
				break;


		default:
			if (getsym()->getProp(INFIX) )
			{
				for( i=0; i<Length(); i++ ) 
				{
					if( i > 0 )
						appbuf( getsym()->getLexem() );

					if( need_bracket(arg(i)) )
					{
						appbuf( "(" );
						arg(i)->infix();
						appbuf( ")" );
					}
					else 
					{
						arg(i)->infix();
					}
				}
			}

			else
			{
				size_t n = Length();
				if( (Is(ABS) || Is(SIGN)) && (Length() == 2) )  {
					if( Second()->evalF() == 0 ) n = 1;
				}

				appbuf( getsym()->getLexemName() );
				if(Length()) appbuf( "(" );

				for( i=0; i<n; i++ )
				{
					if( i > 0 ) appbuf(",");
						arg(i)->infix();
				}

				if(n) appbuf(")");
			}
			break;
	}
}
