
/*  -*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// LEX file   PointsLexer.l
//
// Analysis of initial points
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2003 by VirCinity IT Consulting 
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
// Changes:
//
*/


%{
#include "PointsLexer.h"

// define new name for yylex method
#undef YY_DECL
#define YY_DECL int PointsLexer::scan( TokenType *pToken )
%}

/**
  Don't allow yywrap. There are no multibuffers allowed.
  */
%option noyywrap

/**
  Use new Name for flexer class
*/
%option yyclass="PointsLexer"
%option prefix="Points"

%%
"<" {
   return (PointsParser::OPEN_POINT);
}

">" {
   return (PointsParser::CLOSE_POINT);
}

"(" {
   return (PointsParser::OPEN_POINT_PAR);
}

")" {
   return (PointsParser::CLOSE_POINT_PAR);
}

"[" {
   return (PointsParser::OPEN_POINT_MATLAB);
}

"]" {
   return (PointsParser::CLOSE_POINT_MATLAB);
}

-?((([0-9]+)|([0-9]*\.[0-9]+))([eE][-+]?[0-9]+)?) {
   // read double
   pToken->token.fVal = atof(yytext);
   return( PointsParser::COORDINATE );
}   

, {
   return (PointsParser::SEPARATOR);
}

[ \t]+  ;            // suppress blanks,tabs

\n {
    // increment lineno and ignore newline
   yylineno++;
}

. {
   // all single characters which not match the rules below
   // will be return as single characters
   return( yytext[ 0 ] );
}   

%%

