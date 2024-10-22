/*  -*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// LEX file   CaseLexer.l                         
//
// Description: lexograpic analysis for Ensight case files
//
// Initial version: 2001-  
//                                               
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting 
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//
*/

%{
#ifdef _WIN32
#define YY_NO_UNISTD_H
#endif

#include "CaseLexer.h"

// define new name for yylex method
#undef YY_DECL
#define YY_DECL int CaseLexer::scan( MyTokenType *pToken )

      void CaseLexer::LexerError( const char msg[] )
      {
      cerr << msg << endl;
      }
%}


/** 
  Don't allow yywrap. There are no multibuffers allowed.
  */
%option noyywrap
%option yylineno

/**
  Use new Name for flexer class
*/
%option yyclass="CaseLexer"
%option prefix="Case"

%x VARS
%{

%}

%%
FORMAT[ \t]* {
    return( CaseParser::FORMAT_SEC );
}

type[ \t]*: {
    return( CaseParser::TYPE );
}

ensight[ \t]* {
    return( CaseParser::ENSIGHTV);
}

gold[ \t]* {
    return( CaseParser::ENSIGHT_GOLD );
}


GEOMETRY[ \t]* {
    return( CaseParser::GEOMETRY_SEC );
}

model[ \t]*: {
    return( CaseParser::MODEL );
}

measured[ \t]*: {
    return( CaseParser::MEASURED );
}

match[ \t]*: {
    return( CaseParser::MATCH );
}

change_coords_only {
    return( CaseParser::CH_CO_ONLY );
}

VARIABLE[ \t]* {
    return( CaseParser::VARIABLE_SEC );
}

constant {
    return( CaseParser::CONSTANT );
}


complex {
    return( CaseParser::COMPLEX );
}

scalar {
    return( CaseParser::SCALAR );
}

vector {
    return( CaseParser::VECTOR );
}

"tensor symm" {
    return( CaseParser::TENSOR_SYMM );
}

"per case"[ \t]*: {
    return( CaseParser::PER_CASE );
}

"per node"[ \t]*: {
    BEGIN VARS;
    return( CaseParser::PER_NODE );
}
 
"per measured node"[ \t]*: {
    BEGIN VARS;
    return( CaseParser::PER_M_NODE );
}

"per element"[ \t]*: {
    BEGIN VARS;
    return( CaseParser::PER_ELEMENT );
}

"per measured element"[ \t]*: {
    BEGIN VARS;
    return( CaseParser::PER_M_ELEMENT );
}

TIME[ \t]* {
    return( CaseParser::TIME_SEC );
}
"time set"[ \t]*: {
    return( CaseParser::TIME_SET );
}

"number of steps"[ \t]*: {
    return( CaseParser::NUM_OF_STEPS );
}

"filename start number"[ \t]*: {
    return( CaseParser::FN_ST_NUM );
}

"filename increment"[ \t]*: {
    return( CaseParser::FN_INCR );
}

"filename numbers"[ \t]*: {
    return( CaseParser::FN_NUMS );
}

"time values"[ \t]*: {
    return( CaseParser::TIME_VAL );
}

FILE[ \t]* {
    return( CaseParser::FILE_SEC );
}
"file set"[ \t]*: {
    return( CaseParser::FILE_SET );
}


<VARS>[^ ]{1,99}[ \t]+[\.a-zA-Z0-9]+[^ ^\n]+[ \t\n]*$ {
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    BEGIN 0;
    return( CaseParser::VAR_POST );
} 


<VARS>[0-9]+[ \t]* {
    // read integer digits
    pToken->token.iVal = atoi(yytext);
    // also provide characters
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::VAR_INT );
}

[0-9]*[a-zA-Z_@]+[a-zA-Z0-9_@]* { 
    // read characters as IDENTIFIER
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::IDENTIFIER );
}

([^ \n\t])+([*])+[^ \n\t]* {
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
   return( CaseParser::ASTNOTFN );
}


[0-9][0-9]?[0-9]?\.[0-9][0-9]?[0-9]?\.[0-9][0-9]?[0-9]?\.[0-9][0-9]?[0-9]? {
    // read integer digits
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::IPADDRESS );
}

[0-9\-]*[\.a-zA-Z_@]+[\.a-zA-Z0-9_@\-]* { 
    // read characters as IDENTIFIER
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::POINT_IDENTIFIER );
}


[0-9]+[ \n\t] {
    // read integer digits
    pToken->token.iVal = atoi(yytext);
    // also provide characters
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::INTEGER );
}

[-+]?([0-9]+|[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?[ \n\t] {
    // read double
    pToken->token.dVal = atof(yytext);
    return( CaseParser::DOUBLE );
}



\"[^\"\n]*\" {
    // read a string value without the quotes
    strncpy( pToken->token.szValue, yytext+1, yyleng-2 );
    pToken->token.szValue[ yyleng-2 ] = '\0';
    return( CaseParser::STRING );
}

"#"[^\n]* {   
    // suppress comments - all lines start with a #
} 

[ \x0d \t]+  ;            // suppress blanks,tabs

\n {
    // let flex handle incrementing yylineno
}

. {     
    // all single characters which not match the rules below
    // will be return as single characters
    return( yytext[ 0 ] );
}

%%

