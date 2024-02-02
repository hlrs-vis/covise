/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// LEX file   CaseLexer.l
//
// Description: lexographic analysis for Ensight case files
//
// Initial version: 2001
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
*/

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

%{
#undef YY_DECL
#include "CaseParser.h"
#include "CaseLexer.h"

CaseLexer::CaseLexer(std::istream *pDesc): CaseFlexLexer(pDesc) {}
CaseLexer::~CaseLexer() = default;
%}


%x VARS
%{

%}

%%
FORMAT[ \t]* {
    return( CaseParser::token::FORMAT_SEC );
}

type[ \t]*: {
    return( CaseParser::token::TYPE );
}

ensight[ \t]* {
    return( CaseParser::token::ENSIGHTV);
}

gold[ \t]* {
    return( CaseParser::token::ENSIGHT_GOLD );
}


GEOMETRY[ \t]* {
    return( CaseParser::token::GEOMETRY_SEC );
}

model[ \t]*: {
    return( CaseParser::token::MODEL );
}

measured[ \t]*: {
    return( CaseParser::token::MEASURED );
}

match[ \t]*: {
    return( CaseParser::token::MATCH );
}

change_coords_only {
    return( CaseParser::token::CH_CO_ONLY );
}

VARIABLE[ \t]* {
    return( CaseParser::token::VARIABLE_SEC );
}

constant {
    return( CaseParser::token::CONSTANT );
}


complex {
    return( CaseParser::token::COMPLEX );
}

scalar {
    return( CaseParser::token::SCALAR );
}

vector {
    return( CaseParser::token::VECTOR );
}

"tensor symm" {
    return( CaseParser::token::TENSOR_SYMM );
}

"per case"[ \t]*: {
    return( CaseParser::token::PER_CASE );
}

"per node"[ \t]*: {
    BEGIN VARS;
    return( CaseParser::token::PER_NODE );
}
 
"per measured node"[ \t]*: {
    BEGIN VARS;
    return( CaseParser::token::PER_M_NODE );
}

"per element"[ \t]*: {
    BEGIN VARS;
    return( CaseParser::token::PER_ELEMENT );
}

"per measured element"[ \t]*: {
    BEGIN VARS;
    return( CaseParser::token::PER_M_ELEMENT );
}

TIME[ \t]* {
    return( CaseParser::token::TIME_SEC );
}
"time set"[ \t]*: {
    return( CaseParser::token::TIME_SET );
}

"number of steps"[ \t]*: {
    return( CaseParser::token::NUM_OF_STEPS );
}

"filename start number"[ \t]*: {
    return( CaseParser::token::FN_ST_NUM );
}

"filename increment"[ \t]*: {
    return( CaseParser::token::FN_INCR );
}

"filename numbers"[ \t]*: {
    return( CaseParser::token::FN_NUMS );
}

"time values"[ \t]*: {
    return( CaseParser::token::TIME_VAL );
}

FILE[ \t]* {
    return( CaseParser::token::FILE_SEC );
}
"file set"[ \t]*: {
    return( CaseParser::token::FILE_SET );
}


<VARS>[^ ]{1,99}[ \t]+[\.a-zA-Z0-9]+[^ ^\n]+[ \t\n]*$ {
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    BEGIN 0;
    return( CaseParser::token::VAR_POST );
} 


<VARS>[0-9]+[ \t]* {
    // read integer digits
    pToken->token.iVal = atol(yytext);
    // also provide characters
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::token::VAR_INT );
}

[0-9]*[a-zA-Z_@]+[a-zA-Z0-9_@]* { 
    // read characters as IDENTIFIER
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::token::IDENTIFIER );
}

([^ \n\t])+([*])+[^ \n\t]* {
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
   return( CaseParser::token::ASTNOTFN );
}


[0-9][0-9]?[0-9]?\.[0-9][0-9]?[0-9]?\.[0-9][0-9]?[0-9]?\.[0-9][0-9]?[0-9]? {
    // read integer digits
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::token::IPADDRESS );
}

[0-9\-]*[\.a-zA-Z_@]+[\.a-zA-Z0-9_@\-]* { 
    // read characters as IDENTIFIER
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::token::POINT_IDENTIFIER );
}


[0-9]+[ \n\t] {
    // read integer digits
    pToken->token.iVal = atol(yytext);
    // also provide characters
    strncpy( pToken->token.szValue, yytext, yyleng );
    pToken->token.szValue[ yyleng ] = '\0';
    return( CaseParser::token::INTEGER );
}

[-+]?([0-9]+|[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)([eE][-+]?[0-9]+)?[ \n\t] {
    // read double
    pToken->token.dVal = atof(yytext);
    return( CaseParser::token::DOUBLE );
}



\"[^\"\n]*\" {
    // read a string value without the quotes
    strncpy( pToken->token.szValue, yytext+1, yyleng-2 );
    pToken->token.szValue[ yyleng-2 ] = '\0';
    return( CaseParser::token::STRING );
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
