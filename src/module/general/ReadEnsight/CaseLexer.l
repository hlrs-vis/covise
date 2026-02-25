/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// LEX file   CaseLexer.l
//
// Description: lexographic analysis for EnSight case files
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
%option nounistd
%option nodefault

/* Enable for debugging
%option warn
%option debug
%option verbose
%option trace
*/

/**
  Use new Name for flexer class
*/
%option c++
%option yyclass="CaseLexer"
%option prefix="Case"

%{
#undef YY_DECL
#include "CaseParser.h"
#include "CaseLexer.h"
#include "CaseParserDriver.h"

CaseLexer::CaseLexer(std::istream *pDesc): CaseFlexLexer(pDesc) {}
CaseLexer::~CaseLexer() = default;

#define CERR std::cerr << "CaseLexer: "

#ifdef NDEBUG
#define YY_USER_ACTION \
driver.setLineNumber(yylineno); \
driver.setText(yytext); \
driver.setState(YY_START);
#else
#define YY_USER_ACTION \
driver.setLineNumber(yylineno); \
driver.setText(yytext); \
driver.setState(YY_START); \
CERR << "state=" << YY_START << ", line " << driver.lineNumber() << ": " << yytext << std::endl;
#endif
%}

/* inclusive start conditions:
   TOKENS: for keywords in FORMAT, SCRIPTS, GEOMETRY, VARIABLE, TIME, FILE sections - before colon
   VALUES: for values in FORMAT, SCRIPTS, GEOMETRY, VARIABLE, TIME, FILE sections - after colon
   MULTILINE_VALUES: for values in TIME section which span several lines - after "time values:" or "filename numbers:" */
%s TOKENS VALUES MULTILINE_VALUES

Dig   [0-9]
Start [^\.\'!@#$^()\[\]*/+,\\\"<>?|[:space:]0-9-]
Char  [^\.\'!@#$^()\[\]*/+,\\\"<>?|[:space:]-]

%%
<INITIAL,TOKENS>FORMAT {
BEGIN TOKENS;
return CaseParser::token::FORMAT_SEC;
}

<TOKENS>type {
    return CaseParser::token::TYPE;
}

<VALUES>ensight {
    return CaseParser::token::ENSIGHT;
}

<VALUES>gold {
    return CaseParser::token::ENSIGHT_GOLD;
}

<INITIAL,TOKENS>SCRIPTS {
    BEGIN TOKENS;
    return CaseParser::token::SCRIPTS_SEC;
}

<TOKENS>metadata {
    return CaseParser::token::METADATA;
}

<TOKENS>python {
    return CaseParser::token::PYTHON;
}

<INITIAL,TOKENS>GEOMETRY {
    BEGIN TOKENS;
    return CaseParser::token::GEOMETRY_SEC;
}

<TOKENS>model {
    return CaseParser::token::MODEL;
}

<TOKENS>measured {
    return CaseParser::token::MEASURED;
}

<TOKENS>match {
    return CaseParser::token::MATCH;
}

<VALUES>change_coords_only {
    return CaseParser::token::CH_CO_ONLY;
}

<VALUES>changing_geometry_per_part {
    return CaseParser::token::CH_GEOM_PER_PART;
}

<INITIAL,TOKENS>VARIABLE {
    BEGIN TOKENS;
    return CaseParser::token::VARIABLE_SEC;
}

<TOKENS>constant {
    return CaseParser::token::CONSTANT;
}


<TOKENS>complex {
    return CaseParser::token::COMPLEX;
}

<TOKENS>scalar {
    return CaseParser::token::SCALAR;
}

<TOKENS>vector {
    return CaseParser::token::VECTOR;
}

<TOKENS>"tensor symm" {
    return CaseParser::token::TENSOR_SYMM;
}

<TOKENS>"tensor asym" {
    return CaseParser::token::TENSOR_ASYMM;
}

<TOKENS>"per case" {
    return CaseParser::token::PER_CASE;
}

<TOKENS>"per node" {
    return CaseParser::token::PER_NODE;
}

<TOKENS>"per measured node" {
    return CaseParser::token::PER_M_NODE;
}

<TOKENS>"per element" {
    return CaseParser::token::PER_ELEMENT;
}

<TOKENS>"per measured element" {
    return CaseParser::token::PER_M_ELEMENT;
}

<TOKENS>: {
    BEGIN VALUES;
}

<INITIAL,TOKENS>TIME {
    BEGIN TOKENS;
    return CaseParser::token::TIME_SEC;
}
<TOKENS,MULTILINE_VALUES>"time set" {
    BEGIN TOKENS;
    return CaseParser::token::TIME_SET;
}

<TOKENS,MULTILINE_VALUES>"number of steps" {
    BEGIN TOKENS;
    return CaseParser::token::NUM_OF_STEPS;
}

<TOKENS,MULTILINE_VALUES>"filename start number" {
    BEGIN TOKENS;
    return CaseParser::token::FN_ST_NUM;
}

<TOKENS,MULTILINE_VALUES>"filename increment" {
    BEGIN TOKENS;
    return CaseParser::token::FN_INCR;
}

<TOKENS,MULTILINE_VALUES>"filename numbers:" {
    BEGIN MULTILINE_VALUES;
    return CaseParser::token::FN_NUMS;
}

<TOKENS,MULTILINE_VALUES>"time values:" {
    BEGIN MULTILINE_VALUES;
    return CaseParser::token::TIME_VAL;
}

<INITIAL,TOKENS>FILE {
    BEGIN TOKENS;
    return CaseParser::token::FILE_SEC;
}

<TOKENS>"file set" {
    return CaseParser::token::FILE_SET;
}



<VALUES>{Start}+{Char}* {
    // read characters as IDENTIFIER for variable descriptions
    pToken->String = std::string(yytext, yytext+yyleng );
    return CaseParser::token::IDENTIFIER;
}

<VALUES,MULTILINE_VALUES>[+-]?{Dig}+ {
    // read integer digits
    pToken->Int = atol(yytext);
    pToken->Double = static_cast<double>(pToken->Int);
    // also provide characters
    pToken->String = std::string(yytext, yytext+yyleng);
    return CaseParser::token::INTEGER;
}

<VALUES,MULTILINE_VALUES>[-+]?({Dig}+\.?|{Dig}*\.{Dig}+)([eE][-+]?{Dig}+)? {
    // read double
    pToken->Double = atof(yytext);
    // also provide characters
    pToken->String = std::string(yytext, yytext+yyleng);
    return CaseParser::token::DOUBLE;
}

<VALUES>["]([^"\\\n]|\\.)*["] {
    // read a string value in quotes
    pToken->String = std::string(yytext+1, yytext+yyleng-1 );
    return CaseParser::token::QUOTED;
}

<*>"#"[^\n]* {
    // suppress comments - from # until end of line
}

<*>[\t\r\f ]+ {
    // suppress blanks,tabs
}

<*>\n {
    // let flex handle incrementing yylineno
    if (YY_START == VALUES) {
        BEGIN TOKENS;
    }
}

. {
    CERR << "line=" << yylineno << ", state=" << YY_START << ": " << yytext << std::endl;
    std::cerr << "   Unknown character: " << yytext[0] << std::endl;
    // all single characters which not match the rules below
    // will be returned as single characters
    return yytext[0];
}

<VALUES>[^[:space:]\"\']+ {
    pToken->String = std::string(yytext, yytext+yyleng );
    return CaseParser::token::TEXT;
}

%%
