/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef YY_Parser_h_included
#define YY_Parser_h_included
#define YY_USE_CLASS

#line 1 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
/* before anything */
#ifdef c_plusplus
#ifndef __cplusplus
#define __cplusplus
#endif
#endif

#line 8 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
#line 5 "dxreader.yacc"

// #define YYSTYPE double

#include <util/coviseCompat.h>

#if defined(__hpux) || defined(__sgi)
#include <alloca.h>
#endif

/* #include <covise/FlexLexer.h> */
#include "attribute.h"
#include "scanner.h"

#include <api/coModule.h>
using namespace covise;
#include "parser.h"
/* The action class is the interface between parser and the rest of the compiler
   depending on its implementation
*/

#include "action.h"

#line 31 "dxreader.yacc"
typedef union
{
    attribute *attr;
} yy_Parser_stype;
#define YY_Parser_STYPE yy_Parser_stype
#define YY_Parser_MEMBERS                               \
    ifstream input;                                     \
    Scanner *lexer;                                     \
    char *currFileName_;                                \
    char *currDirName_;                                 \
    bool isOpen_;                                       \
    bool isCorrect_;                                    \
    actionClass *action;                                \
    virtual ~Parser()                                   \
    {                                                   \
        input.close();                                  \
        delete currFileName_;                           \
        delete currDirName_;                            \
    }                                                   \
    Parser(actionClass *act, const char *fileName)      \
    {                                                   \
        currFileName_ = new char[1 + strlen(fileName)]; \
        strcpy(currFileName_, fileName);                \
        currDirName_ = new char[1 + strlen(fileName)];  \
        strcpy(currDirName_, fileName);                 \
        myDirName(currDirName_);                        \
        action = act;                                   \
        input.open(fileName, ios::in);                  \
        isOpen_ = !(!input);                            \
        isCorrect_ = true;                              \
        lexer = new Scanner(&input);                    \
    }                                                   \
    bool isOpen()                                       \
    {                                                   \
        return isOpen_;                                 \
    }                                                   \
    bool isCorrect()                                    \
    {                                                   \
        return isCorrect_;                              \
    }                                                   \
    void myDirName(char *filename)                      \
    {                                                   \
        int len = strlen(filename);                     \
        int i;                                          \
        for (i = len - 1; i >= 0; i--)                  \
        {                                               \
            if (filename[i] == '/')                     \
            {                                           \
                filename[i] = '\0';                     \
                return;                                 \
            }                                           \
        }                                               \
    }                                                   \
    void setCurrFileName(const char *name)              \
    {                                                   \
        delete[] currFileName_;                         \
        currFileName_ = new char[1 + strlen(name)];     \
        strcpy(currFileName_, name);                    \
    }                                                   \
    char *getCurrFileName()                             \
    {                                                   \
        return currFileName_;                           \
    }                                                   \
    void setCurrDirName(const char *name)               \
    {                                                   \
        delete[] currDirName_;                          \
        currDirName_ = new char[1 + strlen(name)];      \
        strcpy(currDirName_, name);                     \
    }                                                   \
    char *getCurrDirName()                              \
    {                                                   \
        return currDirName_;                            \
    }

#define YY_Parser_LEX_BODY     \
    {                          \
        return lexer->yylex(); \
    }
#define YY_Parser_ERROR_BODY                                                                               \
    {                                                                                                      \
        char comsg[4096];                                                                                  \
        sprintf(comsg, "Syntax error in line %d; %s not recognized", lexer->getLineNo(), lexer->YYText()); \
        Covise::sendError(comsg);                                                                          \
        isCorrect_ = false;                                                                                \
    }

#line 21 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
/* %{ and %header{ and %union, during decl */
#ifndef YY_Parser_COMPATIBILITY
#ifndef YY_USE_CLASS
#define YY_Parser_COMPATIBILITY 1
#else
#define YY_Parser_COMPATIBILITY 0
#endif
#endif

#if YY_Parser_COMPATIBILITY != 0
/* backward compatibility */
#ifdef YYLTYPE
#ifndef YY_Parser_LTYPE
#define YY_Parser_LTYPE YYLTYPE
/* WARNING obsolete !!! user defined YYLTYPE not reported into generated header */
/* use %define LTYPE */
#endif
#endif
/*#ifdef YYSTYPE*/
#ifndef YY_Parser_STYPE
#define YY_Parser_STYPE YYSTYPE
/* WARNING obsolete !!! user defined YYSTYPE not reported into generated header */
/* use %define STYPE */
#endif
/*#endif*/
#ifdef YYDEBUG
#ifndef YY_Parser_DEBUG
#define YY_Parser_DEBUG YYDEBUG
/* WARNING obsolete !!! user defined YYDEBUG not reported into generated header */
/* use %define DEBUG */
#endif
#endif
/* use goto to be compatible */
#ifndef YY_Parser_USE_GOTO
#define YY_Parser_USE_GOTO 1
#endif
#endif

/* use no goto to be clean in C++ */
#ifndef YY_Parser_USE_GOTO
#define YY_Parser_USE_GOTO 0
#endif

#ifndef YY_Parser_PURE

#line 65 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"

#line 65 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
/* YY_Parser_PURE */
#endif

#line 68 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"

#line 68 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
/* prefix */

#ifndef YY_Parser_DEBUG

#line 71 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"

#line 71 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
/* YY_Parser_DEBUG */
#endif

#ifndef YY_Parser_LSP_NEEDED

#line 75 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"

#line 75 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
/* YY_Parser_LSP_NEEDED*/
#endif

/* DEFAULT LTYPE*/
#ifdef YY_Parser_LSP_NEEDED
#ifndef YY_Parser_LTYPE
#ifndef BISON_YYLTYPE_ISDECLARED
#define BISON_YYLTYPE_ISDECLARED
typedef struct yyltype
{
    int timestamp;
    int first_line;
    int first_column;
    int last_line;
    int last_column;
    char *text;
} yyltype;
#endif

#define YY_Parser_LTYPE yyltype
#endif
#endif

/* DEFAULT STYPE*/
#ifndef YY_Parser_STYPE
#define YY_Parser_STYPE int
#endif

/* DEFAULT MISCELANEOUS */
#ifndef YY_Parser_PARSE
#define YY_Parser_PARSE yyparse
#endif

#ifndef YY_Parser_LEX
#define YY_Parser_LEX yylex
#endif

#ifndef YY_Parser_LVAL
#define YY_Parser_LVAL yylval
#endif

#ifndef YY_Parser_LLOC
#define YY_Parser_LLOC yylloc
#endif

#ifndef YY_Parser_CHAR
#define YY_Parser_CHAR yychar
#endif

#ifndef YY_Parser_NERRS
#define YY_Parser_NERRS yynerrs
#endif

#ifndef YY_Parser_DEBUG_FLAG
#define YY_Parser_DEBUG_FLAG yydebug
#endif

#ifndef YY_Parser_ERROR
#define YY_Parser_ERROR yyerror
#endif

#ifndef YY_Parser_PARSE_PARAM
#ifndef __STDC__
#ifndef __cplusplus
#ifndef YY_USE_CLASS
#define YY_Parser_PARSE_PARAM
#ifndef YY_Parser_PARSE_PARAM_DEF
#define YY_Parser_PARSE_PARAM_DEF
#endif
#endif
#endif
#endif
#ifndef YY_Parser_PARSE_PARAM
#define YY_Parser_PARSE_PARAM void
#endif
#endif

/* TOKEN C */
#ifndef YY_USE_CLASS

#ifndef YY_Parser_PURE
#ifndef yylval
extern YY_Parser_STYPE YY_Parser_LVAL;
#else
#if yylval != YY_Parser_LVAL
extern YY_Parser_STYPE YY_Parser_LVAL;
#else
#warning "Namespace conflict, disabling some functionality (bison++ only)"
#endif
#endif
#endif

#line 169 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
#define INTEGERVALUE 258
#define FLOATVALUE 259
#define STRINGVALUE 260
#define OBJECT 261
#define CLASS 262
#define ARRAY 263
#define FIELD 264
#define MULTIGRID 265
#define GROUP 266
#define FILENAME 267
#define MEMBER 268
#define TYPE 269
#define BINARY 270
#define ASCII 271
#define COMMA 272
#define SHAPE 273
#define DOUBLE 274
#define FLOAT 275
#define INT 276
#define UINT 277
#define SHORT 278
#define USHORT 279
#define BYTE 280
#define UBYTE 281
#define RANK 282
#define ITEMS 283
#define DATA 284
#define FOLLOWS 285
#define FILE 286
#define ATTRIBUTEREF 287
#define ATTRIBUTEELTYPE 288
#define ATTRIBUTENAME 289
#define ATTRIBUTEDEP 290
#define CPOSITIONS 291
#define CCONNECTIONS 292
#define CDATA 293
#define VALUE 294
#define STRING 295
#define LSB 296
#define MSB 297
#define END 298

#line 169 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
/* #defines token */
/* after #define tokens, before const tokens S5*/
#else
#ifndef YY_Parser_CLASS
#define YY_Parser_CLASS Parser
#endif

#ifndef YY_Parser_INHERIT
#define YY_Parser_INHERIT
#endif

#ifndef YY_Parser_MEMBERS
#define YY_Parser_MEMBERS
#endif

#ifndef YY_Parser_LEX_BODY
#define YY_Parser_LEX_BODY
#endif

#ifndef YY_Parser_ERROR_BODY
#define YY_Parser_ERROR_BODY
#endif

#ifndef YY_Parser_CONSTRUCTOR_PARAM
#define YY_Parser_CONSTRUCTOR_PARAM
#endif
/* choose between enum and const */
#ifndef YY_Parser_USE_CONST_TOKEN
#define YY_Parser_USE_CONST_TOKEN 0
/* yes enum is more compatible with flex,  */
/* so by default we use it */
#endif
#if YY_Parser_USE_CONST_TOKEN != 0
#ifndef YY_Parser_ENUM_TOKEN
#define YY_Parser_ENUM_TOKEN yy_Parser_enum_token
#endif
#endif

class YY_Parser_CLASS YY_Parser_INHERIT
{
public:
#if YY_Parser_USE_CONST_TOKEN != 0
/* static const int token ... */

#line 212 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
    static const int INTEGERVALUE;
    static const int FLOATVALUE;
    static const int STRINGVALUE;
    static const int OBJECT;
    static const int CLASS;
    static const int ARRAY;
    static const int FIELD;
    static const int MULTIGRID;
    static const int GROUP;
    static const int FILENAME;
    static const int MEMBER;
    static const int TYPE;
    static const int BINARY;
    static const int ASCII;
    static const int COMMA;
    static const int SHAPE;
    static const int DOUBLE;
    static const int FLOAT;
    static const int INT;
    static const int UINT;
    static const int SHORT;
    static const int USHORT;
    static const int BYTE;
    static const int UBYTE;
    static const int RANK;
    static const int ITEMS;
    static const int DATA;
    static const int FOLLOWS;
    static const int FILE;
    static const int ATTRIBUTEREF;
    static const int ATTRIBUTEELTYPE;
    static const int ATTRIBUTENAME;
    static const int ATTRIBUTEDEP;
    static const int CPOSITIONS;
    static const int CCONNECTIONS;
    static const int CDATA;
    static const int VALUE;
    static const int STRING;
    static const int LSB;
    static const int MSB;
    static const int END;

#line 212 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
/* decl const */
#else
    enum YY_Parser_ENUM_TOKEN
    {
        YY_Parser_NULL_TOKEN = 0

#line 215 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
        ,
        INTEGERVALUE = 258,
        FLOATVALUE = 259,
        STRINGVALUE = 260,
        OBJECT = 261,
        CLASS = 262,
        ARRAY = 263,
        FIELD = 264,
        MULTIGRID = 265,
        GROUP = 266,
        FILENAME = 267,
        MEMBER = 268,
        TYPE = 269,
        BINARY = 270,
        ASCII = 271,
        COMMA = 272,
        SHAPE = 273,
        DOUBLE = 274,
        FLOAT = 275,
        INT = 276,
        UINT = 277,
        SHORT = 278,
        USHORT = 279,
        BYTE = 280,
        UBYTE = 281,
        RANK = 282,
        ITEMS = 283,
        DATA = 284,
        FOLLOWS = 285,
        FILE = 286,
        ATTRIBUTEREF = 287,
        ATTRIBUTEELTYPE = 288,
        ATTRIBUTENAME = 289,
        ATTRIBUTEDEP = 290,
        CPOSITIONS = 291,
        CCONNECTIONS = 292,
        CDATA = 293,
        VALUE = 294,
        STRING = 295,
        LSB = 296,
        MSB = 297,
        END = 298

#line 215 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
        /* enum token */
    }; /* end of enum declaration */
#endif
public:
    int YY_Parser_PARSE(YY_Parser_PARSE_PARAM);
    virtual void YY_Parser_ERROR(char *msg) YY_Parser_ERROR_BODY;
#ifdef YY_Parser_PURE
#ifdef YY_Parser_LSP_NEEDED
    virtual int YY_Parser_LEX(YY_Parser_STYPE *YY_Parser_LVAL, YY_Parser_LTYPE *YY_Parser_LLOC) YY_Parser_LEX_BODY;
#else
    virtual int YY_Parser_LEX(YY_Parser_STYPE *YY_Parser_LVAL) YY_Parser_LEX_BODY;
#endif
#else
    virtual int YY_Parser_LEX() YY_Parser_LEX_BODY;
    YY_Parser_STYPE YY_Parser_LVAL;
#ifdef YY_Parser_LSP_NEEDED
    YY_Parser_LTYPE YY_Parser_LLOC;
#endif
    int YY_Parser_NERRS;
    int YY_Parser_CHAR;
#endif
#if YY_Parser_DEBUG != 0
public:
    int YY_Parser_DEBUG_FLAG; /*  nonzero means print parse trace	*/
#endif
public:
    YY_Parser_CLASS(YY_Parser_CONSTRUCTOR_PARAM);

public:
    YY_Parser_MEMBERS
};
/* other declare folow */
#endif

#if YY_Parser_COMPATIBILITY != 0
/* backward compatibility */
/* Removed due to bison problems
 /#ifndef YYSTYPE
 / #define YYSTYPE YY_Parser_STYPE
 /#endif*/

#ifndef YYLTYPE
#define YYLTYPE YY_Parser_LTYPE
#endif
#ifndef YYDEBUG
#ifdef YY_Parser_DEBUG
#define YYDEBUG YY_Parser_DEBUG
#endif
#endif

#endif
/* END */

#line 267 "/home/clearcase/extern_libs/amd64/bison++/share/bison++/bison.h"
#endif
