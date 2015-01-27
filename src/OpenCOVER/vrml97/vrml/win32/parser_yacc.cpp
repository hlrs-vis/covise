/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Using locations.  */
#define YYLSP_NEEDED 0

/* Substitute the variable and function names.  */
#define yyparse parserparse
#define yylex parserlex
#define yyerror parsererror
#define yylval parserlval
#define yychar parserchar
#define yydebug parserdebug
#define yynerrs parsernerrs

/* Tokens.  */
#ifndef YYTOKENTYPE
#define YYTOKENTYPE
/* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
enum yytokentype
{
    IDENTIFIER = 258,
    DEF = 259,
    USE = 260,
    PROTO = 261,
    EXTERNPROTO = 262,
    TO = 263,
    IS = 264,
    ROUTE = 265,
    SFN_NULL = 266,
    EVENTIN = 267,
    EVENTOUT = 268,
    FIELD = 269,
    EXPOSEDFIELD = 270,
    SF_BOOL = 271,
    SF_COLOR = 272,
    SF_FLOAT = 273,
    SF_INT32 = 274,
    SF_ROTATION = 275,
    SF_TIME = 276,
    SF_IMAGE = 277,
    SF_STRING = 278,
    SF_VEC2F = 279,
    SF_VEC3F = 280,
    MF_COLOR = 281,
    MF_FLOAT = 282,
    MF_INT32 = 283,
    MF_ROTATION = 284,
    VRML_MF_STRING = 285,
    MF_VEC2F = 286,
    MF_VEC3F = 287,
    SF_NODE = 288,
    MF_NODE = 289
};
#endif
/* Tokens.  */
#define IDENTIFIER 258
#define DEF 259
#define USE 260
#define PROTO 261
#define EXTERNPROTO 262
#define TO 263
#define IS 264
#define ROUTE 265
#define SFN_NULL 266
#define EVENTIN 267
#define EVENTOUT 268
#define FIELD 269
#define EXPOSEDFIELD 270
#define SF_BOOL 271
#define SF_COLOR 272
#define SF_FLOAT 273
#define SF_INT32 274
#define SF_ROTATION 275
#define SF_TIME 276
#define SF_IMAGE 277
#define SF_STRING 278
#define SF_VEC2F 279
#define SF_VEC3F 280
#define MF_COLOR 281
#define MF_FLOAT 282
#define MF_INT32 283
#define MF_ROTATION 284
#define VRML_MF_STRING 285
#define MF_VEC2F 286
#define MF_VEC3F 287
#define SF_NODE 288
#define MF_NODE 289

/* Copy the first part of user declarations.  */
#line 9 "parser.yy"

#ifndef YYINITDEPTH
#define YYINITDEPTH 400
#endif

#include "config.h"

#include <stdio.h> // sprintf
#include <string.h>

// Get rid of this and calls to free() (lexer uses strdup)...
#include <stdlib.h>

#include <vector>
using std::vector;
#include <list>
using std::list;

#include "System.h"
#include "VrmlScene.h"
#include "VrmlField.h"

#include "VrmlNode.h"
#include "VrmlNamespace.h"
#include "VrmlNodeType.h"

#include "VrmlNodeScript.h"

#include "VrmlSFNode.h"
#include "VrmlMFNode.h"

#define parserlex lexerlex
using namespace covise;

// It would be nice to remove these globals...

// The defined node types (built in and PROTOd) and DEFd nodes
VrmlNamespace *yyNodeTypes = 0;

// The parser builds a scene graph rooted at this list of nodes.
VrmlMFNode *yyParsedNodes = 0;

// Where the world is being read from (needed to resolve relative URLs)
Doc *yyDocument = 0;

// Currently-being-defined proto.  Prototypes may be nested, so a stack
// is needed. I'm using a list because the STL stack API is still in flux.

static list<VrmlNodeType *> currentProtoStack;

// This is used to keep track of which field in which type of node is being
// parsed.  Field are nested (nodes are contained inside MFNode/SFNode fields)
// so a stack of these is needed. I'm using a list because the STL stack API
// is still in flux.

typedef VrmlField::VrmlFieldType FieldType;

typedef struct
{
    VrmlNode *node;
    const VrmlNodeType *nodeType;
    const char *fieldName;
    FieldType fieldType;
} FieldRec;

static list<FieldRec *> currentField;

// Name for current node being defined.

static char *nodeName = 0;

// This is used when the parser knows what kind of token it expects
// to get next-- used when parsing field values (whose types are declared
// and read by the parser) and at certain other places:
extern int expectToken;
extern int expectCoordIndex;
extern int expectTexCoordIndex;

// Current line number (set by lexer)
extern int currentLineNumber;

// Some helper routines defined below:
static void beginProto(const char *);
static void endProto(VrmlField *url);

// PROTO interface handlers
static FieldType addField(const char *type, const char *name);
static FieldType addEventIn(const char *type, const char *name);
static FieldType addEventOut(const char *type, const char *name);
static FieldType addExposedField(const char *type, const char *name);

static void setFieldDefault(const char *fieldName, VrmlField *value);
static FieldType fieldType(const char *type);
static void enterNode(const char *name);
static VrmlNode *exitNode();

// Node fields
static void enterField(const char *name);
static void exitField(VrmlField *value);
static void expect(FieldType type);

// Script fields
static bool inScript();
static void addScriptEventIn(const char *type, const char *name);
static void addScriptEventOut(const char *type, const char *name);
static void enterScriptField(const char *type, const char *name);
static void enterScriptExposedField(const char *type, const char *name);
static void exitScriptExposedField(VrmlField *value);
static void addScriptExposedField(const char *type, const char *name);
static void exitScriptField(VrmlField *value);

static VrmlMFNode *nodeListToMFNode(vector<VrmlNode *> *nodeList);

static vector<VrmlNode *> *addNodeToList(vector<VrmlNode *> *nodeList, VrmlNode *node);

static void addNode(VrmlNode *);
static void addRoute(const char *, const char *, const char *, const char *);

static VrmlField *addIS(const char *);
static VrmlField *addEventIS(const char *, const char *);

static VrmlNode *lookupNode(const char *);

void yyerror(const char *);
int yylex(void);

/* Enabling traces.  */
#ifndef YYDEBUG
#define YYDEBUG 1
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
#undef YYERROR_VERBOSE
#define YYERROR_VERBOSE 1
#else
#define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
#define YYTOKEN_TABLE 0
#endif

#if !defined YYSTYPE && !defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 139 "parser.yy"
{
    char *string;
    VrmlField *field;
    VrmlNode *node;
    vector<VrmlNode *> *nodeList;
}
/* Line 187 of yacc.c.  */
#line 309 "parser.tab.c"
YYSTYPE;
#define yystype YYSTYPE /* obsolescent; will be withdrawn */
#define YYSTYPE_IS_DECLARED 1
#define YYSTYPE_IS_TRIVIAL 1
#endif

/* Copy the second part of user declarations.  */

/* Line 216 of yacc.c.  */
#line 322 "parser.tab.c"

#ifdef short
#undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif(defined __STDC__ || defined __C99__FUNC__ \
      || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
#ifdef __SIZE_TYPE__
#define YYSIZE_T __SIZE_TYPE__
#elif defined size_t
#define YYSIZE_T size_t
#elif !defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
                            || defined __cplusplus || defined _MSC_VER)
#include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#define YYSIZE_T size_t
#else
#define YYSIZE_T unsigned int
#endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T)-1)

#ifndef YY_
#if YYENABLE_NLS
#if ENABLE_NLS
#include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#define YY_(msgid) dgettext("bison-runtime", msgid)
#endif
#endif
#ifndef YY_
#define YY_(msgid) msgid
#endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if !defined lint || defined __GNUC__
#define YYUSE(e) ((void)(e))
#else
#define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
#define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID(int i)
#else
static int
    YYID(i) int i;
#endif
{
    return i;
}
#endif

#if !defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

#ifdef YYSTACK_USE_ALLOCA
#if YYSTACK_USE_ALLOCA
#ifdef __GNUC__
#define YYSTACK_ALLOC __builtin_alloca
#elif defined __BUILTIN_VA_ARG_INCR
#include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#elif defined _AIX
#define YYSTACK_ALLOC __alloca
#elif defined _MSC_VER
#include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#define alloca _alloca
#else
#define YYSTACK_ALLOC alloca
#if !defined _ALLOCA_H && !defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
                                                 || defined __cplusplus || defined _MSC_VER)
#include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#ifndef _STDLIB_H
#define _STDLIB_H 1
#endif
#endif
#endif
#endif
#endif

#ifdef YYSTACK_ALLOC
/* Pacify GCC's `empty if-body' warning.  */
#define YYSTACK_FREE(Ptr) \
    do                    \
    { /* empty */         \
        ;                 \
    } while (YYID(0))
#ifndef YYSTACK_ALLOC_MAXIMUM
/* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#endif
#else
#define YYSTACK_ALLOC YYMALLOC
#define YYSTACK_FREE YYFREE
#ifndef YYSTACK_ALLOC_MAXIMUM
#define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#endif
#if (defined __cplusplus && !defined _STDLIB_H \
     && !((defined YYMALLOC || defined malloc) \
          && (defined YYFREE || defined free)))
#include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#ifndef _STDLIB_H
#define _STDLIB_H 1
#endif
#endif
#ifndef YYMALLOC
#define YYMALLOC malloc
#if !defined malloc && !defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
                                              || defined __cplusplus || defined _MSC_VER)
void *malloc(YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#endif
#endif
#ifndef YYFREE
#define YYFREE free
#if !defined free && !defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
                                            || defined __cplusplus || defined _MSC_VER)
void free(void *); /* INFRINGES ON USER NAME SPACE */
#endif
#endif
#endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */

#if (!defined yyoverflow      \
     && (!defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
    yytype_int16 yyss;
    YYSTYPE yyvs;
};

/* The size of the maximum gap between one aligned stack and the next.  */
#define YYSTACK_GAP_MAXIMUM (sizeof(union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
#define YYSTACK_BYTES(N)                            \
    ((N) * (sizeof(yytype_int16) + sizeof(YYSTYPE)) \
     + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
#ifndef YYCOPY
#if defined __GNUC__ && 1 < __GNUC__
#define YYCOPY(To, From, Count) \
    __builtin_memcpy(To, From, (Count) * sizeof(*(From)))
#else
#define YYCOPY(To, From, Count)             \
    do                                      \
    {                                       \
        YYSIZE_T yyi;                       \
        for (yyi = 0; yyi < (Count); yyi++) \
            (To)[yyi] = (From)[yyi];        \
    } while (YYID(0))
#endif
#endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
#define YYSTACK_RELOCATE(Stack)                                          \
    do                                                                   \
    {                                                                    \
        YYSIZE_T yynewbytes;                                             \
        YYCOPY(&yyptr->Stack, Stack, yysize);                            \
        Stack = &yyptr->Stack;                                           \
        yynewbytes = yystacksize * sizeof(*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof(*yyptr);                            \
    } while (YYID(0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL 3
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST 147

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS 40
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS 32
/* YYNRULES -- Number of rules.  */
#define YYNRULES 80
/* YYNRULES -- Number of states.  */
#define YYNSTATES 141

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK 2
#define YYMAXUTOK 289

#define YYTRANSLATE(YYX) \
    ((unsigned int)(YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] = {
    0, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 39, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 35, 2, 36, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 37, 2, 38, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 1, 2, 3, 4,
    5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint8 yyprhs[] = {
    0, 0, 3, 5, 6, 9, 11, 13, 15, 17,
    18, 23, 26, 28, 30, 31, 41, 42, 43, 52,
    53, 56, 60, 64, 65, 71, 72, 78, 79, 82,
    86, 90, 94, 98, 107, 108, 114, 115, 118, 119,
    123, 125, 127, 131, 135, 136, 142, 143, 149, 150,
    157, 158, 165, 166, 173, 175, 177, 179, 181, 183,
    185, 187, 189, 191, 193, 195, 197, 199, 201, 203,
    205, 207, 210, 213, 216, 219, 223, 227, 231, 233,
    234
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] = {
    41, 0, -1, 42, -1, -1, 42, 43, -1, 44,
    -1, 46, -1, 58, -1, 59, -1, -1, 4, 3,
    45, 59, -1, 5, 3, -1, 47, -1, 49, -1,
    -1, 6, 3, 48, 35, 52, 36, 37, 42, 38,
    -1, -1, -1, 7, 3, 50, 35, 56, 36, 51,
    69, -1, -1, 52, 53, -1, 12, 3, 3, -1,
    13, 3, 3, -1, -1, 14, 3, 3, 54, 69,
    -1, -1, 15, 3, 3, 55, 69, -1, -1, 56,
    57, -1, 12, 3, 3, -1, 13, 3, 3, -1,
    14, 3, 3, -1, 15, 3, 3, -1, 10, 3,
    39, 3, 8, 3, 39, 3, -1, -1, 3, 60,
    37, 61, 38, -1, -1, 61, 62, -1, -1, 3,
    63, 69, -1, 58, -1, 46, -1, 12, 3, 3,
    -1, 13, 3, 3, -1, -1, 14, 3, 3, 64,
    69, -1, -1, 15, 3, 3, 65, 69, -1, -1,
    12, 3, 3, 66, 9, 3, -1, -1, 13, 3,
    3, 67, 9, 3, -1, -1, 15, 3, 3, 68,
    9, 3, -1, 26, -1, 27, -1, 28, -1, 29,
    -1, 30, -1, 31, -1, 32, -1, 16, -1, 17,
    -1, 18, -1, 22, -1, 19, -1, 20, -1, 23,
    -1, 21, -1, 24, -1, 25, -1, 33, 44, -1,
    33, 11, -1, 34, 70, -1, 9, 3, -1, 33,
    9, 3, -1, 34, 9, 3, -1, 35, 71, 36,
    -1, 44, -1, -1, 71, 44, -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] = {
    0, 167, 167, 170, 172, 176, 177, 178, 182, 183,
    183, 185, 189, 190, 194, 194, 200, 202, 200, 206,
    208, 212, 214, 216, 216, 219, 219, 224, 226, 230,
    232, 234, 236, 241, 247, 247, 251, 253, 257, 257,
    259, 260, 263, 265, 267, 267, 270, 270, 273, 273,
    275, 275, 277, 277, 282, 283, 284, 285, 286, 287,
    288, 290, 291, 292, 293, 294, 295, 296, 297, 298,
    299, 301, 302, 303, 305, 306, 307, 312, 313, 317,
    319
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] = {
    "$end", "error", "$undefined", "IDENTIFIER", "DEF", "USE", "PROTO",
    "EXTERNPROTO", "TO", "IS", "ROUTE", "SFN_NULL", "EVENTIN", "EVENTOUT",
    "FIELD", "EXPOSEDFIELD", "SF_BOOL", "SF_COLOR", "SF_FLOAT", "SF_INT32",
    "SF_ROTATION", "SF_TIME", "SF_IMAGE", "SF_STRING", "SF_VEC2F",
    "SF_VEC3F", "MF_COLOR", "MF_FLOAT", "MF_INT32", "MF_ROTATION",
    "VRML_MF_STRING", "MF_VEC2F", "MF_VEC3F", "SF_NODE", "MF_NODE", "'['",
    "']'", "'{'", "'}'", "'.'", "$accept", "vrmlscene", "declarations",
    "declaration", "nodeDeclaration", "@1", "protoDeclaration", "proto",
    "@2", "externproto", "@3", "@4", "interfaceDeclarations",
    "interfaceDeclaration", "@5", "@6", "externInterfaceDeclarations",
    "externInterfaceDeclaration", "routeDeclaration", "node", "@7",
    "nodeGuts", "nodeGut", "@8", "@9", "@10", "@11", "@12", "@13",
    "fieldValue", "mfnodeValue", "nodes", 0
};
#endif

#ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] = {
    0, 256, 257, 258, 259, 260, 261, 262, 263, 264,
    265, 266, 267, 268, 269, 270, 271, 272, 273, 274,
    275, 276, 277, 278, 279, 280, 281, 282, 283, 284,
    285, 286, 287, 288, 289, 91, 93, 123, 125, 46
};
#endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] = {
    0, 40, 41, 42, 42, 43, 43, 43, 44, 45,
    44, 44, 46, 46, 48, 47, 50, 51, 49, 52,
    52, 53, 53, 54, 53, 55, 53, 56, 56, 57,
    57, 57, 57, 58, 60, 59, 61, 61, 63, 62,
    62, 62, 62, 62, 64, 62, 65, 62, 66, 62,
    67, 62, 68, 62, 69, 69, 69, 69, 69, 69,
    69, 69, 69, 69, 69, 69, 69, 69, 69, 69,
    69, 69, 69, 69, 69, 69, 69, 70, 70, 71,
    71
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] = {
    0, 2, 1, 0, 2, 1, 1, 1, 1, 0,
    4, 2, 1, 1, 0, 9, 0, 0, 8, 0,
    2, 3, 3, 0, 5, 0, 5, 0, 2, 3,
    3, 3, 3, 8, 0, 5, 0, 2, 0, 3,
    1, 1, 3, 3, 0, 5, 0, 5, 0, 6,
    0, 6, 0, 6, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 3, 3, 3, 1, 0,
    2
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] = {
    3, 0, 2, 1, 34, 0, 0, 0, 0, 0,
    4, 5, 6, 12, 13, 7, 8, 0, 9, 11,
    14, 16, 0, 36, 0, 0, 0, 0, 0, 10,
    19, 27, 0, 38, 0, 0, 0, 0, 35, 41,
    40, 37, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 20, 0, 0, 0, 0,
    17, 28, 0, 0, 61, 62, 63, 65, 66, 68,
    64, 67, 69, 70, 54, 55, 56, 57, 58, 59,
    60, 0, 0, 39, 42, 43, 44, 46, 0, 0,
    0, 0, 3, 0, 0, 0, 0, 0, 0, 74,
    0, 72, 71, 0, 79, 78, 73, 0, 0, 0,
    0, 0, 21, 22, 23, 25, 0, 29, 30, 31,
    32, 18, 33, 75, 76, 0, 0, 0, 45, 47,
    0, 0, 0, 15, 77, 80, 49, 51, 53, 24,
    26
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] = {
    -1, 1, 2, 10, 11, 24, 12, 13, 25, 14,
    26, 97, 42, 55, 131, 132, 43, 61, 15, 16,
    17, 28, 41, 45, 109, 110, 107, 108, 111, 83,
    106, 125
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -90
static const yytype_int8 yypact[] = {
    -90, 3, 79, -90, -90, 1, 4, 7, 26, 27,
    -90, -90, -90, -90, -90, -90, -90, -6, -90, -90,
    -90, -90, 9, -90, 35, 11, 14, 48, -1, -90,
    -90, -90, 45, -90, 51, 52, 84, 85, -90, -90,
    -90, -90, 21, 66, 87, 43, 88, 89, 90, 91,
    92, 93, 94, 95, 62, -90, 97, 98, 100, 101,
    -90, -90, 67, 102, -90, -90, -90, -90, -90, -90,
    -90, -90, -90, -90, -90, -90, -90, -90, -90, -90,
    -90, 36, 23, -90, 99, 103, -90, -90, 104, 106,
    107, 108, -90, 110, 111, 112, 113, 43, 114, -90,
    115, -90, -90, 116, -90, -90, -90, 117, 118, 43,
    43, 119, -90, -90, -90, -90, 12, -90, -90, -90,
    -90, -90, -90, -90, -90, 20, 120, 121, -90, -90,
    122, 43, 43, -90, -90, -90, -90, -90, -90, -90,
    -90
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] = {
    -90, -90, 28, -90, -81, -90, 105, -90, -90, -90,
    -90, -90, -90, -90, -90, -90, -90, -90, 109, 123,
    -90, -90, -90, -90, -90, -90, -90, -90, -90, -89,
    -90, -90
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -51
static const yytype_int16 yytable[] = {
    102, 105, 33, 3, 18, 7, 8, 19, 121, 9,
    20, 34, 35, 36, 37, 4, 5, 6, 7, 8,
    128, 129, 9, 4, 5, 6, 4, 5, 6, 21,
    22, 23, 103, 50, 51, 52, 53, 38, 4, 4,
    5, 6, 139, 140, 135, 100, 30, 101, 27, 31,
    133, 32, 63, 44, 46, 47, 134, 54, 104, 64,
    65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 56, 57,
    58, 59, 4, 5, 6, 7, 8, 48, 49, 9,
    62, 84, 85, 86, 87, 88, 89, 90, 91, 92,
    93, 94, 60, 95, 96, 99, 98, 112, -48, 113,
    114, 115, -50, 117, 118, 119, 120, 122, 123, 124,
    116, 0, 0, 136, 137, 138, 126, 127, 130, 0,
    0, 0, 0, 39, 0, 0, 0, 40, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 29
};

static const yytype_int16 yycheck[] = {
    81, 82, 3, 0, 3, 6, 7, 3, 97, 10,
    3, 12, 13, 14, 15, 3, 4, 5, 6, 7,
    109, 110, 10, 3, 4, 5, 3, 4, 5, 3,
    3, 37, 9, 12, 13, 14, 15, 38, 3, 3,
    4, 5, 131, 132, 125, 9, 35, 11, 39, 35,
    38, 3, 9, 8, 3, 3, 36, 36, 35, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 12, 13,
    14, 15, 3, 4, 5, 6, 7, 3, 3, 10,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 37,
    3, 3, 36, 3, 3, 3, 39, 3, 9, 3,
    3, 3, 9, 3, 3, 3, 3, 3, 3, 3,
    92, -1, -1, 3, 3, 3, 9, 9, 9, -1,
    -1, -1, -1, 28, -1, -1, -1, 28, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, 24
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] = {
    0, 41, 42, 0, 3, 4, 5, 6, 7, 10,
    43, 44, 46, 47, 49, 58, 59, 60, 3, 3,
    3, 3, 3, 37, 45, 48, 50, 39, 61, 59,
    35, 35, 3, 3, 12, 13, 14, 15, 38, 46,
    58, 62, 52, 56, 8, 63, 3, 3, 3, 3,
    12, 13, 14, 15, 36, 53, 12, 13, 14, 15,
    36, 57, 3, 9, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 69, 3, 3, 3, 3, 3, 3,
    3, 3, 37, 3, 3, 3, 3, 51, 39, 3,
    9, 11, 44, 9, 35, 44, 70, 66, 67, 64,
    65, 68, 3, 3, 3, 3, 42, 3, 3, 3,
    3, 69, 3, 3, 3, 71, 9, 9, 69, 69,
    9, 54, 55, 38, 36, 44, 3, 3, 3, 69,
    69
};

#define yyerrok (yyerrstatus = 0)
#define yyclearin (yychar = YYEMPTY)
#define YYEMPTY (-2)
#define YYEOF 0

#define YYACCEPT goto yyacceptlab
#define YYABORT goto yyabortlab
#define YYERROR goto yyerrorlab

/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL goto yyerrlab

#define YYRECOVERING() (!!yyerrstatus)

#define YYBACKUP(Token, Value)                            \
    do                                                    \
        if (yychar == YYEMPTY && yylen == 1)              \
        {                                                 \
            yychar = (Token);                             \
            yylval = (Value);                             \
            yytoken = YYTRANSLATE(yychar);                \
            YYPOPSTACK(1);                                \
            goto yybackup;                                \
        }                                                 \
        else                                              \
        {                                                 \
            yyerror(YY_("syntax error: cannot back up")); \
            YYERROR;                                      \
        }                                                 \
    while (YYID(0))

#define YYTERROR 1
#define YYERRCODE 256

/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
#define YYLLOC_DEFAULT(Current, Rhs, N)                                                    \
    do                                                                                     \
        if (YYID(N))                                                                       \
        {                                                                                  \
            (Current).first_line = YYRHSLOC(Rhs, 1).first_line;                            \
            (Current).first_column = YYRHSLOC(Rhs, 1).first_column;                        \
            (Current).last_line = YYRHSLOC(Rhs, N).last_line;                              \
            (Current).last_column = YYRHSLOC(Rhs, N).last_column;                          \
        }                                                                                  \
        else                                                                               \
        {                                                                                  \
            (Current).first_line = (Current).last_line = YYRHSLOC(Rhs, 0).last_line;       \
            (Current).first_column = (Current).last_column = YYRHSLOC(Rhs, 0).last_column; \
        }                                                                                  \
    while (YYID(0))
#endif

/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
#if YYLTYPE_IS_TRIVIAL
#define YY_LOCATION_PRINT(File, Loc)              \
    fprintf(File, "%d.%d-%d.%d",                  \
            (Loc).first_line, (Loc).first_column, \
            (Loc).last_line, (Loc).last_column)
#else
#define YY_LOCATION_PRINT(File, Loc) ((void)0)
#endif
#endif

/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
#define YYLEX yylex(YYLEX_PARAM)
#else
#define YYLEX yylex()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

#ifndef YYFPRINTF
#include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#define YYFPRINTF fprintf
#endif

#define YYDPRINTF(Args)     \
    do                      \
    {                       \
        if (yydebug)        \
            YYFPRINTF Args; \
    } while (YYID(0))

#define YY_SYMBOL_PRINT(Title, Type, Value, Location) \
    do                                                \
    {                                                 \
        if (yydebug)                                  \
        {                                             \
            YYFPRINTF(stderr, "%s ", Title);          \
            yy_symbol_print(stderr, Type, Value);     \
            YYFPRINTF(stderr, "\n");                  \
        }                                             \
    } while (YYID(0))

/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print(FILE *yyoutput, int yytype, YYSTYPE const *const yyvaluep)
#else
static void
    yy_symbol_value_print(yyoutput, yytype, yyvaluep)
        FILE *yyoutput;
int yytype;
YYSTYPE const *const yyvaluep;
#endif
{
    if (!yyvaluep)
        return;
#ifdef YYPRINT
    if (yytype < YYNTOKENS)
        YYPRINT(yyoutput, yytoknum[yytype], *yyvaluep);
#else
    YYUSE(yyoutput);
#endif
    switch (yytype)
    {
    default:
        break;
    }
}

/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print(FILE *yyoutput, int yytype, YYSTYPE const *const yyvaluep)
#else
static void
    yy_symbol_print(yyoutput, yytype, yyvaluep)
        FILE *yyoutput;
int yytype;
YYSTYPE const *const yyvaluep;
#endif
{
    if (yytype < YYNTOKENS)
        YYFPRINTF(yyoutput, "token %s (", yytname[yytype]);
    else
        YYFPRINTF(yyoutput, "nterm %s (", yytname[yytype]);

    yy_symbol_value_print(yyoutput, yytype, yyvaluep);
    YYFPRINTF(yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print(yytype_int16 *bottom, yytype_int16 *top)
#else
static void
    yy_stack_print(bottom, top)
        yytype_int16 *bottom;
yytype_int16 *top;
#endif
{
    YYFPRINTF(stderr, "Stack now");
    for (; bottom <= top; ++bottom)
        YYFPRINTF(stderr, " %d", *bottom);
    YYFPRINTF(stderr, "\n");
}

#define YY_STACK_PRINT(Bottom, Top)          \
    do                                       \
    {                                        \
        if (yydebug)                         \
            yy_stack_print((Bottom), (Top)); \
    } while (YYID(0))

/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print(YYSTYPE *yyvsp, int yyrule)
#else
static void
    yy_reduce_print(yyvsp, yyrule)
        YYSTYPE *yyvsp;
int yyrule;
#endif
{
    int yynrhs = yyr2[yyrule];
    int yyi;
    unsigned long int yylno = yyrline[yyrule];
    YYFPRINTF(stderr, "Reducing stack by rule %d (line %lu):\n",
              yyrule - 1, yylno);
    /* The symbols being reduced.  */
    for (yyi = 0; yyi < yynrhs; yyi++)
    {
        fprintf(stderr, "   $%d = ", yyi + 1);
        yy_symbol_print(stderr, yyrhs[yyprhs[yyrule] + yyi],
                        &(yyvsp[(yyi + 1) - (yynrhs)]));
        fprintf(stderr, "\n");
    }
}

#define YY_REDUCE_PRINT(Rule)             \
    do                                    \
    {                                     \
        if (yydebug)                      \
            yy_reduce_print(yyvsp, Rule); \
    } while (YYID(0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
#define YYDPRINTF(Args)
#define YY_SYMBOL_PRINT(Title, Type, Value, Location)
#define YY_STACK_PRINT(Bottom, Top)
#define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */

/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
#define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
#define YYMAXDEPTH 10000
#endif

#if YYERROR_VERBOSE

#ifndef yystrlen
#if defined __GLIBC__ && defined _STRING_H
#define yystrlen strlen
#else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen(const char *yystr)
#else
static YYSIZE_T
    yystrlen(yystr)
        const char *yystr;
#endif
{
    YYSIZE_T yylen;
    for (yylen = 0; yystr[yylen]; yylen++)
        continue;
    return yylen;
}
#endif
#endif

#ifndef yystpcpy
#if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#define yystpcpy stpcpy
#else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy(char *yydest, const char *yysrc)
#else
static char *
    yystpcpy(yydest, yysrc) char *yydest;
const char *yysrc;
#endif
{
    char *yyd = yydest;
    const char *yys = yysrc;

    while ((*yyd++ = *yys++) != '\0')
        continue;

    return yyd - 1;
}
#endif
#endif

#ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr(char *yyres, const char *yystr)
{
    if (*yystr == '"')
    {
        YYSIZE_T yyn = 0;
        char const *yyp = yystr;

        for (;;)
            switch (*++yyp)
            {
            case '\'':
            case ',':
                goto do_not_strip_quotes;

            case '\\':
                if (*++yyp != '\\')
                    goto do_not_strip_quotes;
            /* Fall through.  */
            default:
                if (yyres)
                    yyres[yyn] = *yyp;
                yyn++;
                break;

            case '"':
                if (yyres)
                    yyres[yyn] = '\0';
                return yyn;
            }
    do_not_strip_quotes:
        ;
    }

    if (!yyres)
        return yystrlen(yystr);

    return yystpcpy(yyres, yystr) - yyres;
}
#endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error(char *yyresult, int yystate, int yychar)
{
    int yyn = yypact[yystate];

    if (!(YYPACT_NINF < yyn && yyn <= YYLAST))
        return 0;
    else
    {
        int yytype = YYTRANSLATE(yychar);
        YYSIZE_T yysize0 = yytnamerr(0, yytname[yytype]);
        YYSIZE_T yysize = yysize0;
        YYSIZE_T yysize1;
        int yysize_overflow = 0;
        enum
        {
            YYERROR_VERBOSE_ARGS_MAXIMUM = 5
        };
        char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
        int yyx;

#if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
#endif
        char *yyfmt;
        char const *yyf;
        static char const yyunexpected[] = "syntax error, unexpected %s";
        static char const yyexpecting[] = ", expecting %s";
        static char const yyor[] = " or %s";
        char yyformat[sizeof yyunexpected
                      + sizeof yyexpecting - 1
                      + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
                         * (sizeof yyor - 1))];
        char const *yyprefix = yyexpecting;

        /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
        int yyxbegin = yyn < 0 ? -yyn : 0;

        /* Stay within bounds of both yycheck and yytname.  */
        int yychecklim = YYLAST - yyn + 1;
        int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
        int yycount = 1;

        yyarg[0] = yytname[yytype];
        yyfmt = yystpcpy(yyformat, yyunexpected);

        for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
            {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                {
                    yycount = 1;
                    yysize = yysize0;
                    yyformat[sizeof yyunexpected - 1] = '\0';
                    break;
                }
                yyarg[yycount++] = yytname[yyx];
                yysize1 = yysize + yytnamerr(0, yytname[yyx]);
                yysize_overflow |= (yysize1 < yysize);
                yysize = yysize1;
                yyfmt = yystpcpy(yyfmt, yyprefix);
                yyprefix = yyor;
            }

        yyf = YY_(yyformat);
        yysize1 = yysize + yystrlen(yyf);
        yysize_overflow |= (yysize1 < yysize);
        yysize = yysize1;

        if (yysize_overflow)
            return YYSIZE_MAXIMUM;

        if (yyresult)
        {
            /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
            char *yyp = yyresult;
            int yyi = 0;
            while ((*yyp = *yyf) != '\0')
            {
                if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
                {
                    yyp += yytnamerr(yyp, yyarg[yyi++]);
                    yyf += 2;
                }
                else
                {
                    yyp++;
                    yyf++;
                }
            }
        }
        return yysize;
    }
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct(const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
    yydestruct(yymsg, yytype, yyvaluep)
        const char *yymsg;
int yytype;
YYSTYPE *yyvaluep;
#endif
{
    YYUSE(yyvaluep);

    if (!yymsg)
        yymsg = "Deleting";
    YY_SYMBOL_PRINT(yymsg, yytype, yyvaluep, yylocationp);

    switch (yytype)
    {

    default:
        break;
    }
}

/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse(void *YYPARSE_PARAM);
#else
int yyparse();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse(void);
#else
int yyparse();
#endif
#endif /* ! YYPARSE_PARAM */

/* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;

/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse(void *YYPARSE_PARAM)
#else
int
    yyparse(YYPARSE_PARAM) void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse(void)
#else
int
yyparse()

#endif
#endif
{

    int yystate;
    int yyn;
    int yyresult;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;
    /* Look-ahead token as an internal (translated) token number.  */
    int yytoken = 0;
#if YYERROR_VERBOSE
    /* Buffer for error messages, and its allocated size.  */
    char yymsgbuf[128];
    char *yymsg = yymsgbuf;
    YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

    /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss = yyssa;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs = yyvsa;
    YYSTYPE *yyvsp;

#define YYPOPSTACK(N) (yyvsp -= (N), yyssp -= (N))

    YYSIZE_T yystacksize = YYINITDEPTH;

    /* The variables used to return semantic value and location from the
     action routines.  */
    YYSTYPE yyval;

    /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
    int yylen = 0;

    YYDPRINTF((stderr, "Starting parse\n"));

    yystate = 0;
    yyerrstatus = 0;
    yynerrs = 0;
    yychar = YYEMPTY; /* Cause a token to be read.  */

    /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

    yyssp = yyss;
    yyvsp = yyvs;

    goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
    /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
    yyssp++;

yysetstate:
    *yyssp = yystate;

    if (yyss + yystacksize - 1 <= yyssp)
    {
        /* Get the current used size of the three stacks, in elements.  */
        YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
        {
            /* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
            YYSTYPE *yyvs1 = yyvs;
            yytype_int16 *yyss1 = yyss;

            /* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
            yyoverflow(YY_("memory exhausted"),
                       &yyss1, yysize * sizeof(*yyssp),
                       &yyvs1, yysize * sizeof(*yyvsp),

                       &yystacksize);

            yyss = yyss1;
            yyvs = yyvs1;
        }
#else /* no yyoverflow */
#ifndef YYSTACK_RELOCATE
        goto yyexhaustedlab;
#else
        /* Extend the stack our own way.  */
        if (YYMAXDEPTH <= yystacksize)
            goto yyexhaustedlab;
        yystacksize *= 2;
        if (YYMAXDEPTH < yystacksize)
            yystacksize = YYMAXDEPTH;

        {
            yytype_int16 *yyss1 = yyss;
            union yyalloc *yyptr = (union yyalloc *)YYSTACK_ALLOC(YYSTACK_BYTES(yystacksize));
            if (!yyptr)
                goto yyexhaustedlab;
            YYSTACK_RELOCATE(yyss);
            YYSTACK_RELOCATE(yyvs);

#undef YYSTACK_RELOCATE
            if (yyss1 != yyssa)
                YYSTACK_FREE(yyss1);
        }
#endif
#endif /* no yyoverflow */

        yyssp = yyss + yysize - 1;
        yyvsp = yyvs + yysize - 1;

        YYDPRINTF((stderr, "Stack size increased to %lu\n",
                   (unsigned long int)yystacksize));

        if (yyss + yystacksize - 1 <= yyssp)
            YYABORT;
    }

    YYDPRINTF((stderr, "Entering state %d\n", yystate));

    goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

    /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

    /* First try to decide what to do without reference to look-ahead token.  */
    yyn = yypact[yystate];
    if (yyn == YYPACT_NINF)
        goto yydefault;

    /* Not known => get a look-ahead token if don't already have one.  */

    /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
    if (yychar == YYEMPTY)
    {
        YYDPRINTF((stderr, "Reading a token: "));
        yychar = YYLEX;
    }

    if (yychar <= YYEOF)
    {
        yychar = yytoken = YYEOF;
        YYDPRINTF((stderr, "Now at end of input.\n"));
    }
    else
    {
        yytoken = YYTRANSLATE(yychar);
        YY_SYMBOL_PRINT("Next token is", yytoken, &yylval, &yylloc);
    }

    /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
    yyn += yytoken;
    if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
        goto yydefault;
    yyn = yytable[yyn];
    if (yyn <= 0)
    {
        if (yyn == 0 || yyn == YYTABLE_NINF)
            goto yyerrlab;
        yyn = -yyn;
        goto yyreduce;
    }

    if (yyn == YYFINAL)
        YYACCEPT;

    /* Count tokens shifted since error; after three, turn off error
     status.  */
    if (yyerrstatus)
        yyerrstatus--;

    /* Shift the look-ahead token.  */
    YY_SYMBOL_PRINT("Shifting", yytoken, &yylval, &yylloc);

    /* Discard the shifted token unless it is eof.  */
    if (yychar != YYEOF)
        yychar = YYEMPTY;

    yystate = yyn;
    *++yyvsp = yylval;

    goto yynewstate;

/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
    yyn = yydefact[yystate];
    if (yyn == 0)
        goto yyerrlab;
    goto yyreduce;

/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
    /* yyn is the number of a rule to reduce with.  */
    yylen = yyr2[yyn];

    /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
    yyval = yyvsp[1 - yylen];

    YY_REDUCE_PRINT(yyn);
    switch (yyn)
    {
    case 5:
#line 176 "parser.yy"
    {
        addNode((yyvsp[(1) - (1)].node));
    }
    break;

    case 9:
#line 183 "parser.yy"
    {
        nodeName = (yyvsp[(2) - (2)].string);
    }
    break;

    case 10:
#line 184 "parser.yy"
    {
        (yyval.node) = (yyvsp[(4) - (4)].node);
        free((yyvsp[(2) - (4)].string));
    }
    break;

    case 11:
#line 185 "parser.yy"
    {
        (yyval.node) = lookupNode((yyvsp[(2) - (2)].string));
        free((yyvsp[(2) - (2)].string));
    }
    break;

    case 14:
#line 194 "parser.yy"
    {
        beginProto((yyvsp[(2) - (2)].string));
    }
    break;

    case 15:
#line 196 "parser.yy"
    {
        endProto(0);
        free((yyvsp[(2) - (9)].string));
    }
    break;

    case 16:
#line 200 "parser.yy"
    {
        beginProto((yyvsp[(2) - (2)].string));
    }
    break;

    case 17:
#line 202 "parser.yy"
    {
        expect(VrmlField::MFSTRING);
    }
    break;

    case 18:
#line 203 "parser.yy"
    {
        endProto((yyvsp[(8) - (8)].field));
        free((yyvsp[(2) - (8)].string));
    }
    break;

    case 21:
#line 212 "parser.yy"
    {
        addEventIn((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
        free((yyvsp[(2) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 22:
#line 214 "parser.yy"
    {
        addEventOut((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
        free((yyvsp[(2) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 23:
#line 216 "parser.yy"
    {
        expect(addField((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string)));
    }
    break;

    case 24:
#line 217 "parser.yy"
    {
        setFieldDefault((yyvsp[(3) - (5)].string), (yyvsp[(5) - (5)].field));
        free((yyvsp[(2) - (5)].string));
        free((yyvsp[(3) - (5)].string));
    }
    break;

    case 25:
#line 219 "parser.yy"
    {
        expect(addExposedField((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string)));
    }
    break;

    case 26:
#line 220 "parser.yy"
    {
        setFieldDefault((yyvsp[(3) - (5)].string), (yyvsp[(5) - (5)].field));
        free((yyvsp[(2) - (5)].string));
        free((yyvsp[(3) - (5)].string));
    }
    break;

    case 29:
#line 230 "parser.yy"
    {
        addEventIn((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
        free((yyvsp[(2) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 30:
#line 232 "parser.yy"
    {
        addEventOut((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
        free((yyvsp[(2) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 31:
#line 234 "parser.yy"
    {
        addField((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
        free((yyvsp[(2) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 32:
#line 236 "parser.yy"
    {
        addExposedField((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
        free((yyvsp[(2) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 33:
#line 242 "parser.yy"
    {
        addRoute((yyvsp[(2) - (8)].string), (yyvsp[(4) - (8)].string), (yyvsp[(6) - (8)].string), (yyvsp[(8) - (8)].string));
        free((yyvsp[(2) - (8)].string));
        free((yyvsp[(4) - (8)].string));
        free((yyvsp[(6) - (8)].string));
        free((yyvsp[(8) - (8)].string));
    }
    break;

    case 34:
#line 247 "parser.yy"
    {
        enterNode((yyvsp[(1) - (1)].string));
    }
    break;

    case 35:
#line 248 "parser.yy"
    {
        (yyval.node) = exitNode();
        free((yyvsp[(1) - (5)].string));
    }
    break;

    case 38:
#line 257 "parser.yy"
    {
        enterField((yyvsp[(1) - (1)].string));
    }
    break;

    case 39:
#line 258 "parser.yy"
    {
        exitField((yyvsp[(3) - (3)].field));
        free((yyvsp[(1) - (3)].string));
    }
    break;

    case 42:
#line 263 "parser.yy"
    {
        addScriptEventIn((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
        free((yyvsp[(2) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 43:
#line 265 "parser.yy"
    {
        addScriptEventOut((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
        free((yyvsp[(2) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 44:
#line 267 "parser.yy"
    {
        enterScriptField((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
    }
    break;

    case 45:
#line 268 "parser.yy"
    {
        exitScriptField((yyvsp[(5) - (5)].field));
        free((yyvsp[(2) - (5)].string));
        free((yyvsp[(3) - (5)].string));
    }
    break;

    case 46:
#line 270 "parser.yy"
    {
        enterScriptExposedField((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
    }
    break;

    case 47:
#line 271 "parser.yy"
    {
        exitScriptExposedField((yyvsp[(5) - (5)].field));
        free((yyvsp[(2) - (5)].string));
        free((yyvsp[(3) - (5)].string));
    }
    break;

    case 48:
#line 273 "parser.yy"
    {
        addScriptEventIn((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
    }
    break;

    case 49:
#line 274 "parser.yy"
    {
        addEventIS((yyvsp[(3) - (6)].string), (yyvsp[(6) - (6)].string));
        free((yyvsp[(2) - (6)].string));
        free((yyvsp[(3) - (6)].string));
        free((yyvsp[(6) - (6)].string));
    }
    break;

    case 50:
#line 275 "parser.yy"
    {
        addScriptEventOut((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
    }
    break;

    case 51:
#line 276 "parser.yy"
    {
        addEventIS((yyvsp[(3) - (6)].string), (yyvsp[(6) - (6)].string));
        free((yyvsp[(2) - (6)].string));
        free((yyvsp[(3) - (6)].string));
        free((yyvsp[(6) - (6)].string));
    }
    break;

    case 52:
#line 277 "parser.yy"
    {
        addScriptExposedField((yyvsp[(2) - (3)].string), (yyvsp[(3) - (3)].string));
    }
    break;

    case 53:
#line 278 "parser.yy"
    {
        addEventIS((yyvsp[(3) - (6)].string), (yyvsp[(6) - (6)].string));
        free((yyvsp[(2) - (6)].string));
        free((yyvsp[(3) - (6)].string));
        free((yyvsp[(6) - (6)].string));
    }
    break;

    case 71:
#line 301 "parser.yy"
    {
        (yyval.field) = new VrmlSFNode((yyvsp[(2) - (2)].node));
    }
    break;

    case 72:
#line 302 "parser.yy"
    {
        (yyval.field) = 0;
    }
    break;

    case 73:
#line 303 "parser.yy"
    {
        (yyval.field) = (yyvsp[(2) - (2)].field);
    }
    break;

    case 74:
#line 305 "parser.yy"
    {
        (yyval.field) = addIS((yyvsp[(2) - (2)].string));
        free((yyvsp[(2) - (2)].string));
    }
    break;

    case 75:
#line 306 "parser.yy"
    {
        (yyval.field) = addIS((yyvsp[(3) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 76:
#line 307 "parser.yy"
    {
        (yyval.field) = addIS((yyvsp[(3) - (3)].string));
        free((yyvsp[(3) - (3)].string));
    }
    break;

    case 77:
#line 312 "parser.yy"
    {
        (yyval.field) = nodeListToMFNode((yyvsp[(2) - (3)].nodeList));
    }
    break;

    case 78:
#line 313 "parser.yy"
    {
        (yyval.field) = new VrmlMFNode((yyvsp[(1) - (1)].node));
    }
    break;

    case 79:
#line 317 "parser.yy"
    {
        (yyval.nodeList) = 0;
    }
    break;

    case 80:
#line 319 "parser.yy"
    {
        (yyval.nodeList) = addNodeToList((yyvsp[(1) - (2)].nodeList), (yyvsp[(2) - (2)].node));
    }
    break;

/* Line 1267 of yacc.c.  */
#line 1892 "parser.tab.c"
    default:
        break;
    }
    YY_SYMBOL_PRINT("-> $$ =", yyr1[yyn], &yyval, &yyloc);

    YYPOPSTACK(yylen);
    yylen = 0;
    YY_STACK_PRINT(yyss, yyssp);

    *++yyvsp = yyval;

    /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

    yyn = yyr1[yyn];

    yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
    if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
        yystate = yytable[yystate];
    else
        yystate = yydefgoto[yyn - YYNTOKENS];

    goto yynewstate;

/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
    /* If not already recovering from an error, report this error.  */
    if (!yyerrstatus)
    {
        ++yynerrs;
#if !YYERROR_VERBOSE
        yyerror(YY_("syntax error"));
#else
        {
            YYSIZE_T yysize = yysyntax_error(0, yystate, yychar);
            if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
            {
                YYSIZE_T yyalloc = 2 * yysize;
                if (!(yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
                    yyalloc = YYSTACK_ALLOC_MAXIMUM;
                if (yymsg != yymsgbuf)
                    YYSTACK_FREE(yymsg);
                yymsg = (char *)YYSTACK_ALLOC(yyalloc);
                if (yymsg)
                    yymsg_alloc = yyalloc;
                else
                {
                    yymsg = yymsgbuf;
                    yymsg_alloc = sizeof yymsgbuf;
                }
            }

            if (0 < yysize && yysize <= yymsg_alloc)
            {
                (void)yysyntax_error(yymsg, yystate, yychar);
                yyerror(yymsg);
            }
            else
            {
                yyerror(YY_("syntax error"));
                if (yysize != 0)
                    goto yyexhaustedlab;
            }
        }
#endif
    }

    if (yyerrstatus == 3)
    {
        /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

        if (yychar <= YYEOF)
        {
            /* Return failure if at end of input.  */
            if (yychar == YYEOF)
                YYABORT;
        }
        else
        {
            yydestruct("Error: discarding",
                       yytoken, &yylval);
            yychar = YYEMPTY;
        }
    }

    /* Else will try to reuse look-ahead token after shifting the error
     token.  */
    goto yyerrlab1;

/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

    /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
    if (/*CONSTCOND*/ 0)
        goto yyerrorlab;

    /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
    YYPOPSTACK(yylen);
    yylen = 0;
    YY_STACK_PRINT(yyss, yyssp);
    yystate = *yyssp;
    goto yyerrlab1;

/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
    yyerrstatus = 3; /* Each real token shifted decrements this.  */

    for (;;)
    {
        yyn = yypact[yystate];
        if (yyn != YYPACT_NINF)
        {
            yyn += YYTERROR;
            if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
                yyn = yytable[yyn];
                if (0 < yyn)
                    break;
            }
        }

        /* Pop the current state because it cannot handle the error token.  */
        if (yyssp == yyss)
            YYABORT;

        yydestruct("Error: popping",
                   yystos[yystate], yyvsp);
        YYPOPSTACK(1);
        yystate = *yyssp;
        YY_STACK_PRINT(yyss, yyssp);
    }

    if (yyn == YYFINAL)
        YYACCEPT;

    *++yyvsp = yylval;

    /* Shift the error token.  */
    YY_SYMBOL_PRINT("Shifting", yystos[yyn], yyvsp, yylsp);

    yystate = yyn;
    goto yynewstate;

/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
    yyresult = 0;
    goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
    yyresult = 1;
    goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
    yyerror(YY_("memory exhausted"));
    yyresult = 2;
/* Fall through.  */
#endif

yyreturn:
    if (yychar != YYEOF && yychar != YYEMPTY)
        yydestruct("Cleanup: discarding lookahead",
                   yytoken, &yylval);
    /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
    YYPOPSTACK(yylen);
    YY_STACK_PRINT(yyss, yyssp);
    while (yyssp != yyss)
    {
        yydestruct("Cleanup: popping",
                   yystos[*yyssp], yyvsp);
        YYPOPSTACK(1);
    }
#ifndef yyoverflow
    if (yyss != yyssa)
        YYSTACK_FREE(yyss);
#endif
#if YYERROR_VERBOSE
    if (yymsg != yymsgbuf)
        YYSTACK_FREE(yymsg);
#endif
    /* Make sure YYID is used.  */
    return YYID(yyresult);
}

#line 322 "parser.yy"

void
yyerror(const char *msg)
{
    System::the->error("Error near line %d: %s\n", currentLineNumber, msg);
    expect(VrmlField::NO_FIELD);
}

static VrmlNamespace *currentScope()
{
    return currentProtoStack.empty() ? yyNodeTypes : currentProtoStack.front()->scope();
}

static void
beginProto(const char *protoName)
{
    // Need to push node namespace as well, since node names DEF'd in the
    // implementations are not visible (USEable) from the outside and vice
    // versa.

    // Any protos in the implementation are in a local namespace:
    VrmlNodeType *t = new VrmlNodeType(protoName);
    t->setScope(currentScope());
    currentProtoStack.push_front(t);
}

static void
endProto(VrmlField *url)
{
    // Make any node names defined in implementation unavailable: ...

    // Add this proto definition:
    if (currentProtoStack.empty())
    {
        yyerror("Error: Empty PROTO stack");
    }
    else
    {
        VrmlNodeType *t = currentProtoStack.front();
        currentProtoStack.pop_front();
        if (url)
            t->setUrl(url, yyDocument);
        currentScope()->addNodeType(t);
    }
}

static int
inProto()
{
    return !currentProtoStack.empty();
}

// Add a field to a PROTO interface

static FieldType
addField(const char *typeString, const char *name)
{
    FieldType type = fieldType(typeString);

    if (type == VrmlField::NO_FIELD)
    {
        char msg[100];
        sprintf(msg, "invalid field type: %s", typeString);
        yyerror(msg);
        return VrmlField::NO_FIELD;
    }

    // Need to add support for Script nodes:
    // if (inScript) ... ???

    if (currentProtoStack.empty())
    {
        yyerror("field declaration outside of prototype");
        return VrmlField::NO_FIELD;
    }
    VrmlNodeType *t = currentProtoStack.front();
    t->addField(name, type);

    return type;
}

static FieldType
addEventIn(const char *typeString, const char *name)
{
    FieldType type = fieldType(typeString);

    if (type == VrmlField::NO_FIELD)
    {
        char msg[100];
        sprintf(msg, "invalid eventIn type: %s", typeString);
        yyerror(msg);

        return VrmlField::NO_FIELD;
    }

    if (currentProtoStack.empty())
    {
        yyerror("eventIn declaration outside of PROTO interface");
        return VrmlField::NO_FIELD;
    }
    VrmlNodeType *t = currentProtoStack.front();
    t->addEventIn(name, type);

    return type;
}

static FieldType
addEventOut(const char *typeString, const char *name)
{
    FieldType type = fieldType(typeString);

    if (type == VrmlField::NO_FIELD)
    {
        char msg[100];
        sprintf(msg, "invalid eventOut type: %s", typeString);
        yyerror(msg);

        return VrmlField::NO_FIELD;
    }

    if (currentProtoStack.empty())
    {
        yyerror("eventOut declaration outside of PROTO interface");
        return VrmlField::NO_FIELD;
    }
    VrmlNodeType *t = currentProtoStack.front();
    t->addEventOut(name, type);

    return type;
}

static FieldType
addExposedField(const char *typeString, const char *name)
{
    FieldType type = fieldType(typeString);

    if (type == VrmlField::NO_FIELD)
    {
        char msg[100];
        sprintf(msg, "invalid exposedField type: %s", typeString);
        yyerror(msg);

        return VrmlField::NO_FIELD;
    }

    if (currentProtoStack.empty())
    {
        yyerror("exposedField declaration outside of PROTO interface");
        return VrmlField::NO_FIELD;
    }
    VrmlNodeType *t = currentProtoStack.front();
    t->addExposedField(name, type);

    return type;
}

static void
setFieldDefault(const char *fieldName, VrmlField *value)
{
    if (currentProtoStack.empty())
    {
        yyerror("field default declaration outside of PROTO interface");
    }
    else
    {
        VrmlNodeType *t = currentProtoStack.front();
        t->setFieldDefault(fieldName, value);
        delete value;
    }
}

static FieldType
fieldType(const char *type)
{
    return VrmlField::fieldType(type);
}

static void
enterNode(const char *nodeTypeName)
{
    const VrmlNodeType *t = currentScope()->findType(nodeTypeName);

    if (t == NULL)
    {
        char tmp[256];
        sprintf(tmp, "Unknown node type '%s'", nodeTypeName);
        yyerror(tmp);
    }
    FieldRec *fr = new FieldRec;

    // Create a new node of type t
    fr->node = t ? t->newNode() : 0;

    // The nodeName needs to be set here before the node contents
    // are parsed because the contents can actually reference the
    // node (eg, in ROUTE statements). USEing the nodeName from
    // inside the node is probably a bad idea, and is probably
    // illegal according to the acyclic requirement, but isn't
    // checked for...
    if (nodeName)
    {
        if (fr->node)
        {
            fr->node->setName(nodeName, currentScope());
        }
        nodeName = 0;
    }

    fr->nodeType = t;
    fr->fieldName = NULL;
    currentField.push_front(fr);
}

static VrmlNode *
exitNode()
{
    FieldRec *fr = currentField.front();
    //assert(fr != NULL);

    VrmlNode *n = fr->node;

    currentField.pop_front();

    delete fr;

    return n;
}

static void
enterField(const char *fieldName)
{
    FieldRec *fr = currentField.front();
    //assert(fr != NULL);

    fr->fieldName = fieldName;
    if (fr->nodeType != NULL)
    {

        // This is wrong - it lets eventIns/eventOuts be in nodeGuts. It
        // should only allow this when followed by IS...

        // enterField is called when parsing eventIn and eventOut IS
        // declarations, in which case we don't need to do anything special--
        // the IS IDENTIFIER will be returned from the lexer normally.
        if (fr->nodeType->hasEventIn(fieldName) || fr->nodeType->hasEventOut(fieldName))
            return;

        fr->fieldType = fr->nodeType->hasField(fieldName);

        if (fr->fieldType != 0)
        { // Let the lexer know what field type to expect:
            expect(fr->fieldType);
            expectCoordIndex = (strcmp(fieldName, "coordIndex") == 0);
            expectTexCoordIndex = (strcmp(fieldName, "texCoordIndex") == 0);
        }
        else
        {
            char msg[256];
            sprintf(msg, "%s nodes do not have %s fields/eventIns/eventOuts",
                    fr->nodeType->getName(), fieldName);
            yyerror(msg);
        }
    }
    // else expect(ANY_FIELD);
}

static void
exitField(VrmlField *fieldValue)
{
    FieldRec *fr = currentField.front();
    //assert(fr != NULL);

    if (fieldValue)
        fr->node->setField(fr->fieldName, *fieldValue);
    delete fieldValue; // Assumes setField is copying fieldValue...

    fr->fieldName = NULL;
    fr->fieldType = VrmlField::NO_FIELD;
}

static bool
inScript()
{
    FieldRec *fr = currentField.front();
    if (fr->nodeType == NULL || strcmp(fr->nodeType->getName(), "Script") != 0)
    {
        yyerror("interface declaration outside of Script");
        return false;
    }
    return true;
}

static void
addScriptEventIn(const char *typeString, const char *name)
{
    if (inScript())
    {
        FieldType type = fieldType(typeString);

        if (type == VrmlField::NO_FIELD)
        {
            char msg[100];
            sprintf(msg, "invalid eventIn type: %s", typeString);
            yyerror(msg);
        }

        ((VrmlNodeScript *)currentField.front()->node)->addEventIn(name, type);
    }
}

static void
addScriptEventOut(const char *typeString, const char *name)
{
    if (inScript())
    {
        FieldType type = fieldType(typeString);

        if (type == VrmlField::NO_FIELD)
        {
            char msg[100];
            sprintf(msg, "invalid eventOut type: %s", typeString);
            yyerror(msg);
        }

        ((VrmlNodeScript *)currentField.front()->node)->addEventOut(name, type);
    }
}

static void
addScriptExposedField(const char *typeString, const char *name)
{
    if (inScript())
    {
        FieldType type = fieldType(typeString);

        if (type == VrmlField::NO_FIELD)
        {
            char msg[100];
            sprintf(msg, "invalid eventOut type: %s", typeString);
            yyerror(msg);
        }

        ((VrmlNodeScript *)currentField.front()->node)->addExposedField(name, type);
    }
}

static void
enterScriptExposedField(const char *typeString, const char *fieldName)
{
    if (inScript())
    {
        FieldRec *fr = currentField.front();
        //assert(fr != NULL);

        fr->fieldName = fieldName;
        fr->fieldType = fieldType(typeString);
        if (fr->fieldType == VrmlField::NO_FIELD)
        {
            char msg[100];
            sprintf(msg, "invalid Script field %s type: %s",
                    fieldName, typeString);
            yyerror(msg);
        }
        else
            expect(fr->fieldType);
    }
}

static void
exitScriptExposedField(VrmlField *value)
{
    if (inScript())
    {
        FieldRec *fr = currentField.front();
        //assert(fr != NULL);

        VrmlNodeScript *s = (VrmlNodeScript *)(fr->node);
        s->addExposedField(fr->fieldName, fr->fieldType, value);
        delete value;
        fr->fieldName = NULL;
        fr->fieldType = VrmlField::NO_FIELD;
    }
}

static void
enterScriptField(const char *typeString, const char *fieldName)
{
    if (inScript())
    {
        FieldRec *fr = currentField.front();
        //assert(fr != NULL);

        fr->fieldName = fieldName;
        fr->fieldType = fieldType(typeString);
        if (fr->fieldType == VrmlField::NO_FIELD)
        {
            char msg[100];
            sprintf(msg, "invalid Script field %s type: %s",
                    fieldName, typeString);
            yyerror(msg);
        }
        else
            expect(fr->fieldType);
    }
}

static void
exitScriptField(VrmlField *value)
{
    if (inScript())
    {
        FieldRec *fr = currentField.front();
        //assert(fr != NULL);

        VrmlNodeScript *s = (VrmlNodeScript *)(fr->node);
        s->addField(fr->fieldName, fr->fieldType, value);
        delete value;
        fr->fieldName = NULL;
        fr->fieldType = VrmlField::NO_FIELD;
    }
}

// Find a node by name (in the current namespace)

static VrmlNode *
lookupNode(const char *name)
{
    // Uwe if not found in Proto, look it up in current Namespace
    VrmlNode *n = currentScope()->findNode(name);
    if (n)
        return n;
    else
        return yyNodeTypes->findNode(name);
}

static VrmlMFNode *nodeListToMFNode(vector<VrmlNode *> *nodeList)
{
    VrmlMFNode *r = 0;
    if (nodeList)
    {
        r = new VrmlMFNode(nodeList->size(), &(*nodeList)[0]);
        delete nodeList;
    }
    return r;
}

static vector<VrmlNode *> *addNodeToList(vector<VrmlNode *> *nodeList,
                                         VrmlNode *node)
{
    if (!nodeList)
        nodeList = new vector<VrmlNode *>();
    nodeList->push_back(node);
    return nodeList;
}

static void addNode(VrmlNode *node)
{
    if (!node)
        return;

    if (inProto())
    {
        VrmlNodeType *t = currentProtoStack.front();
        t->addNode(node); // add node to PROTO definition
    }
    else // top level
    { // add node to scene graph
        if (!yyParsedNodes)
            yyParsedNodes = new VrmlMFNode(node);
        else
            yyParsedNodes->addNode(node);
    }
}

static void addRoute(const char *fromNodeName,
                     const char *fromFieldName,
                     const char *toNodeName,
                     const char *toFieldName)
{
    VrmlNode *fromNode = lookupNode(fromNodeName);
    VrmlNode *toNode = lookupNode(toNodeName);

    if (!fromNode || !toNode)
    {
        char msg[256];
        sprintf(msg, "invalid %s node name \"%s\" in ROUTE statement.",
                fromNode ? "destination" : "source",
                fromNode ? toNodeName : fromNodeName);
        yyerror(msg);
    }
    else
    {
        fromNode->addRoute(fromFieldName, toNode, toFieldName);
    }
}

// Store the information linking the current field and node to
// to the PROTO interface field with the PROTO definition.

static VrmlField *addIS(const char *isFieldName)
{
    if (!isFieldName)
    {
        yyerror("invalid IS field name (null)");
        return 0;
    }

    FieldRec *fr = currentField.front();
    if (!fr)
    {
        char msg[256];
        sprintf(msg, "IS statement (%s) without field declaration",
                isFieldName);
        yyerror(msg);
        return 0;
    }

    if (inProto())
    {
        VrmlNodeType *t = currentProtoStack.front();

        if (!t)
        {
            yyerror("invalid PROTO for IS statement");
            return 0;
        }
        else if (!fr->fieldName)
        {
            char msg[256];
            sprintf(msg, "invalid IS interface name (%s) in PROTO %s",
                    isFieldName, t->getName());
            yyerror(msg);
        }

        else
            t->addIS(isFieldName, fr->node, fr->fieldName);
    }

    // Not in PROTO, must be a Script field
    else if (fr->nodeType && strcmp(fr->nodeType->getName(), "Script") == 0)
    {
        return new VrmlSFNode(lookupNode(isFieldName));
    }

    else
    {
        char msg[256];
        sprintf(msg, "IS statement (%s) must be in a PROTO or Script.",
                isFieldName);
        yyerror(msg);
    }

    // Nothing is stored for IS'd fields in the PROTO implementation
    return 0;
}

static VrmlField *addEventIS(const char *fieldName,
                             const char *isFieldName)
{
    FieldRec *fr = currentField.front();
    if (!fr)
    {
        char msg[256];
        sprintf(msg, "IS statement (%s) with no eventIn/eventOut declaration",
                isFieldName);
        yyerror(msg);
    }
    fr->fieldName = fieldName;
    addIS(isFieldName);
    fr->fieldName = 0;
    return 0;
}

// This switch is necessary so the VrmlNodeType code can be independent
// of the parser tokens.

void
expect(FieldType type)
{
    switch (type)
    {
    case VrmlField::SFBOOL:
        expectToken = SF_BOOL;
        break;
    case VrmlField::SFCOLOR:
        expectToken = SF_COLOR;
        break;
    case VrmlField::SFFLOAT:
        expectToken = SF_FLOAT;
        break;
    case VrmlField::SFIMAGE:
        expectToken = SF_IMAGE;
        break;
    case VrmlField::SFINT32:
        expectToken = SF_INT32;
        break;
    case VrmlField::SFROTATION:
        expectToken = SF_ROTATION;
        break;
    case VrmlField::SFSTRING:
        expectToken = SF_STRING;
        break;
    case VrmlField::SFTIME:
        expectToken = SF_TIME;
        break;
    case VrmlField::SFVEC2F:
        expectToken = SF_VEC2F;
        break;
    case VrmlField::SFVEC3F:
        expectToken = SF_VEC3F;
        break;

    case VrmlField::MFCOLOR:
        expectToken = MF_COLOR;
        break;
    case VrmlField::MFFLOAT:
        expectToken = MF_FLOAT;
        break;
    case VrmlField::MFINT32:
        expectToken = MF_INT32;
        break;
    case VrmlField::MFROTATION:
        expectToken = MF_ROTATION;
        break;
    case VrmlField::MFSTRING:
        expectToken = VRML_MF_STRING;
        break;
    case VrmlField::MFVEC2F:
        expectToken = MF_VEC2F;
        break;
    case VrmlField::MFVEC3F:
        expectToken = MF_VEC3F;
        break;

    case VrmlField::MFNODE:
        expectToken = MF_NODE;
        break;
    case VrmlField::SFNODE:
        expectToken = SF_NODE;
        break;
    default:
        expectToken = 0;
        break;
    }
}
