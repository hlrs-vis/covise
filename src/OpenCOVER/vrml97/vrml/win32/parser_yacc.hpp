/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton interface for Bison's Yacc-like parsers in C

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

#if !defined YYSTYPE && !defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
#line 139 "parser.yy"
{
    char *string;
    covise::VrmlField *field;
    covise::VrmlNode *node;
    vector<covise::VrmlNode *> *nodeList;
}
/* Line 1489 of yacc.c.  */
#line 124 "parser.tab.h"
YYSTYPE;
#define yystype YYSTYPE /* obsolescent; will be withdrawn */
#define YYSTYPE_IS_DECLARED 1
#define YYSTYPE_IS_TRIVIAL 1
#endif

extern YYSTYPE parserlval;
