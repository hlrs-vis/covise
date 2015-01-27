/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*=========================================================================

  Program:   Visualization Toolkit
  Module:    $RCSfile: vtkParse.h,v $

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef VTK_PARSE_H
#define VTK_PARSE_H

#include <string>
#include <map>
#include <vector>
#include <cfloat>
#include <climits>

#define MAX_ARGS 20

typedef struct _FunctionInfo
{
    char *Name;
    char *ClassName;
    int NumberOfArguments;
    int ArrayFailure;
    int IsPureVirtual;
    int IsPublic;
    int IsProtected;
    int IsOperator;
    int HaveHint;
    int HintSize;
    int ArgTypes[MAX_ARGS];
    int ArgCounts[MAX_ARGS];
    char *ArgClasses[MAX_ARGS];
    int ReturnType;
    char *ReturnClass;
    char *Comment;
    char *Signature;
    int IsLegacy;
} FunctionInfo;

typedef struct _FileInfo
{
    int HasDelete;
    int IsAbstract;
    int IsConcrete;
    char *ClassName;
    char *FileName;
    char *OutputFileName;

    char *SuperClasses[10];
    int NumberOfSuperClasses;
    int NumberOfFunctions;
    FunctionInfo Functions[1000];
    char *NameComment;
    char *Description;
    char *Caveats;
    char *SeeAlso;
} FileInfo;

extern FileInfo data;
extern FunctionInfo *currentFunction;

extern FILE *fhint;
extern char temps[2048];
extern int in_public;
extern int in_protected;
extern int HaveComment;
extern char CommentText[50000];
extern int CommentState;
extern int openSig;
extern int invertSig;
extern unsigned int sigAllocatedLength;

extern void InitFunction(FunctionInfo *func);
extern int vtkParseparse(void);

extern FILE *yyin, *yyout;

extern int yylineno;

struct Param
{
    Param()
        : getter(false)
        , setter(false)
        , ispublic(true)
        , hasindex(false)
    {
    }
    virtual ~Param()
    {
    }
    bool getter;
    bool setter;
    bool ispublic;
    bool hasindex;
    std::string name;
};

struct ScalarParam : public Param
{
    ScalarParam()
        : choice(false)
        , clamped(false)
        , isbool(false)
        , isint(false)
        , isfloat(false)
    {
    }
    bool choice;
    bool clamped;
    bool isbool;
    bool isint;
    bool isfloat;
};

struct StringParam : public Param
{
    StringParam()
        : filename(false)
    {
    }
    bool filename;
};

struct VecParam : public Param
{
    VecParam()
        : dim(0)
        , isfloat(false)
        , isdouble(false)
        , ischar(false)
        , isunsigned(false)
    {
    }
    int dim;
    bool isfloat;
    bool isdouble;
    bool ischar;
    bool isunsigned;
};

typedef std::map<std::string, Param *> ParamMap;

extern ParamMap paramMap;

typedef std::vector<std::string> ParamBlackList;
extern ParamBlackList paramBlackList;

#endif
