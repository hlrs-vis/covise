/**************************************************
 * VRML 2.0 Parser
 * Copyright (C) 1996 Silicon Graphics, Inc.
 *
 * Author(s): Gavin Bell
 *            Daniel Woods (first port)
 **************************************************
 */
%{

#include "config.h"
#include <string.h>

#include <vector>
using std::vector;

#include "VrmlNode.h"
#include "System.h"

#include "VrmlMFBool.h"
#include "VrmlMFColor.h"
#include "VrmlMFColorRGBA.h"
#include "VrmlMFDouble.h"
#include "VrmlMFFloat.h"
#include "VrmlMFInt.h"
#include "VrmlMFRotation.h"
#include "VrmlMFString.h"
#include "VrmlMFTime.h"
#include "VrmlMFVec2f.h"
#include "VrmlMFVec3f.h"
#include "VrmlMFVec2d.h"
#include "VrmlMFVec3d.h"

#include "VrmlSFBool.h"
#include "VrmlSFColor.h"
#include "VrmlSFColorRGBA.h"
#include "VrmlSFDouble.h"
#include "VrmlSFFloat.h"
#include "VrmlSFImage.h"
#include "VrmlSFInt.h"
#include "VrmlSFRotation.h"
#include "VrmlSFString.h"
#include "VrmlSFTime.h"
#include "VrmlSFVec2f.h"
#include "VrmlSFVec3f.h"
#include "VrmlSFVec2d.h"
#include "VrmlSFVec3d.h"

#ifdef _WIN32
#include <io.h>
#define YY_NO_UNISTD_H
#endif

using namespace vrml;

#include "parser_yacc.hpp"

#define yylval parserlval
#define yyerror parsererror

#define YY_NO_UNPUT 1    /* Not using yyunput/yyless */

#define DEBUG_SSCANF 1   /* errors in return value processing fixed */


char *yyinStr = 0;                      /* For input from strings */
int (*yyinFunc)(char *, int) = 0;       /* For input from functions */

#if HAVE_LIBPNG || HAVE_ZLIB
#include <zlib.h>

gzFile yygz = 0;                        /* For input from gzipped files */

#define YY_INPUT(buf,result,max_size) \
   if (yyinStr) { \
      for (result=0; result<max_size && *yyinStr; ) \
         buf[result++] = *yyinStr++; \
   } else if (yyinFunc) { \
      if ((result = (*yyinFunc)( buf, max_size )) == -1) \
         YY_FATAL_ERROR( "cb input in flex scanner failed" ); \
   } else if (yygz) { \
      if ((result = gzread( yygz, buf, max_size )) == -1) \
         YY_FATAL_ERROR( "gz input in flex scanner failed" ); \
   } else if ( ((result = fread( buf, 1, max_size, yyin )) == 0) \
               && ferror( yyin ) ) \
   YY_FATAL_ERROR( "yyin input in flex scanner failed" );
#else
# define YY_INPUT(buf,result,max_size) \
   if (yyinStr) { \
      for (result=0; result<max_size && *yyinStr; ) \
         buf[result++] = *yyinStr++; \
   } else if ( ((result = fread( buf, 1, max_size, yyin )) == 0) \
               && ferror( yyin ) ) \
   YY_FATAL_ERROR( "input in flex scanner failed" );
#endif

   /* Current line number */
int currentLineNumber = 1;

extern void yyerror(const char *);

   /* The YACC parser sets this to a token to direct the lexer */
   /* in cases where just syntax isn't enough: */
int expectToken = 0;
int expectCoordIndex = 0;
int expectTexCoordIndex = 0;

   /* True when parsing a multiple-valued field: */
static int parsing_mf = 0;

   /* These are used when parsing SFImage fields: */
static int sfImageIntsParsed = 0;
static int sfImageIntsExpected = 0;
static int sfImageNC = 0;
static unsigned char *sfImagePixels = 0;
static unsigned int sfImageMask[] = { 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000 };

   /* These are used when parsing SFString fields: */
#define STRING_SIZE_INCR 1000
static int sfStringMax = 0;
static int sfStringN = 0;
static char *sfStringChars = 0;

   /* flag if code has a X3DV (not VRML97) header */
static int x3d = 0;
   /* flag if code has PROFILE in X3DV header */
static int profile = 0;

static void checkStringSize(int nmore)
{
   if (sfStringN+nmore >= sfStringMax)
   {
      int incr = STRING_SIZE_INCR > nmore ? STRING_SIZE_INCR : (nmore+1);
      char *p = new char[sfStringMax += incr];
      if (sfStringChars)
      {
        strcpy(p, sfStringChars);
        delete [] sfStringChars;
      }
      sfStringChars = p;
   }
}

static void initString()
{
   checkStringSize(0);
   sfStringN = 0;
   sfStringChars[0] = 0;
}

   /* These are used when parsing MF* fields */
// replaced vector by array: speedup 50%
//static vector<int> mfInts;
//static vector<float> mfFloats;
//static vector<char*> mfStrs;
#define IALL 10000
#define FALL 100000
#define DALL 100000
#define SALL 100
#define BALL 100000
static int *mfInts=new int[IALL];
static float *mfFloats=new float[FALL];
static double *mfDoubles=new double[DALL];
static const char **mfStrs=new const char*[SALL];
static bool *mfBools=new bool[BALL];
unsigned int intSize=0;
unsigned int floatSize=0; 
unsigned int doubleSize=0; 
unsigned int strSize=0;
unsigned int boolSize=0;
unsigned int intAllocSize=IALL;
unsigned int floatAllocSize=FALL; 
unsigned int doubleAllocSize=DALL; 
unsigned int strAllocSize=SALL;
unsigned int boolAllocSize=BALL;


#define addInt(i) {mfInts[intSize]=i; intSize++; if(intSize>=intAllocSize) {int *oldData = mfInts; mfInts = new int[intAllocSize+IALL]; memcpy(mfInts,oldData,intAllocSize*sizeof(int)); intAllocSize+=IALL; delete[] oldData;}}
#define addFloat(i) {mfFloats[floatSize]=i; floatSize++; if(floatSize>=floatAllocSize) {float *oldData = mfFloats; mfFloats = new float[floatAllocSize+FALL]; memcpy(mfFloats,oldData,floatAllocSize*sizeof(float)); floatAllocSize+=FALL; delete[] oldData;}}
#define addDouble(i) {mfDoubles[doubleSize]=i; doubleSize++; if(doubleSize>=doubleAllocSize) {double *oldData = mfDoubles; mfDoubles = new double[doubleAllocSize+DALL]; memcpy(mfDoubles,oldData,doubleAllocSize*sizeof(double)); doubleAllocSize+=DALL; delete[] oldData;}}
#define addStr(i) {mfStrs[strSize]=i; strSize++; if(strSize>=strAllocSize) {const char **oldData = mfStrs; mfStrs = new const char*[strAllocSize+SALL]; memcpy(mfStrs,oldData,strAllocSize*sizeof(const char*)); strAllocSize+=SALL; delete[] oldData;}}
#define addBool(i) {fprintf(stderr, "addbool");mfBools[boolSize]=i; boolSize++; if(boolSize>=boolAllocSize) {bool *oldData = mfBools; mfBools = new bool[boolAllocSize+BALL]; memcpy(mfBools,oldData,boolAllocSize*sizeof(bool)); boolAllocSize+=BALL; delete[] oldData;}}

#if 0
#ifdef __cplusplus
extern "C"
#endif
int yywrap(void);
#endif


static char *skip_ws(char *s)
{
   while (*s == ' ' ||
          *s == '\f' ||
          *s == '\n' ||
          *s == '\r' ||
          *s == '\t' ||
          *s == '\v' ||
          *s == ',' ||
          *s == '#')
   {
      if (*s == '#')
      {
         while (*s && *s != '\n') ++s;
      }
      else
      {
         if (*s++ == '\n') ++currentLineNumber;
      }
   }
   return s;
}


%}

   /* Normal state:  parsing nodes.  The initial start state is used */
   /* only to recognize the VRML header. */
%x NODE

   /* Start tokens for all of the field types, */
   /* except for MFNode and SFNode, which are almost completely handled */
   /* by the parser: */
%x SFB SFC SFCR SFD SFF SFIMG SFI SFR SFS SFT SFV2 SFV3 SFV2D SFV3D
%x MFB MFC MFCR MFD MFF MFI MFR MFS MFT MFV2 MFV3 MFV2D MFV3D
%x IN_SFS IN_MFS IN_SFIMG

bool (TRUE|FALSE)

   /* Big hairy expression for floating point numbers: 1.E36*/
float ([-+]?((([0-9]+\.?)|([0-9]*\.?[0-9]+))([eE][+\-]?[0-9]+)?)) 


   /* Ints are decimal or hex (0x##): */
int ([-+]?([0-9]+)|(0[xX][0-9a-fA-F]*))

   /* Whitespace.  Using this pattern can screw up currentLineNumber, */
   /* so it is only used wherever it is really convenient and it is */
   /* extremely unlikely that the user will put in a carriage return */
   /* (example: between the floats in an SFVec3f) */
ws ([ \t\r\n,]|(#.*))+
   /* And the same pattern without the newline */
wsnnl ([ \t\r,]|(#.*))

   /* Legal characters to start an identifier */
idStartChar ([^\x30-\x39\x00-\x20\x22\x23\x27\x2b-\x2e\x5b-\x5d\x7b\x7d])
   /* Legal other characters in an identifier */
idRestChar  ([^\x00-\x20\x22\x23\x27\x2c\x2e\x5b-\x5d\x7b\x7d])

%%

%{
   /* Switch into a new start state if the parser */
   /* just told us that we've read a field name */
   /* and should expect a field value (or IS) */
   if (expectToken != 0) {
#if DEBUG
      extern int yy_flex_debug;
      if (yy_flex_debug)
         fprintf(stderr,"LEX--> Start State %d\n", expectToken);
#endif
      
   /*
    * Annoying.  This big switch is necessary because
    * LEX wants to assign particular numbers to start
    * tokens, and YACC wants to define all the tokens
    * used, too.  Sigh.
    */
   switch(expectToken) {
      case SF_BOOL: BEGIN SFB; break;
      case SF_COLOR: BEGIN SFC; break;
      case SF_COLOR_RGBA: BEGIN SFCR; break;
      case SF_DOUBLE: BEGIN SFD; break;
      case SF_FLOAT: BEGIN SFF; break;
      case SF_IMAGE: BEGIN SFIMG; break;
      case SF_INT32: BEGIN SFI; break;
      case SF_ROTATION: BEGIN SFR; break;
      case SF_STRING: BEGIN SFS; break;
      case SF_TIME: BEGIN SFT; break;
      case SF_VEC2F: BEGIN SFV2; break;
      case SF_VEC3F: BEGIN SFV3; break;
      case SF_VEC2D: BEGIN SFV2D; break;
      case SF_VEC3D: BEGIN SFV3D; break;
      case MF_BOOL: BEGIN MFB; break;
      case MF_COLOR: BEGIN MFC; break;
      case MF_COLOR_RGBA: BEGIN MFCR; break;
      case MF_DOUBLE: BEGIN MFD; break;
      case MF_FLOAT: BEGIN MFF; break;
      case MF_INT32: BEGIN MFI; break;
      case MF_ROTATION: BEGIN MFR; break;
      case VRML_MF_STRING: BEGIN MFS; break;
      case MF_TIME: BEGIN MFT; break;
      case MF_VEC2F: BEGIN MFV2; break;
      case MF_VEC3F: BEGIN MFV3; break;
      case MF_VEC2D: BEGIN MFV2D; break;
      case MF_VEC3D: BEGIN MFV3D; break;

      /* SFNode and MFNode are special.  Here the lexer just returns */
      /* "marker tokens" so the parser knows what type of field is */
      /* being parsed; unlike the other fields, parsing of SFNode/MFNode */
      /* field happens in the parser. */
      case MF_NODE: expectToken = 0; return MF_NODE;
      case SF_NODE: expectToken = 0; return SF_NODE;
        
      default: yyerror("ACK: Bad expectToken"); break;
      }
   }
%}

   /* This is more complicated than they really need to be because */
   /* I was ambitious and made the whitespace-matching rule aggressive */
<INITIAL>"#VRML V2.0 utf8".*\n{wsnnl}* { BEGIN NODE; currentLineNumber = 2; }
<INITIAL>"#X3D V3."[0-9]*" utf8".*\n{wsnnl}* { x3d = true; BEGIN NODE; currentLineNumber = 2; return X3D;}

   /* The lexer is in the NODE state when parsing nodes, either */
   /* top-level nodes in the .wrl file, in a prototype implementation, */
   /* or when parsing the contents of SFNode or MFNode fields. */
<NODE>PROTO                     { return PROTO; }
<NODE>EXTERNPROTO               { return EXTERNPROTO; }
<NODE>DEF                       { return DEF; }
<NODE>USE                       { return USE; }
<NODE>TO                        { return TO; }
<NODE>IS                        { return IS; }
<NODE>ROUTE                     { return ROUTE; }
<NODE>NULL                      { return SFN_NULL; }
<NODE>eventIn                   { return EVENTIN; }
<NODE>eventOut                  { return EVENTOUT; }
<NODE>field                     { return FIELD; }
<NODE>exposedField              { return EXPOSEDFIELD; }
<NODE>inputOnly      { if (x3d) return EVENTIN; 
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }
<NODE>outputOnly     { if (x3d) return EVENTOUT; 
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }
<NODE>initializeOnly { if (x3d) return FIELD; 
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }
<NODE>inputOutput    { if (x3d) return EXPOSEDFIELD; 
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }

<NODE>PROFILE        { if (x3d) 
                       {
                          profile = true;
                          return PROFILE;
                       }
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }

<NODE>COMPONENT      { if (profile) 
                          return COMPONENT;
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }

<NODE>META           { if (profile) 
                          return META;
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }


<NODE>EXPORT         { if (x3d) 
                          return EXPORT;
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }

<NODE>IMPORT         { if (x3d) 
                          return IMPORT;
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }

<NODE>AS             { if (x3d) 
                          return AS;
                       yylval.string = strdup(yytext);
                       return IDENTIFIER;
                     }


   /* Legal identifiers. */
<NODE>{idStartChar}{idRestChar}*   { yylval.string = strdup(yytext);
                                     return IDENTIFIER; 
                                   }

   /* All fields may have an IS declaration: */
<SFB,SFC,SFCR,SFD,SFF,SFIMG,SFI,SFR,SFS,SFT,SFV2,SFV3,SFV2D,SFV3D>IS { 
                                                                BEGIN NODE;
                                                                expectToken = 0;
                                                                yyless(0);
                                                              }

<MFB,MFC,MFCR,MFD,MFF,MFI,MFR,MFS,MFT,MFV2,MFV3,MFV2D,MFV3D>IS { 
                                                                BEGIN NODE;
                                                                expectToken = 0;
                                                                yyless(0); /* put back the IS */
                                                              }

   /* All MF field types other than MFNode are completely parsed here */
   /* in the lexer, and one token is returned to the parser.  They all */
   /* share the same rules for open and closing brackets: */
<MFB,MFC,MFCR,MFD,MFF,MFI,MFR,MFS,MFT,MFV2,MFV3,MFV2D,MFV3D>\[ { 
                                               if (parsing_mf) yyerror("Double [");
                                               parsing_mf = 1;
                                               /* mfInts.erase(mfInts.begin(), mfInts.end());
                                                  mfFloats.erase(mfFloats.begin(), mfFloats.end());
                                                  mfStrs.erase(mfStrs.begin(), mfStrs.end());
                                                  mfBool.erase(mfBool.begin(), mfBool.end());
                                               */
                                               intSize=0;
                                               floatSize=0;
                                               strSize=0;
                                               boolSize=0;
                                             }

<MFB,MFC,MFCR,MFD,MFF,MFI,MFR,MFS,MFT,MFV2,MFV3,MFV2D,MFV3D>\]   { 
                                               if (! parsing_mf) yyerror("Unmatched ]");
                                               int fieldType = expectToken;
                                               switch (fieldType) {
                                                 case MF_BOOL:
                                                    if(boolSize>0)
                                                       yylval.field = new VrmlMFBool(boolSize, mfBools);
                                                    else
                                                       yylval.field = new VrmlMFBool();
                                                    break;
                                                 case MF_COLOR:
                                                    if(floatSize>0)
                                                       yylval.field = new VrmlMFColor(floatSize / 3, mfFloats);
                                                    else
                                                       yylval.field = new VrmlMFColor();
                                                    break;
                                                 case MF_COLOR_RGBA:
                                                    if(floatSize>0)
                                                       yylval.field = new VrmlMFColorRGBA(floatSize / 4, mfFloats);
                                                    else
                                                       yylval.field = new VrmlMFColorRGBA();
                                                    break;
                                                 case MF_FLOAT:
                                                    if(floatSize>0)
                                                       yylval.field = new VrmlMFFloat(floatSize, mfFloats);
                                                    else
                                                       yylval.field = new VrmlMFFloat();
                                                    break;
                                                 case MF_DOUBLE:
                                                    if(doubleSize>0)
                                                       yylval.field = new VrmlMFDouble(doubleSize, mfDoubles);
                                                    else
                                                       yylval.field = new VrmlMFDouble();
                                                    break;
                                                 case MF_INT32:
                                                    if ((expectCoordIndex || expectTexCoordIndex)&&
                                                        intSize > 0 && -1 != mfInts[intSize-1])
                                                       addInt(-1);
                                                    if(intSize)
                                                       yylval.field = new VrmlMFInt(intSize, mfInts);
                                                    else
                                                       yylval.field = new VrmlMFInt();
                                                    break;
                                                 case MF_ROTATION:
                                                    if(floatSize)
                                                       yylval.field = new VrmlMFRotation(floatSize / 4, mfFloats);
                                                    else
                                                       yylval.field = new VrmlMFRotation();
                                                    break;
                                                 case VRML_MF_STRING:
                                                    {
                                                    if(strSize)
                                                       yylval.field = new VrmlMFString(strSize, mfStrs);
                                                    else
                                                       yylval.field = new VrmlMFString();
                                                    unsigned int i;
                                                    for(i=0;i<strSize;i++)
                                                       delete[] mfStrs[i];
                                                    //mfStrs.erase(mfStrs.begin(), mfStrs.end());
                                                    strSize=0;
                                                    }
                                                    break;
                                                 case MF_TIME:
                                                    if(doubleSize>0)
                                                       yylval.field = new VrmlMFTime(doubleSize, mfDoubles);
                                                    else
                                                       yylval.field = new VrmlMFTime();
                                                    break;
                                                 case MF_VEC2F:
                                                    if(floatSize)
                                                       yylval.field = new VrmlMFVec2f(floatSize / 2, mfFloats);
                                                    else
                                                       yylval.field = new VrmlMFVec2f();
                                                    break;
                                                 case MF_VEC3F:
                                                    if(floatSize)
                                                       yylval.field = new VrmlMFVec3f(floatSize / 3, mfFloats);
                                                    else
                                                       yylval.field = new VrmlMFVec3f();
                                                    break;
                                                 case MF_VEC2D:
                                                    if(doubleSize)
                                                       yylval.field = new VrmlMFVec2d(doubleSize / 2, mfDoubles);
                                                    else
                                                       yylval.field = new VrmlMFVec2d();
                                                    break;
                                                 case MF_VEC3D:
                                                    if(doubleSize)
                                                       yylval.field = new VrmlMFVec3d(doubleSize / 3, mfDoubles);
                                                    else
                                                       yylval.field = new VrmlMFVec3d();
                                                    break;
                                               }
                                               BEGIN NODE;
                                               parsing_mf = 0;
                                               expectToken = 0;
    
                                               /*mfFloats.erase(mfFloats.begin(), mfFloats.end());
                                               mfInts.erase(mfInts.begin(), mfInts.end());
                                               mfBools.erase(mfBools.begin(), mfBools.end());*/
      
                                               intSize=0;
                                               floatSize=0;
                                               return fieldType;
                                             }


<SFB>{bool}  { BEGIN NODE; 
               expectToken = 0;
               int b;
               if (strcmp(yytext, "TRUE")==0)
                  b = 1;
               else if (strcmp(yytext, "FALSE")==0)
                  b = 0;
               else
               {
                  System::the->warn("\"");
                  System::the->warn(yytext);
                  System::the->warn("\" unsupported, TRUE or FALSE are only allowed for a SFBOOL field\n");
               }
               yylval.field = new VrmlSFBool(b!=0); 
               return SF_BOOL; 
             }

<MFB>{bool}   { 
               int b;
               if (strcmp(yytext, "TRUE")==0)
                  b = true;
               else if (strcmp(yytext, "FALSE")==0)
                  b = false;
               else
               {
                  System::the->warn("\"");
                  System::the->warn(yytext);
                  System::the->warn("\" unsupported, TRUE or FALSE are only allowed for a MFBOOL field\n");
               }
               if (parsing_mf)
               {
                  addBool(b!=0);
               }
               else 
               {
                  /* No open bracket means a single value: */
                  yylval.field = new VrmlMFBool(b!=0);
                  BEGIN NODE; 
                  expectToken = 0;
                  return MF_BOOL;
               }
             }

<SFI>{int}   { BEGIN NODE; 
               expectToken = 0;
               yylval.field = new VrmlSFInt(strtol(yytext,0,0));
               return SF_INT32;
             }

<MFI>{int}   { int i = strtol(yytext,0,0);
               if (parsing_mf)
               {
                  addInt(i);
               }
               else 
               {
                  /* No open bracket means a single value: */
                  yylval.field = new VrmlMFInt(i);
                  BEGIN NODE; 
                  expectToken = 0;
                  return MF_INT32;
               }
             }

   /* All the floating-point types are pretty similar: */
<SFF>{float}   { yylval.field = new VrmlSFFloat((float)atof(yytext));
                 BEGIN NODE; 
                 expectToken = 0;
                 return SF_FLOAT;
               }

<MFF>{float}   { float f = (float)atof(yytext);
                 if (parsing_mf)
                 {
                    addFloat(f);
                 }
                 else 
                 {    
                    /* No open bracket means a single value: */
                    yylval.field = new VrmlMFFloat(f);
                    BEGIN NODE;
                    expectToken = 0;
                    return MF_FLOAT;
                 }
               }

<SFD>{float}   { yylval.field = new VrmlSFDouble(atof(yytext));
                 BEGIN NODE; 
                 expectToken = 0;
                 return SF_DOUBLE;
               }

<MFD>{float}   { double f = atof(yytext);
                 if (parsing_mf)
                 {
                    addDouble(f);
                 }
                 else 
                 {    
                    /* No open bracket means a single value: */
                    yylval.field = new VrmlMFDouble(f);
                    BEGIN NODE;
                    expectToken = 0;
                    return MF_DOUBLE;
                 }
               }

<SFV2>{float}{ws}{float} { float x = 0.0, y = 0.0;
                           int n = 0;
                           char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                           int retval;
                           retval=sscanf(s,"%g%n", &x, &n);
                           if (retval<1)
                               std::cerr<<"lexer.ll SFV2: sscanf failed"<<std::endl;
                           s = skip_ws(s+n);
                           retval=sscanf(s,"%g", &y);
                           if (retval!=1)
                              std::cerr<<"lexer.ll SFV2: sscanf failed"<<std::endl;
#else
                           x = (float)atof(s);
                           while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                              s++;
                           s = skip_ws(s);
                           y = (float)atof(s);
#endif

                           yylval.field = new VrmlSFVec2f(x,y);
                           BEGIN NODE; 
                           expectToken = 0;
                           return SF_VEC2F; 
                         }


<SFV2D>{float}{ws}{float} { double x = 0.0, y = 0.0;
                           int n = 0;
                           char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                           int retval;
                           retval=sscanf(s,"%lg%n", &x, &n);
                           if (retval<1)
                               std::cerr<<"lexer.ll SFV2D: sscanf failed"<<std::endl;
                           s = skip_ws(s+n);
                           retval=sscanf(s,"%lg", &y);
                           if (retval!=1)
                              std::cerr<<"lexer.ll SFV2D: sscanf failed"<<std::endl;
#else
                           x = atof(s);
                           while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                              s++;
                           s = skip_ws(s);
                           y = atof(s);
#endif

                           yylval.field = new VrmlSFVec2d(x,y);
                           BEGIN NODE; 
                           expectToken = 0;
                           return SF_VEC2D; 
                         }

<MFV2>{float}{ws}{float}   { float x = 0.0, y = 0.0;
                             int n = 0;
                             char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                             int retval;

                             retval=sscanf(s,"%g%n", &x, &n);
                             if (retval<1)
                                std::cerr<<"lexer.ll MFV2: sscanf failed"<<std::endl;
                             s = skip_ws(s+n);
                             retval=sscanf(s,"%g", &y);
                             if (retval!=1)
                                std::cerr<<"lexer.ll MFV2: sscanf failed"<<std::endl;
#else
                             x = (float)atof(s);
                             while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                s++;
                             s = skip_ws(s);
                             y = (float)atof(s);
#endif
                             if (parsing_mf) 
                             {
                                addFloat(x);
                                addFloat(y);
                             } 
                             else 
                             {
                                yylval.field = new VrmlMFVec2f(x,y);
                                BEGIN NODE; 
                                expectToken = 0;
                                return MF_VEC2F;
                             }
                           }

<MFV2D>{float}{ws}{float}  { double x = 0.0, y = 0.0;
                             int n = 0;
                             char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                             int retval;

                             retval=sscanf(s,"%lg%n", &x, &n);
                             if (retval<1)
                                std::cerr<<"lexer.ll MFV2D: sscanf failed"<<std::endl;
                             s = skip_ws(s+n);
                             retval=sscanf(s,"%lg", &y);
                             if (retval!=1)
                                std::cerr<<"lexer.ll MFV2D: sscanf failed"<<std::endl;
#else
                             x = atof(s);
                             while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                s++;
                             s = skip_ws(s);
                             y = atof(s);
#endif
                             if (parsing_mf) 
                             {
                                addDouble(x);
                                addDouble(y);
                             } 
                             else 
                             {
                                yylval.field = new VrmlMFVec2d(x,y);
                                BEGIN NODE; 
                                expectToken = 0;
                                return MF_VEC2D;
                             }
                           }

<SFV3>({float}{ws}){2}{float}   { float x = 0.0, y = 0.0, z = 0.0;
                                  int n = 0;
                                  char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                  int retval;

                                  retval=sscanf(s,"%g%n", &x, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFV3: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g%n", &y, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFV3: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g", &z);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll SFV3: sscanf failed"<<std::endl;
#else
                                  x = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  y = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  z = (float)atof(s);
#endif

                                  yylval.field = new VrmlSFVec3f(x,y,z);

                                  BEGIN NODE; 
                                  expectToken = 0; 
                                  return SF_VEC3F;
                                }

<SFV3D>({float}{ws}){2}{float}  { double x = 0.0, y = 0.0, z = 0.0;
                                  int n = 0;
                                  char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                  int retval;

                                  retval=sscanf(s,"%lg%n", &x, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFV3D: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%lg%n", &y, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFV3D: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%lg", &z);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll SFV3D: sscanf failed"<<std::endl;
#else
                                  x = atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  y = atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  z = atof(s);
#endif

                                  yylval.field = new VrmlSFVec3d(x,y,z);

                                  BEGIN NODE; 
                                  expectToken = 0; 
                                  return SF_VEC3D;
                                }

<MFV3>({float}{ws}){2}{float}   { float x = 0.0, y = 0.0, z = 0.0;
                                  int n = 0;
                                  char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                  int retval;

                                  retval=sscanf(s,"%g%n", &x, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll MFV3: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g%n", &y, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll MFV3: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g", &z);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll MFV3: sscanf failed"<<std::endl;
#else
                                  x = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  y = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  z = (float)atof(s);
#endif

                                  if (parsing_mf) 
                                  {
                                     addFloat(x);
                                     addFloat(y);
                                     addFloat(z);
                                  } 
                                  else 
                                  {
                                     yylval.field = new VrmlMFVec3f(x,y,z);
                                     BEGIN NODE; 
                                     expectToken = 0;
                                     return MF_VEC3F;
                                  }
                                }

<MFV3D>({float}{ws}){2}{float}  { double x = 0.0, y = 0.0, z = 0.0;
                                  int n = 0;
                                  char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                  int retval;

                                  retval=sscanf(s,"%lg%n", &x, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll MFV3D: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%lg%n", &y, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll MFV3D: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%lg", &z);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll MFV3D: sscanf failed"<<std::endl;
#else
                                  x = atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  y = atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  z = atof(s);
#endif

                                  if (parsing_mf) 
                                  {
                                     addDouble(x);
                                     addDouble(y);
                                     addDouble(z);
                                  } 
                                  else 
                                  {
                                     yylval.field = new VrmlMFVec3d(x,y,z);
                                     BEGIN NODE; 
                                     expectToken = 0;
                                     return MF_VEC3D;
                                  }
                                }

<SFR>({float}{ws}){3}{float}   { float x = 0.0, y = 0.0, z = 0.0, r = 0.0;
                                 int n = 0;
                                 char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                 int retval;
                                 retval=sscanf(s,"%g%n", &x, &n);
                                 if (retval<1)
                                    std::cerr<<"lexer.ll SFR: sscanf failed"<<std::endl;
                                 s = skip_ws(s+n);
                                 retval=sscanf(s,"%g%n", &y, &n);
                                 if (retval<1)
                                    std::cerr<<"lexer.ll SFR: sscanf failed"<<std::endl;
                                 s = skip_ws(s+n);
                                 retval=sscanf(s,"%g%n", &z, &n);
                                 if (retval<1)
                                    std::cerr<<"lexer.ll SFR: sscanf failed"<<std::endl;
                                 s = skip_ws(s+n);
                                 retval=sscanf(s,"%g", &r);
                                 if (retval!=1)
                                    std::cerr<<"lexer.ll SFR: sscanf failed"<<std::endl;
#else
                                 x = (float)atof(s);
                                 while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                    s++;
                                 s = skip_ws(s);
                                 y = (float)atof(s);
                                 while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                    s++;
                                 s = skip_ws(s);
                                 z = (float)atof(s);
                                 while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                    s++;
                                 s = skip_ws(s);
                                 r = (float)atof(s);
#endif
                                 yylval.field = new VrmlSFRotation(x,y,z,r);
                                 BEGIN NODE; 
                                 expectToken = 0; 
                                 return SF_ROTATION;
                               }

<MFR>({float}{ws}){3}{float}   { float x = 0.0, y = 0.0, z = 0.0, r = 0.0;
                                 int n = 0;
                                 char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                 int retval;
                                 retval=sscanf(s,"%g%n", &x, &n);
                                 if (retval<1)
                                    std::cerr<<"lexer.ll MFR: sscanf failed"<<std::endl;
                                 s = skip_ws(s+n);
                                 retval=sscanf(s,"%g%n", &y, &n);
                                 if (retval<1)
                                    std::cerr<<"lexer.ll MFR: sscanf failed"<<std::endl;
                                 s = skip_ws(s+n);
                                 retval=sscanf(s,"%g%n", &z, &n);
                                 if (retval<1)
                                    std::cerr<<"lexer.ll MFR: sscanf failed"<<std::endl;
                                 s = skip_ws(s+n);
                                 retval=sscanf(s,"%g", &r);
                                 if (retval!=1)
                                    std::cerr<<"lexer.ll MFR: sscanf failed"<<std::endl;
#else
                                  
                                 x = (float)atof(s);
                                 while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                    s++;
                                 s = skip_ws(s);
                                 y = (float)atof(s);
                                 while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                    s++;
                                 s = skip_ws(s);
                                 z = (float)atof(s);
                                 while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                    s++;
                                 s = skip_ws(s);
                                 r = (float)atof(s);
#endif

                                 if (parsing_mf) 
                                 {
                                    addFloat(x);
                                    addFloat(y);
                                    addFloat(z);
                                    addFloat(r);
                                 } 
                                 else 
                                 {
                                    yylval.field = new VrmlMFRotation(x,y,z,r);
                                    BEGIN NODE; 
                                    expectToken = 0;
                                    return MF_ROTATION;
                                  }
                               }

<SFC>({float}{ws}){2}{float}    { float r = 0.0, g = 0.0, b = 0.0;
                                  int n = 0;
                                  char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                  int retval;
                                  retval=sscanf(s,"%g%n", &r, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFC: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g%n", &g, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFC: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g", &b);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll SFC: sscanf failed"<<std::endl;
#else
                                  r = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  g = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  b = (float)atof(s);
#endif

                                  yylval.field = new VrmlSFColor(r,g,b);
                                  BEGIN NODE; expectToken = 0; 
                                  return SF_COLOR;
                                }

<SFCR>({float}{ws}){3}{float}   { float r = 0.0, g = 0.0, b = 0.0, a = 1.0;
                                  int n = 0;
                                  char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                  int retval;
                                  retval=sscanf(s,"%g%n", &r, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFCR: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g%n", &g, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFCR: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g", &b);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll SFCR: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g", &a);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll SFCR: sscanf failed"<<std::endl;
#else
                                  r = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  g = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  b = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  a = (float)atof(s);
#endif

                                  yylval.field = new VrmlSFColorRGBA(r,g,b,a);
                                  BEGIN NODE; expectToken = 0; 
                                  return SF_COLOR_RGBA;
                                }

<MFC>({float}{ws}){2}{float}   { float r = 0.0, g = 0.0, b = 0.0;
                                 int n = 0;
                                 char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                 int retval;
                                 retval=sscanf(s,"%g%n", &r, &n);
                                 if (retval<1)
                                    std::cerr<<"lexer.ll MFC: sscanf failed"<<std::endl;
                                 s = skip_ws(s+n);
                                 retval=sscanf(s,"%g%n", &g, &n);
                                 if (retval<1)
                                    std::cerr<<"lexer.ll MFC: sscanf failed"<<std::endl;
                                 s = skip_ws(s+n);
                                 retval=sscanf(s,"%g", &b);
                                 if (retval!=1)
                                    std::cerr<<"lexer.ll MFC: sscanf failed"<<std::endl;
#else
                                 r = (float)atof(s);
                                 while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                    s++;
                                 s = skip_ws(s);
                                 g = (float)atof(s);
                                 while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                    s++;
                                 s = skip_ws(s);
                                 b = (float)atof(s);
#endif
                                 if (parsing_mf) {
                                    addFloat(r);
                                    addFloat(g);
                                    addFloat(b);
                                 } else {
                                   yylval.field = new VrmlMFColor(r,g,b);
                                   BEGIN NODE;
                                   expectToken = 0;
                                   return MF_COLOR;
                                 }
                               }

<MFCR>({float}{ws}){3}{float}   { float r = 0.0, g = 0.0, b = 0.0, a = 1.0;
                                  int n = 0;
                                  char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                  int retval;
                                  retval=sscanf(s,"%g%n", &r, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll MFCR: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g%n", &g, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll MFCR: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g", &b);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll MFCR: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%g", &a);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll MFCR: sscanf failed"<<std::endl;
 #else
                                  r = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  g = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  b = (float)atof(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  a = (float)atof(s);
 #endif
                                  if (parsing_mf) {
                                     addFloat(r);
                                     addFloat(g);
                                     addFloat(b);
                                     addFloat(a);
                                  } else {
                                    yylval.field = new VrmlMFColorRGBA(r,g,b,a);
                                    BEGIN NODE;
                                    expectToken = 0;
                                    return MF_COLOR_RGBA;
                                  }
                                }

<SFT>{float}                    { yylval.field = new VrmlSFTime(atof(yytext));
                                  BEGIN NODE; 
                                  expectToken = 0;
                                  return SF_TIME;
                                }

<MFT>{float}   { double f = atof(yytext);
                 if (parsing_mf)
                 {
                    addDouble(f);
                 }
                 else 
                 {    
                    /* No open bracket means a single value: */
                    yylval.field = new VrmlMFTime(f);
                    BEGIN NODE;
                    expectToken = 0;
                    return MF_TIME;
                 }
               }


   /* SFString/MFString */
<SFS>\"                         { BEGIN IN_SFS; initString(); }
<MFS>\"                         { BEGIN IN_MFS; initString(); }

   /* Anything besides open-quote (or whitespace) is an error: */
<SFS>[^ \"\t\r\,\n]+            { yyerror("SFString missing open-quote");
                                  yylval.field = 0;
                                  BEGIN NODE; 
                                  expectToken = 0;
                                  return SF_STRING;
                                }

   /* Expect open-quote, open-bracket, or whitespace: */
<MFS>[^ \[\]\"\t\r\,\n]+        { yyerror("MFString missing open-quote");
                                  yylval.field = 0;
                                  BEGIN NODE; expectToken = 0;
                                  return VRML_MF_STRING;
                                }

   /* Backslashed-quotes and backslashed-backslashes are OK: */
<IN_SFS,IN_MFS>\\\"             { checkStringSize(1);
                                  strcpy(sfStringChars+sfStringN++,"\""); 
                                }
<IN_SFS,IN_MFS>\\\\             { checkStringSize(1);
                                  strcpy(sfStringChars+sfStringN++,"\\"); 
                                }
<IN_SFS,IN_MFS>\\n              { checkStringSize(2);
                                  strcpy(sfStringChars+sfStringN, "\\n");
                                  sfStringN += 2;
                                }
<IN_SFS,IN_MFS>\\r              { checkStringSize(2);
                                  strcpy(sfStringChars+sfStringN, "\\r");
                                  sfStringN += 2;
                                }
<IN_SFS,IN_MFS>\\t              { checkStringSize(2);
                                  strcpy(sfStringChars+sfStringN, "\\t");
                                  sfStringN += 2;
                                }
<IN_SFS,IN_MFS>\\'              { checkStringSize(2);
                                  strcpy(sfStringChars+sfStringN, "\\'");
                                  sfStringN += 2;
                                }
   /* XXX: add more escaped characters? */

   /* Newline characters are OK: */
<IN_SFS,IN_MFS>\n               { checkStringSize(1);
                                  strcpy(sfStringChars+sfStringN++,"\n");
                                  ++currentLineNumber;
                                }

   /* Eat anything besides quotes, backslashed (escaped) chars and newlines. */
<IN_SFS,IN_MFS>[^\"\n\\]+       { checkStringSize(yyleng);
                                  strcpy(sfStringChars+sfStringN,yytext);
                                  sfStringN += yyleng;
                                }

   /* Quote ends the string: */
<IN_SFS>\"                      { yylval.field = new VrmlSFString(sfStringChars);
                                  BEGIN NODE; expectToken = 0;
                                  return SF_STRING;
                                }

<IN_MFS>\"                      { if (parsing_mf) 
                                  {
                                    char *s = new char[strlen(sfStringChars)+1];
                                    strcpy(s,sfStringChars);
                                    addStr(s);
                                    BEGIN MFS;
                                  } 
                                  else 
                                  {
                                    yylval.field = new VrmlMFString(sfStringChars);
                                    BEGIN NODE; expectToken = 0;
                                    return VRML_MF_STRING;
                                  }
                                }

   /* SFImage: width height numComponents then width*height integers: */
<SFIMG>{int}{ws}{int}{ws}{int}  { int w = 0, h = 0, nc = 0, n = 0;
                                  unsigned char *pixels = 0;
                                  char *s = &yytext[0];
#ifdef DEBUG_SSCANF
                                  size_t retval;

                                  retval=sscanf(s,"%d%n", &w, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFIMG: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%d%n", &h, &n);
                                  if (retval<1)
                                     std::cerr<<"lexer.ll SFIMG: sscanf failed"<<std::endl;
                                  s = skip_ws(s+n);
                                  retval=sscanf(s,"%d", &nc);
                                  if (retval!=1)
                                     std::cerr<<"lexer.ll SFIMG: sscanf failed"<<std::endl;
#else
                                  w = atoi(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  h = atoi(s);
                                  while(*s && *s !=' ' && *s !='\t'&& *s !='\r'&& *s !='\n')
                                     s++;
                                  s = skip_ws(s);
                                  nc = atoi(s);
#endif

                                  sfImageIntsExpected = w*h;
                                  sfImageIntsParsed = 0;
                                  sfImageNC = nc;

                                  if (sfImageIntsExpected > 0)
                                    pixels = new unsigned char[nc*w*h];

                                  sfImagePixels = pixels;
                                  memset(pixels,0,nc*w*h);

                                  yylval.field = new VrmlSFImage(w,h,nc,pixels);

                                  if (sfImageIntsExpected > 0) {
                                    BEGIN IN_SFIMG;
                                  } else {
                                    BEGIN NODE; expectToken = 0;
                                    return SF_IMAGE;
                                  }
                                }

<IN_SFIMG>{int}         { unsigned long pixval = strtol(yytext, 0, 0);

                          int i, j = sfImageNC * sfImageIntsParsed++;
                          for (i=0; i<sfImageNC; ++i)
                             sfImagePixels[i+j] = (char)((sfImageMask[i] & pixval) >> (8*i));
                          if (sfImageIntsParsed == sfImageIntsExpected) 
                          {
                             BEGIN NODE; expectToken = 0;
                             return SF_IMAGE;
                          }
                        }


   /* Whitespace rules apply to all start states except inside strings: */

<INITIAL,NODE,SFB,SFC,SFCR,SFD,SFF,SFIMG,SFI,SFR,SFS,SFT,SFV2,SFV3,SFV2D,SFV3D,MFB,MFC,MFCR,MFD,MFF,MFI,MFR,MFS,MFT,MFV2,MFV3,MFV2D,MFV3D,IN_SFIMG>{
  {wsnnl}+                ;

        /* This is also whitespace, but we'll keep track of line number */
        /* to report in errors: */
  {wsnnl}*\n{wsnnl}*        { ++currentLineNumber; }
}

   /* This catch-all rule catches anything not covered by any of */
   /* the above: */
<*>.                        { return yytext[0]; }

%%

/* Set up to read from string s. Reading from strings skips the header */

void yystring(char *s) 
{
   yyin = 0;
#if HAVE_LIBPNG || HAVE_ZLIB
   yygz = 0;
#endif
   yyinStr = s;
   yyinFunc = 0;
   BEGIN NODE;
   expectToken = 0;
   parsing_mf = 0;
   currentLineNumber = 1;
}

/* Set up to read from function f. */

void yyfunction( int (*f)(char *, int) )
{
   yyin = 0;
#if HAVE_LIBPNG || HAVE_ZLIB
   yygz = 0;
#endif
   yyinStr = 0;
   yyinFunc = f;
   BEGIN INITIAL;
   expectToken = 0;
   parsing_mf = 0;
   currentLineNumber = 1;
}

int yywrap(void) 
{
   x3d = 0;
   profile = 0;
   yyinStr = 0;
   yyinFunc = 0;
   BEGIN INITIAL;
   return 1; 
}

