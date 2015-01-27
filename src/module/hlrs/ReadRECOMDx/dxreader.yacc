
%name Parser


%header{

// #define YYSTYPE double

#include <covise/covise.h>


#if defined(__hpux) || defined(__sgi)
#include <alloca.h>
#endif

/* #include <FlexLexer.h> */
#include "attribute.h"
#include "scanner.h"



#include <api/coModule.h>
using namespace covise;

#include "parser.h"
/* The action class is the interface between parser and the rest of the compiler
   depending on its implementation
*/

#include "action.h"

%}
%union
{
        attribute *attr;
};


%define MEMBERS \
ifstream input;\
Scanner *lexer;\
char *currFileName_;\
char *currDirName_;\
bool isOpen_;\
bool isCorrect_;\
actionClass *action;\
virtual ~Parser(){input.close();delete currFileName_;delete currDirName_;}\
Parser(actionClass *act,const char *fileName)\
{ \
       currFileName_=new char[1+strlen(fileName)];\
       strcpy(currFileName_, fileName);\
       currDirName_= new char[1+strlen(fileName)];\
       strcpy(currDirName_, fileName);\
       myDirName(currDirName_); \
       action=act;\
       input.open(fileName, ios::in);\
       char c;\
       input.get(c);\
       input.seekg(0);\
       isOpen_= input.good();\
       isCorrect_=true;\
       lexer = new Scanner(&input);\
}\
bool isOpen() { return isOpen_;}\
bool isCorrect() { return isCorrect_;}\
void myDirName(char *filename) \
{\
        int len=strlen(filename);\
        int i;\
        for(i=len-1; i>=0; i--) {\
          if(filename[i]=='/') {\
            filename[i]='\0';\
            return; \
          }       \
        }\
}\
void setCurrFileName(const char *name){ \
        delete [] currFileName_;\
        currFileName_=new char[1+strlen(name)];\
         strcpy(currFileName_, name);}\
char *getCurrFileName(){return currFileName_;}\
void setCurrDirName(const char *name){ \
        delete [] currDirName_;\
        currDirName_=new char[1+strlen(name)];\
         strcpy(currDirName_, name);}\
char *getCurrDirName(){return currDirName_;}\



%define LEX_BODY {return lexer->yylex();}
%define ERROR_BODY { char comsg[4096]; sprintf(comsg, "Syntax error in line %d; %s not recognized", lexer->getLineNo(), lexer->YYText()); Covise::sendError(comsg); isCorrect_=false;}







  /* BISON Declarations */
%token 
<attr> INTEGERVALUE 
<attr> FLOATVALUE 
<attr> STRINGVALUE 


%type <attr> stringvalue
%type <attr> number
/* %type <attr> dataformat*/


%token
OBJECT
CLASS
ARRAY
FIELD
MULTIGRID
GROUP
FILENAME
MEMBER
TYPE
BINARY
ASCII
COMMA
SHAPE
DOUBLE
FLOAT
INT
UINT
SHORT
USHORT
BYTE
UBYTE
RANK
ITEMS
DATA
FOLLOWS
FILE
ATTRIBUTEREF
ATTRIBUTEELTYPE
ATTRIBUTENAME
ATTRIBUTEDEP
CPOSITIONS
CCONNECTIONS
CDATA
VALUE
STRING
INTEGERVALUE
FLOATVALUE
STRINGVALUE
LSB
MSB
END
%%

input: objects END { YYACCEPT;}
        | objects { YYACCEPT;}
;

objects:
        | objects object
        | objects error
;

object: field
        | array
        | multigrid
        | group
;

field: OBJECT number CLASS FIELD fieldspec         {
                                                      action->setCurrName($2->getInt());
                                                      action->setCurrObjectClass(Parser::FIELD);
                                                      action->objects_[action->getCurrName()]=action->getCurrObject();
                                                      action->newCurrent();
                                                   }
        | OBJECT stringvalue CLASS FIELD fieldspec {
                                                      action->setCurrName($2->getString());
                                                      action->setCurrObjectClass(Parser::FIELD);
                                                      
                                                      action->objects_[action->getCurrName()]=action->getCurrObject();
                                                      action->newCurrent();
                                                   }
;
fieldspec: 
        | fieldspec component 
        | fieldspec attributes
;

multigrid: OBJECT stringvalue CLASS MULTIGRID multigridspec {
   action->setCurrName($2->getString());
   action->setCurrObjectClass(Parser::MULTIGRID);
   action->objects_[action->getCurrName()]=action->getCurrObject();
   action->newCurrent();
   }
| OBJECT number CLASS MULTIGRID multigridspec {
   action->setCurrName($2->getInt());
   action->setCurrObjectClass(Parser::MULTIGRID);
   action->objects_[action->getCurrName()]=action->getCurrObject();
   action->newCurrent();
   }
;
multigridspec: 
        | multigridspec MEMBER STRING stringvalue value stringvalue { action->addMember($4->getString(), $6->getString());}
        | multigridspec MEMBER stringvalue value stringvalue{ action->addMember($3->getString(), $5->getString());}
        | multigridspec MEMBER number value stringvalue{ action->addMember($3->getString(), $5->getString());}
        | multigridspec MEMBER number value number{ action->addMember($3->getString(), $5->getString());}
;


group: OBJECT stringvalue CLASS GROUP groupspec {
   action->setCurrName($2->getString());
   action->setCurrObjectClass(Parser::GROUP);
   action->objects_[action->getCurrName()]=action->getCurrObject();
   action->newCurrent();
   }
| OBJECT number CLASS GROUP groupspec {
   action->setCurrName($2->getInt());
   action->setCurrObjectClass(Parser::GROUP);
   action->objects_[action->getCurrName()]=action->getCurrObject();
   action->newCurrent();
   }
;
groupspec: 
        | groupspec MEMBER STRING stringvalue value stringvalue { action->addMember($4->getString(), $6->getString());}
        | groupspec MEMBER stringvalue value stringvalue{ action->addMember($3->getString(), $5->getString());}
        | groupspec MEMBER number value stringvalue{ action->addMember($3->getString(), $5->getString());}
;

component:   CPOSITIONS value number       { action->setCurrPositions($3->getInt());}
        |  CDATA value number              { action->setCurrData($3->getInt());}
        |  CCONNECTIONS value number       { action->setCurrConnections($3->getInt());}
        |  CPOSITIONS value stringvalue    { action->setCurrPositions($3->getString());}
        |  CDATA value stringvalue         { action->setCurrData($3->getString());}
        |  CCONNECTIONS value stringvalue  { action->setCurrConnections($3->getString());}
;
value:
        | VALUE
;


attributes: 
        | attributes ATTRIBUTEREF STRING stringvalue    { action->setCurrRef($4->getString());}
        | attributes ATTRIBUTEELTYPE STRING stringvalue { action->setCurrElementType($4->getString());}
        | attributes ATTRIBUTEDEP STRING stringvalue    { action->setCurrAttributeDep($4->getString());}
        | attributes ATTRIBUTENAME STRING stringvalue   { action->setCurrAttributeName($4->getString());}
;

array: OBJECT number CLASS ARRAY arraymodifiers datasource attributes {
                                                                        action->setCurrName($2->getInt());
                                                                        action->setCurrObjectClass(Parser::ARRAY);
                                                                        
                                                                        //store the recognized array object
                                                                        //in action->arrays
                                                                        action->arrays_[action->getCurrName()]=action->getCurrObject();
                                                                        //and create a new current object to be recognized
                                                                        action->newCurrent();
                                                                      }
        | OBJECT stringvalue CLASS ARRAY arraymodifiers datasource attributes {
                                                                        action->setCurrName($2->getString());
                                                                        action->setCurrObjectClass(Parser::ARRAY);
                                                                        
                                                                        //store the recognized array object
                                                                        //in action->arrays
                                                                        action->arrays_[action->getCurrName()]=action->getCurrObject();
                                                                        //and create a new current object to be recognized
                                                                        action->newCurrent();
                                                                    }
;
arraymodifiers:                                 
        | arraymodifiers RANK number            {action->setCurrRank($3->getInt());}
        | arraymodifiers ITEMS number           {action->setCurrItems($3->getInt());}
        | arraymodifiers TYPE type              
        | arraymodifiers SHAPE number           {action->setCurrShape($3->getInt());}
;

type: DOUBLE             {action->setCurrType(Parser::DOUBLE);}
        | FLOAT          {action->setCurrType(Parser::FLOAT);}
        | INT            {action->setCurrType(Parser::INT);}
        | UINT           {action->setCurrType(Parser::UINT);}
        | SHORT          {action->setCurrType(Parser::SHORT);}
        | USHORT         {action->setCurrType(Parser::USHORT);}
        | BYTE           {action->setCurrType(Parser::BYTE);}
        | UBYTE          {action->setCurrType(Parser::UBYTE);}
;

datasource: DATA FOLLOWS                                {action->setCurrFollows(true);}
        | dataformat DATA FILE stringvalue COMMA number {
                                                                action->setCurrFollows(false);
                                                                action->setCurrFileName(getCurrDirName(),$4->getString());
                                                                action->setCurrDataOffset($6->getInt());
                                                        }
        | dataformat DATA number                        {
                                                                action->setCurrFollows(false);
                                                                action->setCurrFileName(NULL);
                                                                action->setCurrDataOffset($3->getInt());
                                                        }
        | DATA FILE stringvalue COMMA number            {
                                                                action->setCurrFollows(false);
                                                                action->setCurrFileName(getCurrDirName(),$3->getString());
                                                                action->setCurrDataOffset($5->getInt());
                                                        }
        | DATA number                                   {
                                                                action->setCurrFollows(false);
                                                                action->setCurrFileName(NULL);
                                                                action->setCurrDataOffset($2->getInt());
                                                        }
;



dataformat:
        | BINARY                {action->setCurrDataFormat(Parser::BINARY)}
        | MSB BINARY            {action->setCurrByteOrder(Parser::MSB);action->setCurrDataFormat(Parser::BINARY);}
        | LSB BINARY            {action->setCurrByteOrder(Parser::LSB);action->setCurrDataFormat(Parser::BINARY);}
        | ASCII                 {action->setCurrDataFormat(Parser::ASCII);}
;


number: INTEGERVALUE { $$ = new attribute(atoi(lexer->YYText())); } 
        |FLOATVALUE  { $$ = new attribute(atof(lexer->YYText())); } 
;

stringvalue:  STRINGVALUE { attribute *temp=new attribute(lexer->YYText());$$ = temp; }
;



%%



