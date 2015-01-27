%{
#include "parser.h"
#include <covise/covise.h>
#define printf(a)
%}
%option yylineno
%option noyywrap
%x STRINGMODE
%x FILEMODE
%x ENDMODE
DIGIT [0-9]

%%
object                            printf("returning object\n");return Parser::OBJECT;
class                             printf("return Parser::CLASS;\n");return Parser::CLASS;
array                             printf("return Parser::ARRAY;\n");return Parser::ARRAY;
field                             printf("return Parser::FIELD;\n");return Parser::FIELD;
multigrid                         printf("return Parser::MULTIGRID;\n");return Parser::MULTIGRID;
group                         printf("return Parser::GROUP;\n");return Parser::GROUP;
type                              printf("return Parser::TYPE;\n");return Parser::TYPE;
double                            printf("return Parser::DOUBLE;\n");return Parser::DOUBLE;
float                             printf("return Parser::FLOAT;\n");return Parser::FLOAT;
int                               printf("return Parser::INT;\n");return Parser::INT;
uint                              printf("return Parser::UINT;\n");return Parser::UINT;
short                             printf("return Parser::SHORT;\n");return Parser::SHORT;
ushort                            printf("return Parser::USHORT;\n");return Parser::USHORT;
byte                              printf("return Parser::BYTE;\n");return Parser::BYTE;
ubyte                             printf("return Parser::UBYTE;\n");return Parser::UBYTE;
shape                             printf("returning shape\n");return Parser::SHAPE;
rank                              printf("return Parser::RANK;\n");return Parser::RANK;
items                             printf("return Parser::ITEMS;\n");return Parser::ITEMS;
data                              printf("return Parser::DATA;\n");return Parser::DATA;
follows                           printf("return Parser::FOLLOWS;\n");return Parser::FOLLOWS;
end                           printf("return Parser::END;\n");BEGIN(ENDMODE);return Parser::END;
End                           printf("return Parser::END;\n");BEGIN(ENDMODE);return Parser::END;
file                              printf("starting filemode \n");BEGIN(FILEMODE);return Parser::FILE;
<FILEMODE>[^ ,\t\n]+              BEGIN(INITIAL);return Parser::STRINGVALUE;
lsb                            printf("return Parser::LSB;\n");return Parser::LSB;
msb                            printf("return Parser::MSB;\n");return Parser::MSB;
binary                            printf("return Parser::BINARY;\n");return Parser::BINARY;
ieee                            printf("return Parser::BINARY;\n");return Parser::BINARY;
text                            printf("return Parser::ASCII;\n");return Parser::ASCII;
ascii                            printf("return Parser::ASCII;\n");return Parser::ASCII;
,                                 printf("return Parser::COMMA;\n");return Parser::COMMA;
member                            printf("return Parser::MEMBER;\n");return Parser::MEMBER;
attribute[ \t]+\"ref\"                         printf("return Parser::ATTRIBUTEREF;\n");return Parser::ATTRIBUTEREF;
attribute[ \t]+\"element\ type\"                     printf("returninf elmentType\n");return Parser::ATTRIBUTEELTYPE;
attribute[ \t]+\"name\"                         printf("return Parser::ATTRIBUTENAME;\n");return Parser::ATTRIBUTENAME;
attribute[ \t]+\"dep\"                     printf("returninf PArser::ATTRIBUTEDEP\n");return Parser::ATTRIBUTEDEP;
component[ \t]+\"positions\"                     printf("return Parser::CPOSITIONS;\n");return Parser::CPOSITIONS;
component[ \t]+\"connections\"                     printf("return Parser::CCONNECTIONS;\n");return Parser::CCONNECTIONS;
component[ \t]+\"data\"                     printf("return Parser::CDATA;\n");return Parser::CDATA;
value                             printf("return Parser::VALUE;\n");return Parser::VALUE;
string                            printf("returning string\n");return Parser::STRING;
\"                                BEGIN(STRINGMODE);
#.*\n                
-?{DIGIT}+                        printf("returning INTEGERVALUE1\n");return Parser::INTEGERVALUE;
-?{DIGIT}+"."{DIGIT}*             printf("returning FLOATVALUE1\n");return Parser::FLOATVALUE;
[^ _:\t\n"]+                       printf("return Parser::STRINGVALUE1;\n");return Parser::STRINGVALUE;
[ \t\n]+       
\n

<STRINGMODE>[^"]*           printf("return Parser::STRINGVALUE3;\n");return Parser::STRINGVALUE;
<STRINGMODE>\"            BEGIN(INITIAL);
<ENDMODE>.*
%%


int Scanner::getLineNo() { return yylineno; }
void Scanner::setHostName(const char */*name*/){}

