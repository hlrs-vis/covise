%{
#define input() getc(pfile)
#define unput(C) ungetc((C),pfile)
double atof();
extern FILE *pfile;
%}
%a 10000
%p 28000
%o 5000
D		[0-9]
E		[DEde][-+]?{D}+
S		[-+]?
%%

{S}{D}+			{/* integer value */
			 yylval.integer_value = atoi(yytext);
			 return(INTEGER_VALUE);
			}

{S}{D}+"."{D}*({E})?	|
{S}{D}*"."{D}+({E})?	|
{S}{D}+{E}		{/* real value */
			 yylval.real_value = (float) atof(yytext);
			 return(REAL_VALUE);
			}

"{"			{return(BEG);   /* left bracket */} 
"}"			{return(END);   /* right bracket */}
","                     {return(SEP);   /* separator */}

\"[^"]*			{/* string (delimeted by ") */
			 if (yytext[yyleng-1] == '\\')
			    yymore();
			 else {
			    input();   /* eat up trailing quote */
			    yylval.string = yytext+1;
			    return(STRING);
			 }
			}

viz_file		{return(VIZ_FILE);}
info			{return(INFO);}
encoding		{return(ENCODING);}
var_list_ptr		{return(VAR_LIST_PTR);}
timeslice_list_ptr	{return(TIMESLICE_LIST_PTR);}
var_list		{return(VAR_LIST);}
var			{return(VAR);}
var_name		{return(VAR_NAME);}
type			{return(TYPE);}
units			{return(UNITS);}
timeslice_list		{return(TIMESLICE_LIST);}
timeslice		{return(TIMESLICE);}
time			{return(TIME);}
cycle			{return(CYCLE);}
data_obj_list_ptr	{return(DATA_OBJ_LIST_PTR);}
data_obj_list		{return(DATA_OBJ_LIST);}
data_obj		{return(DATA_OBJ);}
obj_name		{return(OBJ_NAME);}
obj_ptr			{return(OBJ_PTR);}
struct_blk		{return(STRUCT_BLK);}
domain_ptr		{return(DOMAIN_PTR);}
values_on               {return(VALUES_ON);}
order 	                {return(ORDER);}
fill			{return(FILL);}
sub_blk			{return(SUBBLK);}
position		{return(POSITION);}
size			{return(SIZE);}
values			{return(VALUES);}
struct_domain		{return(STRUCT_DOMAIN);}
geometry		{return(GEOMETRY);}
coords			{return(COORDS);}

[^ "{},\t\n]*		{/* other words */
			 yylval.string = yytext;
			 return(WORD);
			}

[ \t\n]*		{/* swallow whitespace */}

%%

