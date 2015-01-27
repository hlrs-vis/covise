%{
#ifndef lint
static char *sccsid[] =
{
    "@(#)SCCSID: SCCS/s.plotread.yacc 1.3",
    "@(#)SCCSID: Version Created: 12/08/93 11:42:14"
};
#endif

#include <stdio.h>
#include "plotread.h"
#include "local_defs.h"
#include "bitio.h"

#define SCALE16(bitfile_ptr, scale, min)  (((float)((unsigned int)InputBits(bitfile_ptr,16))/65535.)*(scale)+(min))

extern Plotfile_Obj expected_obj;
extern void *return_obj;
extern FILE *pfile;
extern Variable_Type current_type;
extern Encoding	current_encoding;

static Encoding encoding;
static Ordering ordering = IJK;
static Variable_Type var_type;
static Plotfile *plotfile=NULL;
static PlotList *v_list=NULL, *ts_list=NULL, *do_list=NULL;
static Variable *var_node;
static Timeslice *ts_node;
static Data_Obj *do_node;
#define MAXNINT 3
static int int_list[MAXNINT], count;
static int position[3], size[3];
static Struct_Block *sb;
static Struct_Domain domain;
static int load_domain_flag = 0;
static float fill = 0.;
static long domain_ptr = 0;
static long loaded_domain = 0;
static long old_loc;
static float *subblk;
static float *fval;
static int first_tlist;
%}

%start top

%union {
   float real_value;
   int integer_value;
   char *string;
}

%token BEG END SEP
%token VIZ_FILE
%token INFO
%token ENCODING
%token VAR_LIST_PTR
%token TIMESLICE_LIST_PTR
%token VAR_LIST
%token VAR
%token VAR_NAME
%token TYPE
%token UNITS
%token TIMESLICE_LIST
%token TIMESLICE
%token TIME
%token CYCLE
%token DATA_OBJ_LIST_PTR
%token DATA_OBJ_LIST
%token DATA_OBJ
%token OBJ_NAME
%token OBJ_PTR
%token STRUCT_BLK
%token DOMAIN_PTR
%token VALUES_ON
%token ORDER
%token FILL
%token SUBBLK
%token POSITION
%token SIZE
%token VALUES
%token STRUCT_DOMAIN
%token GEOMETRY
%token COORDS
%token <integer_value> INTEGER_VALUE
%token <real_value> REAL_VALUE
%token <string> STRING WORD

%type <string> string word info var_name obj_name type units
%type <integer_value> var_list_ptr timeslice_list_ptr data_obj_list_ptr obj_ptr cycle
%type <integer_value> t_ptr_or_null
%type <real_value> time real_const

%%

top	:	viz_file	{YYACCEPT;}
	|	var_list	{YYACCEPT;}
	|	timeslice_lists	{YYACCEPT;}
	|	data_obj_list	{YYACCEPT;}
	|	struct_blk	{YYACCEPT;}
	;

viz_file
	:	VIZ_FILE BEG
		{
			 if (expected_obj != VIZ_FILE_OBJ) 
			    	myerror("unexpected vizfile object");

			 /*set up plotfile (init values)*/
			 plotfile = (Plotfile *) UALLOC(Plotfile, 1);
			 plotfile->file = pfile;
			 plotfile->encoding = NATIVE;
			 plotfile->info = "";
			 plotfile->var_list_ptr = 0;
			 plotfile->timeslice_list_ptr = 0;
			 first_tlist = 1;

			 /*set return object to point to plotfile*/
			 return_obj = (void *) plotfile;
		}
		header_list
		END
	;

header_list
	:	header_obj
	|	header_list header_obj
	;

header_obj
	:	info			{plotfile->info = $1;}
	|	encoding		{plotfile->encoding = encoding;}
	|	var_list_ptr		{plotfile->var_list_ptr = $1;}
	|	timeslice_list_ptr	{plotfile->timeslice_list_ptr = $1;}
	;

info	:	INFO BEG
		string
		END		{$$ = $3;}
	;

string  :	STRING
		{
			 char *temp_str;
			 temp_str = (char *) UALLOC(char, yyleng+1);
			 strcpy(temp_str, $1);
			 $$ = temp_str;
		}
	;

encoding:	ENCODING BEG
		word
		END		{encoding = str_to_encoding($3);}
	;

word	:	WORD
		{
			char *temp_str;
                        temp_str = (char *) UALLOC(char, yyleng+1);
                        strcpy(temp_str, $1);
                        $$ = temp_str;
		}
        ;

var_list_ptr :	VAR_LIST_PTR BEG
		INTEGER_VALUE
		END		{$$ = $3;}
	;

timeslice_list_ptr :	TIMESLICE_LIST_PTR BEG
		INTEGER_VALUE
		END		{$$ = $3;}
	;

var_list:	VAR_LIST BEG
		{
			if (expected_obj != VAR_OBJ)
                        	myerror("unexpected variable object");

			 /*start var list*/
                         v_list = (PlotList *) UALLOC(PlotList, 1);
                         init_list(v_list);

                         /*set return object*/
                         return_obj = (void *) v_list;
		}
		v_list
		END
	;

v_list	:	var
	|	v_list var
	;

var	:	VAR BEG
		{
			/*allocate, add var to var list*/
			 var_node = (Variable *) UALLOC(Variable, 1);
			 add_to_list(v_list, var_node);

			 /*set up defaults*/
			 var_node->var_name = "";
			 var_node->type = REAL;
			 var_node->units = "";
		}
		vars
                END
	;

vars
	: var_obj
	| vars var_obj
	;

var_obj
	: var_name 	{var_node->var_name = $1;}
	| type		{var_node->type = var_type;}
	| units		{var_node->units = $1;}
	;

var_name:	VAR_NAME BEG
		string
		END		{$$ = $3;}
	;

type:		TYPE BEG
		word
		END		{var_type = str_to_var_type($3);}
	;

units
	:	UNITS BEG
		string 
		END		{$$ = $3;}
	;

timeslice_lists	
	:	timeslice_list
	|	timeslice_lists timeslice_list
	;

timeslice_list
	:	TIMESLICE_LIST BEG
		{
			if (expected_obj != TIMESLICE_OBJ)
                        	myerror("unexpected timeslice list object");
                        if (first_tlist) 
			{
                            /*start ts_list*/
                            ts_list = (PlotList *) UALLOC(PlotList,1);
                            init_list(ts_list);

                            /*set return object*/
                            return_obj = (void *) ts_list;
			    first_tlist = 0;
			 }
		}
		ts_list
		t_ptr_or_null
		END
			{fseek(pfile, $5, 0);}
	;

ts_list	:	ts
	|	ts_list ts
	;

ts	:	TIMESLICE BEG
		{
			/*allocate, add timeslice to ts_list*/
			 ts_node = (Timeslice *) UALLOC(Timeslice, 1);
			 add_to_list(ts_list, ts_node);
		}
		time
		cycle
		data_obj_list_ptr
		END
		{
			ts_node->time = $4;
			ts_node->cycle = $5;
			ts_node->data_obj_list_ptr = $6;
		}
	;

time	:	TIME BEG
		real_const
		END		{$$ = $3;}
	;

real_const:	REAL_VALUE	{$$ = $1;}
	|	INTEGER_VALUE	{$$ = $1;}
	;

cycle	:	CYCLE BEG
		INTEGER_VALUE
		END		{$$ = $3;}
	;

t_ptr_or_null
	:	timeslice_list_ptr {$$ = $1;}
	|	/* null */	   {$$ = 0;}
	;

data_obj_list_ptr:	DATA_OBJ_LIST_PTR BEG
		INTEGER_VALUE
		END		{$$ = $3;}
	;

data_obj_list
	:	DATA_OBJ_LIST BEG
		{
			if (expected_obj != DATA_LIST_OBJ) 
				myerror("unexpected data object");

			/*start data object list*/
			do_list = (PlotList *) UALLOC(PlotList, 1);
			init_list(do_list);

			/*set return object*/
			return_obj = (void *) do_list;
		}
		do_list
		END
	;

do_list	:	data_obj
	|	do_list data_obj
	;

data_obj:	DATA_OBJ BEG
		{
			/*allocate, add data obj to data obj list*/
			do_node = (Data_Obj *) UALLOC(Data_Obj, 1);
			add_to_list(do_list, do_node);
		}
                obj_name
		var_name
		obj_ptr
		END
		{
			do_node->obj_name = $4;
			do_node->var_name = $5;
			do_node->obj_ptr = $6;
		}
	;

obj_name:	OBJ_NAME BEG
		string
		END		{$$ = $3;}
	;

obj_ptr :	OBJ_PTR BEG
		INTEGER_VALUE
		END		{$$ = $3;}
	;

struct_blk
	:	STRUCT_BLK BEG
		{
			if (expected_obj != STRUCT_BLK_OBJ) 
				myerror("unexpected struct_block object");

			/*allocate struct block*/
			sb = (Struct_Block *) UALLOC(Struct_Block, 1);
			sb->domain = &domain;
			sb->values = NULL;
			return_obj = (void *) sb;
		}

		sb_attrs
		{
			register int i,nval;
			/*allocate memory for values, init*/
			nval=1;
			for (i=0;i<domain.ndim;i++) nval=nval*domain.size[i];
			sb_init_fill(nval);
		}

		subblk_list
		END
	;

sb_attrs:	sb_attr
	|	sb_attrs sb_attr
	;

sb_attr	:	domain_ptr
	|	values_on
	|	order
	|	fill
	;

domain_ptr
	:	DOMAIN_PTR BEG
		INTEGER_VALUE		{domain_ptr = $3;}
		END
                load_domain	{/*load domain if not loaded*/}
	;

load_domain
	:	{
			/* if domain not loaded, then go parse & load domain */
			if (domain_ptr != loaded_domain) 
			{
		    		old_loc = ftell(pfile);
		    		fseek(pfile,domain_ptr,0);
		    		load_domain_flag = 1;
		    		/*
		    		printf("old loc %o\n", old_loc);
		    		printf("loading domain at loc %o\n", domain_ptr);
		    		*/
		 	}
		}
		domain_or_null
		{/* if loaded domain, reset file location */
		if (load_domain_flag) 
		{
			if (loaded_domain != domain_ptr) myerror("domain not found");
			fseek(pfile,old_loc,0);
			load_domain_flag = 0;
			/*
				printf("loc restored to %o\n", old_loc);
		    	*/
		 }
		}
	;

domain_or_null
	:	struct_domain	{loaded_domain = domain_ptr;}
	|	{/* null -- if didn't find domain, match nothing */}
	;

values_on
	:	VALUES_ON BEG
		word	
		END
	;

order
	:	ORDER BEG
		word	
		END		{ordering = str_to_ordering($3);}
	;

fill
	:	FILL BEG
		word 
		END
	;

subblk_list
	:	subblk
	|	subblk_list subblk
	;

subblk:		SUBBLK BEG
		subblk_attrs
		subblk_values
		END
	;

subblk_attrs
	:	subblk_attr
	|	subblk_attrs subblk_attr
	;

subblk_attr
	:	block_position
		{
			if (count != domain.ndim) myerror("invalid subblock position");
		}
	|	block_size
		{
			if (count != domain.ndim) myerror("invalid subblock size");
		}
	;

block_position
	:	POSITION BEG
		{	/*prepare to receive int list*/
			 count = 0;
		}
		int_list
		END	{
				register int i;
				for (i=0;i<count;i++) position[i] = int_list[i];
			 	/*
			 	printf("count: %d\n", count);
			 	printf("position:");
                         	for (i=0;i<count;i++) printf(" %d",position[i]);
			 	printf("\n");
			 	*/
			}
	;

block_size
	:	SIZE BEG
		{/*prepare to receive int list*/
			 count = 0;
		}
		int_list
		END	{
				register int i;
				for (i=0;i<count;i++) size[i] = int_list[i];
			 	/*
				printf("count: %d\n", count);
			 	printf("size:");
                         	for (i=0;i<count;i++) printf(" %d",size[i]);
			 	printf("\n");
			 	*/
			}
	;

int_list:	INTEGER_VALUE		{add_to_int_list($1);
				 /*
				 printf("int_list start: %d\n", $1);
				 */
				}
	|	int_list SEP INTEGER_VALUE
				{add_to_int_list($3);
				 /*
				 printf("int_list append: %d\n", $3);
				 */
				}
	;

subblk_values
	:	VALUES BEG
		{
			register int i,nval;

			nval=1;
        		for (i=0;i<domain.ndim;i++) nval *= size[i];

			/* allocate and read into subblock */
			subblk = read_subblk(nval);

                	/*copy into structured block*/
			copy_subblk_into_sb();

			/*free subblock*/
                	UFREE(subblk);
		}
		END
	;

struct_domain
	:	STRUCT_DOMAIN BEG
		{
			register int i;
			domain.ndim = 0;
			domain.geom = RECTANGULAR;
			domain.coords = NULL;
			for (i=0;i<3;i++) 
			{
				domain.size[i] = 0;
				domain.units[i] = NULL;
			}
			 /*
			 printf("found domain\n");
			 */
		}
		sd_attrs
		sd_coords
		END
		{
			 /*
			 register int i;
			 printf("domain:\n");
			 printf("   ndim: %d  size:", domain.ndim);
			 for (i=0;i<domain.ndim;i++)
			    printf(" %d", domain.size[i]);
			 printf("\n");
			 */
		}
	;

sd_attrs:	sd_attr
	|	sd_attrs sd_attr
	;

sd_attr	:	block_size
		{
			register int i;
			if (count<1 || count>3) myerror("invalid block size");
			domain.ndim = count;
			for (i=0;i<domain.ndim;i++) domain.size[i]=size[i];
		}
	|	sd_geom
	|	sd_units
	;

sd_geom	:	GEOMETRY BEG
		word
		END
	;

sd_units:	UNITS BEG
		string_list
		END
	;

string_list
	:	string
	|	string_list SEP string
	;

word_list
	:	word
	|	word_list SEP word
	;

sd_coords
	:	COORDS BEG
		{
			UFREE(domain.coords);
			domain.coords = (float *) read_struct_coords();
		}
		END
	|	{/* null -- coords optional */}
	;

%%

#include "lex.yy.c"

void yyerror(s)
char *s;
{
   printf("**plotread:yyerror: %s\n", s);
   printf("     current file location %o (octal)\n", ftell(pfile));
   /*exit(1);*/
}

void myerror(s)
char *s;
{
   printf("**plotread:myerror: %s\n", s);
   printf("     current file location %o (octal)\n", ftell(pfile));
   /*exit(1);*/
}

void init_list(list)
PlotList *list;
{
   list->front = NULL;
   list->end = NULL;
}

void add_to_list(list, data)
PlotList *list;
void *data;
{
   List_Node *node;

   /* build list node */
   node = (List_Node *) UALLOC(List_Node, 1);
   node->data = data;
   node->next = NULL;

   /* add to list */
   if (list->end == NULL) 
   {         
	 /* if list empty */
   	 list->front = node;
     	 list->end = node;
   } 
   else 
   {                         
	/* else list not empty */
   	(list->end)->next = node;
   	list->end = node;
   }
}

void add_to_int_list(i)
int i;
{
   if (count >= MAXNINT) myerror("int list too long");
   int_list[count]=i;
   count++;
}

Encoding str_to_encoding(s)
char *s;
{
   if (strcmp(s,"NATIVE") == 0) 
   {
   	return(NATIVE);
   } 
   else if (strcmp(s,"XDR") == 0) 
   {
   	return(XDR);
   } 
   else if (strcmp(s,"BIT16") == 0) 
   {
   	return(BIT16);
   } 
   else 
   {
   	myerror("invalid encoding");
   }
}

Ordering str_to_ordering(s)
char *s;
{
   if (strcmp(s,"IJK") == 0) 
   {
  	 return(IJK);
   } 
   else if (strcmp(s,"KJI") == 0) 
   {
   	return(KJI);
   } 
   else 
   {
   	myerror("invalid ordering");
   }
}

Variable_Type str_to_var_type(s)
char *s;
{
   if (strcmp(s,"REAL") == 0) 
   {
   	return(REAL);
   } 
   else if (strcmp(s,"DOUBLE") == 0) 
   {
   	return(DOUBLE);
   } 
   else if (strcmp(s,"INTEGER") == 0) 
   {
   	return(INTEGER);
   } 
   else if (strcmp(s,"BYTE") == 0) 
   {
   	return(BYTE);
   } 
   else 
   {
   	myerror("invalid variable_type");
   }
}


void copy_subblk_into_sb()
{
   register int i,j,k,n;
   int index1, index2;

   /* set up pointer which treats sb->values as array of floats */
   fval = (float *) sb->values;

   switch (domain.ndim) 
   {
   	case 1:                /* 1-D */
      		n=0;
      		for (i=position[0]-1; i<(position[0]+size[0]-1); i++) 
      		{
       			  fval[i] = subblk[n];
       			  n++;
     		 }
      		break;

	case 2:                /* 2-D */
      		index1 = domain.size[1];
      		n=0;
      		switch (ordering)
      		{	
			case IJK:
				for (i=position[0]-1; i<(position[0]+size[0]-1); i++) 
				{
					for (j=position[1]-1; j<(position[1]+size[1]-1); j++) 
					{
						fval[i*index1 + j] = subblk[n];
						n++;
					}
				} 
				break;

			case KJI:
               			for (j=position[1]-1; j<(position[1]+size[1]-1); j++)
                		{
                       			for (i=position[0]-1; i<(position[0]+size[0]-1); i++)
                        		{
                            			   fval[i*index1 + j] = subblk[n];
                             			   n++;
		                        }
               			 }
				break;
	
			default:
				myerror("invalid ordering");
				break;
		}  /* end switch ordering */

      break;  /* end 2-D case */

   case 3:                /* 3-D */
      n=0;
      index2 = domain.size[1] * domain.size[2];
      index1 = domain.size[2];
      switch (ordering)
      {
        case IJK:
		for (i=position[0]-1; i<(position[0]+size[0]-1); i++) 
		{
			for (j=position[1]-1; j<(position[1]+size[1]-1); j++) 
			{
				for (k=position[2]-1; k<(position[2]+size[2]-1); k++) 
				{
					fval[i*index2 + j*index1 + k] = subblk[n];
					n++;
				}
			}
		}
		break;

	case KJI:
                for (k=position[2]-1; k<(position[2]+size[2]-1); k++)
                {
                        for (j=position[1]-1; j<(position[1]+size[1]-1); j++)
                        {
                                for (i=position[0]-1; i<(position[0]+size[0]-1); i++)
                                {
                                        fval[k + j*index1 + i*index2] = subblk[n];
                                        n++;
                                }
                        }
                }
                break;

	default:
                myerror("invalid ordering");
                break;
        }  /* end switch ordering */

      break;   /* end 3-D case */

   default:
      myerror("copy_subblk_into_sb: bad ndim");
   }
}

void
sb_init_fill(nval)
int 	nval;
{
	register int i;

	/* 
	 *	for now, lets convert all values
	 *	to float for subsequent avs use
	 */
	fval = (float *) UALLOC(float, nval);
	if (fval == (float *)0)
		myerror("sb_init_fill: Ran out of memory trying to allocate values");
	for (i=0;i<nval;i++) fval[i]=fill;
	sb->values = (void *) fval;

}


float *
read_bit16_real(nval, into)
int	nval;
float	*into;
{
	float	min, max, scale, chunk16;
	float	*value;
	int 	run, accum;
	unsigned int count;
	register int i;
	BIT_FILE *bitfile_ptr;

	/*
	 *      Initialize bits structure
	 */
	bitfile_ptr = AllocInputBitFile(pfile);
	if (bitfile_ptr == NULL)
		myerror("read_bit16_real: Not enough memory to allocate a bits structure");

	/*
         * if space to copy values into not already provided,
	 * allocate a subblock worth of space to
	 * decode into
	 */
        if (!into)
        {
		value = (float *) UALLOC(float, nval);      
		if (value == (float *)0)
			myerror("read_bit16_real: Ran out of memory trying to allocate values");
        }
        else
	        value = into;

        /* pick off min, max info; compute scale */
	fscanf(pfile, "%g;%g;", &min, &max);
	scale = max - min;

	/*
	 *      decode values
	 */
	accum = 0;
	while (accum < nval)
	{
       		/*
         	 *      Get 1st bit and see if run(1) or not               
         	 */
        	run = InputBit(bitfile_ptr);

        	/*
        	 *      Get count in next 15 bits, 0 count = 32768
       		 *      For runs, count is number of times
        	 *      to repeat next value - otherwise
        	 *      count is the number of values to follow
        	 */
        	count = (int) InputBits(bitfile_ptr, 15);
        	if (count == 0) count = 32768;
        	if (count+accum > nval) 
                	myerror("Number of values found so far is greater than expected - using expected # values.");
        	if (run)
        	{
               		chunk16 = SCALE16(bitfile_ptr, scale, min);
                	for (i=accum; i<count+accum; i++) value[i] = chunk16;
        	}
        	else
                	for (i=accum; i<count+accum; i++)
                       		 value[i] = SCALE16(bitfile_ptr, scale, min);

        	accum += count;

	}  /* end while */ 

	FreeInputBitFile(bitfile_ptr);
	return(value);
}

float *
read_bit16_byte(nval)
int	nval;
{
}

float *	
read_native_struct_coords(nval)
int	nval;
{

	/* NOTE:  This routine will not really handle any type other
	 *	than float properly
 	 */
	subblk = (float *) UALLOC(float, nval);
	if (subblk == (float *)0)
		myerror("read_native_struct_coords: Ran out of memory trying to allocate coords");
       	fread((char *)subblk, sizeof(float), nval, pfile);
	return(subblk);
}

float *
read_bit16_struct_coords(nval)
int nval;
{
        float *coords, *into;
        int i;

	/* allocate coords array */
	coords = (float *) UALLOC(float, nval);
	if (coords == (float *)0)
		myerror("read_bit16_struct_coords: Ran out of memory trying to allocate coords");

        /* 
	   x,y,z coords must be read as separate 16-bit encoded sequences
           since minmax info is specified separately for each
        */
        into = coords;
        for (i=0; i<domain.ndim; i++) {
		read_bit16_real(domain.size[i], into);
		into += domain.size[i];
	}

	return(coords);
}

float *	
read_native_struct_subblk(nval)
int	nval;
{

	/* NOTE:  This routine will not really handle any type other
	 *	than float properly
 	 */
	switch (current_type)
	{
		case REAL:
        		subblk = (float *) UALLOC(float, nval);
			if (subblk == (float *)0)
				myerror("read_native_struct_subblk: Ran out of memory trying to allocate values");
        		fread((char *)subblk, sizeof(float), nval, pfile);
			return(subblk);
			break;

		case DOUBLE:
			/*  for now fake it with reals since the generator
			 *  is producing reals...............
        		subblk = (double *) UALLOC(double, nval);
			if (subblk == (float *)0)
				myerror("read_native_struct_subblk: Ran out of memory trying to allocate values");
        		fread((char *)subblk, sizeof(double), nval, pfile);
			 */
        		subblk = (float *) UALLOC(float, nval);
        		fread((char *)subblk, sizeof(float), nval, pfile);
			return(subblk);
			break;

		case INTEGER:
        		subblk = (float *) UALLOC(int, nval);
			if (subblk == (float *)0)
				myerror("read_native_struct_subblk: Ran out of memory trying to allocate values");
        		fread((char *)subblk, sizeof(int), nval, pfile);
			return(subblk);
			break;

		case BYTE:
        		subblk = (float *) UALLOC(char, nval);
			if (subblk == (float *)0)
				myerror("read_native_struct_subblk: Ran out of memory trying to allocate values");
        		fread((char *)subblk, sizeof(char), nval, pfile);
			return(subblk);
			break;

		default:
			myerror("Bad current type");
			break;
	}
	
}

float *	
read_bit16_struct_subblk(nval)
int	nval;
{
	void	*values;
	switch (current_type)
	{
		case REAL:
			return(read_bit16_real(nval, NULL));
			break;

		case DOUBLE:
			return(read_bit16_real(nval, NULL));
			break;

		case INTEGER:
			return(read_bit16_real(nval, NULL));
			break;

		case BYTE:
			return(read_bit16_byte(nval));
			break;

		default:
			myerror("Bad current type");
			break;
	}
}

float *
read_xdr_struct_subblk(nval)
int	nval;
{
}

float *
read_xdr_struct_coords(nval)
int	nval;
{
}

float *
read_subblk(nval)
int	nval;
{
	switch (current_encoding)
	{
		case NATIVE:
			return(read_native_struct_subblk(nval));
			break;

		case BIT16:
			return(read_bit16_struct_subblk(nval));
			break;

		case XDR:
			return(read_xdr_struct_subblk(nval));
			break;

		default:
			myerror("current encoding unrecognizable");
			break;
	}

}

float *
read_struct_coords()
{
	register int i, nval, ncoord;
	float *tmp;
	float *coord_ptr;
	float *coords;

        nval=0;
        for (i=0;i<domain.ndim;i++) nval = nval+domain.size[i];

        switch (current_encoding)
        {
                case NATIVE:
                        return(read_native_struct_coords(nval));
                        break;

                case BIT16:
			return(read_bit16_struct_coords(nval));
                        break;

                case XDR:
                        return(read_xdr_struct_coords(nval));
                        break;

                default:
                        myerror("current encoding unrecognizable");
                        break;
        }

}

