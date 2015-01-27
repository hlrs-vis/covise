
%{
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include <string>
//#ifdef __sgi
using namespace std;
//#endif

#include <iostream>
#include <iomanip>
using namespace std;

static int accum=0;
static string nodes;

static void
dumpNodes(ostream *outs,int* accumL)
{
   if(accumL == 0){
      return;
   }
   *outs << nodes.c_str() << endl;
   nodes = "";
   *accumL = 0;
}
%}

/**
  Don't allow yywrap. There are no multibuffers allowed.
  */
%option noyywrap


%%

"*"DATABASE_HISTORY_NODE\n {
   (*yyout) << YYText();
}

"*"DATABASE_HISTORY_SHELL\n  {
   dumpNodes(yyout,&accum);
   (*yyout) << YYText();
}

"*"END\n {
   dumpNodes(yyout,&accum);
   (*yyout) << YYText();
}

.+\n {
   const char *buf = YYText();
   float info;
   int nums = sscanf(buf,"%g",&info); 
   if(nums==1){
      int ints = int(rint(info));
      // add node to nodes
      ++accum;
      char form[32];
      sprintf(form,"%10d",ints);
      nodes += form;
      if(accum==8){ // flush
         dumpNodes(yyout,&accum);
      }
   }  
   else{
      LexerError(buf);
      exit(-10);
   }  
}  


%%

