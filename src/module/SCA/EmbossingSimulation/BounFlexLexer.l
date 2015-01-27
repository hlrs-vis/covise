
%{
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include <string>
using namespace std;

#include <iostream>
#include <iomanip>
using namespace std;

static enum State {SET_NODE_INI=0,SET_NODE_LIST=1,BOUNDARY_SPC_SET=2,\
            CONSTRAINED_NODE_SET=3,END=4} state = SET_NODE_INI;
static int accum;
static string nodes;

static void
dumpNodes(ostream *outs,int* accumL)
{
   if(accumL == 0){
      return;
   }
   *outs << nodes.c_str() << endl;
   nodes="";
   *accumL = 0;
}
%}

/**
  Don't allow yywrap. There are no multibuffers allowed.
  */
%option noyywrap


%%

"*"SET_NODE_LIST\n {
   state = SET_NODE_INI;
   (*yyout) << YYText();
}

"*"BOUNDARY_SPC_SET\n  {
   if(state==SET_NODE_LIST){
      dumpNodes(yyout,&accum);
   }
   state = BOUNDARY_SPC_SET;
   (*yyout) << YYText();
}

"*"CONSTRAINED_NODE_SET\n  {
   if(state==SET_NODE_LIST){
      dumpNodes(yyout,&accum);
   }
   state = CONSTRAINED_NODE_SET;
   (*yyout)<< YYText();
}

"*"END\n  {
   if(state==SET_NODE_LIST){
      dumpNodes(yyout,&accum);
   }
   state = END;
   (*yyout)<< YYText();
}

.+\n {
   const char *buf = YYText();
   switch(state){
   case SET_NODE_INI:
      {
      float info[5];
      int nums = sscanf(buf,"%g %g %g %g %g",\
                     info,info+1,info+2,info+3,info+4);
      if(nums==5){
         int nodLab = int(rint(*info));
         (*yyout).setf(ios::right);
         (*yyout) << setw(10) << nodLab;
         (*yyout).setf(ios::fixed);
         (*yyout).precision(3); 
         (*yyout) << setw(10) << info[1];
         (*yyout) << setw(10) << info[2];
         (*yyout) << setw(10) << info[3];
         (*yyout) << setw(10) << info[4] << endl;
         state = SET_NODE_LIST;
         accum = 0;
      }  
      else{
         LexerError(buf);
         exit(-5);
      }  
      }  
      break;
   case SET_NODE_LIST:
      {
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
         exit(-6);
      }  
      }  
      break;
   case BOUNDARY_SPC_SET:
      {
      float info[8];
      int nums = sscanf(buf,"%g %g %g %g %g %g %g %g",\
                     info+0,info+1,info+2,info+3,info+4,\
                     info+5,info+6,info+7);
      if(nums==8){
         int ints[8];
         ints[0] = int(rint(info[0]));
         ints[1] = int(rint(info[1]));
         ints[2] = int(rint(info[2]));
         ints[3] = int(rint(info[3]));
         ints[4] = int(rint(info[4]));
         ints[5] = int(rint(info[5]));
         ints[6] = int(rint(info[6]));
         ints[7] = int(rint(info[7]));
         (*yyout).setf(ios::right);
         (*yyout) << setw(10) << ints[0];
         (*yyout) << setw(10) << ints[1];
         (*yyout) << setw(10) << ints[2];
         (*yyout) << setw(10) << ints[3];
         (*yyout) << setw(10) << ints[4];
         (*yyout) << setw(10) << ints[5];
         (*yyout) << setw(10) << ints[6];
         (*yyout) << setw(10) << ints[7] << endl;
      }  
      else{
         LexerError(buf);
         exit(-7);
      }  
      }  
      break;
   case CONSTRAINED_NODE_SET:
      {
      float info[2];
      int nums = sscanf(buf,"%g %g",\
                     info+0,info+1);
      if(nums==2){
         int ints[2];
         ints[0] = int(rint(info[0]));
         ints[1] = int(rint(info[1]));
         (*yyout).setf(ios::right);
         (*yyout) << setw(10) << ints[0];
         (*yyout) << setw(10) << ints[1] << endl;
      }  
      else{
         LexerError(buf);
         exit(-8);
      }  
      }  
      break;
   default:
      LexerError("Forbidden state");
      exit(-9);
   }
}

%%

