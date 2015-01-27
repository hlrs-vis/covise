
%{
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include <iostream>
#include <iomanip>
using namespace std;

static enum State {NODE=0,ELEMENT_SHELL=1,ELEMENT_SOLID=2} state;
%}

/**
  Don't allow yywrap. There are no multibuffers allowed.
  */
%option noyywrap


%%

"*"NODE\n {
   state = NODE;
   (*yyout) << YYText();
}

"*"ELEMENT_SHELL\n  {
   state = ELEMENT_SHELL;
   (*yyout) << YYText();
}

"*"ELEMENT_SOLID\n  {
   state = ELEMENT_SOLID;
   (*yyout)<< YYText();
}

"*"END\n  {
   (*yyout)<< YYText();
}

.+\n {
   const char *buf = YYText();
   switch(state){
   case NODE:
      {
      float info[6];
      int nums = sscanf(buf,"%g %g %g %g %g %g",\
                     info,info+1,info+2,info+3,info+4,info+5);
      if(nums==6){
         int nodLab = int(rint(*info));
         int tail1 = int(rint(info[4]));
         int tail2 = int(rint(info[5]));
         (*yyout).setf(ios::right);
         (*yyout) << setw(8) << nodLab;
         (*yyout).setf(ios::uppercase);
         (*yyout).setf(ios::scientific);
         (*yyout).precision(4); 
         (*yyout) << setw(16) << info[1];
         (*yyout) << setw(16) << info[2];
         (*yyout) << setw(16) << info[3];
         (*yyout) << setw(8) << tail1 << setw(8) << tail2 << endl;
      }  
      else{
         LexerError(buf);
         exit(-2);
      }  
      }  
      break;
   case ELEMENT_SHELL:
      {
      float info[6];
      int nums = sscanf(buf,"%g %g %g %g %g %g",\
                     info,info+1,info+2,info+3,info+4,info+5);
      if(nums==6){
         int ints[6];
         ints[0] = int(rint(info[0]));
         ints[1] = int(rint(info[1]));
         ints[2] = int(rint(info[2]));
         ints[3] = int(rint(info[3]));
         ints[4] = int(rint(info[4]));
         ints[5] = int(rint(info[5]));
         (*yyout).setf(ios::right);
         (*yyout) << setw(8) << ints[0];
         (*yyout) << setw(8) << ints[1];
         (*yyout) << setw(8) << ints[2];
         (*yyout) << setw(8) << ints[3];
         (*yyout) << setw(8) << ints[4];
         (*yyout) << setw(8) << ints[5] << endl;
      }  
      else{
         LexerError(buf);
         exit(-3);
      }  
      }  
      break;
   case ELEMENT_SOLID:
      {
      float info[10];
      int nums = sscanf(buf,"%g %g %g %g %g %g %g %g %g %g",\
                     info+0,info+1,info+2,info+3,info+4,\
                     info+5,info+6,info+7,info+8,info+9);
      if(nums==10){
         int ints[10];
         ints[0] = int(rint(info[0]));
         ints[1] = int(rint(info[1]));
         ints[2] = int(rint(info[2]));
         ints[3] = int(rint(info[3]));
         ints[4] = int(rint(info[4]));
         ints[5] = int(rint(info[5]));
         ints[6] = int(rint(info[6]));
         ints[7] = int(rint(info[7]));
         ints[8] = int(rint(info[8]));
         ints[9] = int(rint(info[9]));
         (*yyout).setf(ios::right);
         (*yyout) << setw(8) << ints[0];
         (*yyout) << setw(8) << ints[1];
         (*yyout) << setw(8) << ints[2];
         (*yyout) << setw(8) << ints[3];
         (*yyout) << setw(8) << ints[4];
         (*yyout) << setw(8) << ints[5];
         (*yyout) << setw(8) << ints[6];
         (*yyout) << setw(8) << ints[7];
         (*yyout) << setw(8) << ints[8];
         (*yyout) << setw(8) << ints[9] << endl;
      }  
      else{
         cerr << "Adios amigos\n";
         LexerError(buf);
         exit(-3);
      }  
      }  
      break;
   }
}

%%

