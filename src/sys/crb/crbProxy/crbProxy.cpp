/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ipc.h>
#include <sys/time.h>
#else
#include <stdio.h>
#include <process.h>
#include <io.h>
#endif

#include <fcntl.h>
#include <stdlib.h>

#include <util/covise_version.h>
#include "CRBConnection.h"
#include <dmgr/dmgr.h>
#include <covise/Covise_Util.h>
#include <config/CoviseConfig.h>

#ifdef _NEC
#include <sys/socke.h>
#endif

int main(int argc, char *argv[])
{

    //moduleList mod;
    //char *instance;
    //int proc_id;

    if (argc != 4)
    {
        cerr << "crbProxy (CoviseRequestBrokerProxy) with inappropriate arguments called\n";
        exit(-1);
    }

#ifdef _WIN32
    WORD wVersionRequested;
    WSADATA wsaData;
    int err;
    wVersionRequested = MAKEWORD(1, 1);

    err = WSAStartup(wVersionRequested, &wsaData);
#endif

    int port = atoi(argv[1]);
    //Host *host = new Host(argv[2]);
    int id = atoi(argv[3]);
    //proc_id = id;
    //instance = argv[4];
    //sprintf(err_name,"err%d",id);

    CRBConnection *datamgrProx = new CRBConnection(port, argv[2], id);

    datamgrProx->execCRB(argv[4]);
    while (1)
    {
        datamgrProx->processMessages();
    }

    /*

   while(1)
   {
      msg = datamgr->wait_for_msg();
      switch(msg->type)
      {
         case UI:                                 // get Message-Keyword
            msg_key=strtok(msg->data,"\n");
            if (strcmp(msg_key,"FILE_SEARCH")==0)
            {
   int i;
   #ifndef _WIN32
   char *tmp;
   #endif
   //char *sfilt; // Filter ohne Pfad
   char *hostname=strtok(NULL,"\n");
   char *user=strtok(NULL,"\n");
   char *mod=strtok(NULL,"\n");
   char *inst=strtok(NULL,"\n");
   char *port=strtok(NULL,"\n");
   char *path=strtok(NULL,"\n");
   char *sfilt=strtok(NULL,"\n");
   if(path[0]==0)
   {
   path[0]='/';
   path[1]=0;
   }
   CharBuffer buf(strlen(path)*20);
   buf+="FILE_SEARCH_RESULT\n";
   buf+=hostname;
   buf+='\n';
   buf+=user;
   buf+='\n';
   buf+=mod;
   buf+='\n';
   buf+=inst;
   buf+='\n';
   buf+=port;
   buf+='\n';
   CharBuffer buf2(strlen(path)*20);

   int num=0;
   #ifndef _WIN32
   Directory *dir=Directory::open(path);
   if(dir)
   {
   for(i=0;i<dir->count();i++)
   {
   tmp=dir->full_name(i);
   if(dir->is_directory(i))
   {
   buf2+=tmp;
   buf2+='\n';
   num++;
   }
   delete[] tmp;
   }
   }
   #endif
   buf+=num;
   buf+='\n';
   buf+=(const char *)buf2;
   num=0;
   CharBuffer buf3(strlen(path)*20);
   #ifndef _WIN32
   if(dir)
   {
   for(i=0;i<dir->count();i++)
   {
   tmp=(char *)dir->name(i);
   if((!dir->is_directory(i))&&(dir->match(tmp,sfilt)))
   {
   buf3+=tmp;
   buf3+='\n';
   num++;
   }
   // the following line seems to be a bad bug
   // and is therefore disabled. awi
   //delete[] tmp;
   }
   delete dir;
   }
   #endif

   buf+=num;
   buf+='\n';
   buf+=(const char *)buf3;
   //delete[] path;
   Message *retmsg= new Message;
   retmsg->type=UI;
   retmsg->data=(char *)((const char *)buf);
   retmsg->data.length()=strlen(retmsg->data)+1;
   datamgr->send_ctl_msg(retmsg);
   retmsg->data=NULL;
   delete retmsg;
   }
   else
   {
   cerr<<"UNKNOWN UI MESSAGE"<<msg_key<<"\n";
   }
   break;
   case CRB_EXEC:
   if(msg->data[0]!='\001')
   {
   char *name=NULL,*cat=NULL;
   int i=0;
   name=msg->data;
   while(msg->data[i])
   {
   if(msg->data[i]==' ')
   {
   msg->data[i]='\0';
   i++;
   cat=msg->data+i;
   break;
   }
   i++;
   }
   while(msg->data[i])
   {
   if(msg->data[i]==' ')
   {
   msg->data[i]='\0';
   i++;
   break;
   }
   i++;
   }
   mod.start(name,cat,msg->data+i);
   }
   else
   {

   char *args[1000];
   int i=1,n=0;
   while(msg->data[i])
   {
   if(msg->data[i]==' ')
   {
   msg->data[i]='\0';
   args[n]=msg->data+i+1;
   //fprintf(stderr,"args %d:%s\n",n,args[n]);
   n++;
   }
   i++;
   }
   args[n]=NULL;
   #ifdef _WIN32
   spawnvp(P_NOWAIT,args[0],(const char * const *)args);
   #else
   int pid = fork();
   if(pid == 0)
   {
   //fprintf(stderr,"args0:%s\n",args[n]);
   execvp(args[0],args);
   }
   else
   {
   // Needed to prevent zombies
   // if childs terminate
   signal(SIGCHLD, SIG_IGN);

   }
   #endif
   }
   break;
   default:
   send_back = datamgr->handle_msg(msg);
   if(send_back == 3)
   {
   delete datamgr;
   exit(0);
   //print_exit(__LINE__, __FILE__, 0);
   }
   if((send_back == 2) && (msg->type != EMPTY))
   msg->conn->send_msg(msg);
   break;
   }
   msg->delete_data();
   datamgr->delete_msg(msg);
   }*/
}
