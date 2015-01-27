/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <cstring>

#include "wce_host.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <netdb.h>

#undef DEBUG

using namespace covise;
char *Host::getNumericIpAddr(const char *address)
{
    Host ch(address);
    char *retVal = new char[strlen(ch.getName()) + 1];
    strcpy(retVal, ch.getName());

    return retVal;
}

char *Host::getSymbolicIpAddr(const char *numericIP)
{
    char *retVal = new char[4096];
    static bool onlyNumeric = false;
    if (!onlyNumeric)
    {
        //try to get the symbolic name of
        //the host
        Host ch(numericIP);

        const char *haddr = (const char *)ch.get_char_address();
        //char buf[1024];
        struct hostent *he = NULL;
        he = gethostbyaddr(haddr, 4, AF_INET);
        if (NULL != he)
        {
            sprintf(retVal, "%s", he->h_name);
// cutting after the 2nd dot is uncommon
#if 0
         char *dot = strchr(retVal,'.');
         if ( (dot) && (*dot))
         {
            dot = strchr(dot+1,'.');
            if (dot)
               *dot = '\0';
         }
#endif
        }
        else
        {
            sprintf(retVal, "%s", numericIP);
            /*//TODO coConfig - das muss wieder richtig geparst werden
         covise::coCoviseConfig::ScopeEntries ipe = covise::coCoviseConfig::getScopeEntries("System.IpTable");
         const char **ipEntries = ipe.getValue();
         const char *last;
         if(NULL!=ipEntries)
         {
            bool gotAll=false;
            bool found=false;
            do
            {
               //An IpTable Entry has the form
               //<symbolic> <numeric>
               //The CoviseConfig::getScopeEntries
               //method gets them word by word
               //so we have to parse two of them
               last=*ipEntries;
               fprintf(stderr, "IPTABLE:%s ", last);
               ipEntries++;
               if(NULL!=*ipEntries)
               {
                  fprintf(stderr, "IPTABLE:%s \n", *ipEntries);
                  if(0==strcmp(numericIP,*ipEntries))
                  {
                     //We found the entry
                     sprintf(retVal,last);
                     found=true;
                  }
                  else
                  {
                     //There is an entry, but it does not match
                     ipEntries++;
                     if(NULL==*ipEntries)
                     {
                        onlyNumeric=true;
                        gotAll=true;
                        sprintf(retVal, "%s", numericIP);
                     }
                  }
               }
               else
               {
                  //We got all entries, the last of which is incomplete
                  onlyNumeric=true;
                  gotAll=true;
                  sprintf(retVal, "%s", numericIP);
               }
            }while((!gotAll)&&(!found));
         }
         else
         {
            onlyNumeric=true;
         }*/
        }
    }
    else
    {
        sprintf(retVal, "%s", numericIP);
    }
    return retVal;
}

void Host::setName(const char *n)
{
    name = new char[1 + strlen(n)];
    strcpy(name, n);
}

void Host::HostNumeric(const char *n)
{
    int tmpaddr[4];
    bool invalidIP = false;
    invalidIP = (n == NULL);
    invalidIP |= (strlen(n) > 15);
    int countNumbers = sscanf(n, "%d.%d.%d.%d", &tmpaddr[0],
                              &tmpaddr[1],
                              &tmpaddr[2],
                              &tmpaddr[3]);
    invalidIP |= (countNumbers != 4);
    if (invalidIP)
    {
        setName("Invalid IP address");
    }
    else
    {
        char_address[0] = tmpaddr[0];
        char_address[1] = tmpaddr[1];
        char_address[2] = tmpaddr[2];
        char_address[3] = tmpaddr[3];
        setName(n);
    }
#ifdef DEBUG
    LOGINFO(name);
#endif
}

void Host::HostSymbolic(const char *n)
{
    struct hostent *hent;
    //The address is not numeric
    //and we try to convert the
    //symbolic address into a numeric one
    //II) By  gethostbyname
    //III) If this fails we get "unresolvable IP address"

    hent = gethostbyname(n);
    if (NULL == hent)
    {
        setName("unresolvable IP address");
        return;
    }
    char_address[0] = *hent->h_addr_list[0];
    char_address[1] = *(hent->h_addr_list[0] + 1);
    char_address[2] = *(hent->h_addr_list[0] + 2);
    char_address[3] = *(hent->h_addr_list[0] + 3);
    char buf[1024];
    sprintf(buf, "%d.%d.%d.%d",
            char_address[0],
            char_address[1],
            char_address[2],
            char_address[3]);
    setName(buf);
}

Host::Host(const char *n, bool numeric)
{
    char_address[0] = 0;
    char_address[1] = 0;
    char_address[2] = 0;
    char_address[3] = 0;

    if ((numeric) || (NULL == n))
    {
        HostNumeric(n);
    }
    else
    {
        HostSymbolic(n);
    }
}

Host::Host(const Host &h)
{
    this->char_address[0] = h.char_address[0];
    this->char_address[1] = h.char_address[1];
    this->char_address[2] = h.char_address[2];
    this->char_address[3] = h.char_address[3];
    name = new char[strlen(h.name) + 1];
    strcpy(name, h.name);
}

Host::Host(unsigned long a)
{
    unsigned char *tmpaddr;
    tmpaddr = (unsigned char *)&a;
    char_address[0] = tmpaddr[0];
    char_address[1] = tmpaddr[1];
    char_address[2] = tmpaddr[2];
    char_address[3] = tmpaddr[3];
    char tmpName[256];
    sprintf(tmpName, "%d.%d.%d.%d",
            char_address[0],
            char_address[1],
            char_address[2],
            char_address[3]);
    name = new char[1 + strlen(tmpName)];
    strcpy(name, tmpName);
}

Host::~Host()
{
    delete[] name;
    //   delete[] sname;
    //    LOGINFO(
    //		  "Delete: be carefull, data might be used elsewhere!!!");
}

Host &Host::operator=(const Host &h)
{
    this->char_address[0] = h.char_address[0];
    this->char_address[1] = h.char_address[1];
    this->char_address[2] = h.char_address[2];
    this->char_address[3] = h.char_address[3];
    name = new char[strlen(h.name) + 1];
    strcpy(name, h.name);
    return *this;
}

void Host::get_char_address(unsigned char *c) const
{
    c[0] = char_address[0];
    c[1] = char_address[1];
    c[2] = char_address[2];
    c[3] = char_address[3];
}

uint32_t Host::get_ipv4() const
{
    return (char_address[0] << 24) | (char_address[1] << 16) | (char_address[2] << 8) | char_address[3];
}

Host::Host()
{
    char buf[4096];
    gethostname(buf, 4096);
    HostSymbolic(buf);
}

Host *Host::get_local_host()
{
    return new Host();
}
