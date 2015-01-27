#!/bin/sh


sed -i -e "
/---- Request/i\\
class _covise__$1;\\
class _covise__$1Response;\\
\\

/---- Methods/i\\
class _covise__$1\\
{ public:\\
   struct soap                         *soap                          ;\\
};\\
\\
class _covise__$1Response\\
{ public:\\
   struct soap                         *soap                          ;\\
};\\
\\

/EndOfDeclarations/i\\
//gsoap covise   service method-style:	  $1 document\\
//gsoap covise   service method-encoding:      $1 literal\\
//gsoap covise   service method-action:	  $1 http://www.hlrs.de/organization/vis/covise/$1\\
\\
int __covise__$1(\\
    _covise__$1*              covise__$1,	///< Request parameter\\
    _covise__$1Response *     covise__$1Response	///< Response parameter\\
);\\
\\

" wscovise.h
