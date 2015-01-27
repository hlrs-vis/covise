#!/bin/sh


sed -i -e "
/---- Request/i\\
class _opencover__$1;\\
class _opencover__$1Response;\\
\\

/---- Methods/i\\
class _opencover__$1\\
{ public:\\
   struct soap                         *soap                          ;\\
};\\
\\
class _opencover__$1Response\\
{ public:\\
   struct soap                         *soap                          ;\\
};\\
\\

/EndOfDeclarations/i\\
//gsoap opencover   service method-style:	  $1 document\\
//gsoap opencover   service method-encoding:      $1 literal\\
//gsoap opencover   service method-action:	  $1 http://www.hlrs.de/organization/vis/opencover/$1\\
\\
int __opencover__$1(\\
    _opencover__$1*              opencover__$1,	///< Request parameter\\
    _opencover__$1Response *     opencover__$1Response	///< Response parameter\\
);\\
\\

" wscover.h
