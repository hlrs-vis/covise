/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <ctype.h>

#include "SCANNER.H"

/*-------------------------------------------------------------*/
/* Memberfunktionen der Klasse scanner                         */
/*-------------------------------------------------------------*/

HlSymEntry *HlScanner::catchNumber(unsigned int c)
{
    HlSymEntry *sym = NULL;
    string lexbuf;

    if (isdigit(c))
    {
        while (isdigit(c) || (c == '.'))
        {
            lexbuf += c;
            c = nextChar();
        }

        backChar(c);
        sym = mSymTab->lookup(NUMBER);
        sym->setVal(atof(lexbuf.c_str()));
        return sym;
    }

    return NULL;
}

HlSymEntry *HlScanner::catchName(unsigned int c)
{
    HlSymEntry *sym = NULL;
    string lexbuf;

    if (isalpha(c))
    {
        while (isalnum(c))
        {
            lexbuf += tolower(c);
            c = nextChar();
        }
        backChar(c);

        sym = mSymTab->lookup(lexbuf);
        if (sym == NULL)
            sym = mSymTab->insert(lexbuf, IDENT);
        return sym;
    }

    return NULL;
}

HlSymEntry *HlScanner::catchOp(unsigned int c)
{
    HlSymEntry *sym = NULL;
    string lexbuf;

    if (c == '<' || c == '>' || c == '=')
    {
        lexbuf = c;
        c = nextChar();
        if (c == '=')
            lexbuf += c;
        else
            backChar(c);

        sym = mSymTab->lookup(lexbuf);
        return sym;
    }

    return NULL;
}

HlSymEntry *HlScanner::nextToken()
{
    unsigned int c;
    HlSymEntry *sym;

    for (;;)
    {
        c = nextChar();

        if (isspace(c))
            continue;

        if ((sym = catchNumber(c)) != NULL)
            break;

        if ((sym = catchName(c)) != NULL)
            break;

        if ((sym = catchOp(c)) != NULL)
            break;

        if (c == EOS)
        {
            sym = mSymTab->lookup(EOS);
            break;
        }

        sym = mSymTab->lookup(string(1, c));

        if (sym->tokenIn(MINUS))
        {
            if (isStartSymbol(mLastToken))
            {
                sym = mSymTab->lookup(UMIN);
            }
        }
        break;
    }

    mLastToken = sym;
    return sym;
}
