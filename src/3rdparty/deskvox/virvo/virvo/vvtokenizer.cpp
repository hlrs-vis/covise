// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvtokenizer.h"

const int vvTokenizer::BLOCK_SIZE    = 4096;
const int vvTokenizer::MAX_TOKEN_LEN = 1024;

//#define VV_STANDALONE      // uncomment for demonstration

//----------------------------------------------------------------------------
/** Constructor.
  The default state is: <UL>
  <LI>lowercase off</LI>
  <LI>EOL is not significant</LI>
  <LI>Numbers are not parsed</LI></UL>
  @param file pointer to source file which must have been opened for reading
*/
vvTokenizer::vvTokenizer(std::ifstream& file)
  : file(file)
{
  // Memory allocation:
  sval = new char[MAX_TOKEN_LEN + 1];
  data = new char[BLOCK_SIZE];

  // Attribute initialization:
  ttype            = VV_NOTHING;
  nval             = 0.0f;
  sval[0]          = '\0';
  pushedBack       = false;
  line             = determineCurrentLine();
  caseConversion   = VV_NONE;
  eolIsSignificant = false;
  blockUsed = 0;
  cur = 0;
  firstPass = true;

  setDefault();                                   // disable number parsing
}

//----------------------------------------------------------------------------
/** Destructor.
  Delete the tokenizer to get a correct value of the file pointer.
  The destructor also makes sure that the file pointer is set to the byte
  after the last tokenizer read.
*/
vvTokenizer::~vvTokenizer()
{
  long offset;

  offset = cur-blockUsed;
  if (!firstPass) --offset;
  file.seekg(offset, file.cur);
  delete[] sval;
  delete[] data;
}

//----------------------------------------------------------------------------
/** Find the current line number. The algorithm scans all characters up
  to the current file pointer value and determines the line number by counting
  the line breaks.
  @return current line number or -1 if there was an error
*/
int vvTokenizer::determineCurrentLine()
{
  int  lineNr = 1;
  std::streamoff current;                                   // current position in file
  long i;
  int  c;

  current = file.tellg();                         // memorize file pointer
  if (current<0) return -1;
  file.seekg(0, std::ios::beg);

  // Count number of line breaks by counting the number
  // of '\r' characters in the file:
  for (i=0; i<current; ++i)
  {
    c = file.get();
    if (c=='\r')
      ++lineNr;
  }

  return lineNr;
}

//----------------------------------------------------------------------------
/** Read a character from the file buffer.
  @return read character or -1 for EOF
*/
int vvTokenizer::readChar()
{
  if (cur >= blockUsed)                           // end of buffer reached?
  {
    if (file.eof()) return -1;
    else
    {
      // Read next block:
      file.read(data, BLOCK_SIZE);
      blockUsed = file.gcount();
      cur = 0;
    }
  }
  return (unsigned char)data[cur++];
}

//----------------------------------------------------------------------------
/** Interpret all displayable characters as alpha characters and the rest
  of ASCII codes as whitespace characters. Displayable characters are ASCII
  codes 33..126 and 192..255.
  This method reverses the effects of setCommentCharacter,
  setWhitespaceCharacter, setParseNumbers and setAlphaCharacters.
*/
void vvTokenizer::setDefault()
{
  int i;

  for (i=0; i<33; ++i)
    ctype[i] = VV_WHITESPACE;
  for (i=33; i<127; ++i)
    ctype[i] = VV_ALPHA;
  for (i=127; i<192; ++i)
    ctype[i] = VV_WHITESPACE;
  for (i=192; i<256; ++i)
    ctype[i] = VV_ALPHA;
}

//----------------------------------------------------------------------------
/** Set the line number.
  The tokenizer uses line number 1 as default for the first line when
  instantiated, so this function only needs to be called when the file
  pointer was not set to the beginning of the file.
  @param lineNumber new line number to be used as current line
*/
void vvTokenizer::setLineNumber(int lineNumber)
{
  line = lineNumber;
}

//----------------------------------------------------------------------------
/** Get the line number. The first line of the file is line 1.
  @return the current line number [1..numLines]
*/
int vvTokenizer::getLineNumber()
{
  return line;
}

//----------------------------------------------------------------------------
/** Get the position of the file pointer of the byte that will be read next.
  This returns what one would expect ftell() would return, but the value
  is corrected by the position in the read-ahead buffer.
  @return the current byte number [1..numLines]
*/
long vvTokenizer::getFilePos()
{
  long pos = static_cast< long >(file.tellg()) - blockUsed + cur;
  if (!firstPass) --pos;
  return pos;
}

//----------------------------------------------------------------------------
/** Set the position of the file pointer to a new position.
  This is useful if part of the file has to be parsed with an external
  routine and after it the tokenizer should regain control.
  @param newFP file pointer the tokenizer should continue at
*/
void vvTokenizer::setFilePos(std::streampos pos)
{
  file.seekg(pos);
  ttype = VV_NOTHING;
  nval = 0.0f;
  sval[0] = '\0';
  pushedBack = false;
  blockUsed = 0;
  cur = 0;
  firstPass = true;
  line = determineCurrentLine();
}

//----------------------------------------------------------------------------
/** Set an alpha character.
  Multiple characters can be defined as alpha characters.
  This operation can only be reversed by @see setAllAlpha.
  @param cc comment character (e.g. '#' for UNIX style comments)
*/
void vvTokenizer::setAlphaCharacter(char cc)
{
  ctype[static_cast<int>(cc)] = VV_ALPHA;
}

//----------------------------------------------------------------------------
/** Set a comment character.
  Multiple characters can be defined as comment characters.
  This operation can only be reversed by @see setDefault.
  @param cc comment character (e.g. '#' for UNIX style comments)
*/
void vvTokenizer::setCommentCharacter(char cc)
{
  ctype[static_cast<int>(cc)] = VV_COMMENT;
}

//----------------------------------------------------------------------------
/** Set a whitespace character, which will be ignored in the parsing process.
  Multiple characters can be defined as whitespace characters.
  This operation can only be reversed by @see setDefault.
  @param wc whitespace character (e.g. '=' to ignore all equal signs)
*/
void vvTokenizer::setWhitespaceCharacter(char wc)
{
  ctype[static_cast<int>(wc)] = VV_WHITESPACE;
}

//----------------------------------------------------------------------------
/** Specifies whether numbers should be parsed by this tokenizer.
 When true and parser encounters a word token that has the format of an
 integer or floating-point number, it treats the token as a
 number rather than a word, by setting the the <CODE>ttype</CODE>
 field to the value <CODE>VV_NUMBER</CODE> and putting the numeric
 value of the token into the <CODE>nval</CODE> field.

 @param pn if true, '0'..'9', '.' and '-' are considered digits,
           if false, the above are considered alphas
*/
void vvTokenizer::setParseNumbers(bool pn)
{
  for (int i='0'; i<='9'; ++i)
    ctype[i] = pn ? VV_DIGIT : VV_ALPHA;
  ctype[static_cast<int>('.')] = pn ? VV_DIGIT : VV_ALPHA;
  ctype[static_cast<int>('-')] = pn ? VV_DIGIT : VV_ALPHA;
  ctype[static_cast<int>('+')] = pn ? VV_DIGIT : VV_ALPHA;
}

//----------------------------------------------------------------------------
/** Determines whether or not ends of line are treated as tokens.
  If the flag argument is true, this tokenizer treats end of lines
  as tokens; the <CODE>nextToken</CODE> method returns
  <CODE>VV_EOL</CODE> and also sets the <CODE>ttype</CODE> field to
  this value when an end of line is read.<P>
  A line is a sequence of characters ending with either a
  carriage-return character (\r) or a newline character ('\n').
  In addition, a carriage-return character followed immediately
  by a newline character is treated as a single end-of-line token.<P>
  If the flag is false, end-of-line characters are
  treated as white space and serve only to separate tokens.

@param eol true indicates that end-of-line characters
are separate tokens, false indicates that
end-of-line characters are white space.
*/
void vvTokenizer::setEOLisSignificant(bool eol)
{
  eolIsSignificant = eol;
}

//----------------------------------------------------------------------------
/** Determines if word tokens are converted to upper or lowercase..
  @param cc force word tokens to be converted to given case
  @see CaseType
*/
void vvTokenizer::setCaseConversion(CaseType cc)
{
  caseConversion = cc;
}

//----------------------------------------------------------------------------
/** Causes the next call to the <CODE>nextToken</CODE> method of this
 tokenizer to return the current value in the <CODE>ttype</CODE>
 field, and not to modify the value in the <CODE>nval</CODE> or
 <CODE>sval</CODE> field.
*/
void vvTokenizer::pushBack()
{
  if (ttype != VV_NOTHING)                        // no-op if nextToken() not called
    pushedBack = true;
}

//----------------------------------------------------------------------------
/** Skip all further tokens until a line break occurs. The next call to
  nextToken() will return the first token after the line break.
  If the last token was ttype==VV_EOL, the entire next line will be
  skipped.
*/
void vvTokenizer::nextLine()
{
  int c;                                          // read character, or -1 for EOF

  if (peekChar!='\n'  &&  peekChar!='\r'  &&  peekChar>=0)
  {
    do                                            // skip until EOF or EOL
    {
      c = readChar();
    } while (c!='\n'  &&  c!='\r'  &&  c>=0);
  }
  else c = peekChar;
  if (c<0)                                        // EOF reached?
  {
    pushBack();
    return;
  }
  if (c=='\r')
  {
    c = readChar();
    if (c=='\n')
      c = readChar();
  }
  else c = readChar();
  ++line;
  peekChar = c;
}

//----------------------------------------------------------------------------
/** Parses the next token from the input file of this tokenizer.
  The type of the next token is returned in the <CODE>ttype</CODE>
  field. Additional information about the token may be in the
  <CODE>nval</CODE> field or the <CODE>sval</CODE> field of this
  tokenizer.<P>
  Typical clients of this class first set up the syntax tables
  and then sit in a loop calling nextToken to parse successive
  tokens until VV_EOF is returned.

  @return the value of the <CODE>ttype</CODE> field
*/
vvTokenizer::TokenType vvTokenizer::nextToken()
{
  int   c;                                        // read character, or -1 for EOF
  int   len;                                      // string length
  int   i;
  CharacterType ct;

  if (pushedBack)                                 // if token was pushed back, re-use the last read token
  {
    pushedBack = false;
    return ttype;
  }

  // Initialization:
  sval[0] = '\0';
  nval = 0.0f;

  if (firstPass)                                  // is this the first pass?
  {
    c = readChar();
    firstPass = false;
  }
  else
    c = peekChar;

  if (c<0 || c>255)
    return ttype = VV_EOF;

  ct = ctype[c];                                  // look up character properties

  // Parse whitespace:
  while (ct==VV_WHITESPACE)
  {
    if (c == '\r')
    {
      ++line;
      c = readChar();
      if (c == '\n')
        c = readChar();
      if (eolIsSignificant)
      {
        peekChar = c;
        return ttype = VV_EOL;
      }
    }
    else
    {
      if (c == '\n')
      {
        ++line;
        if (eolIsSignificant)
        {
          peekChar = readChar();
          return ttype = VV_EOL;
        }
      }
      c = readChar();
    }
    if (c<0 || c>255)
      return ttype = VV_EOF;
    ct = ctype[c];
  }

  // Parse comment:
  if (ct==VV_COMMENT)
  {
    while ((c = readChar())!='\n'  &&  c!='\r'  &&  c>=0)
    {
      // skip until EOF or EOL
    }
    if (c=='\r')
    {
      ++line;
      c = readChar();
      if (c=='\n') c = readChar();
    }
    else if (c=='\n')
    {
      ++line;
      c = readChar();
    }
    peekChar = c;
    return nextToken();                           // call self recursively
  }

  // Parse token:
  len = 0;
  while ((ct==VV_DIGIT || ct==VV_ALPHA) && len<MAX_TOKEN_LEN)
  {
    sval[len] = (char)c;
    ++len;
    c = readChar();
    ct = ctype[c];
  }
  sval[len] = '\0';
  peekChar = c;
  if (isNumberToken(sval))
  {
    nval = (float)atof(sval);
    return ttype = VV_NUMBER;
  }
  else                                            // word token
  {
    if (caseConversion!=VV_NONE)                  // check for case conversion
    {
      for (i=0; i<len; ++i)                       // convert each character
      {
        if (caseConversion==VV_UPPER)
          sval[i] = (char)toupper((int)sval[i]);
        else
          sval[i] = (char)tolower((int)sval[i]);
      }
    }
    return ttype = VV_WORD;
  }
}

//----------------------------------------------------------------------------
/** Finds out if the passed string is a number, i.e. all
  its characters are digits. Note: there might be false positives,
  e.g. "12-23...3" would be considered a number.<BR>
  If setParseNumbers() is set to false, this function will always
  return false.
  @param str string to check
  @return true if string is a number
*/
bool vvTokenizer::isNumberToken(char* str)
{
  int i;
  bool isNumber = true;

  for (i=0; i<int(strlen(str)); ++i)
  {
    if ((ctype[static_cast<int>(str[i])] & VV_DIGIT) == 0 && str[i]!='e' && str[i]!='E')
    {
      isNumber = false;
      break;
    }
  }
  return isNumber;
}

//============================================================================
// Functions for STANDALONE mode
//============================================================================

#ifdef VV_STANDALONE

#include <iostream>

using namespace std;

int main(int argc, char** argv)
{
  vvTokenizer* tokenizer;                         // ASCII file tokenizer
  vvTokenizer::TokenType ttype;                   // currently processed token type
  FILE* fp;                                       // volume file pointer
  bool done;

  if (argc!=2)
  {
    cout << "Tokenizer Test. Syntax: tokentext <filename.txt>" << endl;
    return(0);
  }

  if ( (fp = fopen(argv[1], "rb")) == NULL)
  {
    cout << "Error: Cannot open input file." << endl;
    return(0);
  }

  // Read file data:
  tokenizer = new vvTokenizer(fp);
  tokenizer->setCommentCharacter('#');
  tokenizer->setEOLisSignificant(false);
  tokenizer->setCaseConversion(vvTokenizer::VV_UPPER);
  tokenizer->setParseNumbers(true);
  tokenizer->setWhitespaceCharacter('=');
  done = false;
  while (!done)
  {
    // Read a token:
    ttype = tokenizer->nextToken();
    switch (ttype)
    {
      case vvTokenizer::VV_WORD:
        cout << "Line " << tokenizer->getLineNumber() <<
          ": Found word token: " << tokenizer->sval << endl;
        break;
      case vvTokenizer::VV_NUMBER:
        cout << "Line " << tokenizer->getLineNumber() <<
          ": Found number token: " << tokenizer->nval << endl;
        break;
      case vvTokenizer::VV_EOL:
        cout << "EOL" << endl;
        break;
      default: done = true;
      break;
    }
  }

  // Clean up:
  delete tokenizer;
  fclose(fp);
  return 1;
}
#endif

/////////////////
// End of File
/////////////////
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
