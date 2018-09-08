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

#ifndef VV_TOKENIZER_H
#define VV_TOKENIZER_H

#include "vvexport.h"

#include <fstream>

/**
 * This class takes an input file and parses it into "tokens",
 * allowing the tokens to be read one at a time.
 * The parsing process is controlled by a number of flags
 * that can be set to various states. The
 * tokenizer can recognize identifiers, numbers, and comments.<P>
 *
 * In addition, an instance has several flags. These flags indicate:
 * <UL>
 * <LI>Whether line terminators are to be returned as tokens or treated
 *     as white space that merely separates tokens.
 * <LI>Whether comments are to be recognized and skipped.
 * <LI>Whether the characters of identifiers are converted to lowercase.
 * <LI>Whether numbers are to be parsed or returned as strings.
 * </UL><P>
 *
 * A typical application first constructs an instance of this class,
 * sets up the flags, and then repeatedly loops calling the
 * <CODE>nextToken</CODE> method in each iteration of the loop until
 * it returns the value <CODE>VV_EOF</CODE>.
 *
 * The default character properties are:<UL>
 * <LI>ASCII codes 0 to 32 and 127 to 191 are whitespace (CT_WHITESPACE)</LI>
 * <LI>All other ASCII codes are alpha characters (CT_ALPHA)</LI></UL>
 *
 * Here is an example code snippet which uses the Tokenizer class to search for
 * the word "FOUND" in the file "testfile.txt" ignoring comments starting with '#':
 * <PRE>
 * vvTokenizer::TokenType ttype;
 * std::ifstream file("testfile.txt");
 * vvTokenizer tokenizer(file);
 * tokenizer.setCommentCharacter('#');
 * tokenizer.setEOLisSignificant(false);
 * tokenizer.setCaseConversion(vvTokenizer::VV_UPPER);
 * tokenizer.setParseNumbers(true);
 * while ((ttype = tokenizer->nextToken()) != vvTokenizer::VV_EOF)
 *   if (strcmp(tokenizer.sval, "FOUND")==0)
 *     break;
 * </PRE>
 * @author Juergen Schulze
 */
class VIRVO_FILEIOEXPORT vvTokenizer
{
  public:
    enum TokenType                                /// token types
    {
      VV_EOF,                                     ///< end of file has been read
      VV_EOL,                                     ///< end of line has been read
      VV_NUMBER,                                  ///< a number token has been read
      VV_WORD,                                    ///< a word token has been read
      VV_NOTHING                                  ///< no token has been read (used for initialization)
    };

    enum CaseType                                 /// attributes for setCaseConversion()
    {
      VV_NONE,                                    ///< no case conversion
      VV_UPPER,                                   ///< force uppercase letters
      VV_LOWER                                    ///< force lowercase letters
    };

  private:
    static const int BLOCK_SIZE;                  ///< data block size to be read at once [bytes]
    static const int MAX_TOKEN_LEN;               ///< maximum token length [characters]
    enum CharacterType                            /// character types for character parsing
    {
      VV_WHITESPACE = 0,                          ///< whitespace character: to be ignored, separates words and numbers
      VV_DIGIT      = 1,                          ///< character numbers can start with
      VV_ALPHA      = 2,                          ///< character words start with
      VV_COMMENT    = 3                           ///< character comments start with
    };
    std::ifstream& file;                          ///< input file
	std::streamsize   blockUsed;                              ///< number of data block bytes used
    char* data;                                   ///< raw data read from input file
    int   cur;                                    ///< index of current data byte
    int   peekChar;                               ///< next character in input file
    bool  pushedBack;                             ///< true to not read the same token again
    int   line;                                   ///< line number of last token read
    CaseType caseConversion;                      ///< case type to convert alpha values to
    bool  eolIsSignificant;                       ///< true if EOL is returned as a token
    CharacterType ctype[256];                     ///< character type list
    bool  firstPass;                              ///< true = first parsing pass

    int  readChar();
    bool isNumberToken(char*);

  public:
    TokenType ttype;                              ///< After a call to the <CODE>nextToken</CODE> method, this field
    ///< contains the type of the token just read.
    ///< Its value is one of the following:
    ///< <UL>
    ///< <LI><CODE>VV_WORD</CODE> indicates that the token is a word.
    ///< <LI><CODE>VV_NUMBER</CODE> indicates that the token is a number.
    ///< <LI><CODE>VV_EOL</CODE> indicates that the end of line has been read.
    ///<     The field can only have this value if the
    ///<     <CODE>eolIsSignificant</CODE> method has been called with the
    ///<     argument <CODE>true</CODE>.
    ///< <LI><CODE>VV_EOF</CODE> indicates that the end of the input file
    ///<     has been reached.
    ///< </UL>

    char* sval;                                   ///< If the current token is a word or number token token,
    ///< this field contains a string giving the characters of
    ///< the word or number token.
    ///< The current token is a word when the value of the
    ///< <CODE>ttype</CODE> field is <CODE>VV_WORD</CODE>,
    ///< it is a number if the value is <CODE>VV_NUMBER</CODE>,

    float nval;                                   ///< If the current token is a number, this field contains the value
    ///< of that number. The current token is a number when the value of
    ///< the <CODE>ttype</CODE> field is <CODE>VV_NUMBER</CODE>.

    vvTokenizer(std::ifstream& file);
    ~vvTokenizer();
    int  determineCurrentLine();
    void setDefault();
    void setLineNumber(int);
    int  getLineNumber();
    long getFilePos();
    void setFilePos(std::streampos pos);
    void setAlphaCharacter(char);
    void setCommentCharacter(char);
    void setWhitespaceCharacter(char);
    void setParseNumbers(bool);
    void setEOLisSignificant(bool);
    void setCaseConversion(CaseType);
    void pushBack();
    void nextLine();
    TokenType nextToken();
};
#endif

// EOF
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
