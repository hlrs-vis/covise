//-*-Mode: C++;-*-
/*

  Parser for Tracer initial points

*/

// the includes
%header{
#include "covise.h"
#include "ApplInterface.h"
#include <alloca.h>

class PointsLexer;


#define register

// define a constant for the maximum length of a string token
#define YYDEBUG 1
%}

%name PointsParser
%define USE_CONST_TOKEN 1

%define CONSTRUCTOR_PARAM const string &initPoints
%define CONSTRUCTOR_INIT : lexer_( NULL ),isOK_( true )
%define CONSTRUCTOR_CODE init( initPoints );

%define MEMBERS \
  public:   virtual   ~PointsParser(); \
  public:   bool      IsOK() const; \
  public:   void      getPoints(float **x_ini,float **y_ini,float **z_ini); \
  public:   int      getNoPoints(); \
  private:  PointsLexer *lexer_; \
  private:  istringstream *inputStream_; \
  private:  vector<float> coordinates_; \
  private:  bool      isOK_;\
  private:  int init( const string &initPoints ); \

%union
{  
    struct {
        char szValue;
        float fVal; 
    } token;

}

%header{
  // Give Bison-tokentType a readable name
  typedef YY_PointsParser_STYPE TokenType;

%}

%token OPEN_POINT CLOSE_POINT COORDINATE SEPARATOR OPEN_POINT_PAR CLOSE_POINT_PAR OPEN_POINT_MATLAB CLOSE_POINT_MATLAB

%%

points:  point
      |  point points
      |  point SEPARATOR points

point:   OPEN_POINT coordinates CLOSE_POINT
      |  OPEN_POINT_PAR coordinates CLOSE_POINT_PAR
      |  OPEN_POINT_MATLAB coordinates CLOSE_POINT_MATLAB

coordinates:  COORDINATE SEPARATOR COORDINATE SEPARATOR COORDINATE 
       {
          coordinates_.push_back( $<token>1.fVal );
          coordinates_.push_back( $<token>3.fVal );
          coordinates_.push_back( $<token>5.fVal );
       }
           |  COORDINATE COORDINATE COORDINATE
       {
          coordinates_.push_back( $<token>1.fVal );
          coordinates_.push_back( $<token>2.fVal );
          coordinates_.push_back( $<token>3.fVal );
       }

%%

#include "PointsLexer.h"

int
PointsParser::init(const string &initPoints)
{
   inputStream_ = new istringstream( initPoints.c_str() );
   lexer_ = new PointsLexer( inputStream_ );
   lexer_->set_debug( 0 );
   return( 0 );
}

PointsParser::~PointsParser()
{
    delete lexer_;

    if( inputStream_ && (*inputStream_) ) {
        delete inputStream_;
    }
}

int
PointsParser::yylex()
{
    return( lexer_->scan( &yylval ) );
}

void
PointsParser::yyerror( char *szErrMsg )
{
    isOK_ = false;
    string mssg(szErrMsg);
    char buf[256];
    sprintf(buf,", %d points had been read when an error was detected.",coordinates_.size()/3);
    mssg += buf;
    Covise::send_error(mssg.c_str());
}

bool
PointsParser::IsOK() const
{
   return isOK_;
}

void
PointsParser::getPoints(float **x_ini,float **y_ini,float **z_ini)
{
   assert(coordinates_.size() % 3 == 0);
   int no_points = coordinates_.size()/3;
   *x_ini = new float[no_points];
   *y_ini = new float[no_points];
   *z_ini = new float[no_points];
   int point;
   vector<float>::iterator p = coordinates_.begin();
   for(point=0;point<no_points;++point){
      (*x_ini)[point] = *p; ++p;
      (*y_ini)[point] = *p; ++p;
      (*z_ini)[point] = *p; ++p;
   }
}

int
PointsParser::getNoPoints()
{
   assert(coordinates_.size() % 3 == 0);
   return (coordinates_.size()/3);
}

