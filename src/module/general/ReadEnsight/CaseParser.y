/*
  Parser for EnSight case files
*/

%require "3.8"
%language "c++"

%locations

%define api.prefix {ensight}

%define parse.trace
%define parse.error detailed
%define parse.lac full

%header


%code provides {
typedef ensight::parser CaseParser;
// Give Bison token Type a readable name
typedef ensight::parser::value_type CaseTokenType;

#define YY_DECL int CaseLexer::scan(CaseTokenType *pToken)
}


// the includes
%code requires
{
#include "CaseFile.h"
#include "DataItem.h"

#include <string>
#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>

class CaseLexer;
class CaseParserDriver;

#define YYDEBUG 1
// define a constant for the maximum length of a string token
#define MaxTokenLength 1024
}

%code top
{
#include "CaseParserDriver.h"
#include "CaseLexer.h"

#define CERR std::cerr << "CaseParser::parse, at " << driver.location << ": "
}

%param { CaseParserDriver &driver }


%union
{
struct {
    char szValue[MaxTokenLength];
    long iVal;
    double dVal;
} token;
}



// work with untyped tokens

%token FORMAT_SEC TYPE GEOMETRY_SEC MODEL MEASURED MATCH CH_CO_ONLY VARIABLE_SEC CONSTANT COMPLEX
%token SCALAR VECTOR PER_CASE PER_NODE PER_ELEMENT TIME_SEC TIME_SET NUM_OF_STEPS
%token FN_ST_NUM FN_INCR
%token TIME_VAL IDENTIFIER POINT_IDENTIFIER INTEGER DOUBLE STRING IPADDRESS
%token VARDESC TENSOR_SYMM
%token ENSIGHTV ENSIGHT_GOLD ASTNOTFN FN_NUMS FLOAT
%token PER_M_NODE PER_M_ELEMENT
%token VAR_POST VAR_POST_TS VAR_INT
%token FILE_SEC FILE_SET

%left LOGICAL_OR
%left LOGICAL_AND

%left '+' '-'
%left '/' '*'
%left U_SUB

%%

//

ecase: section 
     | ecase section

section: sect_key  spec

sect_key: FORMAT_SEC 
        | GEOMETRY_SEC
        | VARIABLE_SEC
        | TIME_SEC
        | FILE_SEC

spec: spec_line
    | spec spec_line


spec_line: type_spec 
         | model_spec 
         | variable_spec
         | ts_spec
         | fs_spec
         | const_spec


ts_spec: ts_hdr ts_opts
         {
	     	     fprintf(stderr, "ts_hdr ts_opts %s %s\n",$<token>1.szValue, $<token>2.szValue);
         }
        |ts_spec ts_hdr ts_opts
         {
	     	     fprintf(stderr, "ts_hdr ts_opts %s %s\n",$<token>1.szValue, $<token>2.szValue);
         }

ts_hdr: TIME_SET INTEGER NUM_OF_STEPS INTEGER 
         {
	     int ts( $<token>2.iVal );
	     int ns( $<token>4.iVal );
	     	     fprintf(stderr, "DEFINITION TIMESET %d  STEPS %d\n", ts, ns);
	     driver.actTs_ = new TimeSet( ts, ns );

	 }
         | TIME_SET INTEGER IDENTIFIER NUM_OF_STEPS INTEGER 
         {
	     int ts( $<token>2.iVal );
	     int ns( $<token>5.iVal );
	          fprintf(stderr, "DEFINITION TIMESET %d  STEPS %d\n", ts, ns);
	     driver.actTs_ = new TimeSet( ts, ns );

	 } 

ts_opts: ts_fnum_sec ts_tval_secc
         | ts_tval_secc
         | ts_fnum_sec
         | ts_fn_start ts_fn_incr ts_tval_secc


ts_fn_start: FN_ST_NUM INTEGER
	{
	     int fs( $<token>2.iVal );
	     driver.tsStart_ = fs;
	     	     fprintf(stderr, " FIELNAME START %d ", fs);
	}

ts_fn_incr: FN_INCR INTEGER
	{
	    int incr( $<token>2.iVal ); 
		fprintf(stderr, " FIELNAME INCREMENT %d ", incr);
	    if (driver.actTs_ != nullptr) {
			int i=driver.tsStart_;
			for (int j=0; j<driver.actTs_->getNumTs(); ++j) {
				driver.actTs_->addFileNr(i);
				i += incr;
			}
			if ( driver.actTs_->getNumTs() == driver.actTs_->size() ) {
				// the time-set is full
				driver.caseFile_.addTimeSet( driver.actTs_ );
				driver.actTs_ = nullptr;		 
			}
	    }
	 }

ts_fnum_sec: FN_NUMS ts_fn_nums 
         {
	     	     fprintf(stderr, " FILENAME NUMBERS ");
	 }

ts_fn_nums: ts_fn_nums INTEGER
         {
		 
	     	     fprintf(stderr, " INTEGER ");
	     int nf( $<token>2.iVal ); 
	     if (driver.actTs_ != nullptr) {
		 driver.actTs_->addFileNr(nf); 		 	     
		 if ( driver.actTs_->getNumTs() == driver.actTs_->size() ) {
		     // the time-set is full
		     driver.caseFile_.addTimeSet( driver.actTs_ );
		     driver.actTs_ = nullptr;		 
		 }
	     }
	 }
         |  INTEGER
         {
	     	     fprintf(stderr, " INTEGER ");
	     int nf( $<token>1.iVal ); 
	     //	     fprintf(stderr, " %d ", nf);
	     if (driver.actTs_ != nullptr) {
		 driver.actTs_->addFileNr(nf); 
		 if ( driver.actTs_->getNumTs() == driver.actTs_->size() ) {
		     // the time-set is full
		     driver.caseFile_.addTimeSet( driver.actTs_ );
		     driver.actTs_ = nullptr;		 
		 }
	     }
	 }

ts_tval_secc: TIME_VAL ts_tvals
         {
	     	     fprintf(stderr, " TIME_VAL ts_tvals ");
	     if ( driver.actTs_ != nullptr ) {
	     }
	 }

ts_tvals: ts_tvals DOUBLE
         {
	     	     fprintf(stderr, " ts_tvals DOUBLE ");
	     float rt( (float) $<token>2.dVal );
	     if (driver.actTs_ != nullptr) {
		 driver.actTs_->addRealTimeVal(rt); 
	     }
	     else {
		 driver.actTs_ = driver.caseFile_.getLastTimeSet();
		 driver.actTs_->addRealTimeVal(rt); 
	     }
	 }
	 | ts_tvals INTEGER
         {
	     	     fprintf(stderr, " ts_tvals INTEGER ");
	     float rt( (float) $<token>2.iVal );
	     if (driver.actTs_ != nullptr) {
		 driver.actTs_->addRealTimeVal(rt); 
	     }
	     else {
		 driver.actTs_ = driver.caseFile_.getLastTimeSet();
		 driver.actTs_->addRealTimeVal(rt); 
	     }
	 }

        | DOUBLE
         {
	     	     fprintf(stderr, " DOUBLE ");
	     float rt( (float)$<token>1.dVal );
	     if (driver.actTs_ != nullptr) {
		 driver.actTs_->addRealTimeVal(rt); 
	     }
	     else {
		 driver.actTs_ = driver.caseFile_.getLastTimeSet();
		 driver.actTs_->addRealTimeVal(rt); 
	     }
	 }
        | INTEGER
         {
	     	     fprintf(stderr, " INTEGER ");
	     float rt( (float)$<token>1.iVal );
	     if (driver.actTs_ != nullptr) {
		 driver.actTs_->addRealTimeVal(rt); 
	     }
	     else {
		 driver.actTs_ = driver.caseFile_.getLastTimeSet();
		 driver.actTs_->addRealTimeVal(rt); 
	     }
	 }
	 
	 

fs_spec: fs_hdr
        |fs_spec fs_hdr
		
fs_hdr: FILE_SET INTEGER NUM_OF_STEPS INTEGER 
         {
	     int ts( $<token>2.iVal );
	     int ns( $<token>4.iVal );
	     	     fprintf(stderr, "DEFINITION FILESET %d  STEPS %d\n", ts, ns);
	     //driver.actTs_ = new TimeSet( ts, ns );

	 }
         | FILE_SET INTEGER IDENTIFIER NUM_OF_STEPS INTEGER 
         {
	     int ts( $<token>2.iVal );
	     int ns( $<token>5.iVal );
	     	     fprintf(stderr, "DEFINITION FILESET %d  STEPS %d\n", ts, ns);
	     //driver.actTs_ = new TimeSet( ts, ns );

	 } 


type_spec: TYPE ENSIGHTV
    {
	//	fprintf(stderr,"  ENSIGHT VERSION 6 found\n");
	driver.caseFile_.setVersion(CaseFile::v6);
    }
    | TYPE ENSIGHTV ENSIGHT_GOLD
    {
	//	fprintf(stderr,"  ENSIGHT GOLD found\n");
	driver.caseFile_.setVersion(CaseFile::gold);
    }


any_identifier: IDENTIFIER
    | POINT_IDENTIFIER
    | STRING

model_spec: MODEL any_identifier 
          {
        	std::string ensight_geofile($<token>2.szValue);
		//fprintf(stderr,"  ENSIGHT MODEL <%s> found\n", ensight_geofile.c_str());
	        driver.caseFile_.setGeoFileNm( ensight_geofile );
          }
          | MODEL INTEGER any_identifier 
          {
	      std::string ensight_geofile($<token>3.szValue);
          // check whether the integer is part of the filename
	  // this integer is separated by a whitespace from the filename thus no need to add it to the filename
	  // opening the file never works if you are not in the same directory as the case file
	  // INTEGER now requires a whitespace, see lexer
          /*struct stat buf;
          stat( ensight_geofile.c_str(), &buf);
          if( !S_ISREG(buf.st_mode) )
          {
             std::string intStr($<token>2.szValue); 
	         driver.caseFile_.setGeoFileNm( intStr+ensight_geofile );
          }
          else
          {*/
	      driver.caseFile_.setGeoFileNm( ensight_geofile );
	      int ts( $<token>2.iVal );
	      driver.caseFile_.setGeoTsIdx(ts);
          //}   
	      fprintf(stderr,"  ENSIGHT MODEL <%s> TIMESET <%d> found\n", ensight_geofile.c_str(), ts);
          }
          | MODEL INTEGER INTEGER any_identifier 
          {
	      std::string ensight_geofile($<token>4.szValue);
	      driver.caseFile_.setGeoFileNm( ensight_geofile );
	      int ts( $<token>2.iVal );
	      driver.caseFile_.setGeoTsIdx(ts);
	      //	      fprintf(stderr,"  ENSIGHT MODEL <%s> TIMESET <%d> found\n", ensight_geofile.c_str(), ts);
          }


          | MODEL INTEGER ASTNOTFN 
          {
	      std::string ensight_geofile($<token>3.szValue);
	      driver.caseFile_.setGeoFileNm( ensight_geofile );
	      int ts( $<token>2.iVal );
	      driver.caseFile_.setGeoTsIdx(ts);
	      fprintf(stderr,"  ENSIGHT MODEL <%s> TIMESET <%d> found\n", ensight_geofile.c_str(), ts);
          }
          // we may find lines like: model bla_geo.**** in this case we set the timeset to 1 
          | MODEL ASTNOTFN 
          {	      
	      std::string ensight_geofile($<token>2.szValue);
	      driver.caseFile_.setGeoFileNm( ensight_geofile );
	      driver.caseFile_.setGeoTsIdx(1);
	      fprintf(stderr,"  ENSIGHT MODEL <%s> TIMESET <%d> found\n", ensight_geofile.c_str(), 1);
          }


          | MEASURED any_identifier 
          {
        	std::string ensight_geofile($<token>2.szValue);
	        driver.caseFile_.setMGeoFileNm( ensight_geofile );
          }
          | MEASURED INTEGER any_identifier 
          {
	      std::string ensight_geofile($<token>3.szValue);
	      driver.caseFile_.setMGeoFileNm( ensight_geofile );
	      int ts( $<token>2.iVal );
	      driver.caseFile_.setGeoTsIdx(ts);
          }
          | MEASURED INTEGER INTEGER any_identifier 
          {
	      std::string ensight_geofile($<token>4.szValue);
	      driver.caseFile_.setMGeoFileNm( ensight_geofile );
	      int ts( $<token>2.iVal );
	      driver.caseFile_.setGeoTsIdx(ts);
          }
          | MEASURED INTEGER ASTNOTFN 
          {
	      std::string ensight_geofile($<token>3.szValue);
	      driver.caseFile_.setMGeoFileNm( ensight_geofile );
	      int ts( $<token>2.iVal );
	      driver.caseFile_.setGeoTsIdx(ts);
				fprintf(stderr,"  ENSIGHT MEASURED <%s> TIMESET <%d> found\n", ensight_geofile.c_str(), ts);
          }
          // we may find lines like: model bla_geo.**** in this case we set the timeset to 1 
          | MEASURED ASTNOTFN 
          {	      
	      std::string ensight_geofile($<token>2.szValue);
	      driver.caseFile_.setMGeoFileNm( ensight_geofile );
	      driver.caseFile_.setGeoTsIdx(1);
          }

        

variable_spec: var_pre IDENTIFIER STRING
	{
                    fprintf(stderr," var_pre STRING %s\n", $<token>2.szValue);
            std::string tmp($<token>2.szValue);
            size_t last = tmp.find(" ");
            if ( last == std::string::npos ) {
                last = tmp.find("\t");
                if ( last == std::string::npos ) {
                    CERR << "filename or description for variable missing" << std::endl;
                }
            }
            std::string desc( tmp.substr( 0, last ) );
            size_t len( tmp.size() );
            size_t snd( tmp.find_first_not_of(" ",last) );
            std::string fname( tmp.substr( snd, len-snd ) );
            //              fprintf(stderr," VAR_POST DE<%s>   FN<%s>\n", desc.c_str(), fname.c_str() );

            driver.actIt_->setDesc( desc );

            driver.actIt_->setFileName( boost::trim_copy(fname) );

            if ( driver.actIt_ != nullptr ) {
                driver.caseFile_.addDataIt( *driver.actIt_ );
            }
            else {
                CERR << "try to add nullptr DataItem" << std::endl;
            }

            delete driver.actIt_;
            driver.actIt_ = nullptr;

        }
		| var_pre VAR_POST
          {
	      	      fprintf(stderr," VAR_POST %s\n", $<token>2.szValue);
	      std::string tmp($<token>2.szValue);
	      size_t last = tmp.find(" ");
	      if ( last == std::string::npos ) {
		  last = tmp.find("\t");
		  if ( last == std::string::npos ) {
		      CERR << "filename or description for variable missing" << std::endl;
		  }
	      }
	      std::string desc( tmp.substr( 0, last ) );
	      size_t len( tmp.size() );
	      size_t snd( tmp.find_first_not_of(" ",last) );
	      std::string fname( tmp.substr( snd, len-snd ) );
	      //	      fprintf(stderr," VAR_POST DE<%s>   FN<%s>\n", desc.c_str(), fname.c_str() );
	      
	      driver.actIt_->setDesc( desc );

	      driver.actIt_->setFileName( boost::trim_copy(fname) );
	      
	      if ( driver.actIt_ != nullptr ) {
		  driver.caseFile_.addDataIt( *driver.actIt_ );
	      }
	      else {
		  CERR << "try to add nullptr DataItem" << std::endl;
	      }

	      delete driver.actIt_;
	      driver.actIt_ = nullptr;

	  }
          | var_pre VAR_INT VAR_POST
          {
	      	      fprintf(stderr," VAR_INT VAR_POST_TS %s\n", $<token>3.szValue);
	      std::string tmp($<token>3.szValue);
	      size_t last = tmp.find(" ");
	      if ( last == std::string::npos ) {
		  last = tmp.find("\t");
		  if ( last == std::string::npos ) {
		      CERR << "filename or description for variable missing" << std::endl;
		  }
	      }
	      std::string desc( tmp.substr( 0, last ) );
	      size_t len( tmp.size() );
	      size_t snd( tmp.find_first_not_of(" ",last) );
	      std::string fname( tmp.substr( snd, len-snd ) );
	      //	      fprintf(stderr," VAR_POST DE<%s>   FN<%s>\n", desc.c_str(), fname.c_str() );
	      
	      driver.actIt_->setDesc( desc );

	      driver.actIt_->setFileName( boost::trim_copy(fname) );
	      
	      if ( driver.actIt_ != nullptr ) {
		  driver.caseFile_.addDataIt( *driver.actIt_ );
	      }
	      else {
		  CERR << "try to add nullptr DataItem" << std::endl;
	      }

	      delete driver.actIt_;
	      driver.actIt_ = nullptr;

	  }
          | var_pre VAR_INT VAR_INT VAR_POST
          {
	      	      fprintf(stderr," VAR_INT VAR_POST_TS %s\n", $<token>4.szValue);
	      std::string tmp($<token>4.szValue);
	      size_t last = tmp.find(" ");
	      if ( last == std::string::npos ) {
		  last = tmp.find("\t");
		  if ( last == std::string::npos ) {
		      CERR << "filename or description for variable missing" << std::endl;
		  }
	      }
	      std::string desc( tmp.substr( 0, last ) );
	      size_t len( tmp.size() );
	      size_t snd( tmp.find_first_not_of(" ",last) );
	      std::string fname( tmp.substr( snd, len-snd ) );
	      //	      fprintf(stderr," VAR_POST DE<%s>   FN<%s>\n", desc.c_str(), fname.c_str() );
	      
	      driver.actIt_->setDesc( desc );

	      driver.actIt_->setFileName( boost::trim_copy(fname) );
	      
	      if ( driver.actIt_ != nullptr ) {
		  driver.caseFile_.addDataIt( *driver.actIt_ );
	      }
	      else {
		  CERR << "try to add nullptr DataItem" << std::endl;
	      }

	      delete driver.actIt_;
	      driver.actIt_ = nullptr;

	  }
          | var_pre IDENTIFIER DOUBLE
          {
	      	      fprintf(stderr," var_type var_rela %s\n", $<token>3.szValue);
	  }

var_pre :  var_type var_rela 
          {
	      	      fprintf(stderr," var_type var_rela\n");
	  }

var_type: SCALAR
          {  
	      driver.actIt_ = new DataItem;
	      	      fprintf(stderr,"     ENSIGHT SCALAR VARIABLE ");
	      driver.actIt_->setType( DataItem::scalar );
	  }
          | VECTOR 
          {  
	      driver.actIt_ = new DataItem;
	      	      fprintf(stderr,"     ENSIGHT VECTOR VARIABLE ");
	      driver.actIt_->setType( DataItem::vector );
	  }
          | TENSOR_SYMM 
          {  
	      driver.actIt_ = new DataItem;
	      	      fprintf(stderr,"     ENSIGHT SYMMETRIC TENSOR VARIABLE ");
	      driver.actIt_->setType( DataItem::tensor );
	  }


var_rela: PER_ELEMENT 
          {  
	      driver.actIt_->setDataType(false);
	      driver.actIt_->setMeasured(false);
	      	      fprintf(stderr," PER ELEMENT DATA");
	  }
          | PER_NODE
          {  
	      driver.actIt_->setDataType(true);
	      driver.actIt_->setMeasured(false);
	      	      fprintf(stderr," PER NODE DATA");
	  }
          | PER_M_NODE
          {  
	      driver.actIt_->setDataType(true);
	      driver.actIt_->setMeasured(true);
	      	      fprintf(stderr," PER NODE DATA");
	  }
          | PER_M_ELEMENT
          {  
	      driver.actIt_->setDataType(false);
	      driver.actIt_->setMeasured(true);
	      	      fprintf(stderr," PER NODE DATA");
	  }
          | PER_CASE
          {  
	      //driver.actIt_->setDataType(DataItem::PerCase);
	      	      fprintf(stderr," PER CASE ");
	  }

const_spec: CONSTANT var_rela POINT_IDENTIFIER INTEGER
          | CONSTANT var_rela POINT_IDENTIFIER DOUBLE
          | CONSTANT var_rela IDENTIFIER DOUBLE
          | CONSTANT var_rela IDENTIFIER INTEGER
          | CONSTANT var_rela INTEGER POINT_IDENTIFIER INTEGER
          | CONSTANT var_rela INTEGER POINT_IDENTIFIER DOUBLE
          | CONSTANT var_rela INTEGER IDENTIFIER INTEGER
          | CONSTANT var_rela INTEGER IDENTIFIER DOUBLE


%%
// end of rule definition
