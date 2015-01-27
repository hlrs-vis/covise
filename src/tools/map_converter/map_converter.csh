#! /bin/tcsh -f
########################################################################
#
#    convert a whole directory or a directory hierarchy
#
#
#    Ralf Mikulla 21.02.2001
#                                             (C) Vircinity 2001
#
########################################################################


########################################################################
#                   	parse  command  line
########################################################################

if (! $?COVISEDIR ) then
   echo COVISEDIR has to be set
   exit 1
endif

set DIRSUFFIX="conv"
set TRANSLATIONS="${COVISEDIR}"/translations.txt
set FORCE=""

set xx =(`getopt ht:s:ofR: $* >& /dev/null`)

if ( $? != 0 ) then
	set ERRFLAG
endif



foreach i ($*)
    switch ($i)
	case -R:	    
 	    set RECURSIVE;
	    set RECDIR=$2; 
	    set FLAG=$i;
	    shift
	    shift
	    echo "recursive operation for dir" $RECDIR
	breaksw
	
	case -s:
	    set DIRSUFFIX=$2
	    shift
	    shift
	    echo "set suffix to" $DIRSUFFIX
	breaksw

	case -t:
	    set TRANSLATIONS=$2
	    shift
	    shift
	    echo "set translations to" $TRANSLATIONS
	breaksw	

        case -o:
            set OVERWRITE
            shift
            echo "converting in place"
        breaksw

        case -f:
            set FORCE=-f
            shift
            echo "overwriting existing files"
        breaksw

        case -h:
            shift
            set SHOWUSAGE
            set EXITCODE=0
	breaksw

	default:
#	   set ERRFLAG 
    endsw
end


if (( $1 == "" ) && ( $2 == "" )) then
   
else if ($2 == "") then
    set NONRECURSIVE
    set DIRIN=$1
    set DIROUT=$DIRIN"_"$DIRSUFFIX
else
    set NONRECURSIVE
    set DIRIN=$1
    set DIROUT=$2
endif


if ($?ERRFLAG) then
    echo "error in " $0
    set SHOWUSAGE
    set EXITCODE=2
endif

if ($?SHOWUSAGE) then
echo "Usage: `basename $0` [-h] [-t <translations>] [-s <dirsuffix>] [-o] [-f] [-R <directory>]\
       (recursively) convert all .net files\
\
  -t   location of translations file ($TRANSLATIONS)\
  -s   suffix for newly created directories ($DIRSUFFIX)\
  -R   directory to convert recursively\
  -o   convert maps in-place, over-writing old maps\
  -f   force overwriting of existing files\
  -h   print this message\
\
  Examples:\
       `basename $0` [-s <dirsuffix>] [-t <translation file>] <dir in> [dir out]\
          convert all net-files in <dir in> to current netfiles in <dir out>\
\
       `basename $0` [-s <dirsuffix>] [-t <translation file>] -R dir\
          convert recursively a hierarchy of directories starting at dir\
"

   exit $EXITCODE
endif



#
# non recursive 
#
if ($?NONRECURSIVE && ! $?OVERWRITE) then
    if (!(-d $DIROUT)) then           
        mkdir $DIROUT                 
        echo $0 " : created " $DIROUT 
    endif         

    foreach ii ($DIRIN/*.net)
	echo "converting " $ii
        if ($?OVERWRITE) then
           map_converter $FORCE -t$TRANSLATIONS -o $ii.new $ii > $ii.out && mv $ii.new $ii
        else
           map_converter $FORCE -t$TRANSLATIONS -o$DIROUT/`basename "$ii"` "$ii" > $DIROUT/`basename $ii`.out 
        endif
    end
                    
endif

#
# recursive mode
#
if ($?RECURSIVE) then

echo "start recursive operation"
set DIRIN=$RECDIR
set DIROUT=$DIRIN"_"$DIRSUFFIX

                                                                                                  
	  foreach ii ($DIRIN/*)                                                                   
	                                                                                          
	          set isnet = `echo $ii | egrep "\.net"`                                          
	          if (( -f $ii ) && ( $isnet != "" )) then                                        
                      if (!( -d $DIROUT )) then                                  
			echo "   --> create " $DIROUT  
                        if(! $?OVERWRITE) then
			    mkdir  $DIROUT
                        endif
		      endif                   

	              echo " converting " $ii 
                      if ($?OVERWRITE) then
                         map_converter $FORCE -t$TRANSLATIONS -o "$ii.new" "$ii" > "$ii".out && mv "$ii.new" "$ii"
                      else
                         map_converter $FORCE -t$TRANSLATIONS -o$DIROUT/`basename "$ii"` "$ii" > $DIROUT/`basename "$ii"`.out    
                      endif
	          endif                                                                           

	          if ((-d $ii) && (($ii != ".") || ($ii != ".."))) then                          
	            set DIROUT=$ii"_"$DIRSUFFIX 
	            echo "run in " $ii                                                            
	            $0 -t $TRANSLATIONS -R $ii -s $DIRSUFFIX                                                                        
	          endif                                                                           
          end                                                                                     
endif
     
exit	                                                                                          

