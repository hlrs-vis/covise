#!/bin/sh

#When debugging set  DEBUG=1
#For installations set  DEBUG=0
DEBUG=1

case `echo -n x` in x) N='-n' C= ;; *) N= C='\c' ;; esac

debug_echo ()
{
   if [ "$DEBUG" = "1" ] ; then
      echo $@
   fi
}

###############################################################
# covinst.sh  -- COVISE installation script.                  #
###############################################################

######################################################################
### this line is automatically adjusted by covadmin - do not touch ###
COVISE_VERSION="2014.11"
######################################################################

echo
echo   "*******************************************************"
printf "* COVISE v%7s Installation script                 *\n" $COVISE_VERSION
echo   "*                                                     *"
echo   "* (c) 1997-2015 HLRS, University of Stuttgart         *"
echo   "* (c) 2001-2002 VirCinity GmbH, Stuttgart             *"
echo   "* (c) 2004-2008 ZAIK, University of Cologne           *"
echo   "*******************************************************"
echo

### use ARCHSUFFIX var if already set
case x$ARCHSUFFIX in
   x)
      ARCHSUFFIX="unknown"
      ARCH=`uname`
      case "$ARCH" in
         HP-UX)	ARCHSUFFIX=hp
            case `uname -a` in HP-UX*10.20*)
               ARCHSUFFIX=hp1020
         esac
         ;;
      SunOS) 	ARCHSUFFIX=sparc	;;
   AIX)   	ARCHSUFFIX=aix  	;;
OSF1)  	ARCHSUFFIX=dec  	;;
       SUPER-UX)	ARCHSUFFIX=nec  	;;
    Linux)
       mach=`uname -m`
       case "$mach" in
          ia64)
             export ARCHSUFFIX=IA64
             ;;
          *)
             if [ -f /lib/libgcc_s.so ]
             then
                export ARCHSUFFIX=gcc3
             else
                export ARCHSUFFIX=linux
             fi
             ;;
       esac
       ;;
    IRIX)  	ARCHSUFFIX=sgin32	;;
 IRIX64)	ARCHSUFFIX=sgin32	;;
     esac

  esac

  ARCH=`uname -m`
  case "$ARCH" in
     "CRAY Y-MP") ARCHSUFFIX=ymp 	;;
  "CRAY C90")  ARCHSUFFIX=c90 	;;
  esac

  ### detect the directory where the install file is
  BASEDIR=`dirname $0`

  ### get a full, complete and direct path with pwd
  BASEDIR=`(cd "$BASEDIR" ; /bin/pwd)`
  NUMDIST=`/bin/ls -dr DIST.*/Platform | cut -f2 -d. | cut -f1 -d/ | wc -l`
  echo NUMDIST is $NUMDIST
  DISTCOUNT=`expr 10 + $NUMDIST`
  echo DISTCOUNT is $DISTCOUNT
  ## continue in the end to be able to start menu_main()


  ##############################################################

  ##############################################################

  menu_main()
  {
     if [ "$avail_dist" = "linked" ]
     then
        install_linked
     else
        ans='none'
        while [ "$ans" != "0" ] ; do
           echo "-----------------------------------------------"
           echo " Choose installation type                      "
           echo "-----------------------------------------------"
           echo "  1) Private Single User Installation          "
           echo "  2) Global Multi-User Installation            "
           echo "-----------------------------------------------"
           echo "  0) Exit                                      "
           echo "-----------------------------------------------"
           read ans
           echo "                                               "
           case $ans in
              0) break		;;
           1) install_private	;;
        2) install_global	;;
     *) echo ""; echo "Invalid choice!"; echo "" ;;
  esac
         done
      fi
   }

   install_private()
   {
      COVISEDIR=$HOME/covise

      ans='none'
      while [ "$ans" != "y" -a  "$ans" != "n" ]
      do
         echo "---------------------------------------------"
         echo " Private Single User Installation            "
         echo " COVISE will be installed in                 "
         echo " $COVISEDIR for user `id -un`              "
         echo " ok to continue (y/n)?                  "
         echo "---------------------------------------------"
         read ans
         echo " "

         if [ "$ans" = "n" ]
         then 
            echo "INFO: leaving installation script"
            exit
         fi 
      done

      #   check for existing installation     
      ans='none'
      if [ -d $COVISEDIR ]
      then

         while [ "$ans" != "y" -a  "$ans" != "n" ]
         do
            oldsuffix=`date +'%d-%b-%Y' | sed 's/ //g'`
            echo "---------------------------------------------"
            echo " Private Single User Installation            "
            echo " COVISE will be installed in                 "
            echo " $COVISEDIR                                "
            echo "---------------------------------------------"
            echo " WARNING: directory $COVISEDIR exists!     "
            echo " do you want to move it in $COVISEDIR.$oldsuffix (y/n)?"
            echo "---------------------------------------------"
            echo "                                             "
            read ans
            echo " "       
            if [ "$ans" = "y" ]
            then
               echo " Moving $COVISEDIR to $COVISEDIR.$oldsuffix"
               mv $COVISEDIR $COVISEDIR.$oldsuffix	
               if [ -e $HOME/COvise ]; then
                  echo " moving $HOME/COvise to $HOME/COvise.$oldsuffix"
                  mv $HOME/COvise $HOME/COvise.$oldsuffix
               fi
            fi
         done
      fi



      option='none'
      done1=" "
      done2=" "
      done3=" "
      done4=" "
      done5=" "
      done6=" "
      done7=" "
      done8=" "

      while true
      do
         READ=1

         echo "-------------------------------------------------"
         echo " Private Installation"
         echo " COVISE will be installed in"
         echo " $COVISEDIR"
         echo ""
         echo " At least option 1 and one of the platform options is necessary"
         echo "-------------------------------------------------"
         echo "      Platform independent files"
         echo " [$done1]  1)  Install required shared files (bitmaps etc.)"
         echo " [$done2]  2)  Install tutorial maps and data"
         echo " [$done3]  3)  Install example maps and data"
         echo " [$done4]  4)  Install documentation"
         echo "      9)  Install all platform-independent files (1-5)"
         echo "-------------------------------------------------"
         echo "      Platform dependent files"
         num=10
         set $avail_dist
         for dist in $avail_dist ; do
            if [ -x $COVISEDIR/$1/bin/covise ]
            then
               printf " [+] %2d)  " $num
            else
               printf " [ ] %2d)  " $num
            fi
            grep '^DESC' "$BASEDIR"/DIST.$dist/Platform | cut -f2- -d' '
            num=`echo 1 + $num | bc -l`
            shift
         done
         echo "     99)  Install all platforms"
         echo "-------------------------------------------------"
         echo "    999)  Install everything"
         echo "-------------------------------------------------"
         echo "      0)  Return to main menu                      "
         echo

         if [ "$READ" = "1" ]
         then
            read option
         else
            option='none'
         fi

         case $option in

            0)
               if [ "$done1" = " " -a "$ARCH_INSTALLED" = "yes" ]; then
                  echo "You still have to install option 1 (required shared files)"
                  sleep 1
               else
                  break
               fi
               ;;

            1)  adjust_dot_files
               install_runtime_shared
               done1="+"
               ;;


            2)  install_tutorial_maps_and_data
               done2="+"
               ;;

            3)  install_example_maps_and_data
               done3="+"
               ;;

            4)  extract_doc
               done4="+"
               ;;


            9|999)  adjust_dot_files
               install_runtime_shared
               install_tutorial_maps_and_data
               install_example_maps_and_data
               extract_doc
               done1="+"
               done2="+"
               done3="+"
               done4="+"
               done5="+"

               ;;

         esac

         if [ $option -gt 9 -a  $option -lt $DISTCOUNT ]
         then
            set $avail_dist
            while [ $option -gt 10 ] ; do
               shift
               option=`echo $option - 1 | bc -l`
            done
            echo "Install binary for $1"
            #exit
            ARCHSUFFIX=$1
            install_runtime_arch

         fi
         if [ "$option" = 99 -o "$option" = 999 ]
         then
            for ARCHSUFFIX in $avail_dist ; do
               install_runtime_arch
            done
         fi
      done
   }

   install_runtime_arch()
   {
      echo "Installing binary for $ARCHSUFFIX..."
      ARCH_INSTALLED=yes
      install_runtime_core
      #   install_runtime_general_modules
      #   install_runtime_example_modules

      #   INSTALLED_RENDERERS=""
      #   for REND in inventor pf osg coin sg; do
      #      install_runtime_renderer
      #   done

      #   if [ "$INSTALLED_RENDERERS" = "" ]; then
      #      echo WARNING: No renderer installed -- you probably want at least one
      #   fi
   }

   install_global()
   {
      # ask where to install

      echo "---------------------------------------------"
      echo " Global Multi-User Installation              "
      echo "---------------------------------------------"
      echo " Enter directory to create covise_$COVISE_VERSION in"
      echo $N " [/usr/local] : $C"

      read INSTPATH
      if [ "$INSTPATH" = "" ]
      then
         INSTPATH=/usr/local
      fi

      INSTDIR=$INSTPATH/covise_$COVISE_VERSION

      if [ ! -d $INSTPATH ]
      then
         echo "INFO: $INSTPATH does not exist. Making directory $INSTPATH"
         mkdir -p $INSTPATH
         if  [ ! -d $INSTPATH ]
         then
            echo "ERROR: could not create directory $INSTPATH. Leaving..."
            exit
         fi
      fi

      if [ ! -d $INSTDIR ]
      then
         mkdir $INSTDIR
      fi

      if [ ! -d $INSTDIR ]
      then
         echo "---------------------------------------------"
         echo " Error: could not create directory $INSTDIR "
         exit
      fi


      COVISEDIR=$INSTDIR/covise

      ### copy the installation script
      if [ ! -f $INSTDIR/covinst.sh ]
      then
         cp "$BASEDIR"/covinst.sh $INSTDIR
      fi

      echo "#! /bin/sh" > $INSTDIR/covise-vars.sh
      echo >> $INSTDIR/covise-vars.sh
      echo "export COVISE_PATH=" >> $INSTDIR/covise-vars.sh
      echo "export ARCHSUFFIX=" >> $INSTDIR/covise-vars.sh
      echo "COVISEDIR=$COVISEDIR" >> $INSTDIR/covise-vars.sh
      echo 'PATH=${COVISEDIR}/bin:${PATH}' >>$INSTDIR/covise-vars.sh
      echo "export COVISEDIR PATH" >> $INSTDIR/covise-vars.sh

      echo "#! /bin/csh" > $INSTDIR/covise-vars.csh
      echo >> $INSTDIR/covise-vars.csh
      echo "unsetenv COVISE_PATH" >> $INSTDIR/covise-vars.csh
      echo "unsetenv ARCHSUFFIX" >> $INSTDIR/covise-vars.csh
      echo "setenv COVISEDIR $COVISEDIR" >> $INSTDIR/covise-vars.csh
      echo 'set path=(${COVISEDIR}/bin $path)' >>$INSTDIR/covise-vars.csh

      ### shared already existent?
      if [ -d $INSTDIR/covise/share/covise/icons ] ; then 
         done1="+"
      else
         done1=" "
      fi

      ### example modules already existent?
      if [ -d $INSTDIR/covise/share/covise/example-data/general/tutorial ] ; then 
         done2="+"
      else
         done2=" "
      fi

      ### tutorial maps/data already existent?
      if [ -d $INSTDIR/covise/data/covise/share/example-data/general/examples ] ; then 
         done3="+"
      else
         done3=" "
      fi

      ### docu already existent?
      if [ -d $INSTDIR/covise/doc ] ; then 
         done4="+"
      else
         done4=" "
      fi

      # @@@@@@@
      option='none'
      while true
      do
         READ=1
         echo "-------------------------------------------------"
         echo " Global Multi-User Installation in $INSTDIR"
         echo "-------------------------------------------------"
         echo ""
         echo " At least option 1 and one of the platform options is necessary"
         echo "-------------------------------------------------"
         echo "      Platform independent files"
         echo " [$done1]  1)  Install required shared files"
         echo " [$done2]  2)  Install tutorial maps and data"
         echo " [$done3]  3)  Install example maps and data"
         echo " [$done4]  4)  Install documentation"
         echo "      9)  Install all platform-independent files (1-5)"
         echo "-------------------------------------------------"
         echo "      Platform dependent files"
         num=10
         set $avail_dist
         for dist in $avail_dist ; do
            if [ -x $INSTDIR/covise/$1/bin/covise ]
            then
               printf " [+] %2d)  " $num
            else
               printf " [ ] %2d)  " $num
            fi
            grep '^DESC' "$BASEDIR"/DIST.$dist/Platform | cut -f2- -d' '
            num=`echo 1 + $num | bc -l`
            shift
         done
         echo "     99)  Install all platforms"
         echo "-------------------------------------------------"
         echo "    999)  Install everything"
         echo "-------------------------------------------------"
         echo "      0)  Finish installation and quit"
         echo

         if [ "$READ" = "1" ]
         then
            read option
         else
            option='none'
         fi

         if [ "$option" = "0" ]
         then
            if [ "$done1" = " " -a "$ARCH_INSTALLED" = "yes" ]; then
               echo "You still have to install option 1 (required shared files)"
               sleep 1
            else
               break
            fi
         fi

         if [ "$option" = "1" ]
         then
            install_runtime_shared
            done1="+"
         fi


         if [ "$option" = "2" ]
         then
            install_tutorial_maps_and_data
            done2="+"
         fi

         if [ "$option" = "3" ]
         then
            install_example_maps_and_data
            done3="+"
         fi


         if [ "$option" = "4" ]
         then
            extract_doc
            done4="+"
         fi


         if [ "$option" = "9" -o "$option" = "999" ]
         then
            install_runtime_shared            ; done1="x"
            install_tutorial_maps_and_data    ; done2="x"
            install_example_maps_and_data     ; done3="x"
            extract_doc                       ; done4="x"
         fi


         if [ $option -gt 9 -a  $option -lt $DISTCOUNT ]
         then
            set $avail_dist
            while [ $option -gt 10 ] ; do
               shift
               option=`echo $option - 1 | bc -l`
            done
            #exit
            ARCHSUFFIX=$1
            install_runtime_arch
         fi
         if [ "$option" = 99 -o "$option" = 999 ]
         then
            for ARCHSUFFIX in $avail_dist ; do
               install_runtime_arch
               echo Installing For $ARCHSUFFIX
            done
         fi
      done

      ### copy covise.config.base to covise.config
      #cp $INSTDIR/covise/.covise-base $INSTDIR/covise/.covise
      #cp $INSTDIR/covise/.common-base $INSTDIR/covise/.common

      PERM_FLAG="none"
      while [ "$PERM_FLAG" != "w" -a "$PERM_FLAG" != "g" ]
      do
         echo "Shall the installation be (w)orld or (g)roup readable? "
         read PERM_FLAG
      done

      echo "recursively changing the permissions of $INSTDIR"
      if [ "$PERM_FLAG" = "w" ]
      then
         echo "to r-xr-xr-x"
         chmod -R 555 $INSTDIR 2>/dev/null
         cd $INSTDIR
         find . -type d -exec chmod u+w {} \;
      else
         echo "to r-xr-x---"
         chmod -R 550 $INSTDIR 2>/dev/null
         cd $INSTDIR
         find . -type d -exec chmod u+w {} \;
      fi    

      echo "-------------------------------------------------"
      echo "Finished global installation"
      echo " "
      echo "    source $INSTDIR/covise-vars.{csh,sh}"
      echo "(depending on your shell) for using covise."
      echo " "
      echo "-------------------------------------------------"
      echo " "
      exit 0    
   }


   #
   # for install_linked only:
   #
   # copy_includes_for_linked() recursively copies all env files which are
   # included by make.env.$ARCHSUFFIX from $GLOBALINSTDIR to $COVISEDIR
   # (I suppose $COVISEDIR would be better, but history...).
   #
   copy_includes_for_linked() {
      grep "^include " $envfile | while read include ; do
      ERRMSG=
      # $include will be s.th. like 'include $(COVISEDIR)/src/make.env.sgi'
      # we have to remove the brackets and any comment
      # and replace COVISEDIR by GLOBALINSTDIR/covise
      include=`echo "$include"	\
         | sed	-e 's/[()]//g'	\
         -e 's/#.*$//'	\
         -e 's;COVISEDIR;GLOBALINSTDIR/covise;'`
      set x $include; shift
      case $# in
         2) eval include="$2"
            if [ -f $include ] ; then
               cp -f "$include" "$COVISEDIR/src/"
               case $? in
                  0) envfile="$include" copy_includes_for_linked ;;
               *) ERRMSG="Copy for line '$*' in '$envfile' failed." ;;
            esac

         else
            ERRMSG="'$include':  No such file."
         fi
         ;;
      *) ERRMSG="Invalid syntax (word count is $#, has to be 2)."
         ;;
   esac
   case x$ERRMSG in x) ;; *)
      echo ""
      echo "--> Unable to copy included env file from global to private src directory."
      echo "--> Line '$*' in '$envfile':"
      echo "--> $ERRMSG"
      echo "--> Please check include and copy manually."
      echo ""
      echo $N "- Press <RETURN> to continue - $C"; read answer < /dev/tty
esac
    done
 }


 install_linked()
 {
    COVISEDIR=$HOME/covise
    ans='none'
    while [ "$ans" != "y" -a  "$ans" != "n" ]
    do
       echo " Linked  Installation for user `id -un`"
       echo "" 
       echo $N " ok to proceed (y/n)? $C"
       read ans
       if [ "$ans" = "n" ]
       then 
          echo "INFO: leaving installation script"
          exit
       fi 
    done

    #   check for existing installation     
    ans='none'
    oldsuffix=`date +'%d-%b-%Y' | sed 's/ //g'`
    if [ -d $COVISEDIR ]
    then

       while [ "$ans" != "y" -a  "$ans" != "n" ]
       do
          echo "---------------------------------------------"
          echo " WARNING: directory $COVISEDIR exists!     "
          echo $N " do you want to move it in covise.$oldsuffix (y/n)? $C"
          read ans

          if [ "$ans" = "y" ]
          then
             echo " Moving $COVISEDIR to $COVISEDIR.$oldsuffix"
             mv $COVISEDIR $COVISEDIR.$oldsuffix	
             echo " moving $HOME/COvise to $HOME/COvise.$oldsuffix"
             mv $HOME/COvise $HOME/COvise.$oldsuffix
          fi
       done
    fi


    #  ask for the global installation directory

    echo "---------------------------------------------"
    echo "" 
    echo $N "Global Installation dir [$BASEDIR] : $C"
    read INSTDIR

    if [ "$INSTDIR" = "" ]
    then
       INSTDIR="$BASEDIR"

    fi

    GLOBALINSTDIR=$INSTDIR

    # check if there is a valid covise
    if [ ! -d $GLOBALINSTDIR ]
    then
       echo "ERROR: $GLOBALINSTDIR does not exist!"
       echo "...leaving the installation script."
       exit
    fi
    if [ ! -d $GLOBALINSTDIR/covise ]
    then
       echo "ERROR: $GLOBALINSTDIR/covise does not exist!"
       echo "...leaving the installation script."
       exit
    fi
    if [ ! -d $GLOBALINSTDIR/covise/$ARCHSUFFIX ]
    then
       if [ $ARCHSUFFIX = "sgin32" ] ; then
          if [ ! -d $GLOBALINSTDIR/covise/sgi64 ]; then
             echo "ERROR: $GLOBALINSTDIR/covise/sgi64 does not exist!"
             echo "as well as $GLOBALINSTDIR/covise/sgin32"
             echo "...leaving the installation script."
             exit
          fi
          ARCHSUFFIX=sgin64
       else
          if [ $ARCHSUFFIX = "linux" ] ; then
             if [ ! -d $GLOBALINSTDIR/covise/gcc3 ]; then
                echo "ERROR: $GLOBALINSTDIR/covise/linux does not exist!"
                echo "as well as $GLOBALINSTDIR/covise/gcc3"
                echo "...leaving the installation script."
                exit
             fi
             ARCHSUFFIX=gcc3
          else
             echo "ERROR: $GLOBALINSTDIR/covise/$ARCHSUFFIX does not exist!"
             echo "...leaving the installation script."
             exit

          fi

       fi
    fi


    # making a local covise directory
    if [ -d $COVISEDIR ]
    then
       echo "$COVISEDIR exists, installing into current installation"
    else
       echo " Making a local covise directory"
       mkdir $COVISEDIR
    fi

    # copy the configuration files
    if [ -f $COVISEDIR/config/config.xml ]
    then
       echo "$COVISEDIR/config/config.xml exists: NOT overwriting current config"
    else
       echo " Copying the configuration files"
       #cp $GLOBALINSTDIR/covise/.covise-base $COVISEDIR/.covise
    fi

    # making a local net and data directory
    # link the global tutorial and example networks and data

    echo " Making a local net and data directory"

    if [ -d $COVISEDIR/net ]
    then
       echo "$COVISEDIR/net exists"
    else
       mkdir $COVISEDIR/net
       if [ -d $GLOBALINSTDIR/covise/net/examples ]
       then
          echo " Making link for general maps"
          ln -s $GLOBALINSTDIR/covise/net/examples $COVISEDIR/net/examples
       fi
       if [ -d $GLOBALINSTDIR/covise/net/tutorial ]
       then
          echo " Making link for general maps"
          ln -s $GLOBALINSTDIR/covise/net/tutorial $COVISEDIR/net/tutorial
       fi

    fi
    if [ -e $COVISEDIR/extern_libs ] ; then 
       echo "$COVISEDIR/extern_libs exists"
    else
       echo " Making link for '.../covise/extern_libs'"
       ln -s $GLOBALINSTDIR/covise/extern_libs $COVISEDIR/extern_libs
    fi
    if [ -d $COVISEDIR/data ]
    then
       echo "$COVISEDIR/data exists"
    else
       mkdir $COVISEDIR/data
       if [ -d $GLOBALINSTDIR/covise/data/general ]
       then
          echo " Making link for general data"
          ln -s $GLOBALINSTDIR/covise/data/general $COVISEDIR/data/
       fi
    fi


    # adjust the COVISE_PATH environment variable
    echo " INFO: adjusting .covise"
    ACT_GLOBAL=`grep 'setenv COVISE_GLOBALINSTDIR' $COVISEDIR/.covise | awk '{print $3}'`

    if [ "$ACT_GLOBAL" = "" ]
    then
       chmod u+w $COVISEDIR/.covise
       mv -f $COVISEDIR/.covise $COVISEDIR/.covise.bak    
       sed -e "s+^setenv COVISE_GLOBALINSTDIR.*$+& $GLOBALINSTDIR+" $COVISEDIR/.covise.bak > $COVISEDIR/.covise
    else
       if [ "$ACT_GLOBAL" != "$GLOBALINSTDIR" ]
       then
          echo "Changing GLOBALINSTDIR from $ACT_GLOBAL to $GLOBALINSTDIR"
          chmod u+w $COVISEDIR/.covise
          mv -f $COVISEDIR/.covise $COVISEDIR/.covise.bak    
          sed -e "s+^setenv COVISE_GLOBALINSTDIR.*$+& $GLOBALINSTDIR+" $COVISEDIR/.covise.bak > $COVISEDIR/.covise
       else
          echo "GLOBALINSTDIR already set to $GLOBALINSTDIR"
       fi
    fi
    echo " INFO: adjusting .common"
    ACT_GLOBAL=`grep 'setenv COVISE_GLOBALINSTDIR' $COVISEDIR/.common | awk '{print $3}'`

    if [ "$ACT_GLOBAL" = "" ]
    then
       chmod u+w $COVISEDIR/.common
       mv -f $COVISEDIR/.common $COVISEDIR/.common.bak    
       sed -e "s+^setenv COVISE_GLOBALINSTDIR.*$+& $GLOBALINSTDIR+" $COVISEDIR/.common.bak > $COVISEDIR/.common
    else
       if [ "$ACT_GLOBAL" != "$GLOBALINSTDIR" ]
       then
          echo "Changing GLOBALINSTDIR from $ACT_GLOBAL to $GLOBALINSTDIR"
          chmod u+w $COVISEDIR/.common
          mv -f $COVISEDIR/.common $COVISEDIR/.common.bak    
          sed -e "s+^setenv COVISE_GLOBALINSTDIR.*$+& $GLOBALINSTDIR+" $COVISEDIR/.common.bak > $COVISEDIR/.common
       else
          echo "GLOBALINSTDIR already set to $GLOBALINSTDIR"
       fi
    fi

    # check if the development option is installed in the global installation
    if [ -d $GLOBALINSTDIR/covise/src/kernel/covise ] 
    then

       #   make a local src directory

       if [ -d $COVISEDIR/src ]
       then
          echo "There appears to be an existing src directory structure."
       else
          echo " Making a local src directory and linking header directories"
          mkdir $COVISEDIR/src
          #ln -s $GLOBALINSTDIR/covise/src/kernel/covise $COVISEDIR/src/kernel/covise
          #ln -s $GLOBALINSTDIR/covise/src/kernel/appl $COVISEDIR/src/kernel/appl
          #ln -s $GLOBALINSTDIR/covise/src/kernel/render $COVISEDIR/src/kernel/render
          #ln -s $GLOBALINSTDIR/covise/src/kernel/dmgr $COVISEDIR/src/kernel/dmgr
       fi

       if [ ! -d $COVISEDIR/$ARCHSUFFIX/bin ]
       then
          mkdir -p $COVISEDIR/$ARCHSUFFIX/bin
          #echo " Linking the libraries"
          #ln -s $GLOBALINSTDIR/covise/$ARCHSUFFIX/bin/lib* $COVISEDIR/$ARCHSUFFIX/bin
          cp $GLOBALINSTDIR/covise/src/make.env.$ARCHSUFFIX $COVISEDIR/src/
          envfile=$GLOBALINSTDIR/covise/src/make.env.$ARCHSUFFIX copy_includes_for_linked
          cp $GLOBALINSTDIR/covise/src/make.ident $COVISEDIR/src/
       fi

       if [ ! -f $COVISEDIR/src/make.rules ]
       then
          cp $GLOBALINSTDIR/covise/src/make.rules $COVISEDIR/src/
       fi

       if [ ! -d $COVISEDIR/src/application ]
       then
          echo " Making a local application directory"
          mkdir -p $COVISEDIR/src/application
          echo " Making a local renderer plugin directory"
          mkdir -p $COVISEDIR/src/renderer/COVER
          echo " Copying the programming examples"
          cp -R $GLOBALINSTDIR/covise/src/application/examples/ $COVISEDIR/src/application/examples/
          cp -R $GLOBALINSTDIR/covise/src/renderer/COVER/plugins/ $COVISEDIR/src/renderer/COVER/plugins/

       fi

       echo " Changing permissions of the programming examples"
       chmod -R u+w $COVISEDIR/src/application
       chmod -R u+w $COVISEDIR/src/renderer
    fi

    cd $COVISEDIR
    find . -type d -exec chmod u+w {} \;

    echo "... done"

    # adjust the .tcshrc/.cshrc
    adjust_dot_files
 }

 install_runtime_shared()
 {
    echo " "
    echo "Installing core system..."

    ARCHFILE="$BASEDIR"/SHARED/coshared.tar.gz
    if [ -f "$ARCHFILE" ]
    then

       if [ ! -d $COVISEDIR ]
       then
          echo "INFO: making directory $COVISEDIR..."	
          mkdir -p $COVISEDIR
       fi

       echo "INFO: going to $COVISEDIR..."
       cd $COVISEDIR

       echo $N "INFO: extracting the COVISE shared system archive..."
       gzip -c -d "$ARCHFILE" | tar xf -
       echo " done."

    else
       echo "ERROR: $ARCHFILE"
       echo "       not found in $BASEDIR"
       echo "       make sure that the archive and the install script"
       echo "       are in the same directory"
       exit
    fi

 }

 create_qt_conf()
 {
    qtdir="$1"

    echo "INFO: generating qt.conf for Qt in $qtdir"
    echo "[Paths]" > "$qtdir/bin/qt.conf"
    echo "Prefix = $qtdir" >> "$qtdir/bin/qt.conf"
    echo "Translations = i18n" >> "$qtdir/bin/qt.conf"

    for i in \
       "$COVISEDIR/$ARCHSUFFIX/bin" \
       "$COVISEDIR/$ARCHSUFFIX/bin/"*.app/Contents/ \
       "$COVISEDIR/$ARCHSUFFIX/bin/Renderer/" \
       "$COVISEDIR/$ARCHSUFFIX/bin/Renderer/"*.app/Contents/ \
       ; do
    mkdir -p "$i/Resources"
    ln -sf "$qtdir/bin/qt.conf" "$i/Resources/qt.conf"
 done
 for i in "$COVISEDIR/$ARCHSUFFIX/bin"; do
    ln -sf "$qtdir/bin/qt.conf" "$i/qt.conf"
 done
}

install_runtime_core()
{
   echo " "
   echo "Installing platform-dependent parts..."

   ARCHFILE="$BASEDIR"/DIST.${ARCHSUFFIX}/co${ARCHSUFFIX}.bin.tar.gz
   if [ -f "$ARCHFILE" ]
   then

      if [ ! -d $COVISEDIR ]
      then
         echo "INFO: making directory $COVISEDIR..."	
         mkdir -p $COVISEDIR
      fi

      if [ ! -d $COVISEDIR/$ARCHSUFFIX ]
      then
         echo "INFO: making directory $COVISEDIR/$ARCHSUFFIX..."
         mkdir -p $COVISEDIR/$ARCHSUFFIX
      fi

      if [ ! -d $COVISEDIR/$ARCHSUFFIX/bin ]
      then
         mkdir -p $COVISEDIR/$ARCHSUFFIX/bin
      fi

      echo "INFO: going to $COVISEDIR..."
      cd $COVISEDIR

      echo $N "INFO: extracting the COVISE core system archive..."
      gzip -c -d "$ARCHFILE" | tar xf -
      echo " done."

      ARCHFILE="$BASEDIR"/DIST.${ARCHSUFFIX}/co${ARCHSUFFIX}.extern_libs.tar.gz
      if [ -f "$ARCHFILE" ]
      then
         echo $N "INFO: extracting the external libraries archive..."
         gzip -c -d "$ARCHFILE" | tar xf -

         for d in $COVISEDIR/extern_libs/{$ARCHSUFFIX,${ARCHSUFFIX%opt}}/qt{4,5}; do
            if [ -d "$d" ]; then
               create_qt_conf "$d"
            fi
         done

         echo " done."
      else
         echo "ERROR: $ARCHFILE"
         echo "       not found in $BASEDIR"
         echo "       make sure that the archive and the install script"
         echo "       are in the same directory"
         exit
      fi

      if [ "$ARCHSUFFIX" = "linux" ]
      then
         echo 'void main(){}' >checkso.cpp
         g++ checkso.cpp -lGL -o checkso 2>errmsg
         if grep 'cannot find' errmsg
         then
            #no GL installed on system, make a symbolic link to Gl which has come with covise
            ln -s $COVISEDIR/extern_libs/OpenGL/lib/libGL.so $COVISEDIR/$ARCHSUFFIX/bin/libGL.so.1
            ln -s $COVISEDIR/extern_libs/OpenGL/lib/libGLU.so $COVISEDIR/$ARCHSUFFIX/bin/libGLU.so.1
         else
            #GL is installed on the system, but perhaps the version numbers do not match
            if ( ldd checksosource | grep libGL.so | grep 'not found' ) 
            then
               debug_echo creating symbolic link for OpenGL
               ln -s `ldd checkso | grep libGL.so | awk '{ print $3;}'` $COVISEDIR/linux/bin/libGL.so.1
               g++ checkso.cpp -lGL -lGLU -o checkso
               ln -s `ldd checkso | grep libGLU.so | awk '{ print $3;}'` $COVISEDIR/linux/bin/libGLU.so.1
            else
               debug_echo OpenGL OK on system
            fi
         fi
         #Now we make the same check for the libstd++
         if ( ldd checksosource | grep libstdc++ | grep 'not found' ) 
         then
            licppname=`ldd checksosource | grep libstdc++| grep 'not found'| awk '{ print $1;}'` 
            debug_echo creating symbolic link for $libcppname
            ln -s `ldd checkso | grep libstdc++ | awk '{ print $3;}'` $COVISEDIR/linux/bin/$licppname
         else
            debug_echo proper libstdc++ version found
         fi
      fi

      ### Copy platform description file
      cp "$BASEDIR"/DIST.${ARCHSUFFIX}/Platform $COVISEDIR/$ARCHSUFFIX/

   else
      echo "ERROR: $ARCHFILE"
      echo "       not found in $BASEDIR"
      echo "       make sure that the archive and the install script"
      echo "       are in the same directory"
      exit
   fi
}

adjust_dot_files()
{
   ans='n'

   echo -n "Modify .cshrc/.tcshrc/.bashrc for COVISE (y/N)? "
   read ans

   if [ "$ans" = "y" -o "$ans" = "Y" ]; then
      adjust_dot_cshrc_files
      adjust_dot_bashrc_files
   else
      echo ""
      echo "Please adjust your environment to"
      echo "- add $COVISEDIR/bin to your PATH."
      echo ""
      sleep 1
   fi
}

adjust_dot_cshrc_files()
{
   ans='none'
   if [ -f $HOME/.tcshrc ]
   then
      while [ "$ans" != "y" -a  "$ans" != "n" ]
      do



         SET_COVISEDIR="setenv COVISEDIR $COVISEDIR"
         #echo "SET_COVISEDIR = $SET_COVISEDIR"

         ACT_SET_COVISEDIR=`cut -f1 -d\# $HOME/.tcshrc | grep "$SET_COVISEDIR"`
         #echo "ACT_SET_COVISEDIR = $ACT_SET_COVISEDIR"


         if [ "$ACT_SET_COVISEDIR" = ""  ]
         then
            echo "INFO: I need to modify your .tcshrc."   
            echo "Do you want me to save your .tcshrc in .tcshrc.old (y/n)?"
            echo ""
            read ans
            if [ "$ans" = "y" ]
            then
               cp $HOME/.tcshrc $HOME/.tcshrc.old
            fi
            echo "setenv COVISEDIR $COVISEDIR" >> $HOME/.tcshrc
            echo "set path=(${COVISEDIR}/bin \$path)" >> $HOME/.tcshrc
            if [ "$ARCHSUFFIX" = "hp"  ]
            then
               echo "setenv OIV_NO_OVERLAYS 1" >> $HOME/.tcshrc
            fi
         else
            ans='y'
            echo "your .tcshrc was already modified"
         fi    

      done
   else
      while [ "$ans" != "y" -a  "$ans" != "n" ]
      do

         SET_COVISEDIR="setenv COVISEDIR $COVISEDIR"
         #echo "SET_COVISEDIR = $SET_COVISEDIR"

         ACT_SET_COVISEDIR=`cut -f1 -d\# $HOME/.cshrc | grep "$SET_COVISEDIR"`
         #echo "ACT_SET_COVISEDIR = $ACT_SET_COVISEDIR"


         if [ "$ACT_SET_COVISEDIR" = ""  ]
         then

            if [ ! -f  $HOME/.cshrc ]
            then
               ans='n'
               echo "INFO: creating new .cshrc."
            else
               echo "INFO: I need to modify your .cshrc."
               echo "Do you want me to save your .cshrc in .cshrc.old (y/n)?"
               echo " "
               read ans
            fi
            if [ "$ans" = "y" ]
            then
               cp $HOME/.cshrc $HOME/.cshrc.old
            fi
            echo "setenv COVISEDIR $COVISEDIR" >> $HOME/.cshrc
            echo "set path=(${COVISEDIR}/bin \$path)" >> $HOME/.cshrc
            if [ "$ARCHSUFFIX" = "hp"  ]
            then
               echo "setenv OIV_NO_OVERLAYS 1" >> $HOME/.cshrc
            fi
         else
            ans='y'
            echo "your .cshrc was already modified"
         fi    
      done
   fi

}

adjust_dot_bashrc_files()
{
   ans='none'
   if [ -f $HOME/.bashrc ]
   then
      while [ "$ans" != "y" -a  "$ans" != "n" ]
      do



         SET_COVISEDIR="export COVISEDIR=$COVISEDIR"
         #echo "SET_COVISEDIR = $SET_COVISEDIR"

         ACT_SET_COVISEDIR=`cut -f1 -d\# $HOME/.bashrc | grep "$SET_COVISEDIR"`
         #echo "ACT_SET_COVISEDIR = $ACT_SET_COVISEDIR"


         if [ "$ACT_SET_COVISEDIR" = ""  ]
         then
            echo "INFO: I need to modify your .bashrc."   
            echo "Do you want me to save your .bashrc in .bashrc.old (y/n)?"
            echo ""
            read ans
            if [ "$ans" = "y" ]
            then
               cp $HOME/.bashrc $HOME/.bashrc.old
            fi
            echo "export COVISEDIR=$COVISEDIR" >> $HOME/.bashrc
            echo "export PATH=${COVISEDIR}/bin:\$PATH" >> $HOME/.bashrc
         else
            ans='y'
            echo "your .bashrc was already modified"
         fi

      done
   else
      while [ "$ans" != "y" -a  "$ans" != "n" ]
      do

         SET_COVISEDIR="export COVISEDIR=$COVISEDIR"
         #echo "SET_COVISEDIR = $SET_COVISEDIR"

         ACT_SET_COVISEDIR=`cut -f1 -d\# $HOME/.bashrc | grep "$SET_COVISEDIR"`
         #echo "ACT_SET_COVISEDIR = $ACT_SET_COVISEDIR"


         if [ "$ACT_SET_COVISEDIR" = ""  ]
         then

            if [ ! -f  $HOME/.bashrc ]
            then
               ans='n'
               echo "INFO: creating new .bashrc."
            else
               echo "INFO: I need to modify your .bashrc."
               echo "Do you want me to save your .bashrc in .bashrc.old (y/n)?"
               echo " "
               read ans
            fi
            if [ "$ans" = "y" ]
            then
               cp $HOME/.bashrc $HOME/.bashrc.old
            fi
            echo "export COVISEDIR=$COVISEDIR" >> $HOME/.bashrc
            echo "export PATH=${COVISEDIR}/bin:\$PATH" >> $HOME/.bashrc
         else
            ans='y'
            echo "your .bashrc was already modified"
         fi
      done
   fi

}


install_runtime_general_modules()
{
   echo " "
   echo "Installing general modules..."

   for ARCHFILE in "$BASEDIR"/DIST.${ARCHSUFFIX}/co${ARCHSUFFIX}.modules.*.tar.gz ; do
      if [ "$ARCHFILE" = "$BASEDIR"/DIST.${ARCHSUFFIX}/co${ARCHSUFFIX}.modules.Examples.tar.gz ]; then
         continue
      fi

      if [ -f "$ARCHFILE" ]
      then

         if [ ! -d $COVISEDIR ]
         then
            echo "INFO: making directory $COVISEDIR"	
            mkdir -p $COVISEDIR
         fi

         if [ ! -d $COVISEDIR/$ARCHSUFFIX ]
         then
            echo "INFO: making directory $COVISEDIR/$ARCHSUFFIX"
            mkdir -p $COVISEDIR/$ARCHSUFFIX
         fi

         if [ ! -d $COVISEDIR/$ARCHSUFFIX/bin ]
         then
            echo "INFO: making directory $COVISEDIR/$ARCHSUFFIX/bin"
            mkdir -p $COVISEDIR/$ARCHSUFFIX/bin
         fi


         echo "INFO: going to $COVISEDIR..."
         cd $COVISEDIR

         echo $N "INFO: extracting the COVISE general modules archive..."
         gzip -c -d "$ARCHFILE" | tar xf -
         echo " done."


      else
         echo "ERROR: $ARCHFILE"
         echo "       not found in $BASEDIR"
         echo "       make sure that the archive and the install script"
         echo "       are in the same directory"
         exit
      fi
   done
}

install_runtime_example_modules()
{
   echo " "
   echo "Installing example modules..."

   ARCHFILE="$BASEDIR"/DIST.${ARCHSUFFIX}/co${ARCHSUFFIX}.modules.Examples.tar.gz
   if [ -f "$ARCHFILE" ]
   then

      if [ ! -d $COVISEDIR ]
      then
         echo "INFO: making directory $COVISEDIR"	
         mkdir -p $COVISEDIR
      fi

      if [ ! -d $COVISEDIR/$ARCHSUFFIX ]
      then
         echo "INFO: making directory $COVISEDIR/$ARCHSUFFIX"
         mkdir -p $COVISEDIR/$ARCHSUFFIX
      fi

      if [ ! -d $COVISEDIR/$ARCHSUFFIX/bin ]
      then
         echo "INFO: making directory $COVISEDIR/$ARCHSUFFIX/bin"
         mkdir -p $COVISEDIR/$ARCHSUFFIX/bin
      fi


      echo "INFO: going to $COVISEDIR..."
      cd $COVISEDIR

      echo $N "INFO: extracting the COVISE example modules archive..."
      gzip -c -d "$ARCHFILE" | tar xf -
      echo " done."

   else
      echo "ERROR: $ARCHFILE"
      echo "       not found in $BASEDIR"
      echo "       make sure that the archive and the install script"
      echo "       are in the same directory"
      exit
   fi
}

install_runtime_renderer()
{
   ARCHFILE="$BASEDIR"/DIST.${ARCHSUFFIX}/co${ARCHSUFFIX}.renderer.$REND.tar.gz
   if [ ! -f "$ARCHFILE" ]; then
      return
   fi

   echo " "
   echo "Installing $REND Renderer..."

   INSTALLED_RENDERERS="$INSTALLED_RENDERERS $REND"

   FATAL=no;  PROJECT=General;  install_runtime_renderer_plugins
   FATAL=no;  PROJECT=Examples; install_runtime_renderer_plugins

   ARCHFILE="$BASEDIR"/DIST.${ARCHSUFFIX}/co${ARCHSUFFIX}.renderer.$REND.tar.gz
   if [ -f "$ARCHFILE" ]
   then

      if [ ! -d $COVISEDIR ]
      then
         echo "INFO: making directory $COVISEDIR"	
         mkdir -p $COVISEDIR
      fi

      if [ ! -d $COVISEDIR/$ARCHSUFFIX ]
      then
         echo "INFO: making directory $COVISEDIR/$ARCHSUFFIX"
         mkdir -p $COVISEDIR/$ARCHSUFFIX
      fi

      if [ ! -d $COVISEDIR/$ARCHSUFFIX/bin ]
      then
         echo "INFO: making directory $COVISEDIR/$ARCHSUFFIX/bin"
         mkdir -p $COVISEDIR/$ARCHSUFFIX/bin
      fi


      echo "INFO: going to $COVISEDIR..."
      cd $COVISEDIR

      echo $N "INFO: extracting the $REND renderer archive..."
      gzip -c -d "$ARCHFILE" | tar xf -
      echo " done."


   else
      echo "ERROR: $ARCHFILE"
      echo "       not found in $BASEDIR"
      echo "       make sure that the archive and the install script"
      echo "       are in the same directory"
      exit
   fi
}

install_runtime_renderer_plugins()
{
   case "$REND" in
      coin|inventor)
         return
         ;;
   esac

   echo " "
   test -z $PROJECT && PROJECT=General
   echo "Installing $PROJECT $REND plugins..."

   ARCHFILE="$BASEDIR"/DIST.${ARCHSUFFIX}/co${ARCHSUFFIX}.${REND}plugins.$PROJECT.tar.gz
   if [ -f "$ARCHFILE" ]
   then

      if [ ! -d $COVISEDIR ]
      then
         echo "INFO: making directory $COVISEDIR"	
         mkdir -p $COVISEDIR
      fi

      if [ ! -d $COVISEDIR/$ARCHSUFFIX ]
      then
         echo "INFO: making directory $COVISEDIR/$ARCHSUFFIX"
         mkdir -p $COVISEDIR/$ARCHSUFFIX
      fi

      if [ ! -d $COVISEDIR/$ARCHSUFFIX/bin ]
      then
         echo "INFO: making directory $COVISEDIR/$ARCHSUFFIX/bin"
         mkdir -p $COVISEDIR/$ARCHSUFFIX/bin
      fi


      echo "INFO: going to $COVISEDIR..."
      cd $COVISEDIR

      echo $N "INFO: extracting the $REND $PROJECT plugins archive..."
      gzip -c -d "$ARCHFILE" | tar xf -
      echo " done."


   else
      if [ "$FATAL" = "yes" ]; then
         echo "ERROR: $ARCHFILE"
         echo "       not found in $BASEDIR"
         echo "       make sure that the archive and the install script"
         echo "       are in the same directory"
         exit
      fi
   fi
}

install_example_maps_and_data()
{
   echo " "
   echo "Installing example maps and data..."

   ARCHFILE="$BASEDIR"/SHARED/coexample.maps_data.tar.gz

   if [ -f "$ARCHFILE" ]
   then

      if [ ! -d $COVISEDIR ]
      then

         echo "INFO: making directory $COVISEDIR..."
         mkdir -p $COVISEDIR
      fi

      cd $COVISEDIR
      echo $N "INFO: extracting example maps and data archive..."
      gzip -c -d "$ARCHFILE" | tar xf -
      echo " done."

   else
      echo "ERROR: $ARCHFILE"
      echo "       not found in $BASEDIR"
      echo "       make sure that the archive and the install script"
      echo "       are in the same directory"
      exit
   fi
}

install_tutorial_maps_and_data()
{
   echo " "
   echo "Installing tutorial maps and data..."

   ARCHFILE="$BASEDIR"/SHARED/cotutorial.maps_data.tar.gz
   if [ -f "$ARCHFILE" ]
   then

      if [ ! -d $COVISEDIR ]
      then

         echo "INFO: making directory $COVISEDIR"
         echo " "	

         mkdir -p $COVISEDIR
      fi

      cd $COVISEDIR
      echo $N "INFO: extracting tutorial maps and data archive..."
      gzip -c -d "$ARCHFILE" | tar xf -
      echo " done."

   else
      echo "ERROR: $ARCHFILE"
      echo "       not found in $BASEDIR"
      echo "       make sure that the archive and the install script"
      echo "       are in the same directory"
      exit
   fi
}


extract_doc()
{
   echo " "
   echo "Installing documentation..."

   if [ ! -d $COVISEDIR ]
   then
      echo "INFO: making directory $COVISEDIR"
      mkdir $COVISEDIR
      if  [ ! -d $COVISEDIR ]
      then
         echo "ERROR: could not create directory: Leaving..."
         exit
      fi
   fi

   for i in pdf html; do
      ARCHFILE="$BASEDIR"/SHARED/codoc.$i.tar.gz

      if [ -f "$ARCHFILE" ]
      then
         cd $COVISEDIR
         gzip -c -d "$ARCHFILE" | tar xf -
      else
         echo " archive file $ARCHFILE not found"

      fi
   done

   echo "changing permissions of the COVISE documentation directory"
   chmod u+w $COVISEDIR/doc

   echo "... done"
}

###############################################################
###############################################################
############# MAIN BIT OF THE SCRIPT #################
###############################################################
###############################################################
echo "Looking for distribution files and directories..."
echo "" 

### this one exists on all distribution disks, but not in global installations
if [ -d "$BASEDIR"/SHARED ] 
then
   echo "Found Distributions under $BASEDIR"
   echo "" 
   cd "$BASEDIR"
   avail_dist=`/bin/ls -dr DIST.*/Platform | cut -f2 -d. | cut -f1 -d/ | sort -u`
   for dist in $avail_dist ; do
      if [ ! -d DIST.$dist ]; then
         avail_dist=`echo $avail_dist | sed -e s/$dist//`
         continue
      fi
      printf "%9s : "   $dist 
      grep '^DESC' "$BASEDIR"/DIST.$dist/Platform | cut -f2- -d' '
   done
   readonly avail_dist

else
   if [ -d "$BASEDIR"/covise ]
   then
      avail_dist="linked"
   else
      echo ""
      echo "!!!!!! No installation directories found !!!!!!"
      echo ""
      echo "Could not find the directories SHARED and DIST.<ARCH>."
      exit
   fi
fi


menu_main


echo ""
echo "Goodbye! I hope your COVISE installation was successful."
echo "If not, please contact me at:"
echo ""
echo "covise-users@hlrs.de"
echo ""
