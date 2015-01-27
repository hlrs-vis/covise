#include "cfx5ext.h"
      
	    
dllexport(user_export)

      
	    
      SUBROUTINE USER_EXPORT ( CZ,DZ,IZ,LZ,RZ )
      IMPLICIT NONE

      INTERFACE
      SUBROUTINE USER_COVISE(ITERATION,CVARa,CZONEa,VAR,NVAR,
     &     inletVelMulti,USER_VARS,
     &                       numCoord,PORT,CZ,DZ,IZ,LZ,RZ)
        implicit none
	INTEGER           ITERATION,NVAR
	CHARACTER*(*)          CVARa,CZONEa
	REAL                     VAR(NVAR)   
	REAL		      inletVelMulti,angular
	CHARACTER*(3*20) USER_VARS,PORT      
        INTEGER              numCoord 
	INTEGER IZ(*)
	CHARACTER CZ(*)*(1)
	DOUBLE PRECISION DZ(*)
	LOGICAL LZ(*)
	REAL RZ(*)  
        
	END  SUBROUTINE USER_COVISE
	END INTERFACE     

      
	    
	    
C    ++++++++++++++++++++++++++++++++++++
C    +                                  +
C    +    Covise SimLib include file    +
C    +                                  +
C    ++++++++++++++++++++++++++++++++++++

C --- Commands
       INTEGER CONOCO,COVINI,COFINI,COEXEC

C --- Data Object Creation
       INTEGER COSU1D,COSU3D

C --- Parameter Requests
       INTEGER COGPFL,COGPSL,COGPIN,COGPCH,COGPBO,COGPTX
 
C --- binary reading/writing
       INTEGER CORECV,COSEND

C --- request verbose level
       INTEGER COVERB

C --- attach an attribute
       INTEGER COATTR

C --- parallel stuff
       INTEGER COPAIN,COPAPO,COPACM,COPAVM,COPANO,COSIPD

       INTEGER COSLEP
       INTEGER CFXMAL
	   


CD User routine: Export data.
CC
CC --------------------
CC        Input
CC --------------------
CC
CC  none
CC
CC --------------------
CC      Modified
CC --------------------
CC
CC  Stacks possibly.
CC
CC --------------------
CC        Output
CC --------------------
CC
CC  none

CC --------------------
CC       Details
CC --------------------
CC
CC  Export seletected variables to formatted file.
CC
CC======================================================================
C
C ------------------------------
C     Preprocessor includes
C ------------------------------
C
#include "MMS.h"
#include "parallel_partitioning.h"
#include "stack_point.h"


C
C ------------------------------
C        Global Parameters
C ------------------------------
C
C
C ------------------------------
C        Argument list
C ------------------------------
C
      INTEGER IZ(*),i
      CHARACTER CZ(*)*(1)
      DOUBLE PRECISION DZ(*)
      LOGICAL LZ(*)
      REAL RZ(*)
      
C
C ------------------------------
C        External routines
C ------------------------------

      INTEGER            LENACT
      EXTERNAL           LENACT
C
C ------------------------------
C        Local Parameters
C ------------------------------
C
      CHARACTER*(*)      ROUTIN
      PARAMETER         (ROUTIN='USER_EXPORT')
C
      CHARACTER*(*)      CFILE
      PARAMETER         (CFILE='../export.dat')
      CHARACTER*(4)      CDTYPE
C
C ------------------------------
C        Local Variables
C ------------------------------
C
      INTEGER            LVAR,NVAR,testvalue,IPARD,NERR
      INTEGER            MSGTAG,numCoord,ISIZE
      INTEGER            MSGTAG2,JADRES,LENVEC,LEVVUL  
      INTEGER            numberinletnodes
      SAVE               numberinletnodes
      DATA numberinletnodes /0/     
      CHARACTER*(4)      CRESLT
      CHARACTER*(MXDNAM) CVAR,CZONE,WHEN,OPER
      CHARACTER*(3*MXDNAM) USER_VARS,SOLVER_VARS,PORT
      CHARACTER*(2*MXDNAM+1) WHERE
      CHARACTER*(50) CDIR,ZONENAME
      CHARACTER number	  

      REAL		 inletVelMulti,inletVelMultiold
      REAL               angular,angularold	  
      LOGICAL            LOIO
      INTEGER            updateInterval,ZNTOTAL,updateIntervalold
      INTEGER            ITERATION
      SAVE               ITERATION
      SAVE               inletVelMultiold,angularold,updateIntervalold                   
      DATA ITERATION /0/
      DATA updateintervalold /1/
      DATA inletVelMultiold /1/  	  
      DATA angularold /0/  
	  
          
c      INTEGER,DIMENSION(:), ALLOCATABLE :: pinletnodes
c       SAVE pinletnodes	  
	                   
C ------------------------------
C        Stack pointers
C ------------------------------
C
      __stack_point__    pVAR,pVELFACTOR,pANGULAR
 
C ---------------------------
C    Executable Statements
C ---------------------------    

C=======================================================================
C     Prologue
C=======================================================================
C
C---- Output control flag.
C
      LOIO = ( (RUNMOD .EQ. __sequential__) .OR.
     &         (PARMOD .EQ. __master__) )

C
C=======================================================================
C     Export Data
C=======================================================================
       IF ( (RUNMOD .EQ. __sequential__) .OR.
     &          (PARMOD .EQ. __master__) ) THEN
        open(unit = 7,file = 'export.out')
        write(7,*) 'user_export file output'
       ENDIF

C  Creating Connection to Covise 
c       IF ( PARMOD .EQ. __master__ ) THEN

       IF ( (RUNMOD .EQ. __sequential__) .OR.
     &          (PARMOD .EQ. __master__) ) THEN
            WRITE (7,*) 'Entering coSleep'
            WRITE (*,*) 'Entering coSleep'
       ENDIF

C 	     IF (COSLEP(30000).NE.0) THEN
            IF ( (RUNMOD .EQ. __sequential__) .OR.
     &          (PARMOD .EQ. __master__) ) THEN
               WRITE (7,*) 'Error in coSleep'
            ENDIF
            WRITE (*,*) 'Error in coSleep'
C         ENDIF
c      ENDIF


cs	  WRITE (*,*) 'Begin of Export'
	  
       IF ( (RUNMOD .EQ. __sequential__) .OR.
     &          (PARMOD .EQ. __master__) ) THEN  
          IF (CONOCO().NE.0) THEN
             CALL COVWSI()
             IF (COVINI().NE.0) THEN
                WRITE (7,*) 'Could not connect to Covise'             
                WRITE (*,*) 'Could not connect to Covise'  
             ELSE
                write(7,*) 'Connection ok, userexport'            
                write(*,*) 'Connection ok, userexport'
             ENDIF
          ENDIF
       ENDIF
		 
c         GET PARAMS from COVISE		  
	   
       IF (ITERATION .EQ. 0) THEN
	 
	  IF ( (RUNMOD .EQ. __sequential__) .OR.
     &          (PARMOD .EQ. __master__) ) THEN 
	         CALL PSHDIR('/FLOW/BOUNDCON/ZN1/BCP3/VARIABLES/VEL_FL1'
     &  	      ,'STOP',CRESLT)
                 CALL MAKDAT('velfactor','REAL','STOP',1,pVELFACTOR
     &                 ,CRESLT)
	         CALL MAKDAT('angular','REAL','STOP',1,pANGULAR
     &			,CRESLT)            
	         CALL POPDIR('STOP',CRESLT)
	       
	  ELSE IF ( PARMOD .EQ. __slave__ ) THEN 
	         
	         CALL PSHDIR('/FLOW/BOUNDCON/ZN1/BCP3/VARIABLES/VEL_FL1'
     &    	      ,'STOP',CRESLT) 
	         CALL MAKDAT('velfactor','REAL','STOP',1,pVELFACTOR
     &                 ,CRESLT)   
	         CALL MAKDAT('angular','REAL','STOP',1,pANGULAR
     &			,CRESLT)
	         CALL POKER('angular',1,angular,'STOP',CRESLT,RZ)
c	        write(*,*)'POKER TUT'
                 CALL POPDIR('STOP',CRESLT)
	       
	  ENDIF
        ENDIF !Iteration EQ 0
			 
          
	IF ( (RUNMOD .EQ. __sequential__) .OR.
     &          (PARMOD .EQ. __master__) ) THEN   
  	   WRITE(*,*)'###### ITERATION (export): ',ITERATION
      updateInterval=1   
	  IF(COGPIN('updateInterval',updateInterval).NE.0) THEN
             WRITE(7,*) 'Could not get updateinterval'
             WRITE(*,*) 'Could not get updateinterval'
          ELSE
	      IF (updateInterval .EQ. updateIntervalold) THEN
	       ELSE
                   WRITE(7,*) 'neuupdateInterval=',updateInterval
                   WRITE(*,*) 'updateInterval=',updateInterval
	      ENDIF	  	  
  	      updateIntervalold = updateInterval	  
          ENDIF	  
                   WRITE(7,*) 'Hallo0'
	  testvalue = COGPFL('inletVelMulti',inletVelMulti)
                   WRITE(7,*) 'Hallo0.5'
	  IF(testvalue .NE.0) THEN
	     WRITE(7,*) 'Could not get inletVelMulti'
	     WRITE(*,*) 'Could not get inletVelMulti'
c	     WRITE(*,*) 'testvalue: ',testvalue
	  ELSE
	      IF (inletVelMulti .EQ. inletVelMultiold) THEN
	      ELSE
	          WRITE(7,*) 'inletVelMulti=',inletVelMulti
	          WRITE(*,*) 'inletVelMulti=',inletVelMulti
	      ENDIF	  	  
	      inletVelMultiold = inletVelMulti
                   WRITE(7,*) 'Hallo1'
	  ENDIF
                   WRITE(7,*) 'Hallo2'
	  testvalue = COGPFL('incidenceAngular',angular)
	  IF(testvalue.NE.0) THEN
	     WRITE(7,*) 'Could not get angular'
	     WRITE(*,*) 'Could not get angular'
c	     WRITE(*,*) 'testvalue: ',testvalue
          ELSE
	     IF (angular .EQ. angularold) THEN
	     ELSE
	          WRITE(7,*) 'angular=',angular
	          WRITE(*,*) 'angular=',angular
	     ENDIF
                   WRITE(7,*) 'Hallo3'
		 angularold = angular
          ENDIF

                   WRITE(7,*) 'Hallo4' 	  	 
	 CALL PSHDIR('/FLOW/BOUNDCON/ZN1/BCP3/VARIABLES/VEL_FL1'
     &    	      ,'STOP',CRESLT)  
                   WRITE(7,*) 'Hallo5'
	 CALL POKER('velfactor',1,inletVelMulti,'STOP',CRESLT,RZ)	         
                   WRITE(7,*) 'Hallo6'
	 CALL POKER('angular',1,angular,'STOP',CRESLT,RZ)
                   WRITE(7,*) 'Hallo7'
	 CALL POPDIR('STOP',CRESLT)
                   WRITE(7,*) 'Hallo8'
             
       ENDIF
      
C---- Set data to export (really should use CONVERT_USER_NAMES instead)
     
c     send UpdateInterval to slaves
	      
      IF ( PARMOD .EQ. __master__ ) THEN
              MSGTAG2=2
	      DO IPARD = 2,NPARD
	          WRITE(7,*) 'vor sndIB1 updateInterval'
		CALL SNDIB1(updateInterval,IPARD,MSGTAG2,.TRUE.,NERR)	
	          WRITE(7,*) 'vor sndIB1 angular'
		CALL SNDRB1(angular,IPARD,MSGTAG2,.TRUE.,NERR)
	          WRITE(7,*) 'vor sndIB1 inletVelMulti'
		CALL SNDRB1(inletVelMulti,IPARD,MSGTAG2,.TRUE.,NERR)     
             ENDDO
      ELSE IF ( PARMOD .EQ. __slave__ ) THEN
              MSGTAG2=2
c	write(*,*)'SLAVE: Empfange angular und inletVelMulti'
	CALL RCVIB1(updateInterval,1,MSGTAG2,.TRUE.,NERR) 	         
        CALL RCVRB1(angular,1,MSGTAG2,.TRUE.,NERR)
	CALL RCVRB1(inletVelMulti,1,MSGTAG2,.TRUE.,NERR) 
	CALL PSHDIR('/FLOW/BOUNDCON/ZN1/BCP3/VARIABLES/VEL_FL1'
     &    	      ,'STOP',CRESLT) 
	CALL POKER('angular',1,angular,'STOP',CRESLT,RZ)
	CALL POKER('velfactor',1,inletVelMulti,'STOP',CRESLT,RZ) 
	CALL POPDIR('STOP',CRESLT)
		    	    
      ENDIF

	IF (MOD(ITERATION,updateInterval) .NE. 0) THEN
	   write(7,*) 'skipping iteration'
	   write(*,*) 'skipping iteration'
	   GOTO 100
	ENDIF
	
	       
c GET MESH INFO AND STORE IT UNDER CDIR     
        CALL USER_GET_MESH_INFO ('','GET','STOP',
     &            'LATEST','','',CDIR,CRESLT,CZ,DZ,IZ,LZ,RZ)	       
c        write(*,*) 'USER_GET_MESH_INFO RESULT: ',CRESLT 
c        write(*,*) 'CDIR: ',CDIR       
        CALL PSHDIR('/USER/'//CDIR,'STOP',CRESLT)
c        write(*,*) 'PSHDIR RESULT: ',CRESLT     
        CALL PEEKI('NZN',1,ZNTOTAL,'STOP',CRESLT, IZ)
c        write(*,*) 'PEEKI RESULT: ',CRESLT   
c        write(*,*) 'ZNTOTAL: ',ZNTOTAL	           
	CALL POPDIR('STOP',CRESLT)
	    	     
c Schleife ueber alle Domains 
	    
      DO i = 1, ZNTOTAL
	
	CALL PSHDIR('/USER/'//CDIR,'STOP',CRESLT)
	CALL PEEKCA ('/USER/MESH_INFO/CZONE',i,CZONE,
     &                      'STOP',CRESLT, CZ)
        CALL POPDIR('STOP',CRESLT)
c	write(*,*)' CZONE(',i,') = ',CZONE
	WHEN  = 'LATEST'    
	 
	USER_VARS='Velocity'   !!!!!!!!!!!!!!!!!!!!!
	CVAR='VEL_FL1'    
	write (number, '(I1)') i  !cast Integer to Char
	PORT='VEL_FL1' // '_' // number       
c	write(*,*)'PORT :',PORT              
c	CALL USER_PRINT_CHAR('USER Name: ',USER_VARS)
c	CALL USER_PRINT_CHAR('Solver Name: ',CVAR)

	WHERE = CZONE(1:LENACT(CZONE)) //'/VERTICES'             	    
	CALL GETVAR(ROUTIN,' ',CVAR,WHERE,WHEN,
     &            'RETURN','STOP',CRESLT, LVAR,NVAR,pVAR,
     &            CZ,DZ,IZ,LZ,RZ)    
	CALL USER_COVISE(ITERATION,CVAR,CZONE,RZ(pVAR),NVAR,
     &             inletVelMulti,USER_VARS,
     &         numCoord,PORT,CZ,DZ,IZ,LZ,RZ)
   
	CALL GETVAR(ROUTIN,' ',CVAR,WHERE,WHEN,
     &            'RELEASE','STOP',CRESLT, LVAR,NVAR,pVAR,
     &             CZ,DZ,IZ,LZ,RZ)    
	    
        
	 PORT='PRES' // '_' // number 
       IF ( (RUNMOD .EQ. __sequential__) .OR.
     &          (PARMOD .EQ. __master__) ) THEN	 
	 write(7,*)'PORT :',PORT
	 ENDIF
	 write(*,*)'PORT :',PORT   
         USER_VARS='Pressure'       !!!!!!!!!!!!!!!!!!!!!!!!!!!
         CALL CONVERT_NAME_U2S('Variable',
     &                        USER_VARS(1:LENACT(USER_VARS)),
     &                        'ZN1',
     &                        CVAR,
     &                        OPER,CRESLT,CZ,DZ,IZ,LZ,RZ)     
c         CALL USER_PRINT_CHAR('USER Name: ',USER_VARS)
c         CALL USER_PRINT_CHAR('Solver Name: ',CVAR)
         WHERE = CZONE(1:LENACT(CZONE)) //'/VERTICES'
         CALL GETVAR(ROUTIN,' ',CVAR,WHERE,WHEN,
     &            'RETURN','STOP',CRESLT, LVAR,NVAR,pVAR,
     &              CZ,DZ,IZ,LZ,RZ)           
         CALL USER_COVISE(ITERATION,CVAR,CZONE,RZ(pVAR),NVAR,
     &             inletVelMulti,USER_VARS,
     &                numCoord,PORT,CZ,DZ,IZ,LZ,RZ)   
         CALL GETVAR(ROUTIN,' ',CVAR,WHERE,WHEN,
     &            'RELEASE','STOP',CRESLT, LVAR,NVAR,pVAR,
     &            CZ,DZ,IZ,LZ,RZ)
c	  write(*,*)'Exporting: ',USER_VARS,'  (',CVAR,')'



	 PORT='TEMP_FL1' // '_' // number 
c	 write(*,*)'PORT :',PORT   
         USER_VARS='Temperature'       !!!!!!!!!!!!!!!!!!!!!!!!!!!
         CVAR='TEMP_FL1'
c         CALL CONVERT_NAME_U2S('Variable',
c     &                        USER_VARS(1:LENACT(USER_VARS)),
c     &                        'ZN1',
c     &                        CVAR,
c     &                        OPER,CRESLT,CZ,DZ,IZ,LZ,RZ)     
c         CALL USER_PRINT_CHAR('USER Name: ',USER_VARS)
c         CALL USER_PRINT_CHAR('Solver Name: ',CVAR)
         WHERE = CZONE(1:LENACT(CZONE)) //'/VERTICES'
         CALL GETVAR(ROUTIN,' ',CVAR,WHERE,WHEN,
     &            'RETURN','STOP',CRESLT, LVAR,NVAR,pVAR,
     &              CZ,DZ,IZ,LZ,RZ)           
         CALL USER_COVISE(ITERATION,CVAR,CZONE,RZ(pVAR),NVAR,
     &             inletVelMulti,USER_VARS,
     &                numCoord,PORT,CZ,DZ,IZ,LZ,RZ)   
         CALL GETVAR(ROUTIN,' ',CVAR,WHERE,WHEN,
     &            'RELEASE','STOP',CRESLT, LVAR,NVAR,pVAR,
     &            CZ,DZ,IZ,LZ,RZ)
            IF ( (RUNMOD .EQ. __sequential__) .OR.
     &          (PARMOD .EQ. __master__) ) THEN
	  write(7,*)'Exporting: ',USER_VARS,'  (',CVAR,')'
	  ENDIF
	  write(*,*)'Exporting: ',USER_VARS,'  (',CVAR,')'

     
      ENDDO !(over Zones)     
      IF (( ((RUNMOD .EQ. __parallel_run__) .AND.
     &          (PARMOD .EQ. __master__) ))  .OR.  
     &          (RUNMOD .EQ. __sequential__)) THEN
       
        testvalue = COFINI()
        if (testvalue.ne.0) then
          WRITE(7,*) 'Lost Connection to Covise 2 '
          WRITE(*,*) 'Lost Connection to Covise 2 '
          STOP
        ELSE
c          WRITE(*,*) 'SENT FINISH MESSAGE: ',CVAR
        ENDIF
        testvalue = COEXEC()
          if (testvalue.ne.0) then
            WRITE(7,*) 'Lost Connection to Covise 3'
            WRITE(*,*) 'Lost Connection to Covise 3'
            STOP
          ELSE
c           WRITE(*,*) 'Exporting: ',CVAR
          ENDIF
      ENDIF  
	        
c      CALL CHMDIR(ROUTIN,'STOP',CRESLT)
c      CALL DELDAT('pFACTOR','STOP',CRESLT) 
c      CALL DELDAT('pANGULAR','STOP',CRESLT)
c      CALL CHGDIR('..','STOP',CRESLT)

 100    ITERATION = ITERATION +1
           IF ( (RUNMOD .EQ. __sequential__) .OR.
     &          (PARMOD .EQ. __master__) ) THEN
	    WRITE(7,*) 'End of Export'
c	    WRITE(*,*) 'End of Export'
        close(unit = 7)
        ENDIF

	END
 
