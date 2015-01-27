C=======================================================================
C @(#) $Id: user_covise.F,v 1.0 2006/04/21 12:00:00 hforkel Exp $
C-----------------------------------------------------------------------
      SUBROUTINE USER_COVISE(ITERATION,CVARa,ZONEa,VAR,NVAR,
     &     inletVelMulti,USER_VARS,
     &                numCoord,PORT,CZ,DZ,IZ,LZ,RZ)

CC
CD Write a global vertex able to a formatted file.
CC
CC --------------------
CC       Input
CC --------------------
CC
CI NFILE        : Fortran I/O channel.
CI OPER         : Operation to export for variable.
CI CVAR         : Variable to export.
CI CZONE        : Zone to export.
CI WHEN         : Time level to export.
CI VAR          : Data array to write out on VERTICES.
CI NVAR         : Array size of VAR.
CC
CC --------------------
CC      Modified
CC --------------------
CC
CV Stacks possibly.
CC
CC --------------------
CC       Output
CC --------------------
CC
CC
CC --------------------
CC       Details
CC --------------------
CC
CC
CC --------------------
CC Modification History
CC --------------------
CC
CC  2006 Apr 21  HF   Created.
CC
CC --------------------
CC Variable Dictionary
CC --------------------
CC
CC --------------------
CC Local Variables & Parameters
CC --------------------
CC
CC======================================================================
C
C ------------------------------
C     Preprocessor Includes
C ------------------------------
C
#include "MMS.h"
#include "cfd_constants.h"
#include "cfd_sysdep.h"
#include "locale_params.h"
#include "parallel_partitioning.h"
#include "stack_point.h"



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
       INTEGER COPAIN,COPAPO,COPACM,COPAVM,COPANO

C
C ------------------------------
C        Global Parameters
C ------------------------------
C
C
C ------------------------------
C        Argument List
C ------------------------------

      INTEGER            ITERATION,NVAR
      
C
      CHARACTER*(*)     CVARa,ZONEa
C
      REAL               VAR(NVAR)

      CHARACTER*(3*MXDNAM) USER_VARS,PORT
     
      INTEGER IZ(*)
      CHARACTER CZ(*)*(1)
      DOUBLE PRECISION DZ(*)
      LOGICAL LZ(*)
      REAL RZ(*)
C
C ------------------------------
C        External Routines
C ------------------------------
C
      INTEGER            LENACT
      EXTERNAL           LENACT
C
C ------------------------------
C        Local Parameters
C ------------------------------
C
      REAL		 inletVelMulti,angular
      PARAMETER         (MSGTAG=__disdat_message__)
C
      CHARACTER*(*)      ROUTIN
      PARAMETER         (ROUTIN='USER_COVISE')
C
C ------------------------------
C        Local Variables
C ------------------------------
     
      CHARACTER* 200  CURDIR
      CHARACTER* 200  CDANAM  
      CHARACTER* 208  CFNAME
      CHARACTER* 208  CLNAME 
   
      	    
      INTEGER LENCD,ISIZE,JADRES,LFNAME,LLNAME,peekNVALUES
      INTEGER LENVEC,LEVVUL
      	  	  	
      CHARACTER*(4)   CERACT
      INTEGER         NERR,NVX,NVXP,NVXG,NVARP,NVARG,NDIM,NCOMPT,SIZEBLOC
      INTEGER         IVAR,IPARD, NKGVxVx, LParF,NParF,k,numinnervertices
      INTEGER         MSGTAG2,testvalue,updateInterval,j,masterinodes
      INTEGER         numCoord,nodes,test,slaveinodes
      INTEGER         ISIZE2,JADRES2,CRESLT2
      INTEGER         OWN_PAR, NUM_PAR
	       
C
      CHARACTER*(4)      CRESLT,CDTYPE,CDTYPE2
      CHARACTER*(MXDNAM) CVAR,ZONE,CNAME
      CHARACTER*(MXPNAM) MESH_DIR

      LOGICAL            LOIO
c      INTEGER, DIMENSION(:) :: pinletnodes
c      INTEGER, DIMENSION(:), ALLOCATABLE :: pNODEPART
c      INTEGER, DIMENSION(:), ALLOCATABLE :: pLOCMAP	  
c      INTEGER, DIMENSION(:), ALLOCATABLE :: pINODEPART
c      INTEGER, DIMENSION(:), ALLOCATABLE :: pINODEMASTER
      INTEGER, DIMENSION(:), ALLOCATABLE :: pISENDARRAY
      REAL, DIMENSION(:), ALLOCATABLE :: pRSENDARRAY 
      REAL, DIMENSION(:), ALLOCATABLE :: pVELx 
      REAL, DIMENSION(:), ALLOCATABLE :: pVELy 
      REAL, DIMENSION(:), ALLOCATABLE :: pVELz
c      SAVE pLOCMAP 
c      SAVE pNODEPART
c      SAVE pINODEPART
c      SAVE pINODEMASTER	  
c      SAVE masterinodes
c      SAVE slaveinodes	  
      
C ------------------------------
C        Stack Pointers
C ------------------------------
C
      __stack_point__    pKGVxVx,pParF,pVARP
     &                   
C
C ------------------------------
C        Data Statements
C ------------------------------
C
C
C ------------------------------
C  Directory on entry:- Any.
C ------------------------------
C
C ------------------------------
C     Executable Statements
C ------------------------------
C
C=======================================================================
C     Prologue
C=======================================================================
C
      CVAR  = CVARa
      ZONE = ZONEa

C
C---- Output control flag.
C
      LOIO = ( (RUNMOD .EQ. __sequential__) .OR.
     &         (PARMOD .EQ. __master__) )
C
C
C---- Path to mesh directory.
C     Path changes for CFX 11.0.

      MESH_DIR = '/FLOW/MESH/LATEST/'// ZONE
C---- CFX 10
C      MESH_DIR = '/FLOW/MESH/'// CZONE
	 
C=======================================================================
C     Serial
C=======================================================================
C
      IF (RUNMOD .EQ. __sequential__) THEN
      	 
c	 CALL PUTDIR(CURDIR,LENCD)
c	 write(*,*)'CURDIR: ',CURDIR   ! = /
c	 CFNAME = CURDIR
c	 CERACT='STOP'
c 101     CONTINUE	     
c	 CALL TARLIS(CFNAME,LFNAME,CLNAME,LLNAME,CDTYPE,
c     &	         CERACT,ISIZE,JADRES,CRESLT)	
c         IF(CRESLT(1:4).EQ. 'GOOD') THEN	    
c          IF(ISIZE.GT.1) THEN        
c          CALL LOCDAT(CFNAME,CDTYPE2,'STOP',ISIZE2,ptest,
c     &  		      CRESLT2)
c	  DO i=0,ISIZE2-1
c	     IF(CDTYPE .EQ. 'REAL') THEN                
c		 WRITE(*,*)RZ(ptest+i)
c	     ELSEIF (CDTYPE .EQ. 'INTR') THEN	      
c		 WRITE(*,*)IZ(ptest+i) 
c	     ELSE
c		 WRITE(*,*)'not REAL and INTR'
c	     ENDIF	 
c	  ENDDO	         	 
c	  ELSE IF (LLNAME.GT.0) THEN
c	     WRITE(*,*)'link file:    ',CFNAME  
c	  ELSE
c	    WRITE(*,*)'subdirectory: ',CFNAME 	     
c	 ENDIF	              
c	 goto 101	 
c	ENDIF 	 
	 
         DO i = 1, 1
c         WRITE(*,*) 'value ',i,' is',VAR(I)
         ENDDO
c        WRITE(*,*) 'CVAR ', CVAR
c	 WRITE(*,*) 'PORT ', PORT    
c         WRITE(*,*)'###### ITERATION (user_covise): ',ITERATION
	     
       IF( CVAR .EQ. 'VEL_FL1') THEN  

	 
	  ALLOCATE (pVELx(NVAR/3))
	  ALLOCATE (pVELy(NVAR/3))
	  ALLOCATE (pVELz(NVAR/3))
	  
	  k=1
	  DO i=0,(NVAR/3-1)
	    pVELx(k)=VAR(3*i+1)
	   k=k+1
	  ENDDO  

	  k=1
	  DO i=0,(NVAR/3-1)
	    pVELy(k)=VAR(3*i+2)
	   k=k+1
	  ENDDO
	  
	  k=1
	  DO i=0,(NVAR/3-1)
	    pVELz(k)=VAR(3*i+3)
	   k=k+1
	  ENDDO 
     
        testvalue = COSU3D(PORT,NVAR/3,pVELx,pVELy,pVELz)  
        
	DEALLOCATE (pVELx)
	DEALLOCATE (pVELy)
	DEALLOCATE (pVELz)
	
	ELSE
	 
	  testvalue = COSU1D(PORT,NVAR,VAR)	  
         IF (testvalue.NE.0) THEN
            WRITE(*,*) 'Lost Connection to Covise 1 ' ,testvalue
         ELSE
c            write(*,*) 'Data sent: ',CVAR
         ENDIF	 
c	 write(*,*) 'End of user_covise'
       ENDIF ! CVAR.EQ.VEL_FL1
			 
			 
C=======================================================================
C     Slave in Parallel Run
C=======================================================================
C
      ELSE IF ( (RUNMOD .EQ. __parallel_run__) .AND.
     &          (PARMOD .EQ. __slave__) ) THEN
C
      CALL GET_PARALLEL_INFO ('OWNPAR',OWN_PAR,CNAME,CRESLT)
c	     write(*,*)'OWN_PAR: ',OWN_PAR 
      CALL GET_PARALLEL_INFO ('NPARD',NUM_PAR,CNAME,CRESLT)
c	  write(*,*)'NUM_PAR ',NUM_PAR
c  get inletVelMulti from Master
	     	         
      MSGTAG2=2	   
      CALL RCVRB1(inletVelMulti,1,MSGTAG2,.TRUE.,NERR)   
c      write(*,*)' SLAVE inletVelMulti: ',inletVelMulti  
		    
	  CALL PSHDIR(MESH_DIR,'STOP',CRESLT)
          CALL PEEKI('NTOTALS',MNT_NVX,NVX,'STOP',CRESLT, IZ)
          CALL LOCDAT('PARALLEL/KGVxVx',CDTYPE,'STOP',NKGVxVx,pKGVxVx,
     &                CRESLT)
          CALL gParFlag(ROUTIN,'VERTICES','RETURN',LParF,NParF,pParF,
     &                  CZ,DZ,IZ,LZ,RZ)
          CALL POPDIR('STOP',CRESLT)
C
C---- Get global vertex numbers and parallel overlap flag.
C
       IF ((ITERATION .EQ. 0) ) THEN

C      Send Mapping to Master

	  ALLOCATE(pISENDARRAY(NVX))
  
          numinnervertices = 0
          DO k = 0, NVX-1
            IF (RZ(pPARF+k) .GT. HALF) THEN
		numinnervertices = numinnervertices + 1
		pISENDARRAY(numinnervertices) = IZ(pKGVxVx+k)		
            ENDIF
          ENDDO
		
c          WRITE(*,*) 'Sending number of nodes to Master'
          CALL SNDIB1(numinnervertices,1,MSGTAG,.TRUE.,NERR)

c          WRITE(*,*) 'Sending Mapping ISENDARRAY to Master'         
          CALL SNDIBU(pISENDARRAY,numinnervertices,1,MSGTAG,.TRUE. 
     &                   ,NERR)
	      
	  DEALLOCATE(pISENDARRAY)    
       
	ENDIF
c        (ITERATION = 0 .and. CVAR.EQ.VEL_FL1 )	      

	      
C---- Send data to master and REMOVE OVERLAPS!!
              
	 IF (CVAR .EQ. 'VEL_FL1') THEN 
	 
	 ALLOCATE(pRSENDARRAY(NVX*3))
	 ALLOCATE(pVELx(NVX))
	 ALLOCATE(pVELy(NVX))
	 ALLOCATE(pVELz(NVX))
	 
	 numinnervertices = 0
	 DO k = 0, NVX-1
	   IF (RZ(pPARF+k) .GT. HALF) THEN
	      numinnervertices = numinnervertices + 1
	      pRSENDARRAY(3*numinnervertices-2) = VAR(3*k+1)
	      pRSENDARRAY(3*numinnervertices-1) = VAR(3*k+2)
	      pRSENDARRAY(3*numinnervertices-0) = VAR(3*k+3)
	   ENDIF
	 ENDDO
	 
c	  WRITE(*,*) 'Sending number of nodes to Master'
          CALL SNDIB1(numinnervertices,1,MSGTAG,.TRUE.,NERR)  !!!numinnervertices
	      
	 DO i=0,(numinnervertices-1)
	    pVELx(1+i)=pRSENDARRAY(3*i+1)
            pVELy(1+i)=pRSENDARRAY(3*i+2) 
	    pVELz(1+i)=pRSENDARRAY(3*i+3)
	 ENDDO
         
	   CALL SNDRBU(pVELx,numinnervertices,1,MSGTAG,.TRUE.
     &                     ,NERR)
	   CALL SNDRBU(pVELy,numinnervertices,1,MSGTAG,.TRUE.
     &                     ,NERR)
	   CALL SNDRBU(pVELz,numinnervertices,1,MSGTAG,.TRUE.
     &                     ,NERR)
	      
        CALL PSHDIR(MESH_DIR,'STOP',CRESLT)
        CALL gParFlag(ROUTIN,'VERTICES','RELEASE',LParF,NParF,pParF,
     &                  CZ,DZ,IZ,LZ,RZ)
        CALL POPDIR('STOP',CRESLT)

	 DEALLOCATE(pRSENDARRAY)
	 DEALLOCATE(pVELx)
	 DEALLOCATE(pVELy)
	 DEALLOCATE(pVELz)
	 
	 
	  
       ELSE !if cvar.EQ. PRES TEMP
	 
	 ALLOCATE(pRSENDARRAY(NVX))    
         numinnervertices = 0
           DO k = 0, NVX-1
              IF (RZ(pPARF+k) .GT. HALF) THEN
		  numinnervertices = numinnervertices + 1
		  pRSENDARRAY(numinnervertices) = VAR(k+1)
c		  pRSENDARRAY(numinnervertices) = OWN_PAR-1   send Partition ID
              ENDIF
           ENDDO
          
c	  WRITE(*,*) 'Sending number of nodes to Master'
          CALL SNDIB1(numinnervertices,1,MSGTAG,.TRUE.,NERR)
          
c          WRITE(*,*) 'Sending VAR RSENDARRAY to Master'
          CALL SNDRBU(pRSENDARRAY,numinnervertices,1,MSGTAG,.TRUE.
     &                     ,NERR)
c           WRITE(*,*) 'Sending VAR RSENDARRAY completed!'
          
	      
	CALL PSHDIR(MESH_DIR,'STOP',CRESLT)
        CALL gParFlag(ROUTIN,'VERTICES','RELEASE',LParF,NParF,pParF,
     &                  CZ,DZ,IZ,LZ,RZ)
        CALL POPDIR('STOP',CRESLT)	      
	      
	DEALLOCATE(pRSENDARRAY)
       
c	write(*,*)'End of user_covise'    
       ENDIF  !CVAR.EQ.VEL
   

C=======================================================================
C     Master in Parallel Run
C=======================================================================
C      
      ELSE IF ( (RUNMOD .EQ. __parallel_run__) .AND.
     &          (PARMOD .EQ. __master__) ) THEN
C

c     send inletVelMulti to slaves
         
        MSGTAG2=2
        DO IPARD = 2,NPARD
c	   WRITE(*,*)'Sending inletVelMulti to partition ',IPARD
	   CALL SNDRB1(inletVelMulti,IPARD,MSGTAG2,.TRUE.,NERR)	
        ENDDO

C---- Get mesh dimensions
    
        CALL PSHDIR(MESH_DIR,'STOP',CRESLT)
          CALL PEEKI('NTOTALS',MNT_NVX,NVX,'STOP',CRESLT, IZ)
          CALL PEEKI('PARALLEL/NTOTALS',MNT_NVX,NVXG,'STOP',CRESLT, IZ)
        CALL POPDIR('STOP',CRESLT)
C
C---- Dimension and number of variable components.
C
        CALL PSHDIR('/FLOW/ALGORITHM/'// ZONE //'/SYSTEM/VARIABLES',
     &              'STOP',CRESLT)
          CALL PEEKI(CVAR//'/IVAR',IONE,IVAR,'STOP',CRESLT, IZ)
          CALL PEEKI('NCOMPT',IVAR,NCOMPT,'STOP',CRESLT, IZ)
        CALL POPDIR('STOP',CRESLT)
C
        NDIM = NVAR / (NVX*NCOMPT)        

C-----------------------------------------------------------------------
C     Set Vertices from Master
C-----------------------------------------------------------------------
C
        CALL PSHDIR(MESH_DIR,'STOP',CRESLT)
C
C---- Get global vertex numbers and parallel overlap flag.
C
          CALL LOCDAT('PARALLEL/KGVxVx',CDTYPE,'STOP',NKGVxVx,pKGVxVx,
     &                CRESLT)            
          CALL gParFlag(ROUTIN,'VERTICES','RETURN',LParF,NParF,pParF,
     &                  CZ,DZ,IZ,LZ,RZ)
          CALL POPDIR('STOP',CRESLT)
C
C-----------------------------------------------------------------------
C     Collect Data from Slaves
C-----------------------------------------------------------------------
C
c send master data to covise

c          WRITE(*,*)'###### ITERATION (user_covise): ',ITERATION
c	  WRITE(*,*)'###### CVAR:                ',CVAR
C        first Iteration: send mappings to COVISE

      IF ((ITERATION .EQ. 0) ) THEN
C        send master mappings to COVISE
C        initialize parallel collect with NPARD partitions and one port (data) for now
c	 write(*,*) '!!!!!!!!!!! JOLF !!!!!!!!!!'
         testvalue = COPAIN(NPARD,1)
         if (testvalue.ne.0) then
c            WRITE(*,*) 'COPAIN error ',testvalue
            STOP
         ELSE
c           WRITE(*,*) 'Parallel data sending initialised'
         ENDIF
         
         
! C        data port = vertex data
			 
	testvalue = COPAPO(PORT,0)
	if (testvalue.ne.0) then
	WRITE(*,*) 'COPAPO error ',testvalue
	STOP
	ELSE
c	WRITE(*,*) 'port 0 is a parallel data port'
	ENDIF
	
	   
	ALLOCATE(pISENDARRAY(NVX))   
c	WRITE(*,*)'pISENDARRAY from master is built....'        
            numinnervertices = 0
            DO k = 0, NVX-1
               IF (RZ(pPARF+k) .GT. HALF) THEN
		   numinnervertices = numinnervertices + 1
		   pISENDARRAY(numinnervertices) = IZ(pKGVxVx+k)
               ENDIF
           ENDDO	   
         
          
c	  WRITE(*,*)'MASTER NVX: ',NVX,' CVAR: ',CVAR
 
C MappingArray global->lokal wird angelegt		   
	  
c          WRITE(*,*)'#### numCoord #### ',numCoord 
		 
c        WRITE(*,*) 'master numinnervertices=',numinnervertices,' CVAR: ',CVAR 
c        WRITE(*,*) 'Sending Mapping ISENDARRAY to COVISE, part 0'     
        testvalue = COPAVM(0,numinnervertices,pISENDARRAY)  
        if (testvalue.ne.0) then
           WRITE(*,*) 'COPAVM error ',testvalue
           STOP
        ELSE
c           WRITE(*,*) 'mapping array part 0 sent to COVISE'
        ENDIF            
	         
	DEALLOCATE(pISENDARRAY)
	      
	  
       DO IPARD = 2,NPARD
C
c	CALL COSLEP(1)	    
C---- Receive data from slave.
c          WRITE(*,*) 'MSGTAG: ',MSGTAG 
c          WRITE(*,*) 'Receiving Number of nodes from node ',IPARD-1
          CALL RCVIB1(NVXP,IPARD,MSGTAG,.TRUE.,NERR)
c          WRITE(*,*)'####    numinnervertives from slave: ',NVXP
c	  WRITE(*,*)'####    CVAR: ',CVAR
	  ALLOCATE(pISENDARRAY(NVXP))    
	      
c          WRITE(*,*) 'Receiving Mapping ISENDARRAY from node ',IPARD-1
          CALL RCVIBU(pISENDARRAY,NVXP,IPARD,MSGTAG,.TRUE.,NERR)
c          WRITE(*,*) 'Send. Mapp. ISENDARRAY to COVISE, part ',IPARD-1
		  
c    numCoord=numinnervertices+NVXP
	      	        
	  testvalue = COPAVM(IPARD-1,NVXP,pISENDARRAY)
          if (testvalue.ne.0) then
             WRITE(*,*) 'COPAVM error ',testvalue
             STOP
          ELSE
c             WRITE(*,*) 'mapping array part ',IPARD-1,' sent to COVISE'
          ENDIF            
          
c	  WRITE(*,*) '...completed'
	  DEALLOCATE(pISENDARRAY)
	      
        ENDDO 
c	  (IPARD)    
         
        ENDIF
C       (ENDIF: Iteration 0 and CVAR.EQ.VEL_FL1)
        

C        Each iteration send data to COVISE
C        Send data from master

         testvalue = COPANO(0)
         IF (testvalue.NE.0) THEN
            WRITE(*,*) 'COPANO error ',testvalue
         ELSE
c            write(*,*) 'COPANO(0)'
         ENDIF         
 
	
	IF (CVAR .EQ. 'VEL_FL1') THEN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	 ALLOCATE(pRSENDARRAY(NVX*3))
	 ALLOCATE(pVELx(NVX))
	 ALLOCATE(pVELy(NVX))
	 ALLOCATE(pVELz(NVX))   
	     
	 numinnervertices = 0
            DO k = 0, NVX-1
               IF (RZ(pPARF+k) .GT. HALF) THEN
		   numinnervertices = numinnervertices + 1
		   pRSENDARRAY(3*numinnervertices-2) = VAR(3*k+1)
		   pRSENDARRAY(3*numinnervertices-1) = VAR(3*k+2)
		   pRSENDARRAY(3*numinnervertices-0) = VAR(3*k+3)
	       ENDIF
            ENDDO
c         WRITE(*,*)'numinnervertices from master: ',numinnervertices,
c     &	                       ' CVAR: ',CVAR  
	 

	  DO i=0,(numinnervertices-1)
	    pVELx(1+i)=pRSENDARRAY(3*i+1)
	    pVELy(1+i)=pRSENDARRAY(3*i+2) 
	    pVELz(1+i)=pRSENDARRAY(3*i+3)
	  ENDDO  
	  
c         !!!!   inletbocos are multiplicated on master  !!!!!!  

c          WRITE(*,*) 'Sending VEL arrays to Covise, part 0'
           testvalue = COSU3D(PORT,numinnervertices,pVELx,pVELy,pVELz)
           IF (testvalue.NE.0) THEN
             WRITE(*,*) 'Lost Connection to Covise 4 ' ,testvalue
           ELSE
c             write(*,*) 'Data sent'
           ENDIF 
	       
	       
C---- Release overlap flag.
C
        CALL PSHDIR(MESH_DIR,'STOP',CRESLT)
        
        CALL gParFlag(ROUTIN,'VERTICES','RELEASE',LParF,NParF,pParF,
     &                  CZ,DZ,IZ,LZ,RZ)
        CALL POPDIR('STOP',CRESLT)	       
	       
	 DEALLOCATE(pRSENDARRAY)
	 DEALLOCATE(pVELx)
	 DEALLOCATE(pVELy)
	 DEALLOCATE(pVELz)   	
		    
	ELSE
	  
	  ALLOCATE(pRSENDARRAY(NVX))    
          numinnervertices = 0
            DO k = 0, NVX-1
               IF (RZ(pPARF+k) .GT. HALF) THEN
		   numinnervertices = numinnervertices + 1
		   pRSENDARRAY(numinnervertices) = VAR(k+1)
c		   pRSENDARRAY(numinnervertices) = 0    for Partition ID
               ENDIF
            ENDDO
c	   WRITE(*,*)'NVX from master: ',NVX,' CVAR: ',CVAR   
c	   WRITE(*,*)'numinnervertices from master: ',numinnervertices
           
c	   WRITE(*,*) 'Sending VAR RSENDARRAY to Covise, part 0'
           testvalue = COSU1D(PORT,numinnervertices,pRSENDARRAY)
           IF (testvalue.NE.0) THEN
             WRITE(*,*) 'Lost Connection to Covise 4 ' ,testvalue
           ELSE
c             write(*,*) 'Data sent'
           ENDIF
C---- Release overlap flag.
C
	 CALL PSHDIR(MESH_DIR,'STOP',CRESLT)
         CALL gParFlag(ROUTIN,'VERTICES','RELEASE',LParF,NParF,pParF,
     &                  CZ,DZ,IZ,LZ,RZ)
         CALL POPDIR('STOP',CRESLT) 
 
	 DEALLOCATE(pRSENDARRAY)
           	       
	ENDIF  !CVAR.EQ.VEL
		   


   
C        Collect and send data from slaves

         DO IPARD = 2,NPARD
	       
C---- Receive data from slave.
         
         testvalue = COPANO(IPARD-1)  
         IF (testvalue.NE.0) THEN
            WRITE(*,*) 'COPANO error ',testvalue
         ELSE
c            write(*,*) 'COPANO(',IPARD-1,')'
         ENDIF      

                  
	  
	  
      IF (CVAR .EQ. 'VEL_FL1') THEN !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	  
c	  WRITE(*,*) 'Receiving Number of nodes from slave'
	  CALL RCVIB1(NVXP,IPARD,MSGTAG,.TRUE.,NERR)
c	  WRITE(*,*) 'NVXP from slave: ',NVXP,'  CVAR: ',CVAR
c	  WRITE(*,*) '...completed'
	 ALLOCATE(pVELx(NVXP))
	 ALLOCATE(pVELy(NVXP))
	 ALLOCATE(pVELz(NVXP))
            
c	    WRITE(*,*) 'Receiving VEL arrays from slave',IPARD
            CALL RCVRBU(pVELx,NVXP,IPARD,MSGTAG,.TRUE.,NERR)
	    CALL RCVRBU(pVELy,NVXP,IPARD,MSGTAG,.TRUE.,NERR)
	    CALL RCVRBU(pVELz,NVXP,IPARD,MSGTAG,.TRUE.,NERR)  
	
c	   WRITE(*,*) 'Sending VEL arrays to Covise, part 0'
           testvalue = COSU3D(PORT,NVXP,pVELx,pVELy,pVELz)
           IF (testvalue.NE.0) THEN
             WRITE(*,*) 'Lost Connection to Covise 5 ' ,testvalue
           ELSE
c             write(*,*) 'Data sent VEL arrays to Covise'
           ENDIF 
	   DEALLOCATE(pVELx)
	   DEALLOCATE(pVELy)
	   DEALLOCATE(pVELz)
	     	    
	ELSE
	  
c	  WRITE(*,*) 'Receiving Number of nodes from slave'
	  CALL RCVIB1(NVXP,IPARD,MSGTAG,.TRUE.,NERR) !!!!!!!!!!!!!!!!!CVAR PRES
c	      WRITE(*,*) 'NVXP from slave: ',NVXP,' CVAR: ',CVAR
c	  WRITE(*,*) '...completed'
	      
	  ALLOCATE(pRSENDARRAY(NVXP))    

c         WRITE(*,*) 'Receiving VAR RSENDARRAY from slave'
          CALL RCVRBU(pRSENDARRAY,NVXP,IPARD,MSGTAG,.TRUE.,NERR)
    
c          WRITE(*,*) 'Sending VAR RSENDARRAY to Covise, part ',IPARD-1
          testvalue = COSU1D(PORT,NVXP,pRSENDARRAY)
          IF (testvalue.NE.0) THEN
             WRITE(*,*) 'Lost Connection to Covise 5 ' ,testvalue
          ELSE
c             write(*,*) 'Data sent RSENDARRAY to Covise'
          ENDIF    
	  DEALLOCATE(pRSENDARRAY)


	ENDIF !CVAR.EQ.VEL
	            
       ENDDO 
c	      (over Slaves)
C    
	      write(*,*)'Exporting: ',USER_VARS,'(',CVAR,')'	      
      ENDIF
C    (IF RUNMOD PARMOD)
      END


