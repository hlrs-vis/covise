C**********************************************************************
      PROGRAM MINISIM
C**********************************************************************
      PARAMETER (IDIM=5)
      PARAMETER (ISIZE=IDIM+2)

      REAL*4 T(ISIZE,ISIZE,ISIZE)
      CHARACTER*256 DIR
C --- INCLUDE 'coSimLib.inc'
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

C --- WSAStartup (Windows)
      IF (COVWSI().NE.0) THEN
         WRITE (*,*) 'WSAStartup failed'
         STOP
      ENDIF
      
C --- This creates the connection with Covise...
      IF (COVINI().NE.0) THEN
         WRITE (*,*) 'Could not connect to Covise'
         STOP
      ENDIF

C --- Preset all values to 0
      
      DO 10 I=1,ISIZE    
         DO 10 J=1,ISIZE    
            DO 10 K=1,ISIZE
               T(I,J,K)=0
 10   CONTINUE            
 
      ICOUNT=0
 
C --- Here is the start of the main simulation loop

 20   CONTINUE
 
C --- Boundary condition settings : insert here for interaction

      IF (COGPSL('Value111',dum1,dum2,T111).ne.0) THEN
         WRITE(*,*) 'Could not get BOCO: Value111 '
      ELSE
         WRITE(*,*) 'RECEIVED BOCO:',T111
      ENDIF

      IF (COGPSL('Value117',dum1,dum2,T117).ne.0) THEN
         WRITE(*,*) 'Could not get BOCO: Value117 '
      ELSE
         WRITE(*,*) 'RECEIVED BOCO:',T117
      ENDIF

      IF (COGPSL('Value777',dum1,dum2,T777).ne.0) THEN
         WRITE(*,*) 'Could not get BOCO: Value777 '
      ELSE
         WRITE(*,*) 'RECEIVED BOCO:',T777
      ENDIF

      IF (COGPFL('relax',RELAX).ne.0) THEN
         WRITE(*,*) 'Could not get RELAX'
      ELSE
         WRITE(*,*) 'RECEIVED RELAX:',RELAX
      ENDIF

      IF (COGPIN('steps',ISTEPS).ne.0) THEN
         WRITE(*,*) 'Could not get ISTEPS'
      ELSE
         WRITE(*,*) 'RECEIVED ISTEPS:',ISTEPS
      ENDIF
      
      IF (COGPTX('dir',DIR).ne.0) THEN
         WRITE(*,*) 'Could not get DIR'
      ELSE
         WRITE(*,*) 'RECEIVED DIR:',DIR
      ENDIF
      
c ----------- test immediate : 
 200  IF (COGPBO('pause',IBOOL).ne.0) THEN
         WRITE(*,*) 'Could not get pause param'
      ELSE
         IF (IBOOL.eq.0) then
            WRITE(*,*) 'RECEIVED pause: FALSE'
         ELSE
            WRITE(*,*) 'RECEIVED pause: TRUE'
            call sleep(2)
            goto 200
         ENDIF
      ENDIF
      
      IF (COGPCH('choice',ICHOI).ne.0) THEN
         WRITE(*,*) 'Could not set  choice'
      ELSE
         WRITE(*,*) 'RECEIVED choice:',ICHOI
      ENDIF
      
      if (ISTEPS.le.1 .or. ISTEPS.gt.10000) ISTEPS=10
      if (RELAX .le. 0.0e0 .or. RELAX .ge. 1.0e0) RELAX=0.5
      
C --- Here is the inner loop, to be executed for convergence

 30   CONTINUE

      DO 60 ILOOP=1,ISTEPS

 
      RES = 0.0
      
      DO 40 I=2,ISIZE-1   
         DO 40 J=2,ISIZE-1    
            DO 40 K=2,ISIZE-1

               T(1,1,1) = T111
               T(1,1,7) = T117
               T(7,7,7) = T777
               T1 = 0
               NUM=0
               DO 50 IX=-1,1
                  DO 50 IY=-1,1
                     DO 50 IZ=-1,1
                        if (i+ix.GE.1.and.i+ix.LE.7.and.
     +                      j+iy.GE.1.and.j+iy.LE.7.and.
     +                      k+iz.GE.1.and.k+iz.LE.7 ) THEN

                           T1 = T1 + T(i+ix,j+iy,k+iz)
                           NUM = NUM+1

                        ENDIF
 50            CONTINUE              
               T1 = T1 / NUM
               RES = RES + ABS( (T(i,j,k) - T1)/T1 )
               T(i,j,k) = T1 * RELAX + T(i,j,k) * (1.0-RELAX)
 40   CONTINUE

 60   CONTINUE   
      
      WRITE(*,*) '#####',res,T(1,1,1),T(2,2,2)
      call sleep(2)
      
C --- Send our data to Covise

      if (COSU1D('data',7*7*7,T).ne.0) then
         WRITE(*,*) 'Lost Connection to Covise'
         STOP
      ELSE
         WRITE(*,*) 'SENT FINISH MESSAGE:'
      ENDIF

      iii=COATTR('data','name','value')
            
C --- Tell Covise that this run is ready

      if (COFINI().ne.0) then
         WRITE(*,*) 'Lost Connection to Covise'
         STOP
      ELSE
         WRITE(*,*) 'SENT FINISH MESSAGE:'
      ENDIF

C --- Tell Covise to run itself again

      if (COEXEC().ne.0) then
         WRITE(*,*) 'Lost Connection to Covise'
         STOP
      ELSE
         WRITE(*,*) 'SENT EXEC MESSAGE:'
      ENDIF
      
      ICOUNT = ICOUNT + 1
      
C --- check whether we are still connected      
      if (CONOCO().NE.0) stop
      
      if (ICOUNT.LT.1000) GOTO 20
      
      END
