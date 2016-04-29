      SUBROUTINE LES_STF(geo_name,rbe_name,erg_name,
     *                   geo_pfad,rbe_pfad,erg_pfad,
     *                   print_all,ober_geom,redu_graph,
     *                   zer_zeig,nzer,ndat_max,dopplapp,
     *                   mit_name,mit_pfad)
c   
      implicit none
      include 'mpif.h'
      include 'common.zer'
c
      integer i,luein,iread,luerr,zer_zeig,nzer,ndat_max,nnn

      parameter(luein=50)

      character*80 geo_name,rbe_name,erg_name,
     *             geo_pfad,rbe_pfad,erg_pfad,
     *             mit_name,mit_pfad

      character*80 steuerfile,zeile,text

      character*1 com         

      logical ober_geom,print_all,
     *        exist_erg,exist_geo,exist_rbe,redu_graph,dopplapp,
     *        exist_mit

      dimension zer_zeig(ndat_max)
c     *****************************************************************

c     *****************************************************************
c     INITIALISIERUNGEN:

      do 111 i=1,ndat_max
         zer_zeig(i)=0
 111  continue

      nzer=0
c     *****************************************************************


      steuerfile='zerno.stf'
      open(luein,file=steuerfile,status='unknown')


c     *****************************************************************
      read(luein,'(a)') com
      read(luein,'(a)') com
      read(luein,'(a)') com
      read(luein,'(a)') com
      read(luein,'(a)') com
      read(luein,'(a)') com
      read(luein,'(a)') com

c     DIMESNION ncd:
      read(luein,'(a)') zeile
      call dprest(zeile,text)
      ncd=iread(text)

c     read(luein,'(a)') com

      if (numprocs.gt.1) then
        parallel=.true.
      else 
        parallel=.false.
      endif

      read(luein,'(a)') com

c     ZERLEGUNG IN WIEVIEL GEBIETE:
      read(luein,'(a)') zeile
      call dprest(zeile,text)
      ngebiet=iread(text)

      if (parallel.and.ngebiet.ne.numprocs) then
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Die Anzahl Gebiete in die zerlegt wird muss '
        write(luerr,*)'gleich der Anzahl gestarteter Prozessoren sein.'
        call erro_ende(myid,parallel,luerr)
      endif

      read(luein,'(a)') com

c     FILENAME FUER  GEOMETRIE-FILES:         
      read(luein,'(a)') zeile
      call dprest(zeile,geo_name)
      if (geo_name.eq.' ') then
        exist_geo=.false.
      else
        exist_geo=.true. 
      endif
      IF (.not.exist_geo) THEN
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Es wurde kein Geometriefile angegeben!'
         call erro_ende(myid,parallel,luerr)
      ENDIF

c     FILENAME FUER RANDBEDINGUNGSFILES:         
      read(luein,'(a)') zeile
      call dprest(zeile,rbe_name)
      if (rbe_name.eq.' ') then
        exist_rbe=.false.
      else
        exist_rbe=.true. 
      endif

c     FILENAME FUER ERGEBNISFILES:               
      read(luein,'(a)') zeile
      call dprest(zeile,erg_name)
      if (erg_name.eq.' ') then
        exist_erg=.false.
      else
        exist_erg=.true. 
      endif

c     FILENAME FUER GEMITTELTE ERGEBNISFILES:               

      read(luein,'(a)') zeile
      call dprest(zeile,mit_name)
      if (mit_name.eq.' ') then
        exist_mit=.false.
      else
        exist_mit=.true. 
      endif

      read(luein,'(a)') com

c     PFADNAME FUER GEOMETRIE-FILES:         
      read(luein,'(a)') zeile
      call dprest(zeile,geo_pfad)

c     PFADNAME FUER RANDBEDINGUNGSFILES:         
      read(luein,'(a)') zeile
      call dprest(zeile,rbe_pfad)

c     PFADNAME FUER ERGEBNISFILES:               
      read(luein,'(a)') zeile
      call dprest(zeile,erg_pfad)

c     PFADNAME FUER GEMITTELTE ERGEBNISFILES:               
      read(luein,'(a)') zeile
      call dprest(zeile,mit_pfad)

      read(luein,'(a)') com

c     LIEGT EINE PARTITION BEREITS VOR ODER NICHT:
      read(luein,'(a)') zeile
      call dprest(zeile,text)
      if (text(1:2).eq.'ja') then
        parti_geo=.false.
        parti_les=.true. 
      else if (text(1:2).eq.'ne') then
        parti_geo=.true. 
        parti_les=.false.
      else
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Falsche Eingabe bzgl. Vorhandensein einer '  
        write(luerr,*)'Partition unter dem Geometrie-Pfad ! '  
        call erro_ende(myid,parallel,luerr)
      endif


c     ZERLEGUNG DER RANDBEDINGUNGEN MIT DIESER PARTITION:
      read(luein,'(a)') zeile
      call dprest(zeile,text)

      IF (parti_les) THEN
         if (text(1:2).eq.'ja') then
           parti_rbe=.true. 
         else if (text(1:2).eq.'ne') then
           parti_rbe=.false.
         else
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Falsche Eingabe bzgl. Zerlegung der       '  
           write(luerr,*)'Randbedingungen mit vorhandener Partition '
           call erro_ende(myid,parallel,luerr)
         endif
      ELSE  
         parti_rbe=.true.
      ENDIF


c     ZERLEGUNG DER ERGEBNISSE MIT DIESER PARTITION:
      read(luein,'(a)') zeile
      call dprest(zeile,text)

      IF (parti_les) THEN
         if (text(1:2).eq.'ja') then
           parti_erg=.true. 
         else if (text(1:2).eq.'ne') then
           parti_erg=.false.
         else
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Falsche Eingabe bzgl. Zerlegung der       '  
           write(luerr,*)'Ergebnisse  mit vorhandener Partition '
           call erro_ende(myid,parallel,luerr)
         endif
      ELSE  
         parti_erg=.true.
      ENDIF

c     ZERLEGUNG DER GEMITTELTE ERGEBNISSE MIT DIESER PARTITION:
      read(luein,'(a)') zeile
      call dprest(zeile,text)

      IF (parti_les) THEN
         if (text(1:2).eq.'ja') then
           parti_mit=.true. 
         else if (text(1:2).eq.'ne') then
           parti_mit=.false.
         else
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Falsche Eingabe bzgl. Zerlegung der       '  
           write(luerr,*)'gemitt. Ergebnisse  mit vorhandener Partition'
           call erro_ende(myid,parallel,luerr)
         endif
c del     ELSE  
c del        parti_mit=.true.
      ENDIF

      read(luein,'(a)') com
      read(luein,'(a)') com
      read(luein,'(a)') com

      read(luein,'(a)') zeile
      call dprest(zeile,text)
      nzer=nzer+1
      if (text(1:2).eq.'ja') then
         zer_zeig(nzer)=1
      else if (text(1:2).eq.'ne') then
         zer_zeig(nzer)=0   
      else
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Falsche Eingabe bei der Auswahl der     '  
        write(luerr,*)'METIS-Routinen.                           '
        call erro_ende(myid,parallel,luerr)
      endif

      read(luein,'(a)') zeile
      call dprest(zeile,text)
      nzer=nzer+1
      if (text(1:2).eq.'ja') then
         zer_zeig(nzer)=1
      else if (text(1:2).eq.'ne') then
         zer_zeig(nzer)=0   
      else
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Falsche Eingabe bei der Auswahl der     '  
        write(luerr,*)'METIS-Routinen.                           '
        call erro_ende(myid,parallel,luerr)
      endif

      read(luein,'(a)') zeile
      call dprest(zeile,text)
      nzer=nzer+1
      if (text(1:2).eq.'ja') then
         zer_zeig(nzer)=1
      else if (text(1:2).eq.'ne') then
         zer_zeig(nzer)=0   
      else
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Falsche Eingabe bei der Auswahl der     '  
        write(luerr,*)'METIS-Routinen.                           '
        call erro_ende(myid,parallel,luerr)
      endif

      read(luein,'(a)') com


c     ZERLEGUNG AUF REDUZIERTEM GRAPH:
      read(luein,'(a)') zeile
      call dprest(zeile,text)
      if (text(1:2).eq.'ja') then
        redu_graph=.true.
      else if (text(1:2).eq.'ne') then
        redu_graph=.false.
      else
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Falsche Eingabe bzgl. Zerlegung auf reduziertem '
        write(luerr,*)'Graph.                      '
        call erro_ende(myid,parallel,luerr)
      endif


      read(luein,'(a)') com
      read(luein,'(a)') com
      read(luein,'(a)') com


c     AUSDRUCK DER DATEN:
      read(luein,'(a)') zeile
      call dprest(zeile,text)
      if (text(1:2).eq.'ja') then
        print_all=.true.
      else if (text(1:2).eq.'ne') then
        print_all=.false.
      else
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Falsche Eingabe bzgl. Ausdruck von Geometrie '  
        write(luerr,*)'und Randbedingungen  !!            '            
        call erro_ende(myid,parallel,luerr)
      endif

      read(luein,'(a)') com

      read(luein,'(a)') zeile
      call dprest(zeile,text)
      if (text(1:2).eq.'ja') then
        ober_geom=.true.
      else if (text(1:2).eq.'ne') then
        ober_geom=.false.
      else
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Falsche Eingabe bzgl. Ausdruck der '            
        write(luerr,*)'Oberflaechengeomtrie !!            '            
        call erro_ende(myid,parallel,luerr)
      endif

      read(luein,'(a)') com
      read(luein,'(a)') com
      read(luein,'(a)') com

c     DOPPELTE UEBERLAPPUNG:
      read(luein,'(a)') zeile
      call dprest(zeile,text)
      if (text(1:2).eq.'ja') then
        dopplapp=.true.
      else if (text(1:2).eq.'ne') then
        dopplapp=.false.
      else
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Falsche Eingabe bzgl. doppelter Ueberlappung '  
        call erro_ende(myid,parallel,luerr)
      endif

      read(luein,'(a)') com
      read(luein,'(a)') com
      read(luein,'(a)') com
c     *****************************************************************

      close(luein)


c     *****************************************************************
c     SETZEN DER ZERLEGUNGSFLAGS FUER RANDBEDINGUNGEN UND ERGEBNISSE:

      IF (parti_geo) THEN
         parti_rbe=.false.
         parti_erg=.false.
         parti_mit=.false.
         if (exist_rbe) then
            parti_rbe=.true.
         endif
         if (exist_erg) then
            parti_erg=.true.
         endif
         if (exist_mit) then
            parti_mit=.true.
         endif

      ELSE IF (parti_les.and.parti_erg) THEN
         if (.not. exist_erg) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Es wurde kein Ergebnisfile angegeben!'
            call erro_ende(myid,parallel,luerr)
         endif

      ELSE IF (parti_les.and.parti_rbe) THEN
         if (.not. exist_rbe) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Es wurde kein Randbedingungsfile angegeben!'
            call erro_ende(myid,parallel,luerr)
         endif
         
      ENDIF 

      if (parti_geo) then
         nnn=0
         do 155 i=1,nzer
            if (zer_zeig(i).eq.1) then
               nnn=nnn+1
            endif 
 155     continue
         if (nnn.eq.0) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Es wurde keine METIS-Routine ausgesucht. '
            call erro_ende(myid,parallel,luerr)
         endif
      endif
c     *****************************************************************

c     *****************************************************************
c     FEHLERMELDUNGEN:

      if (ober_geom) then
c        call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine LES_STF'
         write(luerr,*)'Die Ausgabe der Oberflaechengeometrie ist '
         write(luerr,*)'zwar implementiert aber nicht getestet.   '
c        call erro_ende(myid,parallel,luerr)
      endif
c     *****************************************************************

      return
      end


