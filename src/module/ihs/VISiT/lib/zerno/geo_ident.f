C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE GEO_IDENT(zeile,file_name,geo_new_format,
     *                     geo_old_format,geo_cut_format,
     *                     para_format)

      implicit none

      include 'common.zer'      

      integer i,j,k,iii,luerr,ilang,lentb,iz_sta,iz_end

      integer zeich_anz,blank_anz,ziffe_anz,
     *        nzeich,nblank,nziffe,nchar,nnn,nnn_ziffe,nnn_zeich

      integer muster_les,muster_cut,muster_old,muster_new,
     *        nhelp_max,nzahl,nzahl_read,
     *        nzahl_cut,nzahl_old,nzahl_new,nzahl_max

      real    rdum,rread

      logical zeich_flag,ziffe_flag,blank_flag,
     *        b_flag,found,para_format

      logical geo_new_format,geo_old_format,geo_cut_format

      parameter (nhelp_max=20)
      
      character*80  comment,file_name,word,zahl_text
      character*100 zeile,text

      character*1  zeich_text,blank_text,ziffe_text

      dimension    zeich_anz(nhelp_max),ziffe_anz(nhelp_max),
     *             blank_anz(nhelp_max)

      dimension    zeich_text(nhelp_max),ziffe_text(nhelp_max),
     *             blank_text(nhelp_max),
     *             zahl_text(nhelp_max)

      dimension    muster_les(nhelp_max),muster_old(nhelp_max),
     *             muster_new(nhelp_max),muster_cut(nhelp_max)
c     ****************************************************************

c     zeile='       -1234,  4.56 , 1e+4 10.54e+2  345  00 '
c     zeile='                                      '

c     ****************************************************************
c     INITIALISIERUNGEN:

      do 11 i=1,nhelp_max
         zeich_anz(i)=0
         ziffe_anz(i)=0
         blank_anz(i)=0
         
         muster_les(i)=0
         muster_old(i)=0
         muster_new(i)=0
         muster_cut(i)=0
 11   continue

      nblank=2
      blank_text(1)=' '
      blank_text(2)=','

      nziffe=12
      ziffe_text(1)='0'
      ziffe_text(2)='1'
      ziffe_text(3)='2'
      ziffe_text(4)='3'
      ziffe_text(5)='4'
      ziffe_text(6)='5'
      ziffe_text(7)='6'
      ziffe_text(8)='7'
      ziffe_text(9)='8'
      ziffe_text(10)='9'
      ziffe_text(11)='+'
      ziffe_text(12)='-'

      nzeich=4
      zeich_text(1)='.'
      zeich_text(2)='e'
      zeich_text(3)='d'
      zeich_text(4)='E'
c     zeich_text(4)='+'
c     zeich_text(5)='-'

      nchar=100
c     ****************************************************************


c     ****************************************************************
c     BELEGUNG DER ERKENNUNGSMUSTER:

c     Markierung mit 1 -> Integer
c     Markierung mit 2 -> Real       


c     Muster des alten Flow-Geometrie-Files:
      if (para_format) then
         nzahl_old=6
         muster_old(1) =1 
         muster_old(2) =2 
         muster_old(3) =2 
         muster_old(4) =2 
         muster_old(5) =1 
         muster_old(6) =1 
      else
         nzahl_old=4
         muster_old(1) =1 
         muster_old(2) =2 
         muster_old(3) =2 
         muster_old(4) =2 
      endif

c     Muster des neuen Flow-Geometrie-Files:
      if (para_format) then
         nzahl_new=7
         muster_new(1) =1 
         muster_new(2) =2 
         muster_new(3) =2 
         muster_new(4) =2 
         muster_new(5) =1 
         muster_new(6) =1 
         muster_new(7) =1 
      else
         nzahl_new=5
         muster_new(1) =1 
         muster_new(2) =2 
         muster_new(3) =2 
         muster_new(4) =2 
         muster_new(5) =1 
      endif

c     Muster des Schnittgitter-Files: 
      nzahl_cut=10
      muster_cut(1) =1 
      muster_cut(2) =2 
      muster_cut(3) =2 
      muster_cut(4) =2 
      muster_cut(5) =2 
      muster_cut(6) =2 
      muster_cut(7) =2 
      muster_cut(8) =1 
      muster_cut(9) =1 
      muster_cut(10)=1 

c     ****************************************************************

c     ****************************************************************
c     ANALYSE DER ZEILE:                                

      ilang=lentb(zeile)
c     write(6,*) ilang
      if (ilang.eq.0) then
         comment='Fehler passiert in File:'
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine GEO_IDENT                '
         write(luerr,*)'Die zu analysierende Zeile ist eine Leerzeile.'
         call char_druck(comment,file_name,luerr)
         call erro_ende(myid,parallel,luerr)
      endif

c     write(6,*) zeile(1:50)

      nnn=0
      do 100 i=1,nchar
         do 120 k=1,nblank
            text=blank_text(k)
            if (zeile(i:i).eq.text) then
               blank_anz(k)=blank_anz(k)+1
               nnn=nnn+1
            endif
 120     continue
         do 110 k=1,nzeich
            text=zeich_text(k)
            if (zeile(i:i).eq.text) then
               zeich_anz(k)=zeich_anz(k)+1
               nnn=nnn+1
            endif
 110     continue
         do 130 k=1,nziffe
            text=ziffe_text(k)
            if (zeile(i:i).eq.text) then
               ziffe_anz(k)=ziffe_anz(k)+1
               nnn=nnn+1
            endif
 130     continue
 100  continue


      if (nnn.ne.nchar) then
         comment='Fehler passiert in File:'
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine GEO_IDENT     '
         write(luerr,*)'Die zu analysierende Zeile '
         ilang=lentb(zeile)
         write(luerr,'(A)') zeile(1:ilang)
         write(luerr,*)'enthaelt unzulaessige Zeichen.'
         write(luerr,*)'Zulaessige Zeichen sind: '
c        write(luerr,'(1x,30(A1,1x))') (blank_text(i),i=2,nblank)
         write(luerr,'(1x,30(A1,1x))') (ziffe_text(i),i=1,nziffe)
         write(luerr,'(1x,30(A1,1x))') (zeich_text(i),i=1,nzeich),
     *                                 (blank_text(i),i=2,2)
         call char_druck(comment,file_name,luerr)
         call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************

c     ****************************************************************
c     BESTIMMUNG DER ANZAHL ZAHLEN:

      text=zeile
      nzahl_read=0
      do 50 i=1,nchar
         ilang=lentb(text)
         if (ilang.eq.0) then
           goto 51 
         endif
         rdum=rread(text)

c        write(6,*) i,ilang,rdum

         nzahl_read=nzahl_read+1
 50   continue

 51   continue

c     write(6,*)'nzahl_read=',nzahl_read
c     ****************************************************************

c     ****************************************************************
c     IDENTIFIKATION DER ZAHLEN:

      iz_sta=1
      iz_end=nchar

      nzahl=0
      do 400 iii=1,nchar

         do 200 i=iz_sta,iz_end  

            blank_flag=.false.
            do 220 k=1,nblank
               text=blank_text(k)
               if (zeile(i:i).eq.text) then
                  blank_flag=.true.
               endif
 220        continue

            IF (.not.blank_flag) THEN
c              Kontrolle ob das Zeichen eine Ziffer ist:
               ziffe_flag=.false.
               do 230 k=1,nziffe
                  text=ziffe_text(k)
                  if (zeile(i:i).eq.text) then
                     ziffe_flag=.true.
                  endif
 230           continue

c              Kontrolle ob das Zeichen ein Real-Zeichen ist.
               zeich_flag=.false.
               do 240 k=1,nzeich 
                  text=zeich_text(k)
                  if (zeile(i:i).eq.text) then
                     zeich_flag=.true.
                  endif
 240           continue

               if (.not.ziffe_flag .and. .not. zeich_flag) then
                 comment='Fehler passiert in File:'
                 call erro_init(myid,parallel,luerr)
                 write(luerr,*)'Fehler in Routine GEO_IDENT     '
                 write(luerr,*)'Die zu analysierende Zeile '
                 ilang=lentb(zeile)
                 write(luerr,'(A)') zeile(1:ilang)
                 write(luerr,*)'enthaelt unzulaessige Zeichen.'
                 write(luerr,'(A23,A1)')' Unzulaessiges Zeichen:',
     *                                    zeile(i:i)
                 call char_druck(comment,file_name,luerr)
                 call erro_ende(myid,parallel,luerr)
               endif

c              Suchen des naechsten Blanks -> Beschriftung mit Zahl
               nnn=0
               nnn_ziffe=0
               nnn_zeich=0
               do 300 j=i,nchar    

c                 Kontrolle ob das Zeichen ein Blank ist:
                  b_flag=.false.
                  do 320 k=1,nblank
                     text=blank_text(k)
                     if (zeile(j:j).eq.text) then
                        b_flag=.true.
                     endif
 320              continue

                  if (.not. b_flag) then
c                    Kontrolle ob das Zeichen eine Ziffer ist:
                     found=.false.
                     do 330 k=1,nziffe
                        text=ziffe_text(k)
                        if (zeile(j:j).eq.text) then
                           nnn_ziffe=nnn_ziffe+1
                           nnn=nnn+1
                           word(nnn:nnn)=zeile(j:j)
                           found=.true.
                           goto 331
                        endif
 330                 continue
 331                 continue

c                    Kontrolle ob das Zeichen ein Real-Zeichen ist.
                     if (.not.found) then
                        do 340 k=1,nzeich 
                           text=zeich_text(k)
                           if (zeile(j:j).eq.text) then
                              nnn_zeich=nnn_zeich+1
                              nnn=nnn+1
                              word(nnn:nnn)=zeile(j:j)
                              found=.true.
                              goto 341
                           endif
 340                    continue
 341                    continue
                     endif

                     if (.not.found) then
                       comment='Fehler passiert in File:'
                       call erro_init(myid,parallel,luerr)
                       write(luerr,*)'Fehler in Routine GEO_IDENT     '
                       write(luerr,*)'Das ',j,'-te Zeichen ist    '
                       write(luerr,*)'unbekannt                  '
                       write(luerr,'(A)') zeile(1:ilang)
                       write(luerr,'(A21,A1)')' Unbekanntes Zeichen:',
     *                                          zeile(j:j)
                       call char_druck(comment,file_name,luerr)
                       call erro_ende(myid,parallel,luerr)
                     endif

                  else
c                   Das Zeichen ist ein Blank -> Zahl ist bestimmt.
                    goto 301
                  endif
               
 300           continue

 301           continue
            
               nzahl=nzahl+1
               if (nnn_zeich.eq.0) then 
c                 Integer-Zahl
                  muster_les(nzahl)=1
               else 
c                 Real-Zahl   
                  muster_les(nzahl)=2
               endif
               if (nnn_zeich.eq.0.and.nnn_ziffe.eq.0) then
                  comment='Fehler passiert in File:'
                  call erro_init(myid,parallel,luerr)
                  write(luerr,*)'Fehler in Routine GEO_IDENT     '
                  write(luerr,*)'Die ',nzahl,'-te Zahl von Zeile    '
                  write(luerr,'(A)') zeile(1:ilang)
                  write(luerr,*)'wurde nicht identifiziert. '
                  write(luerr,*) word(1:nnn)                    
                  call char_druck(comment,file_name,luerr)
                  call erro_ende(myid,parallel,luerr)
               endif

c              if (nnn.ne.0) then
c                 write(6,*) i,j,' word=',word(1:nnn)
c              endif

               zahl_text(nzahl)=word(1:nnn)

               if (nzahl.eq.nzahl_read) then
c                 Alle Zahlen sind identifiziert.
                  goto 401
               endif

               iz_sta=j 
               goto 201

            ENDIF
 200     continue
 201     continue

 400  continue           

      comment='Fehler passiert in File:'
      call erro_init(myid,parallel,luerr)
      write(luerr,*)'Fehler in Routine GEO_IDENT     '
      write(luerr,*)'Schleife 400 wurde vollstaendig durchlaufen '
      write(luerr,*)'ohne dass alle Zahlen identifiziert wurden. '
      write(luerr,'(A)') zeile(1:ilang)
      write(luerr,*)'nzahl_read=',nzahl_read     
      write(luerr,*)'nzahl     =',nzahl          
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)

 401  continue
c     ****************************************************************

c     ****************************************************************
c     ANALYSE DES MUSTERS:

      geo_old_format=.false.
      geo_new_format=.false.
      geo_cut_format=.false.

      nzahl_max=MAX(nzahl_old,nzahl_new,nzahl_cut)

      nnn=0
      do 501 i=1,nzahl_max
         if (muster_les(i).eq.muster_old(i)) then
           nnn=nnn+1
         endif
 501  continue
      if (nnn.eq.nzahl_max) then
         geo_old_format=.true. 
         goto 555
      endif

      nnn=0
      do 502 i=1,nzahl_max
         if (muster_les(i).eq.muster_new(i)) then
           nnn=nnn+1
         endif
 502  continue
      if (nnn.eq.nzahl_max) then
         geo_new_format=.true.
         goto 555
      endif

      nnn=0
      do 503 i=1,nzahl_max
         if (muster_les(i).eq.muster_cut(i)) then
           nnn=nnn+1
         endif
 503  continue
      if (nnn.eq.nzahl_max) then
         geo_cut_format=.true.
         goto 555
      endif


      comment='Fehler passiert in File:'
      call erro_init(myid,parallel,luerr)
      write(luerr,*)'Fehler in Routine GEO_IDENT     '
      write(luerr,*)'Das Muster von Zeile '
      ilang=lentb(zeile)
      write(luerr,'(A)') zeile(1:ilang)
      write(luerr,*)'kann keinem Geometrieformat zugewiesen werden.'
      write(luerr,*)'                                           '
      write(luerr,*)'Eingelesenes Muster   :',(muster_les(k),k=1,nzahl)
      write(luerr,*)'Muster alter Flow-File:',
     *                                 (muster_old(k),k=1,nzahl_old)
      write(luerr,*)'Muster neuer Flow-File:',
     *                                 (muster_new(k),k=1,nzahl_new)
      write(luerr,*)'Muster Schnittgeometie:',
     *                                 (muster_cut(k),k=1,nzahl_cut)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)


 555  continue
c     ****************************************************************

c     write(6,*) zeile(1:50)
c     write(6,*) (muster_les(k),k=1,nzahl)
c     stop

      return
      end


