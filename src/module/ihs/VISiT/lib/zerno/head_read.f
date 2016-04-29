C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE HEAD_READ(lu,file_name,format_read,zeile_dim)

      implicit none

      include 'common.zer'      

      integer lu,i,k,nhead_max,luerr,nnn_zeich,nnn_blank,nnn_text,
     *        nnn_ziff,ilang,lentb

      logical format_read,kommentar

      parameter (nhead_max=1000)
      
      character*80 comment,zeile_dim,file_name
      character*100 text                            
      character*1  zeich_text,blank_text
c     ****************************************************************


c     ****************************************************************
c     FEHLERMELDUNGEN:

      comment='Fehler passiert in File:'
      if (.not.format_read) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine HEAD_READ                '
         write(luerr,*)'Unformatiertes Einlesen des Headers wurde  '
         write(luerr,*)'noch nie getestet.                         '
         call char_druck(comment,file_name,luerr)
         call erro_ende(myid,parallel,luerr)
      endif                       
c     ****************************************************************


c     ****************************************************************
c     EINLESEN DES HEADERS:                             

      zeich_text='#'
      blank_text=' '

      do 11 i=1,nhead_max
         if (format_read) then
            read(lu,'(A)') text
         else
            read(lu) text
         endif

c        Auswertung von text:
         nnn_zeich=80
         nnn_blank=0
         kommentar=.false.
         do 12 k=1,nnn_zeich
            if (text(k:k).eq.blank_text(1:1)) then
               nnn_blank=nnn_blank+1
            else 
               if (text(k:k).eq.zeich_text(1:1)) then
                  kommentar=.true.
               endif
            endif
 12      continue

         if (nnn_blank.eq.nnn_zeich) then
c           Leere Zeile
            kommentar=.true.
         endif

         if (.not.kommentar) then                   
c           Zeile text ist die Dimensionszeile:
            zeile_dim(1:80)=text(1:80)
            goto 99
         endif

 11   continue

      comment='Fehler passiert in File:'
      call erro_init(myid,parallel,luerr)
      write(luerr,*)'Fehler in Routine HEAD_READ                '
      write(luerr,*)'Es wurden ',nhead_max,' Kommentarzeilen    '
      write(luerr,*)'eingelesen, ohne die Dimensionszeile zu finden.'
      call char_druck(comment,file_name,luerr)
      write(luerr,*)'Parameter nhead_max in Routine HEAD_READ erhoehen.'
      call erro_ende(myid,parallel,luerr)

 99   continue

c     Dimensionszeile darf nur Ziffern enthalten.
      nnn_ziff=0
      nnn_text=0
      do 15 k=1,80
         if (zeile_dim(k:k).ne.blank_text(1:1)) then
            if (zeile_dim(k:k).eq.'0') then
              nnn_ziff=nnn_ziff+1
            else if (zeile_dim(k:k).eq.'1') then
              nnn_ziff=nnn_ziff+1
            else if (zeile_dim(k:k).eq.'2') then
              nnn_ziff=nnn_ziff+1
            else if (zeile_dim(k:k).eq.'3') then
              nnn_ziff=nnn_ziff+1
            else if (zeile_dim(k:k).eq.'4') then
              nnn_ziff=nnn_ziff+1
            else if (zeile_dim(k:k).eq.'5') then
              nnn_ziff=nnn_ziff+1
            else if (zeile_dim(k:k).eq.'6') then
              nnn_ziff=nnn_ziff+1
            else if (zeile_dim(k:k).eq.'7') then
              nnn_ziff=nnn_ziff+1
            else if (zeile_dim(k:k).eq.'8') then
              nnn_ziff=nnn_ziff+1
            else if (zeile_dim(k:k).eq.'9') then
              nnn_ziff=nnn_ziff+1
            else
              nnn_text=nnn_text+1
            endif
         endif
 15   continue

      if (nnn_text.ne.0) then
         ilang=lentb(zeile_dim)
         comment='Fehler passiert in File:'
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine HEAD_READ                '
         write(luerr,*)'Die identifizierte Dimensionszeile         '
         write(luerr,'(A)') zeile_dim(1:ilang)
         if (nnn_text.eq.1) then
         write(luerr,*)'enthaelt ',nnn_text,' unzulaessiges Zeichen.'
         else 
         write(luerr,*)'enthaelt ',nnn_text,' unzulaessige Zeichen.'
         endif
         call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************


c      write(6,*) 'Kontrolle ob alle   Zeichen eine Ziffer ist.'

c        write(6,'(A)') zeile_dim(1:60)

      return
      end
