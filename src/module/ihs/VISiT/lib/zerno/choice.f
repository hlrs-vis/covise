C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE CHOICE(help_dat,help_num,help_neu,nnn,
     *                  parti_error,nparti)
c
      implicit none

      include 'common.zer'
c
      integer   help_dat,help_num,help_neu,nnn,
     *          parti_error,nparti

      integer i,ipar,nnn_old,par_min,anz_min

      dimension help_dat(nparti),help_num(nparti),help_neu(nparti),
     *          parti_error(nparti)
c     ****************************************************************


c     Diese Routine sortiert die Daten auf Feld help_dat in
c     aufsteigender Reihenfolge und fuehrt das dazugehoerige
c     Nummernfeld help_num mit. Ausserdem bestimmt die Routine
c     wie oft der kleinste Eintrag auf Feld help_dat vorkommt.
c     Die Nummern dieser Eintraege werden hintereinander auf
c     Feld help_num geschrieben.


c     ****************************************************************
c     AUSWAHL NACH DER KLEINSTEN ANZAHL AUF FELD help_dat:

      nnn_old=nnn 

c     Bestimmung des kleinsten zulaessigen Wertes auf Feld help_dat
      anz_min=1000000000
      do 100 i=1,nnn_old
         ipar=help_num(i)
         if (parti_error(ipar).eq.0) then
            if (help_dat(i).lt.anz_min) then
               anz_min=help_dat(i)
               par_min=ipar             
            endif
         endif
 100  continue


c     Bestimmung wie oft dieser Wert auf Feld help_dat vorkommt:    
      nnn=0
      do 200 i=1,nnn_old
         ipar=help_num(i)
         if (parti_error(ipar).eq.0) then
            if (help_dat(i).eq.anz_min) then
               nnn=nnn+1
               help_neu(nnn)=ipar
            endif
         endif
 200  continue

      do 300 i=1,nnn
         help_num(i)=help_neu(i)
 300  continue
c     ****************************************************************

      return
      end


