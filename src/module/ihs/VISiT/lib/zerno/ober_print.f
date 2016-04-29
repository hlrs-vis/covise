C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE OBER_PRINT(coord,coord_num,
     *                      elfla_adr,elfla_kno,
     *                      nelfla,nelfla_max,nkd_obe,
     *                      farb_geb,farb_adr,nfarb_geb,perm_geb_inv,
     *                      geb_num,geb_adr,col_num,col_adr,
     *                      zeig,folg,
     *                      ober_name_geo,ober_name_ses)
  
      implicit none     

      include 'common.zer'

      integer    elfla_adr,elfla_kno,coord_num,
     *           nelfla_max,nelfla,zeig,folg,
     *           farb_geb,farb_adr,nfarb_geb,perm_geb_inv,
     *           ifarb,igeb,iii,ncofla,kn_num,ipoin,nkd_obe,
     *           geb_num,geb_adr,col_num,col_adr,icol

      integer    i,j,k,lu,icom,lentb,inam,nnn_geb,nnn_col,num_1,num_2,
     *           anz,nfarb_patran,help(8),kn_max

      real       coord

      character*80 ober_name_geo,ober_name_ses,
     *             zeil_1,zeil_2,comment

      character*85 zeile1,zeile2,zeile3,zeile4,zeile5,zeile6,
     *             zeile7,zeile8

      character*10 gruppe
      character*4  otto   

      logical session_print

      parameter (lu=77,nfarb_patran=14)

      dimension elfla_adr(ngebiet+1),elfla_kno(nelfla_max,nkd_obe),
     *          farb_geb(ngebiet),farb_adr(ngebiet+1),
     *          perm_geb_inv(ngebiet),coord_num(npoin_max),
     *          geb_num(ngebiet+1),geb_adr(ngebiet+1),
     *          col_num(ngebiet+1),col_adr(ngebiet+1),
     *          coord(npoin_max,ncd),zeig(npoin_max),folg(npoin_max)
c     *****************************************************************


c     *****************************************************************
c     FARBBESTIMMUNG FUER AUSGABE:

      session_print=.true.

      if (ngebiet.le.nfarb_patran) then
c        Fuer alle Gebiet gibt es genuegend Farben -> Farben aus
c        Colorierung werden nicht verwendet

         nfarb_geb=ngebiet 
         farb_adr(1)=1
         do 100 i=1,ngebiet
           perm_geb_inv(i)=i
           farb_geb(i)=i
           farb_adr(i+1)=farb_adr(i)+1
 100     continue
      else 
         if (nfarb_geb.gt.nfarb_patran) then
            if (myid.eq.0) then
               write(6,*)'                                            '
               write(6,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
               write(6,*)'Meldung von Routine OBER_PRINT'
               write(6,*)'In Patran sind nicht genuegend Farben '
               write(6,*)'vorhanden um die Zerlegung farblich   '
               write(6,*)'darzustellen. Zur Darstellung '
               write(6,*)'waeren ',nfarb_geb,' Farben notwendig.'
               write(6,*)'In Patran stehen aber nur ',nfarb_patran 
               write(6,*)'Farben zur Verfuegung.                   '
               write(6,*)'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
            endif
            session_print=.false.
         endif
      endif

c     write(lupar,*)'geb_num  farbe '
c     do 14 ifarb=1,nfarb_geb
c        do 13 iii=farb_adr(ifarb),farb_adr(ifarb+1)-1
c          igln_alt=perm_geb_inv(iii)
c          write(lupar,111) igln_alt,farb_geb(igln_alt)
c13      continue
c14   continue
c111  format(3(i3,1x),3x,30(i3,1x))
c     *****************************************************************

c     *****************************************************************
c     BESTIMMUNG DER AUSZUGEBENDEN KNOTEN:

      do 301 i=1,npoin
         zeig(i)=0
 301  continue

      kn_max=0
      ncofla=0
      do 300 i=1,nelfla
         do 310 k=1,nkd_obe
            ipoin=elfla_kno(i,k)
            if (zeig(ipoin).eq.0) then
               ncofla=ncofla+1
               zeig(ipoin)=1
               folg(ncofla)=ipoin
            endif
            kn_max=MAX(kn_max,coord_num(ipoin))
 310     continue
 300  continue
c     *****************************************************************

c     *****************************************************************
c     AUSDRUCK DER GEOMETRIE:

      open(lu,file=ober_name_geo,status='unknown')
      do 101 i=1,10
         write(lu,*)'Oberflaechengeometrie '
 101  continue
      write(lu,99) ncofla,nelfla,0,0,ncofla,nelfla,kn_max,nelfla

      do 105 i=1,ncofla
         kn_num=coord_num(folg(i))
         if (ncd.eq.2) then
            write(lu,98) kn_num,(coord(folg(i),j),j=1,ncd),0.0
         else if (ncd.eq.3) then
            write(lu,98) kn_num,(coord(folg(i),j),j=1,ncd)
         endif
 105  continue

c     Elemente der Oberflaechenelemente:  
      col_adr(1)=1
      geb_adr(1)=1
      do 110 ifarb=1,nfarb_geb
         nnn_col=0
         col_num(ifarb)=ifarb
         do 120 iii=farb_adr(ifarb),farb_adr(ifarb+1)-1

            nnn_geb=0
            igeb=perm_geb_inv(iii)
            geb_num(iii)=igeb 
            do 130 i=elfla_adr(igeb),elfla_adr(igeb+1)-1
               nnn_geb=nnn_geb+1
               nnn_col=nnn_col+1
               do 140 k=1,nkd_obe
                  help(k)=coord_num(elfla_kno(i,k))
 140           continue
               write(lu,97) (help(k),k=1,nkd_obe)

 130        continue                                        
            geb_adr(iii+1)=geb_adr(iii)+nnn_geb
 120     continue
         col_adr(ifarb+1)=col_adr(ifarb)+nnn_col
 110  continue

      close(lu)
      comment='File geschrieben:'
      call char_druck(comment,ober_name_geo,6)

 99   format(10(i6,1x))
 98   format(i8,3x,3(f15.6,1x))
 97   format(8(i8,1x))
c     *****************************************************************



c     *****************************************************************
c     AUSDRUCK AUF PROTOKOLL-FILE:

      do 600 i=2,80  
           zeil_1(i-1:i)='*'
           zeil_2(i-1:i)='-'
 600  continue

      comment='Elementnumerierung in File:'
      icom=lentb(comment)
      inam=lentb(ober_name_geo)
      comment(icom+2:icom+inam+1)=ober_name_geo(1:inam)

      write(lupro,*)
      write(lupro,901) zeil_1    
      write(lupro,555) comment(1:icom+inam+1)
      write(lupro,555) zeil_2(1:icom)
      write(lupro,*)
      do 701 i=1,ngebiet
         igeb=geb_num(i)
         num_1=geb_adr(i)
         num_2=geb_adr(i+1)-1
         anz=num_2-num_1
         if (anz.lt.0) then
            anz=0
         endif
         write(lupro,799)'Prozessor',igeb,':',num_1,'-',num_2,
     *                 'Anzahl:',anz
 701  continue

c     if (parallel) then
         comment='Farbnumerierung in File:'
         icom=lentb(comment)
         inam=lentb(ober_name_geo)
         comment(icom+2:icom+inam+1)=ober_name_geo(1:inam)

         write(lupro,*)
         write(lupro,555) comment(1:icom+inam+1)
         write(lupro,555) zeil_2(1:icom)
         write(lupro,*)
         do 702 i=1,nfarb_geb 
            icol=col_num(i)
            num_1=col_adr(i)
            num_2=col_adr(i+1)-1
            anz=num_2-num_1
            write(lupro,799)'Farbe    ',icol,':',num_1,'-',num_2,
     *                    'Anzahl:',anz           
 702     continue
c     endif

      write(lupro,901) zeil_1    
      write(lupro,*)
 799  format(1x,A9,1x,i3,A1,1x,i7,1x,A1,i7,3x,A7,i7)
 901  format(1x,A70)
 555  format(1x,A)
c     *****************************************************************


c     *****************************************************************
c     AUSDRUCK AUF SESSION-FILE:

      zeile1='pref_graphics_set( [TRUE, FALSE, FALSE, FALSE, FALSE , FAL
     *SE, FALSE, FALSE,   @'

      zeile2='FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FA
     *LSE, FALSE, FALSE,   @'

      zeile3='FALSE, FALSE], 2, 8, 13, 1, TRUE, TRUE, TRUE, TRUE, FALSE,
     * TRUE, TRUE, TRUE,   @'

      zeile4='2, 5, 5, 10, 1, 1, 1, 4 )'

      zeile5='ga_light_post( "directional_1" )'
      zeile6='ga_light_post( "directional_2" )'
      zeile7='ga_light_post( "directional_3" )'
      zeile8='ga_display_showedges_set( "general", 0 )'

      IF (session_print) THEN

          open(lu,file=ober_name_ses,status='unknown')

          write(lu,'(A)')zeile1           
          write(lu,'(A)')zeile2           
          write(lu,'(A)')zeile3           
          write(lu,'(A)')zeile4           
          write(lu,'(A)')'               '
          write(lu,'(A)')zeile5           
          write(lu,'(A)')zeile6           
          write(lu,'(A)')zeile7           
          write(lu,'(A)')zeile8           

c         Gruppenname:
          gruppe='kern_'

          do 800 ifarb=1,nfarb_geb

            write(otto,'(i4.4)') ifarb
            gruppe(6:9)=otto(1:4)

            write(lu,'(A)')'               '
            write(lu,910) gruppe
            write(lu,911) gruppe,col_adr(ifarb),col_adr(ifarb+1)-1
            write(lu,912) gruppe
            write(lu,913) gruppe,col_num(ifarb)
 800     continue

         close(lu)
         comment='File geschrieben:'
         call char_druck(comment,ober_name_ses,6)

      ENDIF

 910  format('ga_group_create( "',A10,'" )')
 911  format('ga_group_entity_add( "',A10,'", "element',
     *i8,':',i8,'" )')
 912  format('ga_group_current_set( "',A10,'" )')
 913  format('ga_group_color_set( "',A10,'",',i2,' )')
c     *****************************************************************


      return
      end
