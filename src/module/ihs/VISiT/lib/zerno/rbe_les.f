C**************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE RBE_LES(elra_kno,elra_ele,elra_num,elra_wer,elra_adr,
     *                   knra_kno,knra_mat,knra_wer,knra_adr,
     *                   coord_zeig,lnods_zeig,
     *                   lnods,coord_num,
     *                   zeiger,rbe_name)
C
      implicit none 

      include 'common.zer'
      include 'mpif.h'

      integer   elra_kno,elra_ele,elra_num,elra_adr,
     *          knra_kno,knra_mat,knra_adr,
     *          coord_zeig,lnods_zeig,
     *          lnods,coord_num,
     *          zeiger

      real    elra_wer,knra_wer,
     *        wert,hh(3),toler,ddd

      integer kn,help1(8),i,j,ityp,kn_neu,el_num,lu,
     *        nnn,mmm,luerr,adr,ityp_knra,ityp_elra,
     *        ndop,ndop_all,fla_num

      character*35 typ_name

      character*80 rbe_name,reihe,comment,file_name

      logical   doppel,format_read

      parameter (lu=50,toler=1.e-06)

      dimension elra_kno(nelra_max,nrbknie),elra_ele(nelra_max),
     *          elra_num(nelra_max),elra_wer(nelra_max),
     *          elra_adr(ntyp_elra+1)

      dimension knra_kno(nknra_max),knra_wer(nknra_max,nwer_knra),
     *          knra_mat(nknra_max,nwer_knra),
     *          knra_adr(ntyp_knra+1)

      dimension coord_zeig(knmax_num),lnods_zeig(elmax_num),
     *          lnods(nelem_max,nkd),coord_num(npoin_max),
     *          zeiger(npoin)
c     ****************************************************************

      file_name=rbe_name
      open(lu,file=file_name,status='old',err=777)
      format_read=.true.
      CALL HEAD_READ(lu,file_name,format_read,reihe)


c     ****************************************************************
c     INITIALISIERUNGEN:

      do 10 i=1,nknra_max
         knra_kno(i)=0
         do 11 j=1,nwer_knra
           knra_mat(i,j)=0
           knra_wer(i,j)=0.0
 11      continue
 10   continue

      do 12 ityp_knra=1,ntyp_knra
        knra_adr(ityp_knra)=0
 12   continue

      do 20 i=1,nelra_max
         elra_ele(i)=0
         elra_num(i)=0
         elra_wer(i)=0.0
         do 21 j=1,nrbknie    
           elra_kno(i,j)=0
 21      continue
 20   continue

      do 22 ityp_elra=1,ntyp_elra
        elra_adr(ityp_elra)=0
 22   continue

      ityp_knra=0
      ityp_elra=0
      nknra=0
      nelra=0
      elra_adr(1)=1
      knra_adr(1)=1
c     ****************************************************************


c     ****************************************************************
c     DISPL-RANDBEDINGUNGEN:

      do 101 i=1,npoin
         zeiger(i)=0
 101  continue

      ityp_knra=ityp_knra+1
      knra_name(ityp_knra)='Displ-Randbedingungen'
      typ_name=knra_name(ityp_knra)

      nnn=0
      do 100 i=1,nrbpoi
       read(lu,*) kn,ityp,wert

       CALL RKN_CHECK(kn,coord_zeig,typ_name)
       kn_neu=coord_zeig(kn)
       if (zeiger(kn_neu).eq.0) then
c         Neuer Randbedingungsknoten:
          nknra=nknra+1
          nnn=nnn+1
          knra_kno(nknra)=kn_neu
          knra_mat(nknra,ityp)=knra_mat(nknra,ityp)+1
          knra_wer(nknra,ityp)=wert     
          zeiger(kn_neu)=nknra
       else 
c         Randbedingungsknoten wurde schon geschrieben
          adr=zeiger(kn_neu)
          if (knra_mat(adr,ityp).eq.0) then
c            In diese Spalte wurde noch nicht geschrieben:
             knra_mat(adr,ityp)=knra_mat(nknra,ityp)+1
             knra_wer(adr,ityp)=wert     
          else
c            In diese Spalte wurde schon geschrieben:
             ddd=abs(knra_wer(adr,ityp)-wert)

c            write(lupar,665) kn,kn_neu,adr,
c    *                    ddd,wert,knra_wer(adr,ityp),toler

             if (ddd.gt.toler) then 
c               Doppelte Randbedingung mit verschiedenen Werten:
                knra_mat(adr,ityp)=knra_mat(adr,ityp)+1
                knra_wer(adr,ityp)=wert     
             endif

          endif
       endif
 100  continue
     
 665  format(3(i6,1x),3x,4(e10.4,1x))

      knra_adr(ityp_knra+1)=knra_adr(ityp_knra)+nnn
c     ****************************************************************


c     ****************************************************************
c     EINLESEN DER WAND-RANDBEDINGUNGEN:          

      ityp_elra=ityp_elra+1
      elra_name(ityp_elra)='Wand-Randbedingungen'
      typ_name=elra_name(ityp_elra)

      nnn=0
      do 200 i=1,nwand

         read(lu,*) (help1(j),j=1,nrbknie),el_num,fla_num,wert
         CALL REL_CHECK(el_num,help1,coord_zeig,lnods_zeig,
     *                  coord_num,lnods,typ_name)

         nelra=nelra+1
         nnn=nnn+1
         elra_ele(nelra)=lnods_zeig(el_num)
         elra_num(nelra)=INT(fla_num)
         elra_wer(nelra)=wert
         do 210 j=1,nrbknie
 	    elra_kno(nelra,j)=coord_zeig(help1(j))
 210     continue

 200  continue

      elra_adr(ityp_elra+1)=elra_adr(ityp_elra)+nnn
c     ****************************************************************


c     ****************************************************************
c     EINLESEN DER DRUCK-RANDBEDINGUNGEN:          

      ityp_elra=ityp_elra+1
      elra_name(ityp_elra)='Druck-Randbedingungen'
      typ_name=elra_name(ityp_elra)

      nnn=0
      do 300 i=1,npres

        read(lu,*) (help1(j),j=1,nrbknie),el_num,fla_num,wert

        CALL REL_CHECK(el_num,help1,coord_zeig,lnods_zeig,
     *                 coord_num,lnods,typ_name)

        nelra=nelra+1
        nnn=nnn+1
        elra_ele(nelra)=lnods_zeig(el_num)
        elra_num(nelra)=INT(fla_num)                
        elra_wer(nelra)=wert                
        do 310 j=1,nrbknie
           elra_kno(nelra,j)=coord_zeig(help1(j))
 310    continue

 300  continue

      elra_adr(ityp_elra+1)=elra_adr(ityp_elra)+nnn
c     ****************************************************************


c     ****************************************************************
c     EINLESEN DER SYMMETRIE-RANDBEDINGUNGEN:          

      ityp_elra=ityp_elra+1
      elra_name(ityp_elra)='Symmetrie-Randbedingungen'
      typ_name=elra_name(ityp_elra)

      nnn=0
      do 400 i=1,nsyme
        read(lu,*) (help1(j),j=1,nrbknie),el_num

        CALL REL_CHECK(el_num,help1,coord_zeig,lnods_zeig,
     *                 coord_num,lnods,typ_name)

        nelra=nelra+1
        nnn=nnn+1
        elra_ele(nelra)=lnods_zeig(el_num)
        do 410 j=1,nrbknie
           elra_kno(nelra,j)=coord_zeig(help1(j))
 410    continue

 400  continue

      elra_adr(ityp_elra+1)=elra_adr(ityp_elra)+nnn
c     ****************************************************************


c     ****************************************************************
c     EINLESEN DER PERIODISCHEN-RANDBEDINGUNGEN:          

      ityp_elra=ityp_elra+1
      elra_name(ityp_elra)='periodischen-Randbedingungen'
      typ_name=elra_name(ityp_elra)

      nnn=0
      do 500 i=1,nzykl
        read(lu,*) (help1(j),j=1,nrbknie),wert,el_num

        CALL REL_CHECK(el_num,help1,coord_zeig,lnods_zeig,
     *                 coord_num,lnods,typ_name)

        nelra=nelra+1
        nnn=nnn+1
        elra_ele(nelra)=lnods_zeig(el_num)
        elra_wer(nelra)=wert                
        do 510 j=1,nrbknie
           elra_kno(nelra,j)=coord_zeig(help1(j))
 510    continue

 500  continue

      elra_adr(ityp_elra+1)=elra_adr(ityp_elra)+nnn
c     ****************************************************************

c     ****************************************************************
c old    EINLESEN DER CONV-RANDBEDINGUNGEN:          
c     EINLESEN DER BILA-RANDBEDINGUNGEN:     

      ityp_elra=ityp_elra+1
      elra_name(ityp_elra)='Conv-Randbedingungen'
      typ_name=elra_name(ityp_elra)

      nnn=0
      do 600 i=1,nconv

        read(lu,*) (help1(j),j=1,nrbknie),el_num,fla_num,wert

        CALL REL_CHECK(el_num,help1,coord_zeig,lnods_zeig,
     *                 coord_num,lnods,typ_name)

        nelra=nelra+1
        nnn=nnn+1
        elra_ele(nelra)=lnods_zeig(el_num)
        elra_num(nelra)=INT(fla_num)
        elra_wer(nelra)=wert
        do 610 j=1,nrbknie
           elra_kno(nelra,j)=coord_zeig(help1(j))
 610    continue

 600  continue

      elra_adr(ityp_elra+1)=elra_adr(ityp_elra)+nnn
c     ****************************************************************


c     ****************************************************************
c     EINLESEN DER TEMP-RANDBEDINGUNGEN:

      do 701 i=1,npoin
         zeiger(i)=0
 701  continue

      ityp_knra=ityp_knra+1
      knra_name(ityp_knra)='Temp-Randbedingungen'
      typ_name=knra_name(ityp_knra)

      nnn=0
      do 700 i=1,ntemp 
	read(lu,*) kn,wert

       CALL RKN_CHECK(kn,coord_zeig,typ_name)
       kn_neu=coord_zeig(kn)
       if (zeiger(kn_neu).eq.0) then
c         Neuer Randbedingungsknoten:
          nknra=nknra+1
          nnn=nnn+1
          knra_kno(nknra)=kn_neu
          knra_mat(nknra,1)=knra_mat(nknra,1)+1
          knra_wer(nknra,1)=wert     
          zeiger(kn_neu)=nknra
       else 
c         Randbedingungsknoten wurde schon geschrieben
          adr=zeiger(kn_neu)
          knra_mat(adr,1)=knra_mat(adr,1)+1
          knra_wer(adr,1)=wert     
       endif
 700  continue
      
      knra_adr(ityp_knra+1)=knra_adr(ityp_knra)+nnn
c     ****************************************************************

     
      close(lu)


c     ****************************************************************
c     DIMENSIONS-KONTROLLEN:

      do 50 ityp=1,ntyp_elra
         nnn=elra_adr(ityp+1)-elra_adr(ityp)
         if (ityp.eq.1) then
           mmm=nwand
         else if (ityp.eq.2) then
           mmm=npres
         else if (ityp.eq.3) then
           mmm=nsyme
         else if (ityp.eq.4) then
           mmm=nzykl
         else if (ityp.eq.5) then
           mmm=nconv
         endif

         if (nnn.ne.mmm) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine RBE_LES !'
           write(luerr,*)'Geschriebene Dimension stimmt mit der  '
           write(luerr,*)'tatsaechlichen Dimension nicht ueberein.'
           write(luerr,*)'Geschriebene  Dimension:',nnn            
           write(luerr,*)'Tatsaechliche Dimension:',mmm            
           write(luerr,'(A)')'Fehler bei ',elra_name(ityp)             
           call erro_ende(myid,parallel,luerr)
         endif
 50   continue

      nnn=elra_adr(ntyp_elra+1)-1 
      if (nelra.ne.nnn) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine RBE_LES !'
           write(luerr,*)'Geschriebene Gesamtanzahl an        '
           write(luerr,*)'Elementrandbedingungen stimmt mit '
           write(luerr,*)'der tatsaechlichen Dimension nicht ueberein.'
           write(luerr,*)'Geschriebene  Dimension:',nelra
           write(luerr,*)'Tatsaechliche Dimension:',nnn            
           call erro_ende(myid,parallel,luerr)
      endif

      nnn=knra_adr(ntyp_knra+1)-1 
      if (nknra.ne.nnn) then
           call erro_init(myid,parallel,luerr)
           write(luerr,*)'Fehler in Routine RBE_LES !'
           write(luerr,*)'Geschriebene Gesamtanzahl an        '
           write(luerr,*)'Knotenrandbedingungen stimmt mit '
           write(luerr,*)'der tatsaechlichen Dimension nicht ueberein.'
           write(luerr,*)'Geschriebene  Dimension:',nknra
           write(luerr,*)'Tatsaechliche Dimension:',nnn            
           call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************


c     ****************************************************************
c     KONTROLLE OB DOPPELTE KNOTEN-RANDBEDINGUNGEN EXISTIEREN:  
      do 800 ityp=1,ntyp_knra
         ndop=0
         do 810 i=knra_adr(ityp),knra_adr(ityp+1)-1
           doppel=.false.
           do 820 j=1,nwer_knra
              if (knra_mat(i,j).gt.1) then
                 doppel=.true.
              endif
 820       continue
           if (doppel) then
c            Doppelte Randbedingung:
             ndop=ndop+1
             zeiger(ndop)=coord_num(knra_kno(i))
           endif
 810     continue
         if (parallel) then
            CALL MPI_ALLREDUCE(ndop,ndop_all,1,MPI_INTEGER,
     *                         MPI_SUM,MPI_COMM_WORLD,ierr)  
         else
            ndop_all=ndop
         endif
         if (myid.eq.0) then
            if (ndop_all.ne.0) then
              write(6,*)'                                    '
              write(6,*)'************ WARNUNG ***************'
              write(6,*)'Es gibt doppelte Randbedingungen !!'
              write(6,*)'in den ',knra_name(ityp)
              write(6,*)'Anzahl             :',ndop_all 
              write(6,*)'Verwendete Toleranz:',toler    
              write(6,*)'************ WARNUNG ***************'
              write(6,*)'                                    '
              write(lupro,*)'                                    '
              write(lupro,*)'************ WARNUNG ***************'
              write(lupro,*)'Es gibt doppelte Randbedingungen !!'
              write(lupro,*)'in den ',knra_name(ityp)
              write(lupro,*)'Anzahl             :',ndop_all 
              write(lupro,*)'Verwendete Toleranz:',toler    
              write(lupro,*)'************ WARNUNG ***************'
              write(lupro,*)'                                    '
            endif
         endif
 800  continue
c     ****************************************************************


c     ****************************************************************
c     AUSDRUCK:
c     do 900 ityp=1,ntyp_knra
c        write(lupar,'(A)') knra_name(ityp)
c        do 910 i=knra_adr(ityp),knra_adr(ityp+1)-1
c            write(lupar,988)knra_kno(i),
c    *                  (knra_mat(i,j),j=1,nwer_knra)
c910     continue                                      
c900  continue                   
c988  format(i6,2x,10(i1,1x))
c     ****************************************************************



c     **********************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:

      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File (rbe_les):'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     **********************************************************

      return
      end 


