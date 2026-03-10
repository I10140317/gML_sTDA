! This file is part of stda.
!
! Copyright (C) 2013-2019 Stefan Grimme
!
! stda is free software: you can redistribute it and/or modify it under
! the terms of the GNU Lesser General Public License as published by
! the Free Software Foundation, either version 3 of the License, or
! (at your option) any later version.
!
! stda is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU Lesser General Public License for more details.
!
! You should have received a copy of the GNU Lesser General Public License
! along with stda.  If not, see <https://www.gnu.org/licenses/>.
!
      program acis_prog
      use stdacommon ! mostly input and primitive data
      use kshiftcommon ! kshiftvariables
      use commonlogicals
      use commonresp
      implicit real*8 (a-h,o-z)

      real*8, allocatable ::cc(:)

      integer, allocatable :: ccspin(:)
      real*8, allocatable ::xyz(:,:)
      real*8  xx(10),alpha,beta,ptlim
      character*79 dummy
      character*79 fname
      character*8 method
      integer imethod,inpchk,mform,nvec
      logical molden,da,chkinp,xtbinp
      integer, dimension(8) :: datetimevals

      call date_and_time(VALUES=datetimevals)
      print '(I0,"-",I0,"-",I0,1X,I0,":",I0,":",I0,".",I3)',
     .      datetimevals(1:3), datetimevals(5:8)

      write(*,'(//
     .          17x,''*********************************************'')')
      write(*,'(17x,''*                                           *'')')
      write(*,'(17x,''*               s  T  D  A                  *'')')
      write(*,'(17x,''*                                           *'')')
      write(*,'(17x,''*                S. Grimme                  *'')')
      write(*,'(17x,''* Mulliken Center for Theoretical Chemistry *'')')
      write(*,'(17x,''*             Universitaet Bonn             *'')')
      write(*,'(17x,''*              Version 1.6.3                *'')')
      write(*,'(17x,''*       Fri Aug 26 14:28:49 CEST 2022       *'')')
      write(*,'(17x,''*********************************************'')')
      write(*,*)
      write(*,'('' Please cite as:'')')
      write(*,'('' S. Grimme, J. Chem. Phys. 138 (2013) 244104'')')
      write(*,'('' M. de Wergifosse, S. Grimme, J. Phys. Chem A'')')
      write(*,'('' 125 (2021) 18 3841-3851'')')
      write(*,*)
      write(*,'('' With contributions from:'')')
      write(*,'('' C. Bannwarth, P. Shushkov, M. de Wergifosse'')')
      write(*,*)
      write(*,'(a,a)')'===============================================',
     .                 '======================='
      write(*,*)
c defaults

      ptlim=1.7976931348623157d308 ! energy range that will be scanned by PT (we use just a large number)
      thre=7.0                     ! energy range for primary CSF space
      alpha=-100.0d0               ! alpha & beta are large negative numbers and can be changed by user input
      beta=-100.0d0                ! otherwise global hybrid defaults will be used

c the following value yields converged UV spectra for several members of
c of the DYE12 set
      thrp=1.d-4

      mform=1 ! mform is the "style" specifier for Molden input, by default try TM input: ORCA/XTB = 0, TM=1,Molpro=2, Terachem/Gaussian=3

      rpachk=.false.  ! sTD-DFT wanted?
      triplet=.false. ! triplet excitations wanted?
      eigvec=.false. ! eigenvector printout wanted?
      nvec=0 ! if so, how many vecs?
      printexciton=.false. ! print information for exciton coupling program
      velcorr=.true. ! by default: use  Rv(corr) in TDA
      aniso=.false. ! print  anisotropic f/R

      chkinp=.false. ! perform input check?
      fname='molden.input'
      xtbinp=.false. !use xtbinput?
      screen=.false. ! prescreen configurations (by Schwarz-type screening)

      ! Kia shifting defaults
      dokshift=.false.
      shftmax=0.50d0 ! maximum Kia shift in eV
      shftwidth=0.10d0 ! damping threshold in eV
      shftsteep=4.0d0 ! steepness

c read the tm2xx file, otherwise (-f option) the tm2molden file
      molden=.true.
      ax=-1
      imethod=0
      inpchk=0
      resp=.false.
      TPA=.false.
      aresp=.false.
      ESA=.false.
      smp2=.false.
      bfw=.false.
      spinflip=.false.
      sf_s2=.false.
      rw=.false.
      rw_dual=.false.
      pt_off=.false.
      optrot=.false.

! check for input file
      inquire(file='.STDA',exist=da)
      if(da)then
        call readinp(ax,thre,alpha,beta)
      endif

      do i=1,command_argument_count()
        call getarg(i,dummy)
        if(index(dummy,'-fo').ne.0)then
           call getarg(i+1,fname)
           molden=.false.
           inpchk=inpchk+1
        endif
        if(index(dummy,'-f').ne.0.and.index(dummy,'-fo').eq.0)then
           call getarg(i+1,fname)
           molden=.true.
           inpchk=inpchk+1
        endif
      enddo

      if(chkinp) mform=0 ! if input check is done, start with 0

ccccccccccccccccccccccccccccccc
c check the input
ccccccccccccccccccccccccccccccc
      if(inpchk.gt.1) stop 'please specify only one input file!'
      if(inpchk.lt.1) stop 'no input file specified!'
      if(molden) then
       write(*,*) 'reading a molden input...'
      end if

ccccccccccccccccccccccccccccccc
c first call to get dimensions
ccccccccccccccccccccccccccccccc
      call date_and_time(VALUES=datetimevals)
      print '(I0,"-",I0,"-",I0,1X,I0,":",I0,":",I0,".",I3)',
     .      datetimevals(1:3), datetimevals(5:8)

      if(molden)then
        inpchk=0 ! use this integer now to determine UKS/RKS
        call readmold0(ncent,nmo,nbf,nprims,fname,inpchk)
        if(imethod.eq.0)imethod=inpchk ! if UKS/RKS has not been specified
c compare input and inpchk

      else if(xtbinp) then ! read parameters from xTB input
       call readxtb0(imethod,ncent,nmo,nbf,nprims)

      else
        call readbas0a(0,ncent,nmo,nbf,nprims,fname)
      endif

      if(nprims.eq.0.or.ncent.eq.0.or.nmo.eq.0)
     .stop 'read error'

ccccccccccccccccccccccccccccccc
c allocate mo vectors
ccccccccccccccccccccccccccccccc
       icdim = nmo*nbf
      if(imethod.eq.2.and..not.molden) then
       icdim=2*nmo*nbf
       nmo = 2*nmo
      endif

      allocate(cc(icdim),stat=ierr)
      if(ierr.ne.0) stop 'allocation failed in main for cc'

      allocate(ccspin(nmo),stat=ierr)
      if(ierr.ne.0) stop 'allocation failed in main for ccspin'

*****************************
* allocate common variables *
*****************************
      allocate(co(ncent,4),exip(nprims),cxip(nprims),occ(nmo),eps(nmo))
      allocate(ipat(nprims),ipty(nprims),ipao(nprims),iaoat(nbf))
      allocate(atnam(ncent),eta(nprims,25))

ccccccccccccccccccccccccccccccccc
c read vectors and basis and ..
ccccccccccccccccccccccccccccccccc
      call date_and_time(VALUES=datetimevals)
      print '(I0,"-",I0,"-",I0,1X,I0,":",I0,":",I0,".",I3)',
     .      datetimevals(1:3), datetimevals(5:8)

      if(molden)then
        call readmold(mform,imethod,ncent,nmo,nbf,nprims,cc,
     .  ccspin,icdim,fname)
      else if(xtbinp) then
        call readxtb(imethod,ncent,nmo,nbf,nprims,cc)
        if(imethod.eq.2) then
          do i=1,nmo/2
            ccspin(i)=1
          enddo
          do i=nmo/2+1,nmo
            ccspin(i)=2
          enddo
        endif
      else
       if(imethod.eq.1) call readbasa(1,imethod,ncent,nmo,nbf,nprims,cc,
     .icdim,fname,iaobas)
       if(imethod.eq.2) call readbasb(1,imethod,ncent,nmo,nbf,nprims,cc,
     .ccspin,icdim,fname,iaobas)
      endif
      if(imethod.eq.1) deallocate( ccspin )

ccccccccccccccccccccccccccccccccc
c precalculate primitive data
ccccccccccccccccccccccccccccccccc
      call intslvm(ncent,nbf,nprims)

      call date_and_time(VALUES=datetimevals)
      print '(I0,"-",I0,"-",I0,1X,I0,":",I0,":",I0,".",I3)',
     .      datetimevals(1:3), datetimevals(5:8)
      end

