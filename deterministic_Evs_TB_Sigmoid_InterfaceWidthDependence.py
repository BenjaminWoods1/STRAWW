"""

    Calculating the deterministic valley splitting (i.e. using the virtual crystal approximation)
    for the sp3d5s* tight binding model as a function of interface width, n_Ge_well_max, and shear strain
    for fixed Germanium oscillation wavelength and electric field

    Here the interface Ge profile uses the logistic function, where the
    Ge profile is given by f(z) = xGe_bar/(1 + exp[-4*(z-zt)/w]), with w being the interface width

"""

### Importing files and create file directory
if True:
    import sys
    import os
    import SiGe_Quantum_Well_sp3d5s2 as SQWTB
    import numpy as np
    import scipy.linalg as linalg
    import scipy.sparse as Spar
    import scipy.sparse.linalg as SparLinalg
    import matplotlib.pyplot as plt
    import parameters as par
    from time import time

    dir1 = "%s/deterministic_Evs_TB_Sigmoid_InterfaceWidthDependence_DATA" % (dir)
    dirS = "%s/deterministic_Evs_TB_Sigmoid_InterfaceWidthDependence_DATA/data" % (dir)
    dirF = "%s/deterministic_Evs_TB_Sigmoid_InterfaceWidthDependence_DATA/figs" % (dir)
    if not os.path.isdir(dir1):
        os.mkdir(dir1)
    if not os.path.isdir(dirS):
        os.mkdir(dirS)
    if not os.path.isdir(dirF):
        os.mkdir(dirF)

    def convert_mins_to_readable_time(minutes):
        if minutes < 2.:
            string = "%d minutes and %d seconds" % (minutes//1.,minutes % 1. * 60.)
        elif minutes < 60.:
            string = "%d minutes" % (minutes)
        else:
            string = "%d hours and %d minutes" % (minutes//60., minutes % 60.)
        return string

### Parameters
if True:
    x_Ge_substrate = 0.3 # Ge concentration in the substrate that defines the in-plane lattice constant
    system = SQWTB.SiGe_Quantum_Well_sp3d5s2(x_Ge_substrate)
    N_bar = 40
    N_well = 130
    N_wiggles = float(sys.argv[1])
    xGe_bar = 1.*x_Ge_substrate
    Fz = float(sys.argv[2]) # electric field in (mV/nm)

### Plot the wave function envelope of the ground valley for the input parameters in the absence of
### shear strain and wiggles to makes sure the barrier and well regions are large enough
if str(sys.argv[3]) == 'wavefunction_plot':

    ### Generate the Ge profile
    N_interface = 7.4; interface_width = (0.543/4)*N_interface
    print("Interface width = %.2f nm" % (interface_width))
    #Ge_arr, z_arr_cont, Ge_arr_cont = SQWTB.Soft_barrier_Wiggle_Well_Ge_profile2(N_bar,N_well,N_interface,10,xGe_bar,0.,z0 = 0,PLOT = True)
    Ge_arr, z_arr_cont, Ge_arr_cont = SQWTB.Sigmoid_barrier_WiggleWell_Ge_profile(N_bar,N_well,N_interface,N_wiggles,xGe_bar,0.0,z0 = 0,PLOT = True)
    z_arr = SQWTB.z_arr_gen(system.a_sub,Ge_arr); #z_arr = z_arr[:] - z_arr[N_bar+N_interface]

    ### Calculate the perpendicular lattice constant (along the growth direction) for pure Si layers
    a_perp_Si = SQWTB.calc_m_perp_lat_constant(system.a_sub,0.)

    ### Generate the Hamiltonian, and calculate the valley states and the valley splitting
    kx = 0; ky = 0; sigma = 1.; num = 4; epsilon_xy = 0.00
    Ham, z_arr = system.Ham_spinless_gen(kx,ky,Fz,Ge_arr,epsilon_xy)
    print("Ham shape", Ham.shape)
    eigs, U = system.solve_Ham(Ham,num,sigma,Return_vecs = True)
    print(eigs[1],eigs[0])
    print(eigs[2],eigs[3])
    E_VS = 1.e6*(eigs[1] - eigs[0])
    print("E_VS = %.5f micro eV" % (E_VS))
    E_VS2 = 1.e6*(eigs[3] - eigs[2])
    print("E_VS2 = %.5f micro eV" % (E_VS2))

    ### Plot the envelope of the valley state
    envelope = system.project_state_onto_valley(U[:,0])
    N_orb = 10
    A = np.square(np.absolute(envelope))
    A_proj = np.zeros(int(A.size/N_orb))
    for i in range(N_orb):
        A_proj[:] += A[i::N_orb]
    z_cont = .1*z_arr_cont*(a_perp_Si/5.430)
    z_wavefunction = (np.arange(A_proj.size) - N_bar)*(0.1*a_perp_Si/4)
    norm_factor = np.max(A_proj)/np.max(Ge_arr_cont)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1); ax1t = ax1.twinx()
    #ax.plot(z_cont,Ge_arr_cont*norm_factor*.8,c = 'r')
    ax1t.plot(z_cont,Ge_arr_cont,c = 'r')
    ax1.fill_between(z_wavefunction,0,A_proj,color = 'darkblue',alpha = 0.5)
    ax1.set_xlabel(r"$z$ (nm)")
    ax1.set_ylabel(r"$|\psi_{+z}|^{2}$",color = 'darkblue')
    ax1t.set_ylabel(r"$n_{Ge}$",color='r')
    ax1t.tick_params(axis='y', labelcolor='r')
    ax1.set_yticklabels([])
    ax1t.set_yticks([0,0.15,0.3])
    ax1.set_ylim(ymin = 0)
    ax1t.set_ylim(ymin = 0,ymax = 0.38)
    ax1.set_xlim(np.min(z_wavefunction),np.max(z_wavefunction))
    plt.subplots_adjust(left = 0.1, bottom=0.16, right=0.85, top=0.96)
    #fig.savefig("%s/EnvelopeWavefunction_Nint=%d_Fz=%.2f.png" % (dirF,N_interface,Fz),dpi = 200)
    plt.show()

### Calculate the valley splitting as a function of shear strain and interface width
### for fixed xGe_well_max
if str(sys.argv[3]) == 'calc_2':

    xGe_well_max = float(sys.argv[4])
    N_interface = np.arange(101)*.2 + 1.
    #print(N_interface); sys.exit()
    epsilon_xy = np.linspace(0.,0.01,51)

    ### Loop through values of N_interface and xGe_well_max
    num = 2
    eig_arr = np.zeros((N_interface.size,epsilon_xy.size,num))
    Evs_arr = np.zeros((N_interface.size,epsilon_xy.size))
    for i in range(N_interface.size):
        start_time = time()

        ### Create Ge concentration profile
        Ge_arr, z_arr_cont, Ge_arr_cont = SQWTB.Sigmoid_barrier_WiggleWell_Ge_profile(N_bar,N_well,N_interface[i],N_wiggles,xGe_bar,xGe_well_max,z0 = 0,PLOT = False)

        ### Loop through the shear strain values
        for j in range(epsilon_xy.size):
            print(epsilon_xy.size - j)

            ### Construct the Hamiltonian and solve for the low-energy states
            kx = 0; ky = 0; sigma = 1.
            Ham, z_arr = system.Ham_spinless_gen(kx,ky,Fz,Ge_arr,epsilon_xy[j])
            if (i== 0) and (j == 0):
                print("Ham shape", Ham.shape)
            eig_arr[i,j,:] = system.solve_Ham(Ham,num,sigma,Return_vecs = False)
            Evs_arr[i,j] = 1.e6*(eig_arr[i,j,1] - eig_arr[i,j,0]) # Valley splitting
            print("Evs = %.2f mu eV" % (Evs_arr[i,j]))

        ### save data
        if i > 5:
            np.save("%s/eigArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz,xGe_well_max),eig_arr)
            np.save("%s/EvsArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz,xGe_well_max),Evs_arr)
            np.save("%s/NinterfaceArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz,xGe_well_max),N_interface)
            np.save("%s/epxyArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz,xGe_well_max),epsilon_xy)

        ### Estimate time remaining
        end_time = time()
        Delta_t = end_time - start_time
        time_remaining = Delta_t * (N_interface.size - i -1) / 60.
        readable_time = convert_mins_to_readable_time(time_remaining)
        if i % 1 == 0:
            print ("Delta_t = %.1f seconds, %d values remain -> etr = %s" % (Delta_t,N_interface.size-i-1,readable_time))

if str(sys.argv[3]) == 'plot_2_fourPanels':

    Evs_max = 500.
    epsilon_xy_max = 0.0054
    w_max = 1.3
    cmap = 'tab20c'

    xGe_well_max1 = float(sys.argv[4])
    xGe_well_max2 = float(sys.argv[5])
    Fz1 = 1.*Fz
    Fz2 = float(sys.argv[6])

    xGe_arr = np.array([xGe_well_max1,xGe_well_max2,xGe_well_max1,xGe_well_max2])
    Fz_arr = np.array([Fz1,Fz1,Fz2,Fz2])

    width = 3.847; height = .8 * width
    plt.rcParams["figure.figsize"] = (width,height)
    fig = plt.figure()
    ys = .15
    x2o = 0.17; y2o = .12
    dx2 = .29; dy2 = .365
    y1o = y2o + .41*ys + dy2
    x4o = x2o + dx2 + 0.04
    xCo = x4o + dx2 + .03
    dxC = .02; dyC = y1o + dy2 - y2o
    ax1 = fig.add_axes([x2o,y1o,dx2,dy2])
    ax2 = fig.add_axes([x2o,y2o,dx2,dy2])
    ax3 = fig.add_axes([x4o,y1o,dx2,dy2])
    ax4 = fig.add_axes([x4o,y2o,dx2,dy2])
    axC = fig.add_axes([xCo,y2o,dxC,dyC])

    levels = np.linspace(0.,Evs_max,600)

    for b in range(4):

        eig_arr = np.load("%s/eigArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz_arr[b],xGe_arr[b]))
        Evs_arr = np.load("%s/EvsArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz_arr[b],xGe_arr[b]))
        N_interface = np.load("%s/NinterfaceArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz_arr[b],xGe_arr[b]))
        epsilon_xy = np.load("%s/epxyArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz_arr[b],xGe_arr[b]))

        X = []; Y = []; Z = []
        for i in range(N_interface.size):
            for j in range(epsilon_xy.size):
                X.append(N_interface[i]*(.1*5.430/4.))
                Y.append(epsilon_xy[j]*1.e2)
                if Evs_arr[i,j] < Evs_max:
                    Z.append(Evs_arr[i,j])
                else:
                    Z.append(Evs_max*0.999)

        if b == 0:
            ax1.set_facecolor('k')
            TC1 = ax1.tricontourf(X,Y,Z,600,cmap = cmap,levels = levels)
            #ax1.set_xlabel(r"$w$ (nm)",labelpad = -1)
            ax1.set_ylabel(r"$\epsilon_{xy}$ (%)")
            ax1.set_xlim(xmax = w_max)
            ax1.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
            ax1.set_yticks([0,0.25,0.5])
            ax1.set_xticklabels([])
        elif b == 1:
            ax2.set_facecolor('k')
            TC2 = ax2.tricontourf(X,Y,Z,600,cmap = cmap,levels = levels)
            ax2.set_xlabel(r"$w$ (nm)",labelpad = -1)
            ax2.set_ylabel(r"$\epsilon_{xy}$ (%)")
            ax2.set_xlim(xmax = w_max)
            ax2.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
            ax2.set_yticks([0,0.25,0.5])
        elif b == 2:
            ax3.set_facecolor('k')
            TC3 = ax3.tricontourf(X,Y,Z,600,cmap = cmap,levels = levels)
            ax3.set_xlim(xmax = w_max)
            ax3.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
            ax3.set_yticks([0,0.25,0.5])
            ax3.set_xticklabels([])
            ax3.set_yticklabels([])
        elif b == 3:
            ax4.set_facecolor('k')
            TC4 = ax4.tricontourf(X,Y,Z,600,cmap = cmap,levels = levels)
            ax4.set_xlim(xmax = w_max)
            ax4.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
            ax4.set_yticks([0,0.25,0.5])
            ax4.set_yticklabels([])
            ax4.set_xlabel(r"$w$ (nm)",labelpad = -1)

    cbar = fig.colorbar(TC1, cax=axC,ticks = [0,int(Evs_max/5),int(2*Evs_max/5),int(3*Evs_max/5),int(4*Evs_max/5),int(Evs_max)])
    cbar.ax.set_yticklabels(['0', '%d' % (int(1*Evs_max/5)), '%d' % (int(2*Evs_max/5)), '%d' % (int(3*Evs_max/5)), '%d' % (int(4*Evs_max/5)), r'$\geq %d$' % (int(5*Evs_max/5))])
    cbar.set_label('$E_{vs}$ ($\mu$eV)',labelpad = -4)

    ax1.text(.04, .92, "(a)", ha="left", va="top", transform=ax1.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    ax2.text(.04, .92, "(c)", ha="left", va="top", transform=ax2.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    ax3.text(.04, .92, "(b)", ha="left", va="top", transform=ax3.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    ax4.text(.04, .92, "(d)", ha="left", va="top", transform=ax4.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    ax1.text(.4, .08, r"$\bar{n}_{Ge} = %d$%%" % (xGe_well_max1*1.e2/2), ha="left", va="bottom", transform=ax1.transAxes,fontsize = 10., c= 'k')
    ax2.text(.28, .08, r"$\bar{n}_{Ge} = %.1f$%%" % (xGe_well_max2*1.e2/2), ha="left", va="bottom", transform=ax2.transAxes,fontsize = 10., c= 'k')
    ax3.text(.4, .08, r"$\bar{n}_{Ge} = %d$%%" % (xGe_well_max1*1.e2/2), ha="left", va="bottom", transform=ax3.transAxes,fontsize = 10., c= 'k')
    ax4.text(.28, .08, r"$\bar{n}_{Ge} = %.1f$%%" % (xGe_well_max2*1.e2/2), ha="left", va="bottom", transform=ax4.transAxes,fontsize = 10., c= 'k')
    ax1.text(.5, 1.03, r"$F_z = %d$ mV/nm" % (Fz1), ha="center", va="bottom", transform=ax1.transAxes,fontsize = 10., c= 'k')
    ax3.text(.5, 1.03, r"$F_z = %d$ mV/nm" % (Fz2), ha="center", va="bottom", transform=ax3.transAxes,fontsize = 10., c= 'k')

    fig.savefig("%s/EvsWidthDependence_Nwiggles=%.2f_Fz1=%.2f_Fz2=%.2f_xGeWellMax1=%.3f_xGeWellMax2=%.3f.png" % (dirF,N_wiggles,Fz1,Fz2,xGe_well_max1,xGe_well_max2),dpi = 300)

    plt.show()


if str(sys.argv[3]) == 'plot_2_fourPanels_NatureVersion':

    Evs_max = 500.
    epsilon_xy_max = 0.0054
    w_max = 1.3
    cmap = 'tab20c'

    xGe_well_max1 = float(sys.argv[4])
    xGe_well_max2 = float(sys.argv[5])
    Fz1 = 1.*Fz
    Fz2 = float(sys.argv[6])

    xGe_arr = np.array([xGe_well_max1,xGe_well_max2,xGe_well_max1,xGe_well_max2])
    Fz_arr = np.array([Fz1,Fz1,Fz2,Fz2])

    width = 3.847; height = .8 * width
    plt.rcParams["figure.figsize"] = (width,height)
    fig = plt.figure()
    ys = .15
    x2o = 0.17; y2o = .12
    dx2 = .29; dy2 = .365
    y1o = y2o + .41*ys + dy2
    x4o = x2o + dx2 + 0.04
    xCo = x4o + dx2 + .03
    dxC = .02; dyC = y1o + dy2 - y2o
    ax1 = fig.add_axes([x2o,y1o,dx2,dy2])
    ax2 = fig.add_axes([x2o,y2o,dx2,dy2])
    ax3 = fig.add_axes([x4o,y1o,dx2,dy2])
    ax4 = fig.add_axes([x4o,y2o,dx2,dy2])
    axC = fig.add_axes([xCo,y2o,dxC,dyC])

    levels = np.linspace(0.,Evs_max,600)

    for b in range(4):

        eig_arr = np.load("%s/eigArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz_arr[b],xGe_arr[b]))
        Evs_arr = np.load("%s/EvsArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz_arr[b],xGe_arr[b]))
        N_interface = np.load("%s/NinterfaceArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz_arr[b],xGe_arr[b]))
        epsilon_xy = np.load("%s/epxyArr_Nwiggles=%.2f_Fz=%.2f_xGeWellMax=%.3f.npy" % (dirS,N_wiggles,Fz_arr[b],xGe_arr[b]))

        X = []; Y = []; Z = []
        for i in range(N_interface.size):
            for j in range(epsilon_xy.size):
                X.append(N_interface[i]*(.1*5.430/4.))
                Y.append(epsilon_xy[j]*1.e2)
                if Evs_arr[i,j] < Evs_max:
                    Z.append(Evs_arr[i,j])
                else:
                    Z.append(Evs_max*0.999)

        if b == 0:
            ax1.set_facecolor('k')
            TC1 = ax1.tricontourf(X,Y,Z,600,cmap = cmap,levels = levels)
            #ax1.set_xlabel(r"$w$ (nm)",labelpad = -1)
            #ax1.set_ylabel(r"$\epsilon_{xy}$ (%)")
            ax1.set_xlim(xmax = w_max)
            ax1.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
            ax1.set_yticks([0,0.25,0.5])
            ax1.set_yticklabels([])
            ax1.set_xticklabels([])
        elif b == 1:
            ax2.set_facecolor('k')
            TC2 = ax2.tricontourf(X,Y,Z,600,cmap = cmap,levels = levels)
            #ax2.set_xlabel(r"$w$ (nm)",labelpad = -1)
            #ax2.set_ylabel(r"$\epsilon_{xy}$ (%)")
            ax2.set_xlim(xmax = w_max)
            ax2.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
            ax2.set_yticks([0,0.25,0.5])
            ax2.set_yticklabels([])
            ax2.set_xticklabels([])
        elif b == 2:
            ax3.set_facecolor('k')
            TC3 = ax3.tricontourf(X,Y,Z,600,cmap = cmap,levels = levels)
            ax3.set_xlim(xmax = w_max)
            ax3.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
            ax3.set_yticks([0,0.25,0.5])
            ax3.set_xticklabels([])
            ax3.set_yticklabels([])
        elif b == 3:
            ax4.set_facecolor('k')
            TC4 = ax4.tricontourf(X,Y,Z,600,cmap = cmap,levels = levels)
            ax4.set_xlim(xmax = w_max)
            ax4.set_ylim(ymin = 0, ymax = epsilon_xy_max*1.e2)
            ax4.set_yticks([0,0.25,0.5])
            ax4.set_yticklabels([])
            ax4.set_xticklabels([])
            #ax4.set_xlabel(r"$w$ (nm)",labelpad = -1)

    cbar = fig.colorbar(TC1, cax=axC,ticks = [0,int(Evs_max/5),int(2*Evs_max/5),int(3*Evs_max/5),int(4*Evs_max/5),int(Evs_max)])
    cbar.ax.set_yticklabels(['0', '%d' % (int(1*Evs_max/5)), '%d' % (int(2*Evs_max/5)), '%d' % (int(3*Evs_max/5)), '%d' % (int(4*Evs_max/5)), r'$\geq %d$' % (int(5*Evs_max/5))])
    cbar.ax.set_yticklabels(['','','','','',''])
    cbar.set_label('$E_{vs}$ ($\mu$eV)',labelpad = -4)
    cbar.set_label('')

    #ax1.text(.04, .92, "(a)", ha="left", va="top", transform=ax1.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    #ax2.text(.04, .92, "(c)", ha="left", va="top", transform=ax2.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    #ax3.text(.04, .92, "(b)", ha="left", va="top", transform=ax3.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    #ax4.text(.04, .92, "(d)", ha="left", va="top", transform=ax4.transAxes,fontsize = 10.5,weight = 'bold', c= 'k')
    #ax1.text(.4, .08, r"$\bar{n}_{Ge} = %d$%%" % (xGe_well_max1*1.e2/2), ha="left", va="bottom", transform=ax1.transAxes,fontsize = 10., c= 'k')
    #ax2.text(.28, .08, r"$\bar{n}_{Ge} = %.1f$%%" % (xGe_well_max2*1.e2/2), ha="left", va="bottom", transform=ax2.transAxes,fontsize = 10., c= 'k')
    #ax3.text(.4, .08, r"$\bar{n}_{Ge} = %d$%%" % (xGe_well_max1*1.e2/2), ha="left", va="bottom", transform=ax3.transAxes,fontsize = 10., c= 'k')
    #ax4.text(.28, .08, r"$\bar{n}_{Ge} = %.1f$%%" % (xGe_well_max2*1.e2/2), ha="left", va="bottom", transform=ax4.transAxes,fontsize = 10., c= 'k')
    #ax1.text(.5, 1.03, r"$F_z = %d$ mV/nm" % (Fz1), ha="center", va="bottom", transform=ax1.transAxes,fontsize = 10., c= 'k')
    #ax3.text(.5, 1.03, r"$F_z = %d$ mV/nm" % (Fz2), ha="center", va="bottom", transform=ax3.transAxes,fontsize = 10., c= 'k')

    fig.savefig("%s/EvsWidthDependence_NatureVersion_Nwiggles=%.2f_Fz1=%.2f_Fz2=%.2f_xGeWellMax1=%.3f_xGeWellMax2=%.3f.png" % (dirF,N_wiggles,Fz1,Fz2,xGe_well_max1,xGe_well_max2),dpi = 300)

    plt.show()
