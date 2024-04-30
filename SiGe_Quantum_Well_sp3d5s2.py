"""
    SiGe quantum well system with shear strain ε_{xy} using the
    sp3d5s* tight-binding model of SiGe alloys from
    PHYSICAL REVIEW B 79, 245201 (2009)

    Spin-orbit coupling is not currently included since project is about increasing valley splitting
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import parameters as par
import scipy.sparse as Spar
import scipy.sparse.linalg as SparLinalg
import scipy.special
from random import random, uniform, randint, seed, choice
import scipy.linalg as linalg
np.set_printoptions(linewidth = 500)

### Functions
if True:

    ### Lattice constant functions
    if True:
        def calc_relaxed_lat_constant(x):
            ### Calcualtes the relaxed lattice constant (in Angstroms) of a bulk Si_{1-x}Ge_{x} alloy
            a_Si = 5.431
            a_Ge = 5.658
            #a_relaxed = (1 - x) * a_Si + x * a_Ge
            a_relaxed = a_Si + 0.2*x + 0.027*(x**2)
            return a_relaxed

        def calc_m_perp_lat_constant(a_par,x_m):
            ### Calculate the perpendicular lattice constant
            ### if the entire quantum well had x_m as its Ge fraction
            ###     * The c12 and c11 are determined from a linear interpolation
            ###       between pure Si and Ge with those values given in
            ###       PHYSICAL REVIEW B 79, 245201 (2009) on pg. 10
            c12_c11 = (63.9 - 15.6 * x_m)/ (165.8 - 34.0*x_m)
            a_m_relaxed = calc_relaxed_lat_constant(x_m)
            a_perp_m = a_m_relaxed * (1. - 2*c12_c11 * (a_par/a_m_relaxed - 1.))
            return a_perp_m

        def z_arr_gen(a_par,Ge_arr_Full):
            ### Generates an array of z-coordinates (in Angstroms) of the atomic layers
            ### for a system with Ge content given by Ge_arr_Full
            z_arr = np.zeros(Ge_arr_Full.size)
            for j in range(Ge_arr_Full.size):
                if j == 0:
                    z_arr[j] = 0.
                else:
                    a_perp_mp1m = .5*( calc_m_perp_lat_constant(a_par,Ge_arr_Full[j-1]) \
                                       + calc_m_perp_lat_constant(a_par,Ge_arr_Full[j]) )
                    z_arr[j] = z_arr[j-1] + a_perp_mp1m/4.
            return z_arr

    ### inter-atomic hopping functions
    if True:

        def calc_interatomic_mtx_element_Si_Si(α,β,l,m,n,d):
            ### Calculates the interatomic matrix element between the
            ### orbital β on the reference Si atom and the α orbital on
            ### the target Si atom
            ###     * l,m,n are the directional cosines of the vector
            ###       going from the reference to the target atom
            ###     * d is the distance between atoms

            d_Si = 5.431 * np.sqrt(3.)/4.
            d_ratio = (d_Si/d) # ratio of unstrained and strained inter-atomic distance

            ### Unstrained band parameters
            Es = -2.55247 # + E_offset
            Ep = 4.48593 # + E_offset
            Ed = 14.01053 # + E_offset
            Es2 = 23.44607 # + E_offset

            Vssσ = -1.86600
            Vspσ = 2.91067
            Vsdσ = -2.23992
            Vss2σ = -1.39107

            Vppσ = 4.08481
            Vppπ = -1.49207
            Vpdσ = -1.66657
            Vpdπ = 2.39936

            Vddσ = -1.82945
            Vddπ = 3.08177
            Vddδ = -1.56676

            Vs2pσ = 3.06822
            Vs2dσ = -0.77711
            Vs2s2σ = -4.51331

            Delta_SO = 3*0.01851

            ### Deformation exponentials
            nssσ = 3.56701
            nss2σ = 1.51967
            nspσ = 2.03530
            nsdσ = 2.14811

            ns2s2σ = 0.64401
            ns2pσ = 1.46652
            ns2dσ = 1.79667

            nppσ = 2.01907
            nppπ = 2.87276
            npdσ = 1.00446
            npdπ = 1.78029

            nddσ = 1.73865
            nddπ = 1.80442
            nddδ = 2.54691
            #bd = 0.443

            ### Calculate the renormalized matrix elements
            Vssσ = Vssσ * (d_ratio**nssσ)
            Vs2sσ = Vss2σ * (d_ratio**nss2σ)
            Vss2σ = Vss2σ * (d_ratio**nss2σ)
            Vpsσ = Vspσ * (d_ratio**nspσ)
            Vspσ = Vspσ * (d_ratio**nspσ)
            Vdsσ = Vsdσ * (d_ratio**nsdσ)
            Vsdσ = Vsdσ * (d_ratio**nsdσ)

            Vs2s2σ = Vs2s2σ * (d_ratio**ns2s2σ)
            Vps2σ = Vs2pσ * (d_ratio**ns2pσ)
            Vs2pσ = Vs2pσ * (d_ratio**ns2pσ)
            Vds2σ = Vs2dσ * (d_ratio**ns2dσ)
            Vs2dσ = Vs2dσ * (d_ratio**ns2dσ)

            Vppσ = Vppσ * (d_ratio**nppσ)
            Vppπ = Vppπ * (d_ratio**nppπ)
            Vdpσ = Vpdσ * (d_ratio**npdσ)
            Vpdσ = Vpdσ * (d_ratio**npdσ)
            Vdpπ = Vpdπ * (d_ratio**npdπ)
            Vpdπ = Vpdπ * (d_ratio**npdπ)

            Vddσ = Vddσ * (d_ratio**nddσ)
            Vddπ = Vddπ * (d_ratio**nddπ)
            Vddδ = Vddδ * (d_ratio**nddδ)


            if (α == 's'):
                if (β == 's'):
                    mtx_elem = Vssσ
                elif (β == 'px'):
                    mtx_elem = l*Vspσ
                elif (β == 'py'):
                    mtx_elem = m*Vspσ
                elif (β == 'pz'):
                    mtx_elem = n*Vspσ
                elif (β == 'xy'):
                    mtx_elem = np.sqrt(3)*l*m*Vsdσ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3)*m*n*Vsdσ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3)*l*n*Vsdσ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vsdσ
                elif β == 'z2mr2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vsdσ
                elif β == 's2':
                    mtx_elem = Vss2σ
                else:
                    print("Error! s")
                    sys.exit()

            elif (α == 'px'):
                if β == 's':
                    mtx_elem = -l*Vpsσ
                elif β == 'px':
                    mtx_elem = l**2 * Vppσ + (1 - l**2) * Vppπ
                elif β == 'py':
                    mtx_elem = l*m*Vppσ - l*m*Vppπ
                elif β == 'pz':
                    mtx_elem = l*n*Vppσ - l*n*Vppπ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3) * l**2 * m * Vpdσ + m*(1 - 2*l**2)*Vpdπ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3) * l**2 * n * Vpdσ + n*(1 - 2*l**2)*Vpdπ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*l*(l**2 - m**2) * Vpdσ + l*(1 - l**2 + m**2) * Vpdπ
                elif β == 'z2mr2':
                    mtx_elem = l*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3) * l * n**2 * Vpdπ
                elif β == 's2':
                    mtx_elem = -l*Vps2σ
                else:
                    print("Error! px")
                    sys.exit()

            elif (α == 'py'):
                if β == 's':
                    mtx_elem = -m*Vpsσ
                elif β == 'px':
                    mtx_elem = l*m*Vppσ - l*m*Vppπ
                elif β == 'py':
                    mtx_elem = m**2 * Vppσ + (1. - m**2)*Vppπ
                elif β == 'pz':
                    mtx_elem = m*n*Vppσ - m*n*Vppπ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3) * m**2 * l * Vpdσ + l*(1-2*m**2)*Vpdπ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3) * m**2 * n * Vpdσ + n*(1 - 2*m**2)*Vpdπ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*m*(l**2 - m**2)*Vpdσ - m*(1 +  l**2 - m**2)*Vpdπ
                elif β == 'z2mr2':
                    mtx_elem = m*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3)*m*n**2*Vpdπ
                elif β == 's2':
                    mtx_elem = -m*Vps2σ
                else:
                    print("Error! py")
                    sys.exit()

            elif (α == 'pz'):
                if β == 's':
                    mtx_elem = -n*Vpsσ
                elif β == 'px':
                    mtx_elem = l*n*Vppσ - l*n*Vppπ
                elif β == 'py':
                    mtx_elem = m*n*Vppσ - m*n*Vppπ
                elif β == 'pz':
                    mtx_elem = n**2 * Vppσ + (1.-n**2)*Vppπ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3) * l*m*n*Vpdσ - 2*l*m*n*Vpdπ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3) * n**2 * m*Vpdσ + m*(1-2*n**2)*Vpdπ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3) * n**2 * l*Vpdσ + l*(1-2*n**2)*Vpdπ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*n*(l**2 - m**2)*Vpdσ - n*(l**2 - m**2)*Vpdπ
                elif β == 'z2mr2':
                    mtx_elem = n*(n**2 - (l**2 + m**2)/2.)*Vpdσ + np.sqrt(3)*n*(l**2 + m**2)*Vpdπ
                elif β == 's2':
                    mtx_elem = -n*Vps2σ
                else:
                    print("Error! pz")
                    sys.exit()

            elif (α == 'xy'):
                if β == 's':
                    mtx_elem = np.sqrt(3)*l*m*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(np.sqrt(3) * l**2 * m * Vdpσ + m*(1 - 2*l**2)*Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(np.sqrt(3) * m**2 * l * Vdpσ + l*(1-2*m**2)*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(np.sqrt(3) * l*m*n*Vdpσ - 2*l*m*n*Vdpπ)
                elif β == 'xy':
                    mtx_elem = 3 * l**2 * m**2 * Vddσ + (l**2 + m**2 - 4 * l**2 * m**2) * Vddπ + (n**2 + l**2 * m**2) * Vddδ
                elif β == 'yz':
                    mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
                elif β == 'zx':
                    mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
                elif β == 'z2mr2':
                    mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
                elif β == 's2':
                    mtx_elem = np.sqrt(3)*l*m*Vds2σ
                else:
                    print("Error! xy")
                    sys.exit()

            elif (α == 'yz'):
                if β == 's':
                    mtx_elem = np.sqrt(3)*m*n*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(np.sqrt(3) * m**2 * n * Vdpσ + n*(1 - 2*m**2)*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(np.sqrt(3) * n**2 * m*Vdpσ + m*(1-2*n**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
                elif β == 'yz':
                    mtx_elem = 3 * m**2 * n**2 * Vddσ + (m**2 + n**2 - 4 * m**2 * n**2)*Vddπ + (l**2 + m**2 * n**2)*Vddδ
                elif β == 'zx':
                    mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
                elif β == 's2':
                    mtx_elem = np.sqrt(3)*m*n*Vds2σ
                else:
                    print("Error! yz")
                    sys.exit()

            elif (α == 'zx'):
                if β == 's':
                    mtx_elem = np.sqrt(3)*l*n*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(np.sqrt(3) * l**2 * n * Vdpσ + n*(1 - 2*l**2)*Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(np.sqrt(3) * n**2 * l*Vdpσ + l*(1-2*n**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
                elif β == 'yz':
                    mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
                elif β == 'zx':
                    mtx_elem = 3 * l**2 * n**2 * Vddσ + (n**2 + l**2 - 4 * l**2 * n**2)*Vddπ + (m**2 + l**2 * n**2)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
                elif β == 's2':
                    mtx_elem = np.sqrt(3)*l*n*Vds2σ
                else:
                    print("Error! zx")
                    sys.exit()

            elif (α == 'x2my2'):
                if β == 's':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*((np.sqrt(3)/2.)*l*(l**2 - m**2) * Vdpσ + l*(1 - l**2 + m**2) * Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*((np.sqrt(3)/2.)*m*(l**2 - m**2)*Vdpσ - m*(1 +  l**2 - m**2)*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*((np.sqrt(3)/2.)*n*(l**2 - m**2)*Vdpσ - n*(l**2 - m**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
                elif β == 'yz':
                    mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
                elif β == 'zx':
                    mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./4.)*(l**2 - m**2)*Vddσ + (l**2 + m**2 - (l**2 - m**2)**2)*Vddπ + (n**2 + (l**2 - m**2)**2/4.)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
                elif β == 's2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vds2σ
                else:
                    print("Error! x2my2")
                    sys.exit()

            elif (α == 'z2mr2'):
                if β == 's':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(l*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3) * l * n**2 * Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(m*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3)*m*n**2*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(n*(n**2 - (l**2 + m**2)/2.)*Vdpσ + np.sqrt(3)*n*(l**2 + m**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)**2 * Vddσ + 3 * n**2 * (l**2 + m**2)*Vddπ + (3./4.)*(l**2 + m**2)**2 * Vddδ
                elif β == 's2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vds2σ
                else:
                    print("Error! z2mr2")
                    sys.exit()

            elif (α == 's2'):
                if β == 's':
                    mtx_elem = Vs2sσ
                elif β == 'px':
                    mtx_elem = l*Vs2pσ
                elif β == 'py':
                    mtx_elem = m*Vs2pσ
                elif β == 'pz':
                    mtx_elem = n*Vs2pσ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3)*l*m*Vs2dσ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3)*m*n*Vs2dσ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3)*l*n*Vs2dσ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2) * Vs2dσ
                elif β == 'z2mr2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vs2dσ
                elif β == 's2':
                    mtx_elem = Vs2s2σ
                else:
                    print("Error! s2")
                    sys.exit()

            else:
                print("Error! α")

            return mtx_elem

        def calc_interatomic_mtx_element_Ge_Ge(α,β,l,m,n,d):
            ### Calculates the interatomic matrix element between the
            ### orbital β on the reference Ge atom and the α orbital on
            ### the target Ge atom
            ###     * l,m,n are the directional cosines of the vector
            ###       going from the reference to the target atom
            ###     * d is the distance between atoms

            d_Ge = 5.658 * np.sqrt(3.)/4.
            d_ratio = (d_Ge/d) # ratio of unstrained and strained inter-atomic distance

            ### Unstrained band parameters
            E_offset = 0.68
            Es = -4.08253 + E_offset
            Ep = 4.63470 + E_offset
            Ed = 12.19526 + E_offset
            Es2 = 23.20167 + E_offset

            Vssσ = -1.49093
            Vspσ = 2.91277
            Vsdσ = -2.10114
            Vss2σ = -1.59479

            Vppσ = 4.36624
            Vppπ = -1.58305
            Vpdσ = -1.60110
            Vpdπ = 2.36977

            Vddσ = -1.15483
            Vddπ = 2.30042
            Vddδ = -1.19386

            Vs2pσ = 2.92036
            Vs2dσ = -0.23561
            Vs2s2σ = -4.86118

            Delta_SO = 3*0.12742

            ### Deformation exponentials
            nssσ = 3.57536
            nss2σ = 1.03634
            nspσ = 2.88203
            nsdσ = 1.89283

            ns2s2σ = 1.07935
            ns2pσ = 2.64809
            ns2dσ = 2.33424

            nppσ = 2.40576
            nppπ = 2.95026
            npdσ = 0.51325
            npdπ = 1.62421

            nddσ = 1.68410
            nddπ = 2.64952
            nddδ = 3.83221


            ### Calculate the renormalized matrix elements
            Vssσ = Vssσ * (d_ratio**nssσ)
            Vs2sσ = Vss2σ * (d_ratio**nss2σ)
            Vss2σ = Vss2σ * (d_ratio**nss2σ)
            Vpsσ = Vspσ * (d_ratio**nspσ)
            Vspσ = Vspσ * (d_ratio**nspσ)
            Vdsσ = Vsdσ * (d_ratio**nsdσ)
            Vsdσ = Vsdσ * (d_ratio**nsdσ)

            Vs2s2σ = Vs2s2σ * (d_ratio**ns2s2σ)
            Vps2σ = Vs2pσ * (d_ratio**ns2pσ)
            Vs2pσ = Vs2pσ * (d_ratio**ns2pσ)
            Vds2σ = Vs2dσ * (d_ratio**ns2dσ)
            Vs2dσ = Vs2dσ * (d_ratio**ns2dσ)

            Vppσ = Vppσ * (d_ratio**nppσ)
            Vppπ = Vppπ * (d_ratio**nppπ)
            Vdpσ = Vpdσ * (d_ratio**npdσ)
            Vpdσ = Vpdσ * (d_ratio**npdσ)
            Vdpπ = Vpdπ * (d_ratio**npdπ)
            Vpdπ = Vpdπ * (d_ratio**npdπ)

            Vddσ = Vddσ * (d_ratio**nddσ)
            Vddπ = Vddπ * (d_ratio**nddπ)
            Vddδ = Vddδ * (d_ratio**nddδ)


            if (α == 's'):
                if (β == 's'):
                    mtx_elem = Vssσ
                elif (β == 'px'):
                    mtx_elem = l*Vspσ
                elif (β == 'py'):
                    mtx_elem = m*Vspσ
                elif (β == 'pz'):
                    mtx_elem = n*Vspσ
                elif (β == 'xy'):
                    mtx_elem = np.sqrt(3)*l*m*Vsdσ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3)*m*n*Vsdσ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3)*l*n*Vsdσ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vsdσ
                elif β == 'z2mr2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vsdσ
                elif β == 's2':
                    mtx_elem = Vss2σ
                else:
                    print("Error! s")
                    sys.exit()

            elif (α == 'px'):
                if β == 's':
                    mtx_elem = -l*Vpsσ
                elif β == 'px':
                    mtx_elem = l**2 * Vppσ + (1 - l**2) * Vppπ
                elif β == 'py':
                    mtx_elem = l*m*Vppσ - l*m*Vppπ
                elif β == 'pz':
                    mtx_elem = l*n*Vppσ - l*n*Vppπ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3) * l**2 * m * Vpdσ + m*(1 - 2*l**2)*Vpdπ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3) * l**2 * n * Vpdσ + n*(1 - 2*l**2)*Vpdπ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*l*(l**2 - m**2) * Vpdσ + l*(1 - l**2 + m**2) * Vpdπ
                elif β == 'z2mr2':
                    mtx_elem = l*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3) * l * n**2 * Vpdπ
                elif β == 's2':
                    mtx_elem = -l*Vps2σ
                else:
                    print("Error! px")
                    sys.exit()

            elif (α == 'py'):
                if β == 's':
                    mtx_elem = -m*Vpsσ
                elif β == 'px':
                    mtx_elem = l*m*Vppσ - l*m*Vppπ
                elif β == 'py':
                    mtx_elem = m**2 * Vppσ + (1. - m**2)*Vppπ
                elif β == 'pz':
                    mtx_elem = m*n*Vppσ - m*n*Vppπ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3) * m**2 * l * Vpdσ + l*(1-2*m**2)*Vpdπ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3) * m**2 * n * Vpdσ + n*(1 - 2*m**2)*Vpdπ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*m*(l**2 - m**2)*Vpdσ - m*(1 +  l**2 - m**2)*Vpdπ
                elif β == 'z2mr2':
                    mtx_elem = m*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3)*m*n**2*Vpdπ
                elif β == 's2':
                    mtx_elem = -m*Vps2σ
                else:
                    print("Error! py")
                    sys.exit()

            elif (α == 'pz'):
                if β == 's':
                    mtx_elem = -n*Vpsσ
                elif β == 'px':
                    mtx_elem = l*n*Vppσ - l*n*Vppπ
                elif β == 'py':
                    mtx_elem = m*n*Vppσ - m*n*Vppπ
                elif β == 'pz':
                    mtx_elem = n**2 * Vppσ + (1.-n**2)*Vppπ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3) * l*m*n*Vpdσ - 2*l*m*n*Vpdπ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3) * n**2 * m*Vpdσ + m*(1-2*n**2)*Vpdπ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3) * n**2 * l*Vpdσ + l*(1-2*n**2)*Vpdπ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*n*(l**2 - m**2)*Vpdσ - n*(l**2 - m**2)*Vpdπ
                elif β == 'z2mr2':
                    mtx_elem = n*(n**2 - (l**2 + m**2)/2.)*Vpdσ + np.sqrt(3)*n*(l**2 + m**2)*Vpdπ
                elif β == 's2':
                    mtx_elem = -n*Vps2σ
                else:
                    print("Error! pz")
                    sys.exit()

            elif (α == 'xy'):
                if β == 's':
                    mtx_elem = np.sqrt(3)*l*m*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(np.sqrt(3) * l**2 * m * Vdpσ + m*(1 - 2*l**2)*Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(np.sqrt(3) * m**2 * l * Vdpσ + l*(1-2*m**2)*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(np.sqrt(3) * l*m*n*Vdpσ - 2*l*m*n*Vdpπ)
                elif β == 'xy':
                    mtx_elem = 3 * l**2 * m**2 * Vddσ + (l**2 + m**2 - 4 * l**2 * m**2) * Vddπ + (n**2 + l**2 * m**2) * Vddδ
                elif β == 'yz':
                    mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
                elif β == 'zx':
                    mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
                elif β == 'z2mr2':
                    mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
                elif β == 's2':
                    mtx_elem = np.sqrt(3)*l*m*Vds2σ
                else:
                    print("Error! xy")
                    sys.exit()

            elif (α == 'yz'):
                if β == 's':
                    mtx_elem = np.sqrt(3)*m*n*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(np.sqrt(3) * m**2 * n * Vdpσ + n*(1 - 2*m**2)*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(np.sqrt(3) * n**2 * m*Vdpσ + m*(1-2*n**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
                elif β == 'yz':
                    mtx_elem = 3 * m**2 * n**2 * Vddσ + (m**2 + n**2 - 4 * m**2 * n**2)*Vddπ + (l**2 + m**2 * n**2)*Vddδ
                elif β == 'zx':
                    mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
                elif β == 's2':
                    mtx_elem = np.sqrt(3)*m*n*Vds2σ
                else:
                    print("Error! yz")
                    sys.exit()

            elif (α == 'zx'):
                if β == 's':
                    mtx_elem = np.sqrt(3)*l*n*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(np.sqrt(3) * l**2 * n * Vdpσ + n*(1 - 2*l**2)*Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(np.sqrt(3) * n**2 * l*Vdpσ + l*(1-2*n**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
                elif β == 'yz':
                    mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
                elif β == 'zx':
                    mtx_elem = 3 * l**2 * n**2 * Vddσ + (n**2 + l**2 - 4 * l**2 * n**2)*Vddπ + (m**2 + l**2 * n**2)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
                elif β == 's2':
                    mtx_elem = np.sqrt(3)*l*n*Vds2σ
                else:
                    print("Error! zx")
                    sys.exit()

            elif (α == 'x2my2'):
                if β == 's':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*((np.sqrt(3)/2.)*l*(l**2 - m**2) * Vdpσ + l*(1 - l**2 + m**2) * Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*((np.sqrt(3)/2.)*m*(l**2 - m**2)*Vdpσ - m*(1 +  l**2 - m**2)*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*((np.sqrt(3)/2.)*n*(l**2 - m**2)*Vdpσ - n*(l**2 - m**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
                elif β == 'yz':
                    mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
                elif β == 'zx':
                    mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./4.)*(l**2 - m**2)*Vddσ + (l**2 + m**2 - (l**2 - m**2)**2)*Vddπ + (n**2 + (l**2 - m**2)**2/4.)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
                elif β == 's2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vds2σ
                else:
                    print("Error! x2my2")
                    sys.exit()

            elif (α == 'z2mr2'):
                if β == 's':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(l*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3) * l * n**2 * Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(m*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3)*m*n**2*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(n*(n**2 - (l**2 + m**2)/2.)*Vdpσ + np.sqrt(3)*n*(l**2 + m**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)**2 * Vddσ + 3 * n**2 * (l**2 + m**2)*Vddπ + (3./4.)*(l**2 + m**2)**2 * Vddδ
                elif β == 's2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vds2σ
                else:
                    print("Error! z2mr2")
                    sys.exit()

            elif (α == 's2'):
                if β == 's':
                    mtx_elem = Vs2sσ
                elif β == 'px':
                    mtx_elem = l*Vs2pσ
                elif β == 'py':
                    mtx_elem = m*Vs2pσ
                elif β == 'pz':
                    mtx_elem = n*Vs2pσ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3)*l*m*Vs2dσ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3)*m*n*Vs2dσ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3)*l*n*Vs2dσ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2) * Vs2dσ
                elif β == 'z2mr2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vs2dσ
                elif β == 's2':
                    mtx_elem = Vs2s2σ
                else:
                    print("Error! s2")
                    sys.exit()

            else:
                print("Error! α")

            return mtx_elem

        def calc_interatomic_mtx_element_Si_Ge(α,β,l,m,n,d):
            ### Calculates the interatomic matrix element between the
            ### orbital β on the reference Ge atom and the α orbital on
            ### the target Si atom
            ###     * l,m,n are the directional cosines of the vector
            ###       going from the reference to the target atom
            ###     * d is the distance between atoms

            d_o = 2.39792
            d_ratio = (d_o/d) # ratio of unstrained and strained inter-atomic distance

            ### Unstrained band parameters

            Vssσ = -1.67650
            Vspσ = 2.82890
            Vpsσ = 3.01033
            Vsdσ = -2.13989
            Vdsσ = -2.04737
            Vss2σ = -1.50940
            Vs2sσ = -1.50314

            Vppσ = 4.21933
            Vppπ = -1.54668
            Vpdσ = -1.43412
            Vdpσ = -1.61322
            Vpdπ = 2.57110
            Vdpπ = 2.43552

            Vddσ = -1.41949
            Vddπ = 2.62540
            Vddδ = -1.39382

            Vs2pσ = 3.06299
            Vps2σ = 2.79296
            Vs2dσ = -0.46386
            Vds2σ = -0.51235
            Vs2s2σ = -4.63349

            ### Deformation exponentials
            nssσ = 3.90172
            nss2σ = 1.03801
            nspσ = 2.37280
            nsdσ = 1.99537

            ns2s2σ = 0.85993
            ns2pσ = 1.94143
            ns2dσ = 2.01051

            nppσ = 2.34995
            nppπ = 3.08150
            npdσ = 0.75549
            npdπ = 1.67031

            nddσ = 1.66975
            nddπ = 2.24973
            nddδ = 3.06305


            ### Calculate the renormalized matrix elements
            Vssσ = Vssσ * (d_ratio**nssσ)
            Vss2σ = Vss2σ * (d_ratio**nss2σ)
            Vs2sσ = Vs2sσ * (d_ratio**nss2σ)
            Vspσ = Vspσ * (d_ratio**nspσ)
            Vpsσ = Vpsσ * (d_ratio**nspσ)
            Vsdσ = Vsdσ * (d_ratio**nsdσ)
            Vdsσ = Vdsσ * (d_ratio**nsdσ)

            Vs2s2σ = Vs2s2σ * (d_ratio**ns2s2σ)
            Vs2pσ = Vs2pσ * (d_ratio**ns2pσ)
            Vps2σ = Vps2σ * (d_ratio**ns2pσ)
            Vs2dσ = Vs2dσ * (d_ratio**ns2dσ)
            Vds2σ = Vds2σ * (d_ratio**ns2dσ)

            Vppσ = Vppσ * (d_ratio**nppσ)
            Vppπ = Vppπ * (d_ratio**nppπ)
            Vpdσ = Vpdσ * (d_ratio**npdσ)
            Vdpσ = Vdpσ * (d_ratio**npdσ)
            Vpdπ = Vpdπ * (d_ratio**npdπ)
            Vdpπ = Vdpπ * (d_ratio**npdπ)

            Vddσ = Vddσ * (d_ratio**nddσ)
            Vddπ = Vddπ * (d_ratio**nddπ)
            Vddδ = Vddδ * (d_ratio**nddδ)


            if (α == 's'):
                if (β == 's'):
                    mtx_elem = Vssσ
                elif (β == 'px'):
                    mtx_elem = l*Vspσ
                elif (β == 'py'):
                    mtx_elem = m*Vspσ
                elif (β == 'pz'):
                    mtx_elem = n*Vspσ
                elif (β == 'xy'):
                    mtx_elem = np.sqrt(3)*l*m*Vsdσ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3)*m*n*Vsdσ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3)*l*n*Vsdσ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vsdσ
                elif β == 'z2mr2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vsdσ
                elif β == 's2':
                    mtx_elem = Vss2σ
                else:
                    print("Error! s")
                    sys.exit()

            elif (α == 'px'):
                if β == 's':
                    mtx_elem = -l*Vpsσ
                elif β == 'px':
                    mtx_elem = l**2 * Vppσ + (1 - l**2) * Vppπ
                elif β == 'py':
                    mtx_elem = l*m*Vppσ - l*m*Vppπ
                elif β == 'pz':
                    mtx_elem = l*n*Vppσ - l*n*Vppπ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3) * l**2 * m * Vpdσ + m*(1 - 2*l**2)*Vpdπ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3) * l**2 * n * Vpdσ + n*(1 - 2*l**2)*Vpdπ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*l*(l**2 - m**2) * Vpdσ + l*(1 - l**2 + m**2) * Vpdπ
                elif β == 'z2mr2':
                    mtx_elem = l*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3) * l * n**2 * Vpdπ
                elif β == 's2':
                    mtx_elem = -l*Vps2σ
                else:
                    print("Error! px")
                    sys.exit()

            elif (α == 'py'):
                if β == 's':
                    mtx_elem = -m*Vpsσ
                elif β == 'px':
                    mtx_elem = l*m*Vppσ - l*m*Vppπ
                elif β == 'py':
                    mtx_elem = m**2 * Vppσ + (1. - m**2)*Vppπ
                elif β == 'pz':
                    mtx_elem = m*n*Vppσ - m*n*Vppπ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3) * m**2 * l * Vpdσ + l*(1-2*m**2)*Vpdπ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3) * m**2 * n * Vpdσ + n*(1 - 2*m**2)*Vpdπ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3) * l*m*n * Vpdσ - 2*l*m*n*Vpdπ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*m*(l**2 - m**2)*Vpdσ - m*(1 +  l**2 - m**2)*Vpdπ
                elif β == 'z2mr2':
                    mtx_elem = m*(n**2 - (l**2 + m**2)/2.)*Vpdσ - np.sqrt(3)*m*n**2*Vpdπ
                elif β == 's2':
                    mtx_elem = -m*Vps2σ
                else:
                    print("Error! py")
                    sys.exit()

            elif (α == 'pz'):
                if β == 's':
                    mtx_elem = -n*Vpsσ
                elif β == 'px':
                    mtx_elem = l*n*Vppσ - l*n*Vppπ
                elif β == 'py':
                    mtx_elem = m*n*Vppσ - m*n*Vppπ
                elif β == 'pz':
                    mtx_elem = n**2 * Vppσ + (1.-n**2)*Vppπ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3) * l*m*n*Vpdσ - 2*l*m*n*Vpdπ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3) * n**2 * m*Vpdσ + m*(1-2*n**2)*Vpdπ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3) * n**2 * l*Vpdσ + l*(1-2*n**2)*Vpdπ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*n*(l**2 - m**2)*Vpdσ - n*(l**2 - m**2)*Vpdπ
                elif β == 'z2mr2':
                    mtx_elem = n*(n**2 - (l**2 + m**2)/2.)*Vpdσ + np.sqrt(3)*n*(l**2 + m**2)*Vpdπ
                elif β == 's2':
                    mtx_elem = -n*Vps2σ
                else:
                    print("Error! pz")
                    sys.exit()

            elif (α == 'xy'):
                if β == 's':
                    mtx_elem = np.sqrt(3)*l*m*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(np.sqrt(3) * l**2 * m * Vdpσ + m*(1 - 2*l**2)*Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(np.sqrt(3) * m**2 * l * Vdpσ + l*(1-2*m**2)*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(np.sqrt(3) * l*m*n*Vdpσ - 2*l*m*n*Vdpπ)
                elif β == 'xy':
                    mtx_elem = 3 * l**2 * m**2 * Vddσ + (l**2 + m**2 - 4 * l**2 * m**2) * Vddπ + (n**2 + l**2 * m**2) * Vddδ
                elif β == 'yz':
                    mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
                elif β == 'zx':
                    mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
                elif β == 'z2mr2':
                    mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
                elif β == 's2':
                    mtx_elem = np.sqrt(3)*l*m*Vds2σ
                else:
                    print("Error! xy")
                    sys.exit()

            elif (α == 'yz'):
                if β == 's':
                    mtx_elem = np.sqrt(3)*m*n*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(np.sqrt(3) * m**2 * n * Vdpσ + n*(1 - 2*m**2)*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(np.sqrt(3) * n**2 * m*Vdpσ + m*(1-2*n**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = 3 * l * m**2 * n * Vddσ + l*n*(1-4*m**2)*Vddπ + l*n*(m**2 - 1)*Vddδ
                elif β == 'yz':
                    mtx_elem = 3 * m**2 * n**2 * Vddσ + (m**2 + n**2 - 4 * m**2 * n**2)*Vddπ + (l**2 + m**2 * n**2)*Vddδ
                elif β == 'zx':
                    mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
                elif β == 's2':
                    mtx_elem = np.sqrt(3)*m*n*Vds2σ
                else:
                    print("Error! yz")
                    sys.exit()

            elif (α == 'zx'):
                if β == 's':
                    mtx_elem = np.sqrt(3)*l*n*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(np.sqrt(3) * l**2 * n * Vdpσ + n*(1 - 2*l**2)*Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(np.sqrt(3) * l*m*n * Vdpσ - 2*l*m*n*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(np.sqrt(3) * n**2 * l*Vdpσ + l*(1-2*n**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = 3 * l**2 * m * n * Vddσ + m*n*(1-4*l**2)*Vddπ + m*n*(l**2 - 1)*Vddδ
                elif β == 'yz':
                    mtx_elem = 3 * l * m * n**2 * Vddσ + l*m*(1-4*n**2)*Vddπ + l*m*(n**2 - 1)*Vddδ
                elif β == 'zx':
                    mtx_elem = 3 * l**2 * n**2 * Vddσ + (n**2 + l**2 - 4 * l**2 * n**2)*Vddπ + (m**2 + l**2 * n**2)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
                elif β == 's2':
                    mtx_elem = np.sqrt(3)*l*n*Vds2σ
                else:
                    print("Error! zx")
                    sys.exit()

            elif (α == 'x2my2'):
                if β == 's':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*((np.sqrt(3)/2.)*l*(l**2 - m**2) * Vdpσ + l*(1 - l**2 + m**2) * Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*((np.sqrt(3)/2.)*m*(l**2 - m**2)*Vdpσ - m*(1 +  l**2 - m**2)*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*((np.sqrt(3)/2.)*n*(l**2 - m**2)*Vdpσ - n*(l**2 - m**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = (3./2.)*l*m*(l**2 - m**2)*Vddσ + 2*l*m*(m**2 - l**2) * Vddπ + l*m*(l**2 - m**2)/2. * Vddδ
                elif β == 'yz':
                    mtx_elem = (3./2.)*m*n*(l**2 - m**2)*Vddσ - m*n*(1 + 2*(l**2 - m**2))*Vddπ + m*n*(1 + (l**2 - m**2)/2.)*Vddδ
                elif β == 'zx':
                    mtx_elem = (3./2.)*n*l*(l**2 - m**2)*Vddσ + n*l*(1 - 2*(l**2 - m**2))*Vddπ - n*l*(1 - (l**2 - m**2)/2.)*Vddδ
                elif β == 'x2my2':
                    mtx_elem = (3./4.)*(l**2 - m**2)*Vddσ + (l**2 + m**2 - (l**2 - m**2)**2)*Vddπ + (n**2 + (l**2 - m**2)**2/4.)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
                elif β == 's2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*Vds2σ
                else:
                    print("Error! x2my2")
                    sys.exit()

            elif (α == 'z2mr2'):
                if β == 's':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vdsσ
                elif β == 'px':
                    mtx_elem = -1*(l*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3) * l * n**2 * Vdpπ)
                elif β == 'py':
                    mtx_elem = -1*(m*(n**2 - (l**2 + m**2)/2.)*Vdpσ - np.sqrt(3)*m*n**2*Vdpπ)
                elif β == 'pz':
                    mtx_elem = -1*(n*(n**2 - (l**2 + m**2)/2.)*Vdpσ + np.sqrt(3)*n*(l**2 + m**2)*Vdpπ)
                elif β == 'xy':
                    mtx_elem = np.sqrt(3)*l*m*(n**2 - (l**2+m**2)/2.)*Vddσ - 2*np.sqrt(3)*l*m * n**2 * Vddπ + np.sqrt(3)*l*m*(1+n**2)/2. * Vddδ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3)*m*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*m*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*m*n*(l**2 + m**2)/2. * Vddδ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3)*l*n*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3)*l*n*(l**2 + m**2 - n**2)*Vddπ - np.sqrt(3)*l*n*(l**2 + m**2)/2. * Vddδ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2)*(n**2 - (l**2 + m**2)/2.)*Vddσ + np.sqrt(3) * n**2 * (m**2 - l**2)*Vddπ + (np.sqrt(3)/4.)*(1+n**2)*(l**2 - m**2)*Vddδ
                elif β == 'z2mr2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)**2 * Vddσ + 3 * n**2 * (l**2 + m**2)*Vddπ + (3./4.)*(l**2 + m**2)**2 * Vddδ
                elif β == 's2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vds2σ
                else:
                    print("Error! z2mr2")
                    sys.exit()

            elif (α == 's2'):
                if β == 's':
                    mtx_elem = Vs2sσ
                elif β == 'px':
                    mtx_elem = l*Vs2pσ
                elif β == 'py':
                    mtx_elem = m*Vs2pσ
                elif β == 'pz':
                    mtx_elem = n*Vs2pσ
                elif β == 'xy':
                    mtx_elem = np.sqrt(3)*l*m*Vs2dσ
                elif β == 'yz':
                    mtx_elem = np.sqrt(3)*m*n*Vs2dσ
                elif β == 'zx':
                    mtx_elem = np.sqrt(3)*l*n*Vs2dσ
                elif β == 'x2my2':
                    mtx_elem = (np.sqrt(3)/2.)*(l**2 - m**2) * Vs2dσ
                elif β == 'z2mr2':
                    mtx_elem = (n**2 - (l**2 + m**2)/2.)*Vs2dσ
                elif β == 's2':
                    mtx_elem = Vs2s2σ
                else:
                    print("Error! s2")
                    sys.exit()

            else:
                print("Error! α")

            return mtx_elem

        def calc_interatomic_mtx_element_Ge_Si(α,β,l,m,n,d):
            ### Calculates the interatomic matrix element between the
            ### orbital β on the reference Si atom and the α orbital on
            ### the target Ge atom
            ###     * l,m,n are the directional cosines of the vector
            ###       going from the reference to the target atom
            ###     * d is the distance between atoms

            mtx_elem = calc_interatomic_mtx_element_Si_Ge(β,α,-l,-m,-n,d)
            return mtx_elem

        def Hopping_mtx_VC_spinless_gen(kx,ky,xGe_1,xGe_2,a_par,a_perp_nP1n,epsilon_xy,zeta,n):
            ### Generates the matrix contain the coupling between layers n and n+1
            ### with Germanium concentrations xGe_1 and xGe_2, respectively
            ###     * No spin included in this function
            ###     * The orbital ordering is {s, s*, px, py, pz, dyz, dxz, dxy, dx2my2, dz2}
            ###     * epsilon_xy = ε_{xy} shear strain

            H_np1n_SiSi = np.zeros((10,10),dtype = 'complex') # Hopping matrix H_{n+1,n} if layers were pure silicon
            H_np1n_GeGe = np.zeros((10,10),dtype = 'complex') # Hopping matrix H_{n+1,n} if layers were pure germanium
            H_np1n_SiGe = np.zeros((10,10),dtype = 'complex') # Hopping matrix H_{n+1,n} if layer 1 was Ge and layer 2 was Si
            H_np1n_GeSi = np.zeros((10,10),dtype = 'complex') # Hopping matrix H_{n+1,n} if layer 1 was Si and layer 2 was Ge

            ### Directional cosines
            if n % 2 == 0: # n is even (sublattice 1)
                ### 1st nearest neighbor
                r1_x = (a_par/4.)*(1 + epsilon_xy)
                r1_y = (a_par/4.)*(1 + epsilon_xy)
                r1_z = (a_perp_nP1n/4.)*(1 - 2*zeta*epsilon_xy)
                d = np.sqrt(r1_x**2 + r1_y**2 + r1_z**2) # bond length
                l1 = r1_x/d
                m1 = r1_y/d
                n1 = r1_z/d
                phase_1 = np.exp(-1j*kx*r1_x - 1j*ky*r1_y)

                ### 2nd nearest neighbor
                r2_x = -r1_x
                r2_y = -r1_y
                r2_z = 1*r1_z
                l2 = r2_x/d
                m2 = r2_y/d
                n2 = r2_z/d
                phase_2 = np.exp(-1j*kx*r2_x - 1j*ky*r2_y)
            elif n % 2 == 1: # n is odd (sublattice 2)
                ### 1st nearest neighbor
                r1_x = (a_par/4.)*(1 - epsilon_xy)
                r1_y = -(a_par/4.)*(1 - epsilon_xy)
                r1_z = (a_perp_nP1n/4.)*(1 + 2*zeta*epsilon_xy)
                d = np.sqrt(r1_x**2 + r1_y**2 + r1_z**2) # bond length
                l1 = r1_x/d
                m1 = r1_y/d
                n1 = r1_z/d
                phase_1 = np.exp(-1j*kx*r1_x - 1j*ky*r1_y)

                ### 2nd nearest neighbor
                r2_x = -r1_x
                r2_y = -r1_y
                r2_z = 1*r1_z
                l2 = r2_x/d
                m2 = r2_y/d
                n2 = r2_z/d
                phase_2 = np.exp(-1j*kx*r2_x - 1j*ky*r2_y)

            ### Calculate interatomic matrix elements
            orb_arr = ['s','s2','px','py','pz','yz','zx','xy','x2my2','z2mr2']
            for i in range(10):
                for j in range(10):
                    H_np1n_SiSi[i,j] += phase_1 * calc_interatomic_mtx_element_Si_Si(orb_arr[i],orb_arr[j],l1,m1,n1,d) \
                                        +phase_2 * calc_interatomic_mtx_element_Si_Si(orb_arr[i],orb_arr[j],l2,m2,n2,d)
                    H_np1n_GeGe[i,j] += phase_1 * calc_interatomic_mtx_element_Ge_Ge(orb_arr[i],orb_arr[j],l1,m1,n1,d) \
                                        +phase_2 * calc_interatomic_mtx_element_Ge_Ge(orb_arr[i],orb_arr[j],l2,m2,n2,d)
                    H_np1n_SiGe[i,j] += phase_1 * calc_interatomic_mtx_element_Si_Ge(orb_arr[i],orb_arr[j],l1,m1,n1,d) \
                                        +phase_2 * calc_interatomic_mtx_element_Si_Ge(orb_arr[i],orb_arr[j],l2,m2,n2,d)
                    H_np1n_GeSi[i,j] += phase_1 * calc_interatomic_mtx_element_Ge_Si(orb_arr[i],orb_arr[j],l1,m1,n1,d) \
                                        +phase_2 * calc_interatomic_mtx_element_Ge_Si(orb_arr[i],orb_arr[j],l2,m2,n2,d)
            H_np1n = ((1-xGe_2)*(1 - xGe_1))*H_np1n_SiSi + (xGe_2*(1 - xGe_1))*H_np1n_GeSi \
                     +((1 - xGe_2)*xGe_1)*H_np1n_SiGe + (xGe_2*xGe_1)*H_np1n_GeGe
            return H_np1n

    ### intra-atomic functions
    if True:

        def Onsite_mtx_spinless_Si(n,a_par,a_perp_nnM1,a_perp_nP1n,epsilon_xy,zeta):
            ### Calculates the onsite spinless Hamiltonian of a
            ### Si atom in the nth layer
            ###     * a_par is the in-plane lattice constant set by the barrier region
            ###     * a_perp_nnM1 is the z-direction lattice constant between the n and n-1 layer
            ###     * a_perp_nP1n is the z-direction lattice constant between the n+1 and n layer
            ###     * epsilon_xy = ε_{xy} shear strain
            ###     * The orbital ordering is {s, s*, px, py, pz, dyz, dxz, dxy, dx2my2, dz2}

            ### onsite strain parameters (in eV)
            alpha_s = -0.13357
            alpha_p = -0.18953
            alpha_d = -0.89046
            alpha_s2 = -0.24373
            beta_p0 = 1.13646
            beta_p1 = -2.76257
            beta_pd0 = -0.13011
            beta_pd1 = -3.28537
            beta_d0 = 3.59603
            beta_sp0 = 1.97665
            beta_s2p0 = -2.18403
            beta_sd0 = 3.06840
            beta_s2d0 = -4.95860

            ### Unstrained orbital energies
            E_offset = 0.
            Es = -2.55247  + E_offset
            Ep = 4.48593  + E_offset
            Ed = 14.01053  + E_offset
            Es2 = 23.44607  + E_offset

            ### nearest neighbor, bond distance, and cosine arrays
            #a_Si = 5.431 # lattice constant of unstrained Si
            r_arr = np.zeros((4,3))
            if n % 2 == 0:
                r1_x = (a_par/4.)*(1 + epsilon_xy)
                r1_y = (a_par/4.)*(1 + epsilon_xy)
                r1_z = (a_perp_nP1n/4.)*(1 - 2*zeta*epsilon_xy)
                r_arr[0,:] = np.array([r1_x,r1_y,r1_z])
                r_arr[1,:] = np.array([-r1_x,-r1_y,r1_z])
                r3_x = -(a_par/4.)*(1 - epsilon_xy)
                r3_y = (a_par/4.)*(1 - epsilon_xy)
                r3_z = -(a_perp_nnM1/4.)*(1 + 2*zeta*epsilon_xy)
                r_arr[2,:] = np.array([r3_x,r3_y,r3_z])
                r_arr[3,:] = np.array([-r3_x,-r3_y,r3_z])
            elif n % 2 == 1:
                r1_x = (a_par/4.)*(1 - epsilon_xy)
                r1_y = -(a_par/4.)*(1 - epsilon_xy)
                r1_z = (a_perp_nP1n/4.)*(1 + 2*zeta*epsilon_xy)
                r_arr[0,:] = np.array([r1_x,r1_y,r1_z])
                r_arr[1,:] = np.array([-r1_x,-r1_y,r1_z])
                r3_x = -(a_par/4.)*(1 + epsilon_xy)
                r3_y = -(a_par/4.)*(1 + epsilon_xy)
                r3_z = -(a_perp_nnM1/4.)*(1 - 2*zeta*epsilon_xy)
                r_arr[2,:] = np.array([r3_x,r3_y,r3_z])
                r_arr[3,:] = np.array([-r3_x,-r3_y,r3_z])
            d_arr = np.zeros(4) # bond distance array
            for j in range(4):
                d_arr[j] = np.linalg.norm(r_arr[j,:])
            cosine_arr = np.zeros((4,3)) # cosine array
            for j in range(4):
                for b in range(3):
                    cosine_arr[j,b] = r_arr[j,b]/d_arr[j]

            ### Calculate the hydrostatic strain
            d_Si_o = 5.431 * np.sqrt(3.)/4. # unstrained bond distance
            hydro_strain = 0
            for j in range(4):
                hydro_strain += .75*(d_arr[j] - d_Si_o)/d_Si_o
            #print("hydro_strain = ", hydro_strain)

            ### Set the initial onsite energies with hydrostatic strain included
            Onsite_mtx = np.zeros((10,10))
            Onsite_mtx[0,0] = Es + alpha_s*hydro_strain
            Onsite_mtx[1,1] = Es2 + alpha_s2*hydro_strain
            Onsite_mtx[2,2] = Ep + alpha_p*hydro_strain
            Onsite_mtx[3,3] = Ep + alpha_p*hydro_strain
            Onsite_mtx[4,4] = Ep + alpha_p*hydro_strain
            Onsite_mtx[5,5] = Ed + alpha_d*hydro_strain
            Onsite_mtx[6,6] = Ed + alpha_d*hydro_strain
            Onsite_mtx[7,7] = Ed + alpha_d*hydro_strain
            Onsite_mtx[8,8] = Ed + alpha_d*hydro_strain
            Onsite_mtx[9,9] = Ed + alpha_d*hydro_strain

            #return Onsite_mtx

            ### Include the non-hyrdrostatic strain effects
            for j in range(4):

                l = cosine_arr[j,0]
                m = cosine_arr[j,1]
                n = cosine_arr[j,2]
                u = (l**2 - m**2)/2.
                v = (3*n**2 -1.)/(2*np.sqrt(3.))

                # p orbitals
                beta_p_j = beta_p0 + beta_p1*(d_arr[j] - d_Si_o)/d_Si_o
                Onsite_mtx[2,2] += beta_p_j*(l**2 - (1./3.))
                Onsite_mtx[3,3] += beta_p_j*(m**2 - (1./3.))
                Onsite_mtx[4,4] += beta_p_j*(n**2 - (1./3.))
                Onsite_mtx[2,3] += beta_p_j*(l*m)
                Onsite_mtx[3,2] += beta_p_j*(l*m)
                #Onsite_mtx[2,4] += beta_p_j*(l*n)
                #Onsite_mtx[4,2] += beta_p_j*(l*n)
                #Onsite_mtx[3,4] += beta_p_j*(m*n)
                #Onsite_mtx[4,3] += beta_p_j*(m*n)

                # d orbitals
                Onsite_mtx[5,5] += beta_d0*(l**2 - (1./3.))
                Onsite_mtx[6,6] += beta_d0*(m**2 - (1./3.))
                Onsite_mtx[7,7] += beta_d0*(n**2 - (1./3.))
                Onsite_mtx[8,8] += beta_d0*(n**2 - (1./3.))
                Onsite_mtx[9,9] += beta_d0*(-n**2 + (1./3.))
                Onsite_mtx[5,6] += beta_d0*(-l*m)
                Onsite_mtx[6,5] += beta_d0*(-l*m)
                Onsite_mtx[7,9] += beta_d0*(2*l*m/np.sqrt(3.))
                Onsite_mtx[9,7] += beta_d0*(2*l*m/np.sqrt(3.))

                # s-p coupling
                Onsite_mtx[0,4] += beta_sp0*n
                Onsite_mtx[4,0] += beta_sp0*n
                Onsite_mtx[1,4] += beta_s2p0*n
                Onsite_mtx[4,1] += beta_s2p0*n

                # s-d coupling
                Onsite_mtx[0,7] += beta_sd0*(l*m)
                Onsite_mtx[7,0] += beta_sd0*(l*m)
                Onsite_mtx[0,9] += beta_sd0*(v)
                Onsite_mtx[9,0] += beta_sd0*(v)
                Onsite_mtx[1,7] += beta_s2d0*(l*m)
                Onsite_mtx[7,1] += beta_s2d0*(l*m)
                Onsite_mtx[1,9] += beta_s2d0*(v)
                Onsite_mtx[9,1] += beta_s2d0*(v)

                # p-d coupling
                beta_pd_j = beta_pd0 + beta_pd1*(d_arr[j] - d_Si_o)/d_Si_o
                Onsite_mtx[2,6] += beta_pd_j*n
                Onsite_mtx[6,2] += beta_pd_j*n
                Onsite_mtx[3,5] += beta_pd_j*n
                Onsite_mtx[5,3] += beta_pd_j*n
                Onsite_mtx[4,9] += beta_pd_j*(2*n/np.sqrt(3.))
                Onsite_mtx[9,4] += beta_pd_j*(2*n/np.sqrt(3.))

            return Onsite_mtx

        def Onsite_mtx_spinless_Ge(n,a_par,a_perp_nnM1,a_perp_nP1n,epsilon_xy,zeta,E_offset = 0.68):
            ### Calculates the onsite spinless Hamiltonian of a
            ### Ge atom in the nth layer
            ###     * We can specify the E_offset of the valence band of Ge with respect to the valence band
            ###       of Si. The "standard" value used in this function of E_offset = 0.68 eV is from
            ###       PHYSICAL REVIEW B 79, 245201 (2009)
            ###     * a_par is the in-plane lattice constant set by the barrier region
            ###     * a_perp_nnM1 is the z-direction lattice constant between the n and n-1 layer
            ###     * a_perp_nP1n is the z-direction lattice constant between the n+1 and n layer
            ###     * epsilon_xy = ε_{xy} shear strain
            ###     * The orbital ordering is {s, s*, px, py, pz, dyz, dxz, dxy, dx2my2, dz2}

            ### onsite strain parameters (in eV)
            alpha_s = -0.33252
            alpha_p = -0.43824
            alpha_d = -0.90486
            alpha_s2 = -0.52062
            beta_p0 = 1.01233
            beta_p1 = -2.53951
            beta_pd0 = -0.22597
            beta_pd1 = -3.77180
            beta_d0 = 1.99217
            beta_sp0 = 1.27627
            beta_s2p0 = -2.02374
            beta_sd0 = 2.38822
            beta_s2d0 = -4.73191

            ### Unstrained orbital energies
            #E_offset = 0.68
            Es = -4.08253 + E_offset
            Ep = 4.63470 + E_offset
            Ed = 12.19526 + E_offset
            Es2 = 23.20167 + E_offset

            ### nearest neighbor, bond distance, and cosine arrays
            #a_Ge = 5.658 # lattice constant of unstrained Ge
            r_arr = np.zeros((4,3))
            if n % 2 == 0:
                r1_x = (a_par/4.)*(1 + epsilon_xy)
                r1_y = (a_par/4.)*(1 + epsilon_xy)
                r1_z = (a_perp_nP1n/4.)*(1 - 2*zeta*epsilon_xy)
                r_arr[0,:] = np.array([r1_x,r1_y,r1_z])
                r_arr[1,:] = np.array([-r1_x,-r1_y,r1_z])
                r3_x = -(a_par/4.)*(1 - epsilon_xy)
                r3_y = (a_par/4.)*(1 - epsilon_xy)
                r3_z = -(a_perp_nnM1/4.)*(1 + 2*zeta*epsilon_xy)
                r_arr[2,:] = np.array([r3_x,r3_y,r3_z])
                r_arr[3,:] = np.array([-r3_x,-r3_y,r3_z])
            elif n % 2 == 1:
                r1_x = (a_par/4.)*(1 - epsilon_xy)
                r1_y = -(a_par/4.)*(1 - epsilon_xy)
                r1_z = (a_perp_nP1n/4.)*(1 + 2*zeta*epsilon_xy)
                r_arr[0,:] = np.array([r1_x,r1_y,r1_z])
                r_arr[1,:] = np.array([-r1_x,-r1_y,r1_z])
                r3_x = -(a_par/4.)*(1 + epsilon_xy)
                r3_y = -(a_par/4.)*(1 + epsilon_xy)
                r3_z = -(a_perp_nnM1/4.)*(1 - 2*zeta*epsilon_xy)
                r_arr[2,:] = np.array([r3_x,r3_y,r3_z])
                r_arr[3,:] = np.array([-r3_x,-r3_y,r3_z])
            d_arr = np.zeros(4) # bond distance array
            for j in range(4):
                d_arr[j] = np.linalg.norm(r_arr[j,:])
            cosine_arr = np.zeros((4,3)) # cosine array
            for j in range(4):
                for b in range(3):
                    cosine_arr[j,b] = r_arr[j,b]/d_arr[j]

            ### Calculate the hydrostatic strain
            d_Ge_o = 5.658 * np.sqrt(3.)/4. # unstrained bond distance
            hydro_strain = 0
            for j in range(4):
                hydro_strain += .75*(d_arr[j] - d_Ge_o)/d_Ge_o
            #print("hydro_strain = ", hydro_strain)

            ### Set the initial onsite energies with hydrostatic strain included
            Onsite_mtx = np.zeros((10,10))
            Onsite_mtx[0,0] = Es + alpha_s*hydro_strain
            Onsite_mtx[1,1] = Es2 + alpha_s2*hydro_strain
            Onsite_mtx[2,2] = Ep + alpha_p*hydro_strain
            Onsite_mtx[3,3] = Ep + alpha_p*hydro_strain
            Onsite_mtx[4,4] = Ep + alpha_p*hydro_strain
            Onsite_mtx[5,5] = Ed + alpha_d*hydro_strain
            Onsite_mtx[6,6] = Ed + alpha_d*hydro_strain
            Onsite_mtx[7,7] = Ed + alpha_d*hydro_strain
            Onsite_mtx[8,8] = Ed + alpha_d*hydro_strain
            Onsite_mtx[9,9] = Ed + alpha_d*hydro_strain

            #return Onsite_mtx

            ### Include the non-hyrdrostatic strain effects
            for j in range(4):

                l = cosine_arr[j,0]
                m = cosine_arr[j,1]
                n = cosine_arr[j,2]
                u = (l**2 - m**2)/2.
                v = (3*n**2 -1.)/(2*np.sqrt(3.))

                # p orbitals
                beta_p_j = beta_p0 + beta_p1*(d_arr[j] - d_Ge_o)/d_Ge_o
                Onsite_mtx[2,2] += beta_p_j*(l**2 - (1./3.))
                Onsite_mtx[3,3] += beta_p_j*(m**2 - (1./3.))
                Onsite_mtx[4,4] += beta_p_j*(n**2 - (1./3.))
                Onsite_mtx[2,3] += beta_p_j*(l*m)
                Onsite_mtx[3,2] += beta_p_j*(l*m)
                Onsite_mtx[2,4] += beta_p_j*(l*n)
                Onsite_mtx[4,2] += beta_p_j*(l*n)
                Onsite_mtx[3,4] += beta_p_j*(m*n)
                Onsite_mtx[4,3] += beta_p_j*(m*n)

                # d orbitals
                Onsite_mtx[5,5] += beta_d0*(l**2 - (1./3.))
                Onsite_mtx[6,6] += beta_d0*(m**2 - (1./3.))
                Onsite_mtx[7,7] += beta_d0*(n**2 - (1./3.))
                Onsite_mtx[8,8] += beta_d0*(n**2 - (1./3.))
                Onsite_mtx[9,9] += beta_d0*(-n**2 + (1./3.))
                Onsite_mtx[5,6] += beta_d0*(-l*m)
                Onsite_mtx[6,5] += beta_d0*(-l*m)
                Onsite_mtx[7,9] += beta_d0*(2*l*m/np.sqrt(3.))
                Onsite_mtx[9,7] += beta_d0*(2*l*m/np.sqrt(3.))

                # s-p coupling
                Onsite_mtx[0,4] += beta_sp0*n
                Onsite_mtx[4,0] += beta_sp0*n
                Onsite_mtx[1,4] += beta_s2p0*n
                Onsite_mtx[4,1] += beta_s2p0*n

                # s-d coupling
                Onsite_mtx[0,7] += beta_sd0*(l*m)
                Onsite_mtx[7,0] += beta_sd0*(l*m)
                Onsite_mtx[0,9] += beta_sd0*(v)
                Onsite_mtx[9,0] += beta_sd0*(v)
                Onsite_mtx[1,7] += beta_s2d0*(l*m)
                Onsite_mtx[7,1] += beta_s2d0*(l*m)
                Onsite_mtx[1,9] += beta_s2d0*(v)
                Onsite_mtx[9,1] += beta_s2d0*(v)

                # p-d coupling
                beta_pd_j = beta_pd0 + beta_pd1*(d_arr[j] - d_Ge_o)/d_Ge_o
                Onsite_mtx[2,6] += beta_pd_j*n
                Onsite_mtx[6,2] += beta_pd_j*n
                Onsite_mtx[3,5] += beta_pd_j*n
                Onsite_mtx[5,3] += beta_pd_j*n
                Onsite_mtx[4,9] += beta_pd_j*(2*n/np.sqrt(3.))
                Onsite_mtx[9,4] += beta_pd_j*(2*n/np.sqrt(3.))

            return Onsite_mtx

        def Onsite_mtx_VC_spinless_gen(xGe,a_par,a_perp_nnM1,a_perp_nP1n,epsilon_xy,zeta,n,E_offset = 0.68):
            ### Calculates the onsite Hamiltonian of a virtual cyrstal
            ### atom in the nth layer with Ge concentration xGe
            ###     * We can specify the E_offset of the valence band of Ge with respect to the valence band
            ###       of Si. The "standard" value used in this function of E_offset = 0.68 eV is from
            ###       PHYSICAL REVIEW B 79, 245201 (2009)
            ###     * a_par is the in-plane lattice constant set by the barrier region
            ###     * a_perp_nnM1 is the z-direction lattice constant between the n and n-1 layer
            ###     * a_perp_nP1n is the z-direction lattice constant between the n+1 and n layer
            ###     * The orbital ordering is {s, s*, px, py, pz, dyz, dxz, dxy, dx2my2, dz2}

            Onsite_mtx = np.zeros((10,10),dtype = 'complex')

            ### Generate the spinless Si and Ge Hamiltonian
            Onsite_mtx_Si_spinless = Onsite_mtx_spinless_Si(n,a_par,a_perp_nnM1,a_perp_nP1n,epsilon_xy,zeta)
            Onsite_mtx_Ge_spinless = Onsite_mtx_spinless_Ge(n,a_par,a_perp_nnM1,a_perp_nP1n,epsilon_xy,zeta,E_offset = E_offset)

            Onsite_mtx[:10,:10] = (1 - xGe)*Onsite_mtx_Si_spinless + xGe*Onsite_mtx_Ge_spinless
            return Onsite_mtx

    ### Germanium concentration profile functions
    if True:

        def Hyberbolic_barrier_Wiggle_Well_Ge_profile(N_bar,N_well,N_interface,N_wiggles,xGe_bar,xGe_well_max,n_offset = 0,PLOT = False):
            ### Generates a Wiggle Well with N_wiggles atomic sites in a period and a barrier that is hyberbolic with a sharpness characterized
            ### by N_interface sites

            def hyperbolic_heaviside(x,Lambda):
                return .5*(1 + np.tanh(x/Lambda))

            ### Define the lattice site of the middle of hyberbolic heaviside functions
            n_o = N_bar + n_offset

            ### Loop through the sites
            N_sites = N_bar + N_well
            Ge_arr = np.zeros(N_sites)
            for n in range(N_sites):
                h_n = hyperbolic_heaviside(n-n_o,N_interface)
                Ge_arr[n] = xGe_bar*(1 - h_n) \
                            + h_n*.5*xGe_well_max*(1 - np.cos(2*np.pi*(n - n_o - 2*N_interface)/N_wiggles) )
            return Ge_arr

        def Soft_barrier_Wiggle_Well_Ge_profile(N_bar,N_well,N_interface,N_wiggles,xGe_bar,xGe_well_max,n_offset = 0,PLOT = False):
            ### Generates a Wiggle Well with N_wiggles atomic sites in a period and a barrier has a thickness of N_interface sites

            ### Define the lattice sites where the barrier and wiggles begin
            n_o = N_bar + n_offset
            n_1 = n_o + N_interface

            ### Loop through the sites
            N_sites = int(N_bar+N_interface+N_well)
            Ge_arr = np.zeros(N_sites)
            n_arr_cont = np.linspace(0.,N_sites,5001)
            Ge_arr_cont = np.zeros(n_arr_cont.size)
            for n in range(N_sites):
                if n < n_o:
                    Ge_arr[n] = xGe_bar
                elif n < n_1:
                    Ge_arr[n] = (xGe_bar/2)*(1 + np.cos(np.pi*(n-n_o)/N_interface))
                else:
                    Ge_arr[n] = (xGe_well_max/2)*(1 - np.cos(2*np.pi*(n-n_1)/N_wiggles))
            for m in range(n_arr_cont.size):
                n = n_arr_cont[m]
                if n < n_o:
                    Ge_arr_cont[m] = xGe_bar
                elif n < n_1:
                    Ge_arr_cont[m] = (xGe_bar/2)*(1 + np.cos(np.pi*(n-n_o)/N_interface))
                else:
                    Ge_arr_cont[m] = (xGe_well_max/2)*(1 - np.cos(2*np.pi*(n-n_1)/N_wiggles))

            if PLOT == True:
                fig = plt.figure(); ax1 = fig.add_subplot(1,1,1)
                ax1.plot(n_arr_cont,Ge_arr_cont,c = 'k')
                ax1.scatter(np.arange(Ge_arr.size),Ge_arr,s = 3,c = 'r',zorder = 50)
                ax1.set_xlabel("layer index")
                ax1.set_ylabel(r"$n_{Ge}$")
                plt.show()

            return Ge_arr, n_arr_cont, Ge_arr_cont

        def Soft_barrier_Wiggle_Well_Ge_profile2(N_bar,N_well,N_interface,N_wiggles,xGe_bar,xGe_well_max,z0 = 0.,PLOT = False):
        ### Generates a Wiggle Well with N_wiggles atomic sites in a period and a barrier has a thickness of N_interface sites
        ###     * The interface runs from (-w+z0) to z0, where w = N_interface*aSi/4.

            def xGe_calc(z,w,lambda_wiggle,z0):
                ### Calculate the Ge concentration at z for an interface width of w
                ###     * The interface runs from (-w+z0) to z0
                z1 = -w + z0 # position where xGe goes to xGe_bar
                if z < z1:
                    return xGe_bar
                elif z < z0:
                    return (xGe_bar/2)*( 1 + np.cos(np.pi*(z-z1)/w) )
                else:
                    return (xGe_well_max/2)*(1 - np.cos(2*np.pi*(z-z0)/lambda_wiggle))

            az = 5.431/4.
            int_width = N_interface*az # size of the interface width
            lambda_Ge_oscilations = N_wiggles*az # lambda of the Ge oscillations
            N_sites = int(N_bar+N_well)
            z_arr = np.arange(N_sites)*az - N_bar*az
            z_arr_cont = np.linspace(z_arr[0],z_arr[-1],5001)

            ### Loop through the sites
            Ge_arr = np.zeros(N_sites)
            Ge_arr_cont = np.zeros(z_arr_cont.size)
            for n in range(N_sites):
                Ge_arr[n] = xGe_calc(z_arr[n],int_width,lambda_Ge_oscilations,z0)
            for m in range(Ge_arr_cont.size):
                Ge_arr_cont[m] = xGe_calc(z_arr_cont[m],int_width,lambda_Ge_oscilations,z0)


            if PLOT == True:
                fig = plt.figure(); ax1 = fig.add_subplot(1,1,1)
                ax1.plot(z_arr_cont/10.,Ge_arr_cont,c = 'k')
                ax1.scatter(z_arr/10.,Ge_arr,s = 3,c = 'r',zorder = 50)
                ax1.set_xlabel("z (nm)")
                ax1.set_ylabel(r"$n_{Ge}$")
                plt.subplots_adjust(left = 0.17, bottom=0.2, right=0.8, top=0.96)
                plt.show()

            return Ge_arr, z_arr_cont, Ge_arr_cont

        def Sigmoid_barrier_Ge_profile(N_bar,N_well,N_interface,xGe_bar,z0 = 0.,PLOT = False,symmetric = False):
            ### Generates a uniform Si well with a sigmoid barrier has a thickness of N_interface sites
            ###     * If symmetric == True, then we create a barrier on the right side of the system as well, where
            ###       the Ge profile about the middle of the quantum well is assumed to be symmetric

            def xGe_calc(z,tau,z0):
                ### Calculate the Ge concentration at z for an interface characterized by tau
                ###     * tau = interface width / 4
                zt = 1*z0 # middle position of the interface
                return xGe_bar/(1 + np.exp((z - zt)/tau))

            az = 5.431/4.
            int_width = N_interface*az # size of the interface width
            tau = int_width/4.
            if symmetric == False:
                N_sites = int(N_bar+N_well)
            elif symmetric == True:
                N_sites = int(2*N_bar+N_well)
            z_arr = np.arange(N_sites)*az - N_bar*az
            z_arr_cont = np.linspace(z_arr[0],z_arr[-1],5001)
            z_ave = np.average(z_arr)

            ### Loop through the sites
            Ge_arr = np.zeros(N_sites)
            Ge_arr_cont = np.zeros(z_arr_cont.size)
            if symmetric == False:
                for n in range(N_sites):
                    Ge_arr[n] = xGe_calc(z_arr[n],tau,z0)
                for m in range(Ge_arr_cont.size):
                    Ge_arr_cont[m] = xGe_calc(z_arr_cont[m],tau,z0)
            elif symmetric == True:
                for n in range(N_sites):
                    if z_arr[n] <= z_ave:
                        Ge_arr[n] = xGe_calc(z_arr[n],tau,z0)
                    else:
                        zn = z_ave - (z_arr[n] - z_ave)
                        Ge_arr[n] = xGe_calc(zn,tau,z0)


                for m in range(Ge_arr_cont.size):
                    if z_arr_cont[m] <= z_ave:
                        Ge_arr_cont[m] = xGe_calc(z_arr_cont[m],tau,z0)
                    else:
                        zm = z_ave - (z_arr_cont[m] - z_ave)
                        Ge_arr_cont[m] = xGe_calc(zm,tau,z0)

            if PLOT == True:
                fig = plt.figure(); ax1 = fig.add_subplot(1,1,1)
                ax1.plot(z_arr_cont/10.,Ge_arr_cont,c = 'k')
                ax1.scatter(z_arr/10.,Ge_arr,s = 3,c = 'r',zorder = 50)
                ax1.set_xlabel("z (nm)")
                ax1.set_ylabel(r"$n_{Ge}$")
                ax1.axvline(x = .1*(z0 + int_width/2),c = 'k',ls = 'dashed' )
                ax1.axvline(x = .1*(z0 - int_width/2),c = 'k',ls = 'dashed' )
                plt.subplots_adjust(left = 0.17, bottom=0.2, right=0.8, top=0.96)
                plt.show()

            return Ge_arr, z_arr_cont, Ge_arr_cont

        def Sigmoid_barrier_WiggleWell_Ge_profile(N_bar,N_well,N_interface,N_wiggles,xGe_bar,xGe_well_max,z0 = 0.,PLOT = False,symmetric = False):
            ### Generates a Wiggle Well with N_wiggles atomic sites in a period a sigmoid barrier has a thickness of N_interface sites
            ###     * If symmetric == True, then we create a barrier on the right side of the system as well, where
            ###       the Ge profile about the middle of the quantum well is assumed to be symmetric

            def xGe_calc(z,tau,z0,lambda_wiggle):
                ### Calculate the Ge concentration at z for an interface characterized by tau
                ###     * tau = interface width / 4
                zt = 1*z0 # middle position of the interface
                barrier_term = xGe_bar/(1 + np.exp((z - zt)/tau))
                WiggleWell_term = (xGe_well_max/2)*(1 - np.cos(2*np.pi*(z-z0)/lambda_wiggle))
                return barrier_term + WiggleWell_term

            az = 5.431/4.
            int_width = N_interface*az # size of the interface width
            tau = int_width/4.
            lambda_Ge_oscilations = N_wiggles*az # lambda of the Ge oscillations
            if symmetric == False:
                N_sites = int(N_bar+N_well)
            elif symmetric == True:
                N_sites = int(2*N_bar+N_well)
            z_arr = np.arange(N_sites)*az - N_bar*az
            z_arr_cont = np.linspace(z_arr[0],z_arr[-1],5001)
            z_ave = np.average(z_arr)

            ### Loop through the sites
            Ge_arr = np.zeros(N_sites)
            Ge_arr_cont = np.zeros(z_arr_cont.size)
            if symmetric == False:
                for n in range(N_sites):
                    Ge_arr[n] = xGe_calc(z_arr[n],tau,z0,lambda_Ge_oscilations)
                for m in range(Ge_arr_cont.size):
                    Ge_arr_cont[m] = xGe_calc(z_arr_cont[m],tau,z0,lambda_Ge_oscilations)
            elif symmetric == True:
                for n in range(N_sites):
                    if z_arr[n] <= z_ave:
                        Ge_arr[n] = xGe_calc(z_arr[n],tau,z0,lambda_Ge_oscilations)
                    else:
                        zn = z_ave - (z_arr[n] - z_ave)
                        Ge_arr[n] = xGe_calc(zn,tau,z0,lambda_Ge_oscilations)


                for m in range(Ge_arr_cont.size):
                    if z_arr_cont[m] <= z_ave:
                        Ge_arr_cont[m] = xGe_calc(z_arr_cont[m],tau,z0,lambda_Ge_oscilations)
                    else:
                        zm = z_ave - (z_arr_cont[m] - z_ave)
                        Ge_arr_cont[m] = xGe_calc(zm,tau,z0,lambda_Ge_oscilations)

            if PLOT == True:
                fig = plt.figure(); ax1 = fig.add_subplot(1,1,1)
                ax1.plot(z_arr_cont/10.,Ge_arr_cont,c = 'k')
                ax1.scatter(z_arr/10.,Ge_arr,s = 3,c = 'r',zorder = 50)
                ax1.set_xlabel("z (nm)")
                ax1.set_ylabel(r"$n_{Ge}$")
                ax1.axvline(x = .1*(z0 + int_width/2),c = 'k',ls = 'dashed' )
                ax1.axvline(x = .1*(z0 - int_width/2),c = 'k',ls = 'dashed' )
                plt.subplots_adjust(left = 0.17, bottom=0.2, right=0.8, top=0.96)
                plt.show()

            return Ge_arr, z_arr_cont, Ge_arr_cont

class SiGe_Quantum_Well_sp3d5s2:

    def __init__(self,x_Ge_sub):

        self.x_Ge_sub = x_Ge_sub                         # Germanium concentration of the substrate
                                                         # that sets the biaxial strain parameters
        self.a_sub = calc_relaxed_lat_constant(x_Ge_sub) # relaxed lattice constant of the substrate
        self.zeta = 0.53                                 # Kleinman's internal strain parameter

    def gen_SiGe_zero_momentum_matrices(self,xGe,E_offset = 0.68,print_info=False):
        ### Generates the hopping and onsite matrices for layers with Ge concentration xGe for kx = ky = 0
        ###    * The onsite matrices are denotes by h_o and h_1
        ###    * The hopping matrices are T_o, which is the hopping matrix in the absence of strain
        ###      and T_ep, which is the correct due to the shear strain
        ###         * The hopping matrix from layer n to n+1 is T_o + epsilon_xy*(-1)^n*T_ep, where
        ###           corrections of order (epsilon_xy)^2 are neglected
        ###    * We change the basis ordering to [s,s2,pz,xy,z2mr2,px,py,zx,yz,x2my2]
        ###      and the sign is flipped on xy, py, and yz orbitals on every odd site such that
        ###      the hopping matrix in the absence of strain is site independent.
        ###    * We also separate out the [s,s2,pz,xy,z2mr2] and [px,py,zx,yz] sectors since they
        ###      are uncoupled at kx = ky = 0
        ###    * If print_info == True, the function prints out the various matrices
        ###    * We can specify the E_offset of the valence band of Ge with respect to the valence band
        ###      of Si. The "standard" value used in this function of E_offset = 0.68 eV is from
        ###      PHYSICAL REVIEW B 79, 245201 (2009)

        ### Calculate the growth (z) direction lattice constant
        a_par = 1*self.a_sub # substrate sets the in-plane lattice constant
        a_perp = calc_m_perp_lat_constant(a_par,xGe)

        idx_arr = np.array([0,1,4,7,9,2,3,6,5,8]) # index array to change the orbital ordering
                                                  # to [s,s2,pz,xy,z2mr2,px,py,zx,yz,x2my2]

        ### Calculate the hopping and onsite matrices for kx = ky = 0 in
        ### the absence of shear strain
        kx = ky = 0
        epsilon_xy = 0
        T_0 = Hopping_mtx_VC_spinless_gen(kx,ky,xGe,xGe,a_par,a_perp,epsilon_xy,self.zeta,0)
        T_0 = T_0[idx_arr,:][:,idx_arr]
        T_1 = Hopping_mtx_VC_spinless_gen(kx,ky,xGe,xGe,a_par,a_perp,epsilon_xy,self.zeta,1)
        T_1 = T_1[idx_arr,:][:,idx_arr]
        h_0 = Onsite_mtx_VC_spinless_gen(xGe,a_par,a_perp,a_perp,epsilon_xy,self.zeta,0,E_offset = E_offset)
        h_0 = h_0[idx_arr,:][:,idx_arr]
        h_1 = Onsite_mtx_VC_spinless_gen(xGe,a_par,a_perp,a_perp,epsilon_xy,self.zeta,1,E_offset = E_offset)
        h_1 = h_1[idx_arr,:][:,idx_arr]

        ### Calculate the hopping matrix for kx = ky = 0 in
        ### the presence of a small shear strain
        epsilon_xy = 1.e-4
        T_0_strained = Hopping_mtx_VC_spinless_gen(kx,ky,xGe,xGe,a_par,a_perp,epsilon_xy,self.zeta,0)
        T_0_strained = T_0_strained[idx_arr,:][:,idx_arr]
        T_1_strained = Hopping_mtx_VC_spinless_gen(kx,ky,xGe,xGe,a_par,a_perp,epsilon_xy,self.zeta,1)
        T_1_strained = T_1_strained[idx_arr,:][:,idx_arr]
        h_0_strained = Onsite_mtx_VC_spinless_gen(xGe,a_par,a_perp,a_perp,epsilon_xy,self.zeta,0,E_offset = E_offset)
        h_0_strained = h_0_strained[idx_arr,:][:,idx_arr]
        h_1_strained = Onsite_mtx_VC_spinless_gen(xGe,a_par,a_perp,a_perp,epsilon_xy,self.zeta,1,E_offset = E_offset)
        h_1_strained = h_1_strained[idx_arr,:][:,idx_arr]

        ### Perform a basis transformation by adding a minus sign to the xy, py, and yz on
        ### every odd site
        U_1 = np.diag(np.array([1,1,1,-1,1,1,-1,1,-1,1]))
        T_0 = np.dot(U_1,T_0)
        T_1 = np.dot(T_1,U_1)
        T_0_strained = np.dot(U_1,T_0_strained)
        T_1_strained = np.dot(T_1_strained,U_1)
        h_1 = np.dot(U_1,np.dot(h_1,U_1))
        h_1_strained = np.dot(U_1,np.dot(h_1_strained,U_1))

        ### Subtract off the unstrained hopping matrices from the strained hopping matrices and
        ### divide off the strain to arrive at the T_epsilon and h_epsilon matrices
        T_ep_0 = (T_0_strained - T_0)/epsilon_xy
        T_ep_1 = (T_1_strained - T_1)/epsilon_xy
        h_ep_0 = (h_0_strained - h_0)/epsilon_xy
        h_ep_1 = (h_1_strained - h_1)/epsilon_xy

        ### hopping components for the orbital set {s,s2,pz,xy,z2mr2}
        self.T_o_pz = T_0[:5,:5].real
        self.T_ep_pz = T_ep_0[:5,:5].real
        self.h_o_pz = h_0[:5,:5].real
        self.h_ep_pz = h_ep_0[:5,:5].real

        ### hopping components for the orbital set {px,py,zx,yz,x2my2}
        self.T_o_px = T_0[5:,5:].real
        self.T_ep_px = T_ep_0[5:,5:].real
        self.h_o_px = h_0[5:,5:].real
        self.h_ep_px = h_ep_0[5:,5:].real

        if print_info == True:

            ### Printing results
            #print()
            print("Unstrained hopping matrices")
            print("T_0");
            print(np.around(T_0.real,decimals = 2)); print()
            print("T_1");
            print(np.around(T_1.real,decimals = 2)); print()
            print("Strained hopping matrices")
            print("T_0_strained");
            print(np.around(T_0_strained.real,decimals = 4)); print()
            print("T_1_strained");
            print(np.around(T_1_strained.real,decimals = 4)); print()

            print("Hopping matrix components");
            print("T_o");
            print(np.around(T_0.real,decimals = 2)); print()
            print("T_ep");
            print(np.around(T_ep_0.real,decimals = 2)); print()

            print("Onsite matrix components"); print()
            print("h_0");
            print(np.around(h_0.real,decimals = 2)); print()
            print("h_1");
            print(np.around(h_1.real,decimals = 2)); print()
            print("h_0_strained");
            print(np.around(h_0_strained.real,decimals = 2)); print()
            print("h_1_strained");
            print(np.around(h_1_strained.real,decimals = 2)); print()
            print("h_1_strained - h_0_strained");
            print(np.around(h_1_strained.real - h_0_strained.real,decimals = 3)); print()
            print("h_ep_0");
            print(np.around(h_ep_0.real,decimals = 2)); print()
            print("h_ep_1");
            print(np.around(h_ep_1.real,decimals = 2)); print()
            print("h_ep_1 + h_ep_0");
            print(np.around(h_ep_1.real + h_ep_0.real,decimals = 2)); print()
            print("h_o");
            print(np.around(h_0.real,decimals = 2)); print()
            print("h_ep");
            print(np.around(h_ep_0.real,decimals = 2)); print()

        return h_0_strained, h_1_strained, T_0_strained, T_1_strained

    def gen_SiGe_onsite_and_hopping_matrices(self,xGe,kx,ky,epsilon_xy,E_offset =  0.68):
        ### Generates the hopping and onsite matrices of the system for given Ge concentration
        ### xGe, in-plane momenta kx and ky, and shear strain epsilon_xy
        ###    * The onsite matrices are denotes by h_o and h_1
        ###    * The hopping matrices are T_0 and T_1
        ###    * We change the basis ordering to [s,s2,pz,xy,z2mr2,px,py,zx,yz,x2my2]
        ###      and the sign is flipped on xy, py, and yz orbitals on every odd site such that
        ###      the hopping matrix in the absence of strain is site independent for kx = ky = 0.
        ###    * We can specify the E_offset of the valence band of Ge with respect to the valence band
        ###      of Si. The "standard" value used in this function of E_offset = 0.68 eV is from
        ###      PHYSICAL REVIEW B 79, 245201 (2009)

        ### Calculate the growth (z) direction lattice constant
        a_par = 1*self.a_sub # substrate sets the in-plane lattice constant
        a_perp = calc_m_perp_lat_constant(a_par,xGe)

        idx_arr = np.array([0,1,4,7,9,2,3,6,5,8]) # index array to change the orbital ordering
                                                  # to [s,s2,pz,xy,z2mr2,px,py,zx,yz,x2my2]

        T_0 = Hopping_mtx_VC_spinless_gen(kx,ky,xGe,xGe,a_par,a_perp,epsilon_xy,self.zeta,0)
        T_0 = T_0[idx_arr,:][:,idx_arr]
        T_1 = Hopping_mtx_VC_spinless_gen(kx,ky,xGe,xGe,a_par,a_perp,epsilon_xy,self.zeta,1)
        T_1 = T_1[idx_arr,:][:,idx_arr]
        h_0 = Onsite_mtx_VC_spinless_gen(xGe,a_par,a_perp,a_perp,epsilon_xy,self.zeta,0,E_offset = E_offset)
        h_0 = h_0[idx_arr,:][:,idx_arr]
        h_1 = Onsite_mtx_VC_spinless_gen(xGe,a_par,a_perp,a_perp,epsilon_xy,self.zeta,1,E_offset = E_offset)
        h_1 = h_1[idx_arr,:][:,idx_arr]

        ### Perform a basis transformation by adding a minus sign to the xy, py, and yz on
        ### every odd site
        U_1 = np.diag(np.array([1,1,1,-1,1,1,-1,1,-1,1]))
        T_0 = np.dot(U_1,T_0)
        T_1 = np.dot(T_1,U_1)
        h_1 = np.dot(U_1,np.dot(h_1,U_1))

        return h_0, h_1, T_0, T_1

    def calc_SiGe_zero_momentum_spectrum(self,xGe,kz_arr,epsilon_xy,E_offset =  0.68):
        ### Calculate the spectrum for kx = ky = 0 for a system with Ge concentration
        ### xGe within the virtual crystal approximation and shear strain epsilon_xy
        ###     * We can specify the E_offset of the valence band of Ge with respect to the valence band
        ###       of Si. The "standard" value used in this function of E_offset = 0.68 eV is from
        ###       PHYSICAL REVIEW B 79, 245201 (2009)

        ### Calculate the growth (z) direction lattice constant
        a_par = 1*self.a_sub # substrate sets the in-plane lattice constant
        a_perp = calc_m_perp_lat_constant(a_par,xGe)

        ### Generate the Hamiltonian components
        kx = 0; ky = 0
        h_0, h_1, T_0, T_1 = self.gen_SiGe_onsite_and_hopping_matrices(xGe,kx,ky,epsilon_xy,E_offset = E_offset)
        h_0_pz = h_0[:5,:5]
        h_1_pz = h_1[:5,:5]
        T_0_pz = T_0[:5,:5]
        T_1_pz = T_1[:5,:5]
        h_0_px = h_0[5:,5:]
        h_1_px = h_1[5:,5:]
        T_0_px = T_0[5:,5:]
        T_1_px = T_1[5:,5:]

        eig_arr_pz = np.zeros((kz_arr.size,10))
        eig_arr_px = np.zeros((kz_arr.size,10))

        for i in range(kz_arr.size):
            #print(kz_arr.size - i)
            H_pz = np.zeros((10,10),dtype = 'complex')
            #H_pz[:5,:5] = self.h_o_pz + epsilon_xy*self.h_ep_pz
            #H_pz[5:,5:] = self.h_o_pz - epsilon_xy*self.h_ep_pz
            #H_pz[5:,:5] = (self.T_o_pz + epsilon_xy*self.T_ep_pz)*np.exp(-1j*kz_arr[i]*a_Si/4) \
            #              + np.transpose(self.T_o_pz - epsilon_xy*self.T_ep_pz)*np.exp(1j*kz_arr[i]*a_Si/4)
            #H_pz[:5,5:] = np.transpose(np.conjugate(H_pz[5:,:5]))
            H_pz[:5,:5] = h_0_pz
            H_pz[5:,5:] = h_1_pz
            H_pz[5:,:5] = T_0_pz*np.exp(-1j*kz_arr[i]*a_perp/4) + np.transpose(T_1_pz)*np.exp(1j*kz_arr[i]*a_perp/4)
            H_pz[:5,5:] = np.transpose(np.conjugate(H_pz[5:,:5]))

            H_px = np.zeros((10,10),dtype = 'complex')
            #H_px[:5,:5] = self.h_o_px + epsilon_xy*self.h_ep_px
            #H_px[5:,5:] = self.h_o_px - epsilon_xy*self.h_ep_px
            #H_px[5:,:5] = (self.T_o_px + epsilon_xy*self.T_ep_px)*np.exp(-1j*kz_arr[i]*a_Si/4) \
            #              + np.transpose(self.T_o_px - epsilon_xy*self.T_ep_px)*np.exp(1j*kz_arr[i]*a_Si/4)
            #H_px[:5,5:] = np.transpose(np.conjugate(H_px[5:,:5]))
            H_px[:5,:5] = h_0_px
            H_px[5:,5:] = h_1_px
            H_px[5:,:5] = T_0_px*np.exp(-1j*kz_arr[i]*a_perp/4) + np.transpose(T_1_px)*np.exp(1j*kz_arr[i]*a_perp/4)
            H_px[:5,5:] = np.transpose(np.conjugate(H_px[5:,:5]))

            eig_arr_pz[i,:], U_pz = linalg.eigh(H_pz)
            eig_arr_px[i,:], U_px = linalg.eigh(H_px)

        return eig_arr_pz, eig_arr_px

    def calc_SiGe_zero_momentum_spectrum_unfolded(self,xGe,kz_arr,E_offset =  0.68):
        ### Calculate the unfolded spectrum for kx = ky = 0 for a system with Ge concentration
        ### xGe within the virtual crystal approximation
        ###     * We can specify the E_offset of the valence band of Ge with respect to the valence band
        ###       of Si. The "standard" value used in this function of E_offset = 0.68 eV is from
        ###       PHYSICAL REVIEW B 79, 245201 (2009)

        ### Calculate the growth (z) direction lattice constant
        a_par = 1*self.a_sub # substrate sets the in-plane lattice constant
        a_perp = calc_m_perp_lat_constant(a_par,xGe)

        ### Generate the Hamiltonian components
        kx = 0; ky = 0
        epsilon_xy = 0
        h_0, h_1, T_0, T_1 = self.gen_SiGe_onsite_and_hopping_matrices(xGe,kx,ky,epsilon_xy,E_offset = E_offset)
        h_0_pz = h_0[:5,:5]
        T_0_pz = T_0[:5,:5]
        h_0_px = h_0[5:,5:]
        T_0_px = T_0[5:,5:]

        eig_arr_pz = np.zeros((kz_arr.size,5))
        eig_arr_px = np.zeros((kz_arr.size,5))

        for i in range(kz_arr.size):
            #print(kz_arr.size - i)
            H_pz = np.zeros((5,5),dtype = 'complex')
            H_pz[:5,:5] = h_0_pz + T_0_pz*np.exp(-1j*kz_arr[i]*a_perp/4) + np.transpose(T_0_pz)*np.exp(1j*kz_arr[i]*a_perp/4)

            H_px = np.zeros((5,5),dtype = 'complex')
            H_px[:5,:5] = h_0_px + T_0_px*np.exp(-1j*kz_arr[i]*a_perp/4) + np.transpose(T_0_px)*np.exp(1j*kz_arr[i]*a_perp/4)

            eig_arr_pz[i,:], U_pz = linalg.eigh(H_pz)
            eig_arr_px[i,:], U_px = linalg.eigh(H_px)

        return eig_arr_pz, eig_arr_px

    def Ham_spinless_gen(self,kx,ky,Ez,Ge_arr,epsilon_xy):
        ### Generates the Hamiltonian for momemtnum (kx,ky) of a SiGe
        ### quantum well where the Ge content of each layer is given
        ### by Ge_arr and shear strain by epsilon_xy
        ###     * The input electric field Ez is given in mV/nm
        ###     * This function does not include the spin degree of freedom

        a_par = 1.*self.a_sub
        z_arr = z_arr_gen(a_par,Ge_arr) # z-coordinates of sites in Angstroms
        N_tot = z_arr.size
        Id = np.eye(10)

        row = []; col = []; data = []
        for n in range(N_tot):
            Vn = 1.e-4 * Ez * z_arr[n]  # potential of nth layer in eV

            ### determine the lattice constants from this layer n to layer n-1 and n+1
            if (n != 0) and (n != N_tot -1):
                a_perp_nnM1 = 4.*(z_arr[n] - z_arr[n-1])
                a_perp_nP1n = 4.*(z_arr[n+1] - z_arr[n])
            elif n == 0:
                a_perp_nP1n = 4.*(z_arr[n+1] - z_arr[n])
                a_perp_nnM1 = 1.*a_perp_nP1n
            elif n == (N_tot - 1):
                a_perp_nnM1 = 4.*(z_arr[n] - z_arr[n-1])
                a_perp_nP1n = 1.*a_perp_nnM1

            ### Onsite terms
            H_onsite = Onsite_mtx_VC_spinless_gen(Ge_arr[n],a_par,a_perp_nnM1,a_perp_nP1n,epsilon_xy,self.zeta,n)
            H_nn =  H_onsite[:,:] + Vn*Id
            for i in range(10):
                for j in range(10):
                    if abs(H_nn[i,j]) > 1.e-12:
                        row.append(n*10+i); col.append(n*10 + j); data.append(H_nn[i,j])

            ### hopping between layers
            if n != (N_tot-1):
                xGe_n = Ge_arr[n]
                xGe_np1 = Ge_arr[n+1]
                H_np1n = Hopping_mtx_VC_spinless_gen(kx,ky,xGe_n,xGe_np1,a_par,a_perp_nP1n,epsilon_xy,self.zeta,n)
                for i in range(10):
                    for j in range(10):
                        if abs(H_np1n[i,j]) > 1.e-12:
                            row.append((n+1)*10+i); col.append(n*10 + j); data.append(H_np1n[i,j])
                            col.append((n+1)*10+i); row.append(n*10 + j); data.append(np.conjugate(H_np1n[i,j]))
        Ham_sparse = Spar.csc_matrix((data,(row,col)), shape=(10*N_tot, 10*N_tot),dtype = 'complex')
        return Ham_sparse, z_arr

    def Ham_spinless_GeOnsiteEnergyModel_gen(self,kx,ky,Ez,Ge_arr,epsilon_xy,E_Ge):
        ### Generates the Hamiltonian for momemtnum (kx,ky) of a SiGe
        ### quantum well where the Ge content of each layer is given
        ### by Ge_arr and shear strain by epsilon_xy
        ###     * In this Ge onsite energy model, the Ge is treated
        ###       as simply an energy onsite energy compared to Si atoms
        ###     * The input electric field Ez is given in mV/nm
        ###     * This function does not include the spin degree of freedom

        a_par = 1.*self.a_sub
        a_perp = calc_m_perp_lat_constant(a_par,0)
        z_arr = (a_perp/4.)*np.arange(Ge_arr.size) # z-coordinates of sites in Angstroms
        N_tot = z_arr.size
        Id = np.eye(10)

        row = []; col = []; data = []
        for n in range(N_tot):
            Vn = 1.e-4 * Ez * z_arr[n] + E_Ge*Ge_arr[n]  # potential of nth layer in eV including a shift from the Germanium concentration

            ### determine the lattice constants from this layer n to layer n-1 and n+1
            if (n != 0) and (n != N_tot -1):
                a_perp_nnM1 = 4.*(z_arr[n] - z_arr[n-1])
                a_perp_nP1n = 4.*(z_arr[n+1] - z_arr[n])
            elif n == 0:
                a_perp_nP1n = 4.*(z_arr[n+1] - z_arr[n])
                a_perp_nnM1 = 1.*a_perp_nP1n
            elif n == (N_tot - 1):
                a_perp_nnM1 = 4.*(z_arr[n] - z_arr[n-1])
                a_perp_nP1n = 1.*a_perp_nnM1


            ### Onsite terms
            xGe_n = 0 # Ge content doesn't change anything in this model besides overall onsite energy shift
            H_onsite = Onsite_mtx_VC_spinless_gen(xGe_n,a_par,a_perp_nnM1,a_perp_nP1n,epsilon_xy,self.zeta,n)
            H_nn =  H_onsite[:,:] + Vn*Id
            for i in range(10):
                for j in range(10):
                    if abs(H_nn[i,j]) > 1.e-12:
                        row.append(n*10+i); col.append(n*10 + j); data.append(H_nn[i,j])

            ### hopping between layers
            if n != (N_tot-1):
                xGe_n = 0 # Ge content doesn't change anything in this model besides overall onsite energy shift
                xGe_np1 = 0
                H_np1n = Hopping_mtx_VC_spinless_gen(kx,ky,xGe_n,xGe_np1,a_par,a_perp_nP1n,epsilon_xy,self.zeta,n)
                for i in range(10):
                    for j in range(10):
                        if abs(H_np1n[i,j]) > 1.e-12:
                            row.append((n+1)*10+i); col.append(n*10 + j); data.append(H_np1n[i,j])
                            col.append((n+1)*10+i); row.append(n*10 + j); data.append(np.conjugate(H_np1n[i,j]))
        Ham_sparse = Spar.csc_matrix((data,(row,col)), shape=(10*N_tot, 10*N_tot),dtype = 'complex')
        return Ham_sparse, z_arr

    def plot_state(self,state,z_arr = -1,pz_sector = False):

        if pz_sector == False:
            N_orb = 10
        elif pz_sector == True:
            N_orb = 5
        A = np.square(np.absolute(state))
        A_proj = np.zeros(int(A.size/N_orb))
        for i in range(N_orb):
            A_proj[:] += A[i::N_orb]

        if (type(z_arr) == int):
            X = np.arange(1,A_proj.size +1)
            x_label = "layer index"
        else:
            X = z_arr[:]/10.
            x_label = r"$z$ (nm)"

        fig = plt.figure()
        width = 3.847; height = .7 * width
        fig.set_size_inches(width,height)
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(X,A_proj, c = 'b')
        ax1.scatter(X,A_proj, c = 'k',s = 5,zorder = 30)
        ax1.set_xlabel(x_label,fontsize = 12)
        ax1.set_ylabel(r"$|\psi|^{2}$",fontsize = 12)
        ax1.set_xlim(xmin = min(1.0*np.min(X),0),xmax = 1.0*np.max(X))
        ax1.set_yticklabels([])
        ax1.grid()
        ax1.axhline(y = 0,c = 'r',lw = .5)
        #ax1.set_ylim(ymin = min(-.05*np.max(Ge_arr_Full),-0.01))
        plt.subplots_adjust(left = 0.12, bottom=0.17, right=0.95, top=0.96)
        plt.show()

    def solve_Ham(self,Ham,num,sigma,which = 'LM',Return_vecs = False,reverse = False):
        ### Finding "num" eigenvalues near E = sigma
        eigs,vecs = SparLinalg.eigsh(Ham,k=num,sigma = sigma, which = which)
        idx = np.argsort(eigs)
        if reverse:
            idx = idx[::-1]
        if Return_vecs:
            return eigs[idx], vecs[:,idx]
        else:
            return eigs[idx]

    def project_state_onto_valley(self,state):
        ### Take only the Fourier components of the state that are positive,
        ### which is essentially projecting the state onto its +z valley component

        def momentum_basis_unitary_matrix_gen(N):
            ### Generates the unitary matrix U with transforms a given matrix
            ### from the real space pseudospin basis to momentum space pseudospin basis

            ### Generate the array of kz values
            kz_arr = np.zeros(N)
            if N % 2 == 0:
                for n in range(N):
                    kz_arr[n] = -np.pi + (2*np.pi*(n+1))/N
            elif N % 2 == 1:
                for n in range(N):
                    kz_arr[n] = -np.pi + np.pi/N + (2*np.pi*(n+1))/N

            N_orb = 10
            U = np.zeros((N_orb*N,N_orb*N),dtype = 'complex')
            for n in range(kz_arr.size): ### Loop through the kz values
                Arr = np.exp(1j*kz_arr[n]*np.arange(N))/np.sqrt(N)
                for alpha in range(N_orb): ### Loop through the orbitals
                    U[alpha::N_orb,N_orb*n + alpha] = Arr[:]
            return U, kz_arr

        ### Generate unitary to transform into momentum space
        N = int(state.size/10)
        U, kz_arr = momentum_basis_unitary_matrix_gen(N)
        U_hc = np.conjugate(np.transpose(U))

        ### Perform the projection
        state_momentum = np.dot(U_hc,state)
        state_momentum[:5*N] = 0. # remove -z valley components
        state_real_space_projected = np.dot(U,state_momentum)
        return state_real_space_projected


















