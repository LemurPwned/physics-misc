import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numba
import time
from numba.experimental import jitclass
MU = 4*np.pi*10e-7

precision = numba.float32
element_spec = [('x0', precision), ('y0', precision), ('x1', precision),
                ('y1', precision), ('w', precision), ('h', precision),
                ('area', precision), ('_centre', precision[:]),
                ('I', precision), ('I_dir', precision[:])]


@jitclass(element_spec)
class Element:
    def __init__(self, x, y, width, height, I=1.0):
        self.x0 = x
        self.y0 = y
        self.x1 = x + width
        self.y1 = y + height

        self.w = width
        self.h = height

        self.area = self.w * self.h

        self.I = I
        self.I_dir = np.array([0, 0, 1.], dtype=np.float32)
        self._centre = np.asarray([(self.x1 + self.x0) / 2,
                                   (self.y1 + self.y0) / 2],
                                  dtype=np.float32)

    def get_contribution_at_p(self, p) -> float:
        dist = np.sqrt(
            np.sum(np.asarray([(p[i] - self._centre[i])**2 for i in (0, 1)])))
        return self.I * self.area / dist

    def get_vector_contribution_at_p(self, p) -> np.ndarray:
        diff_vector = p - self._centre
        perp_vector = np.cross(diff_vector, self.I_dir)
        dist = np.sqrt(np.sum(np.power(diff_vector, 2)))
        mag = self.I * self.area / dist
        return mag * perp_vector / np.linalg.norm(perp_vector)

    def set_current(self, I):
        self.I = I


class Composite:
    w: float
    h: float
    l: float
    area: float
    elements: List[Element]
    resistance: float
    # offset: Tuple[float, float]
    Nx: int
    Ny: int

    def __init__(self, w, h, l, discretisation, offset, resistivity=1):
        """
        :param size:
            (W, L, H) aka (x, y, z)
        :param discretisation:
            sizes of tiny element
        """
        (self.w, self.h, self.l) = (w, h, l)
        self.area = self.w * self.h
        self.elements = mesh_discretisation(self.w,
                                            self.h,
                                            *discretisation,
                                            offset=offset)
        self.resistance = self.get_resistance(resistivity, self.l)
        # self.offset = offset
        (self.Nx, self.Ny) = discretisation

    def set_current(self, I):
        element: Element
        for element in self.elements:
            element.set_current(I)

    @staticmethod
    # @numba.jit(nopython=True, parallel=True)
    def calculate_contribs(elements, Xs, Ys):
        # vals = []
        assert (Xs.shape == Ys.shape)
        # for xp, yp in zip(Xs, Ys):
        res = np.zeros_like(Xs)
        for i in numba.prange(Xs.shape[0]):
            for j in numba.prange(Xs.shape[1]):
                p = (Xs[i, j], Ys[i, j])
                # points_contributions = 0.0
                el: Element
                for i_el in numba.prange(len(elements)):
                    res[i, j] += elements[i_el].get_contribution_at_p(
                        p)
                # vals.append(points_contributions)
        return res  # np.asarray(vals)

    @staticmethod
    # @numba.jit(nopython=True, parallel=True)
    def calculate_vector_contribs(elements, Xs, Ys):
        elen = len(elements)
        vals = np.zeros((*Xs.shape, 2), dtype=np.float32)
        for i in numba.prange(Xs.shape[0]):
            for j in numba.prange(Xs.shape[1]):
                for i_el in numba.prange(elen):
                    vals[i, j, :] += elements[i_el].get_vector_contribution_at_p(
                        np.asarray([Xs[i, j], Ys[i, j]]))
        return vals

    def get_resistance(self, resistivity, length):
        # assume all elements are of the same size
        return material_resistance(resistivity, self.area, self.l)

    def set_per_element_current(self, current):
        # the current is evenly distributed along the cross-sec.
        per_element_current = current
        self.set_current(per_element_current)
        return per_element_current


class CompositeStructure:
    def __init__(self, composites, power=10, current_density=0):
        self.composites: List[Composite] = composites
        # calculate the total resistance
        if current_density:
            composite: Composite
            for composite in composites:
                composite.set_per_element_current(current_density)
        else:
            R_tot = sum([composite.resistance for composite in composites])
            I_tot = np.sqrt(power / R_tot)
            for composite in composites:
                rel_I = I_tot * R_tot / composite.resistance
                composite.set_per_element_current(rel_I)

    @staticmethod
    @numba.jit(nopython=False, parallel=True)
    def calculate_contribs(composites, Xs, Ys):
        assert Xs.shape == Ys.shape
        total_contribs = np.zeros_like(Xs).astype(np.float32)
        for composite in composites:
            total_contribs += Composite.calculate_contribs(
                composite.elements, Xs, Ys)
        return total_contribs


def mesh_discretisation(swidth, sheight, rwidth, rheight,
                        offset=(0, 0, 0)) -> List[Element]:
    """
    :param offset:
        top left corner (for the first block, it's usually (0,0,0) 
        i.e. origin point)
    Length => Y
    Height => Z
    Width => X
    """
    Nx, Ny = int(swidth / rwidth), int(sheight / rheight)
    (xoff, yoff) = offset
    element_groups: List[Element] = []
    for ix in range(Nx):
        for iy in range(Ny):
            element_groups.append(
                Element(xoff + ix * rwidth, yoff + iy * rheight, rwidth,
                        rheight))
    return element_groups


def material_resistance(resistivity, area, length):
    return resistivity * length / area


def generate_vector_field():
    """
    Granica interfejsu 
    +- 0.5 nm od interfejsu 
    ~ 100 Oe
    j = 4.2e10 A/m2 
    """
    # 4nm , 20 mu dl szer, res 59 uOhm/cm
    sizeW = 20e-6  # nm
    sizeL = 20e-6  # nm
    sizeH = 4e-9  # nm
    current = 4.2e10  # density
    structure = [
        # ("Co", 62.4e-9),
        ("Pt", 10.6e-8)
        # ("Co", 62.4e-9),
    ]

    Nx = 1
    Ny = 3

    Dx = 1e-7
    Dy = 1e-9
    x = np.arange(-sizeW * Nx, (Nx + 1) * sizeW, Dx)
    y = np.arange(-sizeH * Ny, (Ny + 1) * sizeH, Dy)
    X, Y = np.meshgrid(x, y)
    print(X.shape, Y.shape)
    composites = []
    CC = np.zeros_like(X).astype(float)
    VC = np.zeros((*X.shape, 2), dtype=np.float32)
    for i, (k, r) in enumerate(structure):
        print(k, r)
        composite = Composite(*(sizeW, sizeH, sizeL),
                              discretisation=(Dx, Dy),
                              offset=(i * sizeW, 0),
                              resistivity=r)
        composite.set_current(current)
        # VC += Composite.calculate_vector_contribs(composite.elements, X, Y)
        composites.append(composite)

    start = time.time()
    struct = CompositeStructure(composites, current_density=current)
    CC = CompositeStructure.calculate_contribs(struct.composites, X, Y)
    end = time.time()
    print(f"Computation lasted: {end-start:.2f}")

    depth_start = 1e-9  # nm
    depth_stop = 5e-9  # nm
    depth_indices = np.argwhere((y >= depth_start) & (y <= depth_stop))
    width = 1000e-9
    width_index = np.argmin(abs(x - width)).ravel()[0]
    print(width_index, x[width_index])
    print(VC.shape, CC.shape, x.shape, y.shape)
    fig, ax = plt.subplots(figsize=(14, 14))
    toOersted = 79.5774715459
    print(CC[depth_indices, width_index].shape,
          CC[depth_indices, width_index].max())
    ax.plot(y[depth_indices], CC[depth_indices, width_index]/toOersted)
    ax.set_xlabel("Depth [nm]")
    ax.set_ylabel("Oersted field [Oe]")

    # _ = ax.pcolor(X, Y, CC)

    # im = ax.quiver(X,
    #                Y,
    #                VC[:, :, 0],
    #                VC[:, :, 1],
    #                units='width',
    #                pivot='tip')
    # # im = ax.quiver(X[::2, ::2],
    # #                Y[::2, ::2],
    # #                VC[::2, ::2, 0],
    # #                VC[::2, ::2, 1],
    # #                units='width',
    # #                pivot='tip')

    # im = ax.scatter()

    # # ax.scatter(X[::2, ::2], Y[::2, ::2], color='r', s=0.5, alpha=0.6)

    # depth_indices = np.argwhere((y >= depth_start) & (y  <= depth_stop))
    # y_depth = y[depth_indices]
    # for i in y_depth:
    #     ax.axhline(y=i, label='Vertical', color='r', alpha=0.5)

    # ax.axis('equal')
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.tick_params(axis='x', rotation=90)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # # _ = fig.colorbar(im, cax=cax, orientation='vertical')

    fig.savefig("Vector_field.png")

    for i, (name, _) in enumerate(structure):
        print(f"Layer {i}={name}")
        xoffset = i * sizeW
        yoffset = 0
        vv = get_vector_value_at_region(X, Y, VC, sizeW, sizeH, xoffset,
                                        yoffset, Dx, Dy)
        mag = np.linalg.norm(vv)
        print(f"\tMagnitude Hoe: {mag:.4f}")
        print(f"\tVector: {vv.tolist()}")


def get_vector_value_at_region(X, Y, VC, sizeW, sizeH, x0, y0, discretisation_x, discretisation_y):
    d_width = int(sizeW / discretisation_x)
    d_height = int(sizeH / discretisation_y)

    # find starting blocks
    x_indx = np.abs(X[0, :] - x0).argmin()
    y_indx = np.abs(Y[0, :] - y0).argmin()

    print(x0, y0, x_indx, y_indx, d_width, d_height)
    return VC[x_indx:(x_indx + d_width),
              y_indx:(y_indx + d_height), :].sum(axis=(0, 1))


if __name__ == "__main__":
    generate_vector_field()
