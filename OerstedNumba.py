from functools import partial
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numba
import numpy as np
from numba import njit
# from numba import jitclass
from numba.experimental import jitclass
from multiprocess.pool import Pool

import itertools as it

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
        self.I_dir = np.array([0., 0., 1.], dtype=np.float32)
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


# test_list = numba.typed.List()
# test_element = Element(0, 0, 0, 0, 1)
# test_list.append(test_element)


@njit(parallel=True, fastmath=True)
def parallel_scalar_contribs(elements, Xs, Ys):
    elen = len(elements)
    assert (Xs.shape == Ys.shape)
    res = np.zeros_like(Xs)
    for i_el in numba.prange(elen):
        for i in range(Xs.shape[0]):
            for j in range(Xs.shape[1]):
                res[i, j] += elements[i_el].get_contribution_at_p(
                    (Xs[i, j], Ys[i, j]))
    return res


@njit(parallel=True)
def parallel_vector_contribs(elements, Xs, Ys):
    elen = len(elements)
    assert (Xs.shape == Ys.shape)
    vals = np.zeros((*Xs.shape, 2), dtype=np.float32)
    for i in numba.prange(Xs.shape[0]):
        for j in numba.prange(Xs.shape[1]):
            for i_el in numba.prange(elen):
                vals[i, j, :] += elements[i_el].get_vector_contribution_at_p(
                    np.asarray([Xs[i, j], Ys[i, j]]))
    return vals


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def multiprocessing_scalar_contribs(elements, Xs, Ys, cores: int = 16):
    final_res = np.zeros_like(Xs)
    with Pool(processes=cores) as pool:
        for res in pool.imap_unordered(
            partial(
                parallel_scalar_contribs, Xs=Xs, Ys=Ys
            ),
            chunks(elements, cores)
        ):
            final_res += res
    return final_res


@jitclass
class Composite:
    w: float
    h: float
    l: float
    area: float
    elements: List[Element]
    resistivity: float
    resistance: float
    current_density: float
    offset: Tuple[float, float]
    Nx: float
    Ny: float

    def __init__(self, size, discretisation, offset, resistivity, current_density):
        """
        :param size:
            (W, L, H) aka (x, y, z)
        :param discretisation:
            sizes of tiny element
        """
        (self.w, self.h, self.l) = size
        self.area = self.w * self.h
        self.resistivity = resistivity

        (self.Nx, self.Ny) = discretisation
        self.offset = offset
        self.current_density = current_density
        elements = self.mesh_discretisation(swidth=self.w,
                                            sheight=self.h,
                                            rwidth=self.Nx,
                                            rheight=self.Ny,
                                            offset=self.offset)
        self.elements = numba.typed.List(elements)
        # self.elements = elements

    def mesh_discretisation(self, swidth, sheight, rwidth, rheight,
                            offset=(0, 0, 0)):
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
        for ix in numba.prange(Nx):
            for iy in numba.prange(Ny):
                element_groups.append(
                    Element(xoff + ix * rwidth, yoff + iy * rheight, rwidth,
                            rheight, self.current_density))
        return element_groups

    def calculate_contribs(self, Xs, Ys):
        elen = len(self.elements)
        assert (Xs.shape == Ys.shape)
        res = np.zeros_like(Xs)
        for i in numba.prange(Xs.shape[0]):
            for j in numba.prange(Xs.shape[1]):
                p = (Xs[i, j], Ys[i, j])
                for i_el in numba.prange(elen):
                    res[i, j] += self.elements[i_el].get_contribution_at_p(
                        p)
        return res
        return parallel_scalar_contribs(self.elements, Xs, Ys)
        # return multiprocessing_scalar_contribs(self.elements, Xs, Ys, cores=16)

    # def calculate_vector_contribs(self, Xs, Ys):
    #     return parallel_vector_contribs(self.elements, Xs, Ys)

    def set_per_element_current(self, current):
        # the current is evenly distributed along the cross-sec.
        per_element_current = current
        self.set_current(per_element_current)
        return per_element_current


class CompositeStructure:
    composites: List[Composite]

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


sizeW = 20e-6  # nm
sizeL = 20e-6  # nm
sizeH = 4e-9  # nm
current_density = 4.2e10  # density

Nx = 1
Ny = 1

Dx = 1e-8
Dy = 1e-9
x = np.arange(0, (Nx + 1) * sizeW, Dx)
y = np.arange(0, (Ny + 1) * sizeH, Dy)
X, Y = np.meshgrid(x, y)

comp = Composite(
    size=(sizeW, sizeH, sizeL),
    discretisation=(Dx, Dy),
    offset=(0, 0),
    resistivity=10.6e-8,
    current_density=current_density
)

start = time.time()
CC = comp.calculate_contribs(X, Y)
# CC = multiprocessing_scalar_contribs(comp.elements, X, Y)
end = time.time()
print(f"Computation lasted: {end-start:.2f}")

depth_start = 1e-9  # nm
depth_stop = 5e-9  # nm
depth_indices = np.argwhere((y >= depth_start) & (y <= depth_stop))
width = 1000e-9
width_index = np.argmin(abs(x - width)).ravel()[0]


fig, ax = plt.subplots(figsize=(14, 14))
toOersted = 79.5774715459
print(CC[depth_indices, width_index].shape,
      CC[depth_indices, width_index].max())
ax.plot(y[depth_indices], CC[depth_indices, width_index]/toOersted)
ax.set_xlabel("Depth [nm]")
ax.set_ylabel("Oersted field [Oe]")
fig.savefig("Vector_field.png")
