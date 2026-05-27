# -*- coding: utf-8 -*-
"""Public refinement entry points for meshes_new."""

import numpy as np
try:
    from .refinement_nvb import refine_nvb, mesh_closure
    from .mesh_checks import check_mesh, check_no_degenerate_cells, check_no_nonmanifold_edges
    from .marking import dorfler_marking
except ImportError:  # direct execution/debugging from this directory
    from refinement_nvb import refine_nvb, mesh_closure
    from mesh_checks import check_mesh, check_no_degenerate_cells, check_no_nonmanifold_edges
    from marking import dorfler_marking


if __name__ == "__main__":
    from simfempy.meshes_new import testmeshes
    import matplotlib.pyplot as plt

    mesh = testmeshes.unitsquare(h=0.5)

    print(mesh.bdrylabels.keys())

    circle=False
    for k in range(5):
        if circle:
            centers = mesh.points[mesh.cells].mean(axis=1)
            marked = centers[:, 0] ** 2 + centers[:, 1] ** 2 < 0.5 ** 2
        else:
            eta = np.random.rand(mesh.ncells)
            marked = dorfler_marking(eta, theta=0.5)

        refine_nvb(mesh, marked, method="NVB3", debug=True)



        print(
            f"it={k+1} npoints={len(mesh.points)} "
            f"ncells={len(mesh.cells)} nfaces={len(mesh.faces)}"
        )
        print(mesh.bdrylabels.keys())

        mesh.plot(
            bdry=True,
            title=(
                f"NVB iteration {k+1}: "
                f"{len(mesh.points)} vertices, {len(mesh.cells)} cells"
            ),
        )
        plt.show()
