import numpy as np
from pathlib import Path

def load_obj(path):
    """Load an OBJ and return (V, UV, N, F_full, mtllib_line)."""
    vertices, uvs, normals, faces = [], [], [], []
    mtl_line = None

    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            head, *rest = line.split()
            if head == 'v':          # vertex
                vertices.append([float(x) for x in rest])
            elif head == 'vt':       # texture‑coord (keep first 2 comps)
                uvs.append([float(x) for x in rest[:2]])
            elif head == 'vn':       # normal
                normals.append([float(x) for x in rest])
            elif head == 'f':        # face; keep v/vt/vn triplets
                faces.append([tuple(int(i) if i else 0 for i in v.split('/'))
                              for v in rest])
            elif head == 'mtllib':   # keep as‑is
                mtl_line = line

    return (np.asarray(vertices, np.float32),
            np.asarray(uvs,      np.float32),
            np.asarray(normals,  np.float32),
            faces,
            mtl_line)


def write_obj(path, V, UV, N, F, mtl_line=None):
    """Write an OBJ with full v / vt / vn / f syntax preserved."""
    with open(path, 'w') as fp:
        if mtl_line:
            fp.write(mtl_line if mtl_line.endswith('\n') else mtl_line + '\n')

        for v in V:
            fp.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for uv in UV:
            fp.write(f"vt {uv[0]} {uv[1]}\n")
        for n in N:
            fp.write(f"vn {n[0]} {n[1]} {n[2]}\n")

        # Faces still use original 1‑based indices
        for face in F:
            fp.write("f " + " ".join(
                f"{vi}/{ti}/{ni}" for vi, ti, ni in face) + "\n")


def laplacian_smooth(V, F_idx, iterations=10, lam=0.3,
                     taubin=False, mu_scale=1.02):
    """
    Uniform Laplacian smoothing.
      V       : (n,3) float32 vertex array (will be copied)
      F_idx   : (m,3) int32 zero‑based vertex‑index faces
      lam     : forward step (0‥1)
      taubin  : if True, run second pass with mu = -lam/mu_scale
    Returns smoothed vertex array.
    """
    n = len(V)
    # build 1‑ring adjacency
    adj = [[] for _ in range(n)]
    for a, b, c in F_idx:
        adj[a] += [b, c]
        adj[b] += [a, c]
        adj[c] += [a, b]
    adj = [list(set(nb)) for nb in adj]        # unique neighbours

    V = V.copy()
    step = lambda arr, step_lam: arr + step_lam * (
        np.array([arr[nb].mean(axis=0) if nb else arr[i]
                  for i, nb in enumerate(adj)]) - arr)

    for _ in range(iterations):
        V = step(V, lam)
        if taubin:
            mu = -lam / mu_scale
            V = step(V, mu)
    return V


if __name__ == "__main__":
    in_path  = Path("logs/name_mesh.obj")
    out_path = Path("logs/name_mesh_smooth.obj")

    V, UV, N, F_full, mtllib = load_obj(in_path)

    # extract vertex indices (OBJ is 1‑based) -> 0‑based NumPy array
    F_idx = np.asarray([[vi - 1 for (vi, _, _) in tri] for tri in F_full],
                       dtype=np.int32)

    V_smooth = laplacian_smooth(
        V, F_idx,
        iterations=10,          # gentler than 25
        lam=0.1,                # smaller λ
        taubin=True             # volume‑preserving
    )

    write_obj(out_path, V_smooth, UV, N, F_full, mtllib)
    print(f"✓ Smoothed mesh written to {out_path.resolve()}")
