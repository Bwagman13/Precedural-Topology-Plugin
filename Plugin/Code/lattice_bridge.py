bl_info = {
    "name": "Lattice Bridge - Generative Strut Networks",
    "author": "Bradley Wagman",
    "version": (0, 5, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar (N) > Lattice Bridge",
    "description": "Branching load-path strut networks with tangent terminations. Cast-aware.",
    "category": "Add Mesh",
}

import bpy
import heapq
import json
import os
import numpy as np
from mathutils import Vector
from bpy.props import (
    FloatProperty, IntProperty, PointerProperty, BoolProperty, EnumProperty
)
from bpy.types import Operator, Panel, PropertyGroup


# ======================================================================
# Sampling
# ======================================================================

def farthest_point_sample(pts, n, seed=0):
    if len(pts) == 0:
        return np.empty(0, dtype=int)
    n = min(n, len(pts))
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(pts)))]
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        if d[i] <= 1e-12:
            break
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(pts - pts[i], axis=1))
    return np.array(idx, dtype=int)


def object_surface_points(obj, depsgraph, target=2500, seed=0):
    """
    Area-weighted samples across the surface, not just its vertices.

    Raw vertices are far too sparse on primitives - a default cylinder has a
    couple of rings and nothing in between. Once a sector mask and a design-space
    mask are both applied, that pool can empty out entirely, and any code that
    reacts by relaxing a constraint will put anchors somewhere absurd. Sampling
    the faces keeps the pool dense enough that the constraints can stay strict.
    """
    eobj = obj.evaluated_get(depsgraph)
    try:
        me = eobj.to_mesh()
    except RuntimeError:
        return np.empty((0, 3))
    mw = eobj.matrix_world
    verts = np.array([(mw @ v.co)[:] for v in me.vertices], dtype=np.float64)

    tris = []
    for poly in me.polygons:
        vi = list(poly.vertices)
        for k in range(1, len(vi) - 1):
            tris.append((vi[0], vi[k], vi[k + 1]))
    eobj.to_mesh_clear()

    if not tris or len(verts) == 0:
        return verts

    tris = np.asarray(tris, dtype=int)
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = float(area.sum())
    if total <= 0.0:
        return verts

    rng = np.random.default_rng(seed)
    n = max(0, target - len(verts))
    if n == 0:
        return verts
    pick = rng.choice(len(tris), size=n, p=area / total)
    u = rng.random(n)
    v = rng.random(n)
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    samples = (a[pick] + (b[pick] - a[pick]) * u[:, None]
               + (c[pick] - a[pick]) * v[:, None])
    return np.vstack([verts, samples])


def surface_normal_at(obj, world_pt):
    """Outward world-space normal of obj nearest to world_pt."""
    p_local = obj.matrix_world.inverted() @ Vector(world_pt)
    ok, loc, nor, _ = obj.closest_point_on_mesh(p_local)
    if not ok:
        return np.array([0.0, 0.0, 1.0])
    n = (obj.matrix_world.to_3x3() @ nor).normalized()
    return np.array(n[:])


def point_is_inside(obj, world_pt, eps=0.0):
    p_local = obj.matrix_world.inverted() @ Vector(world_pt)
    ok, loc, nor, _ = obj.closest_point_on_mesh(p_local)
    if not ok:
        return False
    return (p_local - loc).dot(nor) < eps


def clearance_from_surface(obj, world_pt):
    """Signed distance to obj's surface. Negative = inside."""
    p_local = obj.matrix_world.inverted() @ Vector(world_pt)
    ok, loc, nor, _ = obj.closest_point_on_mesh(p_local)
    if not ok:
        return 0.0
    d = (p_local - loc).length
    return -d if (p_local - loc).dot(nor) < 0 else d


def stem_direction(obj, surf_pt, nrm, domain, other_centre, dist):
    """
    Which way is 'out of this body and into the design space'?

    A raw surface normal points out of the body, but on a curved wall that can
    aim straight away from the design space - e.g. anchors on the upper flank of
    a dome, whose normals face up and out. Following those, the stem lands
    outside the dome and its strut spears back through the wall to reach it.
    That is what puts nubs through the surface. Pick the side that actually
    leads into the design space instead of trusting the normal blindly.
    """
    nrm = np.asarray(nrm, dtype=np.float64)
    for cand in (nrm, -nrm):
        pt = np.asarray(surf_pt) + cand * dist
        if point_is_inside(obj, pt):
            continue                      # that way is into its own body
        if domain is not None and not point_is_inside(domain, pt):
            continue                      # that way leaves the design space
        return cand
    v = np.asarray(other_centre) - np.asarray(surf_pt)
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-12 else nrm


def bury_anchor(obj, surf_pt, normal, want, radius, max_steps=16):
    """
    Sink an anchor until its whole flared disc clears the wall.

    The cap has a radius, so burying the centreline by the nominal amount still
    leaves the rim proud wherever the surface curves away or the strut meets it
    off-square. March inward until the clearance beneath the point exceeds the
    strut radius, so the entire end face is under the surface.
    """
    step = max(radius * 0.35, 1e-6)
    p = np.asarray(surf_pt, dtype=np.float64) - np.asarray(normal) * want
    for _ in range(max_steps):
        if -clearance_from_surface(obj, p) >= radius:
            return p, True
        p = p - np.asarray(normal) * step
    return p, False


def combined_bounds(objs, margin=0.0):
    lo = np.array([np.inf] * 3)
    hi = np.array([-np.inf] * 3)
    for o in objs:
        for corner in o.bound_box:
            w = np.array((o.matrix_world @ Vector(corner))[:])
            lo = np.minimum(lo, w)
            hi = np.maximum(hi, w)
    span = hi - lo
    return lo - span * margin, hi + span * margin


def blue_noise_domain(preserve, domain_obj, count, seed, margin=0.05):
    """Minimum-spacing scatter so the web grades evenly instead of clumping."""
    rng = np.random.default_rng(seed)
    ref = [domain_obj] if domain_obj else preserve
    lo, hi = combined_bounds(ref, 0.0 if domain_obj else margin)
    vol = float(np.prod(np.maximum(hi - lo, 1e-9)))
    spacing = (vol / max(count, 1)) ** (1.0 / 3.0) * 0.72

    out, attempts, budget = [], 0, count * 150
    while len(out) < count and attempts < budget:
        attempts += 1
        p = lo + rng.random(3) * (hi - lo)
        if domain_obj and not point_is_inside(domain_obj, p):
            continue
        if any(point_is_inside(o, p) for o in preserve):
            continue
        if out and np.min(np.linalg.norm(np.asarray(out) - p, axis=1)) < spacing:
            continue
        out.append(p)
    return (np.asarray(out) if out else np.empty((0, 3))), spacing


def pick_anchors(obj, surf, other_centre, domain_obj, want_off, count, seed):
    """
    An anchor is only usable if a stem can leave the body there and land in the
    design space. Points on a flank that faces away - the top half of a dome,
    say - fail that test no matter which way the stem points, and a strut
    reaching them has to spear back out through the wall. Reject them at
    selection rather than trying to fix the geometry afterwards.
    """
    # The design space is a hard boundary, never a preference. The old code
    # only applied it when enough candidates survived, so narrowing the pool -
    # with a symmetry sector, say - made it fail open and scatter anchors right
    # across the body, far outside the box.
    cand = surf
    if domain_obj is not None:
        mask = np.array([point_is_inside(domain_obj, q, eps=1e-9) for q in surf])
        cand = surf[mask]
        if len(cand) < 3:
            return None, None, None, -1

    if len(cand) > 500:
        cand = cand[farthest_point_sample(cand, 500, seed)]

    # Try the full stem length first, then progressively shorter ones. A stem
    # that will not fit is shortened, never abandoned - the old code fell
    # through to an unvalidated direction and fired struts off into open air.
    keep, dirs, dists = [], [], []
    for q in cand:
        n = surface_normal_at(obj, q)
        placed = False
        for scale in (1.0, 0.7, 0.45, 0.25):
            for c in (n, -n):
                pt = q + c * want_off * scale
                if point_is_inside(obj, pt):
                    continue
                if domain_obj is not None and not point_is_inside(domain_obj, pt):
                    continue
                keep.append(q)
                dirs.append(c)
                dists.append(want_off * scale)
                placed = True
                break
            if placed:
                break

    if len(keep) >= max(count, 3):
        cand = np.asarray(keep)
        dvec = np.asarray(dirs)
        dlen = np.asarray(dists)
        viable = len(keep)
    else:
        d = np.linalg.norm(cand - other_centre, axis=1)
        cand = cand[np.argsort(d)[: max(count * 6, count)]]
        dvec = other_centre - cand
        nrm = np.linalg.norm(dvec, axis=1)
        nrm[nrm < 1e-12] = 1.0
        dvec = dvec / nrm[:, None]
        dlen = np.full(len(cand), want_off * 0.25)
        viable = 0

    idx = farthest_point_sample(cand, count, seed)
    return cand[idx], dvec[idx], dlen[idx], viable


# ======================================================================
# Graph
# ======================================================================

def build_knn_graph(pts, k, max_len, skip=0):
    """skip = number of leading nodes excluded (embedded anchors route only
    through their own standoff, never straight into the web)."""
    n = len(pts)
    if n < 2:
        return []
    k = max(1, min(k, n - 1))
    diff = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    if skip:
        D[:skip, :] = np.inf
        D[:, :skip] = np.inf
    order = np.argsort(D, axis=1)[:, :k]
    edges = set()
    for i in range(skip, n):
        for j in order[i]:
            j = int(j)
            if D[i, j] <= max_len:
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def drop_edges_through(pts, edges, blockers, protect=(), samples=3):
    """Remove struts tunnelling through preserve bodies. Anchor stems are
    exempt - they are meant to be buried."""
    if not blockers:
        return edges
    prot = set(protect)
    keep = []
    for a, b in edges:
        if (a, b) in prot:
            keep.append((a, b))
            continue
        pa, pb = pts[a], pts[b]
        if any(
            point_is_inside(o, pa + (pb - pa) * (s / (samples + 1.0)))
            for s in range(1, samples + 1)
            for o in blockers
        ):
            continue
        keep.append((a, b))
    return keep


def force_connect(pts, edges, from_idx, pool_start, k=3):
    eset = set(edges)
    if len(pts) <= pool_start:
        return sorted(eset)
    for a in from_idx:
        d = np.linalg.norm(pts[pool_start:] - pts[a], axis=1)
        for j in np.argsort(d)[:k]:
            b = int(j) + pool_start
            eset.add((min(a, b), max(a, b)))
    return sorted(eset)


def anchored_component(edges, src, dst):
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, best = set(), None
    for start in list(src) + list(dst):
        if start in seen or start not in adj:
            continue
        comp, stack = set(), [start]
        while stack:
            u = stack.pop()
            if u in comp:
                continue
            comp.add(u)
            for v in adj.get(u, ()):
                if v not in comp:
                    stack.append(v)
        seen |= comp
        if any(s in comp for s in src) and any(t in comp for t in dst):
            if best is None or len(comp) > len(best):
                best = comp
    if best is None:
        return [], set()
    return [e for e in edges if e[0] in best and e[1] in best], best


def dijkstra(adj, source, n):
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[source] = 0.0
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, prev


def reinforced_load_paths(pts, edges, src_nodes, dst_nodes, rounds=8,
                          bundling=0.6, evaporate=0.25, wander=0.15, seed=0):
    """
    Iterative path reinforcement. Used struts get cheaper each round, so routes
    bundle into a few trunks that branch to the anchors instead of every pair
    finding its own way. Unused struts evaporate and are deleted.
    """
    n = len(pts)
    rng = np.random.default_rng(seed)
    base, jitter = {}, {}
    for e in edges:
        base[e] = float(np.linalg.norm(pts[e[0]] - pts[e[1]]))
        jitter[e] = 1.0 + rng.random() * wander
    pher = {e: 0.0 for e in edges}

    for _ in range(max(1, rounds)):
        adj = [[] for _ in range(n)]
        for e in edges:
            a, b = e
            w = base[e] * jitter[e] / ((1.0 + pher[e]) ** bundling)
            adj[a].append((b, w))
            adj[b].append((a, w))
        fresh = {e: 0.0 for e in edges}
        for s in src_nodes:
            dist, prev = dijkstra(adj, int(s), n)
            for t in dst_nodes:
                t = int(t)
                if dist[t] == float("inf"):
                    continue
                cur = t
                while prev[cur] != -1:
                    p = prev[cur]
                    fresh[(min(p, cur), max(p, cur))] += 1.0
                    cur = p
        for e in edges:
            pher[e] = pher[e] * (1.0 - evaporate) + fresh[e]
    return {e: v for e, v in pher.items() if v > 1e-9}


def relax_nodes(pts, edges, fixed, iterations, strength,
                preserve, domain_obj):
    if iterations <= 0:
        return pts
    pts = pts.copy()
    nbr = [[] for _ in range(len(pts))]
    for a, b in edges:
        nbr[a].append(b)
        nbr[b].append(a)
    fixed = set(fixed)
    for _ in range(iterations):
        new = pts.copy()
        for i in range(len(pts)):
            if i in fixed or not nbr[i]:
                continue
            cand = pts[i] + (pts[nbr[i]].mean(axis=0) - pts[i]) * strength
            if domain_obj is not None and not point_is_inside(domain_obj, cand):
                continue
            if any(point_is_inside(o, cand) for o in preserve):
                continue
            new[i] = cand
        pts = new
    return pts


def orthonormal_basis(axis):
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    a = a / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
    tmp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, tmp)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1), a


def sector_mask(pts, axis_pt, axis_dir, mode, count):
    """Which points fall in the fundamental domain we actually solve in."""
    if mode == "NONE" or len(pts) == 0:
        return np.ones(len(pts), dtype=bool)
    e1, e2, _ = orthonormal_basis(axis_dir)
    v = np.asarray(pts) - np.asarray(axis_pt)
    x, y = v @ e1, v @ e2
    if mode == "MIRROR":
        return x >= -1e-12
    ang = np.arctan2(y, x) % (2.0 * np.pi)
    span = (2.0 * np.pi / count) if mode == "RADIAL" else (np.pi / count)
    return ang <= span + 1e-9


def symmetry_transforms(mode, count, axis_dir):
    e1, e2, a = orthonormal_basis(axis_dir)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])

    def rot(t):
        return np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)

    if mode == "MIRROR":
        return [np.eye(3), np.eye(3) - 2.0 * np.outer(e1, e1)]
    if mode == "RADIAL":
        return [rot(2.0 * np.pi * k / count) for k in range(count)]
    if mode == "DIHEDRAL":
        M = np.eye(3) - 2.0 * np.outer(e2, e2)
        out = []
        for k in range(count):
            R = rot(2.0 * np.pi * k / count)
            out.append(R)
            out.append(R @ M)
        return out
    return [np.eye(3)]


def replicate_symmetry(pts, edges, load, fixed, mode, count, axis_pt, axis_dir,
                       weld_tol):
    """
    Solve once in a sector, then copy it around the axis.

    Symmetry has to be imposed on the solve, not applied afterwards - path
    finding is seeded and jittered, so two independent solves of mirrored
    regions never agree. Replicating one solved sector is the only way to get
    symmetry that is exact rather than approximate.
    """
    mats = symmetry_transforms(mode, count, axis_dir)
    if len(mats) <= 1:
        return pts, edges, load, fixed

    axis_pt = np.asarray(axis_pt, dtype=np.float64)
    base = np.asarray(pts) - axis_pt
    n = len(pts)

    new_pts, new_load, new_fixed, new_edges = [], [], set(), []
    for c, R in enumerate(mats):
        off = c * n
        new_pts.append(base @ R.T + axis_pt)
        new_load.append(np.asarray(load))
        new_fixed |= {i + off for i in fixed}
        new_edges.extend([(a + off, b + off) for a, b in edges])

    P = np.vstack(new_pts)
    L = np.concatenate(new_load)

    # Points sitting on the axis land on top of each other under every
    # transform; weld them so the hub is one node, not N stacked ones.
    key = {}
    remap = np.arange(len(P))
    for i, k in enumerate(map(tuple, np.round(
            P / max(weld_tol, 1e-9)).astype(np.int64))):
        if k in key:
            remap[i] = key[k]
        else:
            key[k] = i
    keep = sorted(set(remap.tolist()))
    idx = {o: i for i, o in enumerate(keep)}
    P2, L2 = P[keep], L[keep]
    F2 = {idx[remap[i]] for i in new_fixed}
    E2 = sorted({(min(idx[remap[a]], idx[remap[b]]),
                  max(idx[remap[a]], idx[remap[b]]))
                 for a, b in new_edges
                 if idx[remap[a]] != idx[remap[b]]})
    return P2, E2, L2, F2


def sector_ids(pts, axis_pt, axis_dir, mode, count):
    e1, e2, _ = orthonormal_basis(axis_dir)
    v = np.asarray(pts) - np.asarray(axis_pt)
    ang = np.arctan2(v @ e2, v @ e1) % (2.0 * np.pi)
    span = {"RADIAL": 2.0 * np.pi / count,
            "MIRROR": np.pi,
            "DIHEDRAL": np.pi / count}.get(mode, 2.0 * np.pi)
    return (ang / span).astype(int)


def stitch_seams(pts, cand, live_edges, axis_pt, axis_dir, mode, count,
                 fraction):
    """
    Tie neighbouring sectors together across the seams.

    Ordinary webbing sorts candidates by length and takes the shortest, and
    within-sector pairs are always shorter than pairs that reach across a seam.
    Left to it, every cross-seam tie loses and the result is N separate ribs
    that happen to share two bodies. Seam ties need their own quota.
    """
    if fraction <= 0.0 or mode == "NONE":
        return live_edges, set()
    sid = sector_ids(pts, axis_pt, axis_dir, mode, count)
    have = set(live_edges)
    pool = [e for e in cand if e not in have and sid[e[0]] != sid[e[1]]]
    if not pool:
        return live_edges, set()
    pool.sort(key=lambda e: float(np.linalg.norm(pts[e[0]] - pts[e[1]])))
    web = set(pool[:max(1, int(round(len(pool) * fraction)))])
    return sorted(have | web), web


def add_webbing(pts, all_edges, live_edges, comp, fraction):
    """
    Cross-braces. Load-path solving keeps only material that carries something,
    which yields clean trunks but a sparse, spidery result. Real webbing has
    thin ties between trunks that carry little on their own. Add back the
    shortest discarded struts between nodes already in the network.
    """
    if fraction <= 0.0:
        return live_edges, set()
    have = set(live_edges)
    pool = [e for e in all_edges
            if e not in have and e[0] in comp and e[1] in comp]
    if not pool:
        return live_edges, set()
    pool.sort(key=lambda e: float(np.linalg.norm(pts[e[0]] - pts[e[1]])))
    take = max(0, int(round(len(pool) * fraction)))
    web = set(pool[:take])
    return sorted(have | web), web


def twist_nodes(pts, movable, centre_a, centre_b, angle):
    """
    Helical shear about the axis joining the two bodies.

    Twist is scaled by sin(pi*t) along that axis, so it is zero at both bodies
    and greatest mid-span. A linear twist would tear the network away from its
    anchors, which are pinned to real surfaces and cannot rotate.
    """
    if abs(angle) < 1e-9:
        return pts
    axis = np.asarray(centre_b) - np.asarray(centre_a)
    L = float(np.linalg.norm(axis))
    if L < 1e-12:
        return pts
    axis = axis / L
    out = pts.copy()
    for i in movable:
        v = pts[i] - centre_a
        t = float(np.clip(np.dot(v, axis) / L, 0.0, 1.0))
        a = angle * np.sin(np.pi * t)
        c, s = np.cos(a), np.sin(a)
        out[i] = (centre_a + v * c + np.cross(axis, v) * s
                  + axis * np.dot(axis, v) * (1.0 - c))
    return out


def subdivide_and_curve(pts, edges, radii, fixed, cuts, iterations, strength,
                        thin_edges=(), r_thin=None):
    """
    Turn the straight-chord graph into swept curves.

    Struts are chords between sparse nodes, so a trunk is a zigzag of straight
    segments and nothing has curvature to blend with. Insert points along every
    edge, then Laplacian-smooth the inserted points while pinning the anchors
    and their stems. Repeated averaging over a subdivided polyline converges to
    a B-spline, so trunks sweep, junctions round off, and struts leave each
    surface square-on before curving away - which is what lets an end actually
    follow the dome instead of stabbing at it.
    """
    pts = list(np.asarray(pts))
    radii = list(np.asarray(radii))
    thin_edges = set(thin_edges)
    new_edges = []
    for a, b in edges:
        if cuts <= 0:
            new_edges.append((a, b))
            continue
        # Skin radius is per-vertex, so a brace between two fat trunk nodes
        # would inherit their bulk. Pin its inserted midpoints thin instead:
        # the brace necks down between the trunks and swells where it meets
        # them, which is what a real tie looks like.
        thin = (a, b) in thin_edges and r_thin is not None
        prev = a
        for i in range(1, cuts + 1):
            t = i / (cuts + 1.0)
            idx = len(pts)
            pts.append(pts[a] * (1.0 - t) + pts[b] * t)
            if thin:
                edge_w = np.sin(np.pi * t) ** 0.5
                radii.append(radii[a] * (1.0 - t) + radii[b] * t
                             + (r_thin - (radii[a] * (1.0 - t)
                                          + radii[b] * t)) * edge_w)
            else:
                radii.append(radii[a] * (1.0 - t) + radii[b] * t)
            new_edges.append((prev, idx))
            prev = idx
        new_edges.append((prev, b))

    pts = np.asarray(pts)
    radii = np.asarray(radii)
    if iterations <= 0 or cuts <= 0:
        return pts, new_edges, radii

    nbr = [[] for _ in range(len(pts))]
    for a, b in new_edges:
        nbr[a].append(b)
        nbr[b].append(a)

    movable = [i for i in range(len(pts)) if i not in fixed and nbr[i]]
    for _ in range(iterations):
        cur = pts.copy()
        for i in movable:
            pts[i] = cur[i] + (cur[nbr[i]].mean(axis=0) - cur[i]) * strength
    return pts, new_edges, radii


# ======================================================================
# Mesh build
# ======================================================================

def build_strut_object(context, pts, edges, radii, name, output_mode,
                       voxel, smooth_factor, smooth_iter, smooth_shade):
    used = sorted({i for e in edges for i in e})
    remap = {o: n for n, o in enumerate(used)}
    me = bpy.data.meshes.new(name)
    me.from_pydata([pts[i].tolist() for i in used],
                   [(remap[a], remap[b]) for a, b in edges], [])
    me.update()

    obj = bpy.data.objects.new(name, me)
    context.collection.objects.link(obj)

    skin = obj.modifiers.new("Skin", type="SKIN")
    skin.use_smooth_shade = True
    sv = me.skin_vertices[0].data
    for o in used:
        r = float(radii[o])
        sv[remap[o]].radius = (r, r)
    sv[0].use_root = True

    if output_mode == "REMESH":
        # Voxel size is deliberately NOT tiny. Fine voxels resolve the gap
        # between near-touching struts and hand back a scattered mess; coarser
        # voxels bridge those sub-voxel gaps into one coherent solid.
        rm = obj.modifiers.new("Remesh", type="REMESH")
        rm.mode = "VOXEL"
        rm.voxel_size = max(voxel, 1e-5)
        rm.use_smooth_shade = smooth_shade
        sm = obj.modifiers.new("Smooth", type="SMOOTH")
        sm.factor = smooth_factor
        sm.iterations = smooth_iter
    else:
        sub = obj.modifiers.new("Subdivision", type="SUBSURF")
        sub.levels = 2
        sub.render_levels = 2

    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


def eval_mesh_world(obj, dg):
    eo = obj.evaluated_get(dg)
    me = eo.to_mesh()
    mw = eo.matrix_world
    verts = [tuple(mw @ v.co) for v in me.vertices]
    faces = [tuple(p.vertices) for p in me.polygons]
    eo.to_mesh_clear()
    return verts, faces


# ======================================================================
# Presets
# ======================================================================

PRESET_KEYS = (
    "nodes", "anchors", "neighbours", "max_len",
    "rounds", "bundling", "evaporate", "wander", "prune", "webbing", "twist", "symmetry", "sym_count", "stitch",
    "embed", "standoff", "flare",
    "relax", "relax_str", "cuts", "curve_iter", "curve_str",
    "auto_thick", "d_min", "d_max", "gamma", "cast_floor",
    "output_mode", "voxel_auto", "voxel", "smooth_factor", "smooth_iter",
    "smooth", "seed", "blend", "blend_iter", "keep_source",
)

BUILTIN_PRESETS = {
    "Knob - Bradley": {
        "nodes": 40, "anchors": 10, "neighbours": 5, "max_len": 0.060,
        "rounds": 8, "bundling": 0.60, "evaporate": 0.25, "wander": 0.30,
        "prune": 0.10, "webbing": 0.0, "twist": 0.0, "symmetry": "NONE", "sym_count": 5, "stitch": 0.25,
        "embed": 2.0, "standoff": 0.20, "flare": 0.20,
        "relax": 10, "relax_str": 0.35,
        "cuts": 4, "curve_iter": 14, "curve_str": 0.50,
        "auto_thick": True, "d_min": 0.003, "d_max": 0.005, "gamma": 0.50,
        "cast_floor": 0.003, "output_mode": "REMESH", "voxel_auto": False,
        "voxel": 0.00135, "smooth_factor": 1.0, "smooth_iter": 4,
        "smooth": True, "seed": 1, "blend": 0.010, "blend_iter": 20,
        "keep_source": True,
    },
    "Radial 5 - Knob": {
        "nodes": 90, "anchors": 6, "neighbours": 5, "max_len": 0.060,
        "rounds": 8, "bundling": 0.55, "evaporate": 0.25, "wander": 0.25,
        "prune": 0.10, "webbing": 0.15, "twist": 0.0,
        "symmetry": "RADIAL", "sym_count": 5, "stitch": 0.30,
        "embed": 2.0, "standoff": 0.20, "flare": 0.60,
        "relax": 10, "relax_str": 0.35,
        "cuts": 5, "curve_iter": 22, "curve_str": 0.50,
        "auto_thick": True, "d_min": 0.003, "d_max": 0.005, "gamma": 0.50,
        "cast_floor": 0.003, "output_mode": "REMESH", "voxel_auto": False,
        "voxel": 0.00135, "smooth_factor": 1.0, "smooth_iter": 4,
        "smooth": True, "seed": 1, "blend": 0.012, "blend_iter": 20,
        "keep_source": True,
    },
    "Webbed": {
        "nodes": 70, "anchors": 12, "neighbours": 7, "max_len": 0.060,
        "rounds": 6, "bundling": 0.35, "evaporate": 0.25, "wander": 0.40,
        "prune": 0.05, "webbing": 0.35, "twist": 0.0, "symmetry": "NONE", "sym_count": 5, "stitch": 0.25,
        "embed": 2.0, "standoff": 0.20, "flare": 0.20,
        "relax": 10, "relax_str": 0.35,
        "cuts": 5, "curve_iter": 20, "curve_str": 0.50,
        "auto_thick": True, "d_min": 0.003, "d_max": 0.005, "gamma": 0.65,
        "cast_floor": 0.003, "output_mode": "REMESH", "voxel_auto": False,
        "voxel": 0.0011, "smooth_factor": 1.0, "smooth_iter": 4,
        "smooth": True, "seed": 1, "blend": 0.010, "blend_iter": 20,
        "keep_source": True,
    },
    "Few Heavy Trunks": {
        "nodes": 30, "anchors": 6, "neighbours": 4, "max_len": 0.060,
        "rounds": 14, "bundling": 1.10, "evaporate": 0.25, "wander": 0.20,
        "prune": 0.30, "webbing": 0.0, "twist": 0.0, "symmetry": "NONE", "sym_count": 5, "stitch": 0.25,
        "embed": 2.0, "standoff": 0.20, "flare": 0.20,
        "relax": 10, "relax_str": 0.35,
        "cuts": 6, "curve_iter": 26, "curve_str": 0.55,
        "auto_thick": True, "d_min": 0.003, "d_max": 0.005, "gamma": 0.40,
        "cast_floor": 0.003, "output_mode": "REMESH", "voxel_auto": False,
        "voxel": 0.0015, "smooth_factor": 1.0, "smooth_iter": 4,
        "smooth": True, "seed": 1, "blend": 0.012, "blend_iter": 24,
        "keep_source": True,
    },
    "Twisted Vine": {
        "nodes": 55, "anchors": 10, "neighbours": 6, "max_len": 0.060,
        "rounds": 8, "bundling": 0.55, "evaporate": 0.25, "wander": 0.45,
        "prune": 0.12, "webbing": 0.18, "twist": 0.79, "symmetry": "NONE", "sym_count": 5, "stitch": 0.25,
        "embed": 2.0, "standoff": 0.20, "flare": 0.20,
        "relax": 8, "relax_str": 0.30,
        "cuts": 6, "curve_iter": 30, "curve_str": 0.55,
        "auto_thick": True, "d_min": 0.003, "d_max": 0.005, "gamma": 0.55,
        "cast_floor": 0.003, "output_mode": "REMESH", "voxel_auto": False,
        "voxel": 0.0012, "smooth_factor": 1.0, "smooth_iter": 4,
        "smooth": True, "seed": 3, "blend": 0.010, "blend_iter": 20,
        "keep_source": True,
    },
}


def preset_dir():
    return bpy.utils.user_resource("SCRIPTS",
                                   path="presets/lattice_bridge", create=True)


def user_preset_names():
    try:
        return sorted(f[:-5] for f in os.listdir(preset_dir())
                      if f.endswith(".json"))
    except OSError:
        return []


def preset_items(self, context):
    items = [(f"@{k}", k, "Built in") for k in BUILTIN_PRESETS]
    items += [(n, n, "Saved") for n in user_preset_names()]
    return items or [("NONE", "No presets", "")]


def apply_preset(props, data):
    for k, v in data.items():
        if k in PRESET_KEYS and hasattr(props, k):
            try:
                setattr(props, k, v)
            except (TypeError, ValueError):
                pass


# ======================================================================
# Properties
# ======================================================================

class LB_Props(PropertyGroup):
    obj_a: PointerProperty(name="Body A", type=bpy.types.Object,
                           poll=lambda s, o: o.type == "MESH")
    obj_b: PointerProperty(name="Body B", type=bpy.types.Object,
                           poll=lambda s, o: o.type == "MESH")
    domain: PointerProperty(name="Design Space", type=bpy.types.Object,
                            poll=lambda s, o: o.type == "MESH")

    preset: EnumProperty(name="Preset", items=preset_items)
    preset_name: bpy.props.StringProperty(name="Name", default="My Preset")

    nodes: IntProperty(name="Nodes", default=40, min=10, max=900,
                       description="Fewer nodes = wider gaps = thicker struts")
    anchors: IntProperty(name="Anchors / Body", default=10, min=2, max=48)
    neighbours: IntProperty(
        name="Connectivity", default=5, min=2, max=12,
        description="Candidate struts per node. Raise for a busier web")
    max_len: FloatProperty(name="Max Strut Length", default=0.060,
                           min=0.0001, unit="LENGTH",
                           description="Set to roughly 2x the reported gap")

    rounds: IntProperty(name="Bundling Rounds", default=8, min=1, max=30)
    bundling: FloatProperty(name="Bundling Strength", default=0.6, min=0.0,
                            max=2.0,
                            description="0 = busy even web with many junctions, "
                                        "1+ = few heavy trunks")
    evaporate: FloatProperty(name="Evaporation", default=0.25, min=0.0, max=0.9)
    wander: FloatProperty(name="Organic Wander", default=0.30, min=0.0, max=1.0)
    prune: FloatProperty(name="Prune Weakest", default=0.10, min=0.0, max=0.95)
    webbing: FloatProperty(
        name="Webbing", default=0.0, min=0.0, max=1.0,
        description="Adds back thin cross-braces between trunks that carry no "
                    "load of their own. This is the junction-count dial - "
                    "bundling makes trunks, webbing ties them together")
    twist: FloatProperty(
        name="Twist", default=0.0, min=-3.15, max=3.15, subtype="ANGLE",
        description="Helical shear about the axis between the bodies. Zero at "
                    "both ends, strongest mid-span")

    symmetry: EnumProperty(
        name="Symmetry",
        items=[("NONE", "None", "Free-grown, no symmetry"),
               ("RADIAL", "Radial", "N identical sectors around the body axis"),
               ("MIRROR", "Mirror", "Bilateral about one plane"),
               ("DIHEDRAL", "Radial + Mirror",
                "N sectors, each mirrored - the most formal option")],
        default="NONE",
        description="Solved in one sector then copied, so symmetry is exact")
    stitch: FloatProperty(
        name="Seam Stitch", default=0.25, min=0.0, max=1.0,
        description="Fraction of cross-seam ties added between neighbouring "
                    "sectors. Without these the sectors are separate ribs that "
                    "merely share the two bodies")
    sym_count: IntProperty(
        name="Sectors", default=5, min=2, max=16,
        description="Number of sectors around the axis. Raise Nodes by roughly "
                    "this factor - each sector only gets a slice of them")

    relax: IntProperty(name="Relax Iterations", default=10, min=0, max=40)
    relax_str: FloatProperty(name="Relax Strength", default=0.35, min=0.0,
                             max=1.0)
    cuts: IntProperty(
        name="Curve Subdivisions", default=4, min=0, max=10,
        description="Points inserted along each strut. 0 gives straight "
                    "chords; higher lets struts actually bend")
    curve_iter: IntProperty(
        name="Curve Smoothing", default=14, min=0, max=80,
        description="Averaging passes over the inserted points. Converges to a "
                    "spline, so trunks sweep instead of zigzagging")
    curve_str: FloatProperty(name="Curve Strength", default=0.5, min=0.0,
                             max=1.0)

    # --- terminations -------------------------------------------------
    embed: FloatProperty(
        name="Embed Depth", default=2.0, min=0.0, max=6.0,
        description="Extra burial past the point where the strut's OUTER "
                    "surface clears the wall, in flared radii. 0 already "
                    "guarantees the full end face is under the surface")
    standoff: FloatProperty(
        name="Normal Standoff", default=0.20, min=0.0, max=3.0,
        description="Length of the square-on stem, in NOMINAL strut radii. "
                    "Short is right - it only has to clear the wall before the "
                    "curve takes over. Long stems read as spikes")
    flare: FloatProperty(
        name="Anchor Flare", default=0.20, min=0.2, max=5.0,
        description="Radius multiplier where the strut enters the body. Below "
                    "1.0 necks the strut down into the surface; above 1.0 "
                    "trumpets it out into a fillet")

    # --- thickness ----------------------------------------------------
    auto_thick: BoolProperty(name="Auto Thickness", default=True)
    d_min: FloatProperty(name="Min Diameter", default=0.003, min=0.00001,
                         unit="LENGTH", description="Across, not a radius")
    d_max: FloatProperty(name="Max Diameter", default=0.007, min=0.00001,
                         unit="LENGTH", description="Across, not a radius")
    gamma: FloatProperty(name="Taper Bias", default=0.5, min=0.05, max=4.0)
    cast_floor: FloatProperty(
        name="Cast Floor (dia)", default=0.003, min=0.0, unit="LENGTH",
        description="3mm is the practical floor for gravity-fed bronze")

    # --- output -------------------------------------------------------
    output_mode: EnumProperty(
        name="Output",
        items=[("REMESH", "Solid (Voxel)", "Watertight, castable"),
               ("SKIN", "Fast (Subsurf)", "Quick preview")],
        default="REMESH")
    voxel_auto: BoolProperty(
        name="Auto Voxel", default=False,
        description="Derive voxel size from strut diameter. Manual is usually "
                    "better - too fine fragments the network")
    voxel: FloatProperty(name="Voxel Size", default=0.00135, min=0.00001,
                         unit="LENGTH")
    smooth_factor: FloatProperty(name="Smooth Factor", default=1.0, min=0.0,
                                 max=2.0)
    smooth_iter: IntProperty(name="Smooth Repeat", default=4, min=0, max=30)
    smooth: BoolProperty(name="Smooth Shade", default=True)
    seed: IntProperty(name="Seed", default=1, min=0)

    # --- fuse ---------------------------------------------------------
    blend: FloatProperty(
        name="Blend Radius", default=0.010, min=0.0001, unit="LENGTH",
        description="How far the tangent fillet reaches from each junction")
    blend_iter: IntProperty(name="Blend Strength", default=20, min=1, max=40)
    keep_source: BoolProperty(name="Keep Originals", default=True)


# ======================================================================
# Generate
# ======================================================================

class LB_OT_generate(Operator):
    bl_idname = "mesh.lattice_bridge_generate"
    bl_label = "Grow Lattice"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        p = context.scene.lattice_bridge
        if not p.obj_a or not p.obj_b or p.obj_a == p.obj_b:
            self.report({"ERROR"}, "Pick two different preserve bodies")
            return {"CANCELLED"}

        dg = context.evaluated_depsgraph_get()
        preserve = [p.obj_a, p.obj_b]
        dom = p.domain

        surf_a = object_surface_points(p.obj_a, dg)
        surf_b = object_surface_points(p.obj_b, dg)
        if len(surf_a) == 0 or len(surf_b) == 0:
            self.report({"ERROR"}, "A preserve body has no geometry")
            return {"CANCELLED"}

        interior, spacing = blue_noise_domain(preserve, dom, p.nodes, p.seed + 2)
        if len(interior) < 8:
            self.report({"ERROR"}, "Could not scatter nodes. Enlarge the "
                                   "Design Space or lower Nodes.")
            return {"CANCELLED"}

        # Provisional diameter, needed before we can place stems.
        d_guess = (spacing * 1.15 * 0.50) if p.auto_thick else p.d_max
        d_guess = max(d_guess, p.cast_floor)

        # Each anchor becomes a buried point plus a standoff point out along the
        # surface normal. The only route in or out of a body is that
        # perpendicular stem, so struts can never graze the surface.
        #
        # Distances are measured from the strut's OUTER SURFACE, not its axis.
        # The end face is a disc of radius r_anchor; burying the centreline
        # alone leaves that disc's rim proud of a curved wall.
        r_anchor = d_guess * p.flare * 0.5      # flared end, for burial
        r_nom = d_guess * 0.5                   # nominal strut, for the stem
        want_emb = r_anchor * (1.0 + p.embed)
        # Stem length keys off the NOMINAL radius and is capped against node
        # spacing. Scaling it by the flare compounded two multipliers and turned
        # a modest stem into a spike.
        want_off = min(r_nom * (0.6 + p.standoff), spacing * 0.75)

        cen_a, cen_b = surf_a.mean(0), surf_b.mean(0)

        # The symmetry axis is the line joining the two bodies - for a knob
        # that is the shifter axis, which is exactly what you want to spin
        # sectors around.
        sym = p.symmetry
        ax_pt = cen_b
        ax_dir = cen_a - cen_b
        if float(np.linalg.norm(ax_dir)) < 1e-9:
            ax_dir = np.array([0.0, 0.0, 1.0])

        sec_a, sec_b = surf_a, surf_b
        if sym != "NONE":
            ma = sector_mask(surf_a, ax_pt, ax_dir, sym, p.sym_count)
            mb = sector_mask(surf_b, ax_pt, ax_dir, sym, p.sym_count)
            if ma.sum() >= 3:
                sec_a = surf_a[ma]
            if mb.sum() >= 3:
                sec_b = surf_b[mb]
            interior = interior[sector_mask(interior, ax_pt, ax_dir, sym,
                                            p.sym_count)]
            if len(interior) < 4:
                self.report({"ERROR"}, "Too few nodes land in one sector. "
                                       "Raise Nodes or lower Sectors.")
                return {"CANCELLED"}

        raw_a, nrm_a, off_d_a, via_a = pick_anchors(
            p.obj_a, sec_a, cen_b, dom, want_off, p.anchors, p.seed)
        raw_b, nrm_b, off_d_b, via_b = pick_anchors(
            p.obj_b, sec_b, cen_a, dom, want_off, p.anchors, p.seed + 1)
        if via_a == -1 or via_b == -1:
            which = "A" if via_a == -1 else "B"
            extra = (" Narrowing by symmetry leaves even less, so try fewer "
                     "Sectors too." if sym != "NONE" else "")
            self.report({"ERROR"},
                        f"Body {which} has almost no surface inside the Design "
                        f"Space. Move or enlarge the box so it overlaps it."
                        + extra)
            return {"CANCELLED"}

        emb_a, emb_b, failed = [], [], 0
        for q, nv in zip(raw_a, nrm_a):
            pt, ok = bury_anchor(p.obj_a, q, nv, want_emb, r_anchor)
            emb_a.append(pt)
            failed += (not ok)
        for q, nv in zip(raw_b, nrm_b):
            pt, ok = bury_anchor(p.obj_b, q, nv, want_emb, r_anchor)
            emb_b.append(pt)
            failed += (not ok)
        emb_a = np.asarray(emb_a)
        emb_b = np.asarray(emb_b)

        off_a = raw_a + nrm_a * off_d_a[:, None]
        off_b = raw_b + nrm_b * off_d_b[:, None]

        na = len(raw_a)
        nb = len(raw_b)
        pts = np.vstack([emb_a, emb_b, off_a, off_b, interior])
        src = list(range(na))
        dst = list(range(na, na + nb))
        n_stem = na + nb                 # embedded anchors
        n_fixed = n_stem * 2             # + standoffs
        stems = [(i, n_stem + i) for i in range(n_stem)]

        max_len = p.max_len
        if max_len < spacing * 1.2:
            max_len = spacing * 2.2

        edges = build_knn_graph(pts, p.neighbours, max_len, skip=n_stem)
        edges = force_connect(pts, edges, range(n_stem, n_fixed), n_fixed, k=3)
        edges = drop_edges_through(pts, edges, preserve, protect=stems)
        edges = sorted(set(edges) | set(stems))

        pher = reinforced_load_paths(pts, edges, src, dst, rounds=p.rounds,
                                     bundling=p.bundling, evaporate=p.evaporate,
                                     wander=p.wander, seed=p.seed + 3)
        if not pher:
            self.report({"ERROR"}, "Bodies never connected. Raise Max Strut "
                                   "Length or Connectivity.")
            return {"CANCELLED"}

        vals = np.array(sorted(pher.values()))
        cut = vals[int(len(vals) * p.prune)] if p.prune > 0 else 0.0
        live = {e: v for e, v in pher.items() if v >= cut}
        for s in stems:                        # stems are never optional
            if s in pher:
                live[s] = pher[s]
        if len(live) < 3:
            live = pher

        live_edges, comp = anchored_component(sorted(live.keys()), src, dst)
        if not live_edges:
            self.report({"ERROR"}, "Network never spans both bodies. Raise "
                                   "Max Strut Length or lower Prune Weakest.")
            return {"CANCELLED"}
        orphans = sum(1 for i in src + dst if i not in comp)

        # Per-node load, resolved before replication so every sector inherits
        # identical thicknesses.
        v = np.array(list(live.values()))
        lo, hi = v.min(), v.max()
        spread = (hi - lo) if hi > lo else 1.0
        load = np.zeros(len(pts))
        for (a, b), c in live.items():
            nrm = (c - lo) / spread
            load[a] = max(load[a], nrm)
            load[b] = max(load[b], nrm)

        entry = set(range(n_stem))               # buried ends
        stemset = set(range(n_stem, n_fixed))    # standoffs
        fixed = set(range(n_fixed))
        sectors = 1

        if sym != "NONE":
            marker = np.zeros(len(pts))
            marker[list(entry)] = 1.0
            marker[list(stemset)] = 2.0
            pts, live_edges, packed, fixed = replicate_symmetry(
                pts, live_edges, np.column_stack([load, marker]), fixed,
                sym, p.sym_count, ax_pt, ax_dir, spacing * 0.05)
            load = packed[:, 0]
            entry = {i for i in range(len(pts)) if packed[i, 1] > 0.5
                     and packed[i, 1] < 1.5}
            stemset = {i for i in range(len(pts)) if packed[i, 1] >= 1.5}
            sectors = len(symmetry_transforms(sym, p.sym_count, ax_dir))
            web = set()

        # Webbing and seam stitching run after replication, so braces can tie
        # neighbouring sectors together rather than only bracing within one.
        if p.webbing > 0.0 or (sym != "NONE" and p.stitch > 0.0):
            cand = build_knn_graph(pts, p.neighbours, max_len)
            cand = drop_edges_through(pts, cand, preserve)
            live_edges, web = add_webbing(pts, cand, live_edges,
                                          set(range(len(pts))), p.webbing)
            live_edges, seam = stitch_seams(pts, cand, live_edges, ax_pt,
                                            ax_dir, sym, p.sym_count, p.stitch)
            web = web | seam

        pts = relax_nodes(pts, live_edges, fixed, p.relax, p.relax_str,
                          preserve, dom)
        movable = [i for i in range(len(pts)) if i not in fixed]
        pts = twist_nodes(pts, movable, cen_a, cen_b, p.twist)

        lengths = np.array([np.linalg.norm(pts[a] - pts[b])
                            for a, b in live_edges])
        med_len = float(np.median(lengths)) if len(lengths) else spacing

        if p.auto_thick:
            d_max = med_len * 0.50
            d_min = d_max * 0.45
        else:
            d_max, d_min = p.d_max, p.d_min
        if d_min > d_max:
            d_min, d_max = d_max, d_min
        if p.cast_floor > 0.0 and d_min < p.cast_floor:
            d_min = p.cast_floor
            d_max = max(d_max, d_min * 1.15)
        crowded = (d_max / med_len) > 0.66
        r_min, r_max = d_min * 0.5, d_max * 0.5

        radii = r_min + (r_max - r_min) * (load ** p.gamma)
        if stemset:
            radii[list(stemset)] = np.maximum(radii[list(stemset)], r_max * 0.9)
        if entry:
            radii[list(entry)] = r_max * p.flare   # trumpet into the body

        # Pin anchors and stems so the square-on entry survives; let everything
        # else relax into curves.
        pts, live_edges, radii = subdivide_and_curve(
            pts, live_edges, radii, set(range(n_fixed)),
            p.cuts, p.curve_iter, p.curve_str,
            thin_edges=web, r_thin=r_min * 0.8)

        obj = build_strut_object(
            context, pts, live_edges, radii, "LatticeBridge", p.output_mode,
            (d_min / 4.3) if p.voxel_auto else p.voxel,
            p.smooth_factor, p.smooth_iter, p.smooth)

        obj["lb_anchors"] = [float(x) for q in np.vstack([raw_a, raw_b])
                             for x in q]
        obj["lb_body_a"] = p.obj_a.name
        obj["lb_body_b"] = p.obj_b.name

        mm = context.scene.unit_settings.scale_length * 1000.0
        sym_txt = "" if sectors <= 1 else f" | {sectors}x sym"
        msg = (f"{len(live_edges)} struts ({len(web)} web){sym_txt} | "
               f"gap {med_len*mm:.1f}mm | dia {d_min*mm:.1f}-{d_max*mm:.1f}mm | "
               f"entry {d_max*p.flare*mm:.1f}mm buried {want_emb*mm:.1f}mm")
        lvl = {"INFO"}
        if failed:
            msg += (f" | {failed} anchors would not bury - is that body a "
                    f"closed solid? An open shell has no inside")
            lvl = {"WARNING"}
        if via_a == 0 or via_b == 0:
            which = "A" if via_a == 0 else "B"
            msg += (f" | Body {which} has no anchor sites facing the design "
                    f"space - move or enlarge the box toward it")
            lvl = {"WARNING"}
        if orphans:
            msg += f" | {orphans} anchors unreachable"
            lvl = {"WARNING"}
        if crowded:
            suggest = max(10, int(p.nodes * (0.60 / (d_max / med_len)) ** 3))
            msg += f" | TOO DENSE, drop Nodes to about {suggest}"
            lvl = {"WARNING"}
        self.report(lvl, msg)
        return {"FINISHED"}


# ======================================================================
# Fuse
# ======================================================================

class LB_OT_fuse(Operator):
    """Union the lattice with both bodies inside one voxel field, then smooth
    ONLY near the junctions. Voxel remesh is a signed distance field, so a
    localised smooth there behaves like a fillet - a true tangent blend instead
    of a boolean seam."""
    bl_idname = "mesh.lattice_bridge_fuse"
    bl_label = "Fuse to Bodies"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        p = context.scene.lattice_bridge
        latt = context.active_object
        if not latt or "lb_anchors" not in latt:
            self.report({"ERROR"}, "Select a generated LatticeBridge object")
            return {"CANCELLED"}
        a = bpy.data.objects.get(latt.get("lb_body_a", ""))
        b = bpy.data.objects.get(latt.get("lb_body_b", ""))
        if not a or not b:
            self.report({"ERROR"}, "Source bodies missing from the scene")
            return {"CANCELLED"}

        dg = context.evaluated_depsgraph_get()
        verts, faces = [], []
        for o in (latt, a, b):
            v, f = eval_mesh_world(o, dg)
            off = len(verts)
            verts.extend(v)
            faces.extend([tuple(i + off for i in poly) for poly in f])

        me = bpy.data.meshes.new("LatticeFused")
        me.from_pydata(verts, [], faces)
        me.update()
        tmp = bpy.data.objects.new("LB_tmp", me)
        context.collection.objects.link(tmp)

        rm = tmp.modifiers.new("Remesh", type="REMESH")
        rm.mode = "VOXEL"
        rm.voxel_size = max(p.voxel, 1e-5)

        dg = context.evaluated_depsgraph_get()
        fv, ff = eval_mesh_world(tmp, dg)
        bpy.data.objects.remove(tmp, do_unlink=True)

        me2 = bpy.data.meshes.new("LatticeFused")
        me2.from_pydata(fv, [], ff)
        me2.update()
        fused = bpy.data.objects.new("LatticeFused", me2)
        context.collection.objects.link(fused)

        # Weight the smooth by proximity to the junctions so the fillet forms
        # there and the dome keeps its clean curvature everywhere else.
        flat = list(latt["lb_anchors"])
        anchors = np.array(flat, dtype=np.float64).reshape(-1, 3)
        vg = fused.vertex_groups.new(name="junctions")
        co = np.array([v.co[:] for v in me2.vertices])
        d = np.min(np.linalg.norm(co[:, None, :] - anchors[None, :, :], axis=2),
                   axis=1)
        w = np.clip(1.0 - d / max(p.blend, 1e-9), 0.0, 1.0) ** 2
        for i, wi in enumerate(w):
            if wi > 0.001:
                vg.add([i], float(wi), "REPLACE")

        sm = fused.modifiers.new("BlendJunctions", type="SMOOTH")
        sm.factor = 1.0
        sm.iterations = p.blend_iter
        sm.vertex_group = vg.name

        if not p.keep_source:
            for o in (latt, a, b):
                bpy.data.objects.remove(o, do_unlink=True)

        for o in context.selected_objects:
            o.select_set(False)
        fused.select_set(True)
        context.view_layer.objects.active = fused
        for poly in me2.polygons:
            poly.use_smooth = True

        self.report({"INFO"}, f"Fused: {len(me2.vertices)} verts, "
                              f"{int((w > 0.001).sum())} verts in blend zone")
        return {"FINISHED"}


class LB_OT_preset_load(Operator):
    bl_idname = "mesh.lattice_bridge_preset_load"
    bl_label = "Load"
    bl_description = "Apply the selected preset"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        p = context.scene.lattice_bridge
        key = p.preset
        if key == "NONE":
            self.report({"ERROR"}, "No presets available")
            return {"CANCELLED"}
        if key.startswith("@"):
            apply_preset(p, BUILTIN_PRESETS[key[1:]])
            self.report({"INFO"}, f"Loaded '{key[1:]}'")
            return {"FINISHED"}
        path = os.path.join(preset_dir(), key + ".json")
        try:
            with open(path, "r") as fh:
                apply_preset(p, json.load(fh))
        except (OSError, ValueError) as e:
            self.report({"ERROR"}, f"Could not read preset: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Loaded '{key}'")
        return {"FINISHED"}


class LB_OT_preset_save(Operator):
    bl_idname = "mesh.lattice_bridge_preset_save"
    bl_label = "Save"
    bl_description = "Save the current settings under the given name"

    def execute(self, context):
        p = context.scene.lattice_bridge
        name = "".join(c for c in p.preset_name
                       if c.isalnum() or c in " _-").strip()
        if not name:
            self.report({"ERROR"}, "Give the preset a name")
            return {"CANCELLED"}
        data = {k: getattr(p, k) for k in PRESET_KEYS if hasattr(p, k)}
        try:
            with open(os.path.join(preset_dir(), name + ".json"), "w") as fh:
                json.dump(data, fh, indent=1)
        except OSError as e:
            self.report({"ERROR"}, f"Could not write preset: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"Saved '{name}'")
        return {"FINISHED"}


class LB_OT_preset_delete(Operator):
    bl_idname = "mesh.lattice_bridge_preset_delete"
    bl_label = "Delete"
    bl_description = "Delete the selected saved preset"

    def execute(self, context):
        p = context.scene.lattice_bridge
        if p.preset.startswith("@") or p.preset == "NONE":
            self.report({"ERROR"}, "Built-in presets cannot be deleted")
            return {"CANCELLED"}
        try:
            os.remove(os.path.join(preset_dir(), p.preset + ".json"))
        except OSError as e:
            self.report({"ERROR"}, f"Could not delete: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, "Deleted")
        return {"FINISHED"}


class LB_OT_make_domain(Operator):
    bl_idname = "mesh.lattice_bridge_domain"
    bl_label = "Add Design Space"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        p = context.scene.lattice_bridge
        if not p.obj_a or not p.obj_b:
            self.report({"ERROR"}, "Pick both bodies first")
            return {"CANCELLED"}
        lo, hi = combined_bounds([p.obj_a, p.obj_b], margin=0.02)
        centre = (lo + hi) / 2.0
        size = hi - lo
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=centre.tolist())
        c = context.active_object
        c.name = "LB_DesignSpace"
        c.scale = size.tolist()
        c.display_type = "WIRE"
        p.domain = c
        self.report({"INFO"}, "Design space added - hug it to the gap between "
                              "the bodies")
        return {"FINISHED"}


# ======================================================================
# UI
# ======================================================================

class LB_PT_panel(Panel):
    bl_label = "Lattice Bridge"
    bl_idname = "LB_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Lattice Bridge"

    def draw(self, context):
        lay = self.layout
        p = context.scene.lattice_bridge

        b = lay.box()
        b.label(text="Presets", icon="PRESET")
        r = b.row(align=True)
        r.prop(p, "preset", text="")
        r.operator("mesh.lattice_bridge_preset_load", text="", icon="IMPORT")
        r.operator("mesh.lattice_bridge_preset_delete", text="", icon="X")
        r = b.row(align=True)
        r.prop(p, "preset_name", text="")
        r.operator("mesh.lattice_bridge_preset_save", text="", icon="FILE_TICK")

        c = lay.column(align=True)
        c.label(text="Preserve Bodies", icon="MESH_UVSPHERE")
        c.prop(p, "obj_a")
        c.prop(p, "obj_b")
        c = lay.column(align=True)
        c.label(text="Design Space", icon="MESH_CUBE")
        c.prop(p, "domain", text="")
        c.operator("mesh.lattice_bridge_domain", icon="ADD")

        b = lay.box()
        b.label(text="Network", icon="MOD_SKIN")
        for k in ("nodes", "anchors", "neighbours", "max_len"):
            b.prop(p, k)

        b = lay.box()
        b.label(text="Load Bundling", icon="FORCE_CHARGE")
        for k in ("rounds", "bundling", "evaporate", "wander", "prune",
                  "webbing", "twist"):
            b.prop(p, k)

        b = lay.box()
        b.label(text="Symmetry", icon="MOD_MIRROR")
        b.prop(p, "symmetry", text="")
        r = b.row()
        r.enabled = p.symmetry in {"RADIAL", "DIHEDRAL"}
        r.prop(p, "sym_count")
        r = b.row()
        r.enabled = p.symmetry != "NONE"
        r.prop(p, "stitch")

        b = lay.box()
        b.label(text="Terminations", icon="SNAP_NORMAL")
        for k in ("embed", "standoff", "flare"):
            b.prop(p, k)

        b = lay.box()
        b.label(text="Cleanup", icon="MOD_SMOOTH")
        b.prop(p, "relax")
        b.prop(p, "relax_str")
        b.separator()
        b.prop(p, "cuts")
        b.prop(p, "curve_iter")
        b.prop(p, "curve_str")

        b = lay.box()
        b.label(text="Struts", icon="MOD_THICKNESS")
        b.prop(p, "auto_thick")
        sub = b.column(align=True)
        sub.enabled = not p.auto_thick
        sub.prop(p, "d_min")
        sub.prop(p, "d_max")
        b.prop(p, "gamma")
        b.prop(p, "cast_floor")

        b = lay.box()
        b.label(text="Output", icon="OUTLINER_OB_MESH")
        b.prop(p, "output_mode", text="")
        b.prop(p, "voxel_auto")
        r = b.row()
        r.enabled = not p.voxel_auto
        r.prop(p, "voxel")
        b.prop(p, "smooth_factor")
        b.prop(p, "smooth_iter")
        b.prop(p, "smooth")
        b.prop(p, "seed")

        lay.separator()
        lay.operator("mesh.lattice_bridge_generate", icon="OUTLINER_OB_MESH")

        b = lay.box()
        b.label(text="Tangent Blend", icon="MOD_BEVEL")
        b.prop(p, "blend")
        b.prop(p, "blend_iter")
        b.prop(p, "keep_source")
        b.operator("mesh.lattice_bridge_fuse", icon="MOD_BOOLEAN")


classes = (LB_Props, LB_OT_generate, LB_OT_fuse, LB_OT_make_domain,
           LB_OT_preset_load, LB_OT_preset_save, LB_OT_preset_delete,
           LB_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.lattice_bridge = PointerProperty(type=LB_Props)


def unregister():
    if hasattr(bpy.types.Scene, "lattice_bridge"):
        del bpy.types.Scene.lattice_bridge
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass


if __name__ == "__main__":
    try:
        unregister()
    except Exception:
        pass
    register()
