import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

def edge_list(faces):
    """
    Extract the list of edges for the icosahedron and the one-ring neighborhood of each vertex.

    """
    # first extract all triangle edges
    el =[]
    for face in faces:
        a, b, c = face
        el.append([min(a,b), max(a,b)])
        el.append([min(a,c), max(a,c)])
        el.append([min(b,c), max(b,c)])
    # remove duplicates from the list
    uel = []
    for edge in el:
        if edge not in uel:
            uel.append(edge)
    uel_rev = []
    for edge in uel:
        uel_rev.append([edge[1], edge[0]])
    all_edges = [*uel, *uel_rev]
    res = np.unique(all_edges, axis=0)
    res = np.split(res, len(all_edges)/5)
    res_ring = []
    for r in res:
        ring = []
        inds = r[:, 1]
        ring.append(inds[0])
        while len(ring) < len(inds):
            for i in inds:
                if [ring[-1], i] in all_edges and i not in ring:
                    ring.append(i)
        curr_ring = np.stack([r[:, 0], np.array(ring)]).T
        res_ring.append(curr_ring)
    #print(len(res))
    return res_ring

def check_counterclockwise(ring, vertices):
    """
    Function to make sure the one-ring neighborhood of each icosahedron vertex
    is ordered counterclockwise.
    """

    vertices = vertices.numpy()
    new_rings = []
    for r in ring:
        inds = r[:, 1]
        inds_rot = np.append(inds[1:], inds[0])
        vecs = vertices[inds_rot, :] - vertices[inds, :]
        vecs_rot = np.append(vecs[1:, :], vecs[:1, :], axis=0)
        norm = vertices[r[0][0]][..., np.newaxis]
        cross_prod = np.cross(vecs, vecs_rot)
        diff = np.squeeze(np.dot(cross_prod, norm))
        assert len(np.unique(np.sign(diff))) == 1
        if not np.all(diff < 0):
            print("Clockwise ring, making it counterclockwise.")
            new_r = np.stack([r[:, 0], inds[::-1]]).T
            new_rings.append(new_r)
        else:
            new_rings.append(r)
    return new_rings

def get_icosahedron_aligned():
    """
    Function to get the icosahedron vertices, the one-ring neighborhood of each vertex,
    and connectivity of the vertices.
    """

    phi = (1.0 + np.sqrt(5.0))/2.0
    # factor that ensure that vertices are norm 1 vectors
    f = 1.0 / (np.sqrt(phi * np.sqrt(5.0)))
    fphi = f * phi
    # The 12 vertices of the icosahedron
    vertices = [
        [ 0.0,  f,  fphi], [ f,  fphi,  0.0], [-f,  fphi,  0.0],
        [-fphi,  0.0,  f], [ 0.0, -f,  fphi], [ fphi,  0.0,  f],
        [ 0.0, -f, -fphi], [ f, -fphi,  0.0], [-f, -fphi,  0.0],
        [-fphi,  0.0, -f], [ 0.0,  f, -fphi], [ fphi,  0.0, -f]
    ]

    vertices = torch.from_numpy(np.array(vertices))
    faces = [
            (0,2,1), (0,2,3), (0,3,4), (0,4,5), (0,1,5),
            (1,10,11), (2,1,10), (2,9,10), (3,2,9), (3,8,9),
            (3,4,8), (4,7,8), (4,5,7), (5,7,11), (5,11,1),
            (6,7,8), (6,8,9),(6,9,10), (6,10,11), (6,7,11)
        ]
    
    edges_ring = edge_list(faces)
    edges_ring = check_counterclockwise(edges_ring, vertices)
    return vertices, edges_ring, np.array(faces)

def get_icosahedron_dirs(vertices, edges, eps):
    """
    Function to get ray directions for each icosahedron vertex.
    The directions are aligned with the edges bewteen the vertex of interest and its neighbors
    """

    all_edges = np.concatenate(edges)
    coords = torch.stack(torch.split(vertices[all_edges[:, 1]], len(edges[0]), dim=0))
    dirs = logmap(vertices, coords, eps)
    return dirs, coords

def dist(x, y, eps):
    """
    x: N x 1 x 3
    y: N x M x 3
    """
    mm_clip = lambda k, l, u: torch.max(l, torch.min(u, k))
    inner = torch.sum(x*y, dim=-1, keepdim=True)    # N x M x 1
    inner = torch.clamp(inner, min=-1+eps, max=1-eps)
    return torch.acos(inner)    # arc length

def proju(x, u):
    """
    x: N x 1 x 3
    u: N x M x 3
    """
    u = u - (x * u).sum(dim=-1, keepdim=True) * x#.transpose(0, 2, 1)
    return u
    
def logmap(x, y, eps):
    """
    Log map

    # param x: N x 3
    # param y: N x M x 3
    """
    x = x.unsqueeze(-1).permute(0, 2, 1)
    u = proju(x, y - x)    # N x M x 3
    dists = dist(x, y, eps)    # N x M x 1
    N, M, _ = dists.shape
    cond = dists > eps
    u_norm = torch.clamp(torch.norm(u, dim=-1, keepdim=True), min=eps)    # N x M x 1
    result = torch.where(
        cond, u * dists / u_norm, u
    )
    return result

def expmap(x, v, eps=1e-7):
    """
    Exponential map 

    # param x: N x 3
    # param v: N x M x 3
    """
    x = x.unsqueeze(-1).permute(0, 2, 1)     # N x 1 x 3
    norm_v = v.norm(dim=-1, keepdim=True)    # N x M x 1
    exp = x * torch.cos(norm_v) + v * torch.sin(norm_v) / norm_v
    retr = x + v
    cond = norm_v > eps
    return torch.where(cond, exp, retr)

def sample_tangent_points(all_directions, points_per_ray, max_len):
    """
    Function to sample points along the rays of each vertex.
    """

    if max_len is None:
        scales = torch.linspace(0, 1, points_per_ray+1, dtype=torch.float64)[1:-1]
    else:
        scales = torch.linspace(0, max_len, points_per_ray+1, dtype=torch.float64)[1:]
    
    n, p, d = all_directions.shape    # 12, 5, 3

    all_points = torch.einsum("npd,s->npds", (all_directions, scales))
    all_points = all_points.permute(0, 2, 1, 3).reshape(n, d, -1)
    all_points = all_points.permute(0, 2, 1)
    return all_points

def get_sphere_points(vertices, edges_ring, sample_per_ray, ray_len, thres=1e-7):
    """
    Function to get the coordinates of points on the sphere around icosahedron vertices 
    to interpolate the function.
    """

    # get the directions of kernel rays at each icosahedron tangent plane from icosahedron edges
    ico_dirs, end_v = get_icosahedron_dirs(vertices, edges_ring, thres)
    if not ray_len is None:
        ico_dirs = ico_dirs / ico_dirs.norm(dim=-1, keepdim=True)

    # sample points along the rays on tangent planes
    sampled_tangent = sample_tangent_points(ico_dirs, sample_per_ray, ray_len)

    # map the rays to the sphere using exponential map
    sphere_coords = expmap(vertices, sampled_tangent, thres)    # 12 x 5 x 3
    if ray_len is None:
        sphere_coords = torch.stack([sphere_coords, end_v], dim=2)
        v, r, s, d = sphere_coords.shape
        sphere_coords = sphere_coords.permute(0, 3, 1, 2).reshape(v, d, -1).permute(0, 2, 1) # 12 x 10 x 3

    # add each icosahedron vertex to its ring-structured patch
    sphere_coords = torch.cat([vertices.unsqueeze(1), sphere_coords], dim=1)
    return sphere_coords

def construct_quaternion(axis, theta):
    """
    Function to construct quaternions from rotation axes and angles.

    # param axis: The 12 vertices of an icosahedron, dimension 12 x 3
    # param theta: The rotation angles, dimension 5
    """
    axis = np.einsum("kd,s->ksd", axis, np.sin(theta / 2)) # 12 x 5 x 3
    theta = np.cos(theta / 2)  # 5,
    theta = np.tile(theta, (len(axis), 1))[..., np.newaxis] # 12 x 5 x 1
    quat = np.concatenate([axis, theta], axis=-1)  # 12 x 5 x 4
    return quat

def compute_rotations(SO2=True):
    """
    Function to compute aligned rotations (no DOF) between the first icosahedron vertex to others.

    Quatertions are used to paraterize the rotations.
    """
    vertices, edge_rings, _ = get_icosahedron_aligned()


    rotation_matrices = [R.from_quat(np.array([0,0,0,1]))]

    # Compute the rotations between the ring of the first vertex and the rings of other vertices
    ring_first = edge_rings[0][:, 1]
    ring_coords_first = vertices[ring_first]
    for i in range(1, len(vertices)):
        ring = edge_rings[i][:, 1]
        ring_coords = vertices[ring]
        r, err = R.align_vectors(ring_coords, ring_coords_first)
        rotation_matrices.append(r)

    # Compute all the SO(2) rotations at each vertex location
    if SO2:
        scales = np.linspace(0, np.pi * 2, len(edge_rings[0]) + 1)[:-1]
        quaternions = construct_quaternion(vertices.numpy(), scales)
        new_rotations = []
        for i in range(len(quaternions)):
            global_rotation = rotation_matrices[i]
            local_rotations = quaternions[i]
            local_rots = []
            for j in range(len(local_rotations)):
                local_rotation = R.from_quat(local_rotations[j])
                local_rots.append((local_rotation * global_rotation).as_matrix())
            local_rots = np.stack(local_rots)
            new_rotations.append(local_rots)
        rotation_matrices = np.stack(new_rotations)

        inv_rotations = []
        for local_rots in rotation_matrices:
            inv_locals = []
            for rot in local_rots:
                inv_rot = np.linalg.inv(rot)
                inv_locals.append(inv_rot)
            inv_locals = np.stack(inv_locals)
            inv_rotations.append(inv_locals)
        inv_rotations = np.stack(inv_rotations)

    else:
        for i in range(len(rotation_matrices)):
            rotation_matrices[i] = rotation_matrices[i].as_matrix()
        rotation_matrices = np.stack(rotation_matrices)

        inv_rotations = []
        for rot in rotation_matrices:
            inv_rot = np.linalg.inv(rot)
            inv_rotations.append(inv_rot)
        inv_rotations = np.stack(inv_rotations)
    return torch.from_numpy(rotation_matrices), torch.from_numpy(inv_rotations)




def compute_rotations_rotvec(SO2=True):
    """
    Function to compute aligned rotations (no DOF) between the first icosahedron vertex to others.

    Parameterized by rotation axes and angles. Equivalent to compute_rotations()
    """
    vertices, edge_rings, _ = get_icosahedron_aligned()
    rotation_matrices = [np.eye(3)]
    ring_first = edge_rings[0][:, 1]
    ring_coords_first = vertices[ring_first].numpy()
    for i in range(1, len(vertices)):
        v = vertices[i]
        ring = edge_rings[i][:, 1]
        ring_coords = vertices[ring].numpy()
        r, err = R.align_vectors(ring_coords, ring_coords_first)
        print(err)
        r = r.as_matrix()
        rotation_matrices.append(r)
    rotation_matrices = torch.tensor(np.stack(rotation_matrices))

    if SO2:
        scales = torch.linspace(0, np.pi*2, len(edge_rings[0])+1)[:-1]
        rot_vecs = torch.einsum("kd,s->ksd", (vertices, scales)).reshape(-1, 3)
        r_mats = R.from_rotvec(rot_vecs.numpy()).as_matrix()
        num_rot, mat_dim1, mat_dim2 = r_mats.shape
        assert num_rot==len(vertices)*len(edge_rings[0])
        r_mats = r_mats.reshape(len(vertices), len(edge_rings[0]), mat_dim1, mat_dim2)
        rotation_matrices = torch.matmul(torch.tensor(r_mats), torch.tensor(rotation_matrices).unsqueeze(1))
    return rotation_matrices