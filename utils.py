from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
import torch
import numpy as np
import pickle as pkl
from pathlib import Path
import subprocess
import random
from nvdiff_render import util
import torchvision
import torchvision.transforms as T

downsampler_512 = T.Resize((512, 512))
tensor_to_img = T.ToPILImage()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_view_obj(n_view, cam_radius=2.8, min=-10,max=-60,hori_min=0,hori_max=2,res=[512, 512], cam_near_far=[0.1, 1000.0], spp=1, is_face=False):
    iter_res = res
    fovy = np.deg2rad(45)
    proj_mtx = util.perspective(
        fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])

    # Random rotation/translation matrix for optimization.
    mv_list, mvp_list, campos_list, direction_list = [], [], [], []
    for view_i in range(n_view):
        angle_x = np.deg2rad(np.random.uniform(min, max)
                             )  # Constrained elevation -50 -70
        angle_y = np.random.uniform(hori_min*np.pi, hori_max * np.pi)  # Full azimuth
        # direction
        # 0 = front, 1 = side, 2 = back, 3 = overhead
        if angle_x is not None:
            direction = 3
        else:
            if 0 <= angle_y <= np.pi / 4 or angle_y > 7 * np.pi / 4:
                direction = 0
            elif np.pi / 4 < angle_y <= 3 * np.pi / 4:
                direction = 1
            elif 3 * np.pi / 4 < angle_y <= 5 * np.pi / 4:
                direction = 2
            elif 5 * np.pi / 4 < angle_y <= 7 * np.pi / 4:
                direction = 1
        mv = util.translate(
            0, 0, -cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mvp = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        mv_list.append(mv[None, ...].cuda())
        mvp_list.append(mvp[None, ...].cuda())
        campos_list.append(campos[None, ...].cuda())
        direction_list.append(direction)

    cam = {
        'mv': torch.cat(mv_list, dim=0),
        'mvp': torch.cat(mvp_list, dim=0),
        'campos': torch.cat(campos_list, dim=0),
        'direction': np.array(direction_list, dtype=np.int32),
        'resolution': iter_res,
        'spp': spp
    }
    return cam


def sample_view_obj3(n_view, cam_radius, res=[512, 512], cam_near_far=[0.1, 1000.0], spp=1, is_face=False):
    iter_res = res
    fovy = np.deg2rad(45)
    proj_mtx = util.perspective(
        fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])

    # Random rotation/translation matrix for optimization.
    mv_list, mvp_list, campos_list, direction_list = [], [], [], []
    for view_i in range(n_view):
        # if view_i == 0:
        #     angle_x = 0.0  # elevation
        #     angle_y = 0.0  # azimuth
        # else:
        #     angle_x = np.deg2rad(np.random.uniform(-10, -20)
        #                          )  # Constrained elevation
        #     angle_y = np.random.uniform(0, 2 * np.pi)  # Full azimuth
        angle_x = np.deg2rad(np.random.uniform(-10, -30)
                             )  # Constrained elevation -50 -70
        angle_y = np.random.uniform(0, 2 * np.pi)  # Full azimuth
        # direction
        # 0 = front, 1 = side, 2 = back, 3 = overhead
        if angle_x < -np.pi / 4:
            direction = 3
        else:
            if 0 <= angle_y <= np.pi / 4 or angle_y > 7 * np.pi / 4:
                direction = 0
            elif np.pi / 4 < angle_y <= 3 * np.pi / 4:
                direction = 1
            elif 3 * np.pi / 4 < angle_y <= 5 * np.pi / 4:
                direction = 2
            elif 5 * np.pi / 4 < angle_y <= 7 * np.pi / 4:
                direction = 1

        mv = util.translate(
            0, 0, -cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mvp = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        mv_list.append(mv[None, ...].cuda())
        mvp_list.append(mvp[None, ...].cuda())
        campos_list.append(campos[None, ...].cuda())
        direction_list.append(direction)

    cam = {
        'mv': torch.cat(mv_list, dim=0),
        'mvp': torch.cat(mvp_list, dim=0),
        'campos': torch.cat(campos_list, dim=0),
        'direction': np.array(direction_list, dtype=np.int32),
        'resolution': iter_res,
        'spp': spp
    }
    return cam


def sample_circle_view(n_view, elev, cam_radius, res=[512, 512], cam_near_far=[0.1, 1000.0], spp=1):
    iter_res = res
    fovy = np.deg2rad(45)
    proj_mtx = util.perspective(
        fovy, iter_res[1] / iter_res[0], cam_near_far[0], cam_near_far[1])

    # Random rotation/translation matrix for optimization.
    mv_list, mvp_list, campos_list, direction_list = [], [], [], []
    angles_y = np.linspace(0.0, 2 * np.pi, n_view)
    for view_i in range(n_view):
        angle_x = elev
        angle_y = angles_y[view_i]
        # direction
        # 0 = front, 1 = side, 2 = back, 3 = side
        if angle_y >= 0 and angle_y <= np.pi / 8 or angle_y > 15 * np.pi / 8:
            direction = 0
        elif angle_y > np.pi / 8 and angle_y <= 7 * np.pi / 8:
            direction = 1
        elif angle_y > 7 * np.pi / 8 and angle_y <= 9 * np.pi / 8:
            direction = 2
        elif angle_y > 9 * np.pi / 8 and angle_y <= 15 * np.pi / 8:
            direction = 3

        mv = util.translate(
            0, 0, -cam_radius) @ (util.rotate_x(angle_x) @ util.rotate_y(angle_y))
        mvp = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]
        mv_list.append(mv[None, ...].cuda())
        mvp_list.append(mvp[None, ...].cuda())
        campos_list.append(campos[None, ...].cuda())
        direction_list.append(direction)

    cam = {
        'mv': torch.cat(mv_list, dim=0),
        'mvp': torch.cat(mvp_list, dim=0),
        'campos': torch.cat(campos_list, dim=0),
        'direction': np.array(direction_list, dtype=np.int32),
        'resolution': iter_res,
        'spp': spp
    }
    return cam


def create_video(img_path, out_path, fps=60):
    '''
    Creates a video from the frame format in the given directory and saves to out_path.
    '''
    command = ['/usr/bin/ffmpeg', '-y', '-r', str(fps), '-i', img_path,
               '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', out_path]
    subprocess.run(command)


def imgcat(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0).squeeze()
        x = x.detach().cpu().numpy()

    print(f'[imgcat] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')

    x = x.astype(np.float32)

    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / \
            (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


#


def load_obj_uv(obj_path, device):
    vert, face, aux = load_obj(obj_path, device=device)
    vt = aux.verts_uvs
    ft = face.textures_idx
    vt = torch.cat((vt[:, [0]], 1.0 - vt[:, [1]]), dim=1)
    return ft, vt, face.verts_idx, vert


# e.g. main_mask = head_mask, sub_mask = face_mask
def get_submasked_faces(main_mask, sub_mask, is_index_mask=False, weight=1.0):
    main_mask_len = main_mask.shape[0]
    faces_mask = np.all(np.isin(main_mask, sub_mask), axis=1)
    regional_main_mask = faces_mask.astype(float).reshape(
        main_mask_len, 1) * weight  # assume the same shape
    if is_index_mask is True:
        # shape [main_mask_len, 1] for face mask with weights only
        return regional_main_mask
    # repeat to fit the faces mask original shape [3931, 3] or [15622, 3]
    else:
        # shape [main_mask_len, 3] for face mask with weights only
        return np.repeat(regional_main_mask, 3, axis=1)


cosine_sim = torch.nn.CosineSimilarity()


def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()


def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def compute_normal(vertices, faces):
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.as_tensor(vertices).float()
    if not isinstance(faces, torch.Tensor):
        faces = torch.as_tensor(faces).long()

    i0, i1, i2 = faces[:, 0].long(), faces[:, 1].long(), faces[:, 2].long()

    v0, v1, v2 = vertices[i0, :], vertices[i1, :], vertices[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    vn = torch.zeros_like(vertices)
    vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    vn = torch.where(dot(vn, vn) > 1e-20, vn,
                     torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))
    vn = safe_normalize(vn)

    face_normals = safe_normalize(face_normals)
    return vn, faces
