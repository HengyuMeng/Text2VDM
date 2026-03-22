from sd import StableDiffusion
from utils import *
from nvdiff_render.obj import *
from nvdiff_render.material import *
from nvdiff_render.texture import *
from nvdiff_render.render import *
from nvdiff_render.mesh import *
import time
import warnings
import torch.nn.functional as F
import argparse
import math
import random
import numpy as np
import torch
from tqdm import tqdm
import os
import pymeshlab
import trimesh
import pytorch3d
import nvdiffrast.torch as dr
import cv2
import numpy as np
import OpenEXR
import Imath
from resize_right import resize, cubic
from largesteps.parameterize import from_differential, to_differential
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

glctx = dr.RasterizeGLContext()
OBJAVERSE_PATH = './data'


def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--decay', type=float, default=0)  # weight decay
    parser.add_argument('--lr_decay', type=float, default=0.9)
    parser.add_argument('--lr_plateau', action='store_true')
    parser.add_argument('--decay_step', type=int, default=500)

    # input
    parser.add_argument('--output_path', type=str, default='./output')
    parser.add_argument('--res', type=int, default=512)
   
    # training
    parser.add_argument('--sd_max_grad_norm', type=float, default=20.0)
    parser.add_argument('--n_iter', type=int,
                        default=10001)  # can be increased
    parser.add_argument('--seed', type=int, default=201)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--sd_min', type=float, default=0.2)
    parser.add_argument('--sd_max', type=float, default=0.98)
    parser.add_argument('--sd_min_l', type=float, default=0.2)
    parser.add_argument('--sd_min_r', type=float, default=0.3)
    parser.add_argument('--sd_max_l', type=float, default=0.5)
    parser.add_argument('--sd_max_r', type=float, default=0.98)
    parser.add_argument('--bg', type=float, default=0.25)
    parser.add_argument('--logging', type=eval,
                        default=True, choices=[True, False])
    parser.add_argument('--n_view', type=int, default=4)
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--log_freq', type=int, default=100)
    parser.add_argument('--gd_scale', type=int, default=100)  # 100
    parser.add_argument('--uv_res', type=int, default=512)

    args = parser.parse_args()
    return args


def seed_all(args):
    # Constrain all sources of randomness
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def report_process(i, loss, exp_name):
    full_loss = 0
    log_message = f'[{exp_name}] iter: {i} '
    for loss_type, loss_val in loss.items():
        full_loss += loss_val
        log_message += f'{loss_type}: {"%.3f" % loss_val} '
    loss['L_all'] = full_loss
    print(log_message)


def compute_sd_step(min, max, iter_frac):
    step = (max - (max - min) * math.sqrt(iter_frac))
    return step


def load_obj_uv(obj_path, device):
    """
    ft(face.textures_idx): LongTensor of texture indices, shape (F, 3). This can be used to index into verts_uvs
    vt(aux.verts_uvs): the uv coordinate per vertex
    face.verts_idx: LongTensor of vertex indices, shape (F, 3)
    vert: FloatTensor of shape (V, 3) giving vertex positions
    """

    vert, face, aux = pytorch3d.io.load_obj(obj_path, device=device)
    vt = aux.verts_uvs
    ft = face.textures_idx
    vt = torch.cat((vt[:, [0]], 1.0 - vt[:, [1]]), dim=1)
    return ft, vt, face.verts_idx, vert


def initialize_mesh():
    obj_f_uv, obj_v_uv, obj_f, obj_v = load_obj_uv(
        obj_path='./base_mesh/plane.obj', device=device)
    mesh = Mesh(obj_v, obj_f, v_tex=obj_v_uv, t_tex_idx=obj_f_uv)
    mesh = unit_size(mesh)
    mesh = auto_normals(mesh)
    mesh = compute_tangents(mesh)
    return mesh


def create_mesh_from_vmap(vmap):
    """
    Create a mesh from vertex map to normalize the order of the vertices, aligned with the pixel arrangement
    advatage: 
        1. allow user using mask to control the generated region of mesh
        2. allow user using initial shape map to control the generated shape of mesh
        3. make the baked VDM align with mesh, convenient for later downstream applications
    """

    index = vmap.shape[0]

    vertices = vmap.reshape(-1, 3)
    faces = []
    for i in range(index-1):
        for j in range(index-1):
            idx1 = i * index + j
            idx2 = idx1 + 1
            idx3 = idx1 + index
            idx4 = idx3 + 1

            faces.append([idx1, idx2, idx3])
            faces.append([idx2, idx4, idx3])

    faces = np.array(faces)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    return mesh


def initialize_dense_mesh(args):

    corase_mesh = initialize_mesh()
    # Convert numpy arrays to torch tensors
    mesh_vertices = corase_mesh.v_pos
    mesh_faces = corase_mesh.t_pos_idx.type(torch.int32)
    mesh_faces_uv = corase_mesh.t_tex_idx.type(torch.int32)
    mesh_vert_uv = corase_mesh.v_tex
    normals = corase_mesh.v_nrm

    # Render UV coordinates to image space
    uv_coords = 2.0 * mesh_vert_uv - 1.0
    uv_coords_exp = torch.concat([uv_coords, torch.zeros_like(uv_coords[..., :1]),
                                  torch.ones_like(uv_coords[..., :1])], dim=-1).unsqueeze(0)  # to [u, v, 0, 1]
    rast_out, _ = dr.rasterize(
        glctx, uv_coords_exp, mesh_faces_uv, resolution=[int(args.res), int(args.res)])

    # get mask
    face_idx = rast_out[..., -1:]
    mask = torch.clamp(face_idx, 0, 1)
    mask = (mask.squeeze(-1) != 0)

    # get vmap (represent vertx per pixel)
    vmap, _ = dr.interpolate(mesh_vertices, rast_out, mesh_faces)
    normals, _ = dr.interpolate(normals, rast_out, mesh_faces)
    normals[normals.isnan()] = 0.0
    normals = torch.nn.functional.normalize(normals, p=2, dim=3)

    mesh = create_mesh_from_vmap(vmap[0].cpu().numpy())

    mesh.export('./dense_mesh/original_dense_mesh.obj')

    os.makedirs('./output/test/tmp', exist_ok=True)
    ms = pymeshlab.MeshSet()

    ms.load_new_mesh('./dense_mesh/original_dense_mesh.obj')

    if not ms.current_mesh().has_wedge_tex_coord():
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(
            textdim=10000)

    ms.save_current_mesh('./output/test/tmp/mesh.obj')

    load_mesh = obj.load_obj('./output/test/tmp/mesh.obj')
    load_mesh = unit_size(load_mesh)
    load_mesh = auto_normals(load_mesh)
    load_mesh = compute_tangents(load_mesh)

    texture_map = texture.create_trainable(np.random.uniform(
        size=[512] * 2 + [3], low=0.0, high=1.0), [512] * 2, True)
    normal_map = texture.create_trainable(
        np.array([0, 0, 1]), [512] * 2, True)
    specular_map = texture.create_trainable(
        np.array([0, 0, 0]), [512] * 2, True)

    material = Material({
        'bsdf': 'diffuse',
        'kd': texture_map,
        'ks': specular_map,
        'normal': normal_map,
    })
    load_mesh.material = material
    return load_mesh, vmap, normals, mask


def normal_render(mesh, cam):
    B = cam['mvp'].shape[0]
    v_clip = torch.bmm(F.pad(mesh.v_pos, pad=(0, 1), mode='constant', value=1.0).unsqueeze(0).expand(B, -1, -1),
                       torch.transpose(cam['mvp'], 1, 2)).float()  # [B, N, 4]

    res = [int(512), int(512)]
    mesh_faces = mesh.t_pos_idx.type(torch.int32)
    rast, _ = dr.rasterize(glctx, v_clip, mesh_faces, res)

    # Interpolate world space position
    alpha, _ = dr.interpolate(torch.ones_like(
        v_clip[..., :1]), rast, mesh_faces)  # [B, H, W, 1]

    vn, _ = compute_normal(v_clip[0, :, :3], mesh_faces)
    normal, _ = dr.interpolate(vn[None, ...].float(), rast, mesh_faces)
    normal = (normal + 1) / 2.
    normal = dr.antialias(normal, rast, v_clip,
                          mesh_faces).clamp(0, 1)  # [H, W, 3]
    alpha = dr.antialias(alpha, rast, v_clip,
                         mesh_faces).clamp(0, 1)
    obj_normal = (normal * alpha + (1 - alpha)
                  * args.bg).permute(0, 3, 1, 2)
    return obj_normal


def initialize_largestep(args, base_mesh, lambda_=15):

    # Compute the system matrix
    M = compute_matrix(base_mesh.v_pos, base_mesh.t_pos_idx,lambda_,cotan=True)  # [V, V]

    # Parameterize
    u = to_differential(M, base_mesh.v_pos)  # [V, 3]
    u.requires_grad = True
    initial_u = to_differential(M, base_mesh.v_pos)
    initial_u.requires_grad = False

    params = []
    params.append({
        'params': u,
        'lr': lr,
        'name': ['u']
    })
    optim = AdamUniform(params, args.learning_rate)
    activate_scheduler = args.lr_decay < 1 and args.decay_step > 0 and not args.lr_plateau
    if activate_scheduler:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=args.decay_step, gamma=args.lr_decay)
    return u, initial_u, M, optim, lr_scheduler


def initialize_exr_map(args):
    offset_exr_map = torch.zeros([1,args.res, args.res, 3], device=device)
    offset_exr_map.requires_grad = True
    optim_exr = torch.optim.Adam([offset_exr_map], 0.0001)
    return offset_exr_map, optim_exr
    

def generate_control_bar(mask_strength, n_iter):
    num_ones = int(mask_strength * n_iter)

    control_bar = torch.cat(
        (torch.ones(num_ones), torch.zeros(n_iter - num_ones))).int()

    return control_bar


def save_tensor_as_exr(tensor, filename):
    assert tensor.shape == (1, 3, 512, 512), "Tensor shape should be [1, 3, 512, 512]"

    tensor = tensor.squeeze(0)  # [3, 512, 512]

    tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
    scale = torch.tensor([-1.0], dtype=torch.float32)
    scale = scale.detach().cpu().numpy().astype(np.float32)
    R = tensor_np[0, :, :].tobytes()
    G = tensor_np[1, :, :].tobytes()
    B = tensor_np[2, :, :].tobytes()

    header = OpenEXR.Header(512, 512)
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                          'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                          'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}

    exr_file = OpenEXR.OutputFile(filename, header)
    exr_file.writePixels({'R': R, 'G': G, 'B': B})
    exr_file.close()


def bake_map_from_mesh(args, mesh_path):
    exp_name = time.strftime('%Y%m%d', time.localtime()) + '_' + args.exp_name
    output_dir = os.path.join('./logs/bake_exr')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dense_mesh, _, _, mask = initialize_dense_mesh(args)
    offset_exr_map,  optim = initialize_exr_map(args)
    path = mesh_path
    vert, face, _ = pytorch3d.io.load_obj(path, device=device)
    final_vertex = vert
    initial_vertex = dense_mesh.v_pos
    for step in tqdm(range(20001)):
        optim.zero_grad()

        new_initail_vertex = initial_vertex.clone().detach()
        new_final_vertex = final_vertex.clone().detach()

        offset = offset_exr_map[mask]

        predicted_vertex = new_initail_vertex + offset

        loss = F.mse_loss(predicted_vertex,new_final_vertex, reduction='mean')

        loss.backward()
        optim.step()

    save_tensor_as_exr(offset_exr_map.permute(0, 3, 1, 2), os.path.join(
        output_dir, '{}.exr'.format(args.identity)))
    
def main(args, guidance):
    exp_name = time.strftime('%Y%m%d', time.localtime()) + '_' + args.exp_name
    output_dir = os.path.join('./logs', exp_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    seed_all(args)
   
    # Get text prompt and tokenize it
    sd_prompt_normal = ", ".join(
        (f"a DSLR normal map of a {args.identity}", "3d model, best quality, high quality, extremely detailed, good geometry"))
    
    # load obj and read uv information
    dense_mesh, _, _, mask = initialize_dense_mesh(args)

    # initialize the direction and volume of the VDM
    if use_shape_map:
        shape_map_t = (torch.tensor(shape_map, device=device) /
                       255.0).permute(2, 0, 1).unsqueeze(0).permute(0, 2, 3, 1)
        shape_vertex = shape_map_t[mask]
        dense_mesh.v_pos += shape_vertex*shape_strength

    # initialize the largestep method
    u, initial_u, M, optim, lr_scheduler = initialize_largestep(
        args, dense_mesh, lambda_)

    # get text embedding
    text_z_normal = []
    for d in ['front', 'side', 'back', 'overhead']:
        # construct dir-encoded text
        text_z_normal.append(guidance.get_text_embeds(
            [f"{sd_prompt_normal}"], [neg_prompt], 1))
   
    text_z_normal = torch.stack(text_z_normal, dim=0) # -> [4, 2, S (77), 1024]

    # Main training loop
    for step in tqdm(range(args.n_iter + 1)):
        cur_iter_frac = step / args.n_iter
        losses = {}
        optim.zero_grad()

        base_mask_t = (torch.tensor(base_mask, device=device) /
                       255.0).permute(2, 0, 1).unsqueeze(0)
        base_mask_geo = resize(base_mask_t, out_shape=(
            args.res, args.res), interp_method=cubic).permute(0, 2, 3, 1)
        
        # ================ vertex prediction ================
        vert_mask = base_mask_geo[mask]
        updated_u = u-initial_u
        control_bar = generate_control_bar(mask_strength, args.n_iter+1)
        use_mask = control_bar[step]
        if use_mask:
            final_u = initial_u + updated_u * vert_mask
        else:
            final_u = initial_u + updated_u

        new_vert = from_differential(M,  final_u, 'Cholesky')

        mesh = Mesh(
            new_vert,
            dense_mesh.t_pos_idx,
            base=dense_mesh 
        )

        # ================ rendering ================
        cam = sample_view_obj(args.n_view)
        obj_normal = normal_render(mesh, cam)

        # =================== SDS loss ==================
        text_z_iter_normal = text_z_normal[cam['direction']]
        all_pos_normal, all_neg_normal = [], []
        for emb_normal in text_z_iter_normal:
            pos, neg = emb_normal.chunk(2)
            all_pos_normal.append(pos)
            all_neg_normal.append(neg)
        text_embedding_normal = torch.cat(
            all_pos_normal + all_neg_normal, dim=0)

        sd_min_step = compute_sd_step(
            args.sd_min_l, args.sd_min_r, cur_iter_frac)
        sd_max_step = compute_sd_step(
            args.sd_max_l, args.sd_max_r, cur_iter_frac)

        sd_loss_normal = guidance.batch_train_step(text_embedding_normal, obj_normal,
                                                   guidance_scale=gd_scale,
                                                   min_step=sd_min_step,
                                                   max_step=sd_max_step,)

        total_loss = sd_loss_normal

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            u, max_norm=args.sd_max_grad_norm)
        optim.step()

        lr_scheduler.step()

        # logging mesh
        if step % 2500 == 0:
            pytorch3d.io.save_obj(
                os.path.join(output_dir, f'mesh_{step:04}.obj'), verts=mesh.v_pos, faces=mesh.t_pos_idx)
            
        if step % args.log_freq == 0 and args.logging:
            with torch.no_grad():
                report_process(step, losses, exp_name)
                torchvision.utils.save_image(
                    obj_normal, os.path.join(output_dir, f'normal_{step:04}.jpg'))
                
    pytorch3d.io.save_obj(
                os.path.join(output_dir, f'mesh_{step:04}.obj'), verts=mesh.v_pos, faces=mesh.t_pos_idx)
    bake_map_from_mesh(args, os.path.join(output_dir, f'mesh_{step:04}.obj'))



if __name__ == '__main__':
    args = parse_args()

    mesh_dicts = {'1': 'a dragon horn+++',}
    neg_prompt = 'worst quality, low quality, bad geometry'
    sd_version = '2.1'
    guidance = StableDiffusion(
        device, sd_version=sd_version, min=args.sd_min, max=args.sd_max)
    
    guidance.eval()
    for p in guidance.parameters():
        p.requires_grad = False
    lr = 0.005
    lambda_ = 15
    mask_strength = 1 # set to 1 for geometric structure VDM generation, set other values [0,1] for surface detail VDM generation
    use_shape_map = True
    shape_strength = 0.8 # control the strength of displacement of vertices for shape map
    gd_scale = 100

    # Load the image control
    path = 'horn'
    base_mask = cv2.imread('./shape_map/'+ path + '_mask.png')
    shape_map = cv2.imread('./shape_map/'+ path + '.png')

    # iterate through the renderpeople items
    for obj_id, caption in mesh_dicts.items():
        args.exp_name = '_'.join((caption.split(' ')[
                                 :])+['lambda{}_mask{}_shape{}_sd{}_gds{}_lr{}_path_{}'.format(lambda_, mask_strength, shape_strength, sd_version, gd_scale, lr, path)])
        args.identity = caption
        main(args, guidance)
        bake_map_from_mesh(args)
