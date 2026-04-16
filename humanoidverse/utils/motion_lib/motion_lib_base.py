import glob
import json
import os.path as osp
from collections.abc import Sequence
from collections import defaultdict
import numpy as np
import joblib
import torch
import torch.multiprocessing as mp
import random

from enum import Enum
from humanoidverse.merge_npz_to_pkl import EXPECTED_FIELD_SET, collect_keys, discover_npz_files
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from pathlib import Path
from easydict import EasyDict
from loguru import logger
from omegaconf import ListConfig
from rich.progress import track
from scipy.spatial.transform import Rotation as sRot
import gc

from humanoidverse.utils.torch_utils import(
    quat_angle_axis,
    quat_inverse,
    quat_mul_norm,
    get_euler_xyz,
    normalize_angle,
    slerp,
    quat_to_exp_map,
    quat_to_angle_axis,
    quat_mul,
    quat_conjugate,
    calc_heading_quat_inv,
    my_quat_rotate,
)

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2

class MotionlibMode(Enum):
    file = 1
    directory = 2
    npz = 3


NPZ_EXPECTED_NUM_DOFS = 29
NPZ_EXPECTED_NUM_BODIES = 30


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


def _fps_to_int(fps_value) -> int:
    fps_array = np.asarray(fps_value)
    if fps_array.size != 1:
        raise ValueError(f"fps must contain exactly one value, got shape {fps_array.shape}")
    fps = int(fps_array.reshape(-1)[0])
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return fps


def _is_motion_root_collection(value) -> bool:
    return isinstance(value, (list, tuple, ListConfig)) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path))
    )


def _normalize_motion_roots(input_root: Path | str | list[str] | tuple[str, ...]) -> list[Path]:
    if isinstance(input_root, (str, Path)):
        roots = [input_root]
    elif _is_motion_root_collection(input_root):
        roots = list(input_root)
    else:
        raise TypeError(f"Unsupported motion root type: {type(input_root)!r}")

    if len(roots) == 0:
        raise ValueError("motion_root must contain at least one directory")

    resolved_roots: list[Path] = []
    for root in roots:
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Motion root does not exist: {root_path}")
        if not root_path.is_dir():
            raise ValueError(f"Motion root must be a directory when scanning npz motions: {root_path}")
        resolved_roots.append(root_path)
    return resolved_roots


def _build_motion_root_prefixes(input_roots: list[Path]) -> dict[Path, str]:
    if len(input_roots) <= 1:
        return {root: "" for root in input_roots}

    basename_counts: dict[str, int] = defaultdict(int)
    for root in input_roots:
        basename = root.name or root.as_posix().strip("/").replace("/", "_")
        basename_counts[basename] += 1

    prefix_counts: dict[str, int] = defaultdict(int)
    prefixes: dict[Path, str] = {}
    for root in input_roots:
        basename = root.name or root.as_posix().strip("/").replace("/", "_")
        prefix_base = basename if basename_counts[basename] == 1 else root.as_posix().strip("/").replace("/", "_")
        prefix_counts[prefix_base] += 1
        prefix = prefix_base if prefix_counts[prefix_base] == 1 else f"{prefix_base}#{prefix_counts[prefix_base]}"
        prefixes[root] = prefix
    return prefixes


def validate_isaaclab_npz_motion(
    npz_path: Path,
    expected_num_dofs: int = NPZ_EXPECTED_NUM_DOFS,
    expected_num_bodies: int = NPZ_EXPECTED_NUM_BODIES,
) -> tuple[bool, str | None]:
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            actual_fields = frozenset(data.files)
            if actual_fields != EXPECTED_FIELD_SET:
                raise ValueError(
                    f"unexpected fields: expected {sorted(EXPECTED_FIELD_SET)}, got {sorted(actual_fields)}"
                )

            fps = _fps_to_int(data["fps"])
            joint_pos = data["joint_pos"]
            joint_vel = data["joint_vel"]
            body_pos = data["body_pos_w"]
            body_quat = data["body_quat_w"]
            body_lin_vel = data["body_lin_vel_w"]
            body_ang_vel = data["body_ang_vel_w"]

            if joint_pos.ndim != 2 or joint_pos.shape[1] != expected_num_dofs:
                raise ValueError(
                    f"joint_pos must have shape (T, {expected_num_dofs}), got {joint_pos.shape}"
                )
            if joint_vel.shape != joint_pos.shape:
                raise ValueError(f"joint_vel must match joint_pos shape {joint_pos.shape}, got {joint_vel.shape}")

            expected_vec_shape = (joint_pos.shape[0], expected_num_bodies, 3)
            expected_quat_shape = (joint_pos.shape[0], expected_num_bodies, 4)
            if body_pos.shape != expected_vec_shape:
                raise ValueError(f"body_pos_w must have shape {expected_vec_shape}, got {body_pos.shape}")
            if body_lin_vel.shape != expected_vec_shape:
                raise ValueError(f"body_lin_vel_w must have shape {expected_vec_shape}, got {body_lin_vel.shape}")
            if body_ang_vel.shape != expected_vec_shape:
                raise ValueError(f"body_ang_vel_w must have shape {expected_vec_shape}, got {body_ang_vel.shape}")
            if body_quat.shape != expected_quat_shape:
                raise ValueError(f"body_quat_w must have shape {expected_quat_shape}, got {body_quat.shape}")
            if joint_pos.shape[0] < 2:
                raise ValueError("motion must contain at least 2 frames")
            _ = fps
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)
    return True, None


def discover_valid_isaaclab_npz_motions(
    input_root: Path | str | list[str] | tuple[str, ...],
    expected_num_dofs: int = NPZ_EXPECTED_NUM_DOFS,
    expected_num_bodies: int = NPZ_EXPECTED_NUM_BODIES,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, str]]]:
    input_roots = _normalize_motion_roots(input_root)
    root_prefixes = _build_motion_root_prefixes(input_roots)

    valid_paths: list[Path] = []
    ordered_keys: list[str] = []
    skipped_files: list[dict[str, str]] = []
    for input_root in input_roots:
        npz_paths, directory_npz_counts = discover_npz_files(input_root=input_root)
        print(f"Discovered {len(npz_paths)} .npz files under {input_root} (including {sum(directory_npz_counts.values())} in directories)")
        root_valid_paths: list[Path] = []
        for i, npz_path in enumerate(npz_paths):
            if i % 1000 == 0:
                print(i)
            is_valid, error_message = validate_isaaclab_npz_motion(
                npz_path=npz_path,
                expected_num_dofs=expected_num_dofs,
                expected_num_bodies=expected_num_bodies,
            )
            if is_valid:
                root_valid_paths.append(npz_path)
            else:
                skipped_files.append({"path": str(npz_path.resolve()), "reason": error_message or "unknown validation error"})

        if not root_valid_paths:
            continue

        keys_by_path = collect_keys(
            npz_paths=root_valid_paths,
            input_root=input_root,
            directory_npz_counts=directory_npz_counts,
        )
        root_prefix = root_prefixes[input_root]
        for path in root_valid_paths:
            key = keys_by_path[path]
            if root_prefix:
                key = f"{root_prefix}/{key}"
            valid_paths.append(path)
            ordered_keys.append(key)

    if not valid_paths:
        formatted_roots = ", ".join(str(root) for root in input_roots)
        raise ValueError(f"No valid .npz motion files found under: {formatted_roots}")

    if len(set(ordered_keys)) != len(ordered_keys):
        raise ValueError("Discovered duplicate motion keys while scanning motion roots; please rename conflicting directories/files")

    return np.array(valid_paths, dtype=object), np.array(ordered_keys), skipped_files


def write_skipped_motion_report(
    report_path: Path | str,
    motion_root: Path | str | list[str] | tuple[str, ...],
    valid_motion_count: int,
    skipped_files: list[dict[str, str]],
) -> None:
    motion_roots = _normalize_motion_roots(motion_root)
    payload = {
        "motion_roots": [str(root) for root in motion_roots],
        "num_valid": valid_motion_count,
        "num_skipped": len(skipped_files),
        "skipped_files": sorted(skipped_files, key=lambda item: item["path"]),
    }
    if len(motion_roots) == 1:
        payload["motion_root"] = str(motion_roots[0])
    report_path = Path(report_path).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

class MotionLibBase():
    def __init__(self, motion_lib_cfg, num_envs, device):
        self.m_cfg = motion_lib_cfg
        self._sim_fps = 1/self.m_cfg.get("step_dt", 1/50)
        self.all_motions_loaded = False
        
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        self._skipped_motion_files: list[dict[str, str]] = []
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / self.m_cfg.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        self.load_data(self.m_cfg.motion_file)
        self.setup_constants(fix_height = motion_lib_cfg.get("fix_height", FixHeightMode.no_fix),  multi_thread = True)
        
        self.smpl_data = None
        smpl_motion_file = motion_lib_cfg.get("smpl_motion_file", None)
        if smpl_motion_file is not None:
            self.smpl_data = joblib.load(smpl_motion_file)
            self.smpl_data = [self.smpl_data[k] for k in self._motion_data_keys]
        return
        
    def load_data(self, motion_file, min_length=-1, im_eval = False):
        force_recursive_npz = bool(self.m_cfg.get("force_recursive_npz", False))
        if _is_motion_root_collection(motion_file):
            self.mode = MotionlibMode.npz
            self._motion_data_load, self._motion_data_keys, self._skipped_motion_files = discover_valid_isaaclab_npz_motions(
                motion_file
            )
            self._motion_data_list = self._motion_data_load
            if self._skipped_motion_files:
                logger.warning(
                    f"Skipped {len(self._skipped_motion_files)} invalid npz motion files while scanning motion roots {motion_file}"
                )
            report_path = self.m_cfg.get("skipped_motion_report_path", None)
            if report_path is not None:
                write_skipped_motion_report(
                    report_path=report_path,
                    motion_root=motion_file,
                    valid_motion_count=len(self._motion_data_list),
                    skipped_files=self._skipped_motion_files,
                )
            self._num_unique_motions = len(self._motion_data_list)
            logger.info(f"Loaded {self._num_unique_motions} motions")
            return

        motion_path = Path(motion_file).expanduser()
        if osp.isfile(motion_file):
            if motion_path.suffix == ".npz":
                is_valid, error_message = validate_isaaclab_npz_motion(motion_path.resolve())
                if not is_valid:
                    raise ValueError(f"Invalid Isaac Lab npz motion file {motion_path}: {error_message}")
                self.mode = MotionlibMode.npz
                self._motion_data_load = np.array([motion_path.resolve()], dtype=object)
                self._motion_data_list = self._motion_data_load
                if motion_path.name == "motion.npz":
                    motion_key = motion_path.parent.name
                else:
                    motion_key = motion_path.stem
                self._motion_data_keys = np.array([motion_key])
                self._skipped_motion_files = []
            else:
                self.mode = MotionlibMode.file
                self._motion_data_load = joblib.load(motion_file)
        else:
            pkl_files = sorted(glob.glob(osp.join(motion_file, "*.pkl")))
            if pkl_files and not force_recursive_npz:
                self.mode = MotionlibMode.directory
                self._motion_data_load = pkl_files
            else:
                self.mode = MotionlibMode.npz
                self._motion_data_load, self._motion_data_keys, self._skipped_motion_files = discover_valid_isaaclab_npz_motions(
                    motion_path
                )
                self._motion_data_list = self._motion_data_load
                if self._skipped_motion_files:
                    logger.warning(
                        f"Skipped {len(self._skipped_motion_files)} invalid npz motion files while scanning {motion_path}"
                    )
                report_path = self.m_cfg.get("skipped_motion_report_path", None)
                if report_path is not None:
                    write_skipped_motion_report(
                        report_path=report_path,
                        motion_root=motion_path,
                        valid_motion_count=len(self._motion_data_list),
                        skipped_files=self._skipped_motion_files,
                    )

        data_list = self._motion_data_load
        if self.mode == MotionlibMode.file:
            if min_length != -1:
                # filtering the data by the length of the motion
                data_list = {k: v for k, v in list(self._motion_data_load.items()) if len(v['pose_quat_global']) >= min_length}
            elif im_eval:
                # sorting the data by the length of the motion
                data_list = {item[0]: item[1] for item in sorted(self._motion_data_load.items(), key=lambda entry: len(entry[1]['pose_quat_global']), reverse=True)}
            else:
                data_list = self._motion_data_load
            self._motion_data_list = np.array(list(data_list.values()))
            self._motion_data_keys = np.array(list(data_list.keys()))
        elif self.mode == MotionlibMode.directory:
            self._motion_data_list = np.array(self._motion_data_load)
            self._motion_data_keys = np.array(self._motion_data_load)
        
        self._num_unique_motions = len(self._motion_data_list)
        if self.mode == MotionlibMode.directory:
            self._motion_data_load = joblib.load(self._motion_data_load[0]) # set self._motion_data_load to a sample of the data 
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread
        
        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches
        
    def update_soft_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only "mostly" trained on "failed" sequences. Auto PMCP. 
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self.update_sampling_prob(self._termination_history)    
            
            print("############################################################ Auto PMCP ############################################################")
            print(f"Training mostly on {len(self._sampling_prob.cpu().nonzero())} seqs ")
            print(self._motion_data_keys[self._sampling_prob.cpu().nonzero()].flatten())
            print(f"###############################################################################################################################")
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches
            
    def update_sampling_prob(self, termination_history):
        if len(termination_history) == len(self._termination_history) and termination_history.sum() > 0:
            self._sampling_prob[:] = termination_history/termination_history.sum()
            if self._sampling_prob[self._curr_motion_ids].sum() == 0:
                self._sampling_prob[self._curr_motion_ids] += 1e-6
                self._sampling_prob[:] = self._sampling_prob[:] / self._sampling_prob[:].sum()
            self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()
            self._termination_history = termination_history
            return True
        else:
            return False

    def update_sampling_weight_by_id(self, priorities: list, motions_id: list, file_name: list | None = None) -> None:
        """
        Update sampling probabilities (_sampling_prob) given motion keys and their priorities.
        This replaces the distribution with the normalized values.
        """
        assert len(motions_id) == len(priorities), "motions_id and priorities must have the same length"
        total = sum(priorities)
        assert total > 0, "Sum of priorities must be greater than zero"
        
        # Normalize priorities
        normalized_priorities = [p / total for p in priorities]
        assert all(torch.isfinite(torch.tensor(normalized_priorities))), "Priorities should be finite"
        
        # Get all motion keys
        all_keys = self._motion_data_keys.tolist()
        new_sampling_prob = torch.zeros(self._num_unique_motions, device=self._device)

        for m, p in zip(motions_id, normalized_priorities):
            # idx = all_keys.index(m)
            new_sampling_prob[m] = p
            if file_name is not None:
                assert all_keys[m] == file_name[m], f"Motion ID {m} does not match file name {file_name[m]}"

        if new_sampling_prob.sum() == 0:
            print("[Warning] No valid motions updated. Falling back to uniform sampling.")
            new_sampling_prob = torch.ones(self._num_unique_motions, device=self._device) / self._num_unique_motions
        else:
            new_sampling_prob /= new_sampling_prob.sum()

        self._sampling_prob = new_sampling_prob
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()
    def get_motion_actions(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        # import ipdb; ipdb.set_trace()
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        action = self._motion_actions[f0l]
        return action

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        if "dof_pos" in self.__dict__:
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.lrs[f0l]
            local_rot1 = self.lrs[f1l]
            
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        if "dof_pos" in self.__dict__: # Robot Joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel
            
        if self.smpl_data is not None:
            smpl_pose0 = self._motion_smpl_poses[f0l]
            smpl_pose1 = self._motion_smpl_poses[f1l]
            smpl_pose = (1.0 - blend) * smpl_pose0 + blend * smpl_pose1
            return_dict.update({"smpl_pose": smpl_pose.clone()})
        
        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1).clone(),
            "motion_aa": self._motion_aa[f0l].clone(),
            "motion_bodies": self._motion_bodies[motion_ids].clone(),
            "rg_pos": rg_pos.clone(),
            "rb_rot": rb_rot.clone(),
            "body_vel": body_vel.clone(),
            "body_ang_vel": body_ang_vel.clone(),
            "rg_pos_t": rg_pos_t.clone(),
            "rg_rot_t": rg_rot_t.clone(),
            "body_vel_t": body_vel_t.clone(),
            "body_ang_vel_t": body_ang_vel_t.clone(),
        })
        return return_dict
    
    def load_motions_for_training(self, max_num_seqs = None):
        if max_num_seqs is not None:
            assert max_num_seqs <= self.num_envs
            
        if self.all_motions_loaded:
            print("All motions already loaded!!! No need to resample.")
            return
        
        if self.m_cfg.get("override_num_motions_to_load", None) is not None:
            max_num_seqs = self.m_cfg.override_num_motions_to_load
        if max_num_seqs is None: # if not specified, load all motions, can OOM if the dataset is too large. 
            max_num_seqs = self._num_unique_motions
            self.all_motions_loaded = True
            self.load_motions(random_sample=False,  num_motions_to_load=self._num_unique_motions)
        else: # 
            if max_num_seqs > self._num_unique_motions: # if specified but more than the number of unique motions, load all motions as well.
                self.all_motions_loaded = True
                self.load_motions(random_sample=False,  num_motions_to_load=self._num_unique_motions)
            else: # if there are more motions than specified, then randomly sample the requested number of motions. 
                self.all_motions_loaded = False
                self.load_motions(random_sample=True, num_motions_to_load=max_num_seqs)
    
    def load_motions_for_evaluation(self, start_idx = 0):
        if self.all_motions_loaded:
            print("All motions already loaded!!! No need to resample.")
            return

        if self._num_unique_motions > self.num_envs: # if number of motions is more than number of envs, then we should only partially load the motions. 
            self.all_motions_loaded = False
            self.load_motions(random_sample=False,  num_motions_to_load=self.num_envs, start_idx=start_idx)
        else:
            self.all_motions_loaded = True  
            self.load_motions(random_sample=False,  num_motions_to_load=self._num_unique_motions, start_idx=start_idx)

    def load_all_motions(self):
        self.all_motions_loaded = True
        self.load_motions(random_sample=False,  num_motions_to_load=self._num_unique_motions)

    def load_motions(self, 
                     random_sample=True, 
                     start_idx=0, 
                     max_len=-1, 
                     target_heading = None, num_motions_to_load = None):
        
        if "gts" in self.__dict__:
            for attr_name in ("gts", "grs", "lrs", "grvs", "gravs", "gavs", "gvs", "dvs", "dof_pos"):
                if hasattr(self, attr_name):
                    delattr(self, attr_name)
            for attr_name in ("gts_t", "grs_t", "gvs_t", "gavs_t"):
                if hasattr(self, attr_name):
                    delattr(self, attr_name)
                
        
        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        has_action = False
        _motion_actions = []
        _motion_smpl_poses = []

        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        if num_motions_to_load is None:
            num_motion_to_load = self.num_envs
        else:
            num_motion_to_load = num_motions_to_load

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_motion_to_load, replacement=True).to(self._device)
        else: # start_idx only used for non-random sampling. 
            sample_idxes = torch.clamp(torch.arange(num_motion_to_load) + start_idx, max = self._num_unique_motions - 1 ).to(self._device)
        
        
        # sample_idxes = torch.tensor([self._motion_data_keys.tolist().index("0-KIT_8_WalkInClockwiseCircle04_poses")]).to(self._device)
        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = [self._motion_data_keys[sample_idxes.cpu()]] if sample_idxes.numel() == 1 else self._motion_data_keys[sample_idxes.cpu()].tolist()
        self._sampling_batch_prob = self._sampling_prob[self._curr_motion_ids] / self._sampling_prob[self._curr_motion_ids].sum()

        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:10]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:10]}, ....")

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        res_acc = {}  # using dictionary ensures order of the results.
        if self.mode == MotionlibMode.npz:
            for motion_index, npz_path in enumerate(track(motion_data_list, description="Loading npz motions...")):
                res_acc[motion_index] = self.load_motion_from_npz(Path(npz_path), target_heading=target_heading, max_len=max_len)
        else:
            if self.smpl_data is not None:
                smpl_data_list = [self.smpl_data[idx] for idx in sample_idxes.cpu().numpy()]
            else:
                smpl_data_list = None
            torch.set_num_threads(1)
            manager = mp.Manager()
            queue = manager.Queue()
            num_jobs = min(mp.cpu_count(), 8)
            
            if num_jobs <= 16 or not self.multi_thread:
                num_jobs = 1
            jobs = motion_data_list
            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            ids = np.arange(len(jobs))

            jobs = [(ids[i:i + chunk], jobs[i:i + chunk], smpl_data_list, self.fix_height, target_heading, max_len) for i in range(0, len(jobs), chunk)]
            job_args = [jobs[i] for i in range(len(jobs))]
            for i in range(1, len(jobs)):
                worker_args = (*job_args[i], queue, i)
                worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
                worker.start()
            res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))
            
            
            for i in track(range(len(jobs) - 1), "Gathering results..."):
                res = queue.get()
                res_acc.update(res)

        
        for f in track(range(len(res_acc)), description="Processing motions..."):
            motion_file_data, curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data['pose_aa'].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            if self.smpl_data is not None:
                _motion_smpl_poses.append(curr_motion['smpl_pose'])
            del curr_motion
        
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(np.concatenate(_motion_aa), device=self._device, dtype=torch.float32)
        if self.smpl_data is not None:
            self._motion_smpl_poses = torch.cat(_motion_smpl_poses, dim=0).float().to(self._device)
        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        
        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).float().to(self._device)
        self._num_motions = len(motions)
        
        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.lrs = torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        self.grvs = torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        self.gravs = torch.cat([m.global_root_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)
        
        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)
        
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)
        motion = motions[0]
        self.num_bodies = self.num_joints
        
        num_motions = self.num_motions()
        total_len = self.get_total_length()
        
        logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")
        
        del motions, _motion_lengths, _motion_fps, _motion_dt, _motion_num_frames, _motion_bodies, _motion_aa, _motion_actions
        torch.cuda.empty_cache()
        gc.collect()
        
    def fix_trans_height(self, pose_aa, trans, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        with torch.no_grad():
            
            mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
            height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
            trans[..., 2] -= height_diff
            
            return trans, height_diff

    def load_motion_from_npz(self, npz_path: Path, target_heading=None, max_len: int = -1):
        with np.load(npz_path, allow_pickle=False) as data:
            body_pos = torch.from_numpy(np.array(data["body_pos_w"], copy=True)).float()
            body_rot_wxyz = torch.from_numpy(np.array(data["body_quat_w"], copy=True)).float()
            body_vel = torch.from_numpy(np.array(data["body_lin_vel_w"], copy=True)).float()
            body_ang_vel = torch.from_numpy(np.array(data["body_ang_vel_w"], copy=True)).float()
            dof_pos = torch.from_numpy(np.array(data["joint_pos"], copy=True)).float()
            dof_vel = torch.from_numpy(np.array(data["joint_vel"], copy=True)).float()
            fps = _fps_to_int(data["fps"])

        seq_len = body_pos.shape[0]
        if max_len == -1 or seq_len < max_len:
            start, end = 0, seq_len
        else:
            start = random.randint(0, seq_len - max_len)
            end = start + max_len

        body_pos = body_pos[start:end]
        body_rot = body_rot_wxyz[start:end][..., [1, 2, 3, 0]]
        body_rot = body_rot / body_rot.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        body_vel = body_vel[start:end]
        body_ang_vel = body_ang_vel[start:end]
        dof_pos = dof_pos[start:end]
        dof_vel = dof_vel[start:end]

        if target_heading is not None:
            target_heading = torch.as_tensor(target_heading, dtype=body_rot.dtype).reshape(1, 4)
            start_root_rot = body_rot[0, 0].reshape(1, 4)
            heading_delta = quat_mul(target_heading, calc_heading_quat_inv(start_root_rot, w_last=True), w_last=True)
            delta_expand = heading_delta.expand(body_pos.shape[0], -1)
            body_pos = my_quat_rotate(delta_expand.repeat_interleave(body_pos.shape[1], dim=0), body_pos.reshape(-1, 3)).view_as(body_pos)
            body_vel = my_quat_rotate(delta_expand.repeat_interleave(body_vel.shape[1], dim=0), body_vel.reshape(-1, 3)).view_as(body_vel)
            body_ang_vel = my_quat_rotate(delta_expand.repeat_interleave(body_ang_vel.shape[1], dim=0), body_ang_vel.reshape(-1, 3)).view_as(body_ang_vel)
            body_rot = quat_mul(
                delta_expand.repeat_interleave(body_rot.shape[1], dim=0),
                body_rot.reshape(-1, 4),
                w_last=True,
            ).view_as(body_rot)

        motion_dict = EasyDict(
            global_translation=body_pos,
            global_rotation=body_rot,
            local_rotation=body_rot.clone(),
            global_root_velocity=body_vel[:, 0].clone(),
            global_root_angular_velocity=body_ang_vel[:, 0].clone(),
            global_velocity=body_vel,
            global_angular_velocity=body_ang_vel,
            dof_pos=dof_pos,
            dof_vels=dof_vel,
            fps=fps,
        )

        extend_config = self.m_cfg.get("extend_config", [])
        if len(extend_config) > 0:
            extend_positions, extend_rotations, extend_velocities, extend_ang_velocities = [], [], [], []
            for extend_cfg in extend_config:
                parent_name = extend_cfg["parent_name"]
                parent_idx = self.skeleton_tree.node_names.index(parent_name)
                parent_rot = body_rot[:, parent_idx]
                parent_pos = body_pos[:, parent_idx]
                parent_vel = body_vel[:, parent_idx]
                parent_ang_vel = body_ang_vel[:, parent_idx]

                offset_local = torch.tensor(extend_cfg["pos"], dtype=body_pos.dtype).unsqueeze(0).repeat(body_pos.shape[0], 1)
                extend_rot_wxyz = torch.tensor(extend_cfg["rot"], dtype=body_pos.dtype)
                extend_rot_xyzw = extend_rot_wxyz[[1, 2, 3, 0]].unsqueeze(0).repeat(body_pos.shape[0], 1)

                rotated_pos_in_parent = my_quat_rotate(parent_rot, offset_local)
                extend_pos = my_quat_rotate(extend_rot_xyzw, rotated_pos_in_parent) + parent_pos
                extend_rot = quat_mul(parent_rot, extend_rot_xyzw, w_last=True)
                extend_vel = parent_vel + torch.cross(parent_ang_vel, offset_local, dim=-1)
                extend_ang_vel = parent_ang_vel

                extend_positions.append(extend_pos)
                extend_rotations.append(extend_rot)
                extend_velocities.append(extend_vel)
                extend_ang_velocities.append(extend_ang_vel)

            motion_dict.global_translation_extend = torch.cat([body_pos, torch.stack(extend_positions, dim=1)], dim=1)
            motion_dict.global_rotation_extend = torch.cat([body_rot, torch.stack(extend_rotations, dim=1)], dim=1)
            motion_dict.global_velocity_extend = torch.cat([body_vel, torch.stack(extend_velocities, dim=1)], dim=1)
            motion_dict.global_angular_velocity_extend = torch.cat([body_ang_vel, torch.stack(extend_ang_velocities, dim=1)], dim=1)

        return {}, motion_dict

    def load_motion_with_skeleton(self,
                                  ids, 
                                  motion_data_list,
                                  smpl_data_list,
                                  fix_height,
                                  target_heading,
                                  max_len, queue, pid):
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}
        
        
        if pid == 0:
            pbar = track(range(len(ids)), description="Loading motions...") 
        else:
            pbar = range(len(ids))
        
        for f in pbar:

            curr_id = ids[f]  # id for this datasample
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]
            
            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len
                
                
            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            
            # import ipdb; ipdb.set_trace()
            if "action" in curr_file.keys():
                self.has_action = True
            
            dt = 1/curr_file['fps']
            
            B, J, N = pose_aa.shape

            if not target_heading is None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ]), w_last=True))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T))
            
            
            if self.mesh_parsers is not None:
                trans, trans_fix = self.fix_trans_height(pose_aa, trans, fix_height_mode = fix_height)
                curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
                if self.smpl_data is not None:
                    skip = int(smpl_data_list[f]['fps'] // 30)
                    curr_motion['smpl_pose'] = torch.tensor(smpl_data_list[f]['pose_aa'][::skip][start:end]).float().to(self._device)
                    assert curr_motion['smpl_pose'].shape[0] == pose_aa.shape[0]
                curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items()})
                # add "action" to curr_motion
                if self.has_action:
                    curr_motion.action = to_torch(curr_file['action']).clone()[start:end]
                res[curr_id] = (curr_file, curr_motion)
            else:
                logger.error("No mesh parser found")
                
        if not queue is None:
            queue.put(res)
        else:
            return res
    

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (self._motion_num_frames * self._sim_fps / self._motion_fps).ceil().int()
        else:
            return (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps[motion_ids]).ceil().int()

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)
    
    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._sampling_batch_prob, num_samples=n, replacement=True).to(self._device)

        return motion_ids
    

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]


    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
        
        return frame_idx0, frame_idx1, blend


    def _get_num_bodies(self):
        return self.num_bodies


    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)
