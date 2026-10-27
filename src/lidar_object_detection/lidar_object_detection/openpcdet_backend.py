"""OpenPCDet inference backend (PointPillars, CenterPoint, ...).

Imported lazily by DetectorModel. Needs torch, OpenPCDet (pcdet) and a checkpoint.
If loading fails the wrapper falls back to the clustering detector.
"""

import logging

import numpy as np

from lidar_object_detection.boxes import Box3D


class OpenPCDetBackend:

    def __init__(self, cfg_file, checkpoint, class_names, aliases=None):
        import torch
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network
        from pcdet.datasets import DatasetTemplate

        self.torch = torch
        cfg_from_yaml_file(cfg_file, cfg)
        self.cfg = cfg
        self.class_names = cfg.CLASS_NAMES
        # Map checkpoint classes onto ours, e.g. a KITTI model's Car -> Vehicle.
        self.aliases = dict(aliases or {})
        self.keep = set(class_names)

        # Dataset shim so the network gets its voxelizer and point encoder
        # without a dataset on disk.
        class _InferDataset(DatasetTemplate):
            def __init__(self):
                super().__init__(
                    dataset_cfg=cfg.DATA_CONFIG,
                    class_names=cfg.CLASS_NAMES,
                    training=False,
                )

        self.dataset = _InferDataset()
        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=self.dataset,
        )
        self.model.load_params_from_file(filename=checkpoint,
                                         logger=logging.getLogger('openpcdet'),
                                         to_cpu=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()

    def infer(self, cloud):
        from pcdet.models import load_data_to_gpu

        points = cloud.astype(np.float32)
        data_dict = self.dataset.prepare_data(data_dict={'points': points})
        data_dict = self.dataset.collate_batch([data_dict])
        if self.device == 'cuda':
            load_data_to_gpu(data_dict)
        with self.torch.no_grad():
            pred_dicts, _ = self.model.forward(data_dict)

        pred = pred_dicts[0]
        boxes = pred['pred_boxes'].cpu().numpy()      # (M, 7): x y z dx dy dz heading
        scores = pred['pred_scores'].cpu().numpy()
        labels = pred['pred_labels'].cpu().numpy()    # 1-indexed into class_names

        out = []
        for b, s, lab in zip(boxes, scores, labels):
            name = self.class_names[int(lab) - 1]
            name = self.aliases.get(name, name)
            if name not in self.keep:
                continue
            out.append(Box3D(name, float(s), float(b[0]), float(b[1]), float(b[2]),
                             float(b[3]), float(b[4]), float(b[5]), float(b[6])))
        return out
