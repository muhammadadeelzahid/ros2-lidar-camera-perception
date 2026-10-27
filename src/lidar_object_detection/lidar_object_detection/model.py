"""3D detector with a learned backend and a clustering fallback.

detect() takes an (N, 4) array of x, y, z, intensity in the LiDAR frame and returns
Box3D detections. A trained OpenPCDet model is used when configured; otherwise
Euclidean clustering on the non-ground points is used.
"""

import numpy as np

from lidar_object_detection.boxes import Box3D


class DetectorModel:

    def __init__(self, config, logger=None):
        self.cfg = config
        self.log = logger
        self.classes = config.get('class_names', ['Pedestrian', 'Vehicle'])
        self.score_threshold = float(config.get('score_threshold', 0.3))
        self.fb = config.get('fallback', {})
        self._deep = None
        self.backend = 'cluster'
        self._maybe_load_deep()

    def _info(self, msg):
        if self.log is not None:
            self.log.info(msg)

    def _maybe_load_deep(self):
        checkpoint = self.cfg.get('checkpoint', '') or ''
        cfg_file = self.cfg.get('openpcdet_cfg', '') or ''
        if not checkpoint or not cfg_file:
            self._info('no checkpoint configured; using geometric fallback detector')
            return
        try:
            from lidar_object_detection.openpcdet_backend import OpenPCDetBackend
            self._deep = OpenPCDetBackend(
                cfg_file, checkpoint, self.classes,
                aliases=self.cfg.get('class_aliases', {}))
            self.backend = 'openpcdet'
            self._info(f'loaded OpenPCDet detector from {checkpoint}')
        except Exception as exc:  # noqa: BLE001 - fall back on any failure
            # A checkpoint was configured, so this is not the expected setup: say so loudly.
            msg = f'could not load OpenPCDet model ({exc!r}); using clustering instead'
            if self.log is not None:
                self.log.warning(msg)
            self._deep = None

    def detect(self, cloud):
        if self._deep is not None:
            boxes = self._deep.infer(cloud)
        else:
            boxes = self._cluster_detect(cloud)
        return [b for b in boxes if b.score >= self.score_threshold]

    # Clustering fallback
    def _cluster_detect(self, cloud):
        from scipy.spatial import cKDTree

        voxel = float(self.fb.get('voxel_size', 0.1))
        ground_z = float(self.fb.get('ground_z', -0.15))
        max_z = float(self.fb.get('max_z', 2.0))
        max_range = float(self.fb.get('max_range', 15.0))
        eps = float(self.fb.get('cluster_eps', 0.5))
        min_pts = int(self.fb.get('min_points', 15))
        score = float(self.fb.get('fixed_score', 0.5))

        pts = cloud[:, :3]
        rng = np.hypot(pts[:, 0], pts[:, 1])
        mask = (pts[:, 2] > ground_z) & (pts[:, 2] < max_z) & (rng < max_range)
        pts = pts[mask]
        if pts.shape[0] < min_pts:
            return []

        pts = self._voxel_downsample(pts, voxel)
        if pts.shape[0] < min_pts:
            return []

        labels = self._euclidean_cluster(pts, eps, cKDTree)
        boxes = []
        for lab in np.unique(labels):
            if lab < 0:
                continue
            cluster = pts[labels == lab]
            if cluster.shape[0] < min_pts:
                continue
            box = self._box_from_cluster(cluster, score)
            if box is not None:
                boxes.append(box)
        return boxes

    @staticmethod
    def _voxel_downsample(pts, voxel):
        keys = np.floor(pts / voxel).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        return pts[np.sort(idx)]

    @staticmethod
    def _euclidean_cluster(pts, eps, cKDTree):
        """Label points by connected components of the eps-neighbour graph."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        n = pts.shape[0]
        pairs = cKDTree(pts).query_pairs(eps, output_type='ndarray')
        graph = coo_matrix((np.ones(len(pairs), dtype=np.int8),
                            (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, labels = connected_components(graph, directed=False)
        return labels

    def _box_from_cluster(self, cluster, score):
        lo = cluster.min(axis=0)
        hi = cluster.max(axis=0)
        size = hi - lo
        center = (lo + hi) / 2.0
        dx, dy, dz = float(size[0]), float(size[1]), float(size[2])
        footprint = max(dx, dy)

        # Reject clearly non-hazard geometry (walls, shelves, ground strips).
        if dz < 0.4 or footprint > 3.0:
            return None
        if footprint < 1.0 and dz > 0.8:
            cls = 'Pedestrian'
        elif dx < 3.0 and dy < 3.0 and dz < 2.0:
            cls = 'Vehicle'
        else:
            return None
        if cls not in self.classes:
            return None
        return Box3D(cls, score, float(center[0]), float(center[1]),
                     float(center[2]), dx, dy, dz, 0.0)
