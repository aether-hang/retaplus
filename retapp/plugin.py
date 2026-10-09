"""STC/HFTA hooks for teacher-guided recovery loops such as SRe2L and EDC."""

import torch
from .topology import ClassTopology
from .transport import assign


class RecoveryPlugin:
    def __init__(self, teacher, real_features, complexity, fetch_patches, slot_count, config):
        self.teacher, self.real_features = teacher, real_features
        self.complexity, self.fetch_patches, self.config = complexity, fetch_patches, config
        self.topology = (ClassTopology(real_features, slot_count, config, config.seed)
                         if slot_count > 1 and config.topology == "hfta" and config.hfta_weight else None)

    @torch.no_grad()
    def refresh(self, all_class_images):
        if self.topology is not None:
            self.topology.refresh(self.teacher(all_class_images)[1])

    def loss(self, active_images, active_indices):
        if self.topology is None:
            return active_images.sum()*0
        features = self.teacher(active_images)[1]
        return self.config.hfta_weight*self.topology.loss(features, active_indices)

    @torch.no_grad()
    def connect(self, active_images):
        features = self.teacher(active_images)[1]
        result = assign(features, self.real_features, self.complexity, self.config)
        anchors = self.fetch_patches(result.indices, active_images.shape[-1])
        return self.config.alpha*active_images + (1-self.config.alpha)*anchors
