"""Vehicles in a camera frame, with Meta's SAM 3.1 (local GPU): one text prompt per vehicle type, boxes and
scores, no training (docs/architecture-next.md section 3).

The checkpoint is models/sam3/sam3.1_multiplex_fp16.safetensors (SAM 3.1, built for video tracking). Only its
text-prompted detector (`detector.*`) is loaded, into SAM 3.1's own detector (the multiplex detector on the
"tri" vision backbone, as sam3.model_builder.build_sam3_multiplex_video_predictor builds it; the plain SAM 3
image model does not fit these weights). Every weight must be filled, except ones the model never takes from
a checkpoint (UNFILLED_OK); a part-loaded model is refused rather than run.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .tools import ROOT

log = logging.getLogger(__name__)

CHECKPOINT = ROOT / "models" / "sam3" / "sam3.1_multiplex_fp16.safetensors"
PROMPTS = {"car": "car", "motorcycle": "two_wheeler", "bus": "bus", "truck": "truck"}   # prompt -> category
CONF = 0.5
DEDUP_IOU = 0.7                     # one vehicle found by two prompts: keep the higher score
# Not in the checkpoint, and not needed from it: the rotary-position tables are computed when the model is built,
# and the text encoder's pooled projection is discarded by detection (VETextEncoder keeps only the token features).
UNFILLED_OK = ("attn.freqs_cis", "language_backbone.encoder.text_projection")


def build_detector_model():
    import pkg_resources
    from sam3 import model_builder as mb
    from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector

    bpe = pkg_resources.resource_filename("sam3", "assets/bpe_simple_vocab_16e6.txt.gz")
    tri = mb._create_multiplex_tri_backbone(compile_mode=None, use_fa3=False, use_rope_real=False)
    return Sam3MultiplexDetector(
        num_feature_levels=1, backbone=mb.SAM3VLBackboneTri(scalp=0, visual=tri, text=mb._create_text_encoder(bpe)),
        transformer=mb._create_sam3_transformer(use_fa3=False), segmentation_head=mb._create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None, input_geometry_encoder=mb._create_geometry_encoder(), use_early_fusion=True,
        use_dot_prod_scoring=True, dot_prod_scoring=mb._create_dot_product_scoring(), supervise_joint_box_scores=True,
        is_multiplex=True)


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    category: str

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, self.y2


def _iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0


class Sam3Detector:
    def __init__(self, checkpoint: Path = CHECKPOINT, device: str = "cuda", conf: float = CONF,
                 prompts: dict[str, str] | None = None):
        import torch
        from safetensors.torch import load_file
        from sam3.model.sam3_image_processor import Sam3Processor

        self.prompts = prompts or PROMPTS
        model = build_detector_model()
        weights = {k[len("detector."):]: v.float() for k, v in load_file(str(checkpoint)).items()
                   if k.startswith("detector.")}
        missing, unexpected = model.load_state_dict(weights, strict=False)
        missing = [k for k in missing if not k.endswith(UNFILLED_OK)]
        if missing or unexpected:
            raise SystemExit(f"{checkpoint} does not fit SAM 3.1's detector: {len(missing)} weights missing "
                             f"{missing[:3]}, {len(unexpected)} unexpected {unexpected[:3]}")
        from sam3.model.data_misc import FindStage

        self.model = model.to(device).eval()
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=conf)
        self.device, self.conf = device, conf
        torch.backends.cuda.matmul.allow_tf32 = True
        # All prompts in one batched pass over the image features; prompt encodings computed once.
        n = len(self.prompts)
        self.find = FindStage(img_ids=torch.zeros(n, dtype=torch.long, device=device),
                              text_ids=torch.arange(n, device=device), input_boxes=None, input_boxes_mask=None,
                              input_boxes_label=None, input_points=None, input_points_mask=None)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
            self.text = self.model.backbone.forward_text(list(self.prompts), device=device)
            self.dummy = self.model._get_dummy_prompt(num_prompts=n)
        log.info("SAM 3.1 detector on %s (%d prompts)", device, n)

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        """Boxes and scores only (no masks): about 2.5 frames/s on a 12-16 GB consumer GPU with 4 prompts."""
        import torch
        from PIL import Image
        from sam3.model import box_ops

        image = Image.fromarray(bgr[:, :, ::-1])
        h, w = bgr.shape[:2]
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.processor.set_image(image)
            state["backbone_out"].update(self.text)
            out = self.model.forward_grounding(backbone_out=state["backbone_out"], find_input=self.find,
                                               geometric_prompt=self.dummy, find_target=None)
            prob = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
            boxes = box_ops.box_cxcywh_to_xyxy(out["pred_boxes"]).float() * torch.tensor([w, h, w, h], device=self.device)
            prob, boxes = prob.float().cpu().numpy(), boxes.cpu().numpy()
        found: list[Detection] = []
        for i, category in enumerate(self.prompts.values()):
            for j in np.nonzero(prob[i] > self.conf)[0]:
                found.append(Detection(*map(float, boxes[i, j]), float(prob[i, j]), category))
        keep: list[Detection] = []
        for d in sorted(found, key=lambda d: -d.score):
            if all(_iou(d, k) < DEDUP_IOU for k in keep):
                keep.append(d)
        return keep
