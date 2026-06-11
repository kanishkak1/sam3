# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
from typing import Dict, List

import numpy as np
import PIL
import time

import torch
from sam3.model import box_ops
from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        
        # Cache for text embeddings to avoid redundant encoding (text encoder has 24 layers!)
        self._text_cache = {}

    @torch.inference_mode()
    def set_image(self, image, state=None):

        start = time.time()
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )

        print("time taken to set image:", round(time.time() - start, 4), "seconds")
        return state

    @torch.inference_mode()
    def set_image_batch(self, images: List[np.ndarray], state=None):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], PIL.Image.Image), (
            "Images must be a list of PIL images"
        )

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        images = [
            self.transform(v2.functional.to_image(image).to(self.device))
            for image in images
        ]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )
        return state

    @torch.inference_mode()
    def set_text_prompt(self, prompt, state: Dict):
        """Sets text prompt(s) and run inference.
        
        Args:
            prompt: Either a string (single prompt or comma-separated) or list of prompts
                   Examples:
                   - "person"
                   - "person, car, dog"  (comma-separated)
                   - ["person", "car", "dog"]  (list)
            state: State dict from set_image()
            
        Returns:
            State dict with detection results. For multiple prompts, includes:
            - boxes: All boxes from all prompts
            - scores: All scores from all prompts  
            - prompt_ids: Which prompt each detection belongs to (0-indexed)
            - prompt_labels: List of prompt strings
        """
        start = time.time()
        
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        # Parse prompts: handle string, comma-separated, or list
        if isinstance(prompt, str):
            prompts = [p.strip() for p in prompt.split(',')]
        else:
            prompts = prompt
        
        # Store prompt info for output
        state["prompt_labels"] = prompts
        state["num_prompts"] = len(prompts)
        
        # Batch encode prompts with caching
        cached_prompts = []
        uncached_prompts = []
        uncached_indices = []
        
        for i, p in enumerate(prompts):
            if p in self._text_cache:
                cached_prompts.append(self._text_cache[p])
            else:
                uncached_prompts.append(p)
                uncached_indices.append(i)
        
        # Encode uncached prompts in batch
        if uncached_prompts:
            text_outputs = self.model.backbone.forward_text(uncached_prompts, device=self.device)
            # Cache each prompt individually for future reuse
            for i, p in enumerate(uncached_prompts):
                # Extract single prompt's features from batch
                # language_features: [seq_len, batch, dim] -> [seq_len, 1, dim]
                # language_mask: [batch, seq_len] -> [1, seq_len]
                # language_embeds: [seq_len, batch, dim] -> [seq_len, 1, dim]
                single_prompt_output = {}
                for k, v in text_outputs.items():
                    if v is None:
                        single_prompt_output[k] = None
                    elif k in ['language_features', 'language_embeds']:
                        # Sequence first format: [seq_len, batch, dim]
                        single_prompt_output[k] = v[:, i:i+1, :]
                    elif k == 'language_mask':
                        # Batch first format: [batch, seq_len]
                        single_prompt_output[k] = v[i:i+1, :]
                    else:
                        # Unknown format, try to handle generically
                        single_prompt_output[k] = v[i:i+1] if v.dim() == 1 else v[:, i:i+1] if v.size(1) > i else v[i:i+1]
                self._text_cache[p] = single_prompt_output
        else:
            print(f'using text cache for all {len(prompts)} prompts')
        
        # Combine all text features (from cache and newly encoded)
        all_text_outputs = []
        for p in prompts:
            all_text_outputs.append(self._text_cache[p])
        
        # Concatenate along batch dimension
        if len(all_text_outputs) == 1:
            combined_text_outputs = all_text_outputs[0]
        else:
            combined_text_outputs = {}
            for k in all_text_outputs[0].keys():
                if all_text_outputs[0][k] is None:
                    combined_text_outputs[k] = None
                elif k in ['language_features', 'language_embeds']:
                    # Sequence first: concatenate along dim=1 (batch)
                    combined_text_outputs[k] = torch.cat([out[k] for out in all_text_outputs], dim=1)
                elif k == 'language_mask':
                    # Batch first: concatenate along dim=0 (batch)
                    combined_text_outputs[k] = torch.cat([out[k] for out in all_text_outputs], dim=0)
                else:
                    # Generic handling
                    combined_text_outputs[k] = torch.cat([out[k] for out in all_text_outputs], dim=0)
        
        # Update backbone with batched text features
        state["backbone_out"].update(combined_text_outputs)

        print("time taken to generate text output:", round(time.time() - start, 4), "seconds")

        # Create or update geometric prompt with correct batch size for multi-prompt
        if "geometric_prompt" not in state:
            start = time.time()
            state["geometric_prompt"] = self.model._get_dummy_prompt(num_prompts=len(prompts))
            print("time taken to generate geometric_prompt:", round(time.time() - start, 4), "seconds")
        else:
            # Check if existing geometric prompt has correct batch size
            existing_geo_prompt = state["geometric_prompt"]
            if existing_geo_prompt.box_mask is not None:
                existing_batch_size = existing_geo_prompt.box_mask.shape[0]
                if existing_batch_size != len(prompts):
                    # Recreate with correct batch size
                    state["geometric_prompt"] = self.model._get_dummy_prompt(num_prompts=len(prompts))

        return self._forward_grounding_batched(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: List, label: bool, state: Dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            visual_prompt = "visual"
            if visual_prompt not in self._text_cache:
                dummy_text_outputs = self.model.backbone.forward_text(
                    [visual_prompt], device=self.device
                )
                self._text_cache[visual_prompt] = dummy_text_outputs
            else:
                dummy_text_outputs = self._text_cache[visual_prompt]
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = [
            "geometric_prompt", "boxes", "masks", "masks_logits", "scores",
            "prompt_ids", "prompt_labels", "num_prompts"  # New multi-prompt fields
        ]
        for key in keys_to_del:
            if key in state:
                del state[key]
    
    def clear_text_cache(self):
        """Clears the text embedding cache. Useful if you want to free up memory."""
        self._text_cache.clear()
    
    def group_results_by_prompt(self, state: Dict):
        """Organize flat results into per-prompt groups.
        
        Useful when you want separate results for each prompt.
        
        Args:
            state: State dict from set_text_prompt() with batched results
            
        Returns:
            Dict mapping prompt string to its detections:
            {
                "person": {"boxes": tensor, "scores": tensor},
                "car": {"boxes": tensor, "scores": tensor},
                ...
            }
        """
        if "prompt_ids" not in state:
            # Single prompt case, return as-is
            return {state.get("prompt_labels", [""])[0]: {
                "boxes": state["boxes"],
                "scores": state["scores"]
            }}
        
        prompt_labels = state["prompt_labels"]
        prompt_ids = state["prompt_ids"]
        boxes = state["boxes"]
        scores = state["scores"]
        
        results = {}
        for i, prompt in enumerate(prompt_labels):
            mask = prompt_ids == i
            results[prompt] = {
                "boxes": boxes[mask],
                "scores": scores[mask]
            }
        
        return results

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict):
        """Single prompt forward grounding (backward compatibility)."""
        start = time.time()
        
        # Temporarily detach the segmentation head so SAM3 skips mask generation
        original_seg_head = self.model.segmentation_head
        self.model.segmentation_head = None
        
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=self.find_stage,
                    geometric_prompt=state["geometric_prompt"],
                    find_target=None,
                )
        finally:
            self.model.segmentation_head = original_seg_head
            
        print("time taken to generate bboxes:", round(time.time() - start, 4), "seconds")
        
        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        # out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        # out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        # out_masks = interpolate(
        #     out_masks.unsqueeze(1),
        #     (img_h, img_w),
        #     mode="bilinear",
        #     align_corners=False,
        # ).sigmoid()

        # state["masks_logits"] = out_masks
        # state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state

    @torch.inference_mode()
    def _forward_grounding_batched(self, state: Dict):
        """Batched forward grounding for multiple prompts on single image.
        
        Processes all prompts in a single forward pass and returns flat output
        with prompt_ids tracking which detection belongs to which prompt.
        """
        start = time.time()
        
        num_prompts = state.get("num_prompts", 1)
        
        # Create batched find_stage for all prompts
        # text_ids=[0,1,2,...] for each prompt, img_ids=[0,0,0,...] for same image
        batched_find_stage = FindStage(
            img_ids=torch.zeros(num_prompts, device=self.device, dtype=torch.long),  # All point to image 0
            text_ids=torch.arange(num_prompts, device=self.device, dtype=torch.long),  # One per prompt
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        
        # Temporarily detach the segmentation head so SAM3 skips mask generation
        original_seg_head = self.model.segmentation_head
        self.model.segmentation_head = None
        
        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=batched_find_stage,
                    geometric_prompt=state["geometric_prompt"],
                    find_target=None,
                )
        finally:
            self.model.segmentation_head = original_seg_head
            
        print("time taken to generate bboxes (batched):", round(time.time() - start, 4), "seconds")
        
        # outputs shape: [num_prompts, num_queries, ...]
        out_bbox = outputs["pred_boxes"]  # [num_prompts, num_queries, 4]
        out_logits = outputs["pred_logits"]  # [num_prompts, num_queries, 1]
        
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)  # [num_prompts, num_queries]
        
        # Collect all valid detections across all prompts
        all_boxes = []
        all_scores = []
        all_prompt_ids = []
        
        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        
        for prompt_idx in range(num_prompts):
            # Filter by confidence threshold for this prompt
            keep = out_probs[prompt_idx] > self.confidence_threshold
            
            if keep.any():
                prompt_boxes = out_bbox[prompt_idx][keep]  # [N, 4]
                prompt_scores = out_probs[prompt_idx][keep]  # [N]
                
                # Convert to xyxy format and scale
                prompt_boxes = box_ops.box_cxcywh_to_xyxy(prompt_boxes)
                prompt_boxes = prompt_boxes * scale_fct[None, :]
                
                all_boxes.append(prompt_boxes)
                all_scores.append(prompt_scores)
                all_prompt_ids.append(
                    torch.full((len(prompt_boxes),), prompt_idx, 
                              dtype=torch.long, device=self.device)
                )
        
        # Concatenate all detections
        if all_boxes:
            state["boxes"] = torch.cat(all_boxes, dim=0)
            state["scores"] = torch.cat(all_scores, dim=0)
            state["prompt_ids"] = torch.cat(all_prompt_ids, dim=0)
        else:
            # No detections found
            state["boxes"] = torch.empty((0, 4), device=self.device)
            state["scores"] = torch.empty((0,), device=self.device)
            state["prompt_ids"] = torch.empty((0,), dtype=torch.long, device=self.device)
        
        return state
