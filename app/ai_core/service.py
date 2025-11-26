import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import copy
import os

# --- 1. CONFIGURATION ---

# Task 1: Informativeness (2 Classes)
# 0: not_informative, 1: informative
INFO_LABELS = [
    "not_informative", 
    "informative"
]

# Task 2: Humanitarian (6 Classes)
# Based on 'labels_task2' in crisismmd_dataset.py
CATEGORY_LABELS = [
    "infrastructure_and_utility_damage",       # 0
    "not_humanitarian",                        # 1
    "other_relevant_information",              # 2
    "rescue_volunteering_or_donation_effort",  # 3
    "vehicle_damage",                          # 4
    "affected_individuals"                     # 5 (Includes injured/dead/missing)
]

# Task 3: Damage Severity (3 Classes)
SEVERITY_LABELS = [
    "little_or_no_damage", # 0
    "mild_damage",         # 1
    "severe_damage"        # 2
]

# --- 2. THE MODEL ARCHITECTURE (From models_clip.py) ---
class CLIP_CrisiKAN(nn.Module):
    def apply_self_attention(self, input_tensor):
        attn_scores = torch.matmul(input_tensor, input_tensor.transpose(-1, -2))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, input_tensor)
        return attn_output
        
    def __init__(self, dim_visual_repr=768, dim_text_repr=768, dim_proj=128):
        super(CLIP_CrisiKAN, self).__init__()
        
        self.dim_visual_repr = dim_visual_repr
        self.dim_text_repr = dim_text_repr
        self.map_dim = dim_proj

        # -------------------------
        # CLIP encoders (frozen)
        # -------------------------
        print("Initializing CLIP backbone...")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)

        for _, p in self.image_encoder.named_parameters():
            p.requires_grad_(False)
        for _, p in self.text_encoder.named_parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout()

        self.image_map = nn.Sequential(
            copy.deepcopy(self.clip.visual_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.dim_visual_repr)
        )
        self.text_map = nn.Sequential(
            copy.deepcopy(self.clip.text_projection),
            nn.ReLU(),
            nn.Linear(self.clip.projection_dim, self.dim_text_repr)
        )

        del self.clip   # free memory

        # --------------------------------
        # Projections to common dimension
        # --------------------------------
        self.proj_visual = nn.Linear(dim_visual_repr, dim_proj)
        self.proj_text = nn.Linear(dim_text_repr, dim_proj)

        self.proj_visual_bn = nn.BatchNorm1d(dim_proj)
        self.proj_text_bn = nn.BatchNorm1d(dim_proj)

        # Guided cross attention masks
        self.layer_attn_visual = nn.Linear(self.dim_visual_repr, dim_proj)
        self.layer_attn_text = nn.Linear(self.dim_text_repr, dim_proj)

        # Fusion + refinement block
        self.fc_as_self_attn = nn.Linear(2*dim_proj, 2*dim_proj)
        self.self_attn_bn = nn.BatchNorm1d(2*dim_proj)

        final_dim = 2 * dim_proj

        # --------------------------------
        # MULTI-TASK CLASSIFIER HEADS
        # --------------------------------
        self.cls_task1 = nn.Linear(final_dim, 2)  # informative
        self.cls_task2 = nn.Linear(final_dim, 6)  # humanitarian
        self.cls_task3 = nn.Linear(final_dim, 3)  # damage

    def forward(self, image, text):
        # -----------------------------
        # CLIP encodings (frozen)
        # -----------------------------
        image_features = self.image_encoder(pixel_values=image).pooler_output
        image_features = self.image_map(image_features)

        text_features = self.text_encoder(**text).pooler_output
        text_features = self.text_map(text_features)

        # Self-attention inside each modality
        f_i_self_attn = self.apply_self_attention(image_features)
        e_i_self_attn = self.apply_self_attention(text_features)

        # Linear projections (Eqn. 3)
        f_i_tilde = F.relu(self.proj_visual_bn(self.proj_visual(image_features)))
        e_i_tilde = F.relu(self.proj_text_bn(self.proj_text(text_features)))

        # Guided cross-attention masks
        alpha_v_i = torch.sigmoid(self.layer_attn_text(e_i_self_attn))
        alpha_e_i = torch.sigmoid(self.layer_attn_visual(f_i_self_attn))

        # Masked representations
        masked_v_i = alpha_v_i * f_i_tilde
        masked_e_i = alpha_e_i * e_i_tilde

        # Joint representation
        joint_repr = torch.cat((masked_v_i, masked_e_i), dim=1)

        # Refinement block
        fused = self.fc_as_self_attn(joint_repr)
        fused = self.self_attn_bn(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)

        # -----------------------------
        # MULTI-TASK OUTPUT
        # -----------------------------
        return {
            "task1": self.cls_task1(fused),
            "task2": self.cls_task2(fused),
            "task3": self.cls_task3(fused)
        }

# --- 3. THE SERVICE HANDLER ---
class CrisisAIService:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"AI Service initialized on: {self.device}")

    def load_model(self):
        print("Loading AI Models...")
        try:
            self.model = CLIP_CrisiKAN()
            
            # Ensure this path matches where you copied the file in Docker
            model_path = "app/ai_core/best.pt" 
            
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                
                # NOTE: strict=False is useful if there are minor naming mismatches 
                # (e.g. if you wrapped the model in DataParallel during training, keys might have 'module.' prefix)
                # If your saved weights have 'module.' prefixes, you might need to strip them or use strict=False
                self.model.load_state_dict(state_dict, strict=False)
                print(f"Weights loaded from {model_path}")
            else:
                print(f"WARNING: {model_path} not found! Model is using random weights.")

            self.model.to(self.device)
            self.model.eval()

            # The Processor handles the raw text/image inputs
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("AI Engine Online.")
            
        except Exception as e:
            print(f"CRITICAL ERROR LOADING AI MODEL: {e}")

    def predict(self, text: str, image_path: str):
        if not self.model: return {"severity": "Error", "category": "Error", "is_informative": False}

        try:
            image = Image.open(image_path).convert("RGB")
            
            # Prepare inputs
            # CLIPProcessor handles both image and text
            inputs = self.processor(
                text=[text], 
                images=image, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=77
            )
            
            pixel_values = inputs['pixel_values'].to(self.device)
            
            # Extract text-specific arguments (input_ids, attention_mask)
            text_inputs = {k: v.to(self.device) for k, v in inputs.items() if k != 'pixel_values'}

            with torch.no_grad():
                # Forward Pass
                outputs = self.model(pixel_values, text_inputs)
                
                # outputs is a dict: {'task1': logits, 'task2': logits, 'task3': logits}
                logits_info = outputs['task1']
                logits_cat = outputs['task2']
                logits_sev = outputs['task3']

                # Decode Indices
                info_idx = logits_info.argmax(dim=1).item()
                cat_idx = logits_cat.argmax(dim=1).item()
                sev_idx = logits_sev.argmax(dim=1).item()

            # Map indices to Strings
            return {
                "is_informative": INFO_LABELS[info_idx] if info_idx < len(INFO_LABELS) else False,
                "category": CATEGORY_LABELS[cat_idx] if cat_idx < len(CATEGORY_LABELS) else "Unknown",
                "severity": SEVERITY_LABELS[sev_idx] if sev_idx < len(SEVERITY_LABELS) else "Unknown"
            }

        except Exception as e:
            print(f"Prediction Error: {e}")
            import traceback
            traceback.print_exc()
            return {"severity": "Error", "category": "Error", "is_informative": False}

ai_engine = CrisisAIService()