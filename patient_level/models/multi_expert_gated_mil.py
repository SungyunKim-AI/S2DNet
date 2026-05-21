import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


class MultiExpertGatedMIL(nn.Module):
    def __init__(self, muscle_dim=192, anatomy_dim=64, hidden_dim=192, num_classes=4, num_heads=4, num_layers=4, ablation_mode='full'):
        super().__init__()
        self.ablation_mode = ablation_mode
        if ablation_mode not in ['full', 'no_muscle', 'no_anatomy']:
            raise ValueError(f"Invalid ablation mode: {ablation_mode}")
        
        self.muscle_dim = muscle_dim
        self.anatomy_dim = anatomy_dim
        self.hidden_dim = hidden_dim
        self.nl_dim = muscle_dim
        self.neuro_dim = muscle_dim * 2
        

        # 1. 전문가별 독립적인 Projection
        self.nl_proj = nn.Sequential(
            nn.Linear(muscle_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.neuro_proj = nn.Sequential(
            nn.Linear(muscle_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 2. 해부학적 게이트
        self.anatomy_to_nl = nn.Sequential(nn.Linear(anatomy_dim, hidden_dim), nn.Sigmoid())
        self.anatomy_to_neuro = nn.Sequential(nn.Linear(anatomy_dim, hidden_dim), nn.Sigmoid())

        # 3. 통합 레이어
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # 4. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. MIL Gated Attention Pooling
        self.attention_V = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

        # 6. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # # 가중치 학습 가능한 파라미터 (이전 제안 반영)
        self.alpha_nl = nn.Parameter(torch.tensor([0.1]))
        self.alpha_neuro = nn.Parameter(torch.tensor([0.1]))

        self.apply(init_weights)

    def forward(self, x, padding_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        nl_embed = x[:, :, :self.nl_dim]
        neuro_embed = x[:, :, self.nl_dim:self.neuro_dim]
        anatomy_info = x[:, :, self.neuro_dim:self.neuro_dim + self.anatomy_dim]

        # --- Step 1: Feature Processing with Ablation Logic ---
        
        # [Ablation: No Muscle] 시그널 정보를 상수로 고정 (1.0)
        if self.ablation_mode == 'no_muscle':
            nl_embed = torch.ones_like(nl_embed)
            neuro_embed = torch.ones_like(neuro_embed)

        h_nl = self.nl_proj(nl_embed)
        h_neuro = self.neuro_proj(neuro_embed)

        # [Ablation: No Anatomy] 게이트 가중치를 0으로 고정하여 Residual만 남김
        if self.ablation_mode == 'no_anatomy':
            gate_nl = torch.zeros_like(h_nl)
            gate_neuro = torch.zeros_like(h_neuro)
        else:
            gate_nl = self.anatomy_to_nl(anatomy_info)
            gate_neuro = self.anatomy_to_neuro(anatomy_info)

        # 최종 임베딩 계산 (Residual Gated Fusion)
        h_nl_gated = h_nl * (1.0 + self.alpha_nl * gate_nl)
        h_neuro_gated = h_neuro * (1.0 + self.alpha_neuro * gate_neuro)
        # h_nl_gated = h_nl * (1.0 + gate_nl)
        # h_neuro_gated = h_neuro * (1.0 + gate_neuro)

        # --- Step 2 ~ 6: 후속 공정 (기존과 동일) ---
        h_combined = torch.cat([h_nl_gated, h_neuro_gated], dim=-1)
        h = self.fusion_fc(h_combined)

        if padding_mask is not None:
            key_padding_mask = ~padding_mask
        else:
            key_padding_mask = None
            
        h_trans = self.transformer(h, src_key_padding_mask=key_padding_mask)

        # Attention Pooling (Masking 적용)
        A_V = self.attention_V(h_trans)
        A_U = self.attention_U(h_trans)
        A_scores = self.attention_weights(A_V * A_U).squeeze(-1)
        
        if padding_mask is not None:
            A_scores = A_scores.masked_fill(~padding_mask, -1e9)
            
        A = F.softmax(A_scores, dim=1)
        M = torch.bmm(A.unsqueeze(1), h_trans).squeeze(1)

        logits = self.classifier(M)
        return logits, A
    