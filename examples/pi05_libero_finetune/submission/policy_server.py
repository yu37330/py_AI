"""ポリシーサーバー（pi0.5 追加学習モデルの実装例）

examples/pi05_libero_finetune/ で追加学習した pi0.5 を提出物の形に
組み込んだもの。submission_template/policy_server.py の MyPolicy だけを
差し替えてあり、それ以外（サーバー部分・シリアライゼーション）は同一である。

前提:
  - `model_weights/` に（LoRAならマージ済みの）チェックポイントが入っていること。
  - 観測は 128x128 で渡される。pi0.5 は 256x256 で学習しているため、
    このサーバー内で拡大してから推論する。

ローカルテスト:
    pip install -r requirements.txt
    python policy_server.py                  # サーバー起動（port 8000）
    python policy_server.py --dry-run        # モデルをロードせず疎通だけ確認

    # 別ターミナルで評価実行
    python -m pipeline --server-url http://localhost:8000 --track track1
"""

import argparse
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import msgpack
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, Response

# チェックポイントの場所。既定はこのファイルの隣の model_weights/。
CKPT_DIR = Path(os.environ.get("PI05_CKPT", Path(__file__).parent / "model_weights"))
# pi0.5 の学習時解像度。採点は 128x128 で渡してくるので拡大する。
IMG_SIZE = 256


# ============================================================
# ポリシーのインターフェース定義（変更不可）
# MyPolicy が満たすべき get_action() / reset() の仕様を定める。
# ============================================================


class BasePolicy(ABC):
    """ポリシーの基底クラス。get_action() と reset() を実装してください。"""

    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        """観測からアクションを推論する。

        Args:
            obs: 環境からの観測。渡されるのは以下の6キーのみ（画像2 + 固有受容4）。
                物体の絶対座標などは渡されない（画像から推定すること）:
                - "agentview_image": (128, 128, 3) uint8  — 正面カメラ
                - "robot0_eye_in_hand_image": (128, 128, 3) uint8  — 手首カメラ
                - "robot0_joint_pos": (7,) float  — アーム関節角
                - "robot0_eef_pos": (3,) float  — エンドエフェクタ位置
                - "robot0_eef_quat": (4,) float  — エンドエフェクタ姿勢(quat)
                - "robot0_gripper_qpos": (2,) float  — グリッパ開度

        Returns:
            action: (7,) float32 — [dx, dy, dz, droll, dpitch, dyaw, gripper]
                絶対座標ではなく **デルタ制御**（LIBERO の OSC_POSE コントローラ）。
                前 6 次元は EEF の相対移動・回転で通常 [-1, 1]、
                7 次元目は gripper 開閉指令。
        """
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        """エピソード開始時に呼ばれる。内部状態をリセットしてください。

        Args:
            instruction: タスクの言語指示（例: "pick up the red mug and place it on the shelf"）
        """
        ...



# ============================================================
# ここを編集する（MyPolicy の中身だけを自分のモデルに置き換える）
# ============================================================


def _resize_uint8(img: np.ndarray, size: int) -> np.ndarray:
    """(H,W,3) uint8 -> (size,size,3) uint8、バイリニア。同サイズなら何もしない。"""
    if img.shape[0] == size and img.shape[1] == size:
        return np.ascontiguousarray(img)
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(
        t, size=(size, size), mode="bilinear", align_corners=False
    )
    return t[0].permute(1, 2, 0).clamp(0, 255).to(torch.uint8).contiguous().numpy()


def _install_transformers_shim() -> None:
    """lerobot 0.4.4 の pi0.5 が前提とする fork 版 transformers との差分を埋める。

    lerobot 0.4.4 は pi0.5 に transformers の fork
    （git+https://github.com/huggingface/transformers@fix/lerobot_openpi）を要求するが、
    提出物の requirements.txt に git 依存は書けない。fork は素の 4.53.x に対して
    実質 2 点しか変えていないため、ここで同じ状態を作る。

      1. transformers.models.siglip.check
         バージョン文字列を確認するだけのモジュール。無いと lerobot が
         ValueError("An incorrect transformer version is used") を出す。
      2. SiglipVisionTransformer で embeddings 出力を bfloat16 にキャストする処理
         pi0.5 は encoder を bfloat16 で持つ一方 embeddings は fp32 のままなので、
         これが無いと dtype が食い違う。キャスト後の値を使うのは encoder だけなので、
         SiglipEncoder.forward の入口で同じキャストを行えば等価である。

    lerobot 0.5.0 以降はこの前提自体が撤去されているため、そちらへ移行できるなら
    このシムは不要になる。
    """
    import sys
    import types

    import transformers
    from transformers.models import siglip
    from transformers.models.siglip import modeling_siglip

    if getattr(modeling_siglip, "_pi05_shim_installed", False):
        return

    if not transformers.__version__.startswith("4.53."):
        raise SystemExit(
            f"transformers 4.53.x が必要です（現在: {transformers.__version__}）。"
            " requirements.txt の pin を確認してください。"
        )

    check = types.ModuleType("transformers.models.siglip.check")
    check.check_whether_transformers_replace_is_installed_correctly = lambda: True
    sys.modules["transformers.models.siglip.check"] = check
    siglip.check = check

    _orig_forward = modeling_siglip.SiglipEncoder.forward

    def _forward(self, inputs_embeds, *args, **kwargs):
        if self.layers and self.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)
        return _orig_forward(self, inputs_embeds, *args, **kwargs)

    modeling_siglip.SiglipEncoder.forward = _forward
    modeling_siglip._pi05_shim_installed = True


class MyPolicy(BasePolicy):
    """追加学習した pi0.5 を lerobot の LIBERO 評価と同じ経路で推論する。

    obs -> action の変換は lerobot 側の実装をそのまま通す:

        gym_obs -> preprocess_observation   (pixels->observation.images.*,
                                             robot_state->observation.robot_state)
                -> LiberoProcessorStep      (画像の次元入れ替え、
                                             state = [eef_pos, quat2axisangle, gripper_qpos])
                -> preprocessor             (チェックポイントの統計で正規化・トークナイズ)
                -> policy.select_action     (10 ステップの action chunk をキュー管理)
                -> postprocessor            (action を逆正規化)

    推論が走るのは chunk が尽きたリクエストだけなので、10 秒のタイムアウト
    （README の「タイムアウト仕様」）に対して効くのは chunk 1 回分の推論時間である。
    """

    def __init__(self, device: str = "cuda"):
        _install_transformers_shim()   # lerobot を import する前に当てる

        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.env_processor import LiberoProcessorStep

        if not CKPT_DIR.is_dir():
            raise SystemExit(
                f"チェックポイントが見つからない: {CKPT_DIR}\n"
                "scripts/merge_lora.py でマージしたものを model_weights/ に配置すること。"
            )

        self.device = device
        self.policy = PI05Policy.from_pretrained(CKPT_DIR).to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=CKPT_DIR,
            preprocessor_overrides={
                "device_processor": {"device": device},
                # 既定は google/paligemma-3b-pt-224（gated かつ要ネットワーク）を指すので、
                # merge_lora.py が同梱したローカルのトークナイザに差し替える。
                "tokenizer_processor": {"tokenizer_name": str(CKPT_DIR / "tokenizer")},
            },
        )
        self.env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
        self.instruction = ""
        print(f"[pi05] loaded {CKPT_DIR} on {device}", flush=True)

    def reset(self, instruction: str = "") -> None:
        self.policy.reset()
        self.instruction = instruction or ""

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        from lerobot.envs.utils import preprocess_observation

        gym_obs = {
            "pixels": {
                "image": _resize_uint8(obs["agentview_image"], IMG_SIZE),
                "image2": _resize_uint8(obs["robot0_eye_in_hand_image"], IMG_SIZE),
            },
            "robot_state": {
                "eef": {
                    "pos": obs["robot0_eef_pos"].astype(np.float32).reshape(1, 3),
                    "quat": obs["robot0_eef_quat"].astype(np.float32).reshape(1, 4),
                },
                "gripper": {
                    "qpos": obs["robot0_gripper_qpos"].astype(np.float32).reshape(1, 2)
                },
            },
        }
        o = preprocess_observation(gym_obs)
        o["task"] = [self.instruction]
        o = self.env_preprocessor(o)
        o = self.preprocessor(o)
        with torch.inference_mode():
            action = self.postprocessor(self.policy.select_action(o))
        a = action.to("cpu").numpy()
        if a.ndim == 2:
            a = a[0]
        return a.astype(np.float32)


class RandomPolicy(BasePolicy):
    """--dry-run 用。モデルをロードせずに疎通だけ確認する。"""

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.random.uniform(-1, 1, size=7).astype(np.float32)

    def reset(self, instruction: str = "") -> None:
        pass

# ============================================================
# 以下は変更不可
# ============================================================


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json
        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--dry-run", action="store_true", help="モデルをロードせず疎通確認のみ"
    )
    args = parser.parse_args()

    set_policy(RandomPolicy() if args.dry_run else MyPolicy(device=args.device))
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
