"""
AI Study Focus & Distraction Detector
======================================

Entry point for the desktop application.
Initialises the GUI and starts the event loop.
"""

import sys
import os
from pathlib import Path

# ─── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Local imports ────────────────────────────────────────────────────────────
import config
from src.utils.helpers import ensure_directories
from src.database.database import DatabaseManager


def _setup_torch_compatibility() -> None:
    """Fix 'operator torchvision::nms does not exist' across Python versions/Windows."""
    import sys
    import types
    try:
        import torch

        def _pure_torch_nms(boxes, scores, iou_threshold):
            if boxes.numel() == 0:
                return torch.empty((0,), dtype=torch.long, device=boxes.device)
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort(descending=True)
            keep = []
            while order.numel() > 0:
                if order.numel() == 1:
                    keep.append(order.item())
                    break
                i = order[0].item()
                keep.append(i)
                xx1 = torch.maximum(x1[i], x1[order[1:]])
                yy1 = torch.maximum(y1[i], y1[order[1:]])
                xx2 = torch.minimum(x2[i], x2[order[1:]])
                yy2 = torch.minimum(y2[i], y2[order[1:]])
                w = torch.clamp(xx2 - xx1, min=0.0)
                h = torch.clamp(yy2 - yy1, min=0.0)
                inter = w * h
                union = areas[i] + areas[order[1:]] - inter
                iou = inter / torch.clamp(union, min=1e-6)
                order = order[1:][iou <= iou_threshold]
            return torch.tensor(keep, dtype=torch.long, device=boxes.device)

        try:
            import torchvision
            torchvision.ops.nms = _pure_torch_nms
        except Exception:
            pass

        if "torchvision.ops" in sys.modules:
            sys.modules["torchvision.ops"].nms = _pure_torch_nms
        else:
            tv = sys.modules.get("torchvision", types.ModuleType("torchvision"))
            tv_ops = types.ModuleType("torchvision.ops")
            tv_ops.nms = _pure_torch_nms
            tv.ops = tv_ops
            sys.modules["torchvision"] = tv
            sys.modules["torchvision.ops"] = tv_ops
    except Exception:
        pass


def main() -> None:
    """Bootstrap the application."""
    # 0. Setup PyTorch / Vision compatibility
    _setup_torch_compatibility()

    # 1. Create required directories if they don't exist
    ensure_directories()

    # 2. Initialise the database (creates tables on first run)
    db = DatabaseManager()
    db.initialise()
    db.close()

    # 3. Launch the GUI
    from src.gui.app import StudyFocusApp

    app = StudyFocusApp()
    app.mainloop()


if __name__ == "__main__":
    main()
